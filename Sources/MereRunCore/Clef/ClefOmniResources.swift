import Foundation
import MLX
import MLXNN
import AudioQwen3ASRModel
import MereRunQwenModel

public enum ClefOmniCatalog {
    public static let modelID = "text-decide-clef-omni"
    public static let repository = "Cloudflare/clef-omni"
    public static let revision = "0db1cd2607d76a7bdb2a382f659e7b313079f84b"
    public static let hubFallback = HubFallbackConfig(repoId: repository, revision: revision, patterns: ClefCatalog.files)
}

struct ClefOmniResources {
    let root: URL

    func configuration() throws -> (Qwen3OmniConfiguration, ClefHeadConfiguration) {
        let config = try Qwen3OmniConfiguration.decode(Data(contentsOf: root.appending(path: "config.json")))
        let head = try JSONDecoder().decode(ClefHeadConfiguration.self, from: Data(contentsOf: root.appending(path: "joint_head_config.json")))
        try head.validate(backboneHiddenSize: config.thinkerConfig.textConfig.hiddenSize)
        // The reference uses Whisper audio and Qwen2-VL image/video preprocessing.
        struct Processor: Decodable {
            struct Features: Decodable { let sampling_rate: Int; let feature_size: Int; let n_fft: Int; let hop_length: Int }
            struct Visual: Decodable {
                let patch_size: Int; let temporal_patch_size: Int; let merge_size: Int
                let image_mean: [Float]; let image_std: [Float]; let resample: Int
                let rescale_factor: Double
                let do_resize: Bool; let do_rescale: Bool; let do_normalize: Bool; let do_convert_rgb: Bool
                var supported: Bool {
                    patch_size == 16 && temporal_patch_size == 2 && merge_size == 2 && resample == 3
                        && image_mean == [0.5, 0.5, 0.5] && image_std == [0.5, 0.5, 0.5]
                        && abs(rescale_factor - 1 / 255.0) < 1e-12
                        && do_resize && do_rescale && do_normalize && do_convert_rgb
                }
            }
            let processor_class: String
            let feature_extractor: Features
            let image_processor: Visual
            let video_processor: Visual
        }
        let processor = try JSONDecoder().decode(Processor.self, from: Data(contentsOf: root.appending(path: "processor_config.json")))
        guard processor.processor_class == "Qwen3OmniMoeProcessor", processor.feature_extractor.sampling_rate == 16_000,
              processor.feature_extractor.feature_size == 128, processor.feature_extractor.n_fft == 400,
              processor.feature_extractor.hop_length == 160,
              processor.image_processor.supported, processor.video_processor.supported else {
            throw ClefError.invalidConfiguration("Clef Omni requires its native 16 kHz / 128-bin Whisper processor.")
        }
        return (config, head)
    }

    func validate(fileManager: FileManager) -> [URL] {
        let required = ["config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                        "joint_head_config.json", "joint_head.safetensors", "model.safetensors.index.json"]
        let missing = required.map { root.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        guard missing.isEmpty else { return missing }
        do {
            _ = try configuration()
            let indexURL = root.appending(path: "model.safetensors.index.json")
            let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: Data(contentsOf: indexURL))
            guard index.weightMap["thinker.model.embed_tokens.weight"] != nil,
                  index.weightMap["thinker.lm_head.weight"] != nil else { return [indexURL] }
            return Set(index.weightMap.filter { $0.key.hasPrefix("thinker.") }.values)
                .map { root.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        } catch { return [root.appending(path: "config.json")] }
    }

    /// Read only shards needed by a thinker component; talker/code2wav arrays never enter the parameter tree.
    func arrays(prefix: String) throws -> [String: MLXArray] {
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self,
            from: Data(contentsOf: root.appending(path: "model.safetensors.index.json")))
        let selected = index.weightMap.filter { $0.key.hasPrefix(prefix) }
        guard !selected.isEmpty else { throw ClefError.invalidWeights("Clef Omni checkpoint has no \(prefix) parameters.") }
        var result: [String: MLXArray] = [:]
        for shard in Set(selected.values).sorted() {
            try Task.checkCancellation()
            let loaded = try MLX.loadArrays(url: root.appending(path: shard))
            for (key, file) in selected where file == shard {
                guard let array = loaded[key] else { throw ClefError.invalidWeights("Missing indexed Clef Omni tensor \(key).") }
                result[String(key.dropFirst(prefix.count))] = array
            }
        }
        return result
    }

    func loadText(_ model: Qwen3OmniThinker) throws {
        var arrays = try arrays(prefix: "thinker.model.")
        let output = try self.arrays(prefix: "thinker.lm_head.")
        guard let weight = output["weight"], output.count == 1 else { throw ClefError.invalidWeights("Invalid Omni output embedding.") }
        arrays["lm_head.weight"] = weight
        for layer in 0..<model.config.numHiddenLayers {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let keys = (0..<model.config.numExperts).map { "layers.\(layer).mlp.experts.\($0).\(projection).weight" }
                let experts = try keys.map { key -> MLXArray in
                    guard let value = arrays.removeValue(forKey: key) else { throw ClefError.invalidWeights("Missing Omni expert \(key).") }
                    return value
                }
                let bank = MLX.stacked(experts, axis: 0)
                MLX.eval(bank)
                arrays["layers.\(layer).mlp.\(projection).weight"] = bank
            }
        }
        try install(arrays, into: model)
    }

    func loadVision(_ model: Qwen3OmniVision) throws {
        var mapped: [String: MLXArray] = [:]
        for (rawKey, rawValue) in try arrays(prefix: "thinker.visual.") {
            var key = rawKey.replacingOccurrences(of: "merger_list.", with: "deepstack_merger_list.")
            if key.hasPrefix("merger.") { key = "patch_merger." + key.dropFirst("merger.".count) }
            key = key.replacingOccurrences(of: ".mlp.0.", with: ".mlp_0.")
                .replacingOccurrences(of: ".mlp.2.", with: ".mlp_2.")
                .replacingOccurrences(of: ".mlp.linear_fc1.", with: ".mlp.fc1.")
                .replacingOccurrences(of: ".mlp.linear_fc2.", with: ".mlp.fc2.")
            let value = key == "patch_embed.proj.weight" ? rawValue.transposed(0, 2, 3, 4, 1) : rawValue
            mapped["tower." + key] = value
        }
        try install(mapped, into: model)
    }

    func loadAudio(_ model: Qwen3ASRAudioTower) throws {
        let mapped = try arrays(prefix: "thinker.audio_tower.").mapValues { $0 }
        var arrays: [String: MLXArray] = [:]
        for (key, value) in mapped {
            arrays[key] = key.hasPrefix("conv2d") && key.hasSuffix(".weight") ? value.transposed(0, 2, 3, 1) : value
        }
        try install(arrays, into: model)
    }

    private func install(_ arrays: [String: MLXArray], into model: Module) throws {
        guard Set(arrays.keys) == Set(model.parameters().flattened().map(\.0)),
              arrays.values.allSatisfy({ $0.dtype == .bfloat16 || $0.dtype == .float32 }) else {
            throw ClefError.invalidWeights("Clef Omni requires complete, unquantized thinker component parameters.")
        }
        try model.update(parameters: ModuleParameters.unflattened(arrays), verify: [.all])
    }
}
