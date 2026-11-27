import Foundation
import MLX
import MLXNN

/// A standalone checkpoint with layer-local tensors in the Swift module layout.
public enum NemotronImageQuantization {
    public static let scheme = "nemotron-image-experts-affine-v1"

    static func expertBits(_ manifest: MereRunModelManifest?) throws -> Int? {
        guard let manifest else { return nil }
        if manifest.precision == .bf16, manifest.quantization == nil { return nil }
        let precisionBits: [MereRunModelManifest.Precision: Int] = [.int4: 4, .int6: 6, .int8: 8]
        guard let precision = manifest.precision, let bits = precisionBits[precision], let quant = manifest.quantization,
              quant.scheme == scheme, quant.groupSize == 64, quant.bits == bits else {
            throw NemotronImageResources.invalid("Expected a BF16 or group-64 affine expert checkpoint.")
        }
        return bits
    }

    static func filename(_ prefix: String) -> String {
        String(prefix.dropLast()) + ".safetensors"
    }

    static var files: [String] {
        (["backbone.embeddings.", "backbone.norm_f.", "lm_head."] + (0..<52).map { "backbone.layers.\($0)." }).map(filename)
    }

    static func validate(_ weights: [String: MLXArray], for module: Module) throws {
        let parameters = module.parameters().flattened()
        guard weights.count == parameters.count else {
            throw NemotronImageResources.invalid("Unexpected tensor count in packed module.")
        }
        for (key, parameter) in parameters {
            let packed = key.contains("mixer.experts.") && key.hasSuffix(".weight")
            guard let value = weights[key], value.shape == parameter.shape, value.dtype == (packed ? .uint32 : .bfloat16) else {
                throw NemotronImageResources.invalid("Invalid packed checkpoint tensor \(key).")
            }
        }
    }

    static func pack(_ weights: [(String, MLXArray)], bits: Int) -> [String: MLXArray] {
        var output: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("mixer.experts."), key.hasSuffix(".weight") {
                let prefix = key.dropLast("weight".count)
                let bank = quantized(value, groupSize: 64, bits: bits)
                output[key] = bank.wq
                output[prefix + "scales"] = bank.scales
                output[prefix + "biases"] = bank.biases
            } else {
                output[key] = value
            }
        }
        return output
    }

    public static func export(source: URL, destination: URL, bits: Int,
                              progress: (String) -> Void) throws -> [URL] {
        let fm = FileManager.default
        guard [4, 6, 8].contains(bits), !fm.fileExists(atPath: destination.path) else {
            throw NemotronImageResources.invalid("Choose 4, 6 or 8 bits and a new output directory.")
        }
        let config = try NemotronImageResources.languageConfig(at: source)
        let community = fm.fileExists(atPath: source.appending(path: "embed.npy").path)
        let decoder = community ? "vq" : NemotronImageResources.decoderDirectory
        try NemotronImageResources.DecoderConfig.validate(at: source.appending(path: decoder))
        var manifest: MereRunModelManifest
        let sourceModel: NemotronOmniCausalLM
        let index: HFSafetensorsIndex?
        if community {
            try NemotronImageCommunityPack.checkRecipe(at: source, bits: bits)
            manifest = .template(for: .nemotronImage)
            let metadata = source.appending(path: ".cache/huggingface/download/config.json.metadata")
            let revision = (try? String(contentsOf: metadata, encoding: .utf8))?.split(separator: "\n").first
            manifest.sources = (manifest.sources ?? []) + [.init(role: "packed-generator-and-decoder", repository: NemotronImageCommunityPack.repository,
                                      revision: revision.map(String.init) ?? "unspecified", destinationPath: ".")]
            sourceModel = NemotronOmniCausalLM(config: config, expertBits: bits)
            index = nil
        } else {
            manifest = try MereRunModelManifest.loadRequired(from: source)
            guard manifest.engine == .nemotronImage, try expertBits(manifest) == nil else {
                throw NemotronImageResources.invalid("Expected the original Nemotron Image checkpoint.")
            }
            sourceModel = NemotronOmniCausalLM(config: config)
            index = try NemotronImageLoader.releasedIndex(at: source)
        }

        let parent = destination.deletingLastPathComponent()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appending(path: ".nemotron-image-\(UUID().uuidString)")
        try fm.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { if fm.fileExists(atPath: staging.path) { try? fm.removeItem(at: staging) } }
        var names: [String] = []
        try NemotronImageLoader.visitModules(sourceModel) { module, prefix in
            progress(prefix)
            let weights: [String: MLXArray]
            if let index {
                weights = pack(try NemotronImageLoader.releasedWeights(at: source, config: config,
                    index: index, module: module, prefix: prefix), bits: bits)
            } else {
                weights = try NemotronImageCommunityPack.weights(at: source, prefix: prefix, module: module)
                try validate(weights, for: module)
            }
            let name = filename(prefix)
            try MLX.save(arrays: weights, url: staging.appending(path: name))
            names.append(name)
        }
        let decoderPath = NemotronImageResources.decoderDirectory
        try fm.createDirectory(at: staging.appending(path: decoderPath), withIntermediateDirectories: false)
        var copies = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                      "unitron_extension_manifest.json"].map { ($0, $0) }
        for name in ["config.json", "model.safetensors"] {
            copies.append(("\(decoder)/\(name)", "\(decoderPath)/\(name)"))
        }
        for name in ["LICENSE", "NOTICE", "README.md"] where fm.fileExists(atPath: source.appending(path: name).path) {
            copies.append((name, name))
        }
        for (from, to) in copies {
            try fm.copyItem(at: source.appending(path: from), to: staging.appending(path: to))
            names.append(to)
        }
        manifest.id = destination.lastPathComponent
        switch bits {
        case 8: manifest.precision = .int8
        case 6: manifest.precision = .int6
        default: manifest.precision = .int4
        }
        manifest.quantization = .init(bits: bits, groupSize: 64, scheme: scheme)
        manifest.createdAt = Date()
        try manifest.write(to: staging)
        try fm.moveItem(at: staging, to: destination)
        return (names + [MereRunModelManifest.filename]).map { destination.appending(path: $0) }
    }
}
