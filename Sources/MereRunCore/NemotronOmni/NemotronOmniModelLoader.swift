import Foundation
import MLX
import MLXNN

struct NemotronOmniLoadedModel: @unchecked Sendable {
    let model: NemotronOmniCausalLM
    let visionTower: NemotronOmniVisionTower
    let soundTower: NemotronOmniSoundTower
    let tokenizer: Q35TokenizerAndTemplate
    let config: NemotronOmniConfig
    let preprocessorConfig: NemotronOmniPreprocessorConfig
    let rootURL: URL
}

struct NemotronOmniExpertWeightKey: Equatable, Sendable {
    enum Projection: String, Sendable {
        case up = "up_proj"
        case down = "down_proj"
    }

    let layer: Int
    let expert: Int
    let projection: Projection

    init?(checkpointKey: String) {
        let parts = checkpointKey.split(separator: ".").map(String.init)
        guard parts.count == 9,
              parts[0] == "language_model",
              parts[1] == "backbone",
              parts[2] == "layers",
              let layer = Int(parts[3]),
              parts[4] == "mixer",
              parts[5] == "experts",
              let expert = Int(parts[6]),
              let projection = Projection(rawValue: parts[7]),
              parts[8] == "weight" else {
            return nil
        }
        self.layer = layer
        self.expert = expert
        self.projection = projection
    }
}

enum NemotronOmniModelLoader {
    static func load(
        rootURL: URL,
        maxContextLength: Int,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> NemotronOmniLoadedModel {
        let rootURL = rootURL.standardizedFileURL
        let missing = NemotronOmniResources.missingTargetFiles(rootURL: rootURL)
        guard missing.isEmpty else {
            throw NemotronOmniError.missingFiles(missing.map(\.lastPathComponent))
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Nemotron Omni config"))
        let config = try JSONDecoder().decode(
            NemotronOmniConfig.self,
            from: Data(contentsOf: rootURL.appendingPathComponent("config.json"))
        )
        let preprocessorConfig = try JSONDecoder().decode(
            NemotronOmniPreprocessorConfig.self,
            from: Data(contentsOf: rootURL.appendingPathComponent("preprocessor_config.json"))
        )
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Nemotron Omni tokenizer"))
        let tokenizer = try Q35TokenizerAndTemplate.load(
            from: rootURL,
            maxLengthOverride: min(maxContextLength, config.maximumSequenceLength)
        )
        let model = NemotronOmniCausalLM(config: config.language.runtimeConfig)
        let visionTower = NemotronOmniVisionTower(config: config)
        let soundTower = NemotronOmniSoundTower(config: config)
        let expertPack: URL
        if let existing = NemotronOmniExpertPack.optimizedURLIfValid(rootURL: rootURL) {
            expertPack = existing
        } else {
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Preparing the native Nemotron Omni BF16 expert cache"
            ))
            expertPack = try NemotronOmniExpertPack.optimize(
                rootURL: rootURL,
                progressHandler: { completed, total in
                    let percent = Int((Double(completed) / Double(total)) * 100)
                    progressHandler?(ChatProgress(
                        stage: .loadingModel,
                        message: "Packing Nemotron Omni experts \(percent)%"
                    ))
                }
            )
        }
        try loadWeights(
            indexURL: rootURL.appendingPathComponent("model.safetensors.index.json"),
            model: model,
            visionTower: visionTower,
            soundTower: soundTower,
            expertPackURL: expertPack,
            progressHandler: progressHandler
        )
        return NemotronOmniLoadedModel(
            model: model,
            visionTower: visionTower,
            soundTower: soundTower,
            tokenizer: tokenizer,
            config: config,
            preprocessorConfig: preprocessorConfig,
            rootURL: rootURL
        )
    }

    private static func loadWeights(
        indexURL: URL,
        model: NemotronOmniCausalLM,
        visionTower: NemotronOmniVisionTower,
        soundTower: NemotronOmniSoundTower,
        expertPackURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws {
        let index = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: indexURL)
        )
        let root = indexURL.deletingLastPathComponent()
        for (shardIndex, filename) in index.shardFilenames.enumerated() {
            let shardURL = root.appendingPathComponent(filename)
            let arrays = try MLX.loadArrays(url: shardURL)
            var languageUpdates: [(String, MLXArray)] = []
            var visionUpdates: [(String, MLXArray)] = []
            var soundUpdates: [(String, MLXArray)] = []
            for (key, value) in arrays where key.hasPrefix("language_model.") {
                if NemotronOmniExpertWeightKey(checkpointKey: key) != nil {
                    continue
                }
                let mappedKey = String(key.dropFirst("language_model.".count))
                let mappedValue: MLXArray
                if mappedKey.hasSuffix(".mixer.conv1d.weight"), value.ndim == 3 {
                    mappedValue = value.transposed(0, 2, 1)
                } else {
                    mappedValue = value
                }
                languageUpdates.append((mappedKey, mappedValue))
            }
            for (key, value) in arrays {
                if key.hasPrefix("vision_model.radio_model.model.") {
                    var mappedKey = "model." + String(
                        key.dropFirst("vision_model.radio_model.model.".count)
                    )
                    mappedKey = mappedKey.replacingOccurrences(
                        of: "patch_generator.cls_token.token",
                        with: "patch_generator.class_token"
                    )
                    visionUpdates.append((mappedKey, value))
                } else if key.hasPrefix("mlp1.") {
                    let suffix = String(key.dropFirst("mlp1.".count))
                    let mappedKey: String
                    switch suffix {
                    case "0.weight": mappedKey = "projector.norm.weight"
                    case "1.weight": mappedKey = "projector.linear1.weight"
                    case "3.weight": mappedKey = "projector.linear2.weight"
                    default: continue
                    }
                    visionUpdates.append((mappedKey, value))
                }
            }
            for (key, value) in arrays where !key.contains("num_batches_tracked") {
                let mappedKey: String
                if key.hasPrefix("sound_encoder.encoder.feature_extractor.") {
                    continue
                } else if key.hasPrefix("sound_encoder.") {
                    mappedKey = String(key.dropFirst("sound_encoder.".count))
                } else if key.hasPrefix("sound_projection.") {
                    mappedKey = "projection." + String(
                        key.dropFirst("sound_projection.".count)
                    )
                } else {
                    continue
                }
                let mappedValue: MLXArray
                if value.ndim == 3,
                   mappedKey.contains(".conv.") {
                    mappedValue = value.transposed(0, 2, 1)
                } else if value.ndim == 4,
                          mappedKey.hasPrefix("encoder.subsampling.layers.") {
                    mappedValue = value.transposed(0, 2, 3, 1)
                } else {
                    mappedValue = value
                }
                soundUpdates.append((mappedKey, mappedValue))
            }
            if !languageUpdates.isEmpty {
                try model.update(
                    parameters: ModuleParameters.unflattened(languageUpdates),
                    verify: .shapeMismatch
                )
            }
            if !visionUpdates.isEmpty {
                try visionTower.update(
                    parameters: ModuleParameters.unflattened(visionUpdates),
                    verify: .shapeMismatch
                )
            }
            if !soundUpdates.isEmpty {
                try soundTower.update(
                    parameters: ModuleParameters.unflattened(soundUpdates),
                    verify: .shapeMismatch
                )
            }
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Loading Nemotron Omni language shard \(shardIndex + 1)/\(index.shardFilenames.count)"
            ))
        }

        try HFSafetensorsWeightsLoader.applyWeights(
            url: expertPackURL,
            to: model,
            dtype: nil,
            verify: .shapeMismatch
        )
        model.train(false)
        visionTower.train(false)
        soundTower.train(false)
    }
}
