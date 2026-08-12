import Foundation
import MLX

struct NemotronHLoadedModel: @unchecked Sendable {
    let model: NemotronHCausalLM
    let tokenizer: Q35TokenizerAndTemplate
    let config: NemotronHConfig
    let dspark: NemotronHDSparkModel?
    let rootURL: URL
}

enum NemotronHModelLoader {
    static func load(
        rootURL: URL,
        dsparkPath: String?,
        maxContextLength: Int,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> NemotronHLoadedModel {
        let rootURL = rootURL.standardizedFileURL
        let missing = NemotronHResources.missingTargetFiles(rootURL: rootURL)
        guard missing.isEmpty else {
            throw NemotronHError.missingFiles(missing.map(\.lastPathComponent))
        }
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Nemotron config"))
        let config = try JSONDecoder().decode(
            NemotronHConfig.self,
            from: Data(contentsOf: rootURL.appendingPathComponent("config.json"))
        )
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Nemotron tokenizer"))
        let tokenizer = try Q35TokenizerAndTemplate.load(
            from: rootURL,
            maxLengthOverride: min(maxContextLength, config.maxPositionEmbeddings)
        )
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Nemotron weights"))
        let model = NemotronHCausalLM(config: config)
        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: rootURL.appendingPathComponent("model.safetensors.index.json"),
            to: model,
            dtype: nil,
            verify: .shapeMismatch,
            progressHandler: { progress in
                progressHandler?(ChatProgress(
                    stage: .loadingModel,
                    message: "Loading Nemotron shard \(progress.shardIndex + 1)/\(progress.shardCount)"
                ))
            }
        )

        let dspark: NemotronHDSparkModel?
        if let dsparkPath {
            let dsparkRoot = URL(fileURLWithPath: dsparkPath).standardizedFileURL
            let missing = NemotronHResources.missingDSparkFiles(rootURL: dsparkRoot)
            guard missing.isEmpty else {
                throw NemotronHError.missingFiles(
                    missing.map { "DSpark/\($0.lastPathComponent)" }
                )
            }
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Nemotron DSpark"))
            let dsparkConfig = try JSONDecoder().decode(
                NemotronHDSparkConfig.self,
                from: Data(contentsOf: dsparkRoot.appendingPathComponent("config.json"))
            )
            try validateCompatibility(target: config, dspark: dsparkConfig)
            let loaded = NemotronHDSparkModel(config: dsparkConfig)
            try HFSafetensorsWeightsLoader.applyWeights(
                url: dsparkRoot.appendingPathComponent("model.safetensors"),
                to: loaded,
                dtype: nil,
                verify: .shapeMismatch
            )
            dspark = loaded
        } else {
            dspark = nil
        }
        return NemotronHLoadedModel(
            model: model,
            tokenizer: tokenizer,
            config: config,
            dspark: dspark,
            rootURL: rootURL
        )
    }

    static func validateCompatibility(
        target: NemotronHConfig,
        dspark: NemotronHDSparkConfig
    ) throws {
        guard target.vocabSize == dspark.vocabSize else {
            throw NemotronHError.incompatibleDSpark("vocabularies differ")
        }
        guard target.hiddenSize == dspark.hiddenSize else {
            throw NemotronHError.incompatibleDSpark("hidden sizes differ")
        }
        guard dspark.speculation.targetLayerIDs.allSatisfy({
            $0 >= 0 && $0 < target.numHiddenLayers
        }) else {
            throw NemotronHError.incompatibleDSpark("target layer IDs are out of range")
        }
    }
}
