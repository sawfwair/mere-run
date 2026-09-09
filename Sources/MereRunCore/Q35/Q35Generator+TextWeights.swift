import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

extension Q35Generator {
    func loadTextWeights(
        into q35Model: Q35Model,
        from resources: Q35Resources,
        groupSize: Int,
        bits: Int
    ) throws {
        let checkpointUsesZeroCenteredNorms = try Self.checkpointUsesZeroCenteredRMSNorm(from: resources)
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            guard let mapped = Self.mapTextWeightKey(key) else { return [] }
            if q35Model.config.tieWordEmbeddings, mapped == "lm_head.weight" {
                return []
            }
            if let splitExperts = Self.splitMappedExpertGateUpWeight(mapped, value) {
                return splitExperts
            }
            let normalizedMapped = Self.normalizeMappedExpertWeightKey(mapped)
            if normalizedMapped.hasSuffix(".conv1d.weight"), value.ndim == 3 {
                return [(normalizedMapped, Self.normalizedLinearAttentionConv1DWeight(value))]
            }
            if Self.isOffsetRMSNormWeight(normalizedMapped) {
                return [(
                    normalizedMapped,
                    Self.normalizedRMSNormWeight(
                        value,
                        checkpointUsesZeroCenteredNorms: checkpointUsesZeroCenteredNorms
                    )
                )]
            }
            return [(normalizedMapped, value)]
        }
        let keyMapper: (String) -> String = { key in
            guard let mapped = Self.mapTextWeightKey(key) else { return "__unused__.\(key)" }
            if q35Model.config.tieWordEmbeddings, mapped == "lm_head.weight" {
                return "__unused__.\(key)"
            }
            return Self.normalizeMappedExpertWeightKey(mapped)
        }
        let quantizedMapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            guard !key.hasPrefix("__unused__.") else { return [] }
            if key.hasSuffix(".conv1d.weight"), value.ndim == 3 {
                return [(key, Self.normalizedLinearAttentionConv1DWeight(value))]
            }
            if Self.isOffsetRMSNormWeight(key) {
                return [(
                    key,
                    Self.normalizedRMSNormWeight(
                        value,
                        checkpointUsesZeroCenteredNorms: checkpointUsesZeroCenteredNorms
                    )
                )]
            }
            return [(key, value)]
        }

        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            if try Self.indexContainsQuantizedWeights(resources.modelIndexURL) {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: resources.modelIndexURL,
                    to: q35Model,
                    groupSize: groupSize,
                    bits: bits,
                    keyMapper: keyMapper,
                    mapper: quantizedMapper
                )
            } else {
                try HFSafetensorsWeightsLoader.applyShardedWeights(
                    indexURL: resources.modelIndexURL,
                    to: q35Model,
                    dtype: .bfloat16,
                    verify: .none,
                    mapper: mapper
                )
            }
            return
        }

        let arrays = try MLX.loadArrays(url: resources.modelWeightsURL)
        if HFSafetensorsWeightsLoader.isQuantized(arrays) {
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                arrays,
                to: q35Model,
                groupSize: groupSize,
                bits: bits,
                keyMapper: keyMapper,
                mapper: quantizedMapper
            )
        } else {
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: resources.modelWeightsURL,
                to: q35Model,
                dtype: .bfloat16,
                verify: .none,
                include: { Self.mapTextWeightKey($0) != nil },
                mapper: mapper,
                batchSize: 32
            )
        }
    }

    func loadQ38NGramEmbeddings(
        into model: Q35Model,
        from resources: Q35Resources,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws {
        let targets = model.q38NGramEmbeddings
        guard !targets.isEmpty else { return }

        let placement = try Q38PLEPlacement.resolve(
            rootURL: resources.rootURL,
            progressHandler: { message in
                progressHandler?(ChatProgress(stage: .loadingModel, message: message))
            }
        )

        for (pleLayerIndex, target) in targets.enumerated() {
            let layerIndex = model.config.textConfig.pleLayerIds[pleLayerIndex] - 1
            let base = "language_model.model.layers.\(layerIndex).ple.ple_embedding.ngram_embedding"
            let dimensions = model.config.textConfig.pleEmbeddingDimensions
                / ((model.config.textConfig.ngramSize - 1) * model.config.textConfig.headsPerNgram)
            target.installDiskTable(try Q38DiskNGramTable(
                indexURL: placement?.indexURL ?? resources.modelIndexURL,
                base: base,
                shardCount: model.config.textConfig.splitNgramParts,
                dimensions: dimensions,
                minimumRowCount: target.minimumRowCount
            ))
        }
    }
}
