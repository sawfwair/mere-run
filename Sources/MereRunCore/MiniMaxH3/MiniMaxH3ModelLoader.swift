import Foundation
import MLX
import MLXNN

public enum MiniMaxH3ModelLoaderError: LocalizedError, Sendable {
    case adaLNCacheRequired

    public var errorDescription: String? {
        switch self {
        case .adaLNCacheRequired:
            return "The compact MiniMax-H3 transformer requires its matching AdaLN cache."
        }
    }
}

public enum MiniMaxH3ModelLoader {
    public static func loadTransformer(
        resources: MiniMaxH3Resources,
        configuration: MiniMaxH3Configuration,
        progressHandler: (@Sendable (HFSafetensorsWeightsLoader.ShardProgress) -> Void)? = nil
    ) throws -> MiniMaxH3Transformer {
        try loadInferenceTransformer(
            resources: resources,
            configuration: configuration,
            cachedAdaLN: nil,
            progressHandler: progressHandler
        )
    }

    static func loadInferenceTransformer(
        resources: MiniMaxH3Resources,
        configuration: MiniMaxH3Configuration,
        cachedAdaLN: MiniMaxH3AdaLNCache?,
        progressHandler: (@Sendable (HFSafetensorsWeightsLoader.ShardProgress) -> Void)? = nil
    ) throws -> MiniMaxH3Transformer {
        let transformer = MiniMaxH3Transformer(
            configuration: .init(configuration),
            includeAdaLN: cachedAdaLN == nil
        )
        guard let layout = resources.transformerWeightsLayout() else {
            throw HFSafetensorsWeightsLoader.LoaderError.shardFileMissing(
                resources.transformerWeightsURL
            )
        }
        switch layout {
        case .shardedBF16(let indexURL):
            try HFSafetensorsWeightsLoader.applyShardedWeights(
                indexURL: indexURL,
                to: transformer,
                dtype: nil,
                verify: [.shapeMismatch],
                mapper: { key, value in
                    releasedBF16TransformerWeight(
                        key: key,
                        value: value,
                        omitCachedAdaLNWeights: cachedAdaLN != nil,
                        headCount: configuration.attentionHeadCount,
                        headDimension: configuration.attentionHeadDimension
                    )
                },
                progressHandler: progressHandler
            )
        case .single(let weightsURL):
            let (loadedWeights, metadata) = try MLX.loadArraysAndMetadata(url: weightsURL)
            if metadata["cache_covered_weights_omitted"] == "true", cachedAdaLN == nil {
                throw MiniMaxH3ModelLoaderError.adaLNCacheRequired
            }
            var weights = loadedWeights
            if cachedAdaLN != nil {
                weights = weights.filter { key, _ in
                    !key.contains(".adaln_proj.") && !key.hasPrefix("time_embedder.")
                }
            }
            if let quantization = configuration.quantization {
                try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                    weights,
                    to: transformer,
                    groupSize: quantization.groupSize,
                    bits: quantization.bits
                )
            } else {
                try transformer.update(
                    parameters: ModuleParameters.unflattened(weights.map { ($0.key, $0.value) }),
                    verify: [.shapeMismatch]
                )
            }
        }
        #if os(Linux)
        if configuration.quantization != nil,
           !resources.usesShardedBF16Transformer,
           Device.defaultDevice().deviceType == .gpu {
            for (_, module) in transformer.leafModules().flattened() {
                (module as? PortableQuantizedLinear)?.cacheDenseFallbackWeight = false
            }
            transformer.usesLayerwiseEvaluation = true
        }
        #endif
        return transformer
    }

    static func releasedBF16TransformerWeight(
        key: String,
        value: MLXArray,
        omitCachedAdaLNWeights: Bool,
        headCount: Int = 56,
        headDimension: Int = 128
    ) -> [(String, MLXArray)] {
        if omitCachedAdaLNWeights,
           (key.contains(".adaln_proj.") || key.hasPrefix("time_embedder.")) {
            return []
        }
        guard key.hasSuffix(".attn.qkv_proj.weight") else {
            return [(key, value)]
        }
        let expectedRows = headCount * 3 * headDimension
        precondition(value.ndim == 2 && value.dim(0) == expectedRows)
        let trailingShape = Array(value.shape.dropFirst())
        let grouped = value.reshaped([headCount, 3, headDimension] + trailingShape)
        let pieces = MLX.split(grouped, parts: 3, axis: 1).map {
            $0.squeezed(axis: 1).reshaped([headCount * headDimension] + trailingShape)
        }
        return [(key, MLX.concatenated(pieces, axis: 0))]
    }

    public static func loadConditioner(
        resources: MiniMaxH3Resources,
        configuration: MiniMaxH3Configuration,
        progressHandler: (@Sendable (HFSafetensorsWeightsLoader.ShardProgress) -> Void)? = nil
    ) throws -> QwenVLEncoder {
        let textConfiguration = QwenTextEncoderConfiguration(
            vocabSize: 151_936,
            hiddenSize: 5_120,
            numHiddenLayers: 50,
            numAttentionHeads: 64,
            numKeyValueHeads: 8,
            intermediateSize: 25_600,
            ropeTheta: 5_000_000,
            maxPositionEmbeddings: 262_144,
            rmsNormEps: 1e-6,
            headDim: 128,
            mropeSection: [24, 20, 20],
            mropeInterleaved: true
        )
        let visionConfiguration = QwenVisionConfiguration(
            depth: 27,
            embedDim: 1_152,
            mlpHiddenDim: 4_304,
            hiddenAct: .geluApproximate,
            numHeads: 16,
            patchSize: 16,
            temporalPatchSize: 2,
            spatialMergeSize: 2,
            inChannels: 3,
            outHiddenDim: 5_120,
            windowSize: 112,
            fullAttentionBlockIndices: [],
            patchEmbedBias: true,
            numPositionEmbeddings: 2_304,
            useLearnedPosEmbed: true,
            deepstackVisualIndexes: [8, 16, 24]
        )
        let encoder = QwenVLEncoder(
            textEncoderConfig: textConfiguration,
            visionConfig: visionConfiguration
        )
        if let quantization = configuration.textEncoderQuantization {
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                MLX.loadArrays(url: resources.textEncoderWeightsURL),
                to: encoder,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                keyMapper: conditionerWeightKey,
                mapper: conditionerWeight
            )
        } else {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: resources.textEncoderWeightsURL,
                to: encoder,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                mapper: { rawKey, value in
                    conditionerWeight(key: conditionerWeightKey(rawKey), value: value)
                }
            )
        }
        return encoder
    }

    public static func loadVideoVAE(
        resources: MiniMaxH3Resources,
        progressHandler: (@Sendable (HFSafetensorsWeightsLoader.ShardProgress) -> Void)? = nil
    ) throws -> MiniMaxH3VideoVAE {
        let model = MiniMaxH3VideoVAE()
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.videoVAEWeightsURL,
            to: model,
            dtype: nil,
            verify: [.shapeMismatch],
            mapper: MiniMaxH3VideoVAE.mapCheckpointWeight
        )
        return model
    }

    public static func loadAudioVAE(
        resources: MiniMaxH3Resources,
        progressHandler: (@Sendable (HFSafetensorsWeightsLoader.ShardProgress) -> Void)? = nil
    ) throws -> MiniMaxH3AudioVAE {
        let model = MiniMaxH3AudioVAE()
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.audioVAEWeightsURL,
            to: model,
            dtype: .float32,
            verify: [.shapeMismatch],
            mapper: MiniMaxH3AudioVAE.mapConvertedWeight
        )
        return model
    }

    static func conditionerWeightKey(_ rawKey: String) -> String {
        var key = rawKey
        if key.hasPrefix("model.") {
            key = "textEncoder.encoder." + String(key.dropFirst("model.".count))
        } else if key.hasPrefix("visual.") {
            key = "visionTower." + String(key.dropFirst("visual.".count))
        }
        key = key.replacingOccurrences(of: ".merger.", with: ".patch_merger.")
        key = key.replacingOccurrences(of: ".patch_merger.norm.", with: ".patch_merger.ln_q.")
        key = key.replacingOccurrences(of: ".patch_merger.linear_fc1.", with: ".patch_merger.mlp_0.")
        key = key.replacingOccurrences(of: ".patch_merger.linear_fc2.", with: ".patch_merger.mlp_2.")
        if key.contains(".deepstack_merger_list.") {
            key = key.replacingOccurrences(of: ".norm.", with: ".ln_q.")
            key = key.replacingOccurrences(of: ".linear_fc1.", with: ".mlp_0.")
            key = key.replacingOccurrences(of: ".linear_fc2.", with: ".mlp_2.")
        }
        key = key.replacingOccurrences(of: ".mlp.linear_fc1.", with: ".mlp.fc1.")
        key = key.replacingOccurrences(of: ".mlp.linear_fc2.", with: ".mlp.fc2.")
        return key
    }

    static func conditionerWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key == "visionTower.patch_embed.proj.weight", value.ndim == 5 {
            return [(key, value.transposed(0, 2, 3, 4, 1))]
        }
        return [(key, value)]
    }
}
