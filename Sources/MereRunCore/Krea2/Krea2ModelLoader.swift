import Foundation
import MLX

public enum Krea2ModelLoader {
    public static func loadTransformer(
        from resources: Krea2Resources,
        configuration: Krea2TransformerConfiguration,
        dtype: DType? = .bfloat16,
        progressHandler: (@Sendable (HFSafetensorsWeightsLoader.ShardProgress) -> Void)? = nil
    ) throws -> Krea2Transformer {
        let transformer = Krea2Transformer(configuration: configuration)
        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: resources.transformerWeightsIndexURL,
            to: transformer,
            dtype: dtype,
            verify: [.shapeMismatch],
            mapper: mapTransformerWeight,
            progressHandler: progressHandler
        )
        return transformer
    }

    public static func loadTextEncoder(
        from resources: Krea2Resources,
        configuration: Krea2TextEncoderConfiguration,
        dtype: DType? = .bfloat16
    ) throws -> QwenEncoder {
        let encoder = QwenEncoder(configuration: configuration.qwenConfiguration)
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.textEncoderWeightsURL,
            to: encoder,
            dtype: dtype,
            verify: [.shapeMismatch],
            mapper: mapTextEncoderWeight
        )
        return encoder
    }

    public static func loadVAE(
        from resources: Krea2Resources,
        configuration: QwenImageEditVAEConfig,
        dtype: DType? = .bfloat16
    ) throws -> QwenImageEditVAE {
        let vae = QwenImageEditVAE(config: configuration)
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.vaeWeightsURL,
            to: vae.underlyingVAE,
            dtype: dtype,
            verify: [.shapeMismatch],
            mapper: QwenImageEditVAE.weightMapper
        )
        return vae
    }

    static func mapTransformerWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        [(key, value)]
    }

    static func mapTextEncoderWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        guard !key.hasPrefix("lm_head."), key != "lm_head" else {
            return []
        }
        if key.hasPrefix("language_model.lm_head") {
            return []
        }
        if key.hasPrefix("language_model.") {
            return QwenEncoder.mapHFSafetensorWeight(
                key: String(key.dropFirst("language_model.".count)),
                value: value
            )
        }
        return QwenEncoder.mapHFSafetensorWeight(key: key, value: value)
    }
}
