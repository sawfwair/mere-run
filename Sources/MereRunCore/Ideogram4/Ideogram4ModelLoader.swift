import Foundation
import MLX

public enum Ideogram4ModelLoader {
    public static func loadConditionalTransformer(
        from resources: Ideogram4Resources,
        dtype: DType? = .bfloat16
    ) throws -> Ideogram4Transformer {
        try loadTransformer(from: resources.transformerURL, dtype: dtype)
    }

    public static func loadUnconditionalTransformer(
        from resources: Ideogram4Resources,
        dtype: DType? = .bfloat16
    ) throws -> Ideogram4Transformer {
        try loadTransformer(from: resources.unconditionalTransformerURL, dtype: dtype)
    }

    public static func loadTransformer(
        from directoryURL: URL,
        dtype: DType? = .bfloat16
    ) throws -> Ideogram4Transformer {
        let configuration = try Ideogram4TransformerConfiguration.load(from: directoryURL)
        let transformer = Ideogram4Transformer(configuration: configuration)
        try SDNQWeightsLoader.applyWeights(
            url: directoryURL.appending(path: "diffusion_pytorch_model.safetensors"),
            to: transformer,
            dtype: dtype
        )
        return transformer
    }

    public static func makeTextEncoder(from resources: Ideogram4Resources) throws -> QwenEncoder {
        let configuration = try Ideogram4TextEncoderConfiguration.load(from: resources.textEncoderURL)
        return QwenEncoder(configuration: configuration.qwenConfiguration)
    }

    public static func loadTextEncoder(
        from resources: Ideogram4Resources,
        dtype: DType? = .bfloat16,
        progressHandler: (@Sendable (HFSafetensorsWeightsLoader.ShardProgress) -> Void)? = nil
    ) throws -> QwenEncoder {
        let encoder = try makeTextEncoder(from: resources)
        let indexURL = resources.textEncoderURL.appending(path: "model.safetensors.index.json")
        let singleURL = resources.textEncoderURL.appending(path: "model.safetensors")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            try SDNQWeightsLoader.applyShardedWeights(
                indexURL: indexURL,
                to: encoder,
                dtype: dtype,
                progressHandler: progressHandler
            )
        } else {
            try SDNQWeightsLoader.applyWeights(
                url: singleURL,
                to: encoder,
                dtype: dtype,
                allowPlainUpdates: true
            )
        }
        return encoder
    }

    public static func loadVAE(
        from resources: Ideogram4Resources,
        dtype: DType? = .bfloat16
    ) throws -> AutoencoderKL {
        let configuration = try Ideogram4VAEConfiguration.load(from: resources.vaeURL)
        let vae = AutoencoderKL(configuration: configuration.autoencoderConfiguration)
        try SDNQWeightsLoader.applyWeights(
            url: resources.vaeURL.appending(path: "diffusion_pytorch_model.safetensors"),
            to: vae,
            dtype: dtype,
            keyMapper: Ideogram4VAEWeights.mapKey,
            allowPlainUpdates: true,
            mapper: Ideogram4VAEWeights.mapParameter
        )
        return vae
    }
}
