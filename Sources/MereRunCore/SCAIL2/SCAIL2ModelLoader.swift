import MLX
import MLXNN

public struct SCAIL2TextConditioning {
    public let prompt: MLXArray
    public let negativePrompt: MLXArray

    public init(prompt: MLXArray, negativePrompt: MLXArray) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
    }
}

public enum SCAIL2ModelLoader {
    static func transformerDType(for key: String) -> DType {
        if key.hasPrefix("time_embedding_")
            || key.hasPrefix("time_projection.")
            || key.hasPrefix("head.")
            || key.hasSuffix("modulation") {
            return .float32
        }
        return .bfloat16
    }

    public static func loadTokenizer(resources: SCAIL2Resources) throws -> Wan2Tokenizer {
        try Wan2Tokenizer.load(from: resources.tokenizerURL)
    }

    public static func loadTextEncoder(
        resources: SCAIL2Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Wan2TextEncoderModel {
        progress?("Loading SCAIL-2 UMT5 encoder")
        let model = Wan2TextEncoderModel()
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.textEncoderIndexURL,
            singleURL: resources.textEncoderURL,
            to: model,
            dtype: .bfloat16,
            verify: .none
        )
        eval(model.parameters().flattened().map(\.1))
        return model
    }

    public static func encodePrompts(
        tokenizer: Wan2Tokenizer,
        encoder: Wan2TextEncoderModel,
        prompt: String,
        negativePrompt: String
    ) throws -> SCAIL2TextConditioning {
        let conditioning = try Wan2ModelLoader.encodePrompts(
            tokenizer: tokenizer,
            encoder: encoder,
            prompt: prompt,
            negativePrompt: negativePrompt
        )
        return SCAIL2TextConditioning(
            prompt: conditioning.prompt,
            negativePrompt: conditioning.negativePrompt
        )
    }

    public static func loadCLIP(
        resources: SCAIL2Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> SCAIL2CLIPVisionModel {
        progress?("Loading SCAIL-2 OpenCLIP visual tower")
        let model = SCAIL2CLIPVisionModel()
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.clipURL,
            to: model,
            dtype: .float16,
            verify: .none
        )
        eval(model.parameters().flattened().map(\.1))
        return model
    }

    public static func loadVAE(
        resources: SCAIL2Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Wan2VAEModel {
        progress?("Loading SCAIL-2 Wan 2.1 VAE")
        let model = Wan2VAEModel(configuration: .wan21)
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.vaeURL,
            to: model,
            dtype: .float32,
            verify: .none
        )
        eval(model.parameters().flattened().map(\.1))
        return model
    }

    public static func loadTransformer(
        resources: SCAIL2Resources,
        configuration: SCAIL2Configuration,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> SCAIL2TransformerModel {
        progress?("Loading SCAIL-2 14B transformer")
        let model = SCAIL2TransformerModel(
            configuration: SCAIL2TransformerConfiguration(configuration)
        )
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.transformerIndexURL,
            singleURL: resources.transformerURL,
            to: model,
            dtype: nil,
            verify: .none,
            mapper: { key, value in
                [(key, value.asType(transformerDType(for: key)))]
            }
        )
        eval(model.parameters().flattened().map(\.1))
        return model
    }
}
