import Foundation
import MLX

public enum Wan2ModelLoaderError: LocalizedError, Sendable {
    case textEncodingFailed

    public var errorDescription: String? {
        switch self {
        case .textEncodingFailed:
            return "Wan2 text encoder did not return the expected batch."
        }
    }
}

public struct Wan2TextConditioning {
    public let prompt: MLXArray
    public let negativePrompt: MLXArray
}

public enum Wan2ModelLoader {
    public static func loadTokenizer(resources: Wan2Resources) throws -> Wan2Tokenizer {
        try Wan2Tokenizer.load(from: resources.tokenizerURL)
    }

    public static func loadTextEncoder(
        resources: Wan2Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Wan2TextEncoderModel {
        progress?("Loading UMT5 encoder")
        let encoder = Wan2TextEncoderModel()
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: resources.textEncoderURL,
            to: encoder,
            dtype: .bfloat16,
            verify: .none,
            batchSize: 12
        )
        eval(encoder.parameters().flattened().map(\.1))
        return encoder
    }

    public static func encodePrompts(
        tokenizer: Wan2Tokenizer,
        encoder: Wan2TextEncoderModel,
        prompt: String,
        negativePrompt: String
    ) throws -> Wan2TextConditioning {
        let promptTokens = tokenizer.encode(prompt)
        let negativeTokens = tokenizer.encode(negativePrompt)
        let promptLength = max(promptTokens.mask.reduce(0, +), 1)
        let negativeLength = max(negativeTokens.mask.reduce(0, +), 1)
        let encodedLength = max(promptLength, negativeLength)
        let promptIDs = Array(promptTokens.tokenIDs.prefix(encodedLength))
        let negativeIDs = Array(negativeTokens.tokenIDs.prefix(encodedLength))
        let promptMask = Array(promptTokens.mask.prefix(encodedLength))
        let negativeMask = Array(negativeTokens.mask.prefix(encodedLength))
        let tokenIDs = MLXArray(promptIDs + negativeIDs, [2, encodedLength])
        let mask = MLXArray(promptMask + negativeMask, [2, encodedLength])
        let encoded = encoder(tokenIDs: tokenIDs, mask: mask)
        eval(encoded)
        guard encoded.dim(0) == 2 else {
            throw Wan2ModelLoaderError.textEncodingFailed
        }
        return Wan2TextConditioning(
            prompt: encoded[0, 0..<promptLength].asType(.bfloat16),
            negativePrompt: encoded[1, 0..<negativeLength].asType(.bfloat16)
        )
    }

    public static func encodePrompts(
        resources: Wan2Resources,
        prompt: String,
        negativePrompt: String,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Wan2TextConditioning {
        progress?("Loading UMT5 tokenizer")
        let tokenizer = try loadTokenizer(resources: resources)
        let encoder = try loadTextEncoder(resources: resources, progress: progress)
        return try encodePrompts(
            tokenizer: tokenizer,
            encoder: encoder,
            prompt: prompt,
            negativePrompt: negativePrompt
        )
    }

    public static func loadTransformer(
        resources: Wan2Resources,
        dtype: DType = .bfloat16,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Wan2TransformerModel {
        progress?("Loading Wan2.2 TI2V transformer")
        let model = Wan2TransformerModel()
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: resources.transformerURL,
            to: model,
            dtype: nil,
            verify: .none,
            mapper: { key, value in
                let targetType: DType = key.hasSuffix("modulation") ? .float32 : dtype
                return [(key, value.asType(targetType))]
            },
            batchSize: 12
        )
        eval(model.parameters().flattened().map(\.1))
        return model
    }

    public static func loadVAE(
        resources: Wan2Resources,
        dtype: DType = .float32,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Wan2VAEModel {
        progress?("Loading Wan2.2 TI2V VAE")
        let model = Wan2VAEModel()
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: resources.vaeURL,
            to: model,
            dtype: dtype,
            verify: .none,
            batchSize: 12
        )
        eval(model.parameters().flattened().map(\.1))
        return model
    }
}
