import Foundation
import MLX
import MLXNN

public enum Wan2ModelLoaderError: LocalizedError, Sendable {
    case textEncodingFailed
    case invalidDreamXCameraWeights(expected: Int, actual: Int)
    case invalidDreamXCausalWeights(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .textEncodingFailed:
            return "Wan2 text encoder did not return the expected batch."
        case .invalidDreamXCameraWeights(let expected, let actual):
            return "DreamX camera adapter must contain \(expected) tensors, found \(actual)."
        case .invalidDreamXCausalWeights(let expected, let actual):
            return "DreamX causal checkpoint must contain \(expected) tensors, found \(actual)."
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
        try applyTransformerWeights(
            url: resources.transformerURL,
            to: model,
            dtype: dtype,
            verify: .none
        )
        eval(model.parameters().flattened().map(\.1))
        return model
    }

    public static func loadDreamXCameraTransformer(
        resources: Wan2Resources,
        cameraWeightsURL: URL,
        dtype: DType = .bfloat16,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Wan2TransformerModel {
        let metadata = try SafetensorsStreamingLoader.metadata(url: cameraWeightsURL)
        let cameraTensorCount = metadata.keys.filter { $0.contains(".cam_self_attn.") }.count
        guard cameraTensorCount == 300, metadata.count == 300 else {
            throw Wan2ModelLoaderError.invalidDreamXCameraWeights(
                expected: 300,
                actual: cameraTensorCount
            )
        }
        progress?("Loading Wan2.2 TI2V transformer")
        let model = Wan2TransformerModel(configuration: Wan2TransformerConfiguration(
            projectiveCameraConditioning: true
        ))
        try applyTransformerWeights(
            url: resources.transformerURL,
            to: model,
            dtype: dtype,
            verify: .none
        )
        progress?("Loading DreamX projective camera attention")
        try applyTransformerWeights(
            url: cameraWeightsURL,
            to: model,
            dtype: dtype,
            verify: .noUnusedKeys
        )
        eval(model.parameters().flattened().map(\.1))
        return model
    }

    public static func loadDreamXCausalTransformer(
        weightsURL: URL,
        dtype: DType = .bfloat16,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Wan2TransformerModel {
        let metadata = try SafetensorsStreamingLoader.metadata(url: weightsURL)
        guard metadata.count == 1_125 else {
            throw Wan2ModelLoaderError.invalidDreamXCausalWeights(
                expected: 1_125,
                actual: metadata.count
            )
        }
        let cameraPrefix = "blocks.0.cam_self_attn."
        guard metadata[cameraPrefix + "q_proj.weight"]?.shape == [768, 3_072],
              metadata[cameraPrefix + "out_proj.weight"]?.shape == [3_072, 768] else {
            throw Wan2ModelLoaderError.invalidDreamXCausalWeights(
                expected: 1_125,
                actual: metadata.count
            )
        }
        progress?("Loading DreamX causal Wan2 transformer")
        let model = Wan2TransformerModel(configuration: Wan2TransformerConfiguration(
            projectiveCameraConditioning: true,
            projectiveCameraAttentionCompression: 4
        ))
        try applyTransformerWeights(
            url: weightsURL,
            to: model,
            dtype: dtype,
            verify: .noUnusedKeys
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

    private static func applyTransformerWeights(
        url: URL,
        to model: Wan2TransformerModel,
        dtype: DType,
        verify: Module.VerifyUpdate
    ) throws {
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: url,
            to: model,
            dtype: nil,
            verify: verify,
            mapper: { key, value in
                let targetType: DType = key.hasSuffix("modulation") ? .float32 : dtype
                return [(key, value.asType(targetType))]
            },
            batchSize: 12
        )
    }
}
