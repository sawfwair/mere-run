import Foundation

public struct LFM2VLVisionConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numChannels: Int
    public let numPatches: Int
    public let patchSize: Int
    public let layerNormEpsilon: Float

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numChannels = "num_channels"
        case numPatches = "num_patches"
        case patchSize = "patch_size"
        case layerNormEpsilon = "layer_norm_eps"
    }
}

public struct LFM2VLConfig: Decodable, Sendable, Hashable {
    public let architectures: [String]
    public let modelType: String
    public let textConfig: LFM2Config
    public let visionConfig: LFM2VLVisionConfig
    public let imageTokenId: Int
    public let imageTokenIndex: Int
    public let downsampleFactor: Int
    public let projectorBias: Bool
    public let projectorHiddenSize: Int
    public let projectorUseLayerNorm: Bool
    public let minImageTokens: Int
    public let maxImageTokens: Int
    public let maxNumPatches: Int
    public let encoderPatchSize: Int
    public let quantization: LFM2QuantizationConfig?

    private enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case imageTokenIndex = "image_token_index"
        case downsampleFactor = "downsample_factor"
        case projectorBias = "projector_bias"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorUseLayerNorm = "projector_use_layernorm"
        case minImageTokens = "min_image_tokens"
        case maxImageTokens = "max_image_tokens"
        case maxNumPatches = "max_num_patches"
        case encoderPatchSize = "encoder_patch_size"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        modelType = try container.decode(String.self, forKey: .modelType)
        textConfig = try container.decode(LFM2Config.self, forKey: .textConfig)
        visionConfig = try container.decode(LFM2VLVisionConfig.self, forKey: .visionConfig)
        imageTokenId = try container.decode(Int.self, forKey: .imageTokenId)
        imageTokenIndex = try container.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? imageTokenId
        downsampleFactor = try container.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 2
        projectorBias = try container.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        projectorHiddenSize = try container.decodeIfPresent(Int.self, forKey: .projectorHiddenSize)
            ?? textConfig.hiddenSize
        projectorUseLayerNorm = try container.decodeIfPresent(Bool.self, forKey: .projectorUseLayerNorm)
            ?? false
        minImageTokens = try container.decodeIfPresent(Int.self, forKey: .minImageTokens) ?? 64
        maxImageTokens = try container.decodeIfPresent(Int.self, forKey: .maxImageTokens) ?? 256
        maxNumPatches = try container.decodeIfPresent(Int.self, forKey: .maxNumPatches)
            ?? maxImageTokens * downsampleFactor * downsampleFactor
        encoderPatchSize = try container.decodeIfPresent(Int.self, forKey: .encoderPatchSize)
            ?? visionConfig.patchSize
        if let direct = try container.decodeIfPresent(LFM2QuantizationConfig.self, forKey: .quantization) {
            quantization = direct
        } else {
            quantization = try container.decodeIfPresent(
                LFM2QuantizationConfig.self,
                forKey: .quantizationConfig
            )
        }
    }
}

struct LFM2VLProcessorConfig: Decodable, Sendable, Hashable {
    struct ImageProcessor: Decodable, Sendable, Hashable {
        let downsampleFactor: Int
        let encoderPatchSize: Int
        let minImageTokens: Int
        let maxImageTokens: Int
        let maxNumPatches: Int

        private enum CodingKeys: String, CodingKey {
            case downsampleFactor = "downsample_factor"
            case encoderPatchSize = "encoder_patch_size"
            case minImageTokens = "min_image_tokens"
            case maxImageTokens = "max_image_tokens"
            case maxNumPatches = "max_num_patches"
        }
    }

    let imageProcessor: ImageProcessor

    private enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
    }
}
