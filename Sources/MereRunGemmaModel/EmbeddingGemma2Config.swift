import Foundation

public enum EmbeddingGemma2Error: LocalizedError {
    case invalidConfiguration(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message), .invalidInput(let message): return message
        }
    }
}

/// Published EmbeddingGemma 2 encoder configuration, independent of causal Gemma 4.
public struct EmbeddingGemma2Config: Decodable, Sendable {
    public let modelType: String
    public let textConfig: TextConfig
    public let visionConfig: EmbeddingGemma2VisionConfig?
    public let audioConfig: EmbeddingGemma2AudioConfig?
    public let imageTokenID: Int?
    public let videoTokenID: Int?
    public let audioTokenID: Int?
    public let boiTokenID: Int?
    public let eoiTokenID: Int?
    public let boaTokenID: Int?
    public let eoaTokenID: Int?
    public static let contextLength = 8_192
    public static let outputDimensions = [128, 256, 512, 768]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config", audioConfig = "audio_config"
        case imageTokenID = "image_token_id", videoTokenID = "video_token_id", audioTokenID = "audio_token_id"
        case boiTokenID = "boi_token_id", eoiTokenID = "eoi_token_id"
        case boaTokenID = "boa_token_id", eoaTokenID = "eoa_token_index"
    }

    public struct LayerConfig: Decodable, Sendable {
        public let headDim: Int
        public let numKeyValueHeads: Int
        enum CodingKeys: String, CodingKey {
            case headDim = "head_dim"
            case numKeyValueHeads = "num_key_value_heads"
        }
    }

    public struct TextConfig: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let headDim: Int
        public let hiddenSizePerLayerInput: Int
        public let embeddingDim: Int
        public let vocabSize: Int
        public let maxPositionEmbeddings: Int
        public let rmsNormEps: Float
        public let slidingWindow: Int
        public let layerTypes: [String]
        public let perLayerConfig: [String: LayerConfig]
        public let ropeParameters: [String: Gemma4TextRopeParameters]
        public let hiddenActivation: String
        public let attentionBias: Bool
        public let bosTokenID: Int
        public let eosTokenID: Int
        public let padTokenID: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type", hiddenSize = "hidden_size", intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers", numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads", headDim = "head_dim"
            case hiddenSizePerLayerInput = "hidden_size_per_layer_input", embeddingDim = "embedding_dim"
            case vocabSize = "vocab_size", maxPositionEmbeddings = "max_position_embeddings"
            case rmsNormEps = "rms_norm_eps", slidingWindow = "sliding_window", layerTypes = "layer_types"
            case perLayerConfig = "per_layer_config", ropeParameters = "rope_parameters"
            case hiddenActivation = "hidden_activation", attentionBias = "attention_bias"
            case bosTokenID = "bos_token_id", eosTokenID = "eos_token_id", padTokenID = "pad_token_id"
        }

        public func layerConfig(at index: Int) -> LayerConfig {
            perLayerConfig[String(format: "%02d", index)] ?? perLayerConfig[String(index)]
                ?? LayerConfig(headDim: headDim, numKeyValueHeads: numKeyValueHeads)
        }
    }

    public func validate() throws {
        let text = textConfig
        guard modelType == "embedding_gemma2", text.modelType == "embedding_gemma2_text",
              text.hiddenActivation == "gelu_pytorch_tanh", !text.attentionBias else {
            throw EmbeddingGemma2Error.invalidConfiguration("Unsupported EmbeddingGemma 2 model or activation configuration.")
        }
        guard [text.hiddenSize, text.intermediateSize, text.numHiddenLayers, text.numAttentionHeads,
               text.numKeyValueHeads, text.headDim, text.hiddenSizePerLayerInput, text.embeddingDim,
               text.vocabSize, text.maxPositionEmbeddings, text.slidingWindow].allSatisfy({ $0 > 0 }),
              text.rmsNormEps.isFinite, text.rmsNormEps > 0,
              text.layerTypes.count == text.numHiddenLayers,
              [text.bosTokenID, text.eosTokenID, text.padTokenID].allSatisfy({ (0..<text.vocabSize).contains($0) }) else {
            throw EmbeddingGemma2Error.invalidConfiguration("Invalid EmbeddingGemma 2 dimensions, layers, or token IDs.")
        }
        for index in 0..<text.numHiddenLayers {
            let layer = text.layerConfig(at: index)
            let type = text.layerTypes[index]
            guard ["full_attention", "sliding_attention"].contains(type),
                  let rope = text.ropeParameters[type], rope.ropeType == "default",
                  rope.ropeTheta.isFinite, rope.ropeTheta > 0,
                  rope.partialRotaryFactor == nil || rope.partialRotaryFactor == 1,
                  layer.headDim > 0, layer.headDim.isMultiple(of: 2), layer.numKeyValueHeads > 0,
                  text.numAttentionHeads.isMultiple(of: layer.numKeyValueHeads) else {
                throw EmbeddingGemma2Error.invalidConfiguration("Unsupported EmbeddingGemma 2 attention layer \(index).")
            }
        }
    }
}
