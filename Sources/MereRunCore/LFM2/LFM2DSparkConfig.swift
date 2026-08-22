import Foundation

public struct LFM2DSparkFeatureConfig: Decodable, Sendable, Hashable {
    public let maskTokenID: Int
    public let targetLayerIDs: [Int]
    public let targetLayerCount: Int

    private enum CodingKeys: String, CodingKey {
        case maskTokenID = "mask_token_id"
        case targetLayerIDs = "target_layer_ids"
        case targetLayerCount = "num_target_layers"
    }
}

public struct LFM2DSparkConfig: Decodable, Sendable, Hashable {
    public let architectures: [String]
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayerCount: Int
    public let attentionHeadCount: Int
    public let keyValueHeadCount: Int
    public let headDimensions: Int
    public let intermediateSize: Int
    public let normEpsilon: Float
    public let vocabularySize: Int
    public let ropeTheta: Float
    public let maximumPositionEmbeddings: Int
    public let layerTypes: [String]
    public let blockSize: Int
    public let features: LFM2DSparkFeatureConfig
    public let markovRank: Int
    public let ropeUsesNeoXLayout: Bool
    public let confidenceHeadEnabled: Bool
    public let markovHeadType: String

    private enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayerCount = "num_hidden_layers"
        case attentionHeadCount = "num_attention_heads"
        case keyValueHeadCount = "num_key_value_heads"
        case headDimensions = "head_dim"
        case intermediateSize = "intermediate_size"
        case normEpsilon = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case maximumPositionEmbeddings = "max_position_embeddings"
        case layerTypes = "layer_types"
        case blockSize = "block_size"
        case features = "dflash_config"
        case markovRank = "markov_rank"
        case ropeUsesNeoXLayout = "rope_is_neox_style"
        case confidenceHeadEnabled = "enable_confidence_head"
        case markovHeadType = "markov_head_type"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        architectures = try container.decode([String].self, forKey: .architectures)
        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayerCount = try container.decode(Int.self, forKey: .hiddenLayerCount)
        attentionHeadCount = try container.decode(Int.self, forKey: .attentionHeadCount)
        keyValueHeadCount = try container.decode(Int.self, forKey: .keyValueHeadCount)
        headDimensions = try container.decode(Int.self, forKey: .headDimensions)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        normEpsilon = try container.decode(Float.self, forKey: .normEpsilon)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        maximumPositionEmbeddings = try container.decode(Int.self, forKey: .maximumPositionEmbeddings)
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        blockSize = try container.decode(Int.self, forKey: .blockSize)
        features = try container.decode(LFM2DSparkFeatureConfig.self, forKey: .features)
        markovRank = try container.decode(Int.self, forKey: .markovRank)
        ropeUsesNeoXLayout = try container.decode(Bool.self, forKey: .ropeUsesNeoXLayout)
        confidenceHeadEnabled = try container.decode(Bool.self, forKey: .confidenceHeadEnabled)
        markovHeadType = try container.decode(String.self, forKey: .markovHeadType)

        guard architectures == ["Lfm2DSparkDraftModel"],
              modelType == "qwen3",
              hiddenLayerCount == 5,
              layerTypes == Array(repeating: "full_attention", count: hiddenLayerCount),
              blockSize == 9,
              markovRank == 256,
              !ropeUsesNeoXLayout,
              confidenceHeadEnabled,
              markovHeadType == "vanilla",
              features.targetLayerIDs.count == hiddenLayerCount else {
            throw DecodingError.dataCorruptedError(
                forKey: .architectures,
                in: container,
                debugDescription: "Unsupported LiquidAI LFM2.5 DSpark checkpoint contract."
            )
        }
    }
}
