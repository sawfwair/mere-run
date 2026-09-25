import Foundation

public struct GLiNEREncoderConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let positionBuckets: Int
    public let layerNormEps: Float
    public let padTokenID: Int
    public let relativeAttention: Bool
    public let shareAttKey: Bool
    public let positionBiasedInput: Bool
    public let posAttType: [String]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads", numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size", maxPositionEmbeddings = "max_position_embeddings"
        case positionBuckets = "position_buckets", layerNormEps = "layer_norm_eps"
        case padTokenID = "pad_token_id", relativeAttention = "relative_attention"
        case shareAttKey = "share_att_key", positionBiasedInput = "position_biased_input"
        case posAttType = "pos_att_type"
    }

    public func validate() throws {
        guard hiddenSize == 1024, intermediateSize == 4096, numAttentionHeads == 16,
              numHiddenLayers == 24, vocabSize == 128_011, maxPositionEmbeddings == 512,
              positionBuckets == 256, padTokenID == 0, relativeAttention,
              shareAttKey, !positionBiasedInput, Set(posAttType) == Set(["c2p", "p2c"]),
              layerNormEps > 0 else {
            throw GLiNERModelError.unsupportedConfiguration
        }
    }
}

public enum GLiNERModelError: LocalizedError {
    case unsupportedConfiguration
    case invalidWeights(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedConfiguration: "Unsupported GLiNER2.5 DeBERTa configuration."
        case .invalidWeights(let detail): "Invalid GLiNER2.5 checkpoint: \(detail)"
        }
    }
}
