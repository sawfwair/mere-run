import Foundation

public struct GLM47FlashQuantization: Codable, Sendable, Hashable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String

    private enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }
}

public struct GLM47FlashConfig: Codable, Sendable, Hashable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let nSharedExperts: Int?
    public let nRoutedExperts: Int?
    public let routedScalingFactor: Float
    public let kvLoraRank: Int
    public let qLoraRank: Int
    public let qkNopeHeadDim: Int
    public let qkRopeHeadDim: Int
    public let vHeadDim: Int
    public let topkMethod: String
    public let normTopkProb: Bool
    public let nGroup: Int
    public let topkGroup: Int
    public let numExpertsPerTok: Int
    public let firstKDenseReplace: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let ropeScaling: [String: Float]?
    public let ropeTraditional: Bool?
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let partialRotaryFactor: Float
    public let tieWordEmbeddings: Bool
    public let numNextNPredictLayers: Int?
    public let eosTokenId: [Int]?
    public let padTokenId: Int?
    public let quantization: GLM47FlashQuantization?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case topkMethod = "topk_method"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeTraditional = "rope_traditional"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case partialRotaryFactor = "partial_rotary_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numNextNPredictLayers = "num_nextn_predict_layers"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case quantization
    }
}
