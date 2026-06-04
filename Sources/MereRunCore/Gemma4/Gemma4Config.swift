import Foundation

public struct Gemma4TextRopeParameters: Decodable, Sendable, Hashable {
    public let ropeType: String?
    public let ropeTheta: Float
    public let partialRotaryFactor: Float?

    private enum CodingKeys: String, CodingKey {
        case ropeType = "rope_type"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
    }
}

public struct Gemma4TextConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numGlobalKeyValueHeads: Int?
    public let headDim: Int
    public let globalHeadDim: Int?
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let vocabSizePerLayerInput: Int
    public let hiddenSizePerLayerInput: Int
    public let padTokenId: Int?
    public let ropeParameters: [String: Gemma4TextRopeParameters]
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let attentionKEqV: Bool
    public let finalLogitSoftcapping: Float?
    public let useDoubleWideMLP: Bool
    public let enableMoEBlock: Bool
    public let numExperts: Int
    public let topKExperts: Int
    public let moeIntermediateSize: Int
    public let numKVSharedLayers: Int
    public let tieWordEmbeddings: Bool

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case padTokenId = "pad_token_id"
        case ropeParameters = "rope_parameters"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case attentionKEqV = "attention_k_eq_v"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMLP = "use_double_wide_mlp"
        case enableMoEBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case numKVSharedLayers = "num_kv_shared_layers"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.numGlobalKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        self.vocabSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput)
            ?? self.vocabSize
        self.hiddenSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput)
            ?? 0
        self.padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId)
        self.ropeParameters = try container.decodeIfPresent([String: Gemma4TextRopeParameters].self, forKey: .ropeParameters)
            ?? Gemma4TextConfig.defaultRopeParameters
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Gemma4TextConfig.defaultLayerTypes(numHiddenLayers: self.numHiddenLayers)
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0
        self.attentionKEqV = try container.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        self.finalLogitSoftcapping = try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        self.useDoubleWideMLP = try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMLP) ?? false
        self.enableMoEBlock = try container.decodeIfPresent(Bool.self, forKey: .enableMoEBlock) ?? false
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts) ?? 0
        self.moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.numKVSharedLayers = try container.decodeIfPresent(Int.self, forKey: .numKVSharedLayers) ?? 0
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }

    private static let defaultRopeParameters: [String: Gemma4TextRopeParameters] = [
        "sliding_attention": Gemma4TextRopeParameters(
            ropeType: "default",
            ropeTheta: 10_000,
            partialRotaryFactor: 1.0
        ),
        "full_attention": Gemma4TextRopeParameters(
            ropeType: "proportional",
            ropeTheta: 1_000_000,
            partialRotaryFactor: 1.0
        ),
    ]

    private static func defaultLayerTypes(numHiddenLayers: Int) -> [String] {
        let pattern = ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
        var result: [String] = []
        result.reserveCapacity(numHiddenLayers)
        while result.count < numHiddenLayers {
            result.append(contentsOf: pattern)
        }
        return Array(result.prefix(numHiddenLayers))
    }
}

public struct Gemma4UnifiedVisionConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let modelPatchSize: Int
    public let mmEmbedDim: Int
    public let mmPosembSize: Int
    public let numSoftTokens: Int
    public let rmsNormEps: Float
    public let outputProjDims: Int

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case modelPatchSize = "model_patch_size"
        case mmEmbedDim = "mm_embed_dim"
        case mmPosembSize = "mm_posemb_size"
        case numSoftTokens = "num_soft_tokens"
        case rmsNormEps = "rms_norm_eps"
        case outputProjDims = "output_proj_dims"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_unified_vision"
        self.patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        self.poolingKernelSize = try container.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        self.modelPatchSize = try container.decodeIfPresent(Int.self, forKey: .modelPatchSize)
            ?? (self.patchSize * self.poolingKernelSize)
        self.mmEmbedDim = try container.decodeIfPresent(Int.self, forKey: .mmEmbedDim) ?? 3_840
        self.mmPosembSize = try container.decodeIfPresent(Int.self, forKey: .mmPosembSize) ?? 1_120
        self.numSoftTokens = try container.decodeIfPresent(Int.self, forKey: .numSoftTokens) ?? 280
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.outputProjDims = try container.decodeIfPresent(Int.self, forKey: .outputProjDims) ?? self.mmEmbedDim
    }
}

public struct Gemma4Config: Decodable, Sendable, Hashable {
    public let modelType: String
    public let architectures: [String]
    public let tieWordEmbeddings: Bool
    public let eosTokenIds: [Int]
    public let textConfig: Gemma4TextConfig
    public let visionConfig: Gemma4UnifiedVisionConfig?
    public let imageTokenId: Int?
    public let boiTokenId: Int?
    public let eoiTokenId: Int?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case tieWordEmbeddings = "tie_word_embeddings"
        case eosTokenId = "eos_token_id"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.textConfig = try container.decode(Gemma4TextConfig.self, forKey: .textConfig)
        self.visionConfig = try container.decodeIfPresent(Gemma4UnifiedVisionConfig.self, forKey: .visionConfig)
        self.imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId)
        self.boiTokenId = try container.decodeIfPresent(Int.self, forKey: .boiTokenId)
        self.eoiTokenId = try container.decodeIfPresent(Int.self, forKey: .eoiTokenId)

        if let direct = try container.decodeIfPresent([Int].self, forKey: .eosTokenId) {
            self.eosTokenIds = direct
        } else if let single = try container.decodeIfPresent(Int.self, forKey: .eosTokenId) {
            self.eosTokenIds = [single]
        } else {
            self.eosTokenIds = []
        }
    }
}
