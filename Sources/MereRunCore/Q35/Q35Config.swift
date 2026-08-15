import Foundation

public struct Q35GenerationConfig: Codable, Sendable, Hashable {
    public let eosTokenIds: [Int]

    private enum CodingKeys: String, CodingKey {
        case eosTokenId = "eos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let single = try? container.decode(Int.self, forKey: .eosTokenId) {
            eosTokenIds = [single]
        } else {
            eosTokenIds = try container.decodeIfPresent([Int].self, forKey: .eosTokenId) ?? []
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(eosTokenIds, forKey: .eosTokenId)
    }
}

public struct Q35QuantizationConfig: Codable, Sendable, Hashable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String

    private enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }

    public init(groupSize: Int, bits: Int, mode: String = "affine") {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        groupSize = try container.decode(Int.self, forKey: .groupSize)
        bits = try container.decode(Int.self, forKey: .bits)
        mode = try container.decodeIfPresent(String.self, forKey: .mode) ?? "affine"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(groupSize, forKey: .groupSize)
        try container.encode(bits, forKey: .bits)
        try container.encode(mode, forKey: .mode)
    }
}

public struct Q35RopeParameters: Codable, Sendable, Hashable {
    public let mropeInterleaved: Bool?
    public let mropeSection: [Int]?
    public let ropeType: String?
    public let ropeTheta: Float
    public let partialRotaryFactor: Float

    private enum CodingKeys: String, CodingKey {
        case mropeInterleaved = "mrope_interleaved"
        case mropeSection = "mrope_section"
        case ropeType = "rope_type"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
    }
}

public struct Q35TextConfig: Codable, Sendable, Hashable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let moeIntermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let normTopKProb: Bool
    public let layerTypes: [String]
    public let mlpOnlyLayers: [Int]
    public let linearNumValueHeads: Int
    public let linearNumKeyHeads: Int
    public let linearKeyHeadDim: Int
    public let linearValueHeadDim: Int
    public let linearConvKernelDim: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let attnOutputGate: Bool
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let fullAttentionInterval: Int?
    public let vocabSize: Int
    public let eosTokenId: Int?
    public let ropeParameters: Q35RopeParameters

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case normTopKProb = "norm_topk_prob"
        case layerTypes = "layer_types"
        case mlpOnlyLayers = "mlp_only_layers"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case attnOutputGate = "attn_output_gate"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case fullAttentionInterval = "full_attention_interval"
        case vocabSize = "vocab_size"
        case eosTokenId = "eos_token_id"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)

        let decodedIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
        let decodedSharedIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize)
        let decodedMoeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)

        guard let resolvedIntermediateSize = decodedIntermediateSize ?? decodedSharedIntermediateSize ?? decodedMoeIntermediateSize else {
            throw DecodingError.dataCorruptedError(
                forKey: .intermediateSize,
                in: container,
                debugDescription: "Missing intermediate size: expected one of text_config.intermediate_size, text_config.shared_expert_intermediate_size, or text_config.moe_intermediate_size."
            )
        }

        self.intermediateSize = resolvedIntermediateSize
        self.sharedExpertIntermediateSize = decodedSharedIntermediateSize ?? resolvedIntermediateSize
        self.moeIntermediateSize = decodedMoeIntermediateSize ?? resolvedIntermediateSize

        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        // Qwen3.5-family checkpoints omit norm_topk_prob and the architecture
        // default is TRUE (HF transformers and mlx_lm both renormalize the
        // top-k router scores). Defaulting to false scales every MoE block's
        // routed output by the sum of top-k softmax mass (<1), which dampens
        // the residual stream a few percent per layer and compounds into
        // repetition-biased logits by mid-stack.
        self.normTopKProb = try container.decodeIfPresent(Bool.self, forKey: .normTopKProb) ?? true
        self.layerTypes = try container.decode([String].self, forKey: .layerTypes)
        self.mlpOnlyLayers = try container.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        self.linearNumValueHeads = try container.decode(Int.self, forKey: .linearNumValueHeads)
        self.linearNumKeyHeads = try container.decode(Int.self, forKey: .linearNumKeyHeads)
        self.linearKeyHeadDim = try container.decode(Int.self, forKey: .linearKeyHeadDim)
        self.linearValueHeadDim = try container.decode(Int.self, forKey: .linearValueHeadDim)
        self.linearConvKernelDim = try container.decode(Int.self, forKey: .linearConvKernelDim)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.attnOutputGate = try container.decode(Bool.self, forKey: .attnOutputGate)
        self.attentionBias = try container.decode(Bool.self, forKey: .attentionBias)
        self.attentionDropout = try container.decode(Float.self, forKey: .attentionDropout)
        self.fullAttentionInterval = try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval)
        self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        self.eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId)
        self.ropeParameters = try container.decode(Q35RopeParameters.self, forKey: .ropeParameters)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(sharedExpertIntermediateSize, forKey: .sharedExpertIntermediateSize)
        try container.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try container.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try container.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(numExperts, forKey: .numExperts)
        try container.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try container.encode(normTopKProb, forKey: .normTopKProb)
        try container.encode(layerTypes, forKey: .layerTypes)
        try container.encode(mlpOnlyLayers, forKey: .mlpOnlyLayers)
        try container.encode(linearNumValueHeads, forKey: .linearNumValueHeads)
        try container.encode(linearNumKeyHeads, forKey: .linearNumKeyHeads)
        try container.encode(linearKeyHeadDim, forKey: .linearKeyHeadDim)
        try container.encode(linearValueHeadDim, forKey: .linearValueHeadDim)
        try container.encode(linearConvKernelDim, forKey: .linearConvKernelDim)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(attnOutputGate, forKey: .attnOutputGate)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(attentionDropout, forKey: .attentionDropout)
        try container.encodeIfPresent(fullAttentionInterval, forKey: .fullAttentionInterval)
        try container.encode(vocabSize, forKey: .vocabSize)
        try container.encodeIfPresent(eosTokenId, forKey: .eosTokenId)
        try container.encode(ropeParameters, forKey: .ropeParameters)
    }

    public var usesMoE: Bool {
        numExperts > 0 && numExpertsPerTok > 0
    }
}

public struct Q35VisionConfig: Codable, Sendable, Hashable {
    public let modelType: String?
    public let depth: Int
    public let hiddenAct: String?
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHeads: Int
    public let outHiddenSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let spatialMergeSize: Int?
    public let numPositionEmbeddings: Int?
    public let deepstackVisualIndexes: [Int]?
    public let fullAttentionBlockIndexes: [Int]?
    public let windowSize: Int?
    public let inChannels: Int
    public let patchEmbedBias: Bool?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case depth
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHeads = "num_heads"
        case outHiddenSize = "out_hidden_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case numPositionEmbeddings = "num_position_embeddings"
        case deepstackVisualIndexes = "deepstack_visual_indexes"
        case fullAttentionBlockIndexes = "fullatt_block_indexes"
        case windowSize = "window_size"
        case inChannels = "in_channels"
        case patchEmbedBias = "patch_embed_bias"
    }
}

public struct Q35Config: Codable, Sendable, Hashable {
    public let modelType: String
    public let architectures: [String]
    public let tieWordEmbeddings: Bool
    public let eosTokenIds: [Int]
    public let imageTokenId: Int?
    public let videoTokenId: Int?
    public let visionStartTokenId: Int?
    public let visionEndTokenId: Int?
    public let quantization: Q35QuantizationConfig?
    public let textConfig: Q35TextConfig
    public let visionConfig: Q35VisionConfig?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case tieWordEmbeddings = "tie_word_embeddings"
        case eosTokenId = "eos_token_id"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case quantization
        case quantizationConfig = "quantization_config"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId)
        self.videoTokenId = try container.decodeIfPresent(Int.self, forKey: .videoTokenId)
        self.visionStartTokenId = try container.decodeIfPresent(Int.self, forKey: .visionStartTokenId)
        self.visionEndTokenId = try container.decodeIfPresent(Int.self, forKey: .visionEndTokenId)
        self.textConfig = try container.decode(Q35TextConfig.self, forKey: .textConfig)
        self.visionConfig = try container.decodeIfPresent(Q35VisionConfig.self, forKey: .visionConfig)

        if let direct = try Self.decodeTokenIDsIfPresent(in: container, forKey: .eosTokenId) {
            self.eosTokenIds = direct
        } else if let nested = textConfig.eosTokenId {
            self.eosTokenIds = [nested]
        } else {
            self.eosTokenIds = []
        }

        if let q = try container.decodeIfPresent(Q35QuantizationConfig.self, forKey: .quantization) {
            self.quantization = q
        } else {
            self.quantization = try container.decodeIfPresent(Q35QuantizationConfig.self, forKey: .quantizationConfig)
        }
    }

    private static func decodeTokenIDsIfPresent(
        in container: KeyedDecodingContainer<CodingKeys>,
        forKey key: CodingKeys
    ) throws -> [Int]? {
        guard container.contains(key) else { return nil }
        if try container.decodeNil(forKey: key) { return nil }
        if let single = try? container.decode(Int.self, forKey: key) {
            return [single]
        }
        if let ids = try? container.decode([Int].self, forKey: key) {
            return ids
        }
        throw DecodingError.typeMismatch(
            [Int].self,
            DecodingError.Context(
                codingPath: container.codingPath + [key],
                debugDescription: "Expected eos_token_id to be either an integer or an array of integers."
            )
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(architectures, forKey: .architectures)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(eosTokenIds, forKey: .eosTokenId)
        try container.encodeIfPresent(imageTokenId, forKey: .imageTokenId)
        try container.encodeIfPresent(videoTokenId, forKey: .videoTokenId)
        try container.encodeIfPresent(visionStartTokenId, forKey: .visionStartTokenId)
        try container.encodeIfPresent(visionEndTokenId, forKey: .visionEndTokenId)
        try container.encodeIfPresent(quantization, forKey: .quantization)
        try container.encode(textConfig, forKey: .textConfig)
        try container.encodeIfPresent(visionConfig, forKey: .visionConfig)
    }
}
