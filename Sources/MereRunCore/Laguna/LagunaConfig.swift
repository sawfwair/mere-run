import Foundation

public struct LagunaQuantizationConfig: Decodable, Sendable, Hashable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String

    private enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }
}

public struct LagunaRopeParameters: Decodable, Sendable, Hashable {
    public let ropeType: String
    public let ropeTheta: Float
    public let factor: Float?
    public let originalMaxPositionEmbeddings: Int?
    public let betaSlow: Float?
    public let betaFast: Float?
    public let attentionFactor: Float?
    public let partialRotaryFactor: Float

    private enum CodingKeys: String, CodingKey {
        case ropeType = "rope_type"
        case ropeTheta = "rope_theta"
        case factor
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case betaSlow = "beta_slow"
        case betaFast = "beta_fast"
        case attentionFactor = "attention_factor"
        case partialRotaryFactor = "partial_rotary_factor"
    }

    public init(
        ropeType: String = "default",
        ropeTheta: Float = 10_000,
        factor: Float? = nil,
        originalMaxPositionEmbeddings: Int? = nil,
        betaSlow: Float? = nil,
        betaFast: Float? = nil,
        attentionFactor: Float? = nil,
        partialRotaryFactor: Float = 1
    ) {
        self.ropeType = ropeType
        self.ropeTheta = ropeTheta
        self.factor = factor
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.betaSlow = betaSlow
        self.betaFast = betaFast
        self.attentionFactor = attentionFactor
        self.partialRotaryFactor = partialRotaryFactor
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.ropeType = try container.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        self.factor = try container.decodeIfPresent(Float.self, forKey: .factor)
        self.originalMaxPositionEmbeddings = try container.decodeIfPresent(
            Int.self,
            forKey: .originalMaxPositionEmbeddings
        )
        self.betaSlow = try container.decodeIfPresent(Float.self, forKey: .betaSlow)
        self.betaFast = try container.decodeIfPresent(Float.self, forKey: .betaFast)
        self.attentionFactor = try container.decodeIfPresent(Float.self, forKey: .attentionFactor)
        self.partialRotaryFactor = try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1
    }
}

public struct LagunaConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numAttentionHeadsPerLayer: [Int]?
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let attentionBias: Bool
    public let gating: String
    public let layerTypes: [String]
    public let slidingWindow: Int
    public let mlpLayerTypes: [String]?
    public let mlpOnlyLayers: [Int]
    public let numExperts: Int
    public let numExpertsPerToken: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let moeRoutedScalingFactor: Float
    public let moeRouterLogitSoftcapping: Float
    public let normTopKProbability: Bool
    public let decoderSparseStep: Int
    public let applyRouterWeightOnInput: Bool
    public let tieWordEmbeddings: Bool
    public let eosTokenIDs: [Int]
    public let ropeParameters: [String: LagunaRopeParameters]
    public let quantization: LagunaQuantizationConfig?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numAttentionHeadsPerLayer = "num_attention_heads_per_layer"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case gating
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case mlpLayerTypes = "mlp_layer_types"
        case mlpOnlyLayers = "mlp_only_layers"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeRoutedScalingFactor = "moe_routed_scaling_factor"
        case moeRouterLogitSoftcapping = "moe_router_logit_softcapping"
        case normTopKProbability = "norm_topk_prob"
        case decoderSparseStep = "decoder_sparse_step"
        case applyRouterWeightOnInput = "moe_apply_router_weight_on_input"
        case tieWordEmbeddings = "tie_word_embeddings"
        case eosTokenID = "eos_token_id"
        case ropeParameters = "rope_parameters"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numAttentionHeadsPerLayer = try container.decodeIfPresent(
            [Int].self,
            forKey: .numAttentionHeadsPerLayer
        )
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.gating = try container.decodeIfPresent(String.self, forKey: .gating) ?? "per-head"
        self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: self.numHiddenLayers)
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        self.mlpLayerTypes = try container.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
        self.mlpOnlyLayers = try container.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? [0]
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerToken = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 1
        self.moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.sharedExpertIntermediateSize = try container.decodeIfPresent(
            Int.self,
            forKey: .sharedExpertIntermediateSize
        ) ?? 0
        self.moeRoutedScalingFactor = try container.decodeIfPresent(
            Float.self,
            forKey: .moeRoutedScalingFactor
        ) ?? 1
        self.moeRouterLogitSoftcapping = try container.decodeIfPresent(
            Float.self,
            forKey: .moeRouterLogitSoftcapping
        ) ?? 0
        self.normTopKProbability = try container.decodeIfPresent(
            Bool.self,
            forKey: .normTopKProbability
        ) ?? true
        self.decoderSparseStep = try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.applyRouterWeightOnInput = try container.decodeIfPresent(
            Bool.self,
            forKey: .applyRouterWeightOnInput
        ) ?? false
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        if let ids = try container.decodeIfPresent([Int].self, forKey: .eosTokenID) {
            self.eosTokenIDs = ids
        } else if let id = try container.decodeIfPresent(Int.self, forKey: .eosTokenID) {
            self.eosTokenIDs = [id]
        } else {
            self.eosTokenIDs = []
        }
        self.ropeParameters = try container.decode([String: LagunaRopeParameters].self, forKey: .ropeParameters)
        self.quantization = try container.decodeIfPresent(LagunaQuantizationConfig.self, forKey: .quantization)

        guard self.modelType == "laguna" else {
            throw DecodingError.dataCorruptedError(
                forKey: .modelType,
                in: container,
                debugDescription: "Expected model_type 'laguna'."
            )
        }
        guard self.layerTypes.count == self.numHiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes,
                in: container,
                debugDescription: "layer_types must contain one entry per hidden layer."
            )
        }
        if let heads = self.numAttentionHeadsPerLayer, heads.count != self.numHiddenLayers {
            throw DecodingError.dataCorruptedError(
                forKey: .numAttentionHeadsPerLayer,
                in: container,
                debugDescription: "num_attention_heads_per_layer must contain one entry per hidden layer."
            )
        }
        if let mlpLayerTypes = self.mlpLayerTypes, mlpLayerTypes.count != self.numHiddenLayers {
            throw DecodingError.dataCorruptedError(
                forKey: .mlpLayerTypes,
                in: container,
                debugDescription: "mlp_layer_types must contain one entry per hidden layer."
            )
        }
        guard self.layerTypes.allSatisfy({ self.ropeParameters[$0] != nil }) else {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeParameters,
                in: container,
                debugDescription: "rope_parameters must define every attention type used by layer_types."
            )
        }
        guard !self.applyRouterWeightOnInput else {
            throw DecodingError.dataCorruptedError(
                forKey: .applyRouterWeightOnInput,
                in: container,
                debugDescription: "moe_apply_router_weight_on_input is not supported by the Laguna runtime."
            )
        }
    }

    func attentionHeads(layerIndex: Int) -> Int {
        numAttentionHeadsPerLayer?[layerIndex] ?? numAttentionHeads
    }

    func ropeParameters(layerIndex: Int) -> LagunaRopeParameters {
        ropeParameters[layerTypes[layerIndex]]!
    }

    func isSparse(layerIndex: Int) -> Bool {
        if let mlpLayerTypes {
            return mlpLayerTypes[layerIndex] == "sparse"
        }
        return !mlpOnlyLayers.contains(layerIndex)
            && numExperts > 0
            && (layerIndex + 1).isMultiple(of: max(1, decoderSparseStep))
    }
}
