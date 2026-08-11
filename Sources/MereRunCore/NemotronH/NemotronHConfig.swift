import Foundation

public struct NemotronHQuantizationConfig: Decodable, Sendable, Hashable {
    public let bits: Int
    public let groupSize: Int
    public let mode: String
    public let globalScale: Bool

    private enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case mode
        case globalScale = "global_scale"
    }
}

public struct NemotronHConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let layersBlockType: [String]
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let maxPositionEmbeddings: Int
    public let normEps: Float
    public let mambaHeadDim: Int
    public let mambaNumHeads: Int
    public let ssmStateSize: Int
    public let nGroups: Int
    public let convKernel: Int
    public let timeStepMin: Float
    public let timeStepMax: Float
    public let nRoutedExperts: Int
    public let nSharedExperts: Int
    public let numExpertsPerToken: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let routedScalingFactor: Float
    public let normTopKProbability: Bool
    public let nGroup: Int
    public let topKGroup: Int
    public let eosTokenIDs: [Int]
    public let quantization: NemotronHQuantizationConfig

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case layersBlockType = "layers_block_type"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case mambaHeadDim = "mamba_head_dim"
        case mambaNumHeads = "mamba_num_heads"
        case ssmStateSize = "ssm_state_size"
        case nGroups = "n_groups"
        case convKernel = "conv_kernel"
        case timeStepMin = "time_step_min"
        case timeStepMax = "time_step_max"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopKProbability = "norm_topk_prob"
        case nGroup = "n_group"
        case topKGroup = "topk_group"
        case eosTokenID = "eos_token_id"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        layersBlockType = try container.decode([String].self, forKey: .layersBlockType)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try container.decode(Int.self, forKey: .headDim)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        normEps = try container.decode(Float.self, forKey: .normEps)
        mambaHeadDim = try container.decode(Int.self, forKey: .mambaHeadDim)
        mambaNumHeads = try container.decode(Int.self, forKey: .mambaNumHeads)
        ssmStateSize = try container.decode(Int.self, forKey: .ssmStateSize)
        nGroups = try container.decode(Int.self, forKey: .nGroups)
        convKernel = try container.decode(Int.self, forKey: .convKernel)
        timeStepMin = try container.decode(Float.self, forKey: .timeStepMin)
        timeStepMax = try container.decode(Float.self, forKey: .timeStepMax)
        nRoutedExperts = try container.decode(Int.self, forKey: .nRoutedExperts)
        nSharedExperts = try container.decode(Int.self, forKey: .nSharedExperts)
        numExpertsPerToken = try container.decode(Int.self, forKey: .numExpertsPerToken)
        moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try container.decode(
            Int.self,
            forKey: .sharedExpertIntermediateSize
        )
        routedScalingFactor = try container.decode(Float.self, forKey: .routedScalingFactor)
        normTopKProbability = try container.decode(Bool.self, forKey: .normTopKProbability)
        nGroup = try container.decode(Int.self, forKey: .nGroup)
        topKGroup = try container.decode(Int.self, forKey: .topKGroup)
        quantization = try container.decode(NemotronHQuantizationConfig.self, forKey: .quantization)
        if let ids = try? container.decode([Int].self, forKey: .eosTokenID) {
            eosTokenIDs = ids
        } else {
            eosTokenIDs = [try container.decode(Int.self, forKey: .eosTokenID)]
        }

        guard modelType == "nemotron_h",
              layersBlockType.count == numHiddenLayers,
              Set(layersBlockType).isSubset(of: ["attention", "mamba", "moe"]),
              mambaHeadDim * mambaNumHeads == 4_096,
              nGroups > 0,
              mambaNumHeads.isMultiple(of: nGroups),
              nRoutedExperts == 128,
              numExpertsPerToken == 6,
              nGroup == 1,
              topKGroup == 1,
              quantization.bits == 4,
              quantization.groupSize == 16,
              quantization.mode == "nvfp4",
              quantization.globalScale else {
            throw DecodingError.dataCorruptedError(
                forKey: .modelType,
                in: container,
                debugDescription: "Unsupported Nemotron-H checkpoint contract."
            )
        }
    }
}

public struct NemotronHDSparkSpeculationConfig: Decodable, Sendable, Hashable {
    public let attentionSinkBias: Bool
    public let causal: Bool
    public let maskTokenID: Int
    public let sampleFromAnchor: Bool
    public let slidingWindow: Int
    public let targetLayerIDs: [Int]
    public let useSlidingWindow: Bool

    private enum CodingKeys: String, CodingKey {
        case attentionSinkBias = "attention_sink_bias"
        case causal
        case maskTokenID = "mask_token_id"
        case sampleFromAnchor = "sample_from_anchor"
        case slidingWindow = "swa_window_size"
        case targetLayerIDs = "target_layer_ids"
        case useSlidingWindow = "use_swa"
    }
}

public struct NemotronHDSparkConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let architectures: [String]
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let eagleAuxHiddenStateLayerIDs: [Int]
    public let blockSize: Int
    public let markovHeadDim: Int
    public let bonusAnchor: Bool
    public let speculation: NemotronHDSparkSpeculationConfig

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case eagleAuxHiddenStateLayerIDs = "eagle_aux_hidden_state_layer_ids"
        case blockSize = "block_size"
        case markovHeadDim = "markov_rank"
        case bonusAnchor = "dspark_bonus_anchor"
        case speculation = "dflash_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures = try container.decode([String].self, forKey: .architectures)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try container.decode(Int.self, forKey: .headDim)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        eagleAuxHiddenStateLayerIDs = try container.decode(
            [Int].self,
            forKey: .eagleAuxHiddenStateLayerIDs
        )
        blockSize = try container.decode(Int.self, forKey: .blockSize)
        markovHeadDim = try container.decode(Int.self, forKey: .markovHeadDim)
        bonusAnchor = try container.decode(Bool.self, forKey: .bonusAnchor)
        speculation = try container.decode(NemotronHDSparkSpeculationConfig.self, forKey: .speculation)

        guard modelType == "qwen3",
              architectures == ["Qwen3DSparkModel"],
              blockSize == 8,
              markovHeadDim == 512,
              speculation.causal,
              speculation.attentionSinkBias,
              speculation.useSlidingWindow,
              !speculation.sampleFromAnchor,
              bonusAnchor,
              speculation.targetLayerIDs.count == numHiddenLayers,
              eagleAuxHiddenStateLayerIDs.count == numHiddenLayers,
              zip(speculation.targetLayerIDs, eagleAuxHiddenStateLayerIDs)
                .allSatisfy({ $1 == $0 + 1 }) else {
            throw DecodingError.dataCorruptedError(
                forKey: .modelType,
                in: container,
                debugDescription: "Unsupported Nemotron DSpark checkpoint contract."
            )
        }
    }
}
