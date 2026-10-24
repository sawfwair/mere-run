import Foundation

public struct NemotronBoltBlockConfig: Decodable, Sendable, Hashable {
    public let blockType: String
    public let moeIntermediateSize: Int?
    public let expertCount: Int?
    public let topK: Int?

    private enum CodingKeys: String, CodingKey {
        case blockType = "block_type"
        case moeIntermediateSize = "moe_intermediate_size"
        case expertCount = "n_routed_experts"
        case topK = "num_experts_per_tok"
    }
}

public struct NemotronBoltLanguageConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let blocks: [NemotronBoltBlockConfig]
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
    public let moeLatentSize: Int
    public let sharedExpertIntermediateSize: Int
    public let routedScalingFactor: Float
    public let normTopKProbability: Bool
    public let scoringFunction: String
    public let nGroup: Int
    public let topKGroup: Int
    public let eosTokenID: Int
    public let hiddenActivation: String
    public let attentionBias: Bool
    public let mlpBias: Bool
    public let useQKNorm: Bool

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case blocks = "block_configs"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "layer_norm_epsilon"
        case mambaHeadDim = "mamba_head_dim"
        case mambaNumHeads = "mamba_num_heads"
        case ssmStateSize = "ssm_state_size"
        case nGroups = "n_groups"
        case convKernel = "conv_kernel"
        case timeStepMin = "time_step_min"
        case moeLatentSize = "moe_latent_size"
        case sharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopKProbability = "norm_topk_prob"
        case scoringFunction = "scoring_func"
        case nGroup = "n_group"
        case topKGroup = "topk_group"
        case eosTokenID = "eos_token_id"
        case hiddenActivation = "mlp_hidden_act"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case useQKNorm = "use_qk_norm"
    }
}

public struct NemotronBoltVisionConfig: Decodable, Sendable, Hashable {
    public let patchSize: Int
    public let videoTemporalPatchSize: Int
    public let hiddenSize: Int
    public let layerCount: Int
    public let headCount: Int
    public let minNumPatches: Int
    public let maxNumPatches: Int

    private enum CodingKeys: String, CodingKey {
        case patchSize = "patch_size"
        case videoTemporalPatchSize = "video_temporal_patch_size"
        case hiddenSize = "hidden_size"
        case layerCount = "num_hidden_layers"
        case headCount = "num_attention_heads"
        case minNumPatches = "min_num_patches"
        case maxNumPatches = "max_num_patches"
    }
}

public struct NemotronBoltConfig: Decodable, Sendable, Hashable {
    public let architectures: [String]
    public let modelType: String
    public let language: NemotronBoltLanguageConfig
    public let vision: NemotronBoltVisionConfig
    public let imageContextTokenID: Int
    public let eosTokenID: Int
    public let visionHiddenSize: Int
    public let projectorHiddenSize: Int
    public let downsampleRatio: Double

    private enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case language = "llm_config"
        case vision = "vision_config"
        case imageContextTokenID = "img_context_token_id"
        case eosTokenID = "eos_token_id"
        case visionHiddenSize = "vit_hidden_size"
        case projectorHiddenSize = "projector_hidden_size"
        case downsampleRatio = "downsample_ratio"
    }
}

extension NemotronBoltConfig {
    public static let upstreamRepoID = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
    public static let upstreamRevision = "96a5d63027ac013c8ef37fb438f3b4ec5fbd3ba6"
    public static let checkpointWeightBytes: Int64 = 24_079_678_824

    public static func load(at root: URL) throws -> Self {
        let config = try JSONDecoder().decode(Self.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        try config.validate()
        return config
    }

    public func validate() throws {
        let c = language
        guard architectures == ["NemotronH_Omni_Reasoning_V3"], modelType == "nemotron_h_omni",
              c.modelType == "nemotron_h_puzzle", c.hiddenSize == 4_096, c.vocabSize == 131_072,
              c.blocks.count == 88, c.blocks.filter({ $0.blockType == "mamba" }).count == 40,
              c.blocks.filter({ $0.blockType == "attention" }).count == 8,
              c.blocks.filter({ $0.blockType == "moe" }).count == 40,
              c.numAttentionHeads == 32, c.numKeyValueHeads == 2, c.headDim == 128,
              c.mambaHeadDim == 64, c.mambaNumHeads == 128, c.ssmStateSize == 128,
              c.nGroups == 8, c.convKernel == 4, c.timeStepMin == 0.001,
              c.normEps == 1e-5, c.moeLatentSize == 1_024, c.sharedExpertIntermediateSize == 5_376,
              c.routedScalingFactor == 5, c.normTopKProbability, c.scoringFunction == "sigmoid",
              c.nGroup == 1, c.topKGroup == 1, c.hiddenActivation == "relu2",
              !c.attentionBias, !c.mlpBias, !c.useQKNorm,
              c.maxPositionEmbeddings == 262_144, imageContextTokenID == 18, eosTokenID == 11,
              visionHiddenSize == 1_280, projectorHiddenSize == 20_480, downsampleRatio == 0.5,
              vision.hiddenSize == 1_280, vision.layerCount == 32, vision.headCount == 16,
              vision.patchSize == 16, vision.videoTemporalPatchSize == 2 else {
            throw NemotronBoltError.invalidCheckpoint("Unsupported Bolt architecture or arithmetic contract")
        }
        for block in c.blocks {
            switch block.blockType {
            case "attention", "mamba": break
            case "moe":
                guard let count = block.expertCount, [240, 304, 368, 432, 496].contains(count),
                      block.topK == 9, let size = block.moeIntermediateSize,
                      [1_536, 2_048, 2_304, 2_560].contains(size) else {
                    throw NemotronBoltError.invalidCheckpoint("Unsupported heterogeneous expert configuration")
                }
            default: throw NemotronBoltError.invalidCheckpoint("Unsupported block type")
            }
        }
    }
}

public enum NemotronBoltError: LocalizedError {
    case invalidCheckpoint(String)
    public var errorDescription: String? {
        switch self {
        case .invalidCheckpoint(let message): "Nemotron 3.5 Bolt: \(message)."
        }
    }
}
