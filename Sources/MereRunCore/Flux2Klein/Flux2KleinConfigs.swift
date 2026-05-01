import Foundation

// MARK: - Transformer Config

public struct Flux2TransformerConfig: Decodable, Sendable {
    public let attentionHeadDim: Int
    public let axesDimsRope: [Int]
    public let eps: Float
    public let guidanceEmbeds: Bool
    public let inChannels: Int
    public let jointAttentionDim: Int
    public let mlpRatio: Float
    public let numAttentionHeads: Int
    public let numLayers: Int
    public let numSingleLayers: Int
    public let outChannels: Int?
    public let patchSize: Int
    public let ropeTheta: Float
    public let timestepGuidanceChannels: Int

    enum CodingKeys: String, CodingKey {
        case attentionHeadDim = "attention_head_dim"
        case axesDimsRope = "axes_dims_rope"
        case eps
        case guidanceEmbeds = "guidance_embeds"
        case inChannels = "in_channels"
        case jointAttentionDim = "joint_attention_dim"
        case mlpRatio = "mlp_ratio"
        case numAttentionHeads = "num_attention_heads"
        case numLayers = "num_layers"
        case numSingleLayers = "num_single_layers"
        case outChannels = "out_channels"
        case patchSize = "patch_size"
        case ropeTheta = "rope_theta"
        case timestepGuidanceChannels = "timestep_guidance_channels"
    }

    public var hiddenSize: Int { numAttentionHeads * attentionHeadDim }
    public var mlpHiddenSize: Int { Int(Float(hiddenSize) * mlpRatio) }
}

// MARK: - Scheduler Config

public struct Flux2SchedulerConfig: Decodable, Sendable {
    public let baseImageSeqLen: Int
    public let baseShift: Float
    public let maxImageSeqLen: Int
    public let maxShift: Float
    public let numTrainTimesteps: Int
    public let shift: Float
    public let useDynamicShifting: Bool
    public let timeShiftType: String?

    enum CodingKeys: String, CodingKey {
        case baseImageSeqLen = "base_image_seq_len"
        case baseShift = "base_shift"
        case maxImageSeqLen = "max_image_seq_len"
        case maxShift = "max_shift"
        case numTrainTimesteps = "num_train_timesteps"
        case shift
        case useDynamicShifting = "use_dynamic_shifting"
        case timeShiftType = "time_shift_type"
    }
}

// MARK: - VAE Config

public struct Flux2VAEConfig: Decodable, Sendable {
    public let actFn: String?
    public let blockOutChannels: [Int]
    public let forceUpcast: Bool?
    public let inChannels: Int
    public let latentChannels: Int
    public let layersPerBlock: Int
    public let midBlockAddAttention: Bool
    public let normNumGroups: Int
    public let outChannels: Int
    public let patchSize: [Int]?
    public let sampleSize: Int?
    public let scalingFactor: Float?
    public let shiftFactor: Float?
    public let usePostQuantConv: Bool?
    public let useQuantConv: Bool?

    enum CodingKeys: String, CodingKey {
        case actFn = "act_fn"
        case blockOutChannels = "block_out_channels"
        case forceUpcast = "force_upcast"
        case inChannels = "in_channels"
        case latentChannels = "latent_channels"
        case layersPerBlock = "layers_per_block"
        case midBlockAddAttention = "mid_block_add_attention"
        case normNumGroups = "norm_num_groups"
        case outChannels = "out_channels"
        case patchSize = "patch_size"
        case sampleSize = "sample_size"
        case scalingFactor = "scaling_factor"
        case shiftFactor = "shift_factor"
        case usePostQuantConv = "use_post_quant_conv"
        case useQuantConv = "use_quant_conv"
    }

    /// VAE scale factor for image → latent conversion
    public var vaeScaleFactor: Int {
        max(1, 1 << max(0, blockOutChannels.count - 1))
    }
}

// MARK: - Text Encoder Config (Qwen3)

public struct Flux2TextEncoderConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let quantizationConfig: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case quantizationConfig = "quantization_config"
    }

    public struct QuantizationConfig: Decodable, Sendable {
        public let quantType: String?
        public let groupSize: Int?
        public let bits: Int?

        enum CodingKeys: String, CodingKey {
            case quantType = "quant_type"
            case groupSize = "group_size"
            case bits
        }
    }
}

// MARK: - Model Index

public struct Flux2ModelIndex: Decodable, Sendable {
    public let scheduler: [String]
    public let textEncoder: [String]
    public let tokenizer: [String]
    public let transformer: [String]
    public let vae: [String]

    enum CodingKeys: String, CodingKey {
        case scheduler
        case textEncoder = "text_encoder"
        case tokenizer
        case transformer
        case vae
    }
}
