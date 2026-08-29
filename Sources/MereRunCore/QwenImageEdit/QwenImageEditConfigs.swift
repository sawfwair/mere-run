import Foundation

// MARK: - Model Index

public struct QwenImageEditModelIndex: Decodable, Sendable, Hashable {
    public let className: String?
    public let diffusersVersion: String?
    public let scheduler: [String]?
    public let textEncoder: [String]?
    public let tokenizer: [String]?
    public let transformer: [String]?
    public let vae: [String]?
    public let processor: [String]?

    enum CodingKeys: String, CodingKey {
        case className = "_class_name"
        case diffusersVersion = "_diffusers_version"
        case scheduler
        case textEncoder = "text_encoder"
        case tokenizer
        case transformer
        case vae
        case processor
    }
}

// MARK: - Scheduler Config

public struct QwenImageEditSchedulerConfig: Decodable, Sendable, Hashable {
    public let numTrainTimesteps: Int
    public let shift: Float
    public let useDynamicShifting: Bool
    public let baseShift: Float?
    public let maxShift: Float?
    public let baseImageSeqLen: Int?
    public let maxImageSeqLen: Int?
    public let invertSigmas: Bool?
    public let shiftTerminal: Float?
    public let timeShiftType: String?

    enum CodingKeys: String, CodingKey {
        case numTrainTimesteps = "num_train_timesteps"
        case shift
        case useDynamicShifting = "use_dynamic_shifting"
        case baseShift = "base_shift"
        case maxShift = "max_shift"
        case baseImageSeqLen = "base_image_seq_len"
        case maxImageSeqLen = "max_image_seq_len"
        case invertSigmas = "invert_sigmas"
        case shiftTerminal = "shift_terminal"
        case timeShiftType = "time_shift_type"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        numTrainTimesteps = try container.decodeIfPresent(Int.self, forKey: .numTrainTimesteps) ?? 1000
        shift = try container.decodeIfPresent(Float.self, forKey: .shift) ?? 1.0
        useDynamicShifting = try container.decodeIfPresent(Bool.self, forKey: .useDynamicShifting) ?? false
        baseShift = try container.decodeIfPresent(Float.self, forKey: .baseShift)
        maxShift = try container.decodeIfPresent(Float.self, forKey: .maxShift)
        baseImageSeqLen = try container.decodeIfPresent(Int.self, forKey: .baseImageSeqLen)
        maxImageSeqLen = try container.decodeIfPresent(Int.self, forKey: .maxImageSeqLen)
        invertSigmas = try container.decodeIfPresent(Bool.self, forKey: .invertSigmas)
        shiftTerminal = try container.decodeIfPresent(Float.self, forKey: .shiftTerminal)
        timeShiftType = try container.decodeIfPresent(String.self, forKey: .timeShiftType)
    }
}

// MARK: - Transformer Config (QwenImageTransformer2DModel)

public struct QwenImageEditTransformerConfig: Decodable, Sendable, Hashable {
    // Core dimensions (computed from attention_head_dim * num_attention_heads)
    public let numAttentionHeads: Int
    public let attentionHeadDim: Int
    public let numLayers: Int

    // Cross-attention dimension (text encoder hidden size)
    public let jointAttentionDim: Int

    // Patch embedding
    public let inChannels: Int
    public let outChannels: Int
    public let patchSize: Int

    // 3D RoPE dimensions [temporal, height, width]
    public let axesDimsRope: [Int]

    // Guidance
    public let guidanceEmbeds: Bool
    public let zeroCondT: Bool

    // Legacy/fallback keys for compatibility
    public let hiddenSize: Int?
    public let numKeyValueHeads: Int?
    public let normEps: Float?
    public let qkNorm: Bool?
    public let ropeTheta: Float?
    public let mlpRatio: Float?
    public let captionProjectionDim: Int?

    enum CodingKeys: String, CodingKey {
        case numAttentionHeads = "num_attention_heads"
        case attentionHeadDim = "attention_head_dim"
        case numLayers = "num_layers"
        case jointAttentionDim = "joint_attention_dim"
        case inChannels = "in_channels"
        case outChannels = "out_channels"
        case patchSize = "patch_size"
        case axesDimsRope = "axes_dims_rope"
        case guidanceEmbeds = "guidance_embeds"
        case zeroCondT = "zero_cond_t"
        // Legacy keys
        case hiddenSize = "hidden_size"
        case numKeyValueHeads = "num_key_value_heads"
        case normEps = "norm_eps"
        case qkNorm = "qk_norm"
        case ropeTheta = "rope_theta"
        case mlpRatio = "mlp_ratio"
        case captionProjectionDim = "caption_projection_dim"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        attentionHeadDim = try container.decodeIfPresent(Int.self, forKey: .attentionHeadDim) ?? 128
        numLayers = try container.decodeIfPresent(Int.self, forKey: .numLayers) ?? 60
        jointAttentionDim = try container.decodeIfPresent(Int.self, forKey: .jointAttentionDim) ?? 3584
        inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 64
        outChannels = try container.decodeIfPresent(Int.self, forKey: .outChannels) ?? 16
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 2
        axesDimsRope = try container.decodeIfPresent([Int].self, forKey: .axesDimsRope) ?? [16, 56, 56]
        guidanceEmbeds = try container.decodeIfPresent(Bool.self, forKey: .guidanceEmbeds) ?? false
        zeroCondT = try container.decodeIfPresent(Bool.self, forKey: .zeroCondT) ?? false

        // Legacy/fallback
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
        normEps = try container.decodeIfPresent(Float.self, forKey: .normEps)
        qkNorm = try container.decodeIfPresent(Bool.self, forKey: .qkNorm)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
        mlpRatio = try container.decodeIfPresent(Float.self, forKey: .mlpRatio)
        captionProjectionDim = try container.decodeIfPresent(Int.self, forKey: .captionProjectionDim)
    }

    /// Computed hidden size = num_attention_heads * attention_head_dim
    public var effectiveHiddenSize: Int {
        hiddenSize ?? (numAttentionHeads * attentionHeadDim)
    }

    public var effectiveHeadDim: Int {
        attentionHeadDim
    }

    public var effectiveNumKVHeads: Int {
        numKeyValueHeads ?? numAttentionHeads
    }

    /// Total RoPE dimension (sum of axes)
    public var ropeDim: Int {
        axesDimsRope.reduce(0, +)
    }
}

// MARK: - VAE Config

public struct QwenImageEditVAEConfig: Decodable, Sendable, Hashable {
    // Core architecture (computed from base_dim + dim_mult OR block_out_channels)
    public let blockOutChannels: [Int]
    public let inChannels: Int
    public let outChannels: Int
    public let latentChannels: Int
    public let layersPerBlock: Int
    public let normNumGroups: Int
    public let scalingFactor: Float
    public let shiftFactor: Float?
    public let temporalCompressionRatio: Int?
    public let midBlockAddAttention: Bool?
    public let latentsMean: [Float]?
    public let latentsStd: [Float]?

    enum CodingKeys: String, CodingKey {
        // Standard diffusers keys
        case blockOutChannels = "block_out_channels"
        case inChannels = "in_channels"
        case outChannels = "out_channels"
        case latentChannels = "latent_channels"
        case layersPerBlock = "layers_per_block"
        case normNumGroups = "norm_num_groups"
        case scalingFactor = "scaling_factor"
        case shiftFactor = "shift_factor"
        case temporalCompressionRatio = "temporal_compression_ratio"
        case midBlockAddAttention = "mid_block_add_attention"
        case latentsMean = "latents_mean"
        case latentsStd = "latents_std"
        // Qwen-specific keys (alternative format)
        case baseDim = "base_dim"
        case dimMult = "dim_mult"
        case zDim = "z_dim"
        case numResBlocks = "num_res_blocks"
        case temperalDownsample = "temperal_downsample"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Handle two possible config formats:
        // 1. Standard diffusers: block_out_channels, latent_channels, etc.
        // 2. Qwen format: base_dim + dim_mult, z_dim, etc.

        if let channels = try container.decodeIfPresent([Int].self, forKey: .blockOutChannels) {
            // Standard format
            self.blockOutChannels = channels
            self.latentChannels = try container.decode(Int.self, forKey: .latentChannels)
            self.layersPerBlock = try container.decodeIfPresent(Int.self, forKey: .layersPerBlock) ?? 2
            self.scalingFactor = try container.decode(Float.self, forKey: .scalingFactor)
            self.temporalCompressionRatio = try container.decodeIfPresent(Int.self, forKey: .temporalCompressionRatio)
        } else if let baseDim = try container.decodeIfPresent(Int.self, forKey: .baseDim),
                  let dimMult = try container.decodeIfPresent([Int].self, forKey: .dimMult) {
            // Qwen format: compute block_out_channels from base_dim * dim_mult
            self.blockOutChannels = dimMult.map { baseDim * $0 }
            self.latentChannels = try container.decodeIfPresent(Int.self, forKey: .zDim) ?? 16
            self.layersPerBlock = try container.decodeIfPresent(Int.self, forKey: .numResBlocks) ?? 2

            // Qwen VAE uses different scaling - compute from z_dim
            // Default scaling factor for Qwen VAE
            self.scalingFactor = try container.decodeIfPresent(Float.self, forKey: .scalingFactor) ?? 0.476986

            // Compute temporal compression from temperal_downsample array
            if let temporalDowns = try container.decodeIfPresent([Bool].self, forKey: .temperalDownsample) {
                let numTemporalDowns = temporalDowns.filter { $0 }.count
                self.temporalCompressionRatio = 1 << numTemporalDowns  // 2^count
            } else {
                self.temporalCompressionRatio = 4
            }
        } else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: container.codingPath,
                    debugDescription: "VAE config must have either block_out_channels or base_dim + dim_mult"
                )
            )
        }

        self.inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
        self.outChannels = try container.decodeIfPresent(Int.self, forKey: .outChannels) ?? 3
        self.normNumGroups = try container.decodeIfPresent(Int.self, forKey: .normNumGroups) ?? 32
        self.shiftFactor = try container.decodeIfPresent(Float.self, forKey: .shiftFactor)
        self.midBlockAddAttention = try container.decodeIfPresent(Bool.self, forKey: .midBlockAddAttention)
        self.latentsMean = try container.decodeIfPresent([Float].self, forKey: .latentsMean)
        self.latentsStd = try container.decodeIfPresent([Float].self, forKey: .latentsStd)
    }

    public var vaeScaleFactor: Int {
        max(1, 1 << max(0, blockOutChannels.count - 1))
    }

    public var latentDivisor: Int {
        vaeScaleFactor
    }
}

// MARK: - Text Encoder Config (Qwen2.5-VL)

public struct QwenImageEditTextEncoderConfig: Decodable, Sendable, Hashable {
    // Core architecture
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int?
    public let headDim: Int?
    public let intermediateSize: Int
    public let vocabSize: Int

    // Vision tower (for VL models)
    public let visionConfig: VisionConfig?

    // Normalization
    public let rmsNormEps: Float?

    // Position embedding
    public let ropeTheta: Float?
    public let maxPositionEmbeddings: Int?
    public let ropeScaling: RopeScaling?

    public struct RopeScaling: Decodable, Sendable, Hashable {
        public let mropeSection: [Int]?

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
        }
    }

    // Token IDs
    public let bosTokenId: Int?
    public let eosTokenId: Int?
    public let padTokenId: Int?

    // Attention
    public let attentionBias: Bool?
    public let attentionDropout: Float?

    // Other
    public let tieWordEmbeddings: Bool?
    public let useCache: Bool?
    public let modelType: String?

    public struct VisionConfig: Decodable, Sendable, Hashable {
        // Qwen2.5-VL vision config uses different key names
        public let depth: Int?  // num_hidden_layers equivalent
        public let hiddenSize: Int?
        public let intermediateSize: Int?
        public let hiddenAct: String?
        public let numHeads: Int?  // num_attention_heads equivalent
        public let patchSize: Int?
        public let spatialPatchSize: Int?
        public let temporalPatchSize: Int?
        public let spatialMergeSize: Int?
        public let outHiddenSize: Int?  // projection target
        public let windowSize: Int?
        public let fullattBlockIndexes: [Int]?

        enum CodingKeys: String, CodingKey {
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenAct = "hidden_act"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case spatialPatchSize = "spatial_patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case outHiddenSize = "out_hidden_size"
            case windowSize = "window_size"
            case fullattBlockIndexes = "fullatt_block_indexes"
        }

        // Convenience accessors with compatibility fallbacks
        public var numHiddenLayers: Int? { depth }
        public var numAttentionHeads: Int? { numHeads }
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case visionConfig = "vision_config"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeScaling = "rope_scaling"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useCache = "use_cache"
        case modelType = "model_type"
    }
}

// MARK: - Combined Config

public struct QwenImageEditModelConfigs: Sendable, Hashable {
    public let modelIndex: QwenImageEditModelIndex
    public let transformer: QwenImageEditTransformerConfig
    public let vae: QwenImageEditVAEConfig
    public let scheduler: QwenImageEditSchedulerConfig
    public let textEncoder: QwenImageEditTextEncoderConfig

    public static func load(
        from resources: QwenImageEditResources,
        decoder: JSONDecoder = JSONDecoder()
    ) throws -> QwenImageEditModelConfigs {
        func decode<T: Decodable>(_ type: T.Type, url: URL) throws -> T {
            try decoder.decode(T.self, from: Data(contentsOf: url))
        }

        return QwenImageEditModelConfigs(
            modelIndex: try decode(QwenImageEditModelIndex.self, url: resources.modelIndexURL),
            transformer: try decode(QwenImageEditTransformerConfig.self, url: resources.transformerConfigURL),
            vae: try decode(QwenImageEditVAEConfig.self, url: resources.vaeConfigURL),
            scheduler: try decode(QwenImageEditSchedulerConfig.self, url: resources.schedulerConfigURL),
            textEncoder: try decode(QwenImageEditTextEncoderConfig.self, url: resources.textEncoderConfigURL)
        )
    }
}
