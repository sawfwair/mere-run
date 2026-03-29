import Foundation

public struct ZImageTurboModelIndex: Decodable, Sendable, Hashable {
    public let className: String?
    public let diffusersVersion: String?
    public let scheduler: [String]?
    public let textEncoder: [String]?
    public let tokenizer: [String]?
    public let transformer: [String]?
    public let vae: [String]?

    enum CodingKeys: String, CodingKey {
        case className = "_class_name"
        case diffusersVersion = "_diffusers_version"
        case scheduler
        case textEncoder = "text_encoder"
        case tokenizer
        case transformer
        case vae
    }
}

public struct ZImageTurboTransformerConfig: Decodable, Sendable, Hashable {
    public let inChannels: Int
    public let dim: Int
    public let nLayers: Int
    public let nRefinerLayers: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let normEps: Float
    public let qkNorm: Bool
    public let capFeatDim: Int
    public let ropeTheta: Float
    public let tScale: Float
    public let axesDims: [Int]
    public let axesLens: [Int]
    public let allPatchSize: [Int]?
    public let allFPatchSize: [Int]?

    enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels"
        case dim
        case nLayers = "n_layers"
        case nRefinerLayers = "n_refiner_layers"
        case nHeads = "n_heads"
        case nKVHeads = "n_kv_heads"
        case normEps = "norm_eps"
        case qkNorm = "qk_norm"
        case capFeatDim = "cap_feat_dim"
        case ropeTheta = "rope_theta"
        case tScale = "t_scale"
        case axesDims = "axes_dims"
        case axesLens = "axes_lens"
        case allPatchSize = "all_patch_size"
        case allFPatchSize = "all_f_patch_size"
    }
}

public struct ZImageTurboVAEConfig: Decodable, Sendable, Hashable {
    public let actFn: String?
    public let blockOutChannels: [Int]
    public let downBlockTypes: [String]?
    public let upBlockTypes: [String]?
    public let forceUpcast: Bool?
    public let inChannels: Int
    public let latentChannels: Int
    public let layersPerBlock: Int
    public let midBlockAddAttention: Bool
    public let normNumGroups: Int
    public let outChannels: Int
    public let sampleSize: Int?
    public let scalingFactor: Float
    public let shiftFactor: Float
    public let usePostQuantConv: Bool?
    public let useQuantConv: Bool?
    public let latentsMean: [Float]?
    public let latentsStd: [Float]?

    enum CodingKeys: String, CodingKey {
        case actFn = "act_fn"
        case blockOutChannels = "block_out_channels"
        case downBlockTypes = "down_block_types"
        case upBlockTypes = "up_block_types"
        case forceUpcast = "force_upcast"
        case inChannels = "in_channels"
        case latentChannels = "latent_channels"
        case layersPerBlock = "layers_per_block"
        case midBlockAddAttention = "mid_block_add_attention"
        case normNumGroups = "norm_num_groups"
        case outChannels = "out_channels"
        case sampleSize = "sample_size"
        case scalingFactor = "scaling_factor"
        case shiftFactor = "shift_factor"
        case usePostQuantConv = "use_post_quant_conv"
        case useQuantConv = "use_quant_conv"
        case latentsMean = "latents_mean"
        case latentsStd = "latents_std"
    }

    public var vaeScaleFactor: Int {
        max(1, 1 << max(0, blockOutChannels.count - 1))
    }

    public var latentDivisor: Int {
        vaeScaleFactor
    }
}

public struct ZImageTurboSchedulerConfig: Decodable, Sendable, Hashable {
    public let numTrainTimesteps: Int
    public let shift: Float
    public let useDynamicShifting: Bool
    public let baseShift: Float?
    public let maxShift: Float?
    public let baseImageSeqLen: Int?
    public let maxImageSeqLen: Int?

    enum CodingKeys: String, CodingKey {
        case numTrainTimesteps = "num_train_timesteps"
        case shift
        case useDynamicShifting = "use_dynamic_shifting"
        case baseShift = "base_shift"
        case maxShift = "max_shift"
        case baseImageSeqLen = "base_image_seq_len"
        case maxImageSeqLen = "max_image_seq_len"
    }
}

public struct ZImageTurboTextEncoderConfig: Decodable, Sendable, Hashable {
    public let attentionBias: Bool?
    public let attentionDropout: Float?
    public let bosTokenId: Int?
    public let eosTokenId: Int?
    public let headDim: Int
    public let hiddenAct: String?
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let maxPositionEmbeddings: Int
    public let modelType: String?
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let slidingWindow: Int?
    public let tieWordEmbeddings: Bool?
    public let torchDtype: String?
    public let useCache: Bool?
    public let useSlidingWindow: Bool?
    public let vocabSize: Int

    enum CodingKeys: String, CodingKey {
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case modelType = "model_type"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case tieWordEmbeddings = "tie_word_embeddings"
        case torchDtype = "torch_dtype"
        case useCache = "use_cache"
        case useSlidingWindow = "use_sliding_window"
        case vocabSize = "vocab_size"
    }
}

public struct ZImageTurboModelConfigs: Sendable, Hashable {
    public let modelIndex: ZImageTurboModelIndex?
    public let transformer: ZImageTurboTransformerConfig
    public let vae: ZImageTurboVAEConfig
    public let scheduler: ZImageTurboSchedulerConfig
    public let textEncoder: ZImageTurboTextEncoderConfig

    public static func load(
        from resources: ZImageTurboResources,
        decoder: JSONDecoder = JSONDecoder(),
        fileManager: FileManager = .default
    ) throws -> ZImageTurboModelConfigs {
        func decode<T: Decodable>(_ type: T.Type, url: URL) throws -> T {
            try decoder.decode(T.self, from: Data(contentsOf: url))
        }

        return ZImageTurboModelConfigs(
            modelIndex: fileManager.fileExists(atPath: resources.modelIndexURL.path)
                ? try decode(ZImageTurboModelIndex.self, url: resources.modelIndexURL)
                : nil,
            transformer: try decode(ZImageTurboTransformerConfig.self, url: resources.transformerConfigURL),
            vae: try decode(ZImageTurboVAEConfig.self, url: resources.vaeConfigURL),
            scheduler: try decode(ZImageTurboSchedulerConfig.self, url: resources.schedulerConfigURL),
            textEncoder: try decode(ZImageTurboTextEncoderConfig.self, url: resources.textEncoderConfigURL)
        )
    }
}
