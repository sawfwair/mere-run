import Foundation

public struct Flux1TransformerConfiguration: Decodable, Sendable, Hashable {
    public let attentionHeadDim: Int
    public let axesDimsRope: [Int]
    public let guidanceEmbeds: Bool
    public let inChannels: Int
    public let jointAttentionDim: Int
    public let numAttentionHeads: Int
    public let numLayers: Int
    public let numSingleLayers: Int
    public let patchSize: Int
    public let pooledProjectionDim: Int

    public var hiddenSize: Int { attentionHeadDim * numAttentionHeads }
    public var outChannels: Int { inChannels }

    private enum CodingKeys: String, CodingKey {
        case attentionHeadDim = "attention_head_dim"
        case axesDimsRope = "axes_dims_rope"
        case guidanceEmbeds = "guidance_embeds"
        case inChannels = "in_channels"
        case jointAttentionDim = "joint_attention_dim"
        case numAttentionHeads = "num_attention_heads"
        case numLayers = "num_layers"
        case numSingleLayers = "num_single_layers"
        case patchSize = "patch_size"
        case pooledProjectionDim = "pooled_projection_dim"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.attentionHeadDim = try container.decode(Int.self, forKey: .attentionHeadDim)
        self.axesDimsRope = try container.decodeIfPresent([Int].self, forKey: .axesDimsRope) ?? [16, 56, 56]
        self.guidanceEmbeds = try container.decode(Bool.self, forKey: .guidanceEmbeds)
        self.inChannels = try container.decode(Int.self, forKey: .inChannels)
        self.jointAttentionDim = try container.decode(Int.self, forKey: .jointAttentionDim)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numLayers = try container.decode(Int.self, forKey: .numLayers)
        self.numSingleLayers = try container.decode(Int.self, forKey: .numSingleLayers)
        self.patchSize = try container.decode(Int.self, forKey: .patchSize)
        self.pooledProjectionDim = try container.decode(Int.self, forKey: .pooledProjectionDim)
    }
}

public struct Flux1VAEConfiguration: Decodable, Sendable, Hashable {
    public let blockOutChannels: [Int]
    public let inChannels: Int
    public let latentChannels: Int
    public let layersPerBlock: Int
    public let normNumGroups: Int
    public let outChannels: Int
    public let scalingFactor: Float
    public let shiftFactor: Float

    private enum CodingKeys: String, CodingKey {
        case blockOutChannels = "block_out_channels"
        case inChannels = "in_channels"
        case latentChannels = "latent_channels"
        case layersPerBlock = "layers_per_block"
        case normNumGroups = "norm_num_groups"
        case outChannels = "out_channels"
        case scalingFactor = "scaling_factor"
        case shiftFactor = "shift_factor"
    }

    public var coreConfiguration: VAEConfig {
        VAEConfig(
            inChannels: inChannels,
            outChannels: outChannels,
            latentChannels: latentChannels,
            scalingFactor: scalingFactor,
            shiftFactor: shiftFactor,
            blockOutChannels: blockOutChannels,
            layersPerBlock: layersPerBlock,
            normNumGroups: normNumGroups
        )
    }
}

public struct Flux1CLIPConfiguration: Decodable, Sendable, Hashable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let layerNormEps: Float
    public let maxPositionEmbeddings: Int
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let projectionDim: Int
    public let vocabSize: Int

    private enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case layerNormEps = "layer_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case projectionDim = "projection_dim"
        case vocabSize = "vocab_size"
    }
}

public struct Flux1T5Configuration: Decodable, Sendable, Hashable {
    public let dFF: Int
    public let dKV: Int
    public let dModel: Int
    public let layerNormEpsilon: Float
    public let numHeads: Int
    public let numLayers: Int
    public let relativeAttentionNumBuckets: Int
    public let vocabSize: Int

    private enum CodingKeys: String, CodingKey {
        case dFF = "d_ff"
        case dKV = "d_kv"
        case dModel = "d_model"
        case layerNormEpsilon = "layer_norm_epsilon"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case relativeAttentionNumBuckets = "relative_attention_num_buckets"
        case vocabSize = "vocab_size"
    }

    public var wanConfiguration: Wan2TextEncoderConfiguration {
        Wan2TextEncoderConfiguration(
            vocabularySize: vocabSize,
            hiddenSize: dModel,
            attentionSize: dKV * numHeads,
            feedForwardSize: dFF,
            headCount: numHeads,
            layerCount: numLayers,
            relativePositionBuckets: relativeAttentionNumBuckets
        )
    }
}

public struct Flux1SchedulerConfiguration: Decodable, Sendable, Hashable {
    public let baseImageSequenceLength: Int
    public let baseShift: Float
    public let maxImageSequenceLength: Int
    public let maxShift: Float
    public let numTrainTimesteps: Int
    public let shift: Float
    public let useDynamicShifting: Bool

    private enum CodingKeys: String, CodingKey {
        case baseImageSequenceLength = "base_image_seq_len"
        case baseShift = "base_shift"
        case maxImageSequenceLength = "max_image_seq_len"
        case maxShift = "max_shift"
        case numTrainTimesteps = "num_train_timesteps"
        case shift
        case useDynamicShifting = "use_dynamic_shifting"
    }
}

public struct Flux1Resources: Sendable, Hashable {
    public static let modelID = "image-flux1-dev"
    public static let upstreamRepoID = "black-forest-labs/FLUX.1-dev"
    public static let upstreamRevision = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
    public static let estimatedDownloadBytes: Int64 = 34_000_000_000
    public static let snapshotPatterns = [
        "LICENSE*",
        "README.md",
        "model_index.json",
        "scheduler/**",
        "text_encoder/**",
        "text_encoder_2/**",
        "tokenizer/**",
        "tokenizer_2/**",
        "transformer/**",
        "vae/**",
    ]

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var transformerDirectoryURL: URL { rootURL.appendingPathComponent("transformer", isDirectory: true) }
    public var transformerConfigURL: URL { transformerDirectoryURL.appendingPathComponent("config.json") }
    public var transformerWeightsIndexURL: URL {
        transformerDirectoryURL.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
    }
    public var clipDirectoryURL: URL { rootURL.appendingPathComponent("text_encoder", isDirectory: true) }
    public var clipConfigURL: URL { clipDirectoryURL.appendingPathComponent("config.json") }
    public var clipWeightsURL: URL { clipDirectoryURL.appendingPathComponent("model.safetensors") }
    public var clipTokenizerURL: URL { rootURL.appendingPathComponent("tokenizer", isDirectory: true) }
    public var clipTokenizerVocabURL: URL { clipTokenizerURL.appendingPathComponent("vocab.json") }
    public var clipTokenizerMergesURL: URL { clipTokenizerURL.appendingPathComponent("merges.txt") }
    public var t5DirectoryURL: URL { rootURL.appendingPathComponent("text_encoder_2", isDirectory: true) }
    public var t5ConfigURL: URL { t5DirectoryURL.appendingPathComponent("config.json") }
    public var t5WeightsIndexURL: URL { t5DirectoryURL.appendingPathComponent("model.safetensors.index.json") }
    public var t5TokenizerURL: URL { rootURL.appendingPathComponent("tokenizer_2", isDirectory: true) }
    public var vaeDirectoryURL: URL { rootURL.appendingPathComponent("vae", isDirectory: true) }
    public var vaeConfigURL: URL { vaeDirectoryURL.appendingPathComponent("config.json") }
    public var vaeWeightsURL: URL { vaeDirectoryURL.appendingPathComponent("diffusion_pytorch_model.safetensors") }
    public var schedulerConfigURL: URL {
        rootURL.appendingPathComponent("scheduler", isDirectory: true).appendingPathComponent("scheduler_config.json")
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        [
            transformerConfigURL,
            transformerWeightsIndexURL,
            clipConfigURL,
            clipWeightsURL,
            clipTokenizerURL.appendingPathComponent("tokenizer_config.json"),
            clipTokenizerVocabURL,
            clipTokenizerMergesURL,
            t5ConfigURL,
            t5WeightsIndexURL,
            t5TokenizerURL.appendingPathComponent("tokenizer.json"),
            vaeConfigURL,
            vaeWeightsURL,
            schedulerConfigURL,
        ].filter { !fileManager.fileExists(atPath: $0.path) }
    }
}
