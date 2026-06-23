import Foundation

public struct Krea2ModelIndex: Decodable, Sendable, Hashable {
    public let className: String?
    public let diffusersVersion: String?
    public let isDistilled: Bool?
    public let patchSize: Int?
    public let textEncoderSelectLayers: [Int]?

    private enum CodingKeys: String, CodingKey {
        case className = "_class_name"
        case diffusersVersion = "_diffusers_version"
        case isDistilled = "is_distilled"
        case patchSize = "patch_size"
        case textEncoderSelectLayers = "text_encoder_select_layers"
    }
}

public struct Krea2SchedulerConfig: Decodable, Sendable, Hashable {
    public let numTrainTimesteps: Int

    private enum CodingKeys: String, CodingKey {
        case numTrainTimesteps = "num_train_timesteps"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.numTrainTimesteps = try container.decodeIfPresent(Int.self, forKey: .numTrainTimesteps) ?? 1000
    }
}

public struct Krea2TransformerConfiguration: Decodable, Sendable, Hashable {
    public let attentionHeadDim: Int
    public let axesDimsRope: [Int]
    public let inChannels: Int
    public let intermediateSize: Int
    public let normEps: Float
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numLayers: Int
    public let numLayerwiseTextBlocks: Int
    public let numRefinerTextBlocks: Int
    public let numTextLayers: Int
    public let ropeTheta: Float
    public let textHiddenDim: Int
    public let textIntermediateSize: Int
    public let textNumAttentionHeads: Int
    public let textNumKeyValueHeads: Int
    public let timestepEmbedDim: Int

    public var hiddenSize: Int {
        numAttentionHeads * attentionHeadDim
    }

    public var patchSize: Int {
        2
    }

    public var latentChannels: Int {
        inChannels / (patchSize * patchSize)
    }

    private enum CodingKeys: String, CodingKey {
        case attentionHeadDim = "attention_head_dim"
        case axesDimsRope = "axes_dims_rope"
        case inChannels = "in_channels"
        case intermediateSize = "intermediate_size"
        case normEps = "norm_eps"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numLayers = "num_layers"
        case numLayerwiseTextBlocks = "num_layerwise_text_blocks"
        case numRefinerTextBlocks = "num_refiner_text_blocks"
        case numTextLayers = "num_text_layers"
        case ropeTheta = "rope_theta"
        case textHiddenDim = "text_hidden_dim"
        case textIntermediateSize = "text_intermediate_size"
        case textNumAttentionHeads = "text_num_attention_heads"
        case textNumKeyValueHeads = "text_num_key_value_heads"
        case timestepEmbedDim = "timestep_embed_dim"
    }
}

public struct Krea2TextEncoderConfiguration: Decodable, Sendable, Hashable {
    public struct RopeParameters: Decodable, Sendable, Hashable {
        public let mropeInterleaved: Bool?
        public let mropeSection: [Int]?
        public let ropeTheta: Float?

        private enum CodingKeys: String, CodingKey {
            case mropeInterleaved = "mrope_interleaved"
            case mropeSection = "mrope_section"
            case ropeTheta = "rope_theta"
        }
    }

    public let headDim: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let maxPositionEmbeddings: Int
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let rmsNormEps: Float
    public let ropeParameters: RopeParameters?
    public let ropeTheta: Float
    public let vocabSize: Int

    private enum RootKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    private enum CodingKeys: String, CodingKey {
        case headDim = "head_dim"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: RootKeys.self)
        if root.contains(.textConfig) {
            let nested = try root.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            self.headDim = try nested.decode(Int.self, forKey: .headDim)
            self.hiddenSize = try nested.decode(Int.self, forKey: .hiddenSize)
            self.intermediateSize = try nested.decode(Int.self, forKey: .intermediateSize)
            self.maxPositionEmbeddings = try nested.decode(Int.self, forKey: .maxPositionEmbeddings)
            self.numAttentionHeads = try nested.decode(Int.self, forKey: .numAttentionHeads)
            self.numHiddenLayers = try nested.decode(Int.self, forKey: .numHiddenLayers)
            self.numKeyValueHeads = try nested.decode(Int.self, forKey: .numKeyValueHeads)
            self.rmsNormEps = try nested.decode(Float.self, forKey: .rmsNormEps)
            self.ropeParameters = try nested.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
            self.ropeTheta = try nested.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? ropeParameters?.ropeTheta
                ?? 5_000_000.0
            self.vocabSize = try nested.decode(Int.self, forKey: .vocabSize)
            return
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.ropeParameters = try container.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? ropeParameters?.ropeTheta
            ?? 5_000_000.0
        self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
    }

    public var qwenConfiguration: QwenTextEncoderConfiguration {
        QwenTextEncoderConfiguration(
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            intermediateSize: intermediateSize,
            ropeTheta: ropeTheta,
            maxPositionEmbeddings: maxPositionEmbeddings,
            rmsNormEps: rmsNormEps,
            headDim: headDim,
            mropeSection: ropeParameters?.mropeSection,
            mropeInterleaved: ropeParameters?.mropeInterleaved ?? false
        )
    }
}

public struct Krea2ModelConfigs: Sendable, Hashable {
    public let modelIndex: Krea2ModelIndex
    public let scheduler: Krea2SchedulerConfig
    public let textEncoder: Krea2TextEncoderConfiguration
    public let transformer: Krea2TransformerConfiguration
    public let vae: QwenImageEditVAEConfig

    public var selectedTextLayers: [Int] {
        modelIndex.textEncoderSelectLayers ?? [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    }

    public static func load(
        from resources: Krea2Resources,
        decoder: JSONDecoder = JSONDecoder()
    ) throws -> Krea2ModelConfigs {
        Krea2ModelConfigs(
            modelIndex: try decode(Krea2ModelIndex.self, at: resources.modelIndexURL, decoder: decoder),
            scheduler: try decode(Krea2SchedulerConfig.self, at: resources.schedulerConfigURL, decoder: decoder),
            textEncoder: try decode(Krea2TextEncoderConfiguration.self, at: resources.textEncoderConfigURL, decoder: decoder),
            transformer: try decode(Krea2TransformerConfiguration.self, at: resources.transformerConfigURL, decoder: decoder),
            vae: try decode(QwenImageEditVAEConfig.self, at: resources.vaeConfigURL, decoder: decoder)
        )
    }

    private static func decode<T: Decodable>(
        _ type: T.Type,
        at url: URL,
        decoder: JSONDecoder
    ) throws -> T {
        let data = try Data(contentsOf: url)
        return try decoder.decode(type, from: data)
    }
}
