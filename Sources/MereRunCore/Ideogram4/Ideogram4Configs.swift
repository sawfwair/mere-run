import Foundation

public struct Ideogram4TransformerConfiguration: Decodable, Sendable, Hashable {
    public var adalnDim: Int
    public var attentionHeadDim: Int
    public var inChannels: Int
    public var intermediateSize: Int
    public var llmFeaturesDim: Int
    public var mropeSection: [Int]
    public var normEps: Float
    public var numAttentionHeads: Int
    public var numLayers: Int
    public var ropeTheta: Float

    public var embeddingDim: Int {
        numAttentionHeads * attentionHeadDim
    }

    public init(
        adalnDim: Int,
        attentionHeadDim: Int,
        inChannels: Int,
        intermediateSize: Int,
        llmFeaturesDim: Int,
        mropeSection: [Int],
        normEps: Float,
        numAttentionHeads: Int,
        numLayers: Int,
        ropeTheta: Float
    ) {
        self.adalnDim = adalnDim
        self.attentionHeadDim = attentionHeadDim
        self.inChannels = inChannels
        self.intermediateSize = intermediateSize
        self.llmFeaturesDim = llmFeaturesDim
        self.mropeSection = mropeSection
        self.normEps = normEps
        self.numAttentionHeads = numAttentionHeads
        self.numLayers = numLayers
        self.ropeTheta = ropeTheta
    }

    private enum CodingKeys: String, CodingKey {
        case adalnDim = "adaln_dim"
        case attentionHeadDim = "attention_head_dim"
        case inChannels = "in_channels"
        case intermediateSize = "intermediate_size"
        case llmFeaturesDim = "llm_features_dim"
        case mropeSection = "mrope_section"
        case normEps = "norm_eps"
        case numAttentionHeads = "num_attention_heads"
        case numLayers = "num_layers"
        case ropeTheta = "rope_theta"
    }

    public static func load(from directoryURL: URL) throws -> Ideogram4TransformerConfiguration {
        let data = try Data(contentsOf: directoryURL.appending(path: "config.json"))
        return try JSONDecoder().decode(Ideogram4TransformerConfiguration.self, from: data)
    }
}

public struct Ideogram4TextEncoderConfiguration: Decodable, Sendable, Hashable {
    public var headDim: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var maxPositionEmbeddings: Int
    public var numAttentionHeads: Int
    public var numHiddenLayers: Int
    public var numKeyValueHeads: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var vocabSize: Int
    public var ropeScaling: RopeScaling?
    public var ropeParameters: RopeScaling?

    public struct RopeScaling: Decodable, Sendable, Hashable {
        public var mropeInterleaved: Bool?
        public var mropeSection: [Int]?
        public var ropeTheta: Float?

        private enum CodingKeys: String, CodingKey {
            case mropeInterleaved = "mrope_interleaved"
            case mropeSection = "mrope_section"
            case ropeTheta = "rope_theta"
        }
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
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        self.ropeScaling = try container.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling)
        self.ropeParameters = try container.decodeIfPresent(RopeScaling.self, forKey: .ropeParameters)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? ropeParameters?.ropeTheta
            ?? 1_000_000.0
    }

    public static func load(from directoryURL: URL) throws -> Ideogram4TextEncoderConfiguration {
        let data = try Data(contentsOf: directoryURL.appending(path: "config.json"))
        return try JSONDecoder().decode(Ideogram4TextEncoderConfiguration.self, from: data)
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
            mropeSection: ropeScaling?.mropeSection ?? ropeParameters?.mropeSection,
            mropeInterleaved: ropeScaling?.mropeInterleaved ?? ropeParameters?.mropeInterleaved ?? false
        )
    }
}

public struct Ideogram4VAEConfiguration: Decodable, Sendable, Hashable {
    public var inChannels: Int
    public var outChannels: Int
    public var latentChannels: Int
    public var blockOutChannels: [Int]
    public var layersPerBlock: Int
    public var normNumGroups: Int
    public var sampleSize: Int
    public var midBlockAddAttention: Bool
    public var useQuantConv: Bool
    public var usePostQuantConv: Bool

    private enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels"
        case outChannels = "out_channels"
        case latentChannels = "latent_channels"
        case blockOutChannels = "block_out_channels"
        case layersPerBlock = "layers_per_block"
        case normNumGroups = "norm_num_groups"
        case sampleSize = "sample_size"
        case midBlockAddAttention = "mid_block_add_attention"
        case useQuantConv = "use_quant_conv"
        case usePostQuantConv = "use_post_quant_conv"
    }

    public static func load(from directoryURL: URL) throws -> Ideogram4VAEConfiguration {
        let data = try Data(contentsOf: directoryURL.appending(path: "config.json"))
        return try JSONDecoder().decode(Ideogram4VAEConfiguration.self, from: data)
    }

    public var autoencoderConfiguration: VAEConfig {
        VAEConfig(
            inChannels: inChannels,
            outChannels: outChannels,
            latentChannels: latentChannels,
            scalingFactor: 1.0,
            shiftFactor: 0.0,
            blockOutChannels: blockOutChannels,
            layersPerBlock: layersPerBlock,
            normNumGroups: normNumGroups,
            sampleSize: sampleSize,
            midBlockAddAttention: midBlockAddAttention,
            useQuantConv: useQuantConv,
            usePostQuantConv: usePostQuantConv
        )
    }
}
