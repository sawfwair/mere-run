import Foundation

public struct HiDreamO1Config: Decodable, Sendable, Hashable {
    public var architectures: [String]
    public var modelType: String
    public var imageTokenId: Int
    public var videoTokenId: Int
    public var visionStartTokenId: Int
    public var visionEndTokenId: Int
    public var textConfig: TextConfig
    public var visionConfig: VisionConfig

    public struct TextConfig: Decodable, Sendable, Hashable {
        public var attentionBias: Bool
        public var attentionDropout: Float
        public var bosTokenId: Int
        public var eosTokenId: Int
        public var headDim: Int
        public var hiddenAct: String
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var maxPositionEmbeddings: Int
        public var numAttentionHeads: Int
        public var numHiddenLayers: Int
        public var numKeyValueHeads: Int
        public var rmsNormEps: Float
        public var ropeTheta: Float
        public var vocabSize: Int
        public var ropeScaling: RopeScaling

        private enum CodingKeys: String, CodingKey {
            case attentionBias = "attention_bias"
            case attentionDropout = "attention_dropout"
            case bosTokenId = "bos_token_id"
            case eosTokenId = "eos_token_id"
            case headDim = "head_dim"
            case hiddenAct = "hidden_act"
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
        }
    }

    public struct RopeScaling: Decodable, Sendable, Hashable {
        public var mropeInterleaved: Bool
        public var mropeSection: [Int]
        public var ropeType: String

        private enum CodingKeys: String, CodingKey {
            case mropeInterleaved = "mrope_interleaved"
            case mropeSection = "mrope_section"
            case ropeType = "rope_type"
        }
    }

    public struct VisionConfig: Decodable, Sendable, Hashable {
        public var deepstackVisualIndexes: [Int]
        public var depth: Int
        public var hiddenAct: String
        public var hiddenSize: Int
        public var inChannels: Int
        public var intermediateSize: Int
        public var numHeads: Int
        public var numPositionEmbeddings: Int
        public var outHiddenSize: Int
        public var patchSize: Int
        public var spatialMergeSize: Int
        public var temporalPatchSize: Int

        private enum CodingKeys: String, CodingKey {
            case deepstackVisualIndexes = "deepstack_visual_indexes"
            case depth
            case hiddenAct = "hidden_act"
            case hiddenSize = "hidden_size"
            case inChannels = "in_channels"
            case intermediateSize = "intermediate_size"
            case numHeads = "num_heads"
            case numPositionEmbeddings = "num_position_embeddings"
            case outHiddenSize = "out_hidden_size"
            case patchSize = "patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
        }
    }

    private enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
    }

    public static func load(from resources: HiDreamO1Resources) throws -> HiDreamO1Config {
        let data = try Data(contentsOf: resources.configURL)
        return try JSONDecoder().decode(HiDreamO1Config.self, from: data)
    }
}
