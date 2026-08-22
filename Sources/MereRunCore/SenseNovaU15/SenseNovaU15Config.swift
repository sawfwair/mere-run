import Foundation

public struct SenseNovaU15Config: Decodable, Sendable, Hashable {
    public let downsampleRatio: Float
    public let patchSize: Int
    public let timestepShift: Float
    public let timeSchedule: String
    public let timeShiftType: String
    public let baseShift: Float
    public let maxShift: Float
    public let baseImageSequenceLength: Int
    public let maxImageSequenceLength: Int
    public let noiseScaleMode: String
    public let noiseScaleBaseImageSequenceLength: Int
    public let addNoiseScaleEmbedding: Bool
    public let noiseScaleMaxValue: Float
    public let noiseScale: Float
    public let tEpsilon: Float
    public let llmConfig: LLMConfig
    public let visionConfig: VisionConfig

    public struct LLMConfig: Decodable, Sendable, Hashable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let headDimension: Int
        public let numberOfAttentionHeads: Int
        public let numberOfKeyValueHeads: Int
        public let numberOfHiddenLayers: Int
        public let vocabularySize: Int
        public let rmsNormEpsilon: Float
        public let ropeTheta: Float
        public let ropeThetaHW: Float
        public let attentionBias: Bool

        private enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case headDimension = "head_dim"
            case numberOfAttentionHeads = "num_attention_heads"
            case numberOfKeyValueHeads = "num_key_value_heads"
            case numberOfHiddenLayers = "num_hidden_layers"
            case vocabularySize = "vocab_size"
            case rmsNormEpsilon = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeThetaHW = "rope_theta_hw"
            case attentionBias = "attention_bias"
        }
    }

    public struct VisionConfig: Decodable, Sendable, Hashable {
        public let hiddenSize: Int
        public let numberOfChannels: Int
        public let patchSize: Int
        public let ropeTheta: Float

        private enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numberOfChannels = "num_channels"
            case patchSize = "patch_size"
            case ropeTheta = "rope_theta_vision"
        }
    }

    private enum CodingKeys: String, CodingKey {
        case downsampleRatio = "downsample_ratio"
        case patchSize = "patch_size"
        case timestepShift = "timestep_shift"
        case timeSchedule = "time_schedule"
        case timeShiftType = "time_shift_type"
        case baseShift = "base_shift"
        case maxShift = "max_shift"
        case baseImageSequenceLength = "base_image_seq_len"
        case maxImageSequenceLength = "max_image_seq_len"
        case noiseScaleMode = "noise_scale_mode"
        case noiseScaleBaseImageSequenceLength = "noise_scale_base_image_seq_len"
        case addNoiseScaleEmbedding = "add_noise_scale_embedding"
        case noiseScaleMaxValue = "noise_scale_max_value"
        case noiseScale = "noise_scale"
        case tEpsilon = "t_eps"
        case llmConfig = "llm_config"
        case visionConfig = "vision_config"
    }

    public static func load(from resources: SenseNovaU15Resources) throws -> SenseNovaU15Config {
        let value = try JSONDecoder().decode(Self.self, from: Data(contentsOf: resources.configURL))
        try value.validate()
        return value
    }

    public func validate() throws {
        guard patchSize > 0, downsampleRatio > 0,
              llmConfig.hiddenSize > 0,
              llmConfig.headDimension.isMultiple(of: 4),
              llmConfig.numberOfAttentionHeads.isMultiple(of: llmConfig.numberOfKeyValueHeads),
              llmConfig.hiddenSize == llmConfig.numberOfAttentionHeads * llmConfig.headDimension,
              visionConfig.patchSize == patchSize else {
            throw SenseNovaU15Error.invalidConfiguration("inconsistent model dimensions")
        }
    }
}
