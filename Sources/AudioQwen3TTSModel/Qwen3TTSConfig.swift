import Foundation

// MARK: - Model Configuration

/// Root config for Qwen3-TTS model
public struct Qwen3TTSConfig: Decodable, Sendable, Hashable {
    public let talkerConfig: TalkerConfig
    public let quantization: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case talkerConfig = "talker_config"
        case quantization
    }

    public var vocabSize: Int { talkerConfig.vocabSize }
    public var hiddenSize: Int { talkerConfig.hiddenSize }
    public var numHiddenLayers: Int { talkerConfig.numHiddenLayers }
    public var numAttentionHeads: Int { talkerConfig.numAttentionHeads }
    public var numKeyValueHeads: Int { talkerConfig.numKeyValueHeads }
    public var intermediateSize: Int { talkerConfig.intermediateSize }
    public var maxPositionEmbeddings: Int { talkerConfig.maxPositionEmbeddings }
    public var ropeTheta: Float { talkerConfig.ropeTheta }
    public var rmsNormEps: Float { talkerConfig.rmsNormEps }
    public var headDim: Int { talkerConfig.headDim }
    public var textVocabSize: Int { talkerConfig.textVocabSize }

    public var computedHeadDim: Int { headDim }

    public static func load(from url: URL) throws -> Qwen3TTSConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Qwen3TTSConfig.self, from: data)
    }

    /// Talker config with the actual model parameters
    public struct TalkerConfig: Decodable, Sendable, Hashable {
        public let vocabSize: Int
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let intermediateSize: Int
        public let maxPositionEmbeddings: Int
        public let ropeTheta: Float
        public let rmsNormEps: Float
        public let headDim: Int
        public let textVocabSize: Int
        public let textHiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case intermediateSize = "intermediate_size"
            case maxPositionEmbeddings = "max_position_embeddings"
            case ropeTheta = "rope_theta"
            case rmsNormEps = "rms_norm_eps"
            case headDim = "head_dim"
            case textVocabSize = "text_vocab_size"
            case textHiddenSize = "text_hidden_size"
        }
    }

    public struct QuantizationConfig: Decodable, Sendable, Hashable {
        public let groupSize: Int
        public let bits: Int

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }
}
