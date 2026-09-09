import Foundation

public struct Qwen3TTSQuantizationConfig: Decodable, Sendable, Hashable {
    public let groupSize: Int
    public let bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }

    public init(groupSize: Int = 64, bits: Int = 4) {
        self.groupSize = groupSize
        self.bits = bits
    }
}

public struct Qwen3TTSRopeScalingConfig: Decodable, Sendable, Hashable {
    public let interleaved: Bool
    public let mropeSection: [Int]
    public let ropeType: String

    enum CodingKeys: String, CodingKey {
        case interleaved
        case mropeSection = "mrope_section"
        case ropeType = "rope_type"
    }

    public init(
        interleaved: Bool = true,
        mropeSection: [Int] = [24, 20, 20],
        ropeType: String = "default"
    ) {
        self.interleaved = interleaved
        self.mropeSection = mropeSection
        self.ropeType = ropeType
    }
}

public struct Qwen3TTSSpeakerEncoderConfig: Decodable, Sendable, Hashable {
    public let melDim: Int
    public let encDim: Int
    public let encChannels: [Int]
    public let encKernelSizes: [Int]
    public let encDilations: [Int]
    public let encAttentionChannels: Int
    public let encRes2netScale: Int
    public let encSeChannels: Int
    public let sampleRate: Int

    enum CodingKeys: String, CodingKey {
        case melDim = "mel_dim"
        case encDim = "enc_dim"
        case encChannels = "enc_channels"
        case encKernelSizes = "enc_kernel_sizes"
        case encDilations = "enc_dilations"
        case encAttentionChannels = "enc_attention_channels"
        case encRes2netScale = "enc_res2net_scale"
        case encSeChannels = "enc_se_channels"
        case sampleRate = "sample_rate"
    }

    public init(
        melDim: Int = 128,
        encDim: Int = 1024,
        encChannels: [Int] = [512, 512, 512, 512, 1536],
        encKernelSizes: [Int] = [5, 3, 3, 3, 1],
        encDilations: [Int] = [1, 2, 3, 4, 1],
        encAttentionChannels: Int = 128,
        encRes2netScale: Int = 8,
        encSeChannels: Int = 128,
        sampleRate: Int = 24000
    ) {
        self.melDim = melDim
        self.encDim = encDim
        self.encChannels = encChannels
        self.encKernelSizes = encKernelSizes
        self.encDilations = encDilations
        self.encAttentionChannels = encAttentionChannels
        self.encRes2netScale = encRes2netScale
        self.encSeChannels = encSeChannels
        self.sampleRate = sampleRate
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        melDim = try container.decodeIfPresent(Int.self, forKey: .melDim) ?? 128
        encDim = try container.decodeIfPresent(Int.self, forKey: .encDim) ?? 1024
        encChannels = try container.decodeIfPresent([Int].self, forKey: .encChannels) ?? [512, 512, 512, 512, 1536]
        encKernelSizes = try container.decodeIfPresent([Int].self, forKey: .encKernelSizes) ?? [5, 3, 3, 3, 1]
        encDilations = try container.decodeIfPresent([Int].self, forKey: .encDilations) ?? [1, 2, 3, 4, 1]
        encAttentionChannels = try container.decodeIfPresent(Int.self, forKey: .encAttentionChannels) ?? 128
        encRes2netScale = try container.decodeIfPresent(Int.self, forKey: .encRes2netScale) ?? 8
        encSeChannels = try container.decodeIfPresent(Int.self, forKey: .encSeChannels) ?? 128
        sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
    }
}

public struct Qwen3TTSModelConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let talkerConfig: Qwen3TTSTalkerConfig
    public let speakerEncoderConfig: Qwen3TTSSpeakerEncoderConfig?
    public let tokenizerConfig: Qwen3TTSTokenizerConfig?
    public let tokenizerType: String
    public let ttsModelSize: String
    public let ttsModelType: String
    public let imStartTokenId: Int
    public let imEndTokenId: Int
    public let ttsPadTokenId: Int
    public let ttsBosTokenId: Int
    public let ttsEosTokenId: Int
    public let sampleRate: Int
    public let quantization: Qwen3TTSQuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case talkerConfig = "talker_config"
        case speakerEncoderConfig = "speaker_encoder_config"
        case tokenizerConfig = "tokenizer_config"
        case tokenizerType = "tokenizer_type"
        case ttsModelSize = "tts_model_size"
        case ttsModelType = "tts_model_type"
        case imStartTokenId = "im_start_token_id"
        case imEndTokenId = "im_end_token_id"
        case ttsPadTokenId = "tts_pad_token_id"
        case ttsBosTokenId = "tts_bos_token_id"
        case ttsEosTokenId = "tts_eos_token_id"
        case sampleRate = "sample_rate"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_tts"
        talkerConfig = try container.decodeIfPresent(Qwen3TTSTalkerConfig.self, forKey: .talkerConfig) ?? Qwen3TTSTalkerConfig()
        speakerEncoderConfig = try container.decodeIfPresent(Qwen3TTSSpeakerEncoderConfig.self, forKey: .speakerEncoderConfig)
        tokenizerConfig = try container.decodeIfPresent(Qwen3TTSTokenizerConfig.self, forKey: .tokenizerConfig)
        tokenizerType = try container.decodeIfPresent(String.self, forKey: .tokenizerType) ?? "qwen3_tts_tokenizer_12hz"
        ttsModelSize = try container.decodeIfPresent(String.self, forKey: .ttsModelSize) ?? "0b6"
        ttsModelType = try container.decodeIfPresent(String.self, forKey: .ttsModelType) ?? "base"
        imStartTokenId = try container.decodeIfPresent(Int.self, forKey: .imStartTokenId) ?? 151644
        imEndTokenId = try container.decodeIfPresent(Int.self, forKey: .imEndTokenId) ?? 151645
        ttsPadTokenId = try container.decodeIfPresent(Int.self, forKey: .ttsPadTokenId) ?? 151671
        ttsBosTokenId = try container.decodeIfPresent(Int.self, forKey: .ttsBosTokenId) ?? 151672
        ttsEosTokenId = try container.decodeIfPresent(Int.self, forKey: .ttsEosTokenId) ?? 151673
        sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
        quantization = try container.decodeIfPresent(Qwen3TTSQuantizationConfig.self, forKey: .quantization)
    }

    public static func load(from url: URL) throws -> Qwen3TTSModelConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Qwen3TTSModelConfig.self, from: data)
    }
}
