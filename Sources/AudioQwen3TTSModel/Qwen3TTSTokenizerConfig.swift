import Foundation

public struct Qwen3TTSTokenizerConfig: Decodable, Sendable, Hashable {
    public let encoderConfig: Qwen3TTSTokenizerEncoderConfig?
    public let decoderConfig: Qwen3TTSTokenizerDecoderConfig
    public let encoderValidNumQuantizers: Int
    public let inputSampleRate: Int
    public let outputSampleRate: Int
    public let decodeUpsampleRate: Int
    public let encodeDownsampleRate: Int

    enum CodingKeys: String, CodingKey {
        case encoderConfig = "encoder_config"
        case decoderConfig = "decoder_config"
        case encoderValidNumQuantizers = "encoder_valid_num_quantizers"
        case inputSampleRate = "input_sample_rate"
        case outputSampleRate = "output_sample_rate"
        case decodeUpsampleRate = "decode_upsample_rate"
        case encodeDownsampleRate = "encode_downsample_rate"
    }

    public init(
        encoderConfig: Qwen3TTSTokenizerEncoderConfig? = nil,
        decoderConfig: Qwen3TTSTokenizerDecoderConfig = Qwen3TTSTokenizerDecoderConfig(),
        encoderValidNumQuantizers: Int = 16,
        inputSampleRate: Int = 24000,
        outputSampleRate: Int = 24000,
        decodeUpsampleRate: Int = 1920,
        encodeDownsampleRate: Int = 1920
    ) {
        self.encoderConfig = encoderConfig
        self.decoderConfig = decoderConfig
        self.encoderValidNumQuantizers = encoderValidNumQuantizers
        self.inputSampleRate = inputSampleRate
        self.outputSampleRate = outputSampleRate
        self.decodeUpsampleRate = decodeUpsampleRate
        self.encodeDownsampleRate = encodeDownsampleRate
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        encoderConfig = try container.decodeIfPresent(Qwen3TTSTokenizerEncoderConfig.self, forKey: .encoderConfig)
        decoderConfig = try container.decodeIfPresent(Qwen3TTSTokenizerDecoderConfig.self, forKey: .decoderConfig)
            ?? Qwen3TTSTokenizerDecoderConfig()
        encoderValidNumQuantizers = try container.decodeIfPresent(Int.self, forKey: .encoderValidNumQuantizers) ?? 16
        inputSampleRate = try container.decodeIfPresent(Int.self, forKey: .inputSampleRate) ?? 24000
        outputSampleRate = try container.decodeIfPresent(Int.self, forKey: .outputSampleRate) ?? 24000
        decodeUpsampleRate = try container.decodeIfPresent(Int.self, forKey: .decodeUpsampleRate) ?? 1920
        encodeDownsampleRate = try container.decodeIfPresent(Int.self, forKey: .encodeDownsampleRate) ?? 1920
    }
}
