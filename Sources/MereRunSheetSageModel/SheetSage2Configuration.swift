import Foundation

public enum SheetSage2Error: LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): "SheetSage2: \(message)" }
    }
}

public struct SheetSage2Configuration: Decodable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let decoderLayers: Int
    public let numAttentionHeads: Int
    public let vocabSize: Int
    public let maxOutputSequence: Int
    public let loraRank: Int
    public let loraAlpha: Float
    public let weightsFormat: String
    public let tokenizerFingerprint: String
    public let baseModelRevision: String
    public let windowSeconds: Double
    public let timeHz: Int
    public let samplingRate: Int
    public let tokenizerSchema: String
    public let backbone: MERT2Configuration
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", hiddenSize = "hidden_size", intermediateSize = "intermediate_size"
        case decoderLayers = "decoder_layers", numAttentionHeads = "num_attention_heads", vocabSize = "vocab_size"
        case maxOutputSequence = "max_output_seq_len", loraRank = "lora_rank", loraAlpha = "lora_alpha"
        case weightsFormat = "weights_format", tokenizerFingerprint = "tokenizer_fingerprint"
        case baseModelRevision = "base_model_revision", backbone = "backbone_config"
        case windowSeconds = "input_audio_length", timeHz = "time_hz", samplingRate = "sampling_rate"
        case tokenizerSchema = "tokenizer_schema_version"
    }

    public func validate() throws {
        guard modelType == "sheetsage2", hiddenSize == 512, intermediateSize == 2048, decoderLayers == 6,
              numAttentionHeads == 8, vocabSize == 31678, maxOutputSequence == 5120, loraRank == 64,
              loraAlpha == 128, weightsFormat == "adapter", tokenizerFingerprint == "5ba3325af0344c7f",
              windowSeconds == 300, timeHz == 100, samplingRate == 24000, tokenizerSchema == "v1",
              baseModelRevision == "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42" else {
            throw SheetSage2Error.invalid("Expected the pinned SheetSage2 adapter architecture and vocabulary.")
        }
        try backbone.validate()
    }
}

public struct MERT2Configuration: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let layers: Int
    public let heads: Int
    public let melBins: Int
    public let channels: [Int]
    public let depths: [Int]
    public let kernel: Int
    public let epsilon: Float
    public let subsamplingEpsilon: Float
    public let theta: Float
    public let samplingRate: Int
    public let fftLength: Int
    public let windowLength: Int
    public let hopLength: Int
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", intermediateSize = "intermediate_size", layers = "num_hidden_layers"
        case heads = "num_attention_heads", melBins = "num_mel_bins", channels = "subsampling_channels"
        case depths = "subsampling_depths", kernel = "conv_depthwise_kernel_size", epsilon = "layer_norm_eps"
        case subsamplingEpsilon = "subsampling_layer_norm_eps", theta = "rotary_embedding_base"
        case samplingRate = "sampling_rate", fftLength = "n_fft", windowLength = "win_length", hopLength = "hop_length"
    }
    public func validate() throws {
        guard hiddenSize == 1024, intermediateSize == 4096, layers == 24, heads == 16, melBins == 128,
              channels == [128, 512, 1024], depths == [3, 4, 5], kernel == 31,
              epsilon == 1e-5, subsamplingEpsilon == 1e-6, theta == 10000 else {
            throw SheetSage2Error.invalid("Expected the pinned MERT2 FullSong architecture.")
        }
        guard samplingRate == 24000, fftLength == 2048, windowLength == 2048, hopLength == 240 else {
            throw SheetSage2Error.invalid("Expected the 24 kHz, 2048-point, 240-hop MERT2 frontend.")
        }
    }
}
