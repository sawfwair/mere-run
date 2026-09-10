import Foundation

public struct MiniMaxH3QuantizationConfiguration: Codable, Hashable, Sendable {
    package init(bits: Int, groupSize: Int, mode: String) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }

    public let bits: Int
    public let groupSize: Int
    public let mode: String

    package enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case mode
    }
}

public struct MiniMaxH3Configuration: Decodable, Hashable, Sendable {
    public let modelType: String
    public let task: String
    public let hiddenSize: Int
    public let layerCount: Int
    public let refinerLayerCount: Int
    public let attentionHeadCount: Int
    public let attentionHeadDimension: Int
    public let feedForwardSize: Int
    public let videoLatentChannels: Int
    public let audioLatentChannels: Int
    public let patchSize: [Int]
    public let textDimension: Int
    public let timeFrequencyDimension: Int
    public let timeEmbeddingHiddenSize: Int
    public let timeEmbeddingDimension: Int
    public let videoFlowShift: Float
    public let audioFlowShift: Float
    public let sampleSteps: Int
    public let quantization: MiniMaxH3QuantizationConfiguration?
    public let textEncoderQuantization: MiniMaxH3QuantizationConfiguration?

    private struct Transformer: Decodable {
        let hiddenSize: Int
        let layerCount: Int
        let attentionHeadCount: Int
        let attentionHeadDimension: Int
        let feedForwardSize: Int
        let videoLatentChannels: Int
        let audioLatentChannels: Int
        let textDimension: Int
        let timeEmbeddingHiddenSize: Int?
        let timeEmbeddingDimension: Int
        let ropeFrequencyCount: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case layerCount = "num_layers"
            case attentionHeadCount = "num_attention_heads"
            case attentionHeadDimension = "attention_head_dim"
            case feedForwardSize = "ffn_hidden_size"
            case videoLatentChannels = "latents_dim"
            case audioLatentChannels = "audio_latents_dim"
            case textDimension = "text_dim"
            case timeEmbeddingHiddenSize = "time_embed_hidden_dim"
            case timeEmbeddingDimension = "time_embed_dim"
            case ropeFrequencyCount = "rope_inv_freq_len"
        }
    }

    private struct SigmaShifts: Decodable {
        let video: Float
        let audio: Float
    }

    package enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case partition
        case transformer
        case sigmaShifts = "sigma_shift_scales"
        case quantization
        case textEncoderQuantization = "text_encoder_quantization"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let transformer = try container.decode(Transformer.self, forKey: .transformer)
        let shifts = try container.decode(SigmaShifts.self, forKey: .sigmaShifts)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.task = try container.decode(String.self, forKey: .partition)
        self.hiddenSize = transformer.hiddenSize
        self.layerCount = transformer.layerCount
        self.refinerLayerCount = 2
        self.attentionHeadCount = transformer.attentionHeadCount
        self.attentionHeadDimension = transformer.attentionHeadDimension
        self.feedForwardSize = transformer.feedForwardSize
        self.videoLatentChannels = transformer.videoLatentChannels
        self.audioLatentChannels = transformer.audioLatentChannels
        self.patchSize = [1, 2, 2]
        self.textDimension = transformer.textDimension
        self.timeFrequencyDimension = 256
        self.timeEmbeddingHiddenSize = transformer.timeEmbeddingHiddenSize ?? transformer.hiddenSize
        self.timeEmbeddingDimension = transformer.timeEmbeddingDimension
        self.videoFlowShift = shifts.video
        self.audioFlowShift = shifts.audio
        self.sampleSteps = 31
        self.quantization = try container.decodeIfPresent(
            MiniMaxH3QuantizationConfiguration.self,
            forKey: .quantization
        )
        self.textEncoderQuantization = try container.decodeIfPresent(
            MiniMaxH3QuantizationConfiguration.self,
            forKey: .textEncoderQuantization
        ) ?? quantization
    }

    public func validationIssues() -> [String] {
        var issues: [String] = []
        if modelType != "minimax_h3" { issues.append("model_type must be minimax_h3") }
        if task != "fl2va" && task != "ref2va" { issues.append("task must be fl2va or ref2va") }
        if hiddenSize != 5_376 { issues.append("hidden_size must be 5376") }
        if layerCount != 50 { issues.append("num_layers must be 50") }
        if refinerLayerCount != 2 { issues.append("num_refiner_layers must be 2") }
        if attentionHeadCount != 56 || attentionHeadDimension != 128 {
            issues.append("attention geometry must be 56 heads x 128")
        }
        if feedForwardSize != 14_336 { issues.append("ffn_dim must be 14336") }
        if videoLatentChannels != 24 || audioLatentChannels != 32 {
            issues.append("video/audio latent channels must be 24/32")
        }
        if patchSize != [1, 2, 2] { issues.append("patch_size must be [1, 2, 2]") }
        if textDimension != 5_120 { issues.append("text_dim must be 5120") }
        if timeFrequencyDimension != 256
            || timeEmbeddingHiddenSize != 5_376
            || timeEmbeddingDimension != 2_688 {
            issues.append("time embedding must be 256 -> 5376 -> 2688")
        }
        if videoFlowShift <= 0 || audioFlowShift <= 0 { issues.append("flow shifts must be positive") }
        if sampleSteps < 2 { issues.append("sample_steps must be at least 2") }
        return issues
    }
}
