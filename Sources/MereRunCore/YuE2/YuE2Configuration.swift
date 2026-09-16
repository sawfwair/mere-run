import Foundation

public enum YuE2Error: LocalizedError {
    case invalidConfiguration(String)
    case invalidRequest(String)
    case invalidWeights(String)
    case invalidAudio(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message): "Invalid YuE2 configuration: \(message)"
        case .invalidRequest(let message): "Invalid YuE2 request: \(message)"
        case .invalidWeights(let message): "Invalid YuE2 checkpoint: \(message)"
        case .invalidAudio(let message): "Invalid YuE2 audio: \(message)"
        }
    }
}

struct YuE2Configuration: Codable, Equatable, Sendable {
    var modelType = "yue2"
    var hiddenSize = 2048
    var numHiddenLayers = 28
    var numAttentionHeads = 16
    var numKeyValueHeads = 8
    var headDim = 128
    var intermediateSize = 6144
    var vocabSize = 184704
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 1_000_000
    var maxPositionEmbeddings = 24576
    var latentType = "vae"
    var latentDim = 64
    var maxLatentFrames = 24576
    var timestepShift: Float = 1
    var tieWordEmbeddings = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", hiddenSize = "hidden_size", numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads", numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim", intermediateSize = "intermediate_size", vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps", ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings", latentType = "latent_type"
        case latentDim = "latent_dim", maxLatentFrames = "max_latent_frames"
        case timestepShift = "timestep_shift", tieWordEmbeddings = "tie_word_embeddings"
    }

    func validateReleased() throws {
        guard self == Self() else {
            throw YuE2Error.invalidConfiguration("Expected the pinned YuE2-3B architecture.")
        }
    }
}

struct YuE2VAEConfiguration: Codable, Sendable {
    struct Decoder: Codable, Sendable {
        var channels = 64
        var channelMultipliers = [1, 2, 4, 8, 16, 32]
        var strides = [2, 2, 4, 4, 5, 6]
        var latentDim = 64
        var outChannels = 2
        var useSnake = true
        var snakeType = "vanilla"
        var finalTanh = false
        var useFilter = false
        var antialiasActivation: Bool?
        var useNearestUpsample: Bool?

        enum CodingKeys: String, CodingKey {
            case channels, strides
            case channelMultipliers = "c_mults", latentDim = "latent_dim", outChannels = "out_channels"
            case useSnake = "use_snake", snakeType = "snake_type", finalTanh = "final_tanh"
            case useFilter = "use_filter", antialiasActivation = "antialias_activation"
            case useNearestUpsample = "use_nearest_upsample"
        }
    }

    var modelType = "yue2_vae"
    var decoderConfig = Decoder()
    var sampleRate = 48000
    var latentDim = 64
    var downsamplingRatio = 1920
    var audioChannels = 2
    var releaseVariant = "standard"
    var decodeCoreFrames = 1024
    var decodeHaloFrames = 16

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", decoderConfig = "decoder_config", sampleRate = "sample_rate"
        case latentDim = "latent_dim", downsamplingRatio = "downsampling_ratio"
        case audioChannels = "audio_channels", releaseVariant = "release_variant"
        case decodeCoreFrames = "decode_core_frames", decodeHaloFrames = "decode_halo_frames"
    }

    func validateReleased() throws {
        let decoder = decoderConfig
        guard modelType == "yue2_vae", sampleRate == 48000, latentDim == 64,
              downsamplingRatio == 1920, audioChannels == 2, releaseVariant == "standard",
              decoder.channels == 64, decoder.channelMultipliers == [1, 2, 4, 8, 16, 32],
              decoder.strides == [2, 2, 4, 4, 5, 6], decoder.latentDim == 64,
              decoder.outChannels == 2, decoder.useSnake, decoder.snakeType == "vanilla",
              !decoder.finalTanh, !decoder.useFilter, decoder.antialiasActivation != true,
              decoder.useNearestUpsample != true, decodeCoreFrames == 1024, decodeHaloFrames == 16 else {
            throw YuE2Error.invalidConfiguration("Expected the pinned standard YuE2 VAE decoder.")
        }
    }
}
