import Foundation

public struct ACEStepConfig: Decodable, Sendable, Hashable {
    public let vocabSize: Int
    public let fsqDim: Int
    public let fsqInputLevels: [Int]
    public let fsqInputNumQuantizers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let encoderHiddenSize: Int
    public let encoderIntermediateSize: Int
    public let encoderNumAttentionHeads: Int
    public let encoderNumKeyValueHeads: Int
    public let headDim: Int
    public let hiddenAct: String
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let useSlidingWindow: Bool
    public let slidingWindow: Int?
    public let layerTypes: [String]?

    public let audioAcousticHiddenDim: Int
    public let poolWindowSize: Int
    public let textHiddenDim: Int
    public let inChannels: Int
    public let patchSize: Int

    public let numLyricEncoderHiddenLayers: Int
    public let numAttentionPoolerHiddenLayers: Int
    public let numAudioDecoderHiddenLayers: Int
    public let timbreHiddenDim: Int
    public let numTimbreEncoderHiddenLayers: Int
    public let timbreFixFrame: Int

    public let isTurbo: Bool

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case fsqDim = "fsq_dim"
        case fsqInputLevels = "fsq_input_levels"
        case fsqInputNumQuantizers = "fsq_input_num_quantizers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case encoderHiddenSize = "encoder_hidden_size"
        case encoderIntermediateSize = "encoder_intermediate_size"
        case encoderNumAttentionHeads = "encoder_num_attention_heads"
        case encoderNumKeyValueHeads = "encoder_num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case useSlidingWindow = "use_sliding_window"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"

        case audioAcousticHiddenDim = "audio_acoustic_hidden_dim"
        case poolWindowSize = "pool_window_size"
        case textHiddenDim = "text_hidden_dim"
        case inChannels = "in_channels"
        case patchSize = "patch_size"

        case numLyricEncoderHiddenLayers = "num_lyric_encoder_hidden_layers"
        case numAttentionPoolerHiddenLayers = "num_attention_pooler_hidden_layers"
        case numAudioDecoderHiddenLayers = "num_audio_decoder_hidden_layers"
        case timbreHiddenDim = "timbre_hidden_dim"
        case numTimbreEncoderHiddenLayers = "num_timbre_encoder_hidden_layers"
        case timbreFixFrame = "timbre_fix_frame"

        case isTurbo = "is_turbo"
    }

    public init(
        vocabSize: Int = 64003,
        fsqDim: Int = 2048,
        fsqInputLevels: [Int] = [8, 8, 8, 5, 5, 5],
        fsqInputNumQuantizers: Int = 1,
        hiddenSize: Int = 2048,
        intermediateSize: Int = 6144,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        encoderHiddenSize: Int? = nil,
        encoderIntermediateSize: Int? = nil,
        encoderNumAttentionHeads: Int? = nil,
        encoderNumKeyValueHeads: Int? = nil,
        headDim: Int = 128,
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 32768,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        useSlidingWindow: Bool = true,
        slidingWindow: Int? = 128,
        layerTypes: [String]? = nil,
        audioAcousticHiddenDim: Int = 64,
        poolWindowSize: Int = 5,
        textHiddenDim: Int = 1024,
        inChannels: Int = 192,
        patchSize: Int = 2,
        numLyricEncoderHiddenLayers: Int = 8,
        numAttentionPoolerHiddenLayers: Int = 2,
        numAudioDecoderHiddenLayers: Int = 24,
        timbreHiddenDim: Int = 64,
        numTimbreEncoderHiddenLayers: Int = 4,
        timbreFixFrame: Int = 750,
        isTurbo: Bool = true
    ) {
        self.vocabSize = vocabSize
        self.fsqDim = fsqDim
        self.fsqInputLevels = fsqInputLevels
        self.fsqInputNumQuantizers = fsqInputNumQuantizers
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.encoderHiddenSize = encoderHiddenSize ?? hiddenSize
        self.encoderIntermediateSize = encoderIntermediateSize ?? intermediateSize
        self.encoderNumAttentionHeads = encoderNumAttentionHeads ?? numAttentionHeads
        self.encoderNumKeyValueHeads = encoderNumKeyValueHeads ?? numKeyValueHeads
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.useSlidingWindow = useSlidingWindow
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes
        self.audioAcousticHiddenDim = audioAcousticHiddenDim
        self.poolWindowSize = poolWindowSize
        self.textHiddenDim = textHiddenDim
        self.inChannels = inChannels
        self.patchSize = patchSize
        self.numLyricEncoderHiddenLayers = numLyricEncoderHiddenLayers
        self.numAttentionPoolerHiddenLayers = numAttentionPoolerHiddenLayers
        self.numAudioDecoderHiddenLayers = numAudioDecoderHiddenLayers
        self.timbreHiddenDim = timbreHiddenDim
        self.numTimbreEncoderHiddenLayers = numTimbreEncoderHiddenLayers
        self.timbreFixFrame = timbreFixFrame
        self.isTurbo = isTurbo
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 64003
        fsqDim = try container.decodeIfPresent(Int.self, forKey: .fsqDim) ?? 2048
        fsqInputLevels = try container.decodeIfPresent([Int].self, forKey: .fsqInputLevels) ?? [8, 8, 8, 5, 5, 5]
        fsqInputNumQuantizers = try container.decodeIfPresent(Int.self, forKey: .fsqInputNumQuantizers) ?? 1
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        encoderHiddenSize = try container.decodeIfPresent(Int.self, forKey: .encoderHiddenSize) ?? hiddenSize
        encoderIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .encoderIntermediateSize) ?? intermediateSize
        encoderNumAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .encoderNumAttentionHeads) ?? numAttentionHeads
        encoderNumKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .encoderNumKeyValueHeads) ?? numKeyValueHeads
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / max(1, numAttentionHeads))
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        useSlidingWindow = try container.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? true
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)

        audioAcousticHiddenDim = try container.decodeIfPresent(Int.self, forKey: .audioAcousticHiddenDim) ?? 64
        poolWindowSize = try container.decodeIfPresent(Int.self, forKey: .poolWindowSize) ?? 5
        textHiddenDim = try container.decodeIfPresent(Int.self, forKey: .textHiddenDim) ?? 1024
        inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 192
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 2

        numLyricEncoderHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numLyricEncoderHiddenLayers) ?? 8
        numAttentionPoolerHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numAttentionPoolerHiddenLayers) ?? 2
        numAudioDecoderHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numAudioDecoderHiddenLayers) ?? numHiddenLayers
        timbreHiddenDim = try container.decodeIfPresent(Int.self, forKey: .timbreHiddenDim) ?? 64
        numTimbreEncoderHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numTimbreEncoderHiddenLayers) ?? 4
        timbreFixFrame = try container.decodeIfPresent(Int.self, forKey: .timbreFixFrame) ?? 750

        isTurbo = try container.decodeIfPresent(Bool.self, forKey: .isTurbo) ?? false
    }

    public var conditionEncoderConfig: ACEStepConfig {
        ACEStepConfig(
            vocabSize: vocabSize,
            fsqDim: fsqDim,
            fsqInputLevels: fsqInputLevels,
            fsqInputNumQuantizers: fsqInputNumQuantizers,
            hiddenSize: encoderHiddenSize,
            intermediateSize: encoderIntermediateSize,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: encoderNumAttentionHeads,
            numKeyValueHeads: encoderNumKeyValueHeads,
            encoderHiddenSize: encoderHiddenSize,
            encoderIntermediateSize: encoderIntermediateSize,
            encoderNumAttentionHeads: encoderNumAttentionHeads,
            encoderNumKeyValueHeads: encoderNumKeyValueHeads,
            headDim: headDim,
            hiddenAct: hiddenAct,
            maxPositionEmbeddings: maxPositionEmbeddings,
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            attentionBias: attentionBias,
            attentionDropout: attentionDropout,
            useSlidingWindow: useSlidingWindow,
            slidingWindow: slidingWindow,
            layerTypes: layerTypes,
            audioAcousticHiddenDim: audioAcousticHiddenDim,
            poolWindowSize: poolWindowSize,
            textHiddenDim: textHiddenDim,
            inChannels: inChannels,
            patchSize: patchSize,
            numLyricEncoderHiddenLayers: numLyricEncoderHiddenLayers,
            numAttentionPoolerHiddenLayers: numAttentionPoolerHiddenLayers,
            numAudioDecoderHiddenLayers: numAudioDecoderHiddenLayers,
            timbreHiddenDim: timbreHiddenDim,
            numTimbreEncoderHiddenLayers: numTimbreEncoderHiddenLayers,
            timbreFixFrame: timbreFixFrame,
            isTurbo: isTurbo
        )
    }
}
