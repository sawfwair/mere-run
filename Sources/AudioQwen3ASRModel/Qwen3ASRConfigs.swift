import Foundation

// MARK: - Qwen3-ASR Configuration Structures

/// Audio encoder configuration
public struct Qwen3ASRAudioEncoderConfig: Sendable, Hashable {
    public let dModel: Int              // 896
    public let numHiddenLayers: Int     // 18
    public let numAttentionHeads: Int   // 14
    public let ffnDim: Int              // 3584
    public let maxSourcePositions: Int  // 1500
    public let numMelBins: Int          // 128
    public let outputDim: Int           // 1024
    public let downsampleHiddenSize: Int // 480
    public let nWindow: Int
    public let nWindowInfer: Int
    public let convChunkSize: Int
    public let layerNormEps: Float

    public init(
        dModel: Int = 896,
        numHiddenLayers: Int = 18,
        numAttentionHeads: Int = 14,
        ffnDim: Int = 3584,
        maxSourcePositions: Int = 1500,
        numMelBins: Int = 128,
        outputDim: Int = 1024,
        downsampleHiddenSize: Int = 480,
        nWindow: Int = 50,
        nWindowInfer: Int = 800,
        convChunkSize: Int = 500,
        layerNormEps: Float = 1e-5
    ) {
        self.dModel = dModel
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.ffnDim = ffnDim
        self.maxSourcePositions = maxSourcePositions
        self.numMelBins = numMelBins
        self.outputDim = outputDim
        self.downsampleHiddenSize = downsampleHiddenSize
        self.nWindow = nWindow
        self.nWindowInfer = nWindowInfer
        self.convChunkSize = convChunkSize
        self.layerNormEps = layerNormEps
    }

    public var headDim: Int { dModel / numAttentionHeads }
}

/// Decoder configuration (Qwen3 architecture)
public struct Qwen3ASRDecoderConfig: Sendable, Hashable {
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
    public let ropeScaling: Qwen3ASRRoPEScalingConfig?
    public let tieWordEmbeddings: Bool

    public init(
        vocabSize: Int = 151936,
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        intermediateSize: Int = 3072,
        maxPositionEmbeddings: Int = 65536,
        ropeTheta: Float = 1_000_000.0,
        rmsNormEps: Float = 1e-6,
        headDim: Int = 128,
        ropeScaling: Qwen3ASRRoPEScalingConfig? = nil,
        tieWordEmbeddings: Bool = true
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.headDim = headDim
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
    }
}

public struct Qwen3ASRRoPEScalingConfig: Sendable, Hashable {
    public let mropeSection: [Int]?
    public let mropeInterleaved: Bool?

    public init(mropeSection: [Int]? = nil, mropeInterleaved: Bool? = nil) {
        self.mropeSection = mropeSection
        self.mropeInterleaved = mropeInterleaved
    }
}

/// Root configuration for Qwen3-ASR model
public struct Qwen3ASRModelConfig: Sendable, Hashable {
    public let audioConfig: Qwen3ASRAudioEncoderConfig
    public let textConfig: Qwen3ASRDecoderConfig
    public let audioTokenId: Int
    public let audioStartTokenId: Int
    public let audioEndTokenId: Int
    public let supportLanguages: [String]?
    public let quantizationBits: Int?
    public let quantizationGroupSize: Int?

    public init(
        audioConfig: Qwen3ASRAudioEncoderConfig = Qwen3ASRAudioEncoderConfig(),
        textConfig: Qwen3ASRDecoderConfig = Qwen3ASRDecoderConfig(),
        audioTokenId: Int = 151676,
        audioStartTokenId: Int = 151669,
        audioEndTokenId: Int = 151670,
        supportLanguages: [String]? = nil,
        quantizationBits: Int? = nil,
        quantizationGroupSize: Int? = nil
    ) {
        self.audioConfig = audioConfig
        self.textConfig = textConfig
        self.audioTokenId = audioTokenId
        self.audioStartTokenId = audioStartTokenId
        self.audioEndTokenId = audioEndTokenId
        self.supportLanguages = supportLanguages
        self.quantizationBits = quantizationBits
        self.quantizationGroupSize = quantizationGroupSize
    }

    public static func load(from url: URL) throws -> Qwen3ASRModelConfig {
        let data = try Data(contentsOf: url)
        let document = try JSONDecoder().decode(Qwen3ASRModelConfigDocument.self, from: data)

        guard let thinkerConfig = document.thinkerConfig else {
            return Qwen3ASRModelConfig()
        }

        let audioConfig = Qwen3ASRAudioEncoderConfig(
            dModel: thinkerConfig.audioConfig?.dModel ?? 896,
            numHiddenLayers: thinkerConfig.audioConfig?.encoderLayers ?? 18,
            numAttentionHeads: thinkerConfig.audioConfig?.encoderAttentionHeads ?? 14,
            ffnDim: thinkerConfig.audioConfig?.encoderFFNDim ?? 3584,
            maxSourcePositions: thinkerConfig.audioConfig?.maxSourcePositions ?? 1500,
            numMelBins: thinkerConfig.audioConfig?.numMelBins ?? 128,
            outputDim: thinkerConfig.audioConfig?.outputDim ?? 1024,
            downsampleHiddenSize: thinkerConfig.audioConfig?.downsampleHiddenSize ?? 480,
            nWindow: thinkerConfig.audioConfig?.nWindow ?? 50,
            nWindowInfer: thinkerConfig.audioConfig?.nWindowInfer ?? 800,
            convChunkSize: thinkerConfig.audioConfig?.convChunkSize ?? 500
        )

        let ropeScaling = thinkerConfig.textConfig?.ropeScaling.map {
            Qwen3ASRRoPEScalingConfig(
                mropeSection: $0.mropeSection,
                mropeInterleaved: $0.mropeInterleaved
            )
        }

        let textConfig = Qwen3ASRDecoderConfig(
            vocabSize: thinkerConfig.textConfig?.vocabSize ?? 151936,
            hiddenSize: thinkerConfig.textConfig?.hiddenSize ?? 1024,
            numHiddenLayers: thinkerConfig.textConfig?.numHiddenLayers ?? 28,
            numAttentionHeads: thinkerConfig.textConfig?.numAttentionHeads ?? 16,
            numKeyValueHeads: thinkerConfig.textConfig?.numKeyValueHeads ?? 8,
            intermediateSize: thinkerConfig.textConfig?.intermediateSize ?? 3072,
            maxPositionEmbeddings: thinkerConfig.textConfig?.maxPositionEmbeddings ?? 65536,
            ropeTheta: thinkerConfig.textConfig?.ropeTheta ?? 1_000_000.0,
            rmsNormEps: thinkerConfig.textConfig?.rmsNormEps ?? 1e-6,
            headDim: thinkerConfig.textConfig?.headDim ?? 128,
            ropeScaling: ropeScaling,
            tieWordEmbeddings: thinkerConfig.textConfig?.tieWordEmbeddings ?? true
        )

        let quantization = document.quantization ?? document.quantizationConfig

        return Qwen3ASRModelConfig(
            audioConfig: audioConfig,
            textConfig: textConfig,
            audioTokenId: thinkerConfig.audioTokenId ?? 151676,
            audioStartTokenId: thinkerConfig.audioStartTokenId ?? 151669,
            audioEndTokenId: thinkerConfig.audioEndTokenId ?? 151670,
            supportLanguages: document.supportLanguages,
            quantizationBits: quantization?.bits,
            quantizationGroupSize: quantization?.groupSize
        )
    }
}

private struct Qwen3ASRModelConfigDocument: Decodable {
    let thinkerConfig: ThinkerConfig?
    let supportLanguages: [String]?
    let quantization: QuantizationConfig?
    let quantizationConfig: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case thinkerConfig = "thinker_config"
        case supportLanguages = "support_languages"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    struct ThinkerConfig: Decodable {
        let audioConfig: AudioConfig?
        let textConfig: TextConfig?
        let audioTokenId: Int?
        let audioStartTokenId: Int?
        let audioEndTokenId: Int?

        enum CodingKeys: String, CodingKey {
            case audioConfig = "audio_config"
            case textConfig = "text_config"
            case audioTokenId = "audio_token_id"
            case audioStartTokenId = "audio_start_token_id"
            case audioEndTokenId = "audio_end_token_id"
        }
    }

    struct AudioConfig: Decodable {
        let dModel: Int?
        let encoderLayers: Int?
        let encoderAttentionHeads: Int?
        let encoderFFNDim: Int?
        let maxSourcePositions: Int?
        let numMelBins: Int?
        let outputDim: Int?
        let downsampleHiddenSize: Int?
        let nWindow: Int?
        let nWindowInfer: Int?
        let convChunkSize: Int?

        enum CodingKeys: String, CodingKey {
            case dModel = "d_model"
            case encoderLayers = "encoder_layers"
            case encoderAttentionHeads = "encoder_attention_heads"
            case encoderFFNDim = "encoder_ffn_dim"
            case maxSourcePositions = "max_source_positions"
            case numMelBins = "num_mel_bins"
            case outputDim = "output_dim"
            case downsampleHiddenSize = "downsample_hidden_size"
            case nWindow = "n_window"
            case nWindowInfer = "n_window_infer"
            case convChunkSize = "conv_chunksize"
        }
    }

    struct TextConfig: Decodable {
        let vocabSize: Int?
        let hiddenSize: Int?
        let numHiddenLayers: Int?
        let numAttentionHeads: Int?
        let numKeyValueHeads: Int?
        let intermediateSize: Int?
        let maxPositionEmbeddings: Int?
        let ropeTheta: Float?
        let rmsNormEps: Float?
        let headDim: Int?
        let ropeScaling: RopeScalingConfig?
        let tieWordEmbeddings: Bool?

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
            case ropeScaling = "rope_scaling"
            case tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    struct RopeScalingConfig: Decodable {
        let mropeSection: [Int]?
        let mropeInterleaved: Bool?

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
            case mropeInterleaved = "mrope_interleaved"
        }
    }

    struct QuantizationConfig: Decodable {
        let bits: Int?
        let groupSize: Int?

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
        }
    }
}
