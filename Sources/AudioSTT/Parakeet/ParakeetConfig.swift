import Foundation

public enum ParakeetVariant: String, Sendable, Hashable {
    case tdt
    case tdtCTC = "tdt_ctc"
    case rnnt
    case ctc
}

public enum ParakeetPackaging: String, Sendable, Hashable {
    case completeMLX = "complete-mlx"
    case coreMLHybrid = "coreml-hybrid-v1"
}

public struct ParakeetPreprocessorConfig: Sendable, Hashable {
    public let sampleRate: Int
    public let normalize: String
    public let windowSize: Double
    public let windowStride: Double
    public let window: String
    public let features: Int
    public let nFFT: Int
    public let dither: Double
    public let padTo: Int
    public let padValue: Double
    public let preemph: Double

    public var winLength: Int {
        Int(windowSize * Double(sampleRate))
    }

    public var hopLength: Int {
        Int(windowStride * Double(sampleRate))
    }
}

public struct ParakeetEncoderConfig: Sendable, Hashable {
    public let featIn: Int
    public let layers: Int
    public let modelDim: Int
    public let heads: Int
    public let ffExpansionFactor: Int
    public let subsamplingFactor: Int
    public let selfAttentionModel: String
    public let subsampling: String
    public let convKernelSize: Int
    public let subsamplingConvChannels: Int
    public let posEmbMaxLen: Int
    public let causalDownsampling: Bool
    public let useBias: Bool
    public let xScaling: Bool
    public let subsamplingConvChunkingFactor: Int

    public var headDim: Int {
        modelDim / max(1, heads)
    }
}

public struct ParakeetPredictNetConfig: Sendable, Hashable {
    public let predHidden: Int
    public let predRnnLayers: Int
    public let rnnHiddenSize: Int?
}

public struct ParakeetRNNTDecoderConfig: Sendable, Hashable {
    public let blankAsPad: Bool
    public let vocabSize: Int
    public let prednet: ParakeetPredictNetConfig
}

public struct ParakeetCTCDecoderConfig: Sendable, Hashable {
    public let featIn: Int
    public let numClasses: Int
    public let vocabulary: [String]
}

public struct ParakeetJointNetConfig: Sendable, Hashable {
    public let jointHidden: Int
    public let activation: String
    public let encoderHidden: Int
    public let predHidden: Int
}

public struct ParakeetJointConfig: Sendable, Hashable {
    public let numClasses: Int
    public let vocabulary: [String]
    public let jointnet: ParakeetJointNetConfig
    public let numExtraOutputs: Int
}

public struct ParakeetModelConfig: Sendable, Hashable {
    public let packaging: ParakeetPackaging
    public let variant: ParakeetVariant
    public let target: String
    public let preprocessor: ParakeetPreprocessorConfig
    public let encoder: ParakeetEncoderConfig
    public let rnntDecoder: ParakeetRNNTDecoderConfig?
    public let ctcDecoder: ParakeetCTCDecoderConfig?
    public let joint: ParakeetJointConfig?
    public let tdtDurations: [Int]?
    public let maxSymbols: Int?
    public let quantizationBits: Int?
    public let quantizationGroupSize: Int?
    public let supportedLanguageCodes: Set<String>

    public var vocabulary: [String] {
        if let joint { return joint.vocabulary }
        if let ctcDecoder { return ctcDecoder.vocabulary }
        return []
    }

    public static func load(from url: URL) throws -> ParakeetModelConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let raw = try decoder.decode(RawConfig.self, from: data)

        let packaging: ParakeetPackaging
        if let format = raw.mere?.format {
            guard let parsed = ParakeetPackaging(rawValue: format) else {
                throw ParakeetConfigError.unsupportedPackaging(format)
            }
            packaging = parsed
        } else {
            packaging = .completeMLX
        }

        let variant = ParakeetModelConfig.detectVariant(
            target: raw.target,
            tdtDurations: raw.modelDefaults?.tdtDurations
        )

        let preprocessor = ParakeetPreprocessorConfig(
            sampleRate: raw.preprocessor.sampleRate,
            normalize: raw.preprocessor.normalize,
            windowSize: raw.preprocessor.windowSize,
            windowStride: raw.preprocessor.windowStride,
            window: raw.preprocessor.window,
            features: raw.preprocessor.features,
            nFFT: raw.preprocessor.nFFT,
            dither: raw.preprocessor.dither,
            padTo: raw.preprocessor.padTo,
            padValue: raw.preprocessor.padValue,
            preemph: raw.preprocessor.preemph
        )

        let encoder = ParakeetEncoderConfig(
            featIn: raw.encoder.featIn,
            layers: raw.encoder.layers,
            modelDim: raw.encoder.modelDim,
            heads: raw.encoder.heads,
            ffExpansionFactor: raw.encoder.ffExpansionFactor,
            subsamplingFactor: raw.encoder.subsamplingFactor,
            selfAttentionModel: raw.encoder.selfAttentionModel,
            subsampling: raw.encoder.subsampling,
            convKernelSize: raw.encoder.convKernelSize,
            subsamplingConvChannels: raw.encoder.subsamplingConvChannels,
            posEmbMaxLen: raw.encoder.posEmbMaxLen,
            causalDownsampling: raw.encoder.causalDownsampling,
            useBias: raw.encoder.useBias,
            xScaling: raw.encoder.xScaling,
            subsamplingConvChunkingFactor: raw.encoder.subsamplingConvChunkingFactor
        )

        let rnntDecoder: ParakeetRNNTDecoderConfig?
        if let rawDecoder = raw.decoder, let prednet = rawDecoder.prednet {
            rnntDecoder = ParakeetRNNTDecoderConfig(
                blankAsPad: rawDecoder.blankAsPad,
                vocabSize: rawDecoder.vocabSize,
                prednet: ParakeetPredictNetConfig(
                    predHidden: prednet.predHidden,
                    predRnnLayers: prednet.predRnnLayers,
                    rnnHiddenSize: prednet.rnnHiddenSize
                )
            )
        } else {
            rnntDecoder = nil
        }

        let ctcDecoder: ParakeetCTCDecoderConfig?
        if let rawDecoder = raw.decoder, !rawDecoder.vocabulary.isEmpty {
            ctcDecoder = ParakeetCTCDecoderConfig(
                featIn: rawDecoder.featIn,
                numClasses: rawDecoder.numClasses,
                vocabulary: rawDecoder.vocabulary
            )
        } else if let aux = raw.auxCTC?.decoder {
            ctcDecoder = ParakeetCTCDecoderConfig(
                featIn: aux.featIn,
                numClasses: aux.numClasses,
                vocabulary: aux.vocabulary
            )
        } else {
            ctcDecoder = nil
        }

        let joint: ParakeetJointConfig?
        if let rawJoint = raw.joint {
            joint = ParakeetJointConfig(
                numClasses: rawJoint.numClasses,
                vocabulary: rawJoint.vocabulary,
                jointnet: ParakeetJointNetConfig(
                    jointHidden: rawJoint.jointnet.jointHidden,
                    activation: rawJoint.jointnet.activation,
                    encoderHidden: rawJoint.jointnet.encoderHidden,
                    predHidden: rawJoint.jointnet.predHidden
                ),
                numExtraOutputs: rawJoint.numExtraOutputs
            )
        } else {
            joint = nil
        }

        let quantization = raw.quantization ?? raw.quantizationConfig

        let supportedLanguageCodes = ParakeetModelConfig.extractSupportedLanguageCodes(
            vocabulary: joint?.vocabulary ?? ctcDecoder?.vocabulary ?? []
        )

        return ParakeetModelConfig(
            packaging: packaging,
            variant: variant,
            target: raw.target,
            preprocessor: preprocessor,
            encoder: encoder,
            rnntDecoder: rnntDecoder,
            ctcDecoder: ctcDecoder,
            joint: joint,
            tdtDurations: raw.modelDefaults?.tdtDurations,
            maxSymbols: raw.decoding?.greedy?.maxSymbols,
            quantizationBits: quantization?.bits,
            quantizationGroupSize: quantization?.groupSize,
            supportedLanguageCodes: supportedLanguageCodes
        )
    }

    private static func detectVariant(target: String, tdtDurations: [Int]?) -> ParakeetVariant {
        if target.contains("ctc_bpe_models.EncDecCTCModelBPE") {
            return .ctc
        }

        let hasTDT = (tdtDurations?.isEmpty == false)
        if target.contains("hybrid_rnnt_ctc") && hasTDT {
            return .tdtCTC
        }
        if target.contains("rnnt_bpe_models") && hasTDT {
            return .tdt
        }
        if target.contains("rnnt_bpe_models") {
            return .rnnt
        }

        return .rnnt
    }

    private static func extractSupportedLanguageCodes(vocabulary: [String]) -> Set<String> {
        guard !vocabulary.isEmpty else { return [] }
        var codes = Set<String>()
        codes.reserveCapacity(256)

        for token in vocabulary {
            guard token.hasPrefix("<|"), token.hasSuffix("|>") else { continue }
            let start = token.index(token.startIndex, offsetBy: 2)
            let end = token.index(token.endIndex, offsetBy: -2)
            let code = String(token[start..<end]).lowercased()
            guard code.count >= 2, code.count <= 3 else { continue }
            guard code.unicodeScalars.allSatisfy({ $0.properties.isAlphabetic }) else { continue }
            codes.insert(code)
        }

        return codes
    }
}

private struct RawConfig: Codable {
    let mere: RawMerePackaging?
    let target: String
    let modelDefaults: RawModelDefaults?
    let preprocessor: RawPreprocessor
    let encoder: RawEncoder
    let decoder: RawDecoder?
    let joint: RawJoint?
    let auxCTC: RawAuxCTC?
    let decoding: RawDecoding?
    let quantization: RawQuantization?
    let quantizationConfig: RawQuantization?

    enum CodingKeys: String, CodingKey {
        case mere
        case target
        case modelDefaults = "model_defaults"
        case preprocessor
        case encoder
        case decoder
        case joint
        case auxCTC = "aux_ctc"
        case decoding
        case quantization
        case quantizationConfig = "quantization_config"
    }
}

private struct RawMerePackaging: Codable {
    let format: String
}

public enum ParakeetConfigError: LocalizedError {
    case unsupportedPackaging(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedPackaging(let format):
            return "Unsupported Parakeet package format: \(format)."
        }
    }
}

private struct RawModelDefaults: Codable {
    let tdtDurations: [Int]?

    enum CodingKeys: String, CodingKey {
        case tdtDurations = "tdt_durations"
    }
}

private struct RawPreprocessor: Codable {
    let sampleRate: Int
    let normalize: String
    let windowSize: Double
    let windowStride: Double
    let window: String
    let features: Int
    let nFFT: Int
    let dither: Double
    let padTo: Int
    let padValue: Double
    let preemph: Double

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case normalize
        case windowSize = "window_size"
        case windowStride = "window_stride"
        case window
        case features
        case nFFT = "n_fft"
        case dither
        case padTo = "pad_to"
        case padValue = "pad_value"
        case preemph
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16_000
        normalize = try container.decodeIfPresent(String.self, forKey: .normalize) ?? "per_feature"
        windowSize = try container.decodeIfPresent(Double.self, forKey: .windowSize) ?? 0.025
        windowStride = try container.decodeIfPresent(Double.self, forKey: .windowStride) ?? 0.01
        window = try container.decodeIfPresent(String.self, forKey: .window) ?? "hann"
        features = try container.decodeIfPresent(Int.self, forKey: .features) ?? 128
        nFFT = try container.decodeIfPresent(Int.self, forKey: .nFFT) ?? 512
        dither = try container.decodeIfPresent(Double.self, forKey: .dither) ?? 0
        padTo = try container.decodeIfPresent(Int.self, forKey: .padTo) ?? 0
        padValue = try container.decodeIfPresent(Double.self, forKey: .padValue) ?? 0
        preemph = try container.decodeIfPresent(Double.self, forKey: .preemph) ?? 0.97
    }
}

private struct RawEncoder: Codable {
    let featIn: Int
    let layers: Int
    let modelDim: Int
    let heads: Int
    let ffExpansionFactor: Int
    let subsamplingFactor: Int
    let selfAttentionModel: String
    let subsampling: String
    let convKernelSize: Int
    let subsamplingConvChannels: Int
    let posEmbMaxLen: Int
    let causalDownsampling: Bool
    let useBias: Bool
    let xScaling: Bool
    let subsamplingConvChunkingFactor: Int

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case layers = "n_layers"
        case modelDim = "d_model"
        case heads = "n_heads"
        case ffExpansionFactor = "ff_expansion_factor"
        case subsamplingFactor = "subsampling_factor"
        case selfAttentionModel = "self_attention_model"
        case subsampling
        case convKernelSize = "conv_kernel_size"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case posEmbMaxLen = "pos_emb_max_len"
        case causalDownsampling = "causal_downsampling"
        case useBias = "use_bias"
        case xScaling = "xscaling"
        case subsamplingConvChunkingFactor = "subsampling_conv_chunking_factor"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        featIn = try container.decodeIfPresent(Int.self, forKey: .featIn) ?? 128
        layers = try container.decodeIfPresent(Int.self, forKey: .layers) ?? 24
        modelDim = try container.decodeIfPresent(Int.self, forKey: .modelDim) ?? 1024
        heads = try container.decodeIfPresent(Int.self, forKey: .heads) ?? 8
        ffExpansionFactor = try container.decodeIfPresent(Int.self, forKey: .ffExpansionFactor) ?? 4
        subsamplingFactor = try container.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8
        selfAttentionModel = try container.decodeIfPresent(String.self, forKey: .selfAttentionModel) ?? "rel_pos"
        subsampling = try container.decodeIfPresent(String.self, forKey: .subsampling) ?? "dw_striding"
        convKernelSize = try container.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 9
        subsamplingConvChannels = try container.decodeIfPresent(Int.self, forKey: .subsamplingConvChannels) ?? 256
        posEmbMaxLen = try container.decodeIfPresent(Int.self, forKey: .posEmbMaxLen) ?? 5000
        causalDownsampling = try container.decodeIfPresent(Bool.self, forKey: .causalDownsampling) ?? false
        useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        xScaling = try container.decodeIfPresent(Bool.self, forKey: .xScaling) ?? false
        subsamplingConvChunkingFactor = try container.decodeIfPresent(Int.self, forKey: .subsamplingConvChunkingFactor) ?? 1
    }
}

private struct RawDecoder: Codable {
    let blankAsPad: Bool
    let vocabSize: Int
    let prednet: RawPrednet?
    let featIn: Int
    let numClasses: Int
    let vocabulary: [String]

    enum CodingKeys: String, CodingKey {
        case blankAsPad = "blank_as_pad"
        case vocabSize = "vocab_size"
        case prednet
        case featIn = "feat_in"
        case numClasses = "num_classes"
        case vocabulary
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        blankAsPad = try container.decodeIfPresent(Bool.self, forKey: .blankAsPad) ?? true
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 0
        prednet = try container.decodeIfPresent(RawPrednet.self, forKey: .prednet)
        featIn = try container.decodeIfPresent(Int.self, forKey: .featIn) ?? 0
        numClasses = try container.decodeIfPresent(Int.self, forKey: .numClasses) ?? 0
        vocabulary = try container.decodeIfPresent([String].self, forKey: .vocabulary) ?? []
    }
}

private struct RawPrednet: Codable {
    let predHidden: Int
    let predRnnLayers: Int
    let rnnHiddenSize: Int?

    enum CodingKeys: String, CodingKey {
        case predHidden = "pred_hidden"
        case predRnnLayers = "pred_rnn_layers"
        case rnnHiddenSize = "rnn_hidden_size"
    }
}

private struct RawJoint: Codable {
    let numClasses: Int
    let vocabulary: [String]
    let jointnet: RawJointNet
    let numExtraOutputs: Int

    enum CodingKeys: String, CodingKey {
        case numClasses = "num_classes"
        case vocabulary
        case jointnet
        case numExtraOutputs = "num_extra_outputs"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        numClasses = try container.decodeIfPresent(Int.self, forKey: .numClasses) ?? 0
        vocabulary = try container.decodeIfPresent([String].self, forKey: .vocabulary) ?? []
        jointnet = try container.decodeIfPresent(RawJointNet.self, forKey: .jointnet) ?? RawJointNet(
            jointHidden: 640,
            activation: "relu",
            encoderHidden: 1024,
            predHidden: 640
        )
        numExtraOutputs = try container.decodeIfPresent(Int.self, forKey: .numExtraOutputs) ?? 0
    }
}

private struct RawJointNet: Codable {
    let jointHidden: Int
    let activation: String
    let encoderHidden: Int
    let predHidden: Int

    enum CodingKeys: String, CodingKey {
        case jointHidden = "joint_hidden"
        case activation
        case encoderHidden = "encoder_hidden"
        case predHidden = "pred_hidden"
    }
}

private struct RawAuxCTC: Codable {
    let decoder: RawCTCDecoder
}

private struct RawCTCDecoder: Codable {
    let featIn: Int
    let numClasses: Int
    let vocabulary: [String]

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case numClasses = "num_classes"
        case vocabulary
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        featIn = try container.decodeIfPresent(Int.self, forKey: .featIn) ?? 0
        numClasses = try container.decodeIfPresent(Int.self, forKey: .numClasses) ?? 0
        vocabulary = try container.decodeIfPresent([String].self, forKey: .vocabulary) ?? []
    }
}

private struct RawDecoding: Codable {
    let greedy: RawGreedy?
}

private struct RawGreedy: Codable {
    let maxSymbols: Int?

    enum CodingKeys: String, CodingKey {
        case maxSymbols = "max_symbols"
    }
}

private struct RawQuantization: Codable {
    let bits: Int?
    let groupSize: Int?

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }
}
