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

public struct Qwen3TTSTalkerCodePredictorConfig: Decodable, Sendable, Hashable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let hiddenAct: String
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let ropeScaling: Qwen3TTSRopeScalingConfig?
    public let attentionBias: Bool
    public let slidingWindow: Int?
    public let layerTypes: [String]?
    public let attentionDropout: Float
    public let numCodeGroups: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case attentionDropout = "attention_dropout"
        case numCodeGroups = "num_code_groups"
    }

    public init(
        vocabSize: Int = 2048,
        hiddenSize: Int = 1024,
        intermediateSize: Int = 3072,
        numHiddenLayers: Int = 5,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 65536,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        ropeScaling: Qwen3TTSRopeScalingConfig? = nil,
        attentionBias: Bool = false,
        slidingWindow: Int? = nil,
        layerTypes: [String]? = nil,
        attentionDropout: Float = 0.0,
        numCodeGroups: Int = 16
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.attentionBias = attentionBias
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes
        self.attentionDropout = attentionDropout
        self.numCodeGroups = numCodeGroups
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 2048
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 5
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 65536
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        ropeScaling = try container.decodeIfPresent(Qwen3TTSRopeScalingConfig.self, forKey: .ropeScaling)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        numCodeGroups = try container.decodeIfPresent(Int.self, forKey: .numCodeGroups) ?? 16
    }
}

public struct Qwen3TTSTalkerConfig: Decodable, Sendable, Hashable {
    public let codePredictorConfig: Qwen3TTSTalkerCodePredictorConfig
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let hiddenAct: String
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let ropeScaling: Qwen3TTSRopeScalingConfig?
    public let attentionBias: Bool
    public let slidingWindow: Int?
    public let attentionDropout: Float
    public let numCodeGroups: Int
    public let textHiddenSize: Int
    public let textVocabSize: Int
    public let codecEosTokenId: Int
    public let codecThinkId: Int
    public let codecNoThinkId: Int
    public let codecThinkBosId: Int
    public let codecThinkEosId: Int
    public let codecPadId: Int
    public let codecBosId: Int
    public let codecLanguageId: [String: Int]?
    public let spkId: [String: [Int]]?
    public let spkIsDialect: [String: String]?

    private enum SpeakerIDValue: Decodable {
        case single(Int)
        case multiple([Int])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let values = try? container.decode([Int].self) {
                self = .multiple(values)
                return
            }
            if let value = try? container.decode(Int.self) {
                self = .single(value)
                return
            }
            if let value = try? container.decode(Double.self) {
                self = .single(Int(value))
                return
            }
            throw DecodingError.typeMismatch(
                SpeakerIDValue.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected speaker id as Int or [Int]."
                )
            )
        }

        var ids: [Int] {
            switch self {
            case .single(let value):
                return [value]
            case .multiple(let values):
                return values
            }
        }
    }

    private enum SpeakerDialectValue: Decodable {
        case string(String)
        case bool(Bool)
        case int(Int)
        case double(Double)

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let value = try? container.decode(String.self) {
                self = .string(value)
                return
            }
            if let value = try? container.decode(Bool.self) {
                self = .bool(value)
                return
            }
            if let value = try? container.decode(Int.self) {
                self = .int(value)
                return
            }
            if let value = try? container.decode(Double.self) {
                self = .double(value)
                return
            }
            throw DecodingError.typeMismatch(
                SpeakerDialectValue.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected dialect flag as String, Bool, Int, or Double."
                )
            )
        }

        var normalized: String {
            switch self {
            case .string(let value):
                return value
            case .bool(let value):
                return value ? "true" : "false"
            case .int(let value):
                return String(value)
            case .double(let value):
                return String(value)
            }
        }
    }

    enum CodingKeys: String, CodingKey {
        case codePredictorConfig = "code_predictor_config"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case slidingWindow = "sliding_window"
        case attentionDropout = "attention_dropout"
        case numCodeGroups = "num_code_groups"
        case textHiddenSize = "text_hidden_size"
        case textVocabSize = "text_vocab_size"
        case codecEosTokenId = "codec_eos_token_id"
        case codecThinkId = "codec_think_id"
        case codecNoThinkId = "codec_nothink_id"
        case codecThinkBosId = "codec_think_bos_id"
        case codecThinkEosId = "codec_think_eos_id"
        case codecPadId = "codec_pad_id"
        case codecBosId = "codec_bos_id"
        case codecLanguageId = "codec_language_id"
        case spkId = "spk_id"
        case spkIsDialect = "spk_is_dialect"
    }

    public init(
        codePredictorConfig: Qwen3TTSTalkerCodePredictorConfig = Qwen3TTSTalkerCodePredictorConfig(),
        vocabSize: Int = 3072,
        hiddenSize: Int = 1024,
        intermediateSize: Int = 3072,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 32768,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        ropeScaling: Qwen3TTSRopeScalingConfig? = Qwen3TTSRopeScalingConfig(),
        attentionBias: Bool = false,
        slidingWindow: Int? = nil,
        attentionDropout: Float = 0.0,
        numCodeGroups: Int = 16,
        textHiddenSize: Int = 2048,
        textVocabSize: Int = 151_936,
        codecEosTokenId: Int = 2150,
        codecThinkId: Int = 2154,
        codecNoThinkId: Int = 2155,
        codecThinkBosId: Int = 2156,
        codecThinkEosId: Int = 2157,
        codecPadId: Int = 2148,
        codecBosId: Int = 2149,
        codecLanguageId: [String: Int]? = nil,
        spkId: [String: [Int]]? = nil,
        spkIsDialect: [String: String]? = nil
    ) {
        self.codePredictorConfig = codePredictorConfig
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.attentionBias = attentionBias
        self.slidingWindow = slidingWindow
        self.attentionDropout = attentionDropout
        self.numCodeGroups = numCodeGroups
        self.textHiddenSize = textHiddenSize
        self.textVocabSize = textVocabSize
        self.codecEosTokenId = codecEosTokenId
        self.codecThinkId = codecThinkId
        self.codecNoThinkId = codecNoThinkId
        self.codecThinkBosId = codecThinkBosId
        self.codecThinkEosId = codecThinkEosId
        self.codecPadId = codecPadId
        self.codecBosId = codecBosId
        self.codecLanguageId = codecLanguageId
        self.spkId = spkId
        self.spkIsDialect = spkIsDialect
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        codePredictorConfig = try container.decodeIfPresent(Qwen3TTSTalkerCodePredictorConfig.self, forKey: .codePredictorConfig)
            ?? Qwen3TTSTalkerCodePredictorConfig()
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 3072
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        ropeScaling = try container.decodeIfPresent(Qwen3TTSRopeScalingConfig.self, forKey: .ropeScaling)
            ?? Qwen3TTSRopeScalingConfig()
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        numCodeGroups = try container.decodeIfPresent(Int.self, forKey: .numCodeGroups) ?? 16
        textHiddenSize = try container.decodeIfPresent(Int.self, forKey: .textHiddenSize) ?? 2048
        textVocabSize = try container.decodeIfPresent(Int.self, forKey: .textVocabSize) ?? 151_936
        codecEosTokenId = try container.decodeIfPresent(Int.self, forKey: .codecEosTokenId) ?? 2150
        codecThinkId = try container.decodeIfPresent(Int.self, forKey: .codecThinkId) ?? 2154
        codecNoThinkId = try container.decodeIfPresent(Int.self, forKey: .codecNoThinkId) ?? 2155
        codecThinkBosId = try container.decodeIfPresent(Int.self, forKey: .codecThinkBosId) ?? 2156
        codecThinkEosId = try container.decodeIfPresent(Int.self, forKey: .codecThinkEosId) ?? 2157
        codecPadId = try container.decodeIfPresent(Int.self, forKey: .codecPadId) ?? 2148
        codecBosId = try container.decodeIfPresent(Int.self, forKey: .codecBosId) ?? 2149
        codecLanguageId = try container.decodeIfPresent([String: Int].self, forKey: .codecLanguageId)
        if let decoded = try container.decodeIfPresent([String: SpeakerIDValue].self, forKey: .spkId) {
            spkId = Dictionary(uniqueKeysWithValues: decoded.map { ($0.key, $0.value.ids) })
        } else {
            spkId = nil
        }
        if let decoded = try container.decodeIfPresent([String: SpeakerDialectValue].self, forKey: .spkIsDialect) {
            spkIsDialect = Dictionary(uniqueKeysWithValues: decoded.map { ($0.key, $0.value.normalized) })
        } else {
            spkIsDialect = nil
        }
    }
}

public struct Qwen3TTSTokenizerDecoderConfig: Decodable, Sendable, Hashable {
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let latentDim: Int
    public let codebookDim: Int
    public let codebookSize: Int
    public let decoderDim: Int
    public let hiddenAct: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let layerScaleInitialScale: Float
    public let maxPositionEmbeddings: Int
    public let headDim: Int
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let numQuantizers: Int
    public let numSemanticQuantizers: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let semanticCodebookSize: Int
    public let slidingWindow: Int
    public let upsampleRates: [Int]
    public let upsamplingRatios: [Int]
    public let vectorQuantizationHiddenDimension: Int

    enum CodingKeys: String, CodingKey {
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case latentDim = "latent_dim"
        case codebookDim = "codebook_dim"
        case codebookSize = "codebook_size"
        case decoderDim = "decoder_dim"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case layerScaleInitialScale = "layer_scale_initial_scale"
        case maxPositionEmbeddings = "max_position_embeddings"
        case headDim = "head_dim"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case numQuantizers = "num_quantizers"
        case numSemanticQuantizers = "num_semantic_quantizers"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case semanticCodebookSize = "semantic_codebook_size"
        case slidingWindow = "sliding_window"
        case upsampleRates = "upsample_rates"
        case upsamplingRatios = "upsampling_ratios"
        case vectorQuantizationHiddenDimension = "vector_quantization_hidden_dimension"
    }

    public init(
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        latentDim: Int = 1024,
        codebookDim: Int = 512,
        codebookSize: Int = 2048,
        decoderDim: Int = 1536,
        hiddenAct: String = "silu",
        hiddenSize: Int = 512,
        intermediateSize: Int = 1024,
        layerScaleInitialScale: Float = 0.01,
        maxPositionEmbeddings: Int = 8000,
        headDim: Int = 64,
        numAttentionHeads: Int = 16,
        numHiddenLayers: Int = 8,
        numKeyValueHeads: Int = 16,
        numQuantizers: Int = 16,
        numSemanticQuantizers: Int = 1,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10000.0,
        semanticCodebookSize: Int = 4096,
        slidingWindow: Int = 72,
        upsampleRates: [Int] = [8, 5, 4, 3],
        upsamplingRatios: [Int] = [2, 2],
        vectorQuantizationHiddenDimension: Int = 512
    ) {
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.latentDim = latentDim
        self.codebookDim = codebookDim
        self.codebookSize = codebookSize
        self.decoderDim = decoderDim
        self.hiddenAct = hiddenAct
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.layerScaleInitialScale = layerScaleInitialScale
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.headDim = headDim
        self.numAttentionHeads = numAttentionHeads
        self.numHiddenLayers = numHiddenLayers
        self.numKeyValueHeads = numKeyValueHeads
        self.numQuantizers = numQuantizers
        self.numSemanticQuantizers = numSemanticQuantizers
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.semanticCodebookSize = semanticCodebookSize
        self.slidingWindow = slidingWindow
        self.upsampleRates = upsampleRates
        self.upsamplingRatios = upsamplingRatios
        self.vectorQuantizationHiddenDimension = vectorQuantizationHiddenDimension
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        latentDim = try container.decodeIfPresent(Int.self, forKey: .latentDim) ?? 1024
        codebookDim = try container.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 512
        codebookSize = try container.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 2048
        decoderDim = try container.decodeIfPresent(Int.self, forKey: .decoderDim) ?? 1536
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 1024
        layerScaleInitialScale = try container.decodeIfPresent(Float.self, forKey: .layerScaleInitialScale) ?? 0.01
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8000
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 8
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 16
        numQuantizers = try container.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? 16
        numSemanticQuantizers = try container.decodeIfPresent(Int.self, forKey: .numSemanticQuantizers) ?? 1
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        semanticCodebookSize = try container.decodeIfPresent(Int.self, forKey: .semanticCodebookSize) ?? 4096
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 72
        upsampleRates = try container.decodeIfPresent([Int].self, forKey: .upsampleRates) ?? [8, 5, 4, 3]
        upsamplingRatios = try container.decodeIfPresent([Int].self, forKey: .upsamplingRatios) ?? [2, 2]
        vectorQuantizationHiddenDimension = try container.decodeIfPresent(Int.self, forKey: .vectorQuantizationHiddenDimension) ?? 512
    }
}

public struct Qwen3TTSTokenizerEncoderConfig: Decodable, Sendable, Hashable {
    public let frameRate: Float
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let audioChannels: Int
    public let codebookDim: Int
    public let codebookSize: Int
    public let compress: Int
    public let dilationGrowthRate: Int
    public let headDim: Int
    public let hiddenAct: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let kernelSize: Int
    public let lastKernelSize: Int
    public let layerScaleInitialScale: Float
    public let maxPositionEmbeddings: Int
    public let normEps: Float
    public let numAttentionHeads: Int
    public let numFilters: Int
    public let numHiddenLayers: Int
    public let numKeyValueHeads: Int
    public let numQuantizers: Int
    public let numResidualLayers: Int
    public let numSemanticQuantizers: Int
    public let residualKernelSize: Int
    public let ropeTheta: Float
    public let samplingRate: Int
    public let slidingWindow: Int
    public let upsamplingRatios: [Int]
    public let useCausalConv: Bool
    public let useConvShortcut: Bool
    public let vectorQuantizationHiddenDimension: Int

    enum CodingKeys: String, CodingKey {
        case frameRate = "frame_rate"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case audioChannels = "audio_channels"
        case codebookDim = "codebook_dim"
        case codebookSize = "codebook_size"
        case compress
        case dilationGrowthRate = "dilation_growth_rate"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case kernelSize = "kernel_size"
        case lastKernelSize = "last_kernel_size"
        case layerScaleInitialScale = "layer_scale_initial_scale"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case numAttentionHeads = "num_attention_heads"
        case numFilters = "num_filters"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case numQuantizers = "num_quantizers"
        case numResidualLayers = "num_residual_layers"
        case numSemanticQuantizers = "num_semantic_quantizers"
        case residualKernelSize = "residual_kernel_size"
        case ropeTheta = "rope_theta"
        case samplingRate = "sampling_rate"
        case slidingWindow = "sliding_window"
        case upsamplingRatios = "upsampling_ratios"
        case useCausalConv = "use_causal_conv"
        case useConvShortcut = "use_conv_shortcut"
        case vectorQuantizationHiddenDimension = "vector_quantization_hidden_dimension"
    }

    public init(
        frameRate: Float = 12.5,
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        audioChannels: Int = 1,
        codebookDim: Int = 256,
        codebookSize: Int = 2048,
        compress: Int = 2,
        dilationGrowthRate: Int = 2,
        headDim: Int = 64,
        hiddenAct: String = "gelu",
        hiddenSize: Int = 512,
        intermediateSize: Int = 2048,
        kernelSize: Int = 7,
        lastKernelSize: Int = 3,
        layerScaleInitialScale: Float = 0.01,
        maxPositionEmbeddings: Int = 8000,
        normEps: Float = 1e-5,
        numAttentionHeads: Int = 8,
        numFilters: Int = 64,
        numHiddenLayers: Int = 8,
        numKeyValueHeads: Int = 8,
        numQuantizers: Int = 32,
        numResidualLayers: Int = 1,
        numSemanticQuantizers: Int = 1,
        residualKernelSize: Int = 3,
        ropeTheta: Float = 10000.0,
        samplingRate: Int = 24000,
        slidingWindow: Int = 250,
        upsamplingRatios: [Int] = [8, 6, 5, 4],
        useCausalConv: Bool = true,
        useConvShortcut: Bool = false,
        vectorQuantizationHiddenDimension: Int = 256
    ) {
        self.frameRate = frameRate
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.audioChannels = audioChannels
        self.codebookDim = codebookDim
        self.codebookSize = codebookSize
        self.compress = compress
        self.dilationGrowthRate = dilationGrowthRate
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.kernelSize = kernelSize
        self.lastKernelSize = lastKernelSize
        self.layerScaleInitialScale = layerScaleInitialScale
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.normEps = normEps
        self.numAttentionHeads = numAttentionHeads
        self.numFilters = numFilters
        self.numHiddenLayers = numHiddenLayers
        self.numKeyValueHeads = numKeyValueHeads
        self.numQuantizers = numQuantizers
        self.numResidualLayers = numResidualLayers
        self.numSemanticQuantizers = numSemanticQuantizers
        self.residualKernelSize = residualKernelSize
        self.ropeTheta = ropeTheta
        self.samplingRate = samplingRate
        self.slidingWindow = slidingWindow
        self.upsamplingRatios = upsamplingRatios
        self.useCausalConv = useCausalConv
        self.useConvShortcut = useConvShortcut
        self.vectorQuantizationHiddenDimension = vectorQuantizationHiddenDimension
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        frameRate = try container.decodeIfPresent(Float.self, forKey: .frameRate) ?? 12.5
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        audioChannels = try container.decodeIfPresent(Int.self, forKey: .audioChannels) ?? 1
        codebookDim = try container.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 256
        codebookSize = try container.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 2048
        compress = try container.decodeIfPresent(Int.self, forKey: .compress) ?? 2
        dilationGrowthRate = try container.decodeIfPresent(Int.self, forKey: .dilationGrowthRate) ?? 2
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2048
        kernelSize = try container.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 7
        lastKernelSize = try container.decodeIfPresent(Int.self, forKey: .lastKernelSize) ?? 3
        layerScaleInitialScale = try container.decodeIfPresent(Float.self, forKey: .layerScaleInitialScale) ?? 0.01
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8000
        normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        numFilters = try container.decodeIfPresent(Int.self, forKey: .numFilters) ?? 64
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 8
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        numQuantizers = try container.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? 32
        numResidualLayers = try container.decodeIfPresent(Int.self, forKey: .numResidualLayers) ?? 1
        numSemanticQuantizers = try container.decodeIfPresent(Int.self, forKey: .numSemanticQuantizers) ?? 1
        residualKernelSize = try container.decodeIfPresent(Int.self, forKey: .residualKernelSize) ?? 3
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        samplingRate = try container.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 250
        upsamplingRatios = try container.decodeIfPresent([Int].self, forKey: .upsamplingRatios) ?? [8, 6, 5, 4]
        useCausalConv = try container.decodeIfPresent(Bool.self, forKey: .useCausalConv) ?? true
        useConvShortcut = try container.decodeIfPresent(Bool.self, forKey: .useConvShortcut) ?? false
        vectorQuantizationHiddenDimension = try container.decodeIfPresent(Int.self, forKey: .vectorQuantizationHiddenDimension) ?? 256
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
