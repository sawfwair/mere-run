import Foundation

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
