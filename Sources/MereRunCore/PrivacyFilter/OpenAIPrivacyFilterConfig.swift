import Foundation

public struct OpenAIPrivacyFilterConfig: Decodable, Hashable, Sendable {
    public struct RopeParameters: Decodable, Hashable, Sendable {
        public var ropeType: String
        public var ropeTheta: Float
        public var factor: Float
        public var betaFast: Float
        public var betaSlow: Float
        public var originalMaxPositionEmbeddings: Int

        enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case ropeTheta = "rope_theta"
            case factor
            case betaFast = "beta_fast"
            case betaSlow = "beta_slow"
            case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        }

        public init(
            ropeType: String = "yarn",
            ropeTheta: Float = 150_000.0,
            factor: Float = 32.0,
            betaFast: Float = 32.0,
            betaSlow: Float = 1.0,
            originalMaxPositionEmbeddings: Int = 4096
        ) {
            self.ropeType = ropeType
            self.ropeTheta = ropeTheta
            self.factor = factor
            self.betaFast = betaFast
            self.betaSlow = betaSlow
            self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        }
    }

    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var slidingWindow: Int
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var attentionBias: Bool
    public var classifierDropout: Float
    public var numLocalExperts: Int
    public var numExpertsPerTok: Int
    public var padTokenID: Int?
    public var eosTokenID: Int?
    public var defaultContextLength: Int?
    public var ropeParameters: RopeParameters
    public var id2label: [Int: String]
    public var label2id: [String: Int]

    public var numLabels: Int {
        id2label.isEmpty ? 33 : id2label.count
    }

    var orderedLabels: [String] {
        let maximumLabelID = id2label.keys.max() ?? -1
        guard maximumLabelID >= 0 else {
            return []
        }
        return (0...maximumLabelID).map { id2label[$0] ?? "" }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case slidingWindow = "sliding_window"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case classifierDropout = "classifier_dropout"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case padTokenID = "pad_token_id"
        case eosTokenID = "eos_token_id"
        case defaultContextLength = "default_n_ctx"
        case ropeParameters = "rope_parameters"
        case id2label
        case label2id
    }

    public init(
        modelType: String = "openai_privacy_filter",
        vocabSize: Int = 200_064,
        hiddenSize: Int = 640,
        intermediateSize: Int = 640,
        numHiddenLayers: Int = 8,
        numAttentionHeads: Int = 14,
        numKeyValueHeads: Int = 2,
        headDim: Int = 64,
        slidingWindow: Int = 128,
        maxPositionEmbeddings: Int = 131_072,
        rmsNormEps: Float = 1e-5,
        attentionBias: Bool = true,
        classifierDropout: Float = 0.0,
        numLocalExperts: Int = 128,
        numExpertsPerTok: Int = 4,
        padTokenID: Int? = 199_999,
        eosTokenID: Int? = 199_999,
        defaultContextLength: Int? = 128_000,
        ropeParameters: RopeParameters = RopeParameters(),
        id2label: [Int: String] = OpenAIPrivacyFilterConfig.defaultIDToLabel,
        label2id: [String: Int] = OpenAIPrivacyFilterConfig.defaultLabelToID
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.slidingWindow = slidingWindow
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.attentionBias = attentionBias
        self.classifierDropout = classifierDropout
        self.numLocalExperts = numLocalExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.padTokenID = padTokenID
        self.eosTokenID = eosTokenID
        self.defaultContextLength = defaultContextLength
        self.ropeParameters = ropeParameters
        self.id2label = id2label
        self.label2id = label2id
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "openai_privacy_filter"
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 200_064
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 640
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 640
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 8
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 14
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 128
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
        self.classifierDropout = try container.decodeIfPresent(Float.self, forKey: .classifierDropout) ?? 0.0
        self.numLocalExperts = try container.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? 128
        self.numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 4
        self.padTokenID = try container.decodeIfPresent(Int.self, forKey: .padTokenID)
        self.eosTokenID = try container.decodeIfPresent(Int.self, forKey: .eosTokenID)
        self.defaultContextLength = try container.decodeIfPresent(Int.self, forKey: .defaultContextLength)
        self.ropeParameters = try container.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters) ?? RopeParameters()

        let rawIDToLabel = try container.decodeIfPresent([String: String].self, forKey: .id2label) ?? [:]
        self.id2label = Dictionary(uniqueKeysWithValues: rawIDToLabel.compactMap { key, value in
            guard let id = Int(key) else { return nil }
            return (id, value)
        })
        if self.id2label.isEmpty {
            self.id2label = Self.defaultIDToLabel
        }

        let rawLabelToID = try container.decodeIfPresent([String: Int].self, forKey: .label2id) ?? [:]
        self.label2id = rawLabelToID.isEmpty ? Self.defaultLabelToID : rawLabelToID
    }

    public static let defaultIDToLabel: [Int: String] = [
        0: "O",
        1: "B-account_number", 2: "I-account_number", 3: "E-account_number", 4: "S-account_number",
        5: "B-private_address", 6: "I-private_address", 7: "E-private_address", 8: "S-private_address",
        9: "B-private_date", 10: "I-private_date", 11: "E-private_date", 12: "S-private_date",
        13: "B-private_email", 14: "I-private_email", 15: "E-private_email", 16: "S-private_email",
        17: "B-private_person", 18: "I-private_person", 19: "E-private_person", 20: "S-private_person",
        21: "B-private_phone", 22: "I-private_phone", 23: "E-private_phone", 24: "S-private_phone",
        25: "B-private_url", 26: "I-private_url", 27: "E-private_url", 28: "S-private_url",
        29: "B-secret", 30: "I-secret", 31: "E-secret", 32: "S-secret",
    ]

    public static let defaultLabelToID: [String: Int] = Dictionary(
        uniqueKeysWithValues: defaultIDToLabel.map { ($0.value, $0.key) }
    )
}
