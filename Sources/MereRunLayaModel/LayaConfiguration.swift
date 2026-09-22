import Foundation

public struct LayaEncoderConfiguration: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var globalAttnEveryNLayers: Int
    public var localAttention: Int
    public var normEps: Float
    public var normBias: Bool
    public var attentionBias: Bool
    public var mlpBias: Bool
    public var hiddenActivation: String
    public var padTokenID: Int
    public var ropeParameters: [String: RotaryParameters]
    public var layerTypes: [String]

    public struct RotaryParameters: Codable, Sendable {
        public var ropeTheta: Float
        public var ropeType: String
        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta", ropeType = "rope_type"
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", hiddenSize = "hidden_size", intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers", numAttentionHeads = "num_attention_heads"
        case vocabSize = "vocab_size", maxPositionEmbeddings = "max_position_embeddings"
        case globalAttnEveryNLayers = "global_attn_every_n_layers", localAttention = "local_attention"
        case normEps = "norm_eps", normBias = "norm_bias", attentionBias = "attention_bias"
        case mlpBias = "mlp_bias", hiddenActivation = "hidden_activation", padTokenID = "pad_token_id"
        case ropeParameters = "rope_parameters", layerTypes = "layer_types"
    }

    public func validate() throws {
        guard modelType == "modernbert", hiddenActivation == "gelu",
              hiddenSize > 0, hiddenSize <= 4_096, numAttentionHeads > 0,
              hiddenSize % numAttentionHeads == 0, (hiddenSize / numAttentionHeads).isMultiple(of: 2),
              intermediateSize > 0, intermediateSize <= 16_384,
              (1...64).contains(numHiddenLayers), (1...512_000).contains(vocabSize),
              (1...8_192).contains(maxPositionEmbeddings), globalAttnEveryNLayers > 0,
              localAttention > 0, normEps.isFinite, normEps > 0,
              (0..<vocabSize).contains(padTokenID), layerTypes.count == numHiddenLayers else {
            throw LayaModelError.invalidConfiguration("Unsupported ModernBERT dimensions or architecture.")
        }
        for (index, kind) in layerTypes.enumerated() {
            let expected = index.isMultiple(of: globalAttnEveryNLayers) ? "full_attention" : "sliding_attention"
            guard kind == expected, let rotary = ropeParameters[kind], rotary.ropeType == "default",
                  rotary.ropeTheta.isFinite, rotary.ropeTheta > 0 else {
                throw LayaModelError.invalidConfiguration("Unsupported attention or RoPE configuration at layer \(index).")
            }
        }
    }
}

public struct LayaAgentConfiguration: Codable, Sendable {
    public var encoder: String
    public var headLayers: Int
    public var maxLength: Int
    public var headMaxLength: Int
    public var actCosts: [String: Double]
    public var temperature: [Double]
    public var temperatureByOptions: [String: Double]

    enum CodingKeys: String, CodingKey {
        case encoder, temperature
        case headLayers = "head_layers", maxLength = "max_len", headMaxLength = "head_max_len"
        case actCosts = "act_costs", temperatureByOptions = "temperature_by_options"
    }

    public func validate(encoder configuration: LayaEncoderConfiguration) throws {
        try configuration.validate()
        guard !encoder.isEmpty, (0...8).contains(headLayers),
              maxLength >= 16, maxLength <= configuration.maxPositionEmbeddings,
              headMaxLength >= 8, headMaxLength < maxLength,
              actCosts.count <= 32, actCosts.values.allSatisfy({ $0.isFinite && $0 >= 0 }),
              temperature.count == 3, temperature.allSatisfy(\.isFinite),
              temperatureByOptions.values.allSatisfy(\.isFinite),
              configuration.hiddenSize % max(1, configuration.hiddenSize / 64) == 0 else {
            throw LayaModelError.invalidConfiguration("Invalid Laya decision head, token budget, or calibration.")
        }
    }

    public func temperature(type: Int, optionCount: Int) -> (raw: Double, applied: Double) {
        let bucket = optionCount <= 2 ? "2" : optionCount <= 5 ? "3-5" : optionCount <= 10 ? "6-10" : "11+"
        let name = ["choice", "score", "noul"][type]
        let raw = temperatureByOptions["\(name):\(bucket)"] ?? temperature[type]
        return (raw, min(5, max(0.5, raw)))
    }
}

public enum LayaModelError: LocalizedError, Sendable {
    case invalidConfiguration(String)
    case invalidWeights(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message), .invalidWeights(let message), .invalidInput(let message):
            return message
        }
    }
}
