import Foundation

public struct LFM2QuantizationConfig: Codable, Sendable, Hashable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String

    private enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }
}

public struct LFM2RopeParameters: Codable, Sendable, Hashable {
    public let ropeTheta: Float?
    public let ropeType: String?

    private enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
    }
}

public struct LFM2Config: Decodable, Sendable, Hashable {
    public let architectures: [String]
    public let bosTokenId: Int?
    public let eosTokenIds: [Int]
    public let padTokenId: Int?
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let numHiddenLayers: Int
    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let normTopKProb: Bool
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let maxPositionEmbeddings: Int
    public let useExpertBias: Bool
    public let numDenseLayers: Int
    public let normEps: Float
    public let convBias: Bool
    public let convLCache: Int
    public let ropeTheta: Float
    public let ropeParameters: LFM2RopeParameters?
    public let layerTypes: [String]
    public let quantization: LFM2QuantizationConfig?
    public let tieWordEmbeddings: Bool

    public var headDim: Int {
        hiddenSize / max(1, numAttentionHeads)
    }

    public var fullAttentionLayerIndexes: Set<Int> {
        Set(layerTypes.enumerated().compactMap { index, layerType in
            layerType == "full_attention" ? index : nil
        })
    }

    private enum CodingKeys: String, CodingKey {
        case architectures
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case blockAutoAdjustFFDimension = "block_auto_adjust_ff_dim"
        case blockFFDimensionMultiplier = "block_ffn_dim_multiplier"
        case blockMultipleOf = "block_multiple_of"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case normTopKProb = "norm_topk_prob"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case useExpertBias = "use_expert_bias"
        case numDenseLayers = "num_dense_layers"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case layerTypes = "layer_types"
        case quantization
        case quantizationConfig = "quantization_config"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        self.bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId)
        self.padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        let configuredIntermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        if try container.decodeIfPresent(Bool.self, forKey: .blockAutoAdjustFFDimension) == true {
            var adjusted = 2 * configuredIntermediateSize / 3
            if let multiplier = try container.decodeIfPresent(
                Double.self,
                forKey: .blockFFDimensionMultiplier
            ) {
                adjusted = Int(multiplier * Double(adjusted))
                let multiple = try container.decodeIfPresent(Int.self, forKey: .blockMultipleOf) ?? 1
                adjusted = multiple * ((adjusted + multiple - 1) / multiple)
            }
            self.intermediateSize = adjusted
        } else {
            self.intermediateSize = configuredIntermediateSize
        }
        self.moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
            ?? intermediateSize
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 1
        self.normTopKProb = try container.decodeIfPresent(Bool.self, forKey: .normTopKProb) ?? true
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.useExpertBias = try container.decodeIfPresent(Bool.self, forKey: .useExpertBias) ?? false
        self.numDenseLayers = try container.decodeIfPresent(Int.self, forKey: .numDenseLayers)
            ?? (modelType == "lfm2" ? numHiddenLayers : 0)
        self.normEps = try container.decode(Float.self, forKey: .normEps)
        self.convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        self.convLCache = try container.decode(Int.self, forKey: .convLCache)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.ropeParameters = try container.decodeIfPresent(LFM2RopeParameters.self, forKey: .ropeParameters)
        self.layerTypes = try container.decode([String].self, forKey: .layerTypes)
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        if let direct = try? container.decodeIfPresent([Int].self, forKey: .eosTokenId) {
            self.eosTokenIds = direct
        } else if let single = try container.decodeIfPresent(Int.self, forKey: .eosTokenId) {
            self.eosTokenIds = [single]
        } else {
            self.eosTokenIds = []
        }

        if let q = try container.decodeIfPresent(LFM2QuantizationConfig.self, forKey: .quantization) {
            self.quantization = q
        } else {
            self.quantization = try container.decodeIfPresent(LFM2QuantizationConfig.self, forKey: .quantizationConfig)
        }
    }
}

public enum LFM2Error: LocalizedError {
    case modelNotLoaded
    case unsupportedModelId(String)
    case missingFiles([String])
    case downloadFailed(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "LFM2 model is not loaded."
        case .unsupportedModelId(let id):
            return "Unsupported LFM2 model id: \(id)"
        case .missingFiles(let files):
            return "Missing required LFM2 files: \(files.joined(separator: ", "))"
        case .downloadFailed(let message):
            return "LFM2 download failed: \(message)"
        case .generationFailed(let message):
            return message
        }
    }
}
