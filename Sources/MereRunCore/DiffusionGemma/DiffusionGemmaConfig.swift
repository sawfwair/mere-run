import Foundation

public struct DiffusionGemmaQuantizationParameters: Decodable, Sendable, Hashable {
    public let groupSize: Int
    public let bits: Int

    private enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}
public struct DiffusionGemmaQuantizationConfig: Decodable, Sendable, Hashable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String
    public let overrides: [String: DiffusionGemmaQuantizationParameters]

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: DiffusionGemmaCodingKey.self)
        groupSize = try container.decode(Int.self, forKey: .init("group_size"))
        bits = try container.decode(Int.self, forKey: .init("bits"))
        mode = try container.decode(String.self, forKey: .init("mode"))

        let reserved = Set(["group_size", "bits", "mode"])
        var decoded: [String: DiffusionGemmaQuantizationParameters] = [:]
        for key in container.allKeys where !reserved.contains(key.stringValue) {
            decoded[key.stringValue] = try container.decode(
                DiffusionGemmaQuantizationParameters.self,
                forKey: key
            )
        }
        overrides = decoded
    }

    public func parameters(for path: String) -> DiffusionGemmaQuantizationParameters {
        overrides[path] ?? DiffusionGemmaQuantizationParameters(
            groupSize: groupSize,
            bits: bits
        )
    }
}

public struct DiffusionGemmaTextConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numGlobalKeyValueHeads: Int
    public let headDim: Int
    public let globalHeadDim: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int
    public let padTokenId: Int
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let finalLogitSoftcapping: Float
    public let numExperts: Int
    public let topKExperts: Int
    public let ropeParameters: [String: Gemma4TextRopeParameters]

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case padTokenId = "pad_token_id"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case ropeParameters = "rope_parameters"
    }
}

public struct DiffusionGemmaConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let architectures: [String]
    public let canvasLength: Int
    public let eosTokenIds: [Int]
    public let imageTokenId: Int?
    public let boiTokenId: Int?
    public let eoiTokenId: Int?
    public let textConfig: DiffusionGemmaTextConfig
    public let quantization: DiffusionGemmaQuantizationConfig

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case canvasLength = "canvas_length"
        case eosTokenId = "eos_token_id"
        case imageTokenId = "image_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case textConfig = "text_config"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        canvasLength = try container.decode(Int.self, forKey: .canvasLength)
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId)
        boiTokenId = try container.decodeIfPresent(Int.self, forKey: .boiTokenId)
        eoiTokenId = try container.decodeIfPresent(Int.self, forKey: .eoiTokenId)
        textConfig = try container.decode(DiffusionGemmaTextConfig.self, forKey: .textConfig)
        quantization = try container.decode(DiffusionGemmaQuantizationConfig.self, forKey: .quantization)
        if let ids = try? container.decode([Int].self, forKey: .eosTokenId) {
            eosTokenIds = ids
        } else {
            eosTokenIds = [try container.decode(Int.self, forKey: .eosTokenId)]
        }
    }
}

public struct DiffusionGemmaGenerationConfig: Decodable, Sendable, Hashable {
    public let confidenceThreshold: Float
    public let eosTokenIds: [Int]
    public let maxDenoisingSteps: Int
    public let maxNewTokens: Int
    public let stabilityThreshold: Int
    public let minimumTemperature: Float
    public let maximumTemperature: Float

    private enum CodingKeys: String, CodingKey {
        case confidenceThreshold = "confidence_threshold"
        case eosTokenId = "eos_token_id"
        case maxDenoisingSteps = "max_denoising_steps"
        case maxNewTokens = "max_new_tokens"
        case stabilityThreshold = "stability_threshold"
        case minimumTemperature = "t_min"
        case maximumTemperature = "t_max"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        confidenceThreshold = try container.decode(Float.self, forKey: .confidenceThreshold)
        eosTokenIds = try container.decode([Int].self, forKey: .eosTokenId)
        maxDenoisingSteps = try container.decode(Int.self, forKey: .maxDenoisingSteps)
        maxNewTokens = try container.decode(Int.self, forKey: .maxNewTokens)
        stabilityThreshold = try container.decode(Int.self, forKey: .stabilityThreshold)
        minimumTemperature = try container.decode(Float.self, forKey: .minimumTemperature)
        maximumTemperature = try container.decode(Float.self, forKey: .maximumTemperature)
    }
}

private struct DiffusionGemmaCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil

    init(_ stringValue: String) {
        self.stringValue = stringValue
    }

    init?(stringValue: String) {
        self.init(stringValue)
    }

    init?(intValue: Int) {
        return nil
    }
}
