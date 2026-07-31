import Foundation

public struct InklingQuantizationConfig: Codable, Sendable, Hashable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String
    public let scope: String

    private enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
        case scope
    }
}

public struct InklingTextConfig: Decodable, Sendable, Hashable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let vocabSize: Int
    public let unpaddedVocabSize: Int?
    public let rmsNormEps: Float
    public let useEmbedNorm: Bool
    public let logitsMUPWidthMultiplier: Float
    public let modelMaxLength: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let swaNumAttentionHeads: Int
    public let swaNumKeyValueHeads: Int
    public let swaHeadDim: Int
    public let slidingWindowSize: Int
    public let localLayerIDs: Set<Int>
    public let dRel: Int
    public let relExtent: Int
    public let logScalingNFloor: Int?
    public let logScalingAlpha: Float
    public let sconvKernelSize: Int
    public let denseMLPIndex: Int
    public let denseIntermediateSize: Int
    public let moeIntermediateSize: Int
    public let routedExpertCount: Int
    public let expertsPerToken: Int
    public let sharedExpertCount: Int
    public let routeScale: Float

    public func layerIsSliding(_ index: Int) -> Bool {
        localLayerIDs.contains(index)
    }

    public func layerIsDense(_ index: Int) -> Bool {
        index < denseMLPIndex
    }

    private enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case unpaddedVocabSize = "unpadded_vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case useEmbedNorm = "use_embed_norm"
        case logitsMUPWidthMultiplier = "logits_mup_width_multiplier"
        case modelMaxLength = "model_max_length"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case swaNumAttentionHeads = "swa_num_attention_heads"
        case swaNumKeyValueHeads = "swa_num_key_value_heads"
        case swaHeadDim = "swa_head_dim"
        case slidingWindowSize = "sliding_window_size"
        case localLayerIDs = "local_layer_ids"
        case dRel = "d_rel"
        case relExtent = "rel_extent"
        case logScalingNFloor = "log_scaling_n_floor"
        case logScalingAlpha = "log_scaling_alpha"
        case sconvKernelSize = "sconv_kernel_size"
        case denseMLPIndex = "dense_mlp_idx"
        case denseIntermediateSize = "dense_intermediate_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case routedExpertCount = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case sharedExpertCount = "n_shared_experts"
        case routeScale = "route_scale"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        unpaddedVocabSize = try container.decodeIfPresent(Int.self, forKey: .unpaddedVocabSize)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        useEmbedNorm = try container.decodeIfPresent(Bool.self, forKey: .useEmbedNorm) ?? true
        logitsMUPWidthMultiplier = try container.decodeIfPresent(
            Float.self,
            forKey: .logitsMUPWidthMultiplier
        ) ?? 1
        modelMaxLength = try container.decodeIfPresent(Int.self, forKey: .modelMaxLength)
            ?? container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? InklingResources.maximumContextLength
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try container.decode(Int.self, forKey: .headDim)
        swaNumAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .swaNumAttentionHeads)
            ?? numAttentionHeads
        swaNumKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .swaNumKeyValueHeads)
            ?? numKeyValueHeads
        swaHeadDim = try container.decodeIfPresent(Int.self, forKey: .swaHeadDim) ?? headDim
        slidingWindowSize = try container.decodeIfPresent(Int.self, forKey: .slidingWindowSize) ?? 512
        localLayerIDs = Set(try container.decodeIfPresent([Int].self, forKey: .localLayerIDs) ?? [])
        dRel = try container.decodeIfPresent(Int.self, forKey: .dRel) ?? 16
        relExtent = try container.decodeIfPresent(Int.self, forKey: .relExtent) ?? 1_024
        logScalingNFloor = try container.decodeIfPresent(Int.self, forKey: .logScalingNFloor)
        logScalingAlpha = try container.decodeIfPresent(Float.self, forKey: .logScalingAlpha) ?? 0.1
        sconvKernelSize = try container.decodeIfPresent(Int.self, forKey: .sconvKernelSize) ?? 4
        denseMLPIndex = try container.decodeIfPresent(Int.self, forKey: .denseMLPIndex) ?? 0
        denseIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .denseIntermediateSize)
            ?? 24_576
        moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
            ?? container.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? 3_072
        routedExpertCount = try container.decodeIfPresent(Int.self, forKey: .routedExpertCount) ?? 256
        expertsPerToken = try container.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? 6
        sharedExpertCount = try container.decodeIfPresent(Int.self, forKey: .sharedExpertCount) ?? 2
        routeScale = try container.decodeIfPresent(Float.self, forKey: .routeScale) ?? 8
    }
}

public struct InklingConfig: Decodable, Sendable, Hashable {
    public let architectures: [String]
    public let modelType: String
    public let eosTokenIDs: [Int]
    public let textConfig: InklingTextConfig
    public let quantization: InklingQuantizationConfig?

    private enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case eosTokenID = "eos_token_id"
        case textConfig = "text_config"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        modelType = try container.decode(String.self, forKey: .modelType)
        textConfig = try container.decode(InklingTextConfig.self, forKey: .textConfig)
        if let multiple = try? container.decodeIfPresent([Int].self, forKey: .eosTokenID) {
            eosTokenIDs = multiple
        } else if let single = try container.decodeIfPresent(Int.self, forKey: .eosTokenID) {
            eosTokenIDs = [single]
        } else {
            eosTokenIDs = []
        }
        quantization = try container.decodeIfPresent(InklingQuantizationConfig.self, forKey: .quantization)
            ?? container.decodeIfPresent(InklingQuantizationConfig.self, forKey: .quantizationConfig)
    }
}

public enum InklingError: LocalizedError {
    case modelNotLoaded
    case unsupportedModelID(String)
    case missingFiles([String])
    case downloadFailed(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Inkling model is not loaded."
        case .unsupportedModelID(let id):
            return "Unsupported Inkling model id: \(id)"
        case .missingFiles(let files):
            return "Missing required Inkling files: \(files.joined(separator: ", "))"
        case .downloadFailed(let message), .generationFailed(let message):
            return message
        }
    }
}
