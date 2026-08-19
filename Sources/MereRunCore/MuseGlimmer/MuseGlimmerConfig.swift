import Foundation

public struct MuseGlimmerQuantizationConfig: Codable, Sendable, Hashable {
    public let groupSize: Int
    public let bits: Int
    public let mode: String

    private enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        groupSize = try container.decode(Int.self, forKey: .groupSize)
        bits = try container.decode(Int.self, forKey: .bits)
        mode = try container.decodeIfPresent(String.self, forKey: .mode) ?? "affine"
    }
}

public struct MuseGlimmerRopeParameters: Codable, Sendable, Hashable {
    public let ropeTheta: Float
    public let ropeType: String

    private enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
    }
}

public struct MuseGlimmerTextConfig: Codable, Sendable, Hashable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let maxPositionEmbeddings: Int
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let layerRopeTheta: [Float]
    public let ropeParameters: MuseGlimmerRopeParameters
    public let rmsNormEps: Float
    public let postNormEps: Float
    public let qkScaleFactor: Float
    public let outputMultiplier: Float
    public let finalLogitSoftcapping: Float
    public let hiddenActivation: String
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let vocabSize: Int
    public let tieWordEmbeddings: Bool
    public let bosTokenId: Int?
    public let eosTokenId: Int?
    public let padTokenId: Int?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case layerRopeTheta = "layer_rope_theta"
        case ropeParameters = "rope_parameters"
        case rmsNormEps = "rms_norm_eps"
        case postNormEps = "post_norm_eps"
        case qkScaleFactor = "qk_scale_factor"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case hiddenActivation = "hidden_activation"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case vocabSize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try container.decode(Int.self, forKey: .headDim)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        layerRopeTheta = try container.decode([Float].self, forKey: .layerRopeTheta)
        ropeParameters = try container.decode(MuseGlimmerRopeParameters.self, forKey: .ropeParameters)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        postNormEps = try container.decode(Float.self, forKey: .postNormEps)
        qkScaleFactor = try container.decode(Float.self, forKey: .qkScaleFactor)
        outputMultiplier = try container.decode(Float.self, forKey: .outputMultiplier)
        finalLogitSoftcapping = try container.decode(Float.self, forKey: .finalLogitSoftcapping)
        hiddenActivation = try container.decode(String.self, forKey: .hiddenActivation)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId)
        eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId)
        padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId)

        guard layerTypes.count == numHiddenLayers,
              layerRopeTheta.count == numHiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes,
                in: container,
                debugDescription: "Muse Glimmer layer_types and layer_rope_theta must match num_hidden_layers."
            )
        }
    }
}

public struct MuseGlimmerVisionConfig: Codable, Sendable, Hashable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let patchSize: Int
    public let patchTemporal: Int
    public let mergeSize: Int
    public let posEmbHeight: Int
    public let posEmbWidth: Int
    public let maxPositionEmbeddings: Int
    public let layerNormEps: Float
    public let hiddenActivation: String
    public let layerTypes: [String]
    public let ropeParameters: MuseGlimmerRopeParameters

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case patchSize = "patch_size"
        case patchTemporal = "patch_temporal"
        case mergeSize = "merge_size"
        case posEmbHeight = "pos_emb_height"
        case posEmbWidth = "pos_emb_width"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layerNormEps = "layer_norm_eps"
        case hiddenActivation = "hidden_act"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }
}

public struct MuseGlimmerConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let architectures: [String]
    public let textConfig: MuseGlimmerTextConfig
    public let visionConfig: MuseGlimmerVisionConfig
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let outHiddenSize: Int
    public let projectorHiddenSize: Int
    public let projectorHiddenActivation: String
    public let quantization: MuseGlimmerQuantizationConfig?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case outHiddenSize = "out_hidden_size"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorHiddenActivation = "projector_hidden_act"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        textConfig = try container.decode(MuseGlimmerTextConfig.self, forKey: .textConfig)
        visionConfig = try container.decode(MuseGlimmerVisionConfig.self, forKey: .visionConfig)
        imageTokenId = try container.decode(Int.self, forKey: .imageTokenId)
        videoTokenId = try container.decode(Int.self, forKey: .videoTokenId)
        outHiddenSize = try container.decode(Int.self, forKey: .outHiddenSize)
        projectorHiddenSize = try container.decode(Int.self, forKey: .projectorHiddenSize)
        projectorHiddenActivation = try container.decode(String.self, forKey: .projectorHiddenActivation)
        quantization = try container.decodeIfPresent(MuseGlimmerQuantizationConfig.self, forKey: .quantization)
            ?? container.decodeIfPresent(MuseGlimmerQuantizationConfig.self, forKey: .quantizationConfig)
    }
}

public struct MuseGlimmerDFlash2Config: Decodable, Sendable, Hashable {
    public let blockSize: Int
    public let convGroupSize: Int
    public let convKernelSize: Int
    public let finalLogitSoftcapping: Float
    public let inputEmbeddingScale: Float
    public let maskTokenId: Int
    public let outputMultiplier: Float
    public let selectorRank: Int
    public let selectorTopK: Int
    public let targetLayerIds: [Int]

    private enum CodingKeys: String, CodingKey {
        case blockSize = "block_size"
        case convGroupSize = "conv_group_size"
        case convKernelSize = "conv_kernel_size"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case inputEmbeddingScale = "input_embedding_scale"
        case maskTokenId = "mask_token_id"
        case outputMultiplier = "output_multiplier"
        case selectorRank = "selector_rank"
        case selectorTopK = "selector_top_k"
        case targetLayerIds = "target_layer_ids"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        blockSize = try container.decode(Int.self, forKey: .blockSize)
        convGroupSize = try container.decode(Int.self, forKey: .convGroupSize)
        convKernelSize = try container.decode(Int.self, forKey: .convKernelSize)
        finalLogitSoftcapping = try container.decode(Float.self, forKey: .finalLogitSoftcapping)
        inputEmbeddingScale = try container.decodeIfPresent(Float.self, forKey: .inputEmbeddingScale) ?? 1
        maskTokenId = try container.decode(Int.self, forKey: .maskTokenId)
        outputMultiplier = try container.decode(Float.self, forKey: .outputMultiplier)
        selectorRank = try container.decode(Int.self, forKey: .selectorRank)
        selectorTopK = try container.decode(Int.self, forKey: .selectorTopK)
        targetLayerIds = try container.decode([Int].self, forKey: .targetLayerIds)
    }
}

public struct MuseGlimmerAssistantConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let architectures: [String]
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let ropeParameters: MuseGlimmerRopeParameters
    public let maxPositionEmbeddings: Int
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let attentionDropout: Float
    public let hiddenActivation: String
    public let vocabSize: Int?
    public let bosTokenId: Int?
    public let eosTokenId: Int?
    public let padTokenId: Int?
    public let blockSize: Int
    public let maskTokenId: Int
    public let targetLayerIds: [Int]
    public let isCausal: Bool?
    public let dflash2: MuseGlimmerDFlash2Config?
    public let quantization: MuseGlimmerQuantizationConfig?

    public var isDFlash2: Bool {
        architectures.contains("DFlash2DraftModel")
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case attentionDropout = "attention_dropout"
        case hiddenActivation = "hidden_act"
        case vocabSize = "vocab_size"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case blockSize = "block_size"
        case maskTokenId = "mask_token_id"
        case targetLayerIds = "target_layer_ids"
        case isCausal = "is_causal"
        case dflash2 = "dflash_config"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try container.decode(Int.self, forKey: .headDim)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        ropeParameters = try container.decode(MuseGlimmerRopeParameters.self, forKey: .ropeParameters)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0
        hiddenActivation = try container.decode(String.self, forKey: .hiddenActivation)
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize)
        bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId)
        eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId)
        padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId)
        dflash2 = try container.decodeIfPresent(MuseGlimmerDFlash2Config.self, forKey: .dflash2)
        blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize)
            ?? dflash2?.blockSize
            ?? Self.missingInteger(.blockSize, in: container)
        maskTokenId = try container.decodeIfPresent(Int.self, forKey: .maskTokenId)
            ?? dflash2?.maskTokenId
            ?? Self.missingInteger(.maskTokenId, in: container)
        targetLayerIds = try container.decodeIfPresent([Int].self, forKey: .targetLayerIds)
            ?? dflash2?.targetLayerIds
            ?? []
        isCausal = try container.decodeIfPresent(Bool.self, forKey: .isCausal)
        quantization = try container.decodeIfPresent(MuseGlimmerQuantizationConfig.self, forKey: .quantization)
            ?? container.decodeIfPresent(MuseGlimmerQuantizationConfig.self, forKey: .quantizationConfig)

        guard layerTypes.count == numHiddenLayers,
              layerTypes.allSatisfy({ $0 == "sliding_attention" }) else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes,
                in: container,
                debugDescription: "Muse Glimmer DFlash requires one sliding_attention entry per assistant layer."
            )
        }
        guard blockSize > 1, !targetLayerIds.isEmpty else {
            throw DecodingError.dataCorruptedError(
                forKey: .blockSize,
                in: container,
                debugDescription: "Muse Glimmer DFlash requires a block larger than one token and target layers."
            )
        }
        if architectures.contains("DFlash2DraftModel") {
            guard let dflash2,
                  let vocabSize,
                  vocabSize > 0,
                  dflash2.convKernelSize > 0,
                  dflash2.convGroupSize > 0,
                  hiddenSize.isMultiple(of: dflash2.convGroupSize),
                  dflash2.selectorRank > 0,
                  dflash2.selectorTopK > 0,
                  dflash2.selectorTopK <= vocabSize else {
                throw DecodingError.dataCorruptedError(
                    forKey: .dflash2,
                    in: container,
                    debugDescription: "Muse Glimmer DFlash2 requires valid convolution, vocabulary, and selector dimensions."
                )
            }
        }
    }

    private static func missingInteger(
        _ key: CodingKeys,
        in container: KeyedDecodingContainer<CodingKeys>
    ) throws -> Int {
        throw DecodingError.keyNotFound(
            key,
            DecodingError.Context(
                codingPath: container.codingPath,
                debugDescription: "Muse Glimmer DFlash requires \(key.stringValue)."
            )
        )
    }
}

struct MuseGlimmerGenerationConfig: Decodable {
    let bosTokenId: Int?
    let eosTokenIds: [Int]
    let padTokenId: Int?
    let maxLength: Int?

    private enum CodingKeys: String, CodingKey {
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case maxLength = "max_length"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId)
        padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId)
        maxLength = try container.decodeIfPresent(Int.self, forKey: .maxLength)
        if let values = try? container.decode([Int].self, forKey: .eosTokenId) {
            eosTokenIds = values
        } else if let value = try container.decodeIfPresent(Int.self, forKey: .eosTokenId) {
            eosTokenIds = [value]
        } else {
            eosTokenIds = []
        }
    }
}
