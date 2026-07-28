import Foundation

public struct LagunaDFlashSpeculationConfig: Decodable, Sendable, Hashable {
    public let blockSize: Int
    public let maskTokenID: Int
    public let numTargetLayers: Int
    public let targetLayerIDs: [Int]
    public let causal: Bool

    private enum CodingKeys: String, CodingKey {
        case blockSize = "block_size"
        case maskTokenID = "mask_token_id"
        case numTargetLayers = "num_target_layers"
        case targetLayerIDs = "target_layer_ids"
        case causal
    }
}

public struct LagunaDFlashConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let vocabSize: Int
    public let draftVocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let attentionBias: Bool
    public let gating: String
    public let layerTypes: [String]
    public let slidingWindow: Int
    public let ropeTheta: Float
    public let eagleAuxHiddenStateLayerIDs: [Int]
    public let dflash: LagunaDFlashSpeculationConfig

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case draftVocabSize = "draft_vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case gating
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case ropeTheta = "rope_theta"
        case eagleAuxHiddenStateLayerIDs = "eagle_aux_hidden_state_layer_ids"
        case dflash = "dflash_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        self.draftVocabSize = try container.decode(Int.self, forKey: .draftVocabSize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.gating = try container.decodeIfPresent(String.self, forKey: .gating) ?? "per-head"
        self.layerTypes = try container.decode([String].self, forKey: .layerTypes)
        self.slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        self.eagleAuxHiddenStateLayerIDs = try container.decode(
            [Int].self,
            forKey: .eagleAuxHiddenStateLayerIDs
        )
        self.dflash = try container.decode(
            LagunaDFlashSpeculationConfig.self,
            forKey: .dflash
        )

        guard modelType == "laguna" else {
            throw DecodingError.dataCorruptedError(
                forKey: .modelType,
                in: container,
                debugDescription: "Expected model_type 'laguna'."
            )
        }
        guard draftVocabSize == vocabSize else {
            throw DecodingError.dataCorruptedError(
                forKey: .draftVocabSize,
                in: container,
                debugDescription: "Laguna DFlash shares the target lm_head and requires matching vocabularies."
            )
        }
        guard layerTypes.count == numHiddenLayers,
              Set(layerTypes).count == 1,
              layerTypes.first == "sliding_attention" else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes,
                in: container,
                debugDescription: "Laguna DFlash requires one sliding_attention entry per draft layer."
            )
        }
        guard !attentionBias else {
            throw DecodingError.dataCorruptedError(
                forKey: .attentionBias,
                in: container,
                debugDescription: "The official Laguna DFlash checkpoint requires bias-free attention."
            )
        }
        guard gating == "per-head" else {
            throw DecodingError.dataCorruptedError(
                forKey: .gating,
                in: container,
                debugDescription: "The official Laguna DFlash checkpoint requires per-head gating."
            )
        }
        guard dflash.causal else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash,
                in: container,
                debugDescription: "The official Laguna DFlash checkpoint requires causal draft attention."
            )
        }
        guard dflash.blockSize > 1,
              !dflash.targetLayerIDs.isEmpty,
              dflash.targetLayerIDs.count == eagleAuxHiddenStateLayerIDs.count,
              zip(dflash.targetLayerIDs, eagleAuxHiddenStateLayerIDs)
                .allSatisfy({ target, auxiliary in auxiliary == target + 1 }) else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash,
                in: container,
                debugDescription: "Laguna DFlash target layer metadata is inconsistent."
            )
        }
    }

    var ropeParameters: LagunaRopeParameters {
        LagunaRopeParameters(ropeTheta: ropeTheta)
    }
}
