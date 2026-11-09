import Foundation

package struct Q35DFlash2Config: Decodable, Sendable {
    package struct Speculation: Decodable, Sendable {
        package let blockSize: Int
        package let convGroupSize: Int
        package let convKernelSize: Int
        package let maskTokenID: Int
        package let selectorRank: Int
        package let selectorTopK: Int
        /// Zero-based target decoder layer outputs, before the final norm.
        package let targetLayerIDs: [Int]

        enum CodingKeys: String, CodingKey {
            case blockSize = "block_size"
            case convGroupSize = "conv_group_size"
            case convKernelSize = "conv_kernel_size"
            case maskTokenID = "mask_token_id"
            case selectorRank = "selector_rank"
            case selectorTopK = "selector_top_k"
            case targetLayerIDs = "target_layer_ids"
        }
    }

    package struct Rope: Decodable, Sendable {
        package let theta: Float
        package let type: String
        enum CodingKeys: String, CodingKey {
            case theta = "rope_theta"
            case type = "rope_type"
        }
    }

    package let architectures: [String]
    package let modelType: String
    package let hiddenSize: Int
    package let intermediateSize: Int
    package let numHiddenLayers: Int
    package let numAttentionHeads: Int
    package let numKeyValueHeads: Int
    package let headDim: Int
    package let numTargetLayers: Int
    package let vocabSize: Int
    package let rmsNormEps: Float
    package let slidingWindow: Int
    package let layerTypes: [String]
    package let attentionBias: Bool
    package let isCausal: Bool
    package let hiddenAct: String
    package let tieWordEmbeddings: Bool
    package let rope: Rope
    package let dflash: Speculation

    enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numTargetLayers = "num_target_layers"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case attentionBias = "attention_bias"
        case isCausal = "is_causal"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
        case rope = "rope_parameters"
        case dflash = "dflash_config"
    }

    package func validate(target: Q35Config) throws {
        let text = target.textConfig
        guard architectures == ["DFlash2DraftModel"], modelType == "qwen3",
              hiddenAct == "silu", !isCausal, !attentionBias, !tieWordEmbeddings,
              rope.type == "default", rope.theta.isFinite, rope.theta > 0,
              rmsNormEps.isFinite, rmsNormEps > 0,
              hiddenSize > 0, intermediateSize > 0, numHiddenLayers > 0, vocabSize > 0,
              numAttentionHeads > 0, numKeyValueHeads > 0,
              numAttentionHeads.isMultiple(of: numKeyValueHeads),
              headDim > 0, headDim.isMultiple(of: 2), slidingWindow > 0,
              layerTypes == Array(repeating: "sliding_attention", count: numHiddenLayers),
              dflash.blockSize > 1, dflash.blockSize <= 9,
              dflash.convKernelSize == 2, dflash.convGroupSize > 0,
              hiddenSize.isMultiple(of: dflash.convGroupSize),
              dflash.selectorRank > 0,
              (1...vocabSize).contains(dflash.selectorTopK),
              (0..<vocabSize).contains(dflash.maskTokenID),
              !dflash.targetLayerIDs.isEmpty,
              dflash.targetLayerIDs == Array(Set(dflash.targetLayerIDs)).sorted(),
              dflash.targetLayerIDs.allSatisfy({ (0..<numTargetLayers).contains($0) }),
              hiddenSize == text.hiddenSize, vocabSize == text.vocabSize,
              numTargetLayers == text.numHiddenLayers,
              !text.usesMoE, !text.isQwen4Exp else {
            throw Q35Error.generationFailed("DFlash 2 configuration is unsupported or does not match the Qwen target.")
        }
    }
}
