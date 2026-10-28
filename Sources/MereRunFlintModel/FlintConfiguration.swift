import Foundation

/// Flint BF16 and the published MLX affine INT4 export.
package struct FlintConfiguration: Codable, Sendable {
    package struct Quantization: Codable, Equatable, Sendable {
        package let bits: Int
        package let groupSize: Int
        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
        }
    }

    package enum LayerType: String, Codable, Sendable {
        case sliding = "sliding_attention"
        case full = "full_attention"
    }

    package let modelType: String
    package let hiddenSize: Int
    package let numHiddenLayers: Int
    package let numAttentionHeads: Int
    package let numKeyValueHeads: Int
    package let headDim: Int
    package let maxPositionEmbeddings: Int
    package let rmsNormEps: Float
    package let vocabSize: Int
    package let ropeTheta: Float
    package let numExperts: Int
    package let numExpertsPerTok: Int
    package let moeIntermediateSize: Int
    package let sharedExpertIntermediateSize: Int
    package let normTopkProb: Bool
    package let slidingWindow: Int
    package let layerTypes: [LayerType]
    package let intermediateSize: Int
    package let mlpOnlyLayers: [Int]
    package let noRopeLayerIndices: [Int]
    package let attentionBias: Bool
    package let hiddenAct: String
    package let softmaxType: String
    package let routerScoreFunction: String
    package let tieWordEmbeddings: Bool
    package let eosTokenID: [Int]
    package let mlxExport: MLXExport?

    package struct MLXExport: Codable, Sendable {
        let bits: Int
        let embeddingBits: Int
        let routedExpertBits: Int
        enum CodingKeys: String, CodingKey {
            case bits
            case embeddingBits = "embedding_bits"
            case routedExpertBits = "routed_expert_bits"
        }
    }
    package let quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case ropeTheta = "rope_theta"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case eosTokenID = "eos_token_id"
        case quantization
        case intermediateSize = "intermediate_size"
        case mlpOnlyLayers = "mlp_only_layers"
        case noRopeLayerIndices = "no_rope_layer_indices"
        case attentionBias = "attention_bias"
        case hiddenAct = "hidden_act"
        case softmaxType = "softmax_type"
        case routerScoreFunction = "router_score_function"
        case tieWordEmbeddings = "tie_word_embeddings"
        case mlxExport = "mlx_export"
    }

    package func validate() throws {
        guard modelType == "flint", hiddenSize > 0, headDim > 0, headDim.isMultiple(of: 2),
              numHiddenLayers > 0, layerTypes.count == numHiddenLayers,
              numAttentionHeads > 0, numKeyValueHeads > 0,
              numAttentionHeads.isMultiple(of: numKeyValueHeads),
              numExpertsPerTok > 0, numExpertsPerTok <= numExperts,
              moeIntermediateSize > 0, sharedExpertIntermediateSize > 0,
              slidingWindow > 0, maxPositionEmbeddings > 0,
              rmsNormEps.isFinite, rmsNormEps > 0,
              ropeTheta.isFinite, ropeTheta > 0,
              vocabSize > 0, !eosTokenID.isEmpty,
              eosTokenID.allSatisfy({ (0..<vocabSize).contains($0) }),
              intermediateSize > 0, !attentionBias, hiddenAct == "silu",
              softmaxType == "learnable", routerScoreFunction == "sigmoid",
              normTopkProb, !tieWordEmbeddings,
              Set(mlpOnlyLayers).count == mlpOnlyLayers.count,
              Set(noRopeLayerIndices).count == noRopeLayerIndices.count,
              (mlpOnlyLayers + noRopeLayerIndices).allSatisfy({ (0..<numHiddenLayers).contains($0) }) else {
            throw FlintModelError.invalidConfiguration
        }
        if let quantization {
            guard quantization.bits == 4, quantization.groupSize == 64,
                  mlxExport?.bits == 4, mlxExport?.routedExpertBits == 4,
                  mlxExport?.embeddingBits == 8,
                  [hiddenSize, intermediateSize, moeIntermediateSize, sharedExpertIntermediateSize,
                   numAttentionHeads * headDim].allSatisfy({ $0.isMultiple(of: 64) }) else {
                throw FlintModelError.invalidQuantization("Expected the published group-64 INT4/INT8 export.")
            }
        } else if mlxExport != nil {
            throw FlintModelError.invalidQuantization("MLX export metadata requires a quantization policy.")
        }
    }

    package func policy(for path: String) -> Quantization? {
        guard let quantization else { return nil }
        return Quantization(bits: path == "model.embed_tokens" ? 8 : quantization.bits,
                            groupSize: quantization.groupSize)
    }


}

package enum FlintModelError: Error, LocalizedError {
    case invalidConfiguration
    case invalidQuantization(String)
    case invalidWeights(String)
    package var errorDescription: String? {
        switch self {
        case .invalidConfiguration: "Invalid Flint architecture configuration."
        case .invalidQuantization(let path): "Invalid Flint affine quantization policy for \(path)."
        case .invalidWeights(let message): "Invalid Flint native weights: \(message)"
        }
    }
}
