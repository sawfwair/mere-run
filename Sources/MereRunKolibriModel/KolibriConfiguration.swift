import Foundation

/// The released Kolibri architecture and an explicit per-projection affine policy.
package struct KolibriConfiguration: Codable, Sendable {
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
    package let eosTokenID: Int
    package let quantization: [String: Quantization]?

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
        case quantization = "mererun_quantization"
    }

    package func validate() throws {
        guard modelType == "kolibri1", hiddenSize > 0, headDim > 0, headDim.isMultiple(of: 2),
              numHiddenLayers > 0, layerTypes.count == numHiddenLayers,
              numAttentionHeads > 0, numKeyValueHeads > 0,
              numAttentionHeads.isMultiple(of: numKeyValueHeads),
              numExpertsPerTok > 0, numExpertsPerTok <= numExperts,
              moeIntermediateSize > 0, sharedExpertIntermediateSize > 0,
              slidingWindow > 0, maxPositionEmbeddings > 0,
              rmsNormEps.isFinite, rmsNormEps > 0,
              ropeTheta.isFinite, ropeTheta > 0,
              vocabSize > 0, (0..<vocabSize).contains(eosTokenID) else {
            throw KolibriModelError.invalidConfiguration
        }
        for (path, policy) in quantization ?? [:] {
            guard [2, 3, 4, 6, 8].contains(policy.bits),
                  [32, 64, 128].contains(policy.groupSize),
                  projectionDimensions(path: path) != nil,
                  let dims = projectionDimensions(path: path),
                  dims.input.isMultiple(of: policy.groupSize),
                  (dims.input * policy.bits).isMultiple(of: 32) else {
                throw KolibriModelError.invalidQuantization(path)
            }
        }
    }

    package func projectionDimensions(path: String) -> (input: Int, output: Int)? {
        if path == "lm_head" { return (hiddenSize, vocabSize) }
        let parts = path.split(separator: ".")
        guard parts.count >= 5, parts[0] == "model", parts[1] == "layers",
              let index = Int(parts[2]), (0..<numHiddenLayers).contains(index) else { return nil }
        if parts.count == 5, parts[3] == "self_attn" {
            switch parts[4] {
            case "q_proj": return (hiddenSize, numAttentionHeads * headDim)
            case "k_proj", "v_proj": return (hiddenSize, numKeyValueHeads * headDim)
            case "o_proj": return (numAttentionHeads * headDim, hiddenSize)
            default: return nil
            }
        }
        guard parts.count == 6, parts[3] == "mlp" else { return nil }
        let width: Int
        switch parts[4] {
        case "experts": width = moeIntermediateSize
        case "shared_experts": width = sharedExpertIntermediateSize
        default: return nil
        }
        switch parts[5] {
        case "gate_proj", "up_proj": return (hiddenSize, width)
        case "down_proj": return (width, hiddenSize)
        default: return nil
        }
    }
}

package enum KolibriModelError: Error, LocalizedError {
    case invalidConfiguration
    case invalidQuantization(String)
    case invalidWeights(String)
    package var errorDescription: String? {
        switch self {
        case .invalidConfiguration: "Invalid Kolibri architecture configuration."
        case .invalidQuantization(let path): "Invalid Kolibri affine quantization policy for \(path)."
        case .invalidWeights(let message): "Invalid Kolibri native weights: \(message)"
        }
    }
}
