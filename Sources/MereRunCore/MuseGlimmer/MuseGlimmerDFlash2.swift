import MereRunTensor
import MLX

typealias MuseGlimmerDFlash2Selection = DFlash2Selection
typealias MuseGlimmerGroupedDynamicCausalConv = DFlash2GroupedDynamicCausalConv
typealias MuseGlimmerDFlash2CandidateSelector = DFlash2CandidateSelector

struct MuseGlimmerAssistantDraftOutput {
    let hidden: MLXArray
    let logits: MLXArray
}

enum MuseGlimmerDFlash2WeightKeys {
    static func normalized(_ key: String) -> String {
        switch key {
        case "fc.weight":
            return "encoder.fc.weight"
        case "hidden_norm.weight":
            return "encoder.output_norm_enc.weight"
        case "candidate_selector.predecessor_codebook":
            return "candidate_selector.predecessor_codebook.weight"
        case "candidate_selector.successor_codebook":
            return "candidate_selector.successor_codebook.weight"
        default:
            return key
        }
    }

    static func mapped(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        [(normalized(key), value)]
    }
}

func museGlimmerGroupedDynamicConvolution(
    hidden: MLXArray, dynamic: MLXArray, base: MLXArray, groupSize: Int
) -> MLXArray {
    dflash2GroupedDynamicConvolution(hidden: hidden, dynamic: dynamic, base: base, groupSize: groupSize)
}
