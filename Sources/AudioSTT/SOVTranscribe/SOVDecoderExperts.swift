/// Experimental resident decoder policies. Quantization repacks only MoE experts;
/// token agreement and transcription quality require separate qualification.
public enum SOVDecoderExperts: String, Sendable, CaseIterable {
    case bf16
    case bf16GEMV = "bf16-gemv"
    case q4
    case q6
    case q6Matrix = "q6-matrix"
    case mixed
    case q8
    case q8Metal = "q8-metal"
    case q8Group32 = "q8-g32"

    var bits: Int { switch self { case .q4: 4; case .q6, .q6Matrix: 6; default: 8 } }
    var requiresMetal: Bool { usesMetal || self == .q6Matrix }
    var usesMetal: Bool { self == .q8Metal || self == .mixed }

    var groupSize: Int? {
        switch self {
        case .bf16, .bf16GEMV: nil
        case .q4, .q6, .q6Matrix, .mixed, .q8, .q8Metal: 64
        case .q8Group32: 32
        }
    }
}
