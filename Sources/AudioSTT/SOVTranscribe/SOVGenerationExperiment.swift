/// Resident-generation policy. Ordinary transcription uses the weight-preserving pipeline.
public enum SOVGenerationExperiment: String, CaseIterable, Sendable {
    case reference
    case pipeline
    case draftQ8 = "draft-q8"
    case retainedAllocator = "retained-allocator"
    case sharedCache = "shared-cache"
    case combined
    case rollback
    case compiled
    case block4 = "block-4"
    case block8 = "block-8"
    case block12 = "block-12"
    case adaptive
    case adaptiveAfterFirst = "adaptive-after-first"
    case vocabulary32k = "vocabulary-32k"

    package var pipelined: Bool { self != .reference && self != .draftQ8 && self != .sharedCache && self != .retainedAllocator }
    package var quantizedDraftHead: Bool { self == .draftQ8 || self == .combined }
    package var retainsAllocator: Bool { self == .retainedAllocator || self == .combined }
    package var reusesAttentionStorage: Bool { self == .sharedCache || self == .combined }
    package var draftTokenLimit: Int {
        switch self { case .block4: 3; case .block8: 7; case .block12: 11; default: 15 }
    }
}
