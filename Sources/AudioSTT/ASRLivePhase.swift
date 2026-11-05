/// Input admission and terminal intent shared by live utterance coordinators.
/// Finish and cancellation tasks remain owned until their native work settles.
enum ASRLivePhase {
    case accepting
    case finishing
    case finished
    case cancelling
    case cancelled
    case failed(Error)

    var isAccepting: Bool {
        if case .accepting = self { return true }
        return false
    }

    var canProcess: Bool {
        switch self {
        case .accepting, .finishing: return true
        case .finished, .cancelling, .cancelled, .failed: return false
        }
    }

    var isCancelled: Bool {
        switch self {
        case .cancelling, .cancelled: return true
        default: return false
        }
    }

    var failure: Error? {
        if case .failed(let error) = self { return error }
        return nil
    }
}
