import StudioKit

/// Cancellation is an outcome even when a process exits cleanly after Stop.
enum StudioConversationAnnouncement {
    static func completion(for state: JobState) -> String? {
        switch state {
        case .cancelled:
            return "Reply stopped."
        case .finished(let exit, _):
            return exit == 0 ? "Reply ready." : "Reply failed. Review the conversation for details."
        case .preflightFailed:
            return "Reply failed. Review the conversation for details."
        case .queued, .running:
            return nil
        }
    }
}
