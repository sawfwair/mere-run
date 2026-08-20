import Foundation

/// The small, deterministic state machine behind LocalEngine residency.
/// Runtime objects stay in LocalEngine; this type only decides when a release
/// starts immediately and when it must wait for active inference to finish.
struct RuntimeResidencyState: Equatable, Sendable {
    enum Activity: Equatable, Sendable {
        case idle
        case generatingImage
        case chatting
        case releasing
    }

    private(set) var activity: Activity = .idle
    private(set) var releasePending = false

    mutating func begin(_ next: Activity) -> Bool {
        guard activity == .idle, next == .generatingImage || next == .chatting else {
            return false
        }
        activity = next
        return true
    }

    /// Returns true when the caller should begin unloading now. Otherwise the
    /// release remains pending until `completeActivity()` is called.
    mutating func requestRelease() -> Bool {
        guard activity != .releasing else { return false }
        releasePending = true
        guard activity == .idle else { return false }
        activity = .releasing
        releasePending = false
        return true
    }

    mutating func beginRelease() -> Bool {
        guard activity == .idle else { return false }
        activity = .releasing
        releasePending = false
        return true
    }

    /// Returns true when a deferred release should start immediately.
    mutating func completeActivity() -> Bool {
        guard activity == .generatingImage || activity == .chatting else {
            return false
        }
        activity = .idle
        guard releasePending else { return false }
        activity = .releasing
        releasePending = false
        return true
    }

    mutating func completeRelease() {
        guard activity == .releasing else { return }
        activity = .idle
        releasePending = false
    }
}
