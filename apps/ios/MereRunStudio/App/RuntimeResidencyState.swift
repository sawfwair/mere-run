import Foundation

/// Owns admission and preparation identity. Runtime objects stay in LocalEngine;
/// a preparation or progress callback can only publish through its current lease.
struct RuntimeResidencyState: Equatable, Sendable {
    enum Activity: Equatable, Sendable {
        case idle
        case generatingImage
        case chatting
        case releasing
    }

    struct ActivityLease: Equatable, Sendable {
        let id = UUID()
        let activity: Activity
        let modelID: String
    }

    struct WarmupLease: Equatable, Sendable {
        let id = UUID()
        let modelID: String
    }

    private enum Phase: Equatable, Sendable {
        case idle
        case warming(WarmupLease)
        case running(ActivityLease, warmup: WarmupLease?, releasePending: Bool)
        case releasing
    }

    private var phase: Phase = .idle
    private(set) var warmChatModelID: String?

    var activity: Activity {
        switch phase {
        case .idle, .warming: return .idle
        case .running(let lease, _, _): return lease.activity
        case .releasing: return .releasing
        }
    }

    var isIdle: Bool {
        if case .idle = phase { return true }
        return false
    }

    var releasePending: Bool {
        if case .running(_, _, releasePending: true) = phase { return true }
        return false
    }

    mutating func beginWarmup(modelID: String) -> WarmupLease? {
        guard case .idle = phase, warmChatModelID != modelID else { return nil }
        let lease = WarmupLease(modelID: modelID)
        phase = .warming(lease)
        return lease
    }

    /// A matching chat can claim its preparation. Image work invalidates preparation and
    /// waits for it to drain before unloading; the old callback cannot publish readiness.
    mutating func begin(_ next: Activity, modelID: String) -> ActivityLease? {
        guard next == .generatingImage || next == .chatting else { return nil }
        let warmup: WarmupLease?
        switch phase {
        case .idle:
            warmup = nil
        case .warming(let current):
            guard next == .generatingImage || current.modelID == modelID else { return nil }
            warmup = next == .chatting ? current : nil
        case .running, .releasing:
            return nil
        }
        let lease = ActivityLease(activity: next, modelID: modelID)
        phase = .running(lease, warmup: warmup, releasePending: false)
        return lease
    }

    @discardableResult
    mutating func completeWarmup(_ lease: WarmupLease, succeeded: Bool) -> Bool {
        switch phase {
        case .warming(let current) where current == lease:
            phase = .idle
        case .running(let current, warmup: let preparation?, releasePending: false) where preparation == lease:
            phase = .running(current, warmup: nil, releasePending: false)
        default:
            return false
        }
        warmChatModelID = succeeded ? lease.modelID : nil
        return true
    }

    func isCurrent(_ lease: ActivityLease) -> Bool {
        if case .running(let current, _, _) = phase { return current == lease }
        return false
    }

    mutating func recordChatPrepared(_ lease: ActivityLease) {
        guard lease.activity == .chatting,
              case .running(let current, warmup: nil, releasePending: false) = phase,
              current == lease else { return }
        warmChatModelID = lease.modelID
    }

    mutating func clearChatModel() { warmChatModelID = nil }

    /// Invalidates warmup immediately. The owner must join the preparation task before unload.
    /// Active inference keeps its lease until completion, then performs the deferred release.
    mutating func requestRelease() -> Bool {
        warmChatModelID = nil
        switch phase {
        case .idle, .warming:
            phase = .releasing
            return true
        case .running(let current, _, _):
            phase = .running(current, warmup: nil, releasePending: true)
            return false
        case .releasing:
            return false
        }
    }

    mutating func beginRelease() -> Bool {
        switch phase {
        case .idle, .warming:
            phase = .releasing
            warmChatModelID = nil
            return true
        case .running, .releasing:
            return false
        }
    }

    /// Only the owning operation may finish. Returns true when its deferred release starts.
    mutating func completeActivity(_ lease: ActivityLease) -> Bool {
        guard case .running(let current, _, let releasing) = phase, current == lease else { return false }
        phase = releasing ? .releasing : .idle
        return releasing
    }

    mutating func completeRelease() {
        guard case .releasing = phase else { return }
        phase = .idle
    }
}
