import Foundation
import MereRunAdmission

public struct RuntimeEvictionCandidate<Key: Hashable & Sendable>: Sendable {
    public let key: Key
    public let sortKey: String
    public let loaded: Bool
    public let ready: Bool
    public let lastAccess: Date?
    public let activeRequests: Int
    public let queuedRequests: Int
    public let pinned: Bool
    public let ttlSeconds: Int?

    public init(
        key: Key, sortKey: String, loaded: Bool, ready: Bool = true, lastAccess: Date?,
        activeRequests: Int, queuedRequests: Int, pinned: Bool, ttlSeconds: Int?
    ) {
        self.key = key
        self.sortKey = sortKey
        self.loaded = loaded
        self.ready = ready
        self.lastAccess = lastAccess
        self.activeRequests = activeRequests
        self.queuedRequests = queuedRequests
        self.pinned = pinned
        self.ttlSeconds = ttlSeconds
    }
}

/// Selects idle, unpinned residents in a stable least-recently-used order.
/// The owner revalidates leases and generation before applying each decision.
public enum RuntimeEvictionPlanner {
    public static func expired<Key>(
        _ candidates: [RuntimeEvictionCandidate<Key>], now: Date, excluding excluded: Set<Key> = []
    ) -> [Key] {
        eligible(candidates, excluding: excluded).filter {
            guard let lastAccess = $0.lastAccess, let ttl = $0.ttlSeconds else { return false }
            return now.timeIntervalSince(lastAccess) >= Double(ttl)
        }.map(\.key)
    }

    public static func memoryPressure<Key>(
        _ candidates: [RuntimeEvictionCandidate<Key>], pressure: RuntimeMemoryPressureLevel,
        excluding excluded: Set<Key> = []
    ) -> [Key] {
        switch pressure {
        case .disabled, .unknown, .nominal:
            return []
        case .elevated:
            return Array(eligible(candidates, excluding: excluded).prefix(1).map(\.key))
        case .critical:
            return eligible(candidates, excluding: excluded).map(\.key)
        }
    }

    private static func eligible<Key>(
        _ candidates: [RuntimeEvictionCandidate<Key>], excluding excluded: Set<Key>
    ) -> [RuntimeEvictionCandidate<Key>] {
        candidates.filter {
            $0.loaded && $0.ready && !excluded.contains($0.key) && !$0.pinned
                && $0.activeRequests == 0 && $0.queuedRequests == 0
        }.sorted {
            let lhs = $0.lastAccess ?? .distantPast
            let rhs = $1.lastAccess ?? .distantPast
            return lhs == rhs ? $0.sortKey < $1.sortKey : lhs < rhs
        }
    }
}
