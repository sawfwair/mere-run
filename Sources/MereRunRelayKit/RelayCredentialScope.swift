import Foundation

/// Binds credential access and refresh writes to one connection lifetime.
/// Invalidate before replacing or clearing the underlying credentials. Requests
/// holding an older client then fail without reading or overwriting the new token.
public final class RelayCredentialScope: RelayCredentialStorage, @unchecked Sendable {
    private let storage: any RelayCredentialStorage
    private let lock = NSLock()
    private var active = true

    public init(storage: any RelayCredentialStorage) { self.storage = storage }

    public func invalidate() { lock.withLock { active = false } }

    public func load() throws -> RelayOAuthTokenSet? {
        try withStorage { try storage.load() }
    }

    public func save(_ tokenSet: RelayOAuthTokenSet) throws {
        try withStorage { try storage.save(tokenSet) }
    }

    public func clear() throws {
        try withStorage { try storage.clear() }
    }

    private func withStorage<Value>(_ operation: () throws -> Value) throws -> Value {
        try lock.withLock {
            guard active else { throw RelayClientError("The relay connection changed. Reconnect before continuing.") }
            return try operation()
        }
    }
}
