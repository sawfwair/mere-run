import Foundation
import XCTest

@MainActor
final class RelaySnapshotStoreTests: XCTestCase {
    func testOlderRefreshCannotOverwriteNewerSnapshot() async {
        let store = RelaySnapshotStore<String>()
        let connection = UUID()
        let gate = SnapshotReadGate()
        let first = Task {
            await store.load(connectionID: connection, read: { try await gate.read() }, isCurrentConnection: { true })
        }
        await gate.waitUntilStarted()
        await store.load(connectionID: connection, read: { "current" }, isCurrentConnection: { true })
        gate.finish(.success("old"))
        await first.value
        XCTAssertEqual(store.value(for: connection), "current")
        XCTAssertNil(store.errorMessage(for: connection))
    }

    func testOlderRefreshFailureCannotReplaceNewerOutcome() async {
        let store = RelaySnapshotStore<String>()
        let connection = UUID()
        let gate = SnapshotReadGate()
        let first = Task {
            await store.load(connectionID: connection, read: { try await gate.read() }, isCurrentConnection: { true })
        }
        await gate.waitUntilStarted()
        await store.load(connectionID: connection, read: { "current" }, isCurrentConnection: { true })
        gate.finish(.failure(URLError(.networkConnectionLost)))
        await first.value
        XCTAssertEqual(store.value(for: connection), "current")
        XCTAssertNil(store.errorMessage(for: connection))
    }

    func testConnectionChangeHidesCachedSnapshotAndErrorBeforeNextLoad() async {
        let store = RelaySnapshotStore<String>()
        let connection = UUID()
        await store.load(connectionID: connection, read: { "old account" }, isCurrentConnection: { true })
        await store.load(connectionID: connection, read: { throw URLError(.notConnectedToInternet) }, isCurrentConnection: { true })
        XCTAssertEqual(store.value(for: connection), "old account")
        XCTAssertNotNil(store.errorMessage(for: connection))
        let replacement = UUID()
        XCTAssertNil(store.value(for: replacement))
        XCTAssertNil(store.errorMessage(for: replacement))
    }

    func testConnectionChangeRejectsResponseBeforeReplacementLoadStarts() async {
        let store = RelaySnapshotStore<String>()
        let original = UUID()
        var connection = original
        let gate = SnapshotReadGate()
        let loading = Task {
            await store.load(connectionID: original, read: { try await gate.read() }, isCurrentConnection: { connection == original })
        }
        await gate.waitUntilStarted()
        connection = UUID()
        gate.finish(.success("retired account"))
        await loading.value
        XCTAssertNil(store.value(for: original))
        XCTAssertNil(store.value(for: connection))
    }

    func testResetRejectsAnUncooperativeRead() async {
        let store = RelaySnapshotStore<String>()
        let connection = UUID()
        let gate = SnapshotReadGate()
        let loading = Task {
            await store.load(connectionID: connection, read: { try await gate.read() }, isCurrentConnection: { true })
        }
        await gate.waitUntilStarted()
        store.reset()
        gate.finish(.failure(URLError(.timedOut)))
        await loading.value
        XCTAssertNil(store.value(for: connection))
        XCTAssertNil(store.errorMessage(for: connection))
    }

    func testCancelledReadCannotPublishEvenWhenProviderReturnsSuccess() async {
        let store = RelaySnapshotStore<String>()
        let connection = UUID()
        await store.load(connectionID: connection, read: { "cached" }, isCurrentConnection: { true })
        let gate = SnapshotReadGate()
        let loading = Task {
            await store.load(connectionID: connection, read: { try await gate.read() }, isCurrentConnection: { true })
        }
        await gate.waitUntilStarted()
        loading.cancel()
        gate.finish(.success("late"))
        await loading.value
        XCTAssertEqual(store.value(for: connection), "cached")
        XCTAssertNil(store.errorMessage(for: connection))
    }

    func testNewConnectionFailureKeepsPreviousAccountDataHidden() async {
        let store = RelaySnapshotStore<String>()
        await store.load(connectionID: UUID(), read: { "old account" }, isCurrentConnection: { true })
        let connection = UUID()
        await store.load(connectionID: connection, read: { throw URLError(.notConnectedToInternet) }, isCurrentConnection: { true },
                         presentError: { _ in "Offline" })
        XCTAssertNil(store.value(for: connection))
        XCTAssertEqual(store.errorMessage(for: connection), "Offline")
        await store.load(connectionID: connection, read: { "new account" }, isCurrentConnection: { true })
        XCTAssertEqual(store.value(for: connection), "new account")
        XCTAssertNil(store.errorMessage(for: connection))
    }
}

@MainActor
private final class SnapshotReadGate {
    private var started = false
    private var startWaiter: CheckedContinuation<Void, Never>?
    private var waiter: CheckedContinuation<String, Error>?

    func read() async throws -> String {
        started = true
        startWaiter?.resume()
        startWaiter = nil
        return try await withCheckedThrowingContinuation { waiter = $0 }
    }

    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { startWaiter = $0 }
    }

    func finish(_ result: Result<String, Error>) { waiter?.resume(with: result); waiter = nil }
}
