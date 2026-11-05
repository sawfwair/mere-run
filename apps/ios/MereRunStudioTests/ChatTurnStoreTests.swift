import Foundation
import XCTest

@MainActor
final class ChatTurnStoreTests: XCTestCase {
    private var root: URL!
    private var threadURL: URL!

    override func setUp() async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("chat-turns-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        threadURL = root.appendingPathComponent("thread.json")
    }

    override func tearDown() async throws { try FileManager.default.removeItem(at: root) }

    func testRelayChangeStopsOnlyTheRemoteTurnAndRejectsItsLateReply() async {
        let store = ChatTurnStore(threadURL: threadURL)
        let remote = ChatReplyGate()
        let sending = Task {
            await store.send("remote question", origin: .remote, generate: { _, _ in try await remote.reply() })
        }
        await remote.waitUntilStarted()
        store.relayConnectionChanged()
        XCTAssertFalse(store.awaitingReply)
        XCTAssertEqual(store.messages.last?.failed, true)
        let afterDisconnect = store.messages
        remote.finish(.success(.init(text: "stale remote reply")))
        await sending.value
        XCTAssertEqual(store.messages, afterDisconnect)

        let local = ChatReplyGate()
        let localSend = Task {
            await store.send("local question", origin: .local, generate: { _, _ in try await local.reply() })
        }
        await local.waitUntilStarted()
        store.relayConnectionChanged()
        XCTAssertTrue(store.awaitingReply)
        XCTAssertEqual(store.activeOrigin, .local)
        local.finish(.success(.init(text: "local reply")))
        await localSend.value
        XCTAssertEqual(store.messages.last?.content, "local reply")
        XCTAssertNil(store.errorMessage)
    }

    func testNewChatRejectsThePreviousReplyInMemoryAndOnDisk() async throws {
        let store = ChatTurnStore(threadURL: threadURL)
        let gate = ChatReplyGate()
        let sending = Task {
            await store.send("old question", origin: .remote, generate: { _, delta in
                delta("old partial")
                return try await gate.reply()
            })
        }
        await gate.waitUntilStarted()
        store.newChat()
        XCTAssertFalse(store.awaitingReply)
        XCTAssertNil(store.streamingReply)
        gate.finish(.success(.init(text: "old reply")))
        await sending.value

        XCTAssertTrue(store.messages.isEmpty)
        XCTAssertNil(store.errorMessage)
        let saved = try JSONDecoder().decode([ChatTurnStore.Message].self, from: Data(contentsOf: threadURL))
        XCTAssertTrue(saved.isEmpty)
    }

    func testNewChatRejectsThePreviousFailure() async {
        let store = ChatTurnStore(threadURL: threadURL)
        let gate = ChatReplyGate()
        let sending = Task {
            await store.send("old", origin: .remote, generate: { _, _ in try await gate.reply() })
        }
        await gate.waitUntilStarted()
        store.newChat()
        gate.finish(.failure(URLError(.networkConnectionLost)))
        await sending.value
        XCTAssertNil(store.errorMessage)
        XCTAssertTrue(store.messages.isEmpty)
    }

    func testPreviousTurnProgressCannotOverwriteANewerReply() async {
        let store = ChatTurnStore(threadURL: threadURL)
        var oldDelta: (@MainActor (String) -> Void)?
        await store.send("first", origin: .local, generate: { _, delta in
            oldDelta = delta
            return .init(text: "first reply")
        })
        let gate = ChatReplyGate()
        let sending = Task {
            await store.send("second", origin: .remote, generate: { _, delta in
                delta("current")
                return try await gate.reply()
            })
        }
        await gate.waitUntilStarted()
        oldDelta?("old")
        XCTAssertEqual(store.streamingReply, "current")
        gate.finish(.success(.init(text: "second reply")))
        await sending.value
        oldDelta?("even later")
        XCTAssertNil(store.streamingReply)
        XCTAssertEqual(store.messages.map(\.content), ["first", "first reply", "second", "second reply"])
    }

    func testPreviousCompletionCannotClearANewerTurnsOwnership() async {
        let store = ChatTurnStore(threadURL: threadURL)
        let old = ChatReplyGate()
        let first = Task { await store.send("old", origin: .remote, generate: { _, _ in try await old.reply() }) }
        await old.waitUntilStarted()
        store.newChat()
        if store.awaitingReply {
            old.finish(.success(.init(text: "old")))
            await first.value
            XCTFail("a reset must release observation of the old turn")
            return
        }
        let current = ChatReplyGate()
        let second = Task {
            await store.send("new", origin: .local, generate: { _, delta in
                delta("current")
                return try await current.reply()
            })
        }
        await current.waitUntilStarted()
        old.finish(.failure(URLError(.networkConnectionLost)))
        await first.value
        XCTAssertTrue(store.awaitingReply)
        XCTAssertEqual(store.activeOrigin, .local)
        XCTAssertEqual(store.streamingReply, "current")
        XCTAssertNil(store.errorMessage)
        XCTAssertEqual(store.messages.map(\.content), ["new"])
        current.finish(.success(.init(text: "new reply")))
        await second.value
        XCTAssertFalse(store.awaitingReply)
    }

    func testNextLocalTurnWaitsForTheRetiredLocalTurnToReleaseTheEngine() async {
        let store = ChatTurnStore(threadURL: threadURL)
        let old = ChatReplyGate()
        let first = Task { await store.send("old", origin: .local, generate: { _, _ in try await old.reply() }) }
        await old.waitUntilStarted()
        store.newChat()
        XCTAssertFalse(store.awaitingReply)

        var generating = false
        let current = ChatReplyGate()
        let second = Task {
            await store.send("new", origin: .local, generate: { _, _ in
                generating = true
                return try await current.reply()
            })
        }
        await Task.yield()
        await Task.yield()
        XCTAssertTrue(store.awaitingReply, "the new turn owns the thread while it waits")
        XCTAssertFalse(generating, "the retired on-device turn still holds the engine")

        old.finish(.success(.init(text: "stale reply")))
        await first.value
        await current.waitUntilStarted()
        XCTAssertTrue(generating)
        current.finish(.success(.init(text: "new reply")))
        await second.value
        XCTAssertEqual(store.messages.map(\.content), ["new", "new reply"])
    }

    func testCallerCancellationRejectsUncooperativeProgressAndCompletion() async {
        let store = ChatTurnStore(threadURL: threadURL)
        let gate = ChatReplyGate()
        var delta: (@MainActor (String) -> Void)?
        let sending = Task {
            await store.send("question", origin: .remote, generate: { _, progress in
                delta = progress
                progress("before cancellation")
                return try await gate.reply()
            })
        }
        await gate.waitUntilStarted()
        sending.cancel()
        delta?("late partial")
        XCTAssertEqual(store.streamingReply, "before cancellation")
        gate.finish(.success(.init(text: "late reply")))
        await sending.value
        XCTAssertFalse(store.awaitingReply)
        XCTAssertEqual(store.messages.last?.failed, true)
        XCTAssertNotEqual(store.messages.last?.content, "late reply")
    }
}

@MainActor
private final class ChatReplyGate {
    private var started = false
    private var startWaiter: CheckedContinuation<Void, Never>?
    private var waiter: CheckedContinuation<ChatTurnStore.Reply, Error>?

    func reply() async throws -> ChatTurnStore.Reply {
        started = true
        startWaiter?.resume()
        startWaiter = nil
        return try await withCheckedThrowingContinuation { waiter = $0 }
    }

    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { startWaiter = $0 }
    }

    func finish(_ result: Result<ChatTurnStore.Reply, Error>) { waiter?.resume(with: result); waiter = nil }
}
