import Foundation
import XCTest

@MainActor
final class RemoteChatReplyTests: XCTestCase {
    func testCancellingAnInFlightURLSessionReadCancelsTheRemoteJob() async {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [HeldChatRequest.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let started = expectation(description: "reply request started")
        let cancelled = expectation(description: "remote job cancellation requested")
        let task = Task {
            do {
                _ = try await RemoteChatReply.observe(read: {
                    let request = URLRequest(url: URL(string: "https://chat-review.invalid/inspect")!)
                    HeldChatRequest.onStart = { started.fulfill() }
                    return try await session.data(for: request)
                }, cancelJob: {
                    XCTAssertFalse(Task.isCancelled, "cleanup must have an uncancelled task for its own URLSession request")
                    cancelled.fulfill()
                })
                XCTFail("the cancelled reply read must fail")
            } catch is CancellationError {
            } catch {
                XCTFail("transport cancellation must be reported as cancellation, got \(error)")
            }
        }
        await fulfillment(of: [started], timeout: 2)
        task.cancel()
        await task.value
        await fulfillment(of: [cancelled], timeout: 2)
        HeldChatRequest.onStart = nil
    }

    func testCancelledTaskCleansUpWhenProviderThrowsADifferentError() async {
        let gate = RemoteReplyGate()
        let cancelled = expectation(description: "remote job cancelled")
        let task = Task {
            do {
                let _: String = try await RemoteChatReply.observe(read: { try await gate.read() }, cancelJob: {
                    XCTAssertFalse(Task.isCancelled)
                    cancelled.fulfill()
                })
                XCTFail("the cancelled reply must fail")
            } catch is CancellationError {
            } catch {
                XCTFail("task cancellation must win over a concurrent transport error")
            }
        }
        await gate.waitUntilStarted()
        task.cancel()
        gate.fail(URLError(.networkConnectionLost))
        await task.value
        await fulfillment(of: [cancelled], timeout: 2)
    }

    func testCooperativeCancellationStillRequestsRemoteCleanup() async {
        let cancelled = expectation(description: "remote job cancelled")
        do {
            let _: String = try await RemoteChatReply.observe(read: { throw CancellationError() }, cancelJob: {
                XCTAssertFalse(Task.isCancelled)
                cancelled.fulfill()
            })
            XCTFail("the reply must fail")
        } catch is CancellationError {
        } catch {
            XCTFail("unexpected error: \(error)")
        }
        await fulfillment(of: [cancelled], timeout: 2)
    }

    func testOrdinaryFailureIsPreservedWithoutCancellingTheJob() async {
        do {
            let _: String = try await RemoteChatReply.observe(read: { throw URLError(.timedOut) }, cancelJob: {
                XCTFail("an ordinary polling error must not cancel the job")
            })
            XCTFail("the polling error must be reported")
        } catch {
            XCTAssertEqual((error as? URLError)?.code, .timedOut)
        }
    }

    func testSuccessfulReplyDoesNotCancelTheJob() async throws {
        let reply = try await RemoteChatReply.observe(read: { "reply" }, cancelJob: {
            XCTFail("successful jobs need no cancellation")
        })
        XCTAssertEqual(reply, "reply")
    }
}

private final class HeldChatRequest: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var startHandler: (@Sendable () -> Void)?

    static var onStart: (@Sendable () -> Void)? {
        get { lock.withLock { startHandler } }
        set { lock.withLock { startHandler = newValue } }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() { Self.onStart?() }
    override func stopLoading() {}
}

@MainActor
private final class RemoteReplyGate {
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

    func fail(_ error: Error) { waiter?.resume(throwing: error); waiter = nil }
}
