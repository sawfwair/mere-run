import AudioCore
import XCTest
@testable import AudioSTT

final class ASRLiveLifecycleTests: XCTestCase {
    private enum Backend: CaseIterable { case resident, qwen }
    fileprivate enum DecodeFailure: Error { case expected }

    func testConcurrentFinishCallsJoinTheSameLiveFlush() async throws {
        for backend in Backend.allCases {
            let gate = LiveDecodeBarrier()
            let live = makeLive(backend, gate: gate)
            try await live.feed(Array(repeating: 0.1, count: 100))
            let first = Task { try await live.finish() }
            await gate.waitUntilStarted()
            let returned = expectation(description: "\(backend) second finish waits for flush")
            returned.isInverted = true
            let second = Task {
                try await live.finish()
                returned.fulfill()
            }
            await fulfillment(of: [returned], timeout: 0.05)
            await gate.resume()
            try await first.value
            try await second.value
            let events = try await collect(live.events)
            XCTAssertEqual(finalReasons(events), [.eof])
            let calls = await gate.calls
            XCTAssertEqual(calls, 1)
        }
    }

    func testCancelDuringLiveFlushDrainsDecodeAndWinsOverEOF() async throws {
        for backend in Backend.allCases {
            let gate = LiveDecodeBarrier()
            let live = makeLive(backend, gate: gate)
            try await live.feed(Array(repeating: 0.1, count: 100))
            let finishing = Task { try await live.finish() }
            await gate.waitUntilStarted()
            let returned = expectation(description: "\(backend) cancellation waits for native flush")
            returned.isInverted = true
            let cancellation = Task {
                await live.cancel()
                returned.fulfill()
            }
            await fulfillment(of: [returned], timeout: 0.05)
            await gate.resume()
            await cancellation.value
            try await finishing.value
            let events = try await collect(live.events)
            XCTAssertEqual(finalReasons(events), [.cancelled])
            XCTAssertFalse(events.contains { if case .commit = $0 { return true }; return false })
        }
    }

    func testFlushFailureClosesTheLiveStreamAndEveryFinishReportsIt() async throws {
        for backend in Backend.allCases {
            let gate = LiveDecodeBarrier()
            let live = makeLive(backend, gate: gate, fail: true)
            try await live.feed(Array(repeating: 0.1, count: 100))
            let closed = expectation(description: "\(backend) flush failure closes event stream")
            let reader = Task {
                do {
                    for try await _ in live.events {}
                    XCTFail("flush failure must fail the event stream")
                } catch DecodeFailure.expected {
                    closed.fulfill()
                } catch {
                    XCTFail("unexpected stream error: \(error)")
                }
            }
            let finishing = Task { try await live.finish() }
            await gate.waitUntilStarted()
            await gate.resume()
            do { try await finishing.value; XCTFail("flush must fail") } catch DecodeFailure.expected {}
            for _ in 0..<2 {
                do { try await live.finish(); XCTFail("finish must retain flush failure") } catch DecodeFailure.expected {}
            }
            await fulfillment(of: [closed], timeout: 2)
            reader.cancel()
            await reader.value
            await live.cancel()
        }
    }

    func testLiveCancellationReachesTheTaskPerformingTheEOFFlush() async throws {
        let gate = LiveDecodeBarrier()
        let requested = expectation(description: "native EOF task receives cancellation")
        let live = ASRUtteranceLiveSession(
            request: ASRStreamingRequest(sampleRate: 1_000),
            configuration: ASRLiveConfiguration(decodeIntervalMs: 10_000, minDecodeAudioMs: 10_000),
            transcribe: { _, _ in
                await withTaskCancellationHandler {
                    await gate.pause()
                } onCancel: {
                    requested.fulfill()
                }
                try Task.checkCancellation()
                return ASRResult(text: "must be discarded")
            }
        )
        try await live.feed(samples: Array(repeating: 0.1, count: 100))
        let finishing = Task { try await live.finish() }
        await gate.waitUntilStarted()
        let cancellation = Task { await live.cancel() }
        await fulfillment(of: [requested], timeout: 2)
        await gate.resume()
        await cancellation.value
        try await finishing.value
        let events = try await collect(live.events)
        XCTAssertEqual(finalReasons(events), [.cancelled])
    }

    func testCancelOwnsAndDiscardsAStreamingSessionCreatedAfterCancellation() async throws {
        let factory = LiveDecodeBarrier()
        let child = HeldLiveStreamingSession(gate: LiveDecodeBarrier(), fail: false)
        let live = Qwen3ASRLiveSession(request: ASRStreamingRequest(sampleRate: 1_000), makeSession: { _ in
            await factory.pause()
            return child
        })
        try await live.feed(samples: Array(repeating: 0.1, count: 100))
        await factory.waitUntilStarted()
        let returned = expectation(description: "cancel must join in-flight session creation")
        returned.isInverted = true
        let cancellation = Task {
            await live.cancel()
            returned.fulfill()
        }
        await fulfillment(of: [returned], timeout: 0.05)
        await factory.resume()
        await cancellation.value
        for _ in 0..<10 { await Task.yield() }
        let counts = await child.counts()
        XCTAssertEqual(counts.fed, 0)
        XCTAssertEqual(counts.cancelled, 1)
        let events = try await collect(live.events)
        XCTAssertEqual(finalReasons(events), [.cancelled])
    }

    private func makeLive(_ backend: Backend, gate: LiveDecodeBarrier, fail: Bool = false) -> LiveSessionHarness {
        let request = ASRStreamingRequest(sampleRate: 1_000, decodeIntervalMs: 1_000, minDecodeAudioMs: 1_000)
        let configuration = ASRLiveConfiguration(decodeIntervalMs: 1_000, minDecodeAudioMs: 1_000)
        switch backend {
        case .resident:
            let live = ASRUtteranceLiveSession(request: request, configuration: configuration, transcribe: { _, _ in
                await gate.pause()
                if fail { throw DecodeFailure.expected }
                return ASRResult(text: "late reply")
            })
            return LiveSessionHarness(events: live.events, feed: { try await live.feed(samples: $0) },
                                      finish: { try await live.finish() }, cancel: { await live.cancel() })
        case .qwen:
            let live = Qwen3ASRLiveSession(request: request, configuration: configuration, makeSession: { _ in
                HeldLiveStreamingSession(gate: gate, fail: fail)
            })
            return LiveSessionHarness(events: live.events, feed: { try await live.feed(samples: $0) },
                                      finish: { try await live.finish() }, cancel: { await live.cancel() })
        }
    }

    private func collect(_ stream: AsyncThrowingStream<ASRLiveEvent, Error>) async throws -> [ASRLiveEvent] {
        var events: [ASRLiveEvent] = []
        for try await event in stream { events.append(event) }
        return events
    }

    private func finalReasons(_ events: [ASRLiveEvent]) -> [ASRLiveFinishReason] {
        events.compactMap { if case .final(let reason) = $0 { reason } else { nil } }
    }
}

private struct LiveSessionHarness: Sendable {
    let events: AsyncThrowingStream<ASRLiveEvent, Error>
    let feed: @Sendable ([Float]) async throws -> Void
    let finish: @Sendable () async throws -> Void
    let cancel: @Sendable () async -> Void
}

private actor LiveDecodeBarrier {
    private(set) var calls = 0
    private var startWaiter: CheckedContinuation<Void, Never>?
    private var waiter: CheckedContinuation<Void, Never>?

    func pause() async {
        calls += 1
        startWaiter?.resume()
        startWaiter = nil
        await withCheckedContinuation { waiter = $0 }
    }

    func waitUntilStarted() async {
        if calls > 0 { return }
        await withCheckedContinuation { startWaiter = $0 }
    }

    func resume() { waiter?.resume(); waiter = nil }
}

private actor HeldLiveStreamingSession: ASRStreamingSession {
    nonisolated let events: AsyncThrowingStream<ASRStreamingEvent, Error>
    private let continuation: AsyncThrowingStream<ASRStreamingEvent, Error>.Continuation
    private let gate: LiveDecodeBarrier
    private let fail: Bool
    private var fed = 0
    private var cancelled = 0

    init(gate: LiveDecodeBarrier, fail: Bool) {
        self.gate = gate
        self.fail = fail
        let stream = AsyncThrowingStream<ASRStreamingEvent, Error>.makeStream()
        events = stream.stream
        continuation = stream.continuation
    }

    func feed(samples: [Float]) { fed += 1 }

    func finish() async throws {
        await gate.pause()
        if fail {
            continuation.finish(throwing: ASRLiveLifecycleTests.DecodeFailure.expected)
            throw ASRLiveLifecycleTests.DecodeFailure.expected
        }
        continuation.yield(.final(result: ASRResult(text: "late reply")))
        continuation.finish()
    }

    func cancel() { cancelled += 1; continuation.finish() }

    func counts() -> (fed: Int, cancelled: Int) { (fed, cancelled) }
}
