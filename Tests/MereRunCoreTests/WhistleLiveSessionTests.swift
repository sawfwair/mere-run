import XCTest
import AudioCore
@testable import AudioSTT

final class WhistleLiveSessionTests: XCTestCase {
    func testCommitRedecodesEntireUtteranceAfterPartial() async throws {
        let probe = LiveWhistleProbe(block: false)
        let live = makeLive(probe)
        let reader = collect(live.events)
        try await live.feed(samples: Array(repeating: 0.1, count: 100))
        await probe.waitUntilStarted()
        try await live.feed(samples: Array(repeating: 0.1, count: 150))
        try await live.finish()
        let events = try await reader.value
        let counts = await probe.counts
        XCTAssertEqual(counts, [100, 200, 250])
        XCTAssertEqual(events.compactMap { event -> String? in if case .partial(let value) = event { return value.text }; return nil }, ["100 samples", "200 samples"])
        XCTAssertEqual(events.compactMap { event -> String? in if case .commit(let value) = event { return value.text }; return nil }, ["250 samples"])
        XCTAssertEqual(events.filter { if case .final = $0 { return true }; return false }.count, 1)
    }

    func testCancellationDuringDecodeDiscardsCommitAndQueueIsBounded() async throws {
        let probe = LiveWhistleProbe(block: true)
        let live = makeLive(probe)
        let reader = collect(live.events)
        try await live.feed(samples: Array(repeating: 0.1, count: 100))
        await probe.waitUntilStarted()
        do {
            try await live.feed(samples: Array(repeating: 0.1, count: 1_001))
            XCTFail("Expected bounded-queue rejection")
        } catch { XCTAssertTrue(error.localizedDescription.contains("backpressure_exceeded")) }
        let finishing = Task { try await live.finish() }
        await live.cancel()
        await probe.release()
        try await finishing.value
        let events = try await reader.value
        XCTAssertFalse(events.contains { if case .commit = $0 { return true }; return false })
        XCTAssertFalse(events.contains { if case .partial = $0 { return true }; return false })
        XCTAssertEqual(events.compactMap { event -> ASRLiveFinishReason? in if case .final(let reason) = event { return reason }; return nil }, [.cancelled])
    }

    func testNonFiniteLiveAudioIsRejected() async throws {
        let live = makeLive(LiveWhistleProbe(block: false))
        do { try await live.feed(samples: [.nan]); XCTFail("Expected invalid audio") }
        catch { XCTAssertTrue(error.localizedDescription.contains("non-finite")) }
        await live.cancel()
    }

    private func makeLive(_ probe: LiveWhistleProbe) -> ASRUtteranceLiveSession {
        ASRUtteranceLiveSession(
            request: ASRStreamingRequest(sampleRate: 1_000, decodeIntervalMs: 100, minDecodeAudioMs: 100),
            configuration: ASRLiveConfiguration(decodeIntervalMs: 100, minDecodeAudioMs: 100, silenceMs: 900,
                                                maxUtteranceMs: 30_000, maxQueuedAudioMs: 1_000),
            redecodeWholeUtterance: true
        ) { samples, _ in await probe.transcribe(samples) }
    }

    private func collect(_ events: AsyncThrowingStream<ASRLiveEvent, Error>) -> Task<[ASRLiveEvent], Error> {
        Task { var result: [ASRLiveEvent] = []; for try await event in events { result.append(event) }; return result }
    }
}

private actor LiveWhistleProbe {
    let block: Bool
    var counts: [Int] = []
    private var started: [CheckedContinuation<Void, Never>] = []
    private var suspended: CheckedContinuation<Void, Never>?

    init(block: Bool) { self.block = block }
    func transcribe(_ samples: [Float]) async -> ASRResult {
        counts.append(samples.count)
        for waiter in started { waiter.resume() }
        started.removeAll()
        if block { await withCheckedContinuation { suspended = $0 } }
        return ASRResult(text: "\(samples.count) samples", language: "en")
    }
    func waitUntilStarted() async {
        if counts.isEmpty { await withCheckedContinuation { started.append($0) } }
    }
    func release() { suspended?.resume(); suspended = nil }
}
