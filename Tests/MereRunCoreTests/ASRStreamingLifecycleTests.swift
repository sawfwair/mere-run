import XCTest
import AudioCore
@testable import AudioSTT

final class ASRStreamingLifecycleTests: XCTestCase {
    private enum DecodeFailure: Error { case expected }

    func testConcurrentFinishCallsBothWaitForTheSameFinalDecode() async throws {
        let gate = StreamingDecodeBarrier()
        let session = heldSession(gate)
        try await session.feed(samples: Array(repeating: 0, count: 100))
        let first = Task { try await session.finish() }
        await gate.waitUntilStarted()
        let returned = expectation(description: "second finish cannot return before decode")
        returned.isInverted = true
        let second = Task {
            try await session.finish()
            returned.fulfill()
        }
        await fulfillment(of: [returned], timeout: 0.05)
        await gate.resume()
        try await first.value
        try await second.value
        let calls = await gate.calls
        XCTAssertEqual(calls, 1)
        var finals = 0
        for try await event in session.events {
            if case .final = event { finals += 1 }
        }
        XCTAssertEqual(finals, 1)
    }

    func testDecodeFailureRejectsFurtherInputAndEveryFinishReportsTheFailure() async throws {
        let session = Qwen3ASRStreamingSession(
            request: ASRStreamingRequest(sampleRate: 1_000, decodeIntervalMs: 100, minDecodeAudioMs: 100),
            decode: { _, _ in throw DecodeFailure.expected }
        )
        try await session.feed(samples: Array(repeating: 0, count: 100))
        do {
            for try await _ in session.events {}
            XCTFail("the stream must fail")
        } catch DecodeFailure.expected {}

        do {
            try await session.feed(samples: [0])
            XCTFail("a failed stream must not accept audio")
        } catch DecodeFailure.expected {}
        for _ in 0..<2 {
            do {
                try await session.finish()
                XCTFail("every finish must report the same decode failure")
            } catch DecodeFailure.expected {}
        }
        await session.cancel()
    }

    func testUnrequestedDecodeCancellationIsReportedAsAFailure() async throws {
        let session = Qwen3ASRStreamingSession(
            request: ASRStreamingRequest(sampleRate: 1_000, decodeIntervalMs: 100, minDecodeAudioMs: 100),
            decode: { _, _ in throw CancellationError() }
        )
        try await session.feed(samples: Array(repeating: 0, count: 100))
        do {
            for try await _ in session.events {}
            XCTFail("a decode that cancels itself must fail the stream, not end it silently")
        } catch is CancellationError {}
        do {
            try await session.finish()
            XCTFail("finish must report the decode failure instead of succeeding without a final")
        } catch is CancellationError {}
    }

    func testCancelWaitsForNativeDecodeAndPreventsFinalPublication() async throws {
        let gate = StreamingDecodeBarrier()
        let session = heldSession(gate)
        try await session.feed(samples: Array(repeating: 0, count: 100))
        let finishing = Task { try await session.finish() }
        await gate.waitUntilStarted()
        let returned = expectation(description: "cancel must join native decode")
        returned.isInverted = true
        let cancel = Task {
            await session.cancel()
            returned.fulfill()
        }
        await fulfillment(of: [returned], timeout: 0.05)
        await gate.resume()
        await cancel.value
        try await finishing.value
        var events = 0
        for try await _ in session.events { events += 1 }
        XCTAssertEqual(events, 0, "cancelled decode cannot publish partials, stats, or a final")
        let calls = await gate.calls
        XCTAssertEqual(calls, 1, "finalization cannot schedule another decode after cancellation")
    }

    func testCancelBeforeFinishDoesNotDecodeOrEmitAFinal() async throws {
        let gate = StreamingDecodeBarrier()
        let session = heldSession(gate)
        try await session.feed(samples: Array(repeating: 0, count: 100))
        await session.cancel()
        try await session.finish()
        var events = 0
        for try await _ in session.events { events += 1 }
        XCTAssertEqual(events, 0)
        let calls = await gate.calls
        XCTAssertEqual(calls, 0)
    }

    private func heldSession(_ gate: StreamingDecodeBarrier) -> Qwen3ASRStreamingSession {
        Qwen3ASRStreamingSession(
            request: ASRStreamingRequest(sampleRate: 1_000, decodeIntervalMs: 10_000, minDecodeAudioMs: 10_000),
            decode: { _, _ in
                await gate.decode()
                return Qwen3ASRStreamingDecodeOutput(result: ASRResult(text: "final"), tokensGenerated: 1)
            }
        )
    }
}

private actor StreamingDecodeBarrier {
    private(set) var calls = 0
    private var started: CheckedContinuation<Void, Never>?
    private var held: CheckedContinuation<Void, Never>?

    func decode() async {
        calls += 1
        started?.resume()
        started = nil
        await withCheckedContinuation { held = $0 }
    }

    func waitUntilStarted() async {
        if calls > 0 { return }
        await withCheckedContinuation { started = $0 }
    }

    func resume() {
        held?.resume()
        held = nil
    }
}
