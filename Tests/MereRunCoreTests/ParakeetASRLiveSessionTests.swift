import XCTest
import AudioCore
@testable import AudioSTT

final class ParakeetASRLiveSessionTests: XCTestCase {
    func testParakeetLiveEmitsPartialCommitStatsAndFinal() async throws {
        let transcriber = FakeParakeetLiveTranscriber(results: ["fast transcript"])
        let live = makeLive(transcriber: transcriber)
        let reader = collect(live.events)

        try await live.feed(samples: samples(count: 100, value: 0.1))
        try await live.feed(samples: samples(count: 90, value: 0))
        try await live.finish(reason: .eof)

        let events = try await reader.value
        let partials = events.compactMap { event -> Qwen3ASRLiveTranscript? in
            guard case .partial(let value) = event else { return nil }
            return value
        }
        let commits = events.compactMap { event -> Qwen3ASRLiveTranscript? in
            guard case .commit(let value) = event else { return nil }
            return value
        }
        XCTAssertEqual(partials.map(\.text), ["fast transcript"])
        XCTAssertEqual(commits.map(\.text), ["fast transcript"])
        XCTAssertEqual(partials.first?.revision, 1)
        XCTAssertEqual(commits.first?.revision, 2)
        XCTAssertTrue(events.contains { if case .stats = $0 { return true }; return false })
        XCTAssertEqual(finalReasons(events), [.eof])
        let callCount = await transcriber.callCount()
        XCTAssertEqual(callCount, 1)
    }

    func testParakeetLiveFinalDecodeIncludesAudioAfterLastPartial() async throws {
        let transcriber = FakeParakeetLiveTranscriber(results: ["draft", "complete"])
        let live = makeLive(transcriber: transcriber)
        let reader = collect(live.events)

        try await live.feed(samples: samples(count: 100, value: 0.1))
        try await live.feed(samples: samples(count: 50, value: 0.1))
        try await live.finish(reason: .stopped)

        let commits = try await reader.value.compactMap { event -> Qwen3ASRLiveTranscript? in
            guard case .commit(let value) = event else { return nil }
            return value
        }
        XCTAssertEqual(commits.map(\.text), ["complete"])
        let sampleCounts = await transcriber.sampleCounts()
        XCTAssertEqual(sampleCounts, [100, 150])
    }

    func testParakeetLiveCancelDiscardsActiveUtterance() async throws {
        let transcriber = FakeParakeetLiveTranscriber(results: ["draft"])
        let live = makeLive(transcriber: transcriber)
        let reader = collect(live.events)

        try await live.feed(samples: samples(count: 100, value: 0.1))
        await live.cancel()

        let events = try await reader.value
        XCTAssertFalse(events.contains { if case .commit = $0 { return true }; return false })
        XCTAssertEqual(finalReasons(events), [.cancelled])
    }

    func testParakeetLiveCommitDecodesOnlyOverlappedTailAfterPartial() async throws {
        let transcriber = FakeParakeetLiveTranscriber(results: ["hello brave", "brave new world"])
        let live = ParakeetASRLiveSession(
            request: ASRStreamingRequest(
                sampleRate: 1_000,
                decodeIntervalMs: 1_000,
                minDecodeAudioMs: 1_000,
                maxQueuedAudioMs: 2_000
            ),
            configuration: Qwen3ASRLiveConfiguration(
                decodeIntervalMs: 1_000,
                minDecodeAudioMs: 1_000,
                silenceMs: 900,
                preRollMs: 300,
                maxUtteranceMs: 30_000,
                maxQueuedAudioMs: 2_000
            ),
            transcribe: { samples, language in
                try await transcriber.transcribe(samples: samples, language: language)
            }
        )
        let reader = collect(live.events)

        try await live.feed(samples: samples(count: 1_000, value: 0.1))
        try await live.feed(samples: samples(count: 600, value: 0.1))
        try await live.finish(reason: .eof)

        let commits = try await reader.value.compactMap { event -> Qwen3ASRLiveTranscript? in
            guard case .commit(let value) = event else { return nil }
            return value
        }
        XCTAssertEqual(commits.map(\.text), ["hello brave new world"])
        let sampleCounts = await transcriber.sampleCounts()
        XCTAssertEqual(sampleCounts, [1_000, 1_100])
    }

    private func makeLive(transcriber: FakeParakeetLiveTranscriber) -> ParakeetASRLiveSession {
        ParakeetASRLiveSession(
            request: ASRStreamingRequest(
                sampleRate: 1_000,
                decodeIntervalMs: 100,
                minDecodeAudioMs: 100,
                maxQueuedAudioMs: 500
            ),
            configuration: Qwen3ASRLiveConfiguration(
                decodeIntervalMs: 100,
                minDecodeAudioMs: 100,
                silenceMs: 90,
                preRollMs: 300,
                maxUtteranceMs: 30_000,
                maxQueuedAudioMs: 500
            ),
            transcribe: { samples, language in
                try await transcriber.transcribe(samples: samples, language: language)
            }
        )
    }

    private func samples(count: Int, value: Float) -> [Float] {
        Array(repeating: value, count: count)
    }

    private func finalReasons(_ events: [Qwen3ASRLiveEvent]) -> [Qwen3ASRLiveFinishReason] {
        events.compactMap { event in
            guard case .final(let reason) = event else { return nil }
            return reason
        }
    }

    private func collect(
        _ stream: AsyncThrowingStream<Qwen3ASRLiveEvent, Error>
    ) -> Task<[Qwen3ASRLiveEvent], Error> {
        Task {
            var events: [Qwen3ASRLiveEvent] = []
            for try await event in stream {
                events.append(event)
            }
            return events
        }
    }
}

private actor FakeParakeetLiveTranscriber {
    private var results: [String]
    private var counts: [Int] = []

    init(results: [String]) {
        self.results = results
    }

    func transcribe(samples: [Float], language: String?) throws -> ASRResult {
        counts.append(samples.count)
        let text = results.isEmpty ? "" : results.removeFirst()
        return ASRResult(
            text: text,
            language: language,
            duration: Double(samples.count) / 1_000
        )
    }

    func callCount() -> Int {
        counts.count
    }

    func sampleCounts() -> [Int] {
        counts
    }
}
