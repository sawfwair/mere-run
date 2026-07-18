import XCTest
import AudioCore
@testable import AudioSTT

final class Qwen3ASRLiveSessionTests: XCTestCase {
    func testSilenceCommitsOnceWithOrderedRevisionsAndAudioClock() async throws {
        let factory = FakeLiveASRFactory(results: ["hello"])
        let live = makeLive(factory: factory)
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
        XCTAssertEqual(partials.count, 1)
        XCTAssertEqual(commits.count, 1)
        XCTAssertEqual(partials.first?.utteranceId, commits.first?.utteranceId)
        XCTAssertEqual(partials.first?.revision, 1)
        XCTAssertEqual(commits.first?.revision, 2)
        XCTAssertGreaterThan(partials.first?.endMs ?? 0, 0)
        XCTAssertEqual(commits.first?.startMs, 0)
        XCTAssertEqual(commits.first?.endMs, 100)
        XCTAssertEqual(finalReasons(events), [.eof])
    }

    func testCancelDiscardsUncommittedPartialAndFinalizesOnce() async throws {
        let factory = FakeLiveASRFactory(results: ["draft"])
        let live = makeLive(factory: factory)
        let reader = collect(live.events)

        try await live.feed(samples: samples(count: 100, value: 0.1))
        await live.cancel()
        await live.cancel()

        let events = try await reader.value
        XCTAssertFalse(events.contains { if case .commit = $0 { return true }; return false })
        XCTAssertEqual(finalReasons(events), [.cancelled])
    }

    func testMaximumUtteranceForcesCommitAndStartsNewIdentifier() async throws {
        let factory = FakeLiveASRFactory(results: ["first", "second"])
        let live = makeLive(factory: factory, maximumMs: 200)
        let reader = collect(live.events)

        try await live.feed(samples: samples(count: 100, value: 0.1))
        try await live.feed(samples: samples(count: 100, value: 0.1))
        try await live.feed(samples: samples(count: 100, value: 0.1))
        try await live.finish(reason: .stopped)

        let events = try await reader.value
        let commits = events.compactMap { event -> Qwen3ASRLiveTranscript? in
            guard case .commit(let value) = event else { return nil }
            return value
        }
        XCTAssertEqual(commits.map { $0.text }, ["first", "second"])
        XCTAssertNotEqual(commits[0].utteranceId, commits[1].utteranceId)
        XCTAssertEqual(finalReasons(events), [.stopped])
    }

    func testLargeTransportChunkStillUsesHundredMillisecondVADWindows() async throws {
        let factory = FakeLiveASRFactory(results: ["kept beginning"])
        let live = makeLive(factory: factory)
        let reader = collect(live.events)
        let transportChunk = samples(count: 200, value: 0)
            + samples(count: 200, value: 0.1)
            + samples(count: 100, value: 0)

        try await live.feed(samples: transportChunk)
        try await live.finish(reason: .eof)

        let commits = try await reader.value.compactMap { event -> Qwen3ASRLiveTranscript? in
            guard case .commit(let value) = event else { return nil }
            return value
        }
        XCTAssertEqual(commits.count, 1)
        XCTAssertEqual(commits.first?.startMs, 0)
        XCTAssertEqual(commits.first?.endMs, 400)
    }

    func testFeedReturnsWhileSilenceCommitDecodeIsStillRunning() async throws {
        let probe = FakeFinishProbe()
        let factory = FakeLiveASRFactory(
            results: ["slow commit"],
            finishDelayNanoseconds: [500_000_000],
            finishProbe: probe
        )
        let live = makeLive(factory: factory)
        let reader = collect(live.events)
        let inputSamples = samples(count: 100, value: 0.1) + samples(count: 90, value: 0)
        let feedTask = Task {
            try await live.feed(samples: inputSamples)
        }

        await probe.waitUntilStarted()
        let clock = ContinuousClock()
        let started = clock.now
        try await feedTask.value
        XCTAssertLessThan(started.duration(to: clock.now), .milliseconds(100))

        try await live.finish(reason: .eof)
        let events = try await reader.value
        XCTAssertEqual(events.compactMap { if case .commit(let value) = $0 { value.text } else { nil } }, ["slow commit"])
        XCTAssertEqual(finalReasons(events), [.eof])
    }

    private func makeLive(factory: FakeLiveASRFactory, maximumMs: Int = 30_000) -> Qwen3ASRLiveSession {
        Qwen3ASRLiveSession(
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
                maxUtteranceMs: maximumMs,
                maxQueuedAudioMs: 500
            ),
            makeSession: { _ in await factory.make() }
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

private actor FakeLiveASRFactory {
    private var results: [String]
    private var finishDelayNanoseconds: [UInt64]
    private let finishProbe: FakeFinishProbe?

    init(
        results: [String],
        finishDelayNanoseconds: [UInt64] = [],
        finishProbe: FakeFinishProbe? = nil
    ) {
        self.results = results
        self.finishDelayNanoseconds = finishDelayNanoseconds
        self.finishProbe = finishProbe
    }

    func make() -> any ASRStreamingSession {
        FakeLiveASRSession(
            result: results.isEmpty ? "" : results.removeFirst(),
            finishDelayNanoseconds: finishDelayNanoseconds.isEmpty ? 0 : finishDelayNanoseconds.removeFirst(),
            finishProbe: finishProbe
        )
    }
}

private actor FakeFinishProbe {
    private var started = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func markStarted() {
        guard !started else { return }
        started = true
        let pending = waiters
        waiters.removeAll()
        pending.forEach { $0.resume() }
    }

    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { waiters.append($0) }
    }
}

private actor FakeLiveASRSession: ASRStreamingSession {
    nonisolated let events: AsyncThrowingStream<ASRStreamingEvent, Error>
    private let continuation: AsyncThrowingStream<ASRStreamingEvent, Error>.Continuation
    private let result: String
    private let finishDelayNanoseconds: UInt64
    private let finishProbe: FakeFinishProbe?
    private var emittedPartial = false
    private var finished = false
    private var sampleCount = 0

    init(
        result: String,
        finishDelayNanoseconds: UInt64,
        finishProbe: FakeFinishProbe?
    ) {
        self.result = result
        self.finishDelayNanoseconds = finishDelayNanoseconds
        self.finishProbe = finishProbe
        var captured: AsyncThrowingStream<ASRStreamingEvent, Error>.Continuation?
        self.events = AsyncThrowingStream { captured = $0 }
        self.continuation = captured!
    }

    func feed(samples: [Float]) async throws {
        sampleCount += samples.count
        guard !emittedPartial else { return }
        emittedPartial = true
        continuation.yield(.partial(text: result))
        continuation.yield(.stats(ASRStreamingStats(
            decodeCount: 1,
            totalAudioSeconds: Double(sampleCount) / 1_000,
            lastDecodeLatencyMs: 5,
            tokensGenerated: 1
        )))
    }

    func finish() async throws {
        guard !finished else { return }
        finished = true
        if finishDelayNanoseconds > 0 {
            await finishProbe?.markStarted()
            try await Task.sleep(nanoseconds: finishDelayNanoseconds)
        }
        continuation.yield(.final(result: ASRResult(
            text: result,
            language: "en",
            duration: Double(sampleCount) / 1_000
        )))
        continuation.finish()
    }

    func cancel() async {
        guard !finished else { return }
        finished = true
        continuation.finish()
    }
}
