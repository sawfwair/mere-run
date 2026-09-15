import Foundation
import XCTest
import AudioCore

final class SpeechSynthesisOperationTests: XCTestCase {
    func testPlansPreserveRequestValuesAndExplicitEncodingDefaults() throws {
        var request = TTSRequest(text: "  Say this exactly.  ", outputURL: URL(fileURLWithPath: "/fixture/output.wav"))
        let file = try SpeechSynthesisPlan(request: request)
        XCTAssertEqual(file.request.text, "  Say this exactly.  ")
        XCTAssertEqual(file.request.voiceDescription, "A calm female voice with clear pronunciation")
        XCTAssertEqual(file.request.language, "auto")
        XCTAssertEqual(file.request.temperature, 0.6)
        XCTAssertEqual(file.request.speed, 1)
        XCTAssertEqual(file.exportPlan.options.format, .pcm16)
        XCTAssertEqual(file.exportPlan.options.normalization, .none)
        XCTAssertEqual(file.exportPlan.options.fadeInMilliseconds, 0)
        XCTAssertEqual(file.exportPlan.options.fadeOutMilliseconds, 0)
        XCTAssertFalse(file.exportPlan.options.dither)
        request.text = "A later edit"
        XCTAssertEqual(file.request.text, "  Say this exactly.  ")
        let stream = try SpeechSynthesisPlan(request: file.request, streamingOptions: TTSStreamingOptions(chunkTokenInterval: 17, emitTokenEvents: false))
        XCTAssertEqual(stream.streamingOptions?.chunkTokenInterval, 17)
        XCTAssertEqual(stream.streamingOptions?.emitTokenEvents, false)
        XCTAssertEqual(stream.exportPlan.options.format, .float32)
        XCTAssertEqual(stream.exportPlan.clipping, .preserveFloatHeadroom)
        var speedHint = file.request
        speedHint.speed = 1.25
        XCTAssertEqual(try SpeechSynthesisPlan(request: speedHint).request.speed, 1.25)
    }

    func testInvalidParametersAndCloneReferencesAreRejectedBeforeExecution() throws {
        let output = URL(fileURLWithPath: "/fixture/output.wav")
        XCTAssertThrowsError(try SpeechSynthesisPlan(request: TTSRequest(text: " \n ", outputURL: output)))
        for temperature: Float in [-1, 2.1, .nan, .infinity, -.infinity] {
            XCTAssertThrowsError(try SpeechSynthesisPlan(request: TTSRequest(text: "hello", temperature: temperature, outputURL: output)))
        }
        for speed: Double in [0, 0.249999999999, 4.000000000001, .nan, .infinity] {
            XCTAssertThrowsError(try SpeechSynthesisPlan.validatedSpeed(speed))
        }
        let request = TTSRequest(text: "hello", outputURL: output)
        XCTAssertThrowsError(try SpeechSynthesisPlan(request: request, streamingOptions: TTSStreamingOptions(chunkTokenInterval: 0)))
        XCTAssertThrowsError(try SpeechSynthesisPlan(request: TTSRequest(text: "hello", voiceMode: .clone, outputURL: output)))
        XCTAssertThrowsError(try SpeechSynthesisPlan(request: TTSRequest(
            text: "hello", voiceMode: .clone,
            cloneReference: TTSCloneReference(audioURL: output, transcript: "  "), outputURL: output
        )))
    }

    func testExecutionUsesTheExactPlanAndWritesLiteralPCM16Samples() async throws {
        let output = try directory().appendingPathComponent("nested/output.wav")
        let plan = try SpeechSynthesisPlan(request: TTSRequest(text: "hello", temperature: 0, outputURL: output))
        let probe = SynthesisProbe()
        let executor = SynthesisFixture(generate: { request in
            await probe.record(request)
            return try AudioWaveform(interleaved: [-2, -0.5, 0, 0.5, 2, .nan], channels: 1, sampleRate: 24_000)
        })
        let outcome = try await SpeechSynthesisOperation.execute(plan, executor: executor)
        let calls = await probe.requests
        XCTAssertEqual(calls, [plan.request])
        XCTAssertEqual(outcome.result.audioURL, output)
        XCTAssertEqual(outcome.result.sampleRate, 24_000)
        XCTAssertEqual(outcome.result.duration, 6.0 / 24_000)
        XCTAssertEqual(outcome.artifact.statistics.replacedNonfiniteSamples, 1)
        XCTAssertEqual(outcome.artifact.statistics.clippedSamples, 2)
        let data = try Data(contentsOf: output)
        XCTAssertEqual(data[20], 1)
        XCTAssertEqual(data[34], 16)
        XCTAssertEqual(read32(data, 40), 12)
        let values = (0..<6).map { Int16(bitPattern: UInt16(data[44 + $0 * 2]) | UInt16(data[45 + $0 * 2]) << 8) }
        XCTAssertEqual(values, [-32_767, -16_383, 0, 16_383, 32_767, 0])
    }

    func testDeletedCloneReferenceDoesNotCallTheExecutorOrCreateOutputDirectories() async throws {
        let folder = try directory()
        let reference = folder.appendingPathComponent("reference.wav")
        try Data("reference".utf8).write(to: reference)
        let output = folder.appendingPathComponent("not-created/output.wav")
        let plan = try SpeechSynthesisPlan(request: TTSRequest(text: "hello", voiceMode: .clone,
            cloneReference: TTSCloneReference(audioURL: reference, transcript: "reference text"), outputURL: output))
        try FileManager.default.removeItem(at: reference)
        let probe = SynthesisProbe()
        do {
            _ = try await SpeechSynthesisOperation.execute(plan, executor: fixture(probe: probe))
            XCTFail("Deleted reference was accepted")
        } catch SpeechSynthesisError.invalidInput(let field, _) {
            XCTAssertEqual(field, .cloneReference)
        }
        let calls = await probe.requests
        XCTAssertTrue(calls.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.deletingLastPathComponent().path))
    }

    func testFailureAndEmptyAudioPreserveExistingOutput() async throws {
        let folder = try directory()
        let output = folder.appendingPathComponent("existing.wav")
        let original = Data("existing audio".utf8)
        try original.write(to: output)
        let plan = try SpeechSynthesisPlan(request: TTSRequest(text: "hello", outputURL: output))
        for executor in [SynthesisFixture(generate: { _ in throw SynthesisFixtureError.expected }),
                         SynthesisFixture(generate: { _ in try AudioWaveform(interleaved: [], channels: 1, sampleRate: 24_000) })] {
            do {
                _ = try await SpeechSynthesisOperation.execute(plan, executor: executor)
                XCTFail("Invalid executor result succeeded")
            } catch {}
            XCTAssertEqual(try Data(contentsOf: output), original)
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path), ["existing.wav"])
        }
    }

    func testCancellationBeforeExecutionDoesNotCallTheExecutor() async throws {
        let output = try directory().appendingPathComponent("output.wav")
        let plan = try SpeechSynthesisPlan(request: TTSRequest(text: "hello", outputURL: output))
        let probe = SynthesisProbe()
        let executor = fixture(probe: probe)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await SpeechSynthesisOperation.execute(plan, executor: executor)
        }
        do { _ = try await task.value; XCTFail("Cancelled operation succeeded") } catch is CancellationError {}
        let calls = await probe.requests
        XCTAssertTrue(calls.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
    }

    func testCancellationAfterUncooperativeGenerationDoesNotPublish() async throws {
        let output = try directory().appendingPathComponent("output.wav")
        let original = Data("previous audio".utf8)
        try original.write(to: output)
        let plan = try SpeechSynthesisPlan(request: TTSRequest(text: "hello", outputURL: output))
        let gate = SynthesisGate()
        let executor = SynthesisFixture(generate: { _ in
            await gate.wait()
            return try AudioWaveform(interleaved: [0.5], channels: 1, sampleRate: 24_000)
        })
        let task = Task { try await SpeechSynthesisOperation.execute(plan, executor: executor) }
        try await waitUntil { await gate.entered }
        task.cancel()
        await gate.open()
        do { _ = try await task.value; XCTFail("Cancelled audio was published") } catch is CancellationError {}
        XCTAssertEqual(try Data(contentsOf: output), original)
    }

    func testStreamingPublishesFinalizedFloatAudioBeforeCompletion() async throws {
        let output = try directory().appendingPathComponent("stream.wav")
        let plan = try streamingPlan(output)
        let events: [TTSStreamingEvent] = [.token(id: 7), .audioChunk(samples: [2, -2], sampleRate: 24_000),
            .audioChunk(samples: [0.25], sampleRate: 24_000), .completed(result: TTSResult(audioURL: output, duration: 3.0 / 24_000))]
        var names: [String] = []
        for try await event in try SpeechSynthesisOperation.stream(plan, executor: eventFixture(events)) {
            switch event {
            case .token: names.append("token")
            case .audioChunk: names.append("audio")
            case .completed(let result):
                names.append("completed")
                let data = try Data(contentsOf: output)
                XCTAssertEqual(read32(data, 40), 12)
                XCTAssertEqual(data[20], 3)
                XCTAssertEqual(data[34], 32)
                XCTAssertEqual((0..<3).map { Float(bitPattern: read32(data, 44 + $0 * 4)) }, [2, -2, 0.25])
                XCTAssertEqual(result.duration, 3.0 / 24_000)
            }
        }
        XCTAssertEqual(names, ["token", "audio", "audio", "completed"])
    }

    func testMalformedAndFailedStreamsPreserveExistingOutputAndRemoveTemporaryFiles() async throws {
        for variant in 0..<5 {
            let folder = try directory()
            let output = folder.appendingPathComponent("existing.wav")
            let original = Data("existing audio".utf8)
            try original.write(to: output)
            let completed = TTSStreamingEvent.completed(result: TTSResult(audioURL: output, duration: 1.0 / 24_000))
            let first = TTSStreamingEvent.audioChunk(samples: [0.5], sampleRate: 24_000)
            let cases: [[TTSStreamingEvent]] = [
                [first],
                [first, .audioChunk(samples: [0.5], sampleRate: 48_000), completed],
                [first, completed, .token(id: 1)],
                [first, .completed(result: TTSResult(audioURL: output, duration: 0, sampleRate: 48_000))],
                [first, completed]
            ]
            let executor = eventFixture(cases[variant], failure: variant == 4)
            do {
                for try await _ in try SpeechSynthesisOperation.stream(streamingPlan(output), executor: executor) {}
                XCTFail("Malformed stream succeeded")
            } catch {}
            XCTAssertEqual(try Data(contentsOf: output), original)
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path), ["existing.wav"])
        }
    }

    func testCancellingStreamConsumerStopsProducerAndDiscardsPartialOutput() async throws {
        let folder = try directory()
        let output = folder.appendingPathComponent("existing.wav")
        let original = Data("existing audio".utf8)
        try original.write(to: output)
        let plan = try streamingPlan(output)
        let probe = SynthesisProbe()
        let executor = SynthesisFixture(stream: { _, _ in
            AsyncThrowingStream { continuation in
                let producer = Task {
                    do {
                        continuation.yield(.audioChunk(samples: [0.5], sampleRate: 24_000))
                        try await Task.sleep(for: .seconds(30))
                        XCTFail("Producer was not cancelled")
                        continuation.finish()
                    } catch {
                        await probe.markStopped()
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { @Sendable reason in
                    if case .cancelled = reason { producer.cancel() }
                }
            }
        })
        let consumer = Task {
            for try await event in try SpeechSynthesisOperation.stream(plan, executor: executor) {
                if case .audioChunk = event { await probe.markReceived() }
                if case .completed = event { XCTFail("Cancelled stream emitted success") }
            }
            try Task.checkCancellation()
        }
        try await waitUntil { await probe.received }
        consumer.cancel()
        do { try await consumer.value; XCTFail("Cancelled consumer succeeded") } catch is CancellationError {}
        try await waitUntil { await probe.stopped }
        try await waitUntil {
            (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) == ["existing.wav"]
        }
        XCTAssertEqual(try Data(contentsOf: output), original)
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func streamingPlan(_ output: URL) throws -> SpeechSynthesisPlan {
        try SpeechSynthesisPlan(request: TTSRequest(text: "hello", outputURL: output), streamingOptions: TTSStreamingOptions())
    }

    private func fixture(probe: SynthesisProbe) -> SynthesisFixture {
        SynthesisFixture(generate: { request in
            await probe.record(request)
            return try AudioWaveform(interleaved: [0.25], channels: 1, sampleRate: 24_000)
        })
    }

    private func eventFixture(_ events: [TTSStreamingEvent], failure: Bool = false) -> SynthesisFixture {
        SynthesisFixture(stream: { _, _ in
            AsyncThrowingStream { continuation in
                for event in events { continuation.yield(event) }
                if failure { continuation.finish(throwing: SynthesisFixtureError.expected) }
                else { continuation.finish() }
            }
        })
    }

    private func read32(_ data: Data, _ offset: Int) -> UInt32 {
        (0..<4).reduce(0) { $0 | UInt32(data[offset + $1]) << ($1 * 8) }
    }

    private func waitUntil(_ predicate: @escaping @Sendable () async -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(3)
        while !(await predicate()) {
            guard ContinuousClock.now < deadline else { throw SynthesisFixtureError.timeout }
            try await Task.sleep(for: .milliseconds(5))
        }
    }
}

private enum SynthesisFixtureError: Error { case expected, timeout }

private struct SynthesisFixture: SpeechSynthesisExecutor {
    var generate: @Sendable (TTSRequest) async throws -> AudioWaveform = { _ in throw SynthesisFixtureError.expected }
    var stream: @Sendable (TTSRequest, TTSStreamingOptions) -> AsyncThrowingStream<TTSStreamingEvent, Error> = { _, _ in
        AsyncThrowingStream { $0.finish(throwing: SynthesisFixtureError.expected) }
    }

    func generate(_ request: TTSRequest, progressHandler: (@Sendable (TTSProgress) -> Void)?) async throws -> AudioWaveform {
        try await generate(request)
    }
    func generateStream(_ request: TTSRequest, options: TTSStreamingOptions) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        stream(request, options)
    }
}

private actor SynthesisProbe {
    private(set) var requests: [TTSRequest] = []
    private(set) var received = false
    private(set) var stopped = false
    func record(_ request: TTSRequest) { requests.append(request) }
    func markReceived() { received = true }
    func markStopped() { stopped = true }
}

private actor SynthesisGate {
    private(set) var entered = false
    private var continuation: CheckedContinuation<Void, Never>?
    func wait() async {
        entered = true
        await withCheckedContinuation { continuation = $0 }
    }
    func open() { continuation?.resume(); continuation = nil }
}
