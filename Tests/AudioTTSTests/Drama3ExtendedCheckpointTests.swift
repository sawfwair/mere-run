import AudioCodecs
import AudioCore
import AudioTTS
import Foundation
import MLX
import MereRunAdmission
import MereRunCore
import XCTest

/// One explicitly selected released-checkpoint case per process, with shared admission.
final class Drama3ExtendedCheckpointTests: XCTestCase {
    private enum Probe: String {
        case paragraph, clone, stream, cloneStream, cancel

        var text: String {
            switch self {
            case .paragraph, .cancel:
                return "Today we are testing a longer paragraph on this laptop. The native speech engine should read every sentence clearly, preserve the pauses between ideas, and finish without losing the final words. After walking through the garden, we stopped beside the river and watched the evening light move across the water. This is the end of the recording."
            case .clone, .cloneStream:
                return "The reference voice is now reading a different sentence, with clear words and a steady pace."
            case .stream:
                return "[excited] Hello! The native speech engine is running on this laptop."
            }
        }

        var isStreaming: Bool { self == .stream || self == .cloneStream || self == .cancel }
        var isClone: Bool { self == .clone || self == .cloneStream }
    }

    func testReleasedCheckpointExtendedPath() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_TEST_DRAMA3_ROOT"],
              let directory = environment["MERERUN_TEST_DRAMA3_EVIDENCE"],
              let name = environment["MERERUN_TEST_DRAMA3_CASE"], let probe = Probe(rawValue: name) else {
            throw XCTSkip("Set the DRAMA3 root, evidence directory, and explicit extended case to qualify a checkpoint.")
        }
        let evidence = URL(fileURLWithPath: directory, isDirectory: true)
        try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
        var request = TTSRequest(text: probe.text, voiceDescription: "", temperature: 0.6, seed: 42,
            outputURL: evidence.appendingPathComponent("native-speech.wav"))
        if probe.isClone {
            let reference = try XCTUnwrap(environment["MERERUN_TEST_DRAMA3_REFERENCE"])
            request.voiceMode = .clone
            request.cloneReference = TTSCloneReference(audioURL: URL(fileURLWithPath: reference),
                transcript: "Hello! The native speech engine is running on this laptop.", language: "en")
        }
        let options: TTSStreamingOptions? = probe.isStreaming ? .init(chunkTokenInterval: 25) : nil
        let plan = try SpeechSynthesisPlan(request: request, streamingOptions: options)
        let coordinator = MachineInferenceCoordinator(stateDirectory:
            MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true))
        let lease = try await coordinator.acquire(.init(label: "DRAMA-3 extended checkpoint: " + probe.rawValue,
            resourceClass: .standard), onWait: { _ in
                FileHandle.standardError.write(Data("DRAMA-3 extended probe is waiting for shared inference capacity.\n".utf8))
            })
        defer { lease.release() }
        #if os(macOS)
        GPU.resetPeakMemory()
        #endif
        let generator = Drama3Generator()
        let executor = Drama3SynthesisExecutor(generator: generator, modelPath: root)
        let trace = Trace(url: evidence.appendingPathComponent("native-events.json"))
        do {
            if probe == .cancel {
                try await cancelAfterFirstChunk(plan, executor: executor, generator: generator, trace: trace)
            } else if probe.isStreaming {
                try await stream(plan, executor: executor, trace: trace, comparison: environment["MERERUN_TEST_DRAMA3_COMPARISON"])
            } else {
                let outcome = try await SpeechSynthesisOperation.execute(plan, executor: executor,
                    progressHandler: { trace.append(stage: $0.stage.rawValue, frames: $0.tokensGenerated) })
                let decoding = try XCTUnwrap(trace.snapshot().first { $0.stage == TTSStage.decoding.rawValue })
                XCTAssertLessThan(decoding.activeMLXBytes, 1_073_741_824)
                if probe == .paragraph { XCTAssertGreaterThan(outcome.result.duration, 10) }
                trace.append(stage: "completed", samples: Int(outcome.result.duration * Double(outcome.result.sampleRate)))
            }
            await generator.unload()
            trace.append(stage: "unloaded")
            XCTAssertLessThan(Memory.activeMemory, 1_073_741_824, "Completed or cancelled requests must release their graphs.")
            try trace.finish()
        } catch {
            await generator.unload()
            trace.append(stage: "failedAndUnloaded")
            try trace.finish()
            throw error
        }
    }

    private func stream(_ plan: SpeechSynthesisPlan, executor: Drama3SynthesisExecutor,
                        trace: Trace, comparison: String?) async throws {
        var samples: [Float] = []
        var tokens = 0, chunks = 0, completions = 0
        for try await event in try SpeechSynthesisOperation.stream(plan, executor: executor) {
            switch event {
            case .token:
                tokens += 1
                if tokens.isMultiple(of: 25) { trace.append(stage: "tokens", frames: tokens) }
            case .audioChunk(let values, let rate):
                XCTAssertEqual(rate, 44_100)
                XCTAssertFalse(values.isEmpty)
                XCTAssertTrue(values.allSatisfy(\.isFinite))
                XCTAssertEqual(completions, 0)
                samples += values
                chunks += 1
                trace.append(stage: "audioChunk", frames: tokens, samples: values.count)
            case .completed(let result):
                completions += 1
                XCTAssertEqual(result.sampleRate, 44_100)
                XCTAssertEqual(result.duration, Double(samples.count) / 44_100, accuracy: 1e-9)
                XCTAssertTrue(FileManager.default.fileExists(atPath: result.audioURL.path))
                trace.append(stage: "completed", frames: tokens, samples: samples.count)
            }
        }
        XCTAssertEqual(completions, 1)
        XCTAssertGreaterThan(chunks, 1)
        XCTAssertEqual(samples.count, tokens * 2_048)
        XCTAssertGreaterThan(samples.reduce(0.0) { $0 + Double($1 * $1) } / Double(samples.count), 1e-8)
        if let comparison {
            let batch = try AudioReader.readAudioBuffer(from: URL(fileURLWithPath: comparison), sampleRate: 44_100, channels: 1)
            XCTAssertEqual(samples.count, batch.samples.count, "Seeded streaming must preserve the batch acoustic-frame sequence.")
            let error = zip(samples, batch.samples).map { abs($0 - $1) }.max() ?? .infinity
            trace.append(stage: "batchComparison", maximumSampleError: Double(error))
            // Batch WAV is PCM16; streaming keeps float samples. Allow quantization and small kernel-shape rounding.
            XCTAssertLessThan(error, 0.0002, "Causal prefix decoding must agree with the seeded batch waveform.")
        }
    }

    private func cancelAfterFirstChunk(_ plan: SpeechSynthesisPlan, executor: Drama3SynthesisExecutor,
                                      generator: Drama3Generator, trace: Trace) async throws {
        let signal = AsyncStream<Void>.makeStream()
        let consumer = Task {
            defer { signal.continuation.finish() }
            for try await event in try SpeechSynthesisOperation.stream(plan, executor: executor) {
                if case .audioChunk(let samples, _) = event {
                    trace.append(stage: "firstChunkBeforeCancellation", samples: samples.count)
                    signal.continuation.yield(())
                    signal.continuation.finish()
                    try await Task.sleep(for: .seconds(30))
                }
            }
            signal.continuation.finish()
        }
        var iterator = signal.stream.makeAsyncIterator()
        _ = await iterator.next()
        trace.append(stage: "cancellationRequested")
        consumer.cancel()
        do {
            try await consumer.value
            XCTFail("The consumer must observe cancellation.")
        } catch is CancellationError {
            trace.append(stage: "consumerCancelled")
        }
        // This actor hop waits for the producer to observe cancellation and release its graphs.
        await generator.unload()
        trace.append(stage: "producerUnloadedAfterCancellation")
        let requested = try XCTUnwrap(trace.snapshot().first { $0.stage == "cancellationRequested" })
        let released = try XCTUnwrap(trace.snapshot().last)
        XCTAssertLessThan(released.elapsedSeconds - requested.elapsedSeconds, 10, "Cancellation must stop generation promptly.")
        XCTAssertTrue(trace.snapshot().contains { $0.stage == "firstChunkBeforeCancellation" })
        XCTAssertFalse(FileManager.default.fileExists(atPath: plan.request.outputURL.path),
            "Cancellation must not publish a partial final WAV.")
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: plan.request.outputURL.deletingLastPathComponent().path)
        XCTAssertFalse(leftovers.contains { $0.hasPrefix(".audio-export-") && $0.hasSuffix(".tmp") })
    }

    private struct Event: Encodable {
        let stage: String
        let elapsedSeconds: Double
        let frames: Int
        let samples: Int
        let activeMLXBytes: Int
        let cachedMLXBytes: Int
        let peakMLXBytes: Int
        let physicalFootprintBytes: UInt64?
        let maximumSampleError: Double?
    }

    private final class Trace: @unchecked Sendable {
        private let lock = NSLock()
        private let started = Date()
        private let url: URL
        private var events: [Event] = []
        private var writeError: Error?

        init(url: URL) { self.url = url }

        func append(stage: String, frames: Int = 0, samples: Int = 0, maximumSampleError: Double? = nil) {
            lock.lock()
            defer { lock.unlock() }
            events.append(Event(stage: stage, elapsedSeconds: Date().timeIntervalSince(started), frames: frames, samples: samples,
                activeMLXBytes: Memory.activeMemory, cachedMLXBytes: Memory.cacheMemory, peakMLXBytes: Memory.peakMemory,
                physicalFootprintBytes: RuntimeMemorySample.current().physicalFootprintBytes,
                maximumSampleError: maximumSampleError))
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(events).write(to: url, options: .atomic)
            } catch { writeError = error }
        }

        func snapshot() -> [Event] {
            lock.lock()
            defer { lock.unlock() }
            return events
        }

        func finish() throws {
            lock.lock()
            defer { lock.unlock() }
            if let writeError { throw writeError }
        }
    }
}
