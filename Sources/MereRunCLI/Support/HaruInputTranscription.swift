#if os(macOS)
import AudioCodecs
import AudioCore
import AudioSTT
import Foundation
import MereRunContract
import MereRunCore

/// Feeds one Parakeet session from the same PCM packets Haru receives. The bounded stream
/// keeps ASR loading, resampling, and decoding off the conversation loop.
struct HaruInputTranscription {
    struct Result: Sendable {
        var segments: [HaruTranscriptSegment] = []
        var error: String?
    }

    private enum InputError: LocalizedError {
        case fellBehind

        var errorDescription: String? { "Microphone transcription fell behind the conversation." }
    }

    private let continuation: AsyncThrowingStream<[Float], Error>.Continuation
    private let worker: Task<Result, Never>

    init() {
        let (packets, continuation) = AsyncThrowingStream<[Float], Error>.makeStream(
            bufferingPolicy: .bufferingOldest(128)
        )
        self.continuation = continuation
        worker = Task { await Self.transcribe(packets) }
    }

    func feed(_ samples: [Float]) {
        if case .dropped = continuation.yield(samples) {
            continuation.finish(throwing: InputError.fellBehind)
        }
    }

    func finish() async -> Result {
        continuation.finish()
        return await worker.value
    }

    func cancel() {
        worker.cancel()
        continuation.finish()
    }

    private static func transcribe(_ packets: AsyncThrowingStream<[Float], Error>) async -> Result {
        guard let model = ModelResolver().resolveIfPresent(.parakeetASR) else {
            let message = "Install Parakeet to transcribe your microphone."
            HaruEventWriter.write(.init(.inputUnavailable, text: message))
            return Result(error: message)
        }
        do {
            let generator = try await CLIParakeetASRLoader.prepare(model: model.rootURL.path, progressHandler: nil)
            let request = ASRStreamingRequest()
            let configuration = ASRLiveConfiguration(
                decodeIntervalMs: 1_000, minDecodeAudioMs: 1_000, silenceMs: 600, maxUtteranceMs: 15_000
            )
            let live = ParakeetASRLiveSession(
                generator: generator, request: request, configuration: configuration, redecodeWholeUtterance: true
            )
            async let collected = collect(live.events)
            do {
                try await feed(packets, to: live)
                try await live.finish(reason: .stopped)
            } catch {
                await live.cancel()
                var result = await collected
                if !Task.isCancelled, result.error == nil {
                    result.error = reportFailure(error)
                }
                return result
            }
            return await collected
        } catch {
            return Result(error: Task.isCancelled ? nil : reportFailure(error))
        }
    }

    private static func feed(
        _ packets: AsyncThrowingStream<[Float], Error>, to live: ParakeetASRLiveSession
    ) async throws {
        var remainder: [Float] = []
        for try await packet in packets {
            try Task.checkCancellation()
            let samples = remainder + packet
            // The 24 → 16 kHz ratio is 3:2. Carry incomplete groups across packet
            // boundaries to prevent cumulative drift in ASR timestamps.
            let count = samples.count - samples.count % 3
            remainder = Array(samples.dropFirst(count))
            if count > 0 {
                try await live.feed(samples: resample(Array(samples.prefix(count))))
            }
        }
        if !remainder.isEmpty {
            try await live.feed(samples: resample(remainder))
        }
    }

    private static func resample(_ samples: [Float]) -> [Float] {
        PCMStreamConverter.resampleLinear(samples, from: HaruDuplexSession.sampleRate, to: 16_000)
    }

    private static func collect(_ events: AsyncThrowingStream<ASRLiveEvent, Error>) async -> Result {
        var result = Result()
        do {
            for try await event in events {
                switch event {
                case .partial(let utterance):
                    HaruEventWriter.write(.init(.inputPartial, text: utterance.text, startMs: utterance.startMs))
                case .commit(let utterance):
                    if !utterance.text.isEmpty {
                        result.segments.append(.init(speaker: .user, startMs: utterance.startMs, text: utterance.text))
                    }
                    HaruEventWriter.write(.init(.inputCommit, text: utterance.text, startMs: utterance.startMs))
                case .stats, .final:
                    break
                }
            }
        } catch {
            if !Task.isCancelled { result.error = reportFailure(error) }
        }
        return result
    }

    private static func reportFailure(_ error: Error) -> String {
        CLIStderr.write("Microphone transcription: \(error.localizedDescription)\n")
        let message = "Microphone transcription stopped. Earlier transcript entries are kept."
        HaruEventWriter.write(.init(.inputUnavailable, text: message))
        return message
    }
}

/// Audio and ASR tasks share stdout. Write each complete JSON line under one lock.
enum HaruEventWriter {
    private static let lock = NSLock()

    static func write(_ event: HaruConversationEvent) {
        lock.withLock {
            do {
                let line = try JSONEncoder().encode(event)
                CLIStdout.write(String(decoding: line, as: UTF8.self) + "\n")
            } catch {
                CLIStderr.write("Could not encode a Haru event: \(error.localizedDescription)\n")
            }
        }
    }
}
#endif
