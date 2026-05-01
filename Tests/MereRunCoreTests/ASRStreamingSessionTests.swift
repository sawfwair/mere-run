import XCTest
@testable import MereRunCore
import AudioCore
@testable import AudioSTT

final class ASRStreamingSessionTests: XCTestCase {
    func testSessionEmitsPartialsOnlyOnChangedTextAndFinalOnce() async throws {
        let outputs = DecodeOutputSequence(
            outputs: [
                Qwen3ASRStreamingDecodeOutput(
                    result: ASRResult(text: "hello", language: "en", duration: 0.1),
                    tokensGenerated: 5
                ),
                Qwen3ASRStreamingDecodeOutput(
                    result: ASRResult(text: "hello", language: "en", duration: 0.2),
                    tokensGenerated: 6
                ),
                Qwen3ASRStreamingDecodeOutput(
                    result: ASRResult(text: "hello world", language: "en", duration: 0.3),
                    tokensGenerated: 7
                ),
            ]
        )

        let session = Qwen3ASRStreamingSession(
            request: ASRStreamingRequest(
                language: "en",
                task: .transcribe,
                maxTokens: 128,
                sampleRate: 16_000,
                decodeIntervalMs: 100,
                minDecodeAudioMs: 100
            ),
            decode: { _ in
                await outputs.next()
            }
        )

        let eventsTask = Task { () throws -> [ASRStreamingEvent] in
            var events: [ASRStreamingEvent] = []
            for try await event in session.events {
                events.append(event)
            }
            return events
        }

        let chunk = Array(repeating: Float(0), count: 1_600)
        try await session.feed(samples: chunk)
        try await session.feed(samples: chunk)
        try await session.finish()

        let events = try await eventsTask.value

        let partials = events.compactMap { event -> String? in
            guard case .partial(let text) = event else { return nil }
            return text
        }
        XCTAssertEqual(partials, ["hello", "hello world"])

        let finals = events.compactMap { event -> ASRResult? in
            guard case .final(let result) = event else { return nil }
            return result
        }
        XCTAssertEqual(finals.count, 1)
        XCTAssertEqual(finals.first?.text, "hello world")

        let stats = events.compactMap { event -> ASRStreamingStats? in
            guard case .stats(let value) = event else { return nil }
            return value
        }
        XCTAssertEqual(stats.map { $0.decodeCount }, [1, 2, 3])
        XCTAssertEqual(stats.last?.tokensGenerated, 7)
    }

    func testFeedAfterFinishThrowsInvalidState() async throws {
        let outputs = DecodeOutputSequence(
            outputs: [
                Qwen3ASRStreamingDecodeOutput(
                    result: ASRResult(text: "done", language: "en", duration: 0.2),
                    tokensGenerated: 3
                )
            ]
        )

        let session = Qwen3ASRStreamingSession(
            request: ASRStreamingRequest(
                language: "en",
                task: .transcribe,
                maxTokens: 128,
                sampleRate: 16_000,
                decodeIntervalMs: 100,
                minDecodeAudioMs: 100
            ),
            decode: { _ in
                await outputs.next()
            }
        )

        let eventsTask = Task {
            for try await _ in session.events {}
        }

        let chunk = Array(repeating: Float(0), count: 1_600)
        try await session.feed(samples: chunk)
        try await session.finish()
        _ = try await eventsTask.value

        do {
            try await session.feed(samples: chunk)
            XCTFail("Expected invalid state when feeding after finish")
        } catch let error as ASRStreamingError {
            guard case .invalidState = error else {
                XCTFail("Unexpected streaming error: \(error)")
                return
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }
}

private actor DecodeOutputSequence {
    private let outputs: [Qwen3ASRStreamingDecodeOutput]
    private var index = 0

    init(outputs: [Qwen3ASRStreamingDecodeOutput]) {
        self.outputs = outputs
    }

    func next() -> Qwen3ASRStreamingDecodeOutput {
        guard !outputs.isEmpty else {
            return Qwen3ASRStreamingDecodeOutput(
                result: ASRResult(text: "", duration: 0),
                tokensGenerated: 0
            )
        }
        let value = outputs[min(index, outputs.count - 1)]
        index += 1
        return value
    }
}
