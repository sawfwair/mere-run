import XCTest
@testable import MereRunCore
import AudioCore

final class GenerationStreamingTypesTests: XCTestCase {
    func testASRStreamingRequestDefaultsAndCodableRoundTrip() throws {
        let request = ASRStreamingRequest()

        XCTAssertNil(request.language)
        XCTAssertEqual(request.task, .transcribe)
        XCTAssertEqual(request.maxTokens, 448)
        XCTAssertEqual(request.sampleRate, 16_000)
        XCTAssertEqual(request.decodeIntervalMs, 500)
        XCTAssertEqual(request.minDecodeAudioMs, 800)

        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(ASRStreamingRequest.self, from: data)
        XCTAssertEqual(decoded, request)
    }

    func testTTSStreamingOptionsDefaultsAndCodableRoundTrip() throws {
        let options = TTSStreamingOptions()

        XCTAssertEqual(options.chunkTokenInterval, 25)
        XCTAssertTrue(options.emitTokenEvents)

        let data = try JSONEncoder().encode(options)
        let decoded = try JSONDecoder().decode(TTSStreamingOptions.self, from: data)
        XCTAssertEqual(decoded, options)
    }

    func testStreamingEventsAreCodableAndHashable() throws {
        let asrEvent = ASRStreamingEvent.final(result: ASRResult(text: "hello", language: "en", duration: 1.0))
        let asrData = try JSONEncoder().encode(asrEvent)
        let asrDecoded = try JSONDecoder().decode(ASRStreamingEvent.self, from: asrData)
        XCTAssertEqual(asrDecoded, asrEvent)

        let ttsEvent = TTSStreamingEvent.completed(
            result: TTSResult(
                audioURL: URL(fileURLWithPath: "/tmp/out.wav"),
                duration: 1.25,
                sampleRate: 24_000
            )
        )
        let ttsData = try JSONEncoder().encode(ttsEvent)
        let ttsDecoded = try JSONDecoder().decode(TTSStreamingEvent.self, from: ttsData)
        XCTAssertEqual(ttsDecoded, ttsEvent)

        let asrSet: Set<ASRStreamingEvent> = [asrEvent, asrDecoded]
        let ttsSet: Set<TTSStreamingEvent> = [ttsEvent, ttsDecoded]
        XCTAssertEqual(asrSet.count, 1)
        XCTAssertEqual(ttsSet.count, 1)
    }

    func testDefaultASRStreamingUnsupportedBackend() async {
        let generator = UnsupportedASRGenerator()

        do {
            _ = try await generator.makeStreamingSession(ASRStreamingRequest())
            XCTFail("Expected unsupported backend error")
        } catch let error as ASRStreamingError {
            guard case .unsupportedBackend(let message) = error else {
                XCTFail("Unexpected ASR streaming error: \(error)")
                return
            }
            XCTAssertFalse(message.isEmpty)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testDefaultTTSStreamingUnsupportedBackend() async {
        let generator = UnsupportedTTSGenerator()
        let request = TTSRequest(text: "hello", outputURL: URL(fileURLWithPath: "/tmp/out.wav"))

        do {
            for try await _ in generator.generateStream(request, options: TTSStreamingOptions()) {
                XCTFail("Unsupported stream should not emit events")
            }
            XCTFail("Expected unsupported backend error")
        } catch let error as ASRStreamingError {
            guard case .unsupportedBackend(let message) = error else {
                XCTFail("Unexpected TTS streaming error: \(error)")
                return
            }
            XCTAssertFalse(message.isEmpty)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }
}

private struct UnsupportedASRGenerator: ASRGenerator {
    func transcribe(
        _ request: ASRRequest,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        ASRResult(text: "", duration: 0)
    }
}

private struct UnsupportedTTSGenerator: TTSGenerator {
    func generate(
        _ request: TTSRequest,
        progressHandler: (@Sendable (TTSProgress) -> Void)?
    ) async throws -> TTSResult {
        TTSResult(audioURL: request.outputURL, duration: 0, sampleRate: 24_000)
    }
}
