import ArgumentParser
import AudioCore
import AudioSTT
import Foundation
import XCTest
@testable import MereRunCLI

final class WhistleCommandTests: XCTestCase {
    func testWhistleStreamingParsesForFileAndRawPCM() throws {
        for selection in [["--backend", "whistle"], ["--model", "speech-asr-whistle"]] {
            _ = try SpeechTranscribe.parse(["audio.wav", "--stream"] + selection)
            _ = try SpeechTranscribe.parse(["-", "--stream", "--input-format", "pcm-s16le", "--sample-rate", "16000", "--jsonl"] + selection)
        }
        _ = try SpeechListen.parse(["--model", "speech-asr-whistle"])
    }

    func testParsesWhistleBackend() throws {
        let command = try SpeechTranscribe.parse(["audio.wav", "--backend", "whistle", "--language", "de", "--no-timestamps"])
        XCTAssertEqual(command.backend.backend, .whistle)
        XCTAssertEqual(command.language, "de")
        XCTAssertFalse(command.timestamps)
    }

    func testWhistleSearchControlsValidateAndParseKeywords() throws {
        let command = try SpeechTranscribe.parse(["audio.wav", "--backend", "whistle", "--beam-size", "3", "--decoder-depth", "4",
                                                 "--keyword", "Siobhan", "--keyword", "Mere Run", "--whistle-weights", "fp32"])
        XCTAssertEqual(command.beamSize, 3)
        XCTAssertEqual(command.decoderDepth, 4)
        XCTAssertEqual(command.keywords, ["Siobhan", "Mere Run"])
        for flags in [["--beam-size", "0"], ["--decoder-depth", "9"], ["--whistle-weights", "bad"], ["--keyword", " "]] {
            XCTAssertThrowsError(try SpeechTranscribe.parse(["audio.wav", "--backend", "whistle"] + flags))
        }
    }

    func testSharedOperationDispatchesWhistleWithoutLoadingOtherBackends() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let audio = root.appendingPathComponent("audio.wav")
        try Data().write(to: audio)
        let outcome = try await CLIASRRouting.transcribe(
            request: ASRRequest(audioURL: audio), preferredBackend: .whistle, executor: WhistleProbe()
        )
        XCTAssertEqual(outcome.backend, .whistle)
        XCTAssertEqual(outcome.plan.modelID, WhistleGenerator.modelID)
        XCTAssertEqual(outcome.result.text, "native Whistle")
    }
}

private struct WhistleProbe: SpeechTranscriptionExecutor {
    func transcribeWhistle(
        request: ASRRequest, modelID: String, modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        ASRResult(text: "native Whistle", language: "en")
    }

    func transcribeQwen(
        request: ASRRequest, modelID: String, modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        XCTFail("Unexpected Qwen dispatch")
        return ASRResult(text: "")
    }

    func transcribeParakeet(
        request: ASRRequest, modelID: String, modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        XCTFail("Unexpected Parakeet dispatch")
        return ASRResult(text: "")
    }
}
