import ArgumentParser
import AudioCore
import AudioSTT
import Foundation
import XCTest
@testable import MereRunCLI
import MereRunCore

final class NemotronTranscribeCommandTests: XCTestCase {
    func testParsesOfflineBackendAndTranslation() throws {
        let command = try SpeechTranscribe.parse(["audio.wav", "--backend", "nemotron", "--task", "translate", "--no-timestamps"])
        XCTAssertEqual(command.backend.backend, .nemotron)
        XCTAssertEqual(command.task.task, .translate)
    }

    func testCapabilityGateSupportsOfflineTranscriptionAndTranslation() throws {
        for flags in [["--backend", "nemotron"], ["--model", NemotronTranscribeResources.modelID, "--task", "translate"],
                      ["--model", NemotronTranscribeResources.bf16ModelID, "--task", "translate"]] {
            let report = try XCTUnwrap(CLICapabilityGate.evaluate(commandLine: ["speech", "transcribe", "audio.wav", "--no-timestamps"] + flags)).report
            XCTAssertEqual(report.family, "nemotron-transcribe")
            XCTAssertEqual(report.violations, [])
        }
        let stream = try XCTUnwrap(CLICapabilityGate.evaluate(commandLine: ["speech", "transcribe", "audio.wav", "--backend", "nemotron", "--stream"])).report
        XCTAssertFalse(stream.violations.isEmpty)
    }

    func testRejectsStreamingBeforeLoading() {
        for selection in [["--backend", "nemotron"], ["--model", NemotronTranscribeResources.modelID],
                          ["--model", NemotronTranscribeResources.bf16ModelID]] {
            XCTAssertThrowsError(try SpeechTranscribe.parse(["audio.wav", "--stream"] + selection))
        }
    }

    func testOperationDispatchesSelectedNemotronTranslation() async throws {
        let audio = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        try Data().write(to: audio)
        defer { try? FileManager.default.removeItem(at: audio) }
        let outcome = try await CLIASRRouting.transcribe(
            request: ASRRequest(audioURL: audio, language: "German", task: .translate),
            preferredBackend: .auto, modelOverride: NemotronTranscribeResources.modelID, executor: NemotronProbe()
        )
        XCTAssertEqual(outcome.backend, .nemotron)
        XCTAssertEqual(outcome.plan.modelID, NemotronTranscribeResources.modelID)
        XCTAssertEqual(outcome.result.text, "Native Nemotron translation")
    }
}

private struct NemotronProbe: SpeechTranscriptionExecutor {
    func transcribeNemotron(request: ASRRequest, modelID: String, modelPath: String?,
                            progressHandler: (@Sendable (ASRProgress) -> Void)?) async throws -> ASRResult {
        XCTAssertEqual(request.task, .translate)
        return ASRResult(text: "Native Nemotron translation", language: "en")
    }
    func transcribeQwen(request: ASRRequest, modelID: String, modelPath: String?,
                        progressHandler: (@Sendable (ASRProgress) -> Void)?) async throws -> ASRResult {
        XCTFail("Unexpected Qwen dispatch")
        return ASRResult(text: "")
    }
    func transcribeParakeet(request: ASRRequest, modelID: String, modelPath: String?,
                            progressHandler: (@Sendable (ASRProgress) -> Void)?) async throws -> ASRResult {
        XCTFail("Unexpected Parakeet dispatch")
        return ASRResult(text: "")
    }
}
