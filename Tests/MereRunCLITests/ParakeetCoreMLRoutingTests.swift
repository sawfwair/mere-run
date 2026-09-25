import ArgumentParser
import AudioCore
import AudioSTT
import Foundation
import XCTest
@testable import MereRunCLI

final class ParakeetCoreMLRoutingTests: XCTestCase {
    func testCoreMLTranscriptionWithoutLanguageHintUsesParakeet() async throws {
        let result = try await CLIASRRouting.transcribe(
            request: ASRRequest(audioURL: try audioFixture()),
            preferredBackend: .parakeet,
            parakeetExecutionProvider: .coreML(artifactURL: URL(fileURLWithPath: "/tmp/coreml")),
            executor: CoreMLRoutingProbe()
        )
        XCTAssertEqual(result.backend, .parakeet)
        XCTAssertEqual(result.result.text, "Parakeet")
    }

    func testDefaultProviderRetainsQwenLanguageFallback() async throws {
        let result = try await CLIASRRouting.transcribe(
            request: ASRRequest(audioURL: try audioFixture(), language: "zz"),
            preferredBackend: .parakeet,
            executor: CoreMLRoutingProbe()
        )
        XCTAssertEqual(result.backend, .qwen)
    }

    func testExplicitCoreMLProviderRejectsLanguageThatRoutesToQwen() async throws {
        let executor = CoreMLRoutingProbe()
        do {
            _ = try await CLIASRRouting.transcribe(
                request: ASRRequest(audioURL: try audioFixture(), language: "zz"),
                preferredBackend: .parakeet,
                parakeetExecutionProvider: .coreML(artifactURL: URL(fileURLWithPath: "/tmp/coreml")),
                executor: executor
            )
            XCTFail("Expected an explicit provider error before executing Qwen")
        } catch let error as ValidationError {
            XCTAssertTrue(error.description.contains("--provider coreml"))
        }
        let calls = await executor.calls
        XCTAssertEqual(calls, 0)
    }

    private func audioFixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let audio = root.appendingPathComponent("audio.wav")
        try Data().write(to: audio)
        return audio
    }
}

private actor CoreMLRoutingProbe: CLIASRTranscriptionExecutor {
    private(set) var calls = 0

    func transcribeQwen(
        request: ASRRequest,
        modelID: String,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        calls += 1
        return ASRResult(text: "unexpected Qwen")
    }

    func transcribeParakeet(
        request: ASRRequest,
        modelID: String,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        calls += 1
        return ASRResult(text: "Parakeet")
    }
}
