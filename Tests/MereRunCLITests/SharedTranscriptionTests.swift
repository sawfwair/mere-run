import Foundation
import XCTest
import AudioCore
import AudioSTT
import MereRunCore
@testable import MereRunCLI

final class SharedTranscriptionTests: XCTestCase {
    func testCLIAndAPIResolveEquivalentPlansForBothNativeBackends() async throws {
        let audio = try audioFixture()
        let cases: [(String, ASRBackend)] = [
            (Qwen3ASRResources.defaultModelId, .qwen),
            (ParakeetResources.defaultModelId, .parakeet)
        ]
        for (modelID, backend) in cases {
            let request = ASRRequest(audioURL: audio, language: "en", maxTokens: 32)
            let cli = try await CLIASRRouting.transcribe(
                request: request, preferredBackend: backend, modelOverride: modelID,
                executor: SharedTranscriptionProbe()
            )
            let api = try APITranscription.plan(
                audioURL: audio,
                options: .init(modelID: modelID, language: "en", responseFormat: "json", task: .transcribe, maxTokens: 32)
            )
            XCTAssertEqual(cli.plan, api)
        }
    }

    func testTranslationSwitchesBothBackendAndBuiltInModelIdentity() throws {
        let audio = try audioFixture()
        let request = ASRRequest(audioURL: audio, language: "en", task: .translate)
        let plan = try SpeechTranscriptionResolver.resolve(
            request: request, preferredBackend: .parakeet, modelOverride: ParakeetResources.defaultModelId
        )
        XCTAssertEqual(plan.decision.backend, .qwen)
        XCTAssertEqual(plan.modelID, Qwen3ASRResources.defaultModelId)
        let api = try APITranscription.plan(
            audioURL: audio,
            options: .init(modelID: ParakeetResources.defaultModelId, language: "en", responseFormat: "json", task: .translate, maxTokens: 448)
        )
        XCTAssertEqual(api, plan)
    }

    func testAPIDefaultTranscriptionAliasCanTranslateWithQwen() throws {
        let audio = try audioFixture()
        let form = MultipartFormData(parts: [
            .init(name: "model", filename: nil, contentType: nil, body: Data("whisper-1".utf8)),
            .init(name: "task", filename: nil, contentType: nil, body: Data("translate".utf8))
        ])
        let options = try APIServerContract.transcriptionPlan(from: form)
        let plan = try APITranscription.plan(audioURL: audio, options: options)
        XCTAssertEqual(plan.decision.backend, .qwen)
        XCTAssertEqual(plan.modelID, Qwen3ASRResources.defaultModelId)
    }

    func testLocalParakeetPathCannotBePassedToQwenTranslation() throws {
        let audio = try audioFixture()
        let root = try temporaryDirectory()
        try Data("{\"target\":\"parakeet\",\"preprocessor\":{},\"encoder\":{}}".utf8).write(to: root.appendingPathComponent("config.json"))
        XCTAssertThrowsError(try SpeechTranscriptionResolver.resolve(
            request: .init(audioURL: audio, task: .translate), preferredBackend: .auto, modelOverride: root.path
        )) { error in
            XCTAssertEqual((error as? SpeechTranscriptionIssue)?.code, "model_backend_mismatch")
        }
    }

    func testCoreMLProviderRejectsRoutingToQwenBeforeExecution() throws {
        let audio = try audioFixture()
        let root = try temporaryDirectory()
        XCTAssertThrowsError(try SpeechTranscriptionResolver.resolve(
            request: .init(audioURL: audio, task: .translate), preferredBackend: .auto,
            parakeetExecutionProvider: .coreML(artifactURL: root)
        )) { error in
            XCTAssertEqual((error as? SpeechTranscriptionIssue)?.code, "incompatible_execution_provider")
        }
    }

    func testLocalModelOverrideIsSharedByCLIAndAPI() async throws {
        let audio = try audioFixture()
        let root = try temporaryDirectory()
        let cli = try await CLIASRRouting.transcribe(
            request: .init(audioURL: audio, language: "en"), preferredBackend: .auto,
            modelOverride: root.path, executor: SharedTranscriptionProbe()
        )
        let api = try APITranscription.plan(
            audioURL: audio,
            options: .init(modelID: root.path, language: "en", responseFormat: "text", task: .transcribe, maxTokens: 448)
        )
        XCTAssertEqual(api, cli.plan)
        XCTAssertEqual(api.modelPath, root.path)
    }

    func testServingUsesOneRequestSlotAndKeepsTheBorrowedRuntimeWarm() async throws {
        let audio = try audioFixture()
        let executor = SharedTranscriptionProbe()
        let services = try services(executor: executor)
        let plan = try SpeechTranscriptionResolver.resolve(request: .init(audioURL: audio), preferredBackend: .qwen)
        for _ in 0..<2 {
            let result = try await services.transcribe(plan)
            XCTAssertEqual(result.result.text, "resident transcript")
        }
        let requestStatus = await services.admission.snapshot()
        XCTAssertEqual(requestStatus.activeRequests, 0)
        XCTAssertEqual(requestStatus.totalAdmittedRequests, 2)
        XCTAssertEqual(requestStatus.totalCompletedRequests, 2)
        let resident = await executor.state()
        XCTAssertEqual(resident.loadCount, 1)
        XCTAssertEqual(resident.completedRequests, 2)
        XCTAssertTrue(resident.ready)
    }

    func testServingReleasesSlotsOnTranscriptionFailureAndCancellation() async throws {
        let audio = try audioFixture()
        for cancelled in [false, true] {
            let services = try services(executor: SharedTranscriptionProbe(cancelled: cancelled, fail: true))
            let plan = try SpeechTranscriptionResolver.resolve(request: .init(audioURL: audio), preferredBackend: .qwen)
            do {
                _ = try await services.transcribe(plan)
                XCTFail("Expected transcription failure")
            } catch is CancellationError {
                XCTAssertTrue(cancelled)
            } catch SharedTranscriptionFailure.expected {
                XCTAssertFalse(cancelled)
            }
            let snapshot = await services.admission.snapshot()
            XCTAssertEqual(snapshot.activeRequests, 0)
            XCTAssertEqual(snapshot.queuedRequests, 0)
            XCTAssertEqual(snapshot.totalAdmittedRequests, 1)
            XCTAssertEqual(snapshot.totalCancelledRequests, cancelled ? 1 : 0)
        }
    }

    func testServingModelMaintenanceUsesScopedAdmission() async throws {
        let services = try services(executor: SharedTranscriptionProbe())
        let loaded = try await services.loadModel(id: "fixture.gguf")
        XCTAssertTrue(loaded.loaded)
        XCTAssertEqual(loaded.activeRequests, 0)
        let unloaded = try await services.unloadModel(id: "fixture.gguf")
        XCTAssertFalse(unloaded.loaded)
        let snapshot = await services.admission.snapshot()
        XCTAssertEqual(snapshot.activeRequests, 0)
        XCTAssertEqual(snapshot.totalAdmittedRequests, 2)
        XCTAssertEqual(snapshot.totalCompletedRequests, 2)
    }

    func testNativeExecutorCannotUseCoreMLForQwenWithoutTheResolver() async throws {
        let audio = try audioFixture()
        let root = try temporaryDirectory()
        let executor = NativeSpeechTranscriptionExecutor(parakeetExecutionProvider: .coreML(artifactURL: root))
        do {
            _ = try await executor.transcribeQwen(
                request: .init(audioURL: audio), modelID: Qwen3ASRResources.defaultModelId,
                modelPath: nil, progressHandler: nil
            )
            XCTFail("Core ML provider reached Qwen execution")
        } catch let issue as SpeechTranscriptionIssue {
            XCTAssertEqual(issue.code, "incompatible_execution_provider")
        }
    }

    func testTranscriptionCommandRejectsNonpositiveTokenLimits() throws {
        for value in ["0", "-1"] {
            XCTAssertThrowsError(try SpeechTranscribe.parse(["audio.wav", "--max-tokens", value]))
        }
    }

    private func services(executor: any SpeechTranscriptionExecutor) throws -> RuntimeServingServices {
        let root = try temporaryDirectory()
        let model = root.appendingPathComponent("fixture.gguf")
        try Data().write(to: model)
        let models = RuntimeModelPool(
            defaultModelID: "fixture.gguf", defaultEngine: .textCode, startupModelPath: model.path,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root), ensureMLXAvailable: {},
            prepareLoadedModel: { _ in }, unloadLoadedModel: { _ in }, clearMLXCache: {}
        )
        return RuntimeServingServices(
            models: models, media: APISidecarModelPool(settingsURL: root.appendingPathComponent("settings.json")),
            admission: RuntimeRequestAdmission(maxActiveRequests: 1), ensureAvailable: {}, transcriptionExecutor: executor
        )
    }

    private func audioFixture() throws -> URL {
        let audio = try temporaryDirectory().appendingPathComponent("audio.wav")
        try Data().write(to: audio)
        return audio
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}

private enum SharedTranscriptionFailure: Error { case expected }

private actor SharedTranscriptionProbe: SpeechTranscriptionExecutor {
    private let slot = ResidentRuntimeSlot<String, Int>()
    private let cancelled: Bool
    private let fail: Bool

    init(cancelled: Bool = false, fail: Bool = false) {
        self.cancelled = cancelled
        self.fail = fail
    }

    func transcribeQwen(
        request: ASRRequest, modelID: String, modelPath: String?, progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        try await transcribe(modelID: modelID)
    }

    func transcribeParakeet(
        request: ASRRequest, modelID: String, modelPath: String?, progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        try await transcribe(modelID: modelID)
    }

    func state() async -> ResidentRuntimeSlotState<String> { await slot.state() }

    private func transcribe(modelID: String) async throws -> ASRResult {
        let cancelled = cancelled
        let fail = fail
        return try await slot.withValue(for: modelID, make: { 1 }, unload: { _ in }) { _ in
            if cancelled { throw CancellationError() }
            if fail { throw SharedTranscriptionFailure.expected }
            return ASRResult(text: "resident transcript")
        }
    }
}
