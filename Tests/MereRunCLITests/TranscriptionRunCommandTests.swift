import AudioCore
import AudioSTT
import Foundation
import MereRunCore
import MereRunExecution
import XCTest
@testable import MereRunCLI

final class TranscriptionRunCommandTests: XCTestCase {
    func testRecordingFlagsAreOptInAndDoNotOverlapLiveStdinOrOutputFiles() throws {
        let root = try root()
        let run = root.appendingPathComponent("run")
        XCTAssertNil(try SpeechTranscribe.parse(["audio.wav"]).runDirectory)
        XCTAssertNil(try APIServe.parse([]).transcriptionRunRecords)
        XCTAssertEqual(try SpeechTranscribe.parse(["audio.wav", "--run-dir", run.path]).runDirectory, run.path)
        XCTAssertEqual(try APIServe.parse(["--transcription-run-records", root.path]).transcriptionRunRecords, root.path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: run.path))
        XCTAssertThrowsError(try SpeechTranscribe.parse(["audio.wav", "--run-dir", run.path, "--stream"]))
        XCTAssertThrowsError(try SpeechTranscribe.parse([
            "-", "--stream", "--input-format", "pcm-s16le", "--sample-rate", "16000", "--jsonl", "--run-dir", run.path
        ]))
        for name in ["transcription-run.json", ".transcription-run.lock", "inputs/audio.wav", "result.json", "transcript.txt"] {
            XCTAssertThrowsError(try SpeechTranscribe.parse([
                "audio.wav", "--run-dir", run.path, "--output", run.appendingPathComponent(name).path
            ]))
        }
    }

    func testCLIAndAPIRecordingKeepEquivalentSettingsAndRetainTemporaryUpload() async throws {
        let fixture = try fixture()
        let request = ASRRequest(audioURL: fixture.audio, language: "en", maxTokens: 29)
        let cli = try session(root: fixture.root, name: "cli", request: request, model: fixture.model)
        let apiRoot = fixture.root.appendingPathComponent("api-runs")
        let cliOutcome = try await CLIASRRouting.transcribe(
            request: request, preferredBackend: .auto, modelOverride: fixture.model.path,
            recording: cli, executor: RecordedTranscriptionExecutor()
        )
        let services = makeServices(root: fixture.root, executor: RecordedTranscriptionExecutor())
        let apiOutcome = try await APITranscription.execute(
            audioURL: fixture.audio,
            options: .init(modelID: fixture.model.path, language: "en", responseFormat: "verbose_json", task: .transcribe, maxTokens: 29),
            recordingRoot: apiRoot, services: services
        )
        let apiDirectories = try FileManager.default.contentsOfDirectory(at: apiRoot, includingPropertiesForKeys: nil)
        XCTAssertEqual(apiDirectories.count, 1)
        let apiDirectory = try XCTUnwrap(apiDirectories.first)
        XCTAssertEqual(cliOutcome.plan.decision, apiOutcome.plan.decision)
        XCTAssertEqual(cliOutcome.plan.modelID, apiOutcome.plan.modelID)
        XCTAssertEqual(cliOutcome.plan.modelPath, apiOutcome.plan.modelPath)
        XCTAssertEqual(cliOutcome.plan.modelMetadata, apiOutcome.plan.modelMetadata)
        XCTAssertEqual(cliOutcome.plan.provider, apiOutcome.plan.provider)
        XCTAssertEqual(cliOutcome.plan.request.maxTokens, apiOutcome.plan.request.maxTokens)
        XCTAssertEqual(cliOutcome.result, apiOutcome.result)
        try FileManager.default.removeItem(at: fixture.audio)
        for directory in [cli.directory, apiDirectory] {
            let record = try SpeechTranscriptionRunRecord.inspect(at: directory)
            XCTAssertTrue(record.canRetry)
            XCTAssertEqual(record.state, .succeeded)
            XCTAssertEqual(try RunArtifact.read(XCTUnwrap(record.input).url), record.input)
        }
        let admission = await services.admission.snapshot()
        XCTAssertEqual(admission.activeRequests, 0)
        XCTAssertEqual(admission.totalAdmittedRequests, 1)
    }

    func testResolutionAndRuntimeAdmissionFailuresPersistFailureRecords() async throws {
        let fixture = try fixture()
        try Data("{\"target\":\"parakeet\",\"preprocessor\":{},\"encoder\":{}}".utf8)
            .write(to: fixture.model.appendingPathComponent("config.json"))
        let request = ASRRequest(audioURL: fixture.audio, task: .translate)
        let cli = try session(root: fixture.root, name: "invalid-model", request: request, model: fixture.model)
        do {
            _ = try await CLIASRRouting.transcribe(
                request: request, preferredBackend: .auto, modelOverride: fixture.model.path,
                recording: cli, executor: RecordedTranscriptionExecutor()
            )
            XCTFail("Incompatible local model was accepted")
        } catch {}
        XCTAssertEqual(try SpeechTranscriptionRunRecord.inspect(at: cli.directory).issue?.code, "model_backend_mismatch")
        let api = try session(root: fixture.root, name: "unavailable", request: request, model: fixture.model)
        let services = makeServices(root: fixture.root, executor: RecordedTranscriptionExecutor(), ensureAvailable: {
            throw SpeechTranscriptionIssue("fixture_runtime_missing", "Runtime unavailable")
        })
        let plan = SpeechTranscriptionPlan(
            request: request, decision: .init(backend: .qwen, reason: "fixture", normalizedLanguageHint: nil),
            modelID: "fixture", modelPath: fixture.model.path
        )
        do { _ = try await services.transcribe(plan, recording: api); XCTFail("Expected runtime failure") } catch {}
        let record = try SpeechTranscriptionRunRecord.inspect(at: api.directory)
        XCTAssertEqual(record.state, .failed)
        XCTAssertEqual(record.issue?.code, "fixture_runtime_missing")
        let admission = await services.admission.snapshot()
        XCTAssertEqual(admission.activeRequests, 0)
    }

    func testInspectionListingAndRetryShareRecordedOperationIdentity() async throws {
        let fixture = try fixture()
        let request = ASRRequest(audioURL: fixture.audio, language: "en", maxTokens: 19)
        let recording = try session(root: fixture.root, name: "run", request: request, model: fixture.model)
        _ = try await CLIASRRouting.transcribe(
            request: request, preferredBackend: .auto, modelOverride: fixture.model.path,
            recording: recording, executor: RecordedTranscriptionExecutor()
        )
        let envelope = RunInspectionAnalyzer(path: recording.directory.path).envelope()
        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.kind, "transcription_run")
        XCTAssertEqual(envelope.result.transcriptionRun?.id, recording.id)
        XCTAssertTrue(envelope.actions.contains { $0.id == "retry-transcription-run" && $0.enabled })
        let list = RunListAnalyzer(root: fixture.root.path).envelope()
        let entry = try XCTUnwrap(list.result.entries.first { $0.kind == "transcription_run" })
        XCTAssertEqual(entry.command, ["speech", "transcribe"])
        XCTAssertEqual(entry.format, "speech.transcribe")
        XCTAssertEqual(entry.artifactCount, 2)
        let coordinator = MachineInferenceCoordinator(
            stateDirectory: fixture.root.appendingPathComponent("admission"),
            hostSnapshot: {
                MachineInferenceHostSnapshot(
                    physicalMemoryBytes: 64 * 1_073_741_824, availableMemoryBytes: 48 * 1_073_741_824,
                    memoryPressure: .nominal, availableDiskBytes: 128 * 1_073_741_824
                )
            }
        )
        let child = try await RunRetry.retryTranscription(
            at: recording.directory, executor: RecordedTranscriptionExecutor(), admission: coordinator
        )
        XCTAssertEqual(child.parentID, recording.id)
        XCTAssertEqual(child.state, .succeeded)
        XCTAssertEqual(child.effective?.request.maxTokens, 19)
        XCTAssertEqual(try coordinator.snapshot().activePermits, 0)
        XCTAssertNil(CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "run", "retry", recording.directory.path]))
        for model in ["speech-asr-qwen3", "speech-asr-parakeet"] {
            XCTAssertEqual(
                CLIInferenceAdmissionClassifier.speechTranscriptionRequest(modelID: model).resourceClass,
                CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "speech", "transcribe", "audio.wav", "--model", model])?.resourceClass
            )
        }
    }

    func testInspectionReportsUnreadableRecordsAndMissingOutputs() async throws {
        let fixture = try fixture()
        let recording = try session(root: fixture.root, name: "run", request: .init(audioURL: fixture.audio), model: fixture.model)
        _ = try await CLIASRRouting.transcribe(
            request: .init(audioURL: fixture.audio), preferredBackend: .auto, modelOverride: fixture.model.path,
            recording: recording, executor: RecordedTranscriptionExecutor()
        )
        try FileManager.default.removeItem(at: recording.directory.appendingPathComponent("transcript.txt"))
        XCTAssertTrue(RunInspectionAnalyzer(path: recording.directory.path).envelope().diagnostics.contains {
            $0.id == "transcription_run_artifact_missing"
        })
        let recordURL = SpeechTranscriptionRunRecord.recordURL(at: recording.directory)
        try Data("broken".utf8).write(to: recordURL)
        let envelope = RunInspectionAnalyzer(path: recordURL.path).envelope()
        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertNil(envelope.result.transcriptionRun)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "transcription_run_unreadable" })
        XCTAssertEqual(try Data(contentsOf: recordURL), Data("broken".utf8))
    }

    private func makeServices(
        root: URL, executor: any SpeechTranscriptionExecutor, ensureAvailable: @escaping @Sendable () throws -> Void = {}
    ) -> RuntimeServingServices {
        let models = RuntimeModelPool(
            defaultModelID: "fixture.gguf", defaultEngine: .textCode, startupModelPath: nil,
            settingsStore: RuntimeModelSettingsStore(modelsDir: root), ensureMLXAvailable: {},
            prepareLoadedModel: { _ in }, unloadLoadedModel: { _ in }, clearMLXCache: {}
        )
        return RuntimeServingServices(
            models: models, media: APISidecarModelPool(settingsURL: root.appendingPathComponent("settings.json")),
            admission: RuntimeRequestAdmission(maxActiveRequests: 1), ensureAvailable: ensureAvailable, transcriptionExecutor: executor
        )
    }

    private func session(root: URL, name: String, request: ASRRequest, model: URL) throws -> SpeechTranscriptionRunSession {
        try SpeechTranscriptionRunSession(
            directory: root.appendingPathComponent(name),
            requested: .init(request: request, preferredBackend: .auto, modelOverride: model.path)
        )
    }

    private func fixture() throws -> (root: URL, audio: URL, model: URL) {
        let root = try root()
        let audio = root.appendingPathComponent("upload.wav")
        try Data("audio fixture".utf8).write(to: audio)
        let model = root.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: false)
        try Data("{}".utf8).write(to: model.appendingPathComponent("config.json"))
        return (root, audio, model)
    }

    private func root() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}

private struct RecordedTranscriptionExecutor: SpeechTranscriptionExecutor {
    func transcribeQwen(
        request: ASRRequest, modelID: String, modelPath: String?, progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        XCTAssertEqual(try Data(contentsOf: request.audioURL), Data("audio fixture".utf8))
        return ASRResult(text: "Recorded transcript", language: request.language)
    }

    func transcribeParakeet(
        request: ASRRequest, modelID: String, modelPath: String?, progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        try await transcribeQwen(request: request, modelID: modelID, modelPath: modelPath, progressHandler: progressHandler)
    }
}
