import Foundation
import XCTest
import MereRunExecution
@testable import AudioCore

final class SpeechTranscriptionRunTests: XCTestCase {
    func testBothBackendsRetainAudioResolvedSettingsAndStructuredAlignment() async throws {
        for backend in [ASRResolvedBackend.qwen, .parakeet] {
            let fixture = try fixture(backend: backend)
            let session = try session(fixture)
            let expected = ASRResult(
                text: "Retained transcript", language: "en", duration: 2,
                tokenAlignments: [.init(text: "Retained", startSeconds: 0.2, durationSeconds: 0.4)],
                sentenceAlignments: [.init(text: "Retained transcript", startSeconds: 0.2, durationSeconds: 1.5, tokens: [])]
            )
            let events = TranscriptionRunEvents()
            let executor = TranscriptionRunExecutor { request in
                XCTAssertNotEqual(request.audioURL, fixture.plan.request.audioURL)
                XCTAssertEqual(try Data(contentsOf: request.audioURL), Data("audio fixture".utf8))
                let active = try SpeechTranscriptionRunRecord.inspect(at: session.directory)
                XCTAssertEqual(active.state, .running)
                XCTAssertEqual(active.requested.request.audioURL, fixture.plan.request.audioURL)
                XCTAssertEqual(active.effective?.request, request)
                return expected
            }
            let outcome = try await SpeechTranscriptionOperation.execute(
                fixture.plan, recording: session, eventHandler: { events.append($0) }, executor: executor
            )
            XCTAssertEqual(outcome.id, session.id)
            let record = try SpeechTranscriptionRunRecord.inspect(at: session.directory)
            XCTAssertEqual(record.state, .succeeded)
            XCTAssertEqual(record.effective?.decision.backend, backend)
            XCTAssertEqual(record.artifacts.count, 2)
            XCTAssertEqual(events.values(), ["started", "succeeded"])
            let json = session.directory.appendingPathComponent("result.json")
            XCTAssertEqual(try RunRecordCodec.decoder().decode(ASRResult.self, from: Data(contentsOf: json)), expected)
            try FileManager.default.removeItem(at: fixture.plan.request.audioURL)
            XCTAssertEqual(try RunArtifact.read(XCTUnwrap(record.input).url), record.input)
            XCTAssertEqual(try String(contentsOf: session.directory.appendingPathComponent("transcript.txt"), encoding: .utf8), expected.text)
        }
    }

    func testFailureCancellationAndOutputFailureNeverRecordSuccess() async throws {
        for mode in ["failure", "cancelled", "output"] {
            let fixture = try fixture()
            let session = try session(fixture)
            let events = TranscriptionRunEvents()
            let executor = TranscriptionRunExecutor { _ in
                if mode == "cancelled" { throw CancellationError() }
                if mode == "failure" { throw SpeechTranscriptionIssue("fixture", "Expected failure") }
                try FileManager.default.createDirectory(at: session.directory.appendingPathComponent("result.json"), withIntermediateDirectories: false)
                return ASRResult(text: "Cannot save")
            }
            do {
                _ = try await SpeechTranscriptionOperation.execute(
                    fixture.plan, recording: session, eventHandler: { events.append($0) }, executor: executor
                )
                XCTFail("Expected a terminal error")
            } catch {
                XCTAssertEqual(error is CancellationError, mode == "cancelled")
            }
            let record = try SpeechTranscriptionRunRecord.inspect(at: session.directory)
            XCTAssertEqual(record.state, mode == "cancelled" ? .cancelled : .failed)
            XCTAssertEqual(events.values(), ["started", mode == "cancelled" ? "cancelled" : "failed"])
            XCTAssertTrue(record.artifacts.isEmpty)
            XCTAssertNotNil(record.effective)
        }
    }

    func testCancellationBeforePreparationDoesNotInvokeTheExecutor() async throws {
        let fixture = try fixture()
        let session = try session(fixture)
        let gate = TranscriptionRunGate()
        let task = Task {
            await gate.wait()
            return try await SpeechTranscriptionOperation.execute(fixture.plan, recording: session, executor: TranscriptionRunExecutor { _ in
                XCTFail("Cancelled request executed")
                return ASRResult(text: "Unexpected")
            })
        }
        try await gate.waitUntilEntered()
        task.cancel()
        await gate.open()
        do { _ = try await task.value; XCTFail("Cancelled request succeeded") } catch is CancellationError {}
        let record = try SpeechTranscriptionRunRecord.inspect(at: session.directory)
        XCTAssertEqual(record.state, .cancelled)
        XCTAssertNil(record.effective)
        XCTAssertNil(record.input)
        XCTAssertFalse(record.canRetry)
    }

    func testCancellationAfterExecutorReturnsRetainsPreparedInputWithoutSuccess() async throws {
        let fixture = try fixture()
        let session = try session(fixture)
        let gate = TranscriptionRunGate()
        let task = Task {
            try await SpeechTranscriptionOperation.execute(fixture.plan, recording: session, executor: TranscriptionRunExecutor { _ in
                await gate.wait()
                return ASRResult(text: "Completed after cancellation")
            })
        }
        try await gate.waitUntilEntered()
        task.cancel()
        await gate.open()
        do { _ = try await task.value; XCTFail("Cancelled request succeeded") } catch is CancellationError {}
        let record = try SpeechTranscriptionRunRecord.inspect(at: session.directory)
        XCTAssertEqual(record.state, .cancelled)
        XCTAssertNotNil(record.input)
        XCTAssertTrue(record.artifacts.isEmpty)
    }

    func testRetryUsesRetainedAudioProviderAndSettingsInNewDirectory() async throws {
        let fixture = try fixture(backend: .parakeet, provider: .coreML(artifactURL: URL(fileURLWithPath: "/fixture/coreml")))
        let session = try session(fixture)
        _ = try await SpeechTranscriptionOperation.execute(fixture.plan, recording: session, executor: TranscriptionRunExecutor())
        let parentURL = SpeechTranscriptionRunRecord.recordURL(at: session.directory)
        let parentBytes = try Data(contentsOf: parentURL)
        try FileManager.default.removeItem(at: fixture.plan.request.audioURL)
        let retry = try SpeechTranscriptionRunSession.retryPlan(at: session.directory)
        XCTAssertNotEqual(retry.session.id, session.id)
        XCTAssertNotEqual(retry.session.directory, session.directory)
        XCTAssertEqual(retry.plan.provider, fixture.plan.provider)
        XCTAssertEqual(retry.plan.request.maxTokens, 37)
        XCTAssertEqual(retry.plan.decision, fixture.plan.decision)
        _ = try await SpeechTranscriptionOperation.execute(retry.plan, recording: retry.session, executor: TranscriptionRunExecutor())
        let child = try SpeechTranscriptionRunRecord.inspect(at: retry.session.directory)
        XCTAssertEqual(child.parentID, session.id)
        XCTAssertEqual(child.state, .succeeded)
        XCTAssertEqual(try Data(contentsOf: parentURL), parentBytes)
        XCTAssertNotEqual(child.input?.url, retry.plan.request.audioURL)
    }

    func testRetryRejectsChangedInputAndChangedOrNewMetadataBeforeCreatingDirectory() async throws {
        for mode in ["input", "config", "new-manifest"] {
            let fixture = try fixture()
            let session = try session(fixture)
            _ = try await SpeechTranscriptionOperation.execute(fixture.plan, recording: session, executor: TranscriptionRunExecutor())
            let record = try SpeechTranscriptionRunRecord.inspect(at: session.directory)
            let target: URL
            switch mode {
            case "input": target = try XCTUnwrap(record.input?.url)
            case "config": target = fixture.root.appendingPathComponent("config.json")
            default: target = fixture.root.appendingPathComponent("manifest.json")
            }
            try Data("changed".utf8).write(to: target)
            let siblings = try FileManager.default.contentsOfDirectory(atPath: fixture.root.path)
            XCTAssertThrowsError(try SpeechTranscriptionRunSession.retryPlan(at: session.directory)) {
                XCTAssertEqual(($0 as? SpeechTranscriptionIssue)?.code, mode == "input" ? "run_input_changed" : "run_model_changed")
            }
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: fixture.root.path), siblings)
        }
    }

    func testActiveAbandonedFutureAndCorruptRecordsAreHandledWithoutDataLoss() throws {
        let fixture = try fixture()
        var session: SpeechTranscriptionRunSession? = try session(fixture)
        let directory = try XCTUnwrap(session?.directory)
        XCTAssertEqual(try SpeechTranscriptionRunRecord.inspect(at: directory).state, .preparing)
        XCTAssertThrowsError(try SpeechTranscriptionRunSession.retryPlan(at: directory)) {
            XCTAssertEqual(($0 as? SpeechTranscriptionIssue)?.code, "run_active")
        }
        _ = try XCTUnwrap(session).prepare(fixture.plan)
        XCTAssertEqual(try SpeechTranscriptionRunRecord.inspect(at: directory).state, .running)
        session = nil
        let interrupted = try SpeechTranscriptionRunRecord.inspect(at: directory)
        XCTAssertEqual(interrupted.state, .interrupted)
        XCTAssertTrue(interrupted.canRetry)
        XCTAssertEqual(try SpeechTranscriptionRunRecord.inspect(at: directory), interrupted)
        let recordURL = SpeechTranscriptionRunRecord.recordURL(at: directory)
        let original = try String(contentsOf: recordURL, encoding: .utf8)
        let future = original.replacingOccurrences(of: "\"schemaVersion\" : 1", with: "\"schemaVersion\" : 99")
        XCTAssertNotEqual(future, original)
        for bytes in [Data(future.utf8), Data("broken".utf8)] {
            try bytes.write(to: recordURL)
            XCTAssertThrowsError(try SpeechTranscriptionRunRecord.inspect(at: directory))
            XCTAssertEqual(try Data(contentsOf: recordURL), bytes)
        }
        XCTAssertThrowsError(try self.session(fixture))
        XCTAssertEqual(try Data(contentsOf: recordURL), Data("broken".utf8))
    }

    private func fixture(
        backend: ASRResolvedBackend = .qwen, provider: ParakeetExecutionProvider = .mlx
    ) throws -> (root: URL, plan: SpeechTranscriptionPlan) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let audio = root.appendingPathComponent("source.wav")
        try Data("audio fixture".utf8).write(to: audio)
        let config = root.appendingPathComponent("config.json")
        try Data("{}".utf8).write(to: config)
        let metadata = try [config, root.appendingPathComponent("manifest.json")].map(RunFileSnapshot.capture)
        return (root, SpeechTranscriptionPlan(
            request: .init(audioURL: audio, language: "en", maxTokens: 37),
            decision: .init(backend: backend, reason: "fixture selection", normalizedLanguageHint: "en"),
            modelID: "fixture-model", modelPath: root.path, provider: provider, modelMetadata: metadata
        ))
    }

    private func session(_ fixture: (root: URL, plan: SpeechTranscriptionPlan)) throws -> SpeechTranscriptionRunSession {
        try SpeechTranscriptionRunSession(
            directory: fixture.root.appendingPathComponent("run"),
            requested: .init(request: fixture.plan.request, preferredBackend: .auto, provider: fixture.plan.provider)
        )
    }
}

private struct TranscriptionRunExecutor: SpeechTranscriptionExecutor {
    var execute: @Sendable (ASRRequest) async throws -> ASRResult = { _ in ASRResult(text: "Fixture transcript") }

    func transcribeQwen(
        request: ASRRequest, modelID: String, modelPath: String?, progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult { try await execute(request) }

    func transcribeParakeet(
        request: ASRRequest, modelID: String, modelPath: String?, progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult { try await execute(request) }
}

private actor TranscriptionRunGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var entered = false
    func wait() async {
        entered = true
        await withCheckedContinuation { continuation = $0 }
    }
    func waitUntilEntered() async throws {
        let start = ContinuousClock.now
        while !entered {
            guard start.duration(to: .now) < .seconds(3) else {
                throw SpeechTranscriptionIssue("fixture_timeout", "The fixture executor did not start.")
            }
            try await Task.sleep(for: .milliseconds(1))
        }
    }
    func open() { continuation?.resume(); continuation = nil }
}

private final class TranscriptionRunEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [String] = []
    func append(_ event: SpeechTranscriptionEvent) {
        lock.withLock {
            switch event {
            case .started: events.append("started")
            case .progress: events.append("progress")
            case .succeeded: events.append("succeeded")
            case .failed: events.append("failed")
            case .cancelled: events.append("cancelled")
            }
        }
    }
    func values() -> [String] { lock.withLock { events } }
}
