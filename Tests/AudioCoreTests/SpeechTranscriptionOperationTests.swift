import Foundation
import XCTest
import AudioCore

final class SpeechTranscriptionOperationTests: XCTestCase {
    func testBothBackendsReceiveTheResolvedRequestAndEmitOneSuccess() async throws {
        for backend in [ASRResolvedBackend.qwen, .parakeet] {
            let plan = try plan(backend: backend)
            let executor = SpeechOperationProbe()
            let events = SpeechEventProbe()
            let id = UUID()
            let outcome = try await SpeechTranscriptionOperation.execute(
                plan, id: id, eventHandler: { events.record($0) }, executor: executor
            )
            XCTAssertEqual(outcome.id, id)
            XCTAssertEqual(outcome.plan, plan)
            XCTAssertEqual(outcome.backend, backend)
            XCTAssertEqual(outcome.result.text, "fixture transcript")
            let calls = await executor.calls
            XCTAssertEqual(calls, [.init(backend: backend, request: plan.request, modelID: plan.modelID, modelPath: plan.modelPath)])
            XCTAssertEqual(events.names(), ["started", "progress", "succeeded"])
            XCTAssertEqual(Set(events.ids()), [id])
        }
    }

    func testValidationAndDeletedInputFailBeforeCallingTheExecutor() async throws {
        let valid = try plan()
        try valid.validate()
        try FileManager.default.removeItem(at: valid.request.audioURL)
        let executor = SpeechOperationProbe()
        let events = SpeechEventProbe()
        do {
            _ = try await SpeechTranscriptionOperation.execute(valid, eventHandler: { events.record($0) }, executor: executor)
            XCTFail("Deleted input was accepted")
        } catch let issue as SpeechTranscriptionIssue {
            XCTAssertEqual(issue.code, "audio_not_found")
        }
        let calls = await executor.calls
        XCTAssertTrue(calls.isEmpty)
        XCTAssertEqual(events.names(), ["started", "failed"])
    }

    func testInvalidTokenLimitAndIncompatibleTranslationFailBeforeExecution() async throws {
        let source = try plan()
        var invalidTokens = source.request
        invalidTokens.maxTokens = 0
        var translation = source.request
        translation.task = .translate
        let cases: [(ASRRequest, ASRResolvedBackend, String)] = [
            (invalidTokens, .qwen, "invalid_max_tokens"),
            (translation, .parakeet, "unsupported_task")
        ]
        for (request, backend, code) in cases {
            let invalid = SpeechTranscriptionPlan(
                request: request,
                decision: .init(backend: backend, reason: "fixture", normalizedLanguageHint: "en"),
                modelID: "fixture-model", modelPath: nil
            )
            let executor = SpeechOperationProbe()
            do {
                _ = try await SpeechTranscriptionOperation.execute(invalid, executor: executor)
                XCTFail("Invalid plan was executed")
            } catch let issue as SpeechTranscriptionIssue {
                XCTAssertEqual(issue.code, code)
            }
            let calls = await executor.calls
            XCTAssertTrue(calls.isEmpty)
        }
    }

    func testExecutorFailureEmitsOneFailureAndPreservesTheError() async throws {
        let source = try plan()
        let executor = SpeechOperationProbe(failure: .expected)
        let events = SpeechEventProbe()
        do {
            _ = try await SpeechTranscriptionOperation.execute(source, eventHandler: { events.record($0) }, executor: executor)
            XCTFail("Expected executor failure")
        } catch SpeechOperationFailure.expected {}
        XCTAssertEqual(events.names(), ["started", "progress", "failed"])
    }

    func testCancellationBeforeExecutionNeverCallsTheExecutor() async throws {
        let source = try plan()
        let gate = SpeechOperationGate()
        let executor = SpeechOperationProbe()
        let events = SpeechEventProbe()
        let task = Task {
            await gate.wait()
            return try await SpeechTranscriptionOperation.execute(source, eventHandler: { events.record($0) }, executor: executor)
        }
        try await waitUntil { await gate.entered }
        task.cancel()
        await gate.open()
        do {
            _ = try await task.value
            XCTFail("Cancelled request succeeded")
        } catch is CancellationError {}
        let calls = await executor.calls
        XCTAssertTrue(calls.isEmpty)
        XCTAssertEqual(events.names(), ["started", "cancelled"])
    }

    func testCancellationAfterAnUncooperativeExecutorReturnsIsNotSuccess() async throws {
        let source = try plan()
        let gate = SpeechOperationGate()
        let executor = SpeechOperationProbe(gate: gate)
        let events = SpeechEventProbe()
        let task = Task {
            try await SpeechTranscriptionOperation.execute(source, eventHandler: { events.record($0) }, executor: executor)
        }
        try await waitUntil { await gate.entered }
        task.cancel()
        await gate.open()
        do {
            _ = try await task.value
            XCTFail("Cancelled backend result was reported as success")
        } catch is CancellationError {}
        XCTAssertEqual(events.names(), ["started", "progress", "cancelled"])
    }

    private func plan(backend: ASRResolvedBackend = .qwen) throws -> SpeechTranscriptionPlan {
        let audio = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        try Data().write(to: audio)
        addTeardownBlock { try? FileManager.default.removeItem(at: audio) }
        return SpeechTranscriptionPlan(
            request: ASRRequest(audioURL: audio, language: "en", maxTokens: 37),
            decision: .init(backend: backend, reason: "fixture", normalizedLanguageHint: "en"),
            modelID: "fixture-model", modelPath: "/fixture/model"
        )
    }

    private func waitUntil(_ condition: () async -> Bool) async throws {
        let start = ContinuousClock.now
        while !(await condition()) {
            guard start.duration(to: .now) < .seconds(3) else { throw SpeechOperationFailure.timeout }
            try await Task.sleep(for: .milliseconds(1))
        }
    }
}

private enum SpeechOperationFailure: Error { case expected, timeout }

private actor SpeechOperationProbe: SpeechTranscriptionExecutor {
    struct Call: Equatable {
        let backend: ASRResolvedBackend
        let request: ASRRequest
        let modelID: String
        let modelPath: String?
    }
    private(set) var calls: [Call] = []
    private let failure: SpeechOperationFailure?
    private let gate: SpeechOperationGate?

    init(failure: SpeechOperationFailure? = nil, gate: SpeechOperationGate? = nil) {
        self.failure = failure
        self.gate = gate
    }

    func transcribeQwen(
        request: ASRRequest, modelID: String, modelPath: String?, progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        try await transcribe(.qwen, request: request, modelID: modelID, modelPath: modelPath, progressHandler: progressHandler)
    }

    func transcribeParakeet(
        request: ASRRequest, modelID: String, modelPath: String?, progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        try await transcribe(.parakeet, request: request, modelID: modelID, modelPath: modelPath, progressHandler: progressHandler)
    }

    private func transcribe(
        _ backend: ASRResolvedBackend, request: ASRRequest, modelID: String, modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        calls.append(.init(backend: backend, request: request, modelID: modelID, modelPath: modelPath))
        progressHandler?(ASRProgress(stage: .transcribing, tokensGenerated: 1))
        await gate?.wait()
        if let failure { throw failure }
        return ASRResult(text: "fixture transcript", language: request.language)
    }
}

private actor SpeechOperationGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private(set) var entered = false
    func wait() async {
        entered = true
        await withCheckedContinuation { continuation = $0 }
    }
    func open() { continuation?.resume(); continuation = nil }
}

private final class SpeechEventProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [(String, UUID)] = []
    func record(_ event: SpeechTranscriptionEvent) {
        lock.lock()
        defer { lock.unlock() }
        switch event {
        case .started(let id): events.append(("started", id))
        case .progress(let id, _): events.append(("progress", id))
        case .succeeded(let outcome): events.append(("succeeded", outcome.id))
        case .failed(let id, _): events.append(("failed", id))
        case .cancelled(let id): events.append(("cancelled", id))
        }
    }
    func names() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return events.map(\.0)
    }
    func ids() -> [UUID] {
        lock.lock()
        defer { lock.unlock() }
        return events.map(\.1)
    }
}
