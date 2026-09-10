import Foundation
import MereRunExecution

/// Resolved backend and model inputs for one file transcription or translation.
public struct SpeechTranscriptionPlan: Sendable, Hashable, Codable {
    public let request: ASRRequest
    public let decision: ASRBackendDecision
    public let modelID: String
    public let modelPath: String?
    public let provider: ParakeetExecutionProvider
    public let modelMetadata: [RunFileSnapshot]

    public init(
        request: ASRRequest, decision: ASRBackendDecision, modelID: String, modelPath: String?,
        provider: ParakeetExecutionProvider = .mlx, modelMetadata: [RunFileSnapshot] = []
    ) {
        self.request = request
        self.decision = decision
        self.modelID = modelID
        self.modelPath = modelPath
        self.provider = provider
        self.modelMetadata = modelMetadata
    }

    /// Checks inputs without loading a checkpoint or acquiring resources.
    public func validate() throws {
        guard request.maxTokens > 0 else {
            throw SpeechTranscriptionIssue("invalid_max_tokens", "Maximum transcription tokens must be positive.")
        }
        guard request.task != .translate || decision.backend == .qwen else {
            throw SpeechTranscriptionIssue("unsupported_task", "Translation requires the Qwen backend.")
        }
        if case .coreML = provider, decision.backend != .parakeet || request.task != .transcribe {
            throw SpeechTranscriptionIssue("incompatible_execution_provider", "Core ML execution requires Parakeet transcription.")
        }
        guard !modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SpeechTranscriptionIssue("model_missing", "A transcription model ID is required.")
        }
        var isDirectory: ObjCBool = false
        guard request.audioURL.isFileURL,
              FileManager.default.fileExists(atPath: request.audioURL.path, isDirectory: &isDirectory),
              !isDirectory.boolValue else {
            throw SpeechTranscriptionIssue("audio_not_found", "Audio file not found: \(request.audioURL.path)")
        }
    }
}

public struct SpeechTranscriptionIssue: Error, LocalizedError, Sendable, Equatable, Codable {
    public let code: String
    public let message: String
    public var errorDescription: String? { message }

    public init(_ code: String, _ message: String) {
        self.code = code
        self.message = message
    }
}

/// Supplies a native generator or borrows one from a resident runtime pool.
public protocol SpeechTranscriptionExecutor: Sendable {
    func transcribeQwen(
        request: ASRRequest, modelID: String, modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult

    func transcribeParakeet(
        request: ASRRequest, modelID: String, modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult
}

public struct SpeechTranscriptionOutcome: Sendable {
    public let id: UUID
    public let result: ASRResult
    public let plan: SpeechTranscriptionPlan
    public var decision: ASRBackendDecision { plan.decision }
    public var backend: ASRResolvedBackend { plan.decision.backend }
}

public enum SpeechTranscriptionEvent: Sendable {
    case started(UUID)
    case progress(UUID, ASRProgress)
    case succeeded(SpeechTranscriptionOutcome)
    case failed(UUID, SpeechTranscriptionIssue)
    case cancelled(UUID)
}

/// Executes the resolved plan and emits one terminal outcome. The executor
/// owns admission and residency; this operation never acquires a second
/// machine reservation or unloads a borrowed runtime.
public enum SpeechTranscriptionOperation {
    public static func execute(
        _ plan: SpeechTranscriptionPlan,
        id: UUID = UUID(),
        recording: SpeechTranscriptionRunSession? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil,
        eventHandler: (@Sendable (SpeechTranscriptionEvent) -> Void)? = nil,
        executor: any SpeechTranscriptionExecutor
    ) async throws -> SpeechTranscriptionOutcome {
        let id = recording?.id ?? id
        eventHandler?(.started(id))
        do {
            try Task.checkCancellation()
            // Files can disappear after observational resolution or preflight.
            try plan.validate()
            let plan = try recording?.prepare(plan) ?? plan
            try Task.checkCancellation()
            let progress: (@Sendable (ASRProgress) -> Void)?
            if progressHandler != nil || eventHandler != nil {
                progress = { value in
                    progressHandler?(value)
                    eventHandler?(.progress(id, value))
                }
            } else {
                progress = nil
            }
            let result: ASRResult
            switch plan.decision.backend {
            case .qwen:
                result = try await executor.transcribeQwen(
                    request: plan.request, modelID: plan.modelID, modelPath: plan.modelPath,
                    progressHandler: progress
                )
            case .parakeet:
                result = try await executor.transcribeParakeet(
                    request: plan.request, modelID: plan.modelID, modelPath: plan.modelPath,
                    progressHandler: progress
                )
            }
            try Task.checkCancellation()
            let outcome = SpeechTranscriptionOutcome(id: id, result: result, plan: plan)
            try recording?.succeed(outcome)
            eventHandler?(.succeeded(outcome))
            return outcome
        } catch {
            if error is CancellationError || Task.isCancelled {
                try recording?.fail(CancellationError())
                eventHandler?(.cancelled(id))
                throw CancellationError()
            }
            try recording?.fail(error)
            let issue = error as? SpeechTranscriptionIssue
                ?? SpeechTranscriptionIssue("transcription_failed", error.localizedDescription)
            eventHandler?(.failed(id, issue))
            throw error
        }
    }
}
