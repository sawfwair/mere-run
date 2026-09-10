import ArgumentParser
import AudioCore
import AudioSTT

// Compatibility names for command fixtures and callers during migration.
typealias CLIASRExecutionResult = SpeechTranscriptionOutcome
typealias CLIASRTranscriptionExecutor = SpeechTranscriptionExecutor

enum CLIASRRouting {
    static func transcribe(
        request: ASRRequest,
        preferredBackend: ASRBackend,
        modelOverride: String? = nil,
        parakeetExecutionProvider: ParakeetExecutionProvider = .mlx,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil,
        recording: SpeechTranscriptionRunSession? = nil,
        executor: (any SpeechTranscriptionExecutor)? = nil
    ) async throws -> SpeechTranscriptionOutcome {
        do {
            let plan = try SpeechTranscriptionResolver.resolve(
                request: request, preferredBackend: preferredBackend, modelOverride: modelOverride,
                parakeetExecutionProvider: parakeetExecutionProvider, captureModelMetadata: recording != nil
            )
            return try await SpeechTranscriptionOperation.execute(
                plan, recording: recording, progressHandler: progressHandler,
                executor: executor ?? NativeSpeechTranscriptionExecutor(parakeetExecutionProvider: parakeetExecutionProvider)
            )
        } catch let issue as SpeechTranscriptionIssue {
            try recording?.fail(issue)
            if issue.code == "incompatible_execution_provider" {
                throw ValidationError(issue.message + " Use --backend qwen without --provider coreml or --coreml-encoder.")
            }
            throw ValidationError(issue.message)
        } catch {
            try recording?.fail(error)
            throw error
        }
    }
}
