import Foundation
import AudioCore

/// Owns a temporary native generator and unloads it on every terminal path.
/// Long-lived callers can supply their own resident executor to the operation.
public struct NativeSpeechTranscriptionExecutor: SpeechTranscriptionExecutor {
    private let parakeetExecutionProvider: ParakeetExecutionProvider

    public init(parakeetExecutionProvider: ParakeetExecutionProvider = .mlx) {
        self.parakeetExecutionProvider = parakeetExecutionProvider
    }

    public func transcribeQwen(
        request: ASRRequest, modelID: String, modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        guard case .mlx = parakeetExecutionProvider else {
            throw SpeechTranscriptionIssue("incompatible_execution_provider", "Core ML execution requires the Parakeet backend.")
        }
        let generator = Qwen3ASRGenerator(modelId: modelID)
        do {
            let result = try await generator.transcribe(request, modelPath: modelPath, progressHandler: progressHandler)
            await generator.unload()
            return result
        } catch {
            await generator.unload()
            throw error
        }
    }

    public func transcribeParakeet(
        request: ASRRequest, modelID: String, modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        let generator = ParakeetGenerator(modelId: modelID, executionProvider: parakeetExecutionProvider)
        do {
            let result = try await generator.transcribe(request, modelPath: modelPath, progressHandler: progressHandler)
            await generator.unload()
            return result
        } catch {
            await generator.unload()
            throw error
        }
    }
}
