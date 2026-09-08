import Foundation
import AudioCore
import AudioSTT
import MereRunCore

/// Applies API model-allowlist policy before using the shared speech resolver.
enum APITranscription {
    static func plan(audioURL: URL, options: APIServerContract.TranscriptionPlan) throws -> SpeechTranscriptionPlan {
        let selection = try resolveModel(options.modelID)
        return try SpeechTranscriptionResolver.resolve(
            request: ASRRequest(audioURL: audioURL, language: options.language, task: options.task, maxTokens: options.maxTokens),
            preferredBackend: selection.backend, modelOverride: selection.modelOverride
        )
    }

    private static func resolveModel(
        _ requestedModel: String
    ) throws -> (modelOverride: String?, backend: ASRBackend) {
        let normalized = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        let asPath = URL(fileURLWithPath: normalized).standardizedFileURL
        if FileManager.default.fileExists(atPath: asPath.path) {
            return (asPath.path, .auto)
        }
        guard let spec = ManagedModelCatalog.spec(for: normalized),
              spec.category == .speechASR else {
            throw APIRequestValidationError.invalidField(
                "model",
                "use a mere.run ASR model id or a local ASR model path"
            )
        }
        let backend: ASRBackend = spec.id.contains("parakeet") ? .parakeet : .qwen
        return (spec.id, backend)
    }

}
