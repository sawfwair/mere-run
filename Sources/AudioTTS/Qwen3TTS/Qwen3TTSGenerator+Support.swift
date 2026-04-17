import Foundation
import MLX

/// Support types and helpers shared across the TTS runtime.
/// Keeping them here keeps the actor file focused on entrypoints.

func mapTalkerWeightKey(_ key: String) -> String {
    guard key.hasPrefix("talker.") else { return key }
    return String(key.dropFirst("talker.".count))
}

public enum Qwen3TTSError: LocalizedError {
    case modelsNotLoaded
    case unsupportedModelId(String)
    case missingFiles([String])
    case cloneAssetsMissing([String])
    case invalidCloneReference(String)
    case noAudioTokensGenerated
    case weightsNotFound(URL)
    case downloadFailed(String)
    case extractionFailed
    case tokenizationFailed
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelsNotLoaded:
            return "TTS models not loaded"
        case .unsupportedModelId(let modelId):
            return "Unsupported model id: \(modelId)"
        case .missingFiles(let files):
            return "Missing required files: \(files.joined(separator: ", "))"
        case .cloneAssetsMissing(let assets):
            return "Missing clone assets: \(assets.joined(separator: ", "))"
        case .invalidCloneReference(let message):
            return "Invalid clone reference: \(message)"
        case .noAudioTokensGenerated:
            return "No audio tokens were generated"
        case .weightsNotFound(let url):
            return "Weights not found at \(url.path)"
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .extractionFailed:
            return "Failed to extract model archive"
        case .tokenizationFailed:
            return "Failed to tokenize input text"
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        }
    }
}

public actor Qwen3TTSModelContainer {
    private var generator: Qwen3TTSGenerator?
    private let repoId: String

    public init(repoId: String = Qwen3TTSResources.defaultModelId) async throws {
        self.repoId = repoId
    }

    public func prepare(
        progressHandler: (@Sendable (Double) -> Void)? = nil
    ) async throws -> Qwen3TTSGenerator {
        let generator = Qwen3TTSGenerator(modelId: repoId)
        progressHandler?(0)
        try await generator.prepare()
        progressHandler?(1.0)
        self.generator = generator
        return generator
    }

    public func cachedGenerator() -> Qwen3TTSGenerator? {
        generator
    }
}
