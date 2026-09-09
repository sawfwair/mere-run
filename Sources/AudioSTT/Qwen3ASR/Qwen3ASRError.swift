import Foundation

// MARK: - Errors

public enum Qwen3ASRError: LocalizedError {
    case modelsNotLoaded
    case unsupportedModelId(String)
    case missingFiles([String])
    case weightsNotFound(URL)
    case downloadFailed(String)
    case extractionFailed
    case transcriptionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelsNotLoaded:
            return "ASR models not loaded"
        case .unsupportedModelId(let modelId):
            return "Unsupported model id: \(modelId)"
        case .missingFiles(let files):
            return "Missing required files: \(files.joined(separator: ", "))"
        case .weightsNotFound(let url):
            return "Weights not found at \(url.path)"
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .extractionFailed:
            return "Failed to prepare model files"
        case .transcriptionFailed(let message):
            return "Transcription failed: \(message)"
        }
    }
}
