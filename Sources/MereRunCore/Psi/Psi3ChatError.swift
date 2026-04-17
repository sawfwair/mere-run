import Foundation

/// Errors from Psi3 chat generation.
public enum Psi3ChatError: LocalizedError {
    case modelNotLoaded
    case unsupportedModelId(String)
    case missingFiles([String])
    case downloadFailed(String)
    case extractionFailed
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Psi model not loaded"
        case .unsupportedModelId(let modelId):
            return "Unsupported model id: \(modelId)"
        case .missingFiles(let files):
            return "Missing required files: \(files.joined(separator: ", "))"
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .extractionFailed:
            return "Failed to extract model archive"
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        }
    }
}
