import Foundation

public enum Q35Error: LocalizedError {
    case modelNotLoaded
    case unsupportedModelId(String)
    case missingFiles([String])
    case downloadFailed(String)
    case extractionFailed
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Qwen-family model not loaded"
        case .unsupportedModelId(let id):
            return "Unsupported Qwen-family model id: \(id)"
        case .missingFiles(let files):
            return "Missing required files: \(files.joined(separator: ", "))"
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .extractionFailed:
            return "Failed to prepare Qwen-family model files"
        case .generationFailed(let message):
            return message
        }
    }
}
