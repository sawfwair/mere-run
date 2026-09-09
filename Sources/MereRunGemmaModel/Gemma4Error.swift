import Foundation

public enum Gemma4Error: LocalizedError {
    case modelNotLoaded
    case missingFiles([String])
    case unsupportedConfiguration(String)
    case unsupportedModelLocation(String)
    case downloadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Gemma4 model is not loaded."
        case .missingFiles(let files):
            return "Missing required Gemma4 files: \(files.joined(separator: ", "))"
        case .unsupportedConfiguration(let message):
            return message
        case .unsupportedModelLocation(let location):
            return "Could not resolve Gemma4 model location: \(location)"
        case .downloadFailed(let message):
            return "Gemma4 download failed: \(message)"
        }
    }
}
