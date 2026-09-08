import Foundation

public enum ParakeetError: LocalizedError {
    case modelNotLoaded
    case unsupportedModelId(String)
    case missingFiles([String])
    case downloadFailed(String)
    case extractionFailed
    case unsupportedCoreMLVariant(String)
    case coreMLUnavailable
    case coreMLProviderRequired
    case missingMLXEncoder
    case coreMLHybridArtifactMismatch
    case decoderBatchTooLarge(actual: Int, maximum: Int)
    case unexpectedDecoderBatchCount(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Parakeet model is not loaded."
        case .unsupportedModelId(let modelId):
            return "Unsupported model id: \(modelId)"
        case .missingFiles(let files):
            return "Parakeet model files missing: \(files.joined(separator: ", "))"
        case .downloadFailed(let message):
            return message
        case .extractionFailed:
            return "Failed to prepare model files"
        case .unsupportedCoreMLVariant(let variant):
            return "The Core ML Parakeet encoder supports TDT checkpoints only; found \(variant)."
        case .coreMLUnavailable:
            return "The Core ML Parakeet encoder is unavailable on this platform."
        case .coreMLProviderRequired:
            return "This Parakeet package requires the Core ML provider."
        case .missingMLXEncoder:
            return "The Parakeet MLX encoder is not available in this package."
        case .coreMLHybridArtifactMismatch:
            return "The compact Parakeet decoder and Core ML encoder must come from the same artifact."
        case .decoderBatchTooLarge(let actual, let maximum):
            return "The Parakeet decoder batch has \(actual) windows; the maximum is \(maximum)."
        case .unexpectedDecoderBatchCount(let expected, let actual):
            return "The Parakeet decoder returned \(actual) windows; expected \(expected)."
        }
    }
}
