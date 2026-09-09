import Foundation
import MLX

public enum LagunaError: LocalizedError {
    case modelPathRequired
    case missingFiles([String])
    case dflashIncompatible(String)
    case modelNotLoaded
    case adapterSwitchDuringActiveGeneration
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelPathRequired:
            return "Laguna requires an installed managed model or an explicit local MLX checkpoint path."
        case .missingFiles(let files):
            return "Laguna checkpoint is missing required files: \(files.joined(separator: ", "))."
        case .dflashIncompatible(let message):
            return "Laguna DFlash checkpoint is incompatible: \(message)"
        case .modelNotLoaded:
            return "Laguna model is not loaded."
        case .adapterSwitchDuringActiveGeneration:
            return "Laguna cannot switch text LoRA adapters while batched generation is active."
        case .generationFailed(let message):
            return "Laguna generation failed: \(message)"
        }
    }
}

public typealias LagunaContinuousBatchingStats = RuntimeDecodeBatchingStats
