import Foundation

/// Resource paths and validation for CodeGen (Qwen3-Coder) models.
public struct CodeGenResources: Sendable, Hashable {
    public static let defaultModelId = "text-code-qwen3"
    public static let r2ArchiveKey = "models/qwen3-coder-next.tar.gz"
    public static let r2ArchiveSize: Int64 = 47_530_770_762  // Q4_K_M GGUF tar.gz
    public static let defaultRepoId = "Qwen/Qwen3-Coder-Next-GGUF"
    public static let defaultRevision = "main"
    public static let managedRelativePath = "\(defaultModelId).gguf"
    public static let hubGGUFPath = "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf"
    public static let snapshotPatterns = ["Qwen3-Coder-Next-Q4_K_M/*"]

    public static let hubFallbackConfig = HubFallbackConfig(
        repoId: defaultRepoId,
        revision: defaultRevision,
        patterns: snapshotPatterns,
        filePath: hubGGUFPath
    )

    public var ggufURL: URL

    public init(ggufURL: URL) {
        self.ggufURL = ggufURL
    }

    /// Validate that the GGUF file exists.
    /// Returns list of missing URLs. Empty list means the file is present.
    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        if !fileManager.fileExists(atPath: ggufURL.path) {
            missing.append(ggufURL)
        }
        return missing
    }
}

/// Errors specific to CodeGen operations.
public enum CodeGenError: Error, LocalizedError {
    case modelNotLoaded
    case unsupportedModelId(String)
    case missingFiles([String])
    case downloadFailed(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Model not loaded"
        case .unsupportedModelId(let id):
            return "Unsupported model ID: \(id)"
        case .missingFiles(let files):
            return "Missing files: \(files.joined(separator: ", "))"
        case .downloadFailed(let reason):
            return "Download failed: \(reason)"
        case .generationFailed(let reason):
            return "Generation failed: \(reason)"
        }
    }
}
