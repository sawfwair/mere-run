import Foundation

/// Resource paths, Hub fallback config, and validation for the
/// DeepSeek V4 Flash premier agent tier.
///
/// The model is a single ~81 GB GGUF served by the bundled `ds4-server`
/// inference binary (see `vendor/ds4/`). Unlike the other text-chat
/// generators in this codebase, the model is **not** loaded in-process:
/// `DeepseekV4FlashGenerator` spawns the vendored binary as a subprocess
/// and proxies OpenAI-compatible chat requests over loopback HTTP.
public struct DeepseekV4FlashResources: Sendable, Hashable {
    public static let defaultModelId = "text-agent-deepseek-v4-flash"
    public static let defaultRepoId = "jedisct1/DeepSeek-V4-Flash-imatrix-aligned"
    public static let defaultRevision = "main"
    public static let defaultContextLength = 65_536
    public static let serverStartupTimeoutSeconds: TimeInterval = 300

    /// Imatrix-tuned q2 GGUF with page-aligned tensor offsets: ~81 GB.
    /// Drop-in replacement for the original `antirez/deepseek-v4-gguf` imatrix
    /// file, with alignment that enables efficient mmap on Apple Silicon and a
    /// refreshed calibration corpus.
    public static let imatrixGGUFFile =
        "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-aligned.gguf"

    /// Previous imatrix filename from the original upstream GGUF. Existing
    /// installs may still point at this file, often through a symlink.
    public static let previousImatrixGGUFFile =
        "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"

    /// Legacy non-imatrix q2 GGUF. Not downloaded by mere.run, but recognized
    /// on disk if a user installed it manually — the resolver still prefers
    /// imatrix when both are present.
    public static let legacyGGUFFile =
        "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"

    /// Back-compat alias: code that referenced `defaultGGUFFile` keeps working.
    public static let defaultGGUFFile = imatrixGGUFFile

    public static let managedRelativePath = "\(defaultModelId).gguf"

    public static let hubFallbackConfig = HubFallbackConfig(
        repoId: defaultRepoId,
        revision: defaultRevision,
        patterns: [imatrixGGUFFile],
        filePath: imatrixGGUFFile
    )

    public var ggufURL: URL

    public init(ggufURL: URL) {
        self.ggufURL = ggufURL
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        fileManager.fileExists(atPath: ggufURL.path) ? [] : [ggufURL]
    }
}

public enum DeepseekV4FlashError: Error, LocalizedError {
    case binaryNotFound(searched: [URL])
    case modelNotFound(URL)
    case downloadFailed(String)
    case serverFailedToStart(String)
    case serverExited(Int32, stderr: String)
    case requestFailed(String)

    public var errorDescription: String? {
        switch self {
        case .binaryNotFound(let searched):
            var lines = ["DeepSeek V4 Flash inference binary (ds4-server) not found."]
            lines.append("Run scripts/rebuild_ds4.sh to populate vendor/ds4/.")
            if !searched.isEmpty {
                lines.append("Searched:")
                lines.append(contentsOf: searched.map { "  - \($0.path)" })
            }
            return lines.joined(separator: "\n")
        case .modelNotFound(let url):
            return "DeepSeek V4 Flash GGUF not found at \(url.path). " +
                "Run: mere.run model pull \(DeepseekV4FlashResources.defaultModelId)"
        case .downloadFailed(let reason):
            return "DeepSeek V4 Flash download failed: \(reason)"
        case .serverFailedToStart(let reason):
            return "ds4-server failed to become ready: \(reason)"
        case .serverExited(let code, let stderr):
            return "ds4-server exited unexpectedly (status \(code)). Last stderr:\n\(stderr)"
        case .requestFailed(let reason):
            return "ds4-server request failed: \(reason)"
        }
    }
}
