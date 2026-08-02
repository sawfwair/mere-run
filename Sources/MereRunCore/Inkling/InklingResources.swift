import Foundation
import MLX

/// Managed metadata for the text-only native MLX Inkling-Small runtime.
public enum InklingResources {
    public static let modelID = "text-chat-inkling-small"

    public static let baseRepoID = "thinkingmachines/Inkling-Small"
    public static let baseRevision = "b2d4f225a02032c5d154bff748ab5a00c5ca26e4"

    public static let artifactRepoID = "Sawfwair/Inkling-Small-MLX-Mixed-2bit"
    public static let artifactRevision = "22c95fc2e30400a893dc9ac9fe4df17f306362ae"
    public static let quantizationBits = 2
    public static let quantizationGroupSize = 128
    public static let quantizationMode = "affine"
    public static let quantizationScope = "routed_experts"

    public static let defaultContextLength = 32_768
    public static let maximumContextLength = 1_048_576
    public static let estimatedDownloadBytes: Int64 = 84_562_401_171

    public static let snapshotPatterns = [
        "LICENSE*",
        "README.md",
        "chat_template.jinja",
        "config.json",
        "conversion.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tiktoken/*",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "*.safetensors",
    ]

    public static let hubFallbackConfig = HubFallbackConfig(
        repoId: artifactRepoID,
        revision: artifactRevision,
        patterns: snapshotPatterns
    )

    public static func handles(modelSpec: String) -> Bool {
        let normalized = modelSpec.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }
        return normalized == modelID
            || normalized == artifactRepoID.lowercased()
            || normalized.contains("inkling-small")
    }

    public static func normalizedRootURL(
        _ rootURL: URL,
        fileManager: FileManager = .default
    ) -> URL {
        let standardized = rootURL.standardizedFileURL
        if fileManager.fileExists(atPath: standardized.appendingPathComponent("config.json").path) {
            return standardized
        }
        if let children = try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: standardized,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for child in children where
                fileManager.fileExists(atPath: child.appendingPathComponent("config.json").path) {
                return child.standardizedFileURL
            }
        }
        return standardized
    }

    public static func validate(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        let root = normalizedRootURL(rootURL, fileManager: fileManager)
        var missing = ["config.json", "tokenizer.json", "tokenizer_config.json"]
            .map { root.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
        let index = root.appendingPathComponent("model.safetensors.index.json")
        let single = root.appendingPathComponent("model.safetensors")
        if fileManager.fileExists(atPath: index.path) {
            do {
                let decoded = try JSONDecoder().decode(
                    HFSafetensorsIndex.self,
                    from: Data(contentsOf: index)
                )
                guard !decoded.shardFilenames.isEmpty else {
                    missing.append(index)
                    return missing
                }
                missing.append(contentsOf: decoded.shardFilenames
                    .map { root.appendingPathComponent($0) }
                    .filter { !fileManager.fileExists(atPath: $0.path) })
            } catch {
                missing.append(index)
            }
        } else if !fileManager.fileExists(atPath: single.path) {
            missing.append(index)
        }
        return missing
    }

    static func mapWeightKey(_ key: String) -> String {
        guard key.hasPrefix("language_model.") else {
            return "_ignored.\(key)"
        }
        return String(key.dropFirst("language_model.".count))
    }

    static func mapWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        guard !key.hasPrefix("_ignored.") else {
            return []
        }
        if key.hasSuffix(".conv.weight"),
           value.ndim == 3,
           value.dim(2) > value.dim(1) {
            return [(key, value.transposed(0, 2, 1))]
        }
        return [(key, value)]
    }
}
