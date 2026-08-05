import Foundation
import MLX

public struct LFM2Resources: Sendable, Hashable {
    public static let defaultModelId = "text-chat-lfm25-a1b-8bit"
    public static let upstreamRepoId = "LiquidAI/LFM2.5-8B-A1B-MLX-8bit"
    public static let upstreamRevision = "984aa3f7b00ab3deb00d987ae79b9bbe326eef3a"
    public static let denseModelId = "text-chat-lfm25-2.6b-4bit"
    public static let denseUpstreamRepoId = "LiquidAI/LFM2.5-2.6B-MLX"
    public static let denseUpstreamRevision = "58e239c769c4eb2b766fee80f0b7228bff837baf"
    public static let denseVariantSubdirectory = "4bit"
    public static let defaultContextLength = 32_768

    public static let snapshotPatterns = [
        "LICENSE*",
        "README.md",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors",
        "model.safetensors.index.json",
        "*.safetensors",
    ]

    public static let denseSnapshotPatterns = [
        "LICENSE*",
        "README.md",
        "\(denseVariantSubdirectory)/chat_template.jinja",
        "\(denseVariantSubdirectory)/config.json",
        "\(denseVariantSubdirectory)/generation_config.json",
        "\(denseVariantSubdirectory)/tokenizer.json",
        "\(denseVariantSubdirectory)/tokenizer_config.json",
        "\(denseVariantSubdirectory)/model.safetensors",
        "\(denseVariantSubdirectory)/model.safetensors.index.json",
    ]

    public static let managedModelIds = [defaultModelId, denseModelId]
    public static let upstreamRepoIds = [upstreamRepoId, denseUpstreamRepoId]

    public static func handles(modelSpec: String) -> Bool {
        let normalized = modelSpec.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }
        return managedModelIds.contains { normalized == $0.lowercased() }
            || upstreamRepoIds.contains { normalized == $0.lowercased() }
            || normalized.contains("lfm2")
            || normalized.contains("liquidai/")
    }

    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var modelIndexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }
    public var modelWeightsURL: URL { rootURL.appending(path: "model.safetensors") }
    public var tokenizerURL: URL { rootURL.appending(path: "tokenizer.json") }
    public var tokenizerConfigURL: URL { rootURL.appending(path: "tokenizer_config.json") }
    public var chatTemplateURL: URL { rootURL.appending(path: "chat_template.jinja") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        if !fileManager.fileExists(atPath: configURL.path) {
            missing.append(configURL)
        }
        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle {
            missing.append(modelIndexURL)
        }
        if !fileManager.fileExists(atPath: tokenizerURL.path) {
            missing.append(tokenizerURL)
        }
        if !fileManager.fileExists(atPath: tokenizerConfigURL.path) {
            missing.append(tokenizerConfigURL)
        }
        return missing
    }

    public static func normalizedRootURL(_ rootURL: URL, fileManager: FileManager = .default) -> URL {
        let standardized = rootURL.standardizedFileURL
        if fileManager.fileExists(atPath: standardized.appendingPathComponent("config.json").path) {
            return standardized
        }
        let denseVariant = standardized.appendingPathComponent(denseVariantSubdirectory, isDirectory: true)
        if fileManager.fileExists(atPath: denseVariant.appendingPathComponent("config.json").path) {
            return denseVariant.standardizedFileURL
        }
        if let children = try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: standardized,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for child in children {
                if fileManager.fileExists(atPath: child.appendingPathComponent("config.json").path) {
                    return child.standardizedFileURL
                }
            }
        }
        return standardized
    }

    static func mapWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key.hasSuffix(".conv.conv.weight"),
           value.ndim == 3,
           value.dim(2) > value.dim(1) {
            return [(key, value.transposed(0, 2, 1))]
        }
        return [(key, value)]
    }
}
