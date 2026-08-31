import Foundation
import MLX

public struct LFM2Resources: Sendable, Hashable {
    public static let defaultModelId = "text-chat-lfm25-a1b-8bit"
    public static let upstreamRepoId = "LiquidAI/LFM2.5-8B-A1B-MLX-8bit"
    public static let upstreamRevision = "984aa3f7b00ab3deb00d987ae79b9bbe326eef3a"
    public static let a1bBF16ModelId = "text-chat-lfm25-a1b-bf16"
    public static let a1bBF16UpstreamRepoId = "LiquidAI/LFM2.5-8B-A1B"
    public static let a1bBF16UpstreamRevision = "b9aebfcbe28b6cb374042f495d733037550ab146"
    public static let a1bBF16EstimatedDownloadBytes: Int64 = 16_936_006_912
    public static let smallModelId = "text-chat-lfm25-1.2b-bf16"
    public static let smallUpstreamRepoId = "LiquidAI/LFM2.5-1.2B-Instruct"
    public static let smallUpstreamRevision = "df58c174f05ff733f83f8cae10ea9298224c8006"
    public static let smallEstimatedDownloadBytes: Int64 = 2_345_555_000
    public static let smallQADModelId = "text-chat-lfm25-1.2b-qad-4bit"
    public static let smallQADRepoId = "Sawfwair/LFM2.5-1.2B-Instruct-QAD-MLX-4bit"
    public static let smallQADRevision = "b1caaa502c5bfd975f5219162eaa90b1e7cc7839"
    public static let smallQADEstimatedDownloadBytes: Int64 = 761_967_712
    public static let denseModelId = "text-chat-lfm25-2.6b-4bit"
    public static let denseUpstreamRepoId = "LiquidAI/LFM2.5-2.6B-MLX"
    public static let denseUpstreamRevision = "58e239c769c4eb2b766fee80f0b7228bff837baf"
    public static let denseVariantSubdirectory = "4bit"
    public static let denseBF16ModelId = "text-chat-lfm25-2.6b-bf16"
    public static let denseBF16UpstreamRepoId = "LiquidAI/LFM2.5-2.6B"
    public static let denseBF16UpstreamRevision = "a334ee78cd38458bb71eda24109ac42dcec1309d"
    public static let denseBF16EstimatedDownloadBytes: Int64 = 5_394_427_456
    public static let denseQADModelId = "text-chat-lfm25-2.6b-qad-4bit"
    public static let denseQADRepoId = "Sawfwair/LFM2.5-2.6B-QAD-MLX-4bit"
    public static let denseQADRevision = "e829e540629759fdb886cb529e4d03640a11ffa7"
    public static let denseQADEstimatedDownloadBytes: Int64 = 1_753_796_391
    public static let visionModelId = "vision-chat-lfm25-3b-8bit"
    public static let visionUpstreamRepoId = "LiquidAI/LFM2.5-VL-3B-MLX-8bit"
    public static let visionUpstreamRevision = "4065d2c056a9c54d44fec67cf651812b55c6673f"
    public static let visionEstimatedDownloadBytes: Int64 = 3_736_739_700
    public static let defaultDSparkModelId = "text-chat-lfm25-a1b-dspark"
    public static let defaultDSparkRepoId = "LiquidAI/LFM2.5-8B-A1B-DSpark"
    public static let defaultDSparkRevision = "5b285c827912834665b1915f171897e49ff0f388"
    public static let defaultDSparkEstimatedDownloadBytes: Int64 = 655_433_500
    public static let smallDSparkModelId = "text-chat-lfm25-1.2b-dspark"
    public static let smallDSparkRepoId = "LiquidAI/LFM2.5-1.2B-Instruct-DSpark"
    public static let smallDSparkRevision = "4876d04848e15a6fd48d7c1481110e7cf5d62621"
    public static let smallDSparkEstimatedDownloadBytes: Int64 = 591_470_500
    public static let denseDSparkModelId = "text-chat-lfm25-2.6b-dspark"
    public static let denseDSparkRepoId = "LiquidAI/LFM2.5-2.6B-DSpark"
    public static let denseDSparkRevision = "458cedab07d0f7b2b05700c77e1aa463d43d6f04"
    public static let denseDSparkEstimatedDownloadBytes: Int64 = 655_433_500
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

    public static let dsparkSnapshotPatterns = [
        "LICENSE", "README.md", "config.json", "model.safetensors",
    ]

    public static let qadSnapshotPatterns = snapshotPatterns + ["MERERUN_CONVERSION.json"]

    public static let visionSnapshotPatterns = [
        "LICENSE*",
        "README.md",
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors",
        "model.safetensors.index.json",
        "*.safetensors",
    ]

    public static let managedModelIds = [
        defaultModelId,
        a1bBF16ModelId,
        smallModelId,
        smallQADModelId,
        denseModelId,
        denseBF16ModelId,
        denseQADModelId,
        visionModelId,
    ]
    public static let upstreamRepoIds = [
        upstreamRepoId,
        a1bBF16UpstreamRepoId,
        smallUpstreamRepoId,
        smallQADRepoId,
        denseUpstreamRepoId,
        denseBF16UpstreamRepoId,
        denseQADRepoId,
        visionUpstreamRepoId,
    ]

    public static func handles(modelSpec: String) -> Bool {
        let normalized = modelSpec.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }
        return managedModelIds.contains { normalized == $0.lowercased() }
            || upstreamRepoIds.contains { normalized == $0.lowercased() }
            || normalized.contains("lfm2")
            || normalized.contains("liquidai/")
    }

    public static func supportsTextLoRATraining(modelSpec: String) -> Bool {
        let normalized = modelSpec.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == defaultModelId.lowercased()
            || normalized == upstreamRepoId.lowercased()
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
    public var processorConfigURL: URL { rootURL.appending(path: "processor_config.json") }

    public func validate(
        fileManager: FileManager = .default,
        requireVisionProcessor: Bool = false
    ) -> [URL] {
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
        if requireVisionProcessor, !fileManager.fileExists(atPath: processorConfigURL.path) {
            missing.append(processorConfigURL)
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

    public static func dsparkModelID(
        for targetModelID: String,
        config: LFM2Config? = nil
    ) -> String? {
        _ = config
        let normalized = targetModelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized == a1bBF16ModelId || normalized == a1bBF16UpstreamRepoId.lowercased() {
            return defaultDSparkModelId
        }
        if normalized == smallModelId || normalized == smallUpstreamRepoId.lowercased() {
            return smallDSparkModelId
        }
        if normalized == denseBF16ModelId || normalized == denseBF16UpstreamRepoId.lowercased() {
            return denseDSparkModelId
        }
        return nil
    }

    public static func installedDSparkPath(
        for targetModelID: String,
        config: LFM2Config? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        if let override = environment["MERERUN_LFM25_DSPARK_PATH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !override.isEmpty {
            return override
        }
        guard let modelID = dsparkModelID(for: targetModelID, config: config) else {
            return nil
        }
        return ManagedModelResolver.resolveInstalledModel(id: modelID)?.path
    }

    public static func missingDSparkFiles(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        ["config.json", "model.safetensors"]
            .map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
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
