import Foundation

public enum LagunaResources {
    public static let modelID = "text-chat-laguna-s-2-1"
    public static let upstreamModelID = "poolside/Laguna-S-2.1-NVFP4-mlx"
    public static let upstreamRevision = "e6a961c3bbebfffd8fa5b42f243e375504f41edd"
    public static let dflashModelID = "text-chat-laguna-s-2-1-dflash"
    public static let dflashUpstreamModelID = "poolside/Laguna-S-2.1-DFlash"
    public static let dflashUpstreamRevision = "b0486d1586daa0d56435c508108171fc1c8daff9"
    public static let estimatedDownloadBytes: Int64 = 71_905_947_593
    public static let dflashEstimatedDownloadBytes: Int64 = 2_229_973_462
    public static let defaultContextLength = 32_768
    public static let recommendedTemperature = 1.0
    public static let recommendedTopP = 1.0
    public static let recommendedTopK = 20
    public static let recommendedMinP = 0.02

    public static let snapshotPatterns = [
        "README.md",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "*.safetensors",
    ]

    public static let dflashSnapshotPatterns = [
        "README.md",
        "config.json",
        "model.safetensors",
    ]

    private static let requiredMetadataFiles = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    ]

    private static let requiredDFlashFiles = [
        "config.json",
        "model.safetensors",
    ]

    public static func isManagedIdentifier(_ raw: String) -> Bool {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == modelID
            || normalized == upstreamModelID.lowercased()
    }

    public static func handles(modelSpec raw: String) -> Bool {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let localName = URL(fileURLWithPath: normalized).lastPathComponent
        return isManagedIdentifier(normalized)
            || localName == upstreamModelID.split(separator: "/").last?.lowercased()
    }

    public static func missingTargetFiles(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        var missing = requiredMetadataFiles
            .map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }

        let indexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        guard fileManager.fileExists(atPath: indexURL.path) else {
            return missing
        }

        do {
            let index = try JSONDecoder().decode(
                HFSafetensorsIndex.self,
                from: Data(contentsOf: indexURL)
            )
            guard !index.shardFilenames.isEmpty else {
                missing.append(indexURL)
                return missing
            }
            missing.append(contentsOf: index.shardFilenames
                .map { rootURL.appendingPathComponent($0) }
                .filter { !fileManager.fileExists(atPath: $0.path) })
        } catch {
            missing.append(indexURL)
        }
        return missing
    }

    public static func missingDFlashFiles(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        requiredDFlashFiles
            .map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    static func validate(rootURL: URL) -> [String] {
        missingTargetFiles(rootURL: rootURL).map(\.lastPathComponent)
    }

    static func validateDFlash(rootURL: URL) -> [String] {
        missingDFlashFiles(rootURL: rootURL).map(\.lastPathComponent)
    }

    static func hasQuantizedSharedExperts(_ index: HFSafetensorsIndex) -> Bool {
        index.weightMap.keys.contains {
            $0.hasSuffix(".mlp.shared_expert.gate_proj.scales")
        }
    }

    public static func installedDFlashPath() -> String? {
        if let override = ProcessInfo.processInfo.environment["MERERUN_LAGUNA_DFLASH_PATH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !override.isEmpty {
            return override
        }
        return ManagedModelResolver.resolveInstalledModel(id: dflashModelID)?.path
    }
}
