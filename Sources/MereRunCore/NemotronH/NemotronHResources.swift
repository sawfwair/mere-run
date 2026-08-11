import Foundation

public enum NemotronHError: LocalizedError {
    case modelPathRequired
    case missingFiles([String])
    case incompatibleDSpark(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelPathRequired:
            "Nemotron 3.5 Lightning requires an installed managed model or an explicit model path."
        case .missingFiles(let files):
            "Nemotron 3.5 Lightning is missing required files: \(files.joined(separator: ", "))."
        case .incompatibleDSpark(let message):
            "Nemotron 3.5 Lightning DSpark is incompatible: \(message)"
        case .generationFailed(let message):
            "Nemotron 3.5 Lightning generation failed: \(message)"
        }
    }
}

public enum NemotronHResources {
    public static let modelID = "text-chat-nemotron-35-lightning"
    public static let artifactRepoID =
        "Sawfwair/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-MLX"
    public static let artifactRevision = "6699e5fd3f0c5b392bb3f8bac2443276bb41958a"
    public static let estimatedDownloadBytes: Int64 = 19_797_633_387
    public static let upstreamRepoID =
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    public static let upstreamRevision = "e0b753dc24903ad4d62f5696077da22020eca89a"
    public static let dsparkModelID = "text-chat-nemotron-35-lightning-dspark"
    public static let dsparkArtifactRepoID =
        "Sawfwair/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-MLX"
    public static let dsparkArtifactRevision = "d30f0914d6bbb6da36302bd9228f92824901e675"
    public static let dsparkEstimatedDownloadBytes: Int64 = 1_349_081_120
    public static let dsparkUpstreamRepoID =
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark"
    public static let dsparkUpstreamRevision = "e3af76fbff445ef795958bee96bc1126af70fd57"
    public static let defaultContextLength = 32_768
    public static let maximumContextLength = 262_144
    public static let recommendedTemperature = 1.0
    public static let recommendedTopP = 0.95
    /// NVIDIA's published single-DGX-Spark recipe uses a three-token DSpark
    /// proposal. The checkpoint's block size is a maximum query canvas, not a
    /// requirement to fill all seven mask positions on every round.
    public static let defaultSpeculativeTokens = 3

    public static let snapshotPatterns = [
        "LICENSE", "README.md", "chat_template.jinja", "config.json",
        "generation_config.json", "mererun_model.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json",
        "model.safetensors.index.json", "*.safetensors",
    ]
    public static let dsparkSnapshotPatterns = [
        "LICENSE", "README.md", "config.json", "mererun_model.json",
        "model.safetensors",
    ]

    public static func handles(modelSpec raw: String) -> Bool {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let localName = URL(fileURLWithPath: normalized).lastPathComponent
        return normalized == modelID
            || normalized == artifactRepoID.lowercased()
            || normalized == upstreamRepoID.lowercased()
            || localName == artifactRepoID.split(separator: "/").last?.lowercased()
    }

    public static func missingTargetFiles(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        let required = [
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "chat_template.jinja", "model.safetensors.index.json",
        ]
        var missing = required.map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
        let indexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        guard fileManager.fileExists(atPath: indexURL.path),
              let index = try? JSONDecoder().decode(
                HFSafetensorsIndex.self,
                from: Data(contentsOf: indexURL)
              ) else {
            return missing
        }
        missing += index.shardFilenames.map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
        return missing
    }

    public static func missingDSparkFiles(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        ["config.json", "model.safetensors"]
            .map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public static func installedDSparkPath() -> String? {
        if let override = ProcessInfo.processInfo.environment[
            "MERERUN_NEMOTRON35_DSPARK_PATH"
        ]?.trimmingCharacters(in: .whitespacesAndNewlines), !override.isEmpty {
            return override
        }
        return ManagedModelResolver.resolveInstalledModel(id: dsparkModelID)?.path
    }
}
