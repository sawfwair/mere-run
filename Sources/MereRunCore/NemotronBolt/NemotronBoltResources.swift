import Foundation
import MereRunNemotronBoltModel

public enum NemotronBoltResources {
    public static let modelID = "vision-chat-nemotron-35-bolt-24gb"
    public static let artifactRepoID =
        "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
    public static let artifactRevision = "96a5d63027ac013c8ef37fb438f3b4ec5fbd3ba6"
    public static let estimatedDownloadBytes: Int64 = 24_120_000_000
    public static let upstreamRepoID =
        "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
    public static let upstreamRevision = "96a5d63027ac013c8ef37fb438f3b4ec5fbd3ba6"
    public static let defaultContextLength = 32_768
    public static let maximumContextLength = 262_144
    public static let recommendedTemperature = 1.0
    public static let recommendedTopP = 0.95
    public static let snapshotPatterns = [
        "LICENSE", "README.md", "chat_template.jinja", "config.json",
        "generation_config.json", "preprocessor_config.json", "hf_quant_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json",
        "model.safetensors.index.json", "*.safetensors",
    ]
    public static func validationMessages(rootURL: URL, fileManager: FileManager = .default) -> [String] {
        do {
            _ = try NemotronBoltConfig.load(at: rootURL)
            _ = try JSONDecoder().decode(NemotronOmniPreprocessorConfig.self,
                from: Data(contentsOf: rootURL.appendingPathComponent("preprocessor_config.json")))
            let index = try JSONDecoder().decode(HFSafetensorsIndex.self,
                from: Data(contentsOf: rootURL.appendingPathComponent("model.safetensors.index.json")))
            let expected = (1...90).map { String(format: "model-%05d-of-00090.safetensors", $0) }
            guard index.shardFilenames == expected,
                  index.metadata?.totalSize == NemotronBoltConfig.checkpointWeightBytes else {
                return ["Index differs from the pinned 90-shard Bolt checkpoint."]
            }
            return missingTargetFiles(rootURL: rootURL, fileManager: fileManager)
                .map { "Missing required file: \($0.lastPathComponent)" }
        } catch { return [error.localizedDescription] }
    }

    public static func handles(modelSpec raw: String) -> Bool {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let localName = URL(fileURLWithPath: normalized).lastPathComponent
        if raw.hasPrefix("/") || raw.hasPrefix(".") {
            let root = URL(fileURLWithPath: raw).standardizedFileURL
            if (try? NemotronBoltConfig.load(at: root)) != nil { return true }
        }
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
            "chat_template.jinja", "preprocessor_config.json", "model.safetensors.index.json",
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

}
