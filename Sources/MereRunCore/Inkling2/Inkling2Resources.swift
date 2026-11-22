import Foundation

public enum Inkling2Resources {
    public static let modelID = "vision-chat-inkling-2-small"
    public static let repository = "thinkingmachines/Inkling-2-Small"
    public static let revision = "aac974122acd33ddb0d3e1f54433a9c5dd2249ce"
    public static let miniModelID = ManagedModelID.inkling2Mini.rawValue
    public static let miniRepository = "thinkingmachines/Inkling-2-Mini"
    public static let miniRevision = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
    public static let miniQ8HeadModelID = ManagedModelID.inkling2MiniQ8Head.rawValue
    public static let miniQ8HeadRepository = "Sawfwair/Inkling-2-Mini-Q8-Head"
    public static let miniQ8HeadRevision = "6e2909a5ed4956cec636e6517d365676086f90df"
    public static let defaultContextLength = 8_192
    public static let maximumContextLength = 1_048_576
    public static let snapshotPatterns = ["config.json", "hf_quant_config.json", "processor_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
        "model.safetensors.index.json", "model-*.safetensors", "README.md", "LICENSE*"]
    static let miniSnapshotPatterns = snapshotPatterns + ["mtp.safetensors"]
    static let miniQ8HeadSnapshotPatterns = miniSnapshotPatterns + [
        "generation_config.json", "tiktoken/tokenizer.model", "UPSTREAM_README.md"
    ]
    private static let miniModels = [miniModelID, miniRepository, miniQ8HeadModelID, miniQ8HeadRepository]

    public static func handles(modelSpec: String) -> Bool {
        ([modelID, repository, repository + "-MXFP8"] + miniModels)
            .contains { $0.caseInsensitiveCompare(modelSpec) == .orderedSame }
            || localModelRoot(for: modelSpec) != nil
    }

    static func isMini(modelSpec: String) -> Bool {
        if miniModels.contains(where: { $0.caseInsensitiveCompare(modelSpec) == .orderedSame }) {
            return true
        }
        return localCheckpoint(for: modelSpec)?.type == "inkling2_loop"
    }

    public static func localModelRoot(for modelSpec: String) -> String? {
        localCheckpoint(for: modelSpec)?.root.path
    }

    private static func localCheckpoint(for modelSpec: String) -> (root: URL, type: String)? {
        struct Envelope: Decodable { let model_type: String }
        let root = URL(fileURLWithPath: (modelSpec as NSString).expandingTildeInPath).standardizedFileURL
        guard let data = try? Data(contentsOf: root.appendingPathComponent("config.json")),
              let config = try? JSONDecoder().decode(Envelope.self, from: data),
              ["inkling2", "inkling2_loop"].contains(config.model_type) else { return nil }
        return (root, config.model_type)
    }

    static func missingFiles(root: URL, requiresDraft: Bool, fileManager: FileManager) -> [URL] {
        var metadata = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"]
        if requiresDraft { metadata.append("mtp.safetensors") }
        var missing = metadata.map { root.appendingPathComponent($0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: indexURL), let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data),
           !index.shardFilenames.isEmpty,
           index.shardFilenames.allSatisfy({ $0 == URL(fileURLWithPath: $0).lastPathComponent && $0.hasSuffix(".safetensors") }) {
            missing += index.shardFilenames.map { root.appendingPathComponent($0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        } else if !missing.contains(indexURL) { missing.append(indexURL) }
        return missing
    }
}
