import Foundation

public enum Inkling2Resources {
    public static let modelID = "vision-chat-inkling-2-small"
    public static let repository = "thinkingmachines/Inkling-2-Small"
    public static let revision = "aac974122acd33ddb0d3e1f54433a9c5dd2249ce"
    public static let defaultContextLength = 8_192
    public static let maximumContextLength = 1_048_576
    public static let snapshotPatterns = ["config.json", "hf_quant_config.json", "processor_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
        "model.safetensors.index.json", "model-*.safetensors", "README.md", "LICENSE*"]

    public static func handles(modelSpec: String) -> Bool {
        [modelID, repository, repository + "-MXFP8"].contains { $0.caseInsensitiveCompare(modelSpec) == .orderedSame }
            || localModelRoot(for: modelSpec) != nil
    }

    public static func localModelRoot(for modelSpec: String) -> String? {
        struct Envelope: Decodable { let model_type: String }
        let root = URL(fileURLWithPath: (modelSpec as NSString).expandingTildeInPath).standardizedFileURL
        guard let data = try? Data(contentsOf: root.appendingPathComponent("config.json")),
              let config = try? JSONDecoder().decode(Envelope.self, from: data), config.model_type == "inkling2" else { return nil }
        return root.path
    }

    static func missingFiles(root: URL, fileManager: FileManager) -> [URL] {
        let metadata = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"]
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
