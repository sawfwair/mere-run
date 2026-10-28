import Foundation

public enum FlintResources {
    public static let modelID = "text-chat-flint-4bit"
    public static let bf16ModelID = "text-chat-flint-bf16"
    public static let repository = "RecursiveSI/flint-int4"
    public static let revision = "3b93f9053b530488405ae9fca51d4900d6ce21f0"
    public static let bf16Repository = "RecursiveSI/flint"
    public static let bf16Revision = "2a247be0c0aec895ccb0ac7ca92229ec6df37288"
    public static let defaultContextLength = 8_192
    public static let maximumContextLength = 262_144
    public static let snapshotPatterns = ["config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "LICENSE",
        "README.md", "*.safetensors", "model.safetensors.index.json", "flint-int4.json"]

    public static func handles(modelSpec: String) -> Bool {
        [modelID, bf16ModelID, repository, bf16Repository].contains { $0.caseInsensitiveCompare(modelSpec) == .orderedSame } || localModelRoot(for: modelSpec) != nil
    }

    public static func localModelRoot(for modelSpec: String) -> String? {
        struct Envelope: Decodable { let model_type: String }
        let root = URL(fileURLWithPath: modelSpec).standardizedFileURL
        guard let data = try? Data(contentsOf: root.appendingPathComponent("config.json")),
              let config = try? JSONDecoder().decode(Envelope.self, from: data), config.model_type == "flint" else { return nil }
        return root.path
    }

    static func missingFiles(root: URL, quantized: Bool, fileManager: FileManager) -> [URL] {
        let shards = quantized ? (0..<5).map { String(format: "model-%05d.safetensors", $0) }
            : (1...15).map { String(format: "model-%05d-of-00015.safetensors", $0) }
        let metadata = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
            + (quantized ? ["flint-int4.json", "special_tokens_map.json"] : ["model.safetensors.index.json"])
        return (metadata + shards).map { root.appendingPathComponent($0) }.filter { !fileManager.fileExists(atPath: $0.path) }
    }
}
