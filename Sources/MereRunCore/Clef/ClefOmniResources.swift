import Foundation
import MereRunQwenModel

public enum ClefOmniCatalog {
    public static let modelID = "text-decide-clef-omni"
    public static let repository = "Cloudflare/clef-omni"
    public static let revision = "0db1cd2607d76a7bdb2a382f659e7b313079f84b"
    public static let hubFallback = HubFallbackConfig(repoId: repository, revision: revision, patterns: ClefCatalog.files)

    package static func validate(root: URL, fileManager: FileManager = .default) -> [URL] {
        let required = ["config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json",
                        "joint_head_config.json", "joint_head.safetensors", "model.safetensors.index.json"]
        let missing = required.map { root.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        guard missing.isEmpty else { return missing }
        do {
            let config = try Qwen3OmniConfiguration.decode(Data(contentsOf: root.appending(path: "config.json")))
            let head = try JSONDecoder().decode(
                ClefHeadConfiguration.self,
                from: Data(contentsOf: root.appending(path: "joint_head_config.json"))
            )
            try head.validate(backboneHiddenSize: config.thinkerConfig.textConfig.hiddenSize)
            let indexURL = root.appending(path: "model.safetensors.index.json")
            let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: Data(contentsOf: indexURL))
            guard index.weightMap["thinker.model.embed_tokens.weight"] != nil,
                  index.weightMap["thinker.lm_head.weight"] != nil else { return [indexURL] }
            return Set(index.weightMap.filter { $0.key.hasPrefix("thinker.") }.values)
                .map { root.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        } catch {
            return [root.appending(path: "config.json")]
        }
    }
}
