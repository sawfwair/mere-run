// Native Swift/MLX reimplementation of the pinned LiquidAI D1 references.
// Modified for mere.run; see THIRD_PARTY_NOTICES.md and licenses/d1-LFM-OPEN-LICENSE.txt.
import Foundation
import MereRunD1Model

public enum D1Catalog {
    public static let modelID = "text-decide-d1-3b-bf16"
    public static let omniModelID = "text-decide-d1-omni-600m-fp32"
    public static let modelIDs = [modelID, omniModelID]
    public static let repository = "LiquidAI/d1-3B"
    public static let revision = "da1fe36a861f24690f27f622dca1d8688503d113"
    public static let omniRepository = "LiquidAI/d1-omni-600M"
    public static let omniRevision = "414f8d6438174f5b2133a9c21a478fc42625e308"
    public static func hubFallback(modelID: String) -> HubFallbackConfig {
        HubFallbackConfig(repoId: modelID == omniModelID ? omniRepository : repository,
            revision: modelID == omniModelID ? omniRevision : revision,
            patterns: ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors", "model.safetensors.index.json", "model-*.safetensors", "LICENSE", "README.md"])
    }
    static func configuration(root: URL) throws -> D1Configuration {
        let config = try JSONDecoder().decode(D1Configuration.self, from: Data(contentsOf: root.appending(path: "config.json")))
        try config.validate()
        return config
    }
    public static func isLocalCheckpoint(_ root: URL) -> Bool {
        struct Identity: Decodable { let model_type: String; let auto_map: [String: String]? }
        guard let data = try? Data(contentsOf: root.appending(path: "config.json")),
              let identity = try? JSONDecoder().decode(Identity.self, from: data) else { return false }
        return identity.model_type == "d1_omni" || identity.auto_map?["AutoModel"] == "modeling_d1.D1Model"
    }
    public static func validate(root: URL, modelID: String, fileManager: FileManager = .default) -> [URL] {
        let configURL = root.appending(path: "config.json")
        var missing = ["config.json", "tokenizer.json", "tokenizer_config.json"].map { root.appending(path: $0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
        let indexURL = root.appending(path: "model.safetensors.index.json")
        if fileManager.fileExists(atPath: indexURL.path) {
            do {
                let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: Data(contentsOf: indexURL))
                if index.weightMap.isEmpty { missing.append(indexURL) }
                missing += Set(index.weightMap.values).map { root.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
            } catch { missing.append(indexURL) }
        } else if !fileManager.fileExists(atPath: root.appending(path: "model.safetensors").path) { missing.append(root.appending(path: "model.safetensors")) }
        guard missing.isEmpty else { return missing }
        do {
            let config = try configuration(root: root)
            if modelIDs.contains(modelID), config.isOmni != (modelID == omniModelID) { return [configURL] }
        } catch { return [configURL] }
        return []
    }
}
