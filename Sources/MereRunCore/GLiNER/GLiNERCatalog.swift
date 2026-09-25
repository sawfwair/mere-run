import Foundation
import MereRunGLiNERModel

public enum GLiNERCatalog {
    public static let modelID = "text-classify-gliner25-decide"
    public static let repository = "fastino/GLiNER2.5-Decide"
    public static let revision = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"
    public static let referenceRevision = "55656fbfa01d3d4a77485e1a1eeeaf682990ccdf"
    public static let files = ["config.json", "encoder_config/config.json", "model.safetensors",
                               "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

    public static let hubFallback = HubFallbackConfig(repoId: repository, revision: revision, patterns: files)

    public static func validate(root: URL, fileManager: FileManager = .default) -> [URL] {
        let missing = files.map { root.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        if !missing.isEmpty { return missing }
        do {
            let config = try JSONDecoder().decode(GLiNEREncoderConfiguration.self,
                from: Data(contentsOf: root.appending(path: "encoder_config/config.json")))
            try config.validate()
            return []
        } catch {
            return [root.appending(path: "encoder_config/config.json")]
        }
    }
}
