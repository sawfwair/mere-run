import Foundation
import MereRunLayaModel

public enum LayaCatalog {
    public static let modelID = "text-decide-laya"
    public static let multilingualID = "text-decide-laya-multilingual"
    public static let typedDecisionsID = "text-decide-laya-typed-decisions"
    public static let repository = "convaiinnovations/laya"
    public static let revision = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
    public static let referenceRevision = "573e5b62696ba441230cd6be71d593331b5d23af"
    public static let modelIDs = [modelID, multilingualID, typedDecisionsID]
    public static let files = ["encoder/config.json", "rl_agent_config.json", "model.safetensors",
                               "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]

    public static func subfolder(modelID: String) -> String {
        switch modelID {
        case multilingualID: "multilingual"
        case typedDecisionsID: "typed-decisions"
        default: ""
        }
    }

    public static func checkpointRoot(_ root: URL, modelID: String) -> URL {
        let folder = subfolder(modelID: modelID)
        return folder.isEmpty ? root : root.appending(path: folder)
    }

    public static func hubFallback(modelID: String) -> HubFallbackConfig {
        let folder = subfolder(modelID: modelID)
        return HubFallbackConfig(repoId: repository, revision: revision,
                                 patterns: files.map { folder.isEmpty ? $0 : "\(folder)/\($0)" })
    }

    public static func validate(root: URL, modelID: String, fileManager: FileManager = .default) -> [URL] {
        let checkpoint = checkpointRoot(root, modelID: modelID)
        let missing = files.map { checkpoint.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        if !missing.isEmpty { return missing }
        do {
            _ = try LayaResources(root: checkpoint).configuration()
            return []
        } catch {
            return [checkpoint.appending(path: "rl_agent_config.json"), checkpoint.appending(path: "encoder/config.json")]
        }
    }
}

public struct LayaResources: Sendable {
    public let root: URL
    public init(root: URL) { self.root = root }

    public func configuration() throws -> (encoder: LayaEncoderConfiguration, agent: LayaAgentConfiguration) {
        let decoder = JSONDecoder()
        let encoder = try decoder.decode(LayaEncoderConfiguration.self, from: Data(contentsOf: root.appending(path: "encoder/config.json")))
        let agent = try decoder.decode(LayaAgentConfiguration.self, from: Data(contentsOf: root.appending(path: "rl_agent_config.json")))
        try agent.validate(encoder: encoder)
        return (encoder, agent)
    }
}
