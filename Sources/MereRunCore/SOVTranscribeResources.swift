import Foundation

public enum SOVTranscribeResources {
    public static let modelID = "speech-asr-sov-transcribe-1"
    public static let draftModelID = "speech-asr-sov-transcribe-1-dflash"
    public static let repository = "salesforce-sovereign-intelligence/SOV-Transcribe-1"
    public static let draftRepository = "salesforce-sovereign-intelligence/SOV-Transcribe-1-DFlash"
    public static let revision = "505d8fae243924c0d8acdc622f7f0abab344eba3"
    public static let draftRevision = "8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05"
    public static let patterns = ["config.json", "model.safetensors.index.json", "model*.safetensors", "tokenizer.json",
                                  "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                                  "processor_config.json", "preprocessor_config.json", "generation_config.json", "README.md", "LICENSE*"]

    public static func handles(_ value: String) -> Bool {
        [modelID, repository.lowercased()].contains(value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }

    public static func isDraft(_ value: String) -> Bool {
        if [draftModelID, draftRepository.lowercased()].contains(value.lowercased()) { return true }
        struct DraftIdentity: Decodable { let architectures: [String]? }
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: value).appendingPathComponent("config.json")),
              let config = try? JSONDecoder().decode(DraftIdentity.self, from: data) else { return false }
        return config.architectures?.contains("DFlashDraftModel") == true
    }

    public static func isLocalModel(_ root: URL) -> Bool {
        struct Identity: Decodable { let model_type: String }
        guard let data = try? Data(contentsOf: root.appendingPathComponent("config.json")),
              let identity = try? JSONDecoder().decode(Identity.self, from: data) else { return false }
        return identity.model_type == "sovereignai_transcribe"
    }

    public static func missingFiles(root: URL, draft: Bool, fileManager: FileManager) -> [URL] {
        let names = draft ? ["config.json"] : ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "processor_config.json", "generation_config.json"]
        var missing = names.map { root.appendingPathComponent($0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        let bundle = root.appendingPathComponent(SOVTranscribeBundleManifest.filename)
        if !draft, fileManager.fileExists(atPath: bundle.path) {
            do {
                let manifest = try JSONDecoder().decode(SOVTranscribeBundleManifest.self, from: Data(contentsOf: bundle))
                try manifest.validate()
                return missing + ([manifest.audio] + manifest.shards).map { root.appendingPathComponent($0.name) }
                    .filter { !fileManager.fileExists(atPath: $0.path) }
            } catch { return missing + [bundle] }
        }
        let index = root.appendingPathComponent("model.safetensors.index.json")
        let weights = root.appendingPathComponent("model.safetensors")
        if fileManager.fileExists(atPath: index.path) {
            if let data = try? Data(contentsOf: index), let decoded = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data),
               !decoded.weightMap.isEmpty {
                missing += decoded.shardFilenames.map { root.appendingPathComponent($0) }.filter { !fileManager.fileExists(atPath: $0.path) }
            } else { missing.append(index) }
        } else if !fileManager.fileExists(atPath: weights.path) { missing.append(weights) }
        return missing
    }
}
