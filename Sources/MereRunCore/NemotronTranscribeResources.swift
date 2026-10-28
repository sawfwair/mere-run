import Foundation

/// Managed INT4/BF16 SALM metadata. Neural computation belongs to AudioNemotronTranscribeModel.
public enum NemotronTranscribeResources {
    public static let modelID = "speech-asr-nemotron-35-transcribe"
    public static let bf16ModelID = "speech-asr-nemotron-35-transcribe-bf16"
    public static let repository = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
    public static let revision = "ef905575a106fdad904f3b0ce661dccad654607d"
    public static let weightBytes: Int64 = 67_496_013_734
    public static let files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    public static let quantizedRepository = "Sawfwair/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-MLX-4bit"
    public static let quantizedRevision = "c20414875a76f4b9de213a4a084c9731069a29b3"
    public static let quantizedDownloadBytes: Int64 = 25_085_565_732
    public static let quantizedFiles = [
        "config.json", "tokenizer.json", "tokenizer_config.json", "expert-quantization.json",
        "model.safetensors.index.json", "MERERUN_CONVERSION.json"
    ] + (1...29).map { String(format: "model-%05d-of-00029.safetensors", $0) }
    public static let quantizedSnapshotPatterns = quantizedFiles + [
        "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/*.txt", "mererun_model.json",
        "UPSTREAM_MODEL_CARD.md", "MODIFICATIONS.md", "SHA256SUMS", "QUALIFICATION.md", "QUALIFICATION.json"
    ]

    public static func managedModelID(for value: String) -> String? {
        switch value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case modelID, quantizedRepository.lowercased(): modelID
        case bf16ModelID, repository.lowercased(): bf16ModelID
        default: nil
        }
    }

    public static func handles(_ value: String) -> Bool {
        managedModelID(for: value) != nil
    }

    public static func managedModelID(at root: URL) -> String {
        FileManager.default.fileExists(atPath: root.appendingPathComponent("expert-quantization.json").path) ? modelID : bf16ModelID
    }

    public static func missingFiles(modelID: String, root: URL, fileManager: FileManager) -> [URL] {
        let required = modelID == bf16ModelID ? files : quantizedFiles
        var missing = required.map { root.appendingPathComponent($0) }.filter { !fileManager.fileExists(atPath: $0.path) }
        let quantization = root.appendingPathComponent("expert-quantization.json")
        if modelID == bf16ModelID, fileManager.fileExists(atPath: quantization.path) {
            missing.append(quantization)
        }
        return missing
    }

    public static func isLocalModel(_ root: URL) -> Bool {
        struct Identity: Decodable { let architectures: [String]? }
        guard let data = try? Data(contentsOf: root.appendingPathComponent("config.json")),
              let config = try? JSONDecoder().decode(Identity.self, from: data) else { return false }
        return config.architectures == ["Nemotron3_5TranscribeForConditionalGeneration"]
    }
}
