import Foundation

/// Metadata for the original BF16 SALM checkpoint. Neural computation belongs to AudioNemotronTranscribeModel.
public enum NemotronTranscribeResources {
    public static let modelID = "speech-asr-nemotron-35-transcribe"
    public static let repository = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
    public static let revision = "ef905575a106fdad904f3b0ce661dccad654607d"
    public static let weightBytes: Int64 = 67_496_013_734
    public static let files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]

    public static func handles(_ value: String) -> Bool {
        value == modelID || value.lowercased() == repository.lowercased()
    }

    public static func isLocalModel(_ root: URL) -> Bool {
        struct Identity: Decodable { let architectures: [String]? }
        guard let data = try? Data(contentsOf: root.appendingPathComponent("config.json")),
              let config = try? JSONDecoder().decode(Identity.self, from: data) else { return false }
        return config.architectures == ["Nemotron3_5TranscribeForConditionalGeneration"]
    }
}
