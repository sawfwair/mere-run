import Foundation

public enum NemotronOmniError: LocalizedError {
    case modelPathRequired
    case missingFiles([String])
    case invalidWeights(String)
    case unsupportedMedia(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelPathRequired:
            "Nemotron 3 Nano Omni requires the installed BF16 model or an explicit model path."
        case .missingFiles(let files):
            "Nemotron 3 Nano Omni is missing required files: \(files.joined(separator: ", "))."
        case .invalidWeights(let message):
            "Nemotron 3 Nano Omni checkpoint weights are incompatible: \(message)"
        case .unsupportedMedia(let message):
            "Nemotron 3 Nano Omni media input is unsupported: \(message)"
        case .generationFailed(let message):
            "Nemotron 3 Nano Omni generation failed: \(message)"
        }
    }
}

/// Pinned artifacts and runtime limits for NVIDIA Nemotron 3 Nano Omni.
///
/// The upstream repository contains executable Python reference code. mere.run
/// downloads it only as provenance for the published checkpoint contract; the
/// native runtime never imports or executes repository code.
public enum NemotronOmniResources {
    public static let modelID = "omni-chat-nemotron3-nano-30b-a3b-bf16"
    public static let upstreamRepoID =
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    public static let upstreamRevision = "24e67ea000b7c2837fc8f9488aa2008524fac8ba"
    public static let estimatedDownloadBytes: Int64 = 66_059_015_328
    public static let checkpointWeightBytes: Int64 = 66_031_270_520
    public static let checkpointShardCount = 17
    public static let packedExpertWeightBytes: Int64 = 58_749_616_128

    /// The composed checkpoint advertises 131,072 tokens even though the
    /// underlying Nemotron-H text backbone has 262,144 positional slots.
    public static let defaultContextLength = 32_768
    public static let maximumContextLength = 131_072
    public static let maximumOutputTokens = 20_480
    public static let maximumVideoDurationSeconds = 120
    public static let maximumAudioDurationSeconds = 3_600
    public static let audioSampleRate = 16_000
    public static let maximumVideoFrames1080p = 128
    public static let maximumVideoFrames720p = 256

    public static let thinkingTemperature = 0.6
    public static let thinkingTopP = 0.95
    public static let instructTemperature = 0.2
    public static let instructTopK = 1

    public static let snapshotPatterns = [
        "README.md",
        "bias.md",
        "explainability.md",
        "privacy.md",
        "safety.md",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
        // Reference implementation files are retained for pinned provenance.
        "__init__.py",
        "audio_model.py",
        "configuration.py",
        "configuration_nemotron_h.py",
        "configuration_radio.py",
        "evs.py",
        "image_processing.py",
        "modeling.py",
        "modeling_nemotron_h.py",
        "processing.py",
        "processing_utils.py",
        "video_io.py",
        "video_processing.py",
    ]

    public static func handles(modelSpec raw: String) -> Bool {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let localName = URL(fileURLWithPath: normalized).lastPathComponent
        return normalized == modelID
            || normalized == upstreamRepoID.lowercased()
            || localName == upstreamRepoID.split(separator: "/").last?.lowercased()
    }

    public static func missingTargetFiles(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        let required = [
            "README.md",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ]
        var missing = required.map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
        let indexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        guard fileManager.fileExists(atPath: indexURL.path),
              let index = try? JSONDecoder().decode(
                HFSafetensorsIndex.self,
                from: Data(contentsOf: indexURL)
              ) else {
            return missing
        }
        missing += index.shardFilenames.map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
        return missing
    }

    public static func validationMessages(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> [String] {
        let missing = missingTargetFiles(rootURL: rootURL, fileManager: fileManager)
        guard missing.isEmpty else {
            return missing.map { "Missing required file: \($0.path)" }
        }

        var messages: [String] = []
        do {
            _ = try JSONDecoder().decode(
                NemotronOmniConfig.self,
                from: Data(contentsOf: rootURL.appendingPathComponent("config.json"))
            )
        } catch {
            messages.append("Invalid pinned Nemotron Omni config.json: \(error.localizedDescription)")
        }
        do {
            _ = try JSONDecoder().decode(
                NemotronOmniPreprocessorConfig.self,
                from: Data(contentsOf: rootURL.appendingPathComponent("preprocessor_config.json"))
            )
        } catch {
            messages.append(
                "Invalid pinned Nemotron Omni preprocessor_config.json: \(error.localizedDescription)"
            )
        }
        do {
            let index = try JSONDecoder().decode(
                HFSafetensorsIndex.self,
                from: Data(contentsOf: rootURL.appendingPathComponent("model.safetensors.index.json"))
            )
            let expectedShards = (1...checkpointShardCount).map {
                String(format: "model-%05d-of-%05d.safetensors", $0, checkpointShardCount)
            }
            guard index.metadata?.totalSize == checkpointWeightBytes,
                  index.shardFilenames == expectedShards else {
                messages.append(
                    "Safetensors index does not match the pinned 17-shard, "
                        + "\(checkpointWeightBytes)-byte checkpoint contract."
                )
                return messages
            }
        } catch {
            messages.append("Invalid pinned Nemotron Omni safetensors index: \(error.localizedDescription)")
        }
        return messages
    }
}
