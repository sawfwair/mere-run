import Foundation

public struct MuseGlimmerResources: Sendable, Hashable {
    public static let modelId = "vision-chat-muse-glimmer-30b"
    public static let upstreamRepoId = "meta-models/Muse-Glimmer-30B"
    public static let upstreamRevision = "f84ecc3a0ea984a4c04542a84269e3d065350a6e"
    public static let artifactRepoId = "Sawfwair/Muse-Glimmer-30B-MLX-4bit"
    public static let artifactRevision = "6532e898dc5c1a55b51b1b108cd36728b79be751"
    public static let defaultContextLength = 131_072
    public static let estimatedDownloadBytes: Int64 = 21_376_480_452
    public static let recommendedTemperature = 1.0
    public static let recommendedTopP = 0.95
    public static let recommendedTopK = 64
    public static let assistantModelId = "vision-chat-muse-glimmer-30b-assistant"
    public static let assistantQuantizedModelId = "vision-chat-muse-glimmer-30b-assistant-q4"
    public static let assistantUpstreamRepoId = "meta-models/Muse-Glimmer-30B-assistant"
    public static let assistantUpstreamRevision = "2c86316d689027b91123638739743fef1d425233"
    public static let assistantEstimatedDownloadBytes: Int64 = 5_111_976_608
    public static let assistantSnapshotPatterns = [
        "LICENSE",
        "README.md",
        "USAGE_POLICY.md",
        "config.json",
        "MERERUN_CONVERSION.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "*.safetensors",
    ]
    public static let snapshotPatterns = [
        "LICENSE",
        "README.md",
        "USAGE_POLICY.md",
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "MERERUN_CONVERSION.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "*.safetensors",
    ]

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL.standardizedFileURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var generationConfigURL: URL { rootURL.appending(path: "generation_config.json") }
    public var processorConfigURL: URL { rootURL.appending(path: "processor_config.json") }
    public var tokenizerURL: URL { rootURL.appending(path: "tokenizer.json") }
    public var tokenizerConfigURL: URL { rootURL.appending(path: "tokenizer_config.json") }
    public var chatTemplateURL: URL { rootURL.appending(path: "chat_template.jinja") }
    public var modelIndexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }
    public var modelWeightsURL: URL { rootURL.appending(path: "model.safetensors") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing = [configURL, generationConfigURL, processorConfigURL, tokenizerURL, tokenizerConfigURL]
            .filter { !fileManager.fileExists(atPath: $0.path) }
        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle {
            missing.append(modelIndexURL)
        }
        return missing
    }

    public static func handles(modelSpec raw: String) -> Bool {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == modelId
            || normalized == upstreamRepoId.lowercased()
            || normalized == artifactRepoId.lowercased()
            || normalized.contains("muse-glimmer")
            || normalized.contains("muse_glimmer")
    }

    public static func isLikelyHubRepoID(_ raw: String) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmed.isEmpty
            && !trimmed.hasPrefix("/")
            && !trimmed.hasPrefix("~")
            && !trimmed.hasPrefix(".")
            && trimmed.split(separator: "/").count == 2
    }

    public static func validateAssistant(
        rootURL: URL,
        fileManager: FileManager = .default
    ) -> [URL] {
        let config = rootURL.appending(path: "config.json")
        let single = rootURL.appending(path: "model.safetensors")
        let index = rootURL.appending(path: "model.safetensors.index.json")
        var missing = fileManager.fileExists(atPath: config.path) ? [] : [config]
        if !fileManager.fileExists(atPath: single.path),
           !fileManager.fileExists(atPath: index.path) {
            missing.append(index)
        }
        return missing
    }
}

public enum MuseGlimmerError: LocalizedError {
    case modelNotLoaded
    case missingFiles([String])
    case generationFailed(String)
    case unsupportedModelID(String)
    case unsupportedConfiguration(String)
    case unsupportedModelLocation(String)
    case downloadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Muse Glimmer is not loaded."
        case .missingFiles(let files):
            return "Missing required Muse Glimmer files: \(files.joined(separator: ", "))."
        case .generationFailed(let message):
            return message
        case .unsupportedModelID(let modelID):
            return "Unsupported Muse Glimmer model id: \(modelID)"
        case .unsupportedConfiguration(let message):
            return message
        case .unsupportedModelLocation(let location):
            return "Could not resolve Muse Glimmer model location: \(location)"
        case .downloadFailed(let message):
            return "Muse Glimmer download failed: \(message)"
        }
    }
}
