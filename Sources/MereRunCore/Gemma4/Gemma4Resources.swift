import Foundation

public struct Gemma4Resources: Sendable, Hashable {
    public static let defaultModelId = "text-chat-gemma4"
    public static let nanoModelId = "text-chat-gemma4-nano"
    public static let maxModelId = "text-chat-gemma4-max"
    public static let turboModelId = "text-chat-gemma4-turbo"
    public static let twelveBModelId = "text-chat-gemma4-12b"
    public static let visionTwelveBModelId = "vision-chat-gemma4-12b"
    public static let nanoUpstreamModelId = "google/gemma-4-E4B-it"
    public static let maxUpstreamModelId = "google/gemma-4-31B-it"
    public static let turboUpstreamModelId = "mlx-community/gemma-4-26b-a4b-it-nvfp4"
    public static let twelveBUpstreamModelId = "google/gemma-4-12B-it"
    public static let defaultUpstreamModelId = maxUpstreamModelId
    public static let defaultContextLength = 32_768
    public static let defaultKVGroupSize = 64
    public static let defaultQuantizedKVStart = 5_000
    public static let defaultKVQuantizationScheme: Gemma4KVQuantizationScheme = .uniform
    public static let defaultTurboKVBits = 4.0
    public static let defaultTurboQuantizedKVStart = 0
    public static let defaultTurboKVQuantizationScheme: Gemma4KVQuantizationScheme = .turboquant
    public static var supportsDefaultTurboKVQuantization: Bool {
        #if os(Linux)
        false
        #else
        true
        #endif
    }
    public static let snapshotPatterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
        "chat_template.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "*.safetensors",
    ]

    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var modelIndexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }
    public var modelWeightsURL: URL { rootURL.appending(path: "model.safetensors") }
    public var tokenizerURL: URL { rootURL.appending(path: "tokenizer.json") }
    public var tokenizerConfigURL: URL { rootURL.appending(path: "tokenizer_config.json") }
    public var processorConfigURL: URL { rootURL.appending(path: "processor_config.json") }
    public var preprocessorConfigURL: URL { rootURL.appending(path: "preprocessor_config.json") }
    public var chatTemplateURL: URL { rootURL.appending(path: "chat_template.jinja") }

    public func validate(fileManager: FileManager = .default, requireUnifiedProcessor: Bool = false) -> [URL] {
        var missing: [URL] = []

        if !fileManager.fileExists(atPath: configURL.path) {
            missing.append(configURL)
        }

        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle {
            missing.append(modelIndexURL)
        }

        if !fileManager.fileExists(atPath: tokenizerURL.path) {
            missing.append(tokenizerURL)
        }

        if !fileManager.fileExists(atPath: tokenizerConfigURL.path) {
            missing.append(tokenizerConfigURL)
        }

        if requireUnifiedProcessor,
           !fileManager.fileExists(atPath: processorConfigURL.path),
           !fileManager.fileExists(atPath: preprocessorConfigURL.path) {
            missing.append(processorConfigURL)
        }

        return missing
    }

    public static func normalizedRootURL(_ rootURL: URL) -> URL {
        rootURL.standardizedFileURL
    }

    public static func handles(modelSpec raw: String) -> Bool {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if normalized.isEmpty || normalized == defaultModelId {
            return true
        }
        return normalized.contains("gemma-4") || normalized.contains("gemma4")
    }

    public static func isLikelyHubRepoID(_ raw: String) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard !trimmed.hasPrefix("/") && !trimmed.hasPrefix("~") && !trimmed.hasPrefix(".") else {
            return false
        }
        return trimmed.split(separator: "/").count == 2
    }

    public static func usesTurboDefaults(modelSpec raw: String) -> Bool {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return normalized == turboModelId || normalized == turboUpstreamModelId.lowercased()
    }

    public static func supportsVision(modelSpec raw: String) -> Bool {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return normalized == visionTwelveBModelId
            || normalized == twelveBUpstreamModelId.lowercased()
    }
}

public enum Gemma4Error: LocalizedError {
    case modelNotLoaded
    case missingFiles([String])
    case unsupportedConfiguration(String)
    case unsupportedModelLocation(String)
    case downloadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Gemma4 model is not loaded."
        case .missingFiles(let files):
            return "Missing required Gemma4 files: \(files.joined(separator: ", "))"
        case .unsupportedConfiguration(let message):
            return message
        case .unsupportedModelLocation(let location):
            return "Could not resolve Gemma4 model location: \(location)"
        case .downloadFailed(let message):
            return "Gemma4 download failed: \(message)"
        }
    }
}
