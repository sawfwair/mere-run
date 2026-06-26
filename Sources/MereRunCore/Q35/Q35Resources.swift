import Foundation

public struct Q35Resources: Sendable, Hashable {
    public struct Profile: Sendable, Hashable {
        public let modelId: String
        public let upstreamRepoId: String
        public let upstreamRevision: String

        public init(
            modelId: String,
            upstreamRepoId: String,
            upstreamRevision: String
        ) {
            self.modelId = modelId
            self.upstreamRepoId = upstreamRepoId
            self.upstreamRevision = upstreamRevision
        }

        public var hubFallbackConfig: HubFallbackConfig {
            HubFallbackConfig(
                repoId: upstreamRepoId,
                revision: upstreamRevision,
                patterns: Q35Resources.snapshotPatterns
            )
        }
    }

    public static let q36NanoModelId = "text-chat-q36-nano"
    public static let infinityParser2FlashModelId = "vision-ocr-infinity-flash"
    public static let infinityParser2ProModelId = "vision-ocr-infinity-pro"
    public static let infinityParser2ProInt8ModelId = "vision-ocr-infinity-pro-int8"
    public static let defaultModelId = q36NanoModelId

    public static let q36NanoUpstreamRepoId = "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    public static let q36NanoUpstreamRevision = "63d520640ca7461f31ba66104612135770090340"
    public static let infinityParser2FlashUpstreamRepoId = "infly/Infinity-Parser2-Flash"
    public static let infinityParser2FlashUpstreamRevision = "30f02aca5ee7c32d22a962cbece5c19611147c9e"
    public static let infinityParser2ProUpstreamRepoId = "infly/Infinity-Parser2-Pro"
    public static let infinityParser2ProUpstreamRevision = "1d070df7db5acca0ffa75596229070a047704f89"
    public static let infinityParser2ProInt8UpstreamRepoId = "Sawfwair/Infinity-Parser2-Pro-Int8"
    public static let infinityParser2ProInt8UpstreamRevision = "main"

    private static let profilesByModelId: [String: Profile] = [
        q36NanoModelId: Profile(
            modelId: q36NanoModelId,
            upstreamRepoId: q36NanoUpstreamRepoId,
            upstreamRevision: q36NanoUpstreamRevision
        ),
        infinityParser2FlashModelId: Profile(
            modelId: infinityParser2FlashModelId,
            upstreamRepoId: infinityParser2FlashUpstreamRepoId,
            upstreamRevision: infinityParser2FlashUpstreamRevision
        ),
        infinityParser2ProModelId: Profile(
            modelId: infinityParser2ProModelId,
            upstreamRepoId: infinityParser2ProUpstreamRepoId,
            upstreamRevision: infinityParser2ProUpstreamRevision
        ),
        infinityParser2ProInt8ModelId: Profile(
            modelId: infinityParser2ProInt8ModelId,
            upstreamRepoId: infinityParser2ProInt8UpstreamRepoId,
            upstreamRevision: infinityParser2ProInt8UpstreamRevision
        ),
    ]

    public static var supportedModelIds: Set<String> {
        Set(profilesByModelId.keys)
    }

    public static func profile(for modelId: String) -> Profile? {
        let normalized = modelId.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return profilesByModelId[normalized]
    }

    public static let defaultContextLength = 16_384
    public static let snapshotPatterns = [
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
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
    public var chatTemplateURL: URL { rootURL.appending(path: "chat_template.jinja") }
    public var processorConfigURL: URL { rootURL.appending(path: "processor_config.json") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
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
        if !fileManager.fileExists(atPath: chatTemplateURL.path)
            && !fileManager.fileExists(atPath: tokenizerConfigURL.path) {
            missing.append(chatTemplateURL)
        }

        return missing
    }

    public static func normalizedRootURL(_ rootURL: URL, fileManager: FileManager = .default) -> URL {
        let standardized = rootURL.standardizedFileURL
        let directConfig = standardized.appendingPathComponent("config.json")
        if fileManager.fileExists(atPath: directConfig.path) {
            return standardized
        }

        if let children = try? fileManager.contentsOfDirectory(
            at: standardized,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for child in children {
                let config = child.appendingPathComponent("config.json")
                if fileManager.fileExists(atPath: config.path) {
                    return child.standardizedFileURL
                }
            }
        }

        return standardized
    }
}

public enum Q35Error: LocalizedError {
    case modelNotLoaded
    case unsupportedModelId(String)
    case missingFiles([String])
    case downloadFailed(String)
    case extractionFailed
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Qwen-family model not loaded"
        case .unsupportedModelId(let id):
            return "Unsupported Qwen-family model id: \(id)"
        case .missingFiles(let files):
            return "Missing required files: \(files.joined(separator: ", "))"
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .extractionFailed:
            return "Failed to prepare Qwen-family model files"
        case .generationFailed(let message):
            return message
        }
    }
}
