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
    public static let bonsai27B1BitModelId = "text-chat-bonsai-27b-1bit"
    public static let bonsai27B2BitModelId = "text-chat-bonsai-27b-2bit"
    public static let ornith9BModelId = "text-agent-ornith-9b"
    public static let ornith35BMLXModelId = "text-agent-ornith-35b-mlx"
    public static let infinityParser2FlashModelId = "vision-ocr-infinity-flash"
    public static let infinityParser2ProModelId = "vision-ocr-infinity-pro"
    public static let infinityParser2ProInt8ModelId = "vision-ocr-infinity-pro-int8"
    public static let defaultModelId = q36NanoModelId

    /// R1-style agent tunes degenerate without reasoning enabled: no-think
    /// output is signature echo or repetition loops at any temperature
    /// (HumanEval no-think scored 1/164 on Ornith 35B vs a ~90% thinking-mode
    /// model card), so these lanes default to thinking-enabled generation.
    public static func thinkingDefault(forModelId modelId: String) -> Bool {
        isBonsai27BModelId(modelId)
            || modelId == ornith9BModelId
            || modelId == ornith35BMLXModelId
    }

    public static func isBonsai27BModelId(_ modelId: String) -> Bool {
        modelId == bonsai27B1BitModelId || modelId == bonsai27B2BitModelId
    }

    public struct RecommendedSampling: Sendable, Hashable {
        public let temperature: Double
        public let topP: Double
        public let topK: Int

        public init(temperature: Double, topP: Double, topK: Int) {
            self.temperature = temperature
            self.topP = topP
            self.topK = topK
        }
    }

    /// Published `generation_config.json` sampling for lanes whose models ship
    /// one; callers use it when the user did not set explicit sampling.
    public static func recommendedSampling(forModelId modelId: String) -> RecommendedSampling? {
        switch modelId {
        case bonsai27B1BitModelId, bonsai27B2BitModelId:
            return RecommendedSampling(temperature: 0.7, topP: 0.95, topK: 20)
        case ornith9BModelId, ornith35BMLXModelId:
            return RecommendedSampling(temperature: 1.0, topP: 0.95, topK: 20)
        default:
            return nil
        }
    }

    public static let q36NanoUpstreamRepoId = "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    public static let q36NanoUpstreamRevision = "63d520640ca7461f31ba66104612135770090340"
    public static let bonsai27B1BitUpstreamRepoId = "prism-ml/Bonsai-27B-mlx-1bit"
    public static let bonsai27B1BitUpstreamRevision = "ef22f239c670078e1507f9769bcaa66657332b96"
    public static let bonsai27B1BitEstimatedDownloadBytes: Int64 = 5_128_837_600
    public static let bonsai27B1BitContextLength = 262_144
    public static let bonsai27B2BitUpstreamRepoId = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
    public static let bonsai27B2BitUpstreamRevision = "70f75f3ad081ab840a42f3304c02c27e7f89bfb7"
    public static let bonsai27B2BitEstimatedDownloadBytes: Int64 = 8_521_085_419
    public static let bonsai27B2BitContextLength = 262_144
    public static let ornith9BUpstreamRepoId = "sahilchachra/ornith-1.0-9b-optiq-5bpw-mlx"
    public static let ornith9BUpstreamRevision = "4f9f4fc2c10ec17cbeb9dae086a7f1272c904e86"
    public static let ornith9BEstimatedDownloadBytes: Int64 = 7 * 1_073_741_824
    public static let ornith35BMLXUpstreamRepoId = "deepreinforce-ai/Ornith-1.0-35B"
    public static let ornith35BMLXUpstreamRevision = "5df2ed3f675c7beaa490328cc70bb573b65fb660"
    public static let ornith35BMLXEstimatedDownloadBytes: Int64 = 18 * 1_073_741_824
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
        bonsai27B1BitModelId: Profile(
            modelId: bonsai27B1BitModelId,
            upstreamRepoId: bonsai27B1BitUpstreamRepoId,
            upstreamRevision: bonsai27B1BitUpstreamRevision
        ),
        bonsai27B2BitModelId: Profile(
            modelId: bonsai27B2BitModelId,
            upstreamRepoId: bonsai27B2BitUpstreamRepoId,
            upstreamRevision: bonsai27B2BitUpstreamRevision
        ),
        ornith9BModelId: Profile(
            modelId: ornith9BModelId,
            upstreamRepoId: ornith9BUpstreamRepoId,
            upstreamRevision: ornith9BUpstreamRevision
        ),
        ornith35BMLXModelId: Profile(
            modelId: ornith35BMLXModelId,
            upstreamRepoId: ornith35BMLXUpstreamRepoId,
            upstreamRevision: ornith35BMLXUpstreamRevision
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

    public static func defaultContextLength(forModelId modelId: String) -> Int {
        switch modelId {
        case bonsai27B1BitModelId:
            bonsai27B1BitContextLength
        case bonsai27B2BitModelId:
            bonsai27B2BitContextLength
        default:
            defaultContextLength
        }
    }

    public static let snapshotPatterns = [
        "LICENSE*",
        "NOTICE*",
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "optiq_metadata.json",
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
