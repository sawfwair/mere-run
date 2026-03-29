import Foundation

public struct Q35Resources: Sendable, Hashable {
    public struct Profile: Sendable, Hashable {
        public let modelId: String
        public let archiveKey: String
        public let archiveSize: Int64
        public let upstreamRepoId: String
        public let upstreamRevision: String

        public init(
            modelId: String,
            archiveKey: String,
            archiveSize: Int64,
            upstreamRepoId: String,
            upstreamRevision: String
        ) {
            self.modelId = modelId
            self.archiveKey = archiveKey
            self.archiveSize = archiveSize
            self.upstreamRepoId = upstreamRepoId
            self.upstreamRevision = upstreamRevision
        }
    }

    public static let defaultModelId = "text-chat-q35"
    public static let nanoModelId = "text-chat-q35-nano"

    public static let r2ArchiveKey = "models/text-chat-q35.tar.gz"
    public static let r2ArchiveSize: Int64 = 55_965_575_798

    public static let upstreamRepoId = "mlx-community/Qwen3.5-122B-A10B-4bit"
    public static let upstreamRevision = "e9c67b08899964be5fdd069bb1b4bc8907fe68f5"

    public static let nanoArchiveKey = "models/text-chat-q35-nano.tar.gz"
    public static let nanoArchiveSize: Int64 = 0

    public static let nanoUpstreamRepoId = "mlx-community/Qwen3.5-35B-A3B-4bit"
    public static let nanoUpstreamRevision = "1e20fd8d42056f870933bf98ca6211024744f7ec"

    private static let profilesByModelId: [String: Profile] = [
        defaultModelId: Profile(
            modelId: defaultModelId,
            archiveKey: r2ArchiveKey,
            archiveSize: r2ArchiveSize,
            upstreamRepoId: upstreamRepoId,
            upstreamRevision: upstreamRevision
        ),
        nanoModelId: Profile(
            modelId: nanoModelId,
            archiveKey: nanoArchiveKey,
            archiveSize: nanoArchiveSize,
            upstreamRepoId: nanoUpstreamRepoId,
            upstreamRevision: nanoUpstreamRevision
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

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Q35 model not loaded"
        case .unsupportedModelId(let id):
            return "Unsupported Q35 model id: \(id)"
        case .missingFiles(let files):
            return "Missing required files: \(files.joined(separator: ", "))"
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .extractionFailed:
            return "Failed to extract Q35 archive"
        }
    }
}
