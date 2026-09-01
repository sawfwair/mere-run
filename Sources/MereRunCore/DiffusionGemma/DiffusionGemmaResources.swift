import Foundation

public struct DiffusionGemmaResources: Sendable, Hashable {
    public static let modelID = "text-chat-diffusiongemma-26b-optiq-4bit"
    public static let upstreamModelID = "mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit"
    public static let upstreamRevision = "30f3c7c7746bf41cfd1a290155cc3b777ab588b9"
    public static let estimatedDownloadBytes: Int64 = 17_852_058_816
    public static let defaultContextLength = 32_768
    public static let maximumCanvasLength = 256
    public static let snapshotPatterns = [
        ".gitattributes",
        "README.md",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "*.safetensors",
        "optiq/optiq_vision.safetensors",
        "optiq_metadata.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL.standardizedFileURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var generationConfigURL: URL { rootURL.appending(path: "generation_config.json") }
    public var chatTemplateURL: URL { rootURL.appending(path: "chat_template.jinja") }
    public var modelIndexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }
    public var tokenizerURL: URL { rootURL.appending(path: "tokenizer.json") }
    public var tokenizerConfigURL: URL { rootURL.appending(path: "tokenizer_config.json") }
    public var visionWeightsURL: URL { rootURL.appending(path: "optiq/optiq_vision.safetensors") }

    public var modelShardURLs: [URL] {
        (1...4).map {
            rootURL.appending(path: String(format: "model-%05d-of-00004.safetensors", $0))
        }
    }

    public func validate(fileManager: FileManager = .default, requireVision: Bool = false) -> [URL] {
        var required = [
            chatTemplateURL,
            configURL,
            generationConfigURL,
            modelIndexURL,
            tokenizerURL,
            tokenizerConfigURL,
        ] + modelShardURLs
        if requireVision {
            required.append(visionWeightsURL)
        }
        return required.filter { !fileManager.fileExists(atPath: $0.path) }
    }
}

public enum DiffusionGemmaError: LocalizedError {
    case missingFiles([String])
    case unsupportedConfiguration(String)
    case unsupportedModelLocation(String)
    case downloadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .missingFiles(let files):
            return "Missing required DiffusionGemma files: \(files.joined(separator: ", "))"
        case .unsupportedConfiguration(let message):
            return message
        case .unsupportedModelLocation(let location):
            return "Could not resolve DiffusionGemma model location: \(location)"
        case .downloadFailed(let message):
            return "DiffusionGemma download failed: \(message)"
        }
    }
}
