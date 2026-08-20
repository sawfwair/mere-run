import Foundation

public enum Qwen3VLEmbeddingCatalog {
    public static let modelID = "vision-embed-qwen3-vl-2b"
    public static let defaultRepoID = "Qwen/Qwen3-VL-Embedding-2B"
    public static let defaultRevision = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
    public static let nativeDimensions = 2_048
    public static let defaultMaxTokens = 8_192
    public static let defaultMinPixels = 4_096
    public static let defaultMaxPixels = 1_843_200

    public static let hubFallbackConfig = HubFallbackConfig(
        repoId: defaultRepoID,
        revision: defaultRevision,
        patterns: [
            "1_Pooling/config.json",
            "added_tokens.json",
            "chat_template.jinja",
            "config.json",
            "config_sentence_transformers.json",
            "merges.txt",
            "model.safetensors",
            "model.safetensors.index.json",
            "modules.json",
            "preprocessor_config.json",
            "sentence_bert_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ]
    )
}

public struct Qwen3VLEmbeddingResources: Sendable, Hashable {
    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL.standardizedFileURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var weightsURL: URL { rootURL.appending(path: "model.safetensors") }
    public var weightsIndexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }
    public var tokenizerConfigURL: URL { rootURL.appending(path: "tokenizer_config.json") }
    public var tokenizerDataURL: URL { rootURL.appending(path: "tokenizer.json") }
    public var vocabURL: URL { rootURL.appending(path: "vocab.json") }
    public var mergesURL: URL { rootURL.appending(path: "merges.txt") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        for required in [configURL, tokenizerConfigURL] where !fileManager.fileExists(atPath: required.path) {
            missing.append(required)
        }

        if !fileManager.fileExists(atPath: weightsURL.path)
            && !fileManager.fileExists(atPath: weightsIndexURL.path) {
            missing.append(weightsURL)
        }

        if !fileManager.fileExists(atPath: tokenizerDataURL.path)
            && !(fileManager.fileExists(atPath: vocabURL.path)
                && fileManager.fileExists(atPath: mergesURL.path)) {
            missing.append(tokenizerDataURL)
        }
        return missing
    }
}

struct Qwen3VLEmbeddingRootConfig: Decodable {
    let textConfig: QwenVLTextEncoderConfig
    let visionConfig: QwenVLTextEncoderConfig.VisionConfig
    let quantization: QwenVLTextEncoderConfig.Quantization?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case quantization
    }
}
