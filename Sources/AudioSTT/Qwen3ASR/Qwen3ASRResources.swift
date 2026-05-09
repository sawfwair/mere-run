import Foundation
import MereRunCore

/// Resource paths and validation for Qwen3-ASR models
public struct Qwen3ASRResources: Sendable, Hashable {
    public static let defaultModelId = "speech-asr-qwen3"
    public static let defaultRepoId = "mlx-community/Qwen3-ASR-1.7B-8bit"
    public static let defaultRevision = "main"
    public static let sampleRate = 16000
    public static let hubFallbackConfig = HubFallbackConfig(
        repoId: defaultRepoId,
        revision: defaultRevision,
        patterns: [
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "*.safetensors",
        ]
    )

    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    // Model weights
    public var modelIndexURL: URL {
        rootURL.appending(path: "model.safetensors.index.json")
    }

    public var modelWeightsURL: URL {
        rootURL.appending(path: "model.safetensors")
    }

    // Tokenizer files
    public var tokenizerJSONURL: URL {
        rootURL.appending(path: "tokenizer.json")
    }

    public var vocabJSONURL: URL {
        rootURL.appending(path: "vocab.json")
    }

    public var mergesTxtURL: URL {
        rootURL.appending(path: "merges.txt")
    }

    public var tokenizerConfigURL: URL {
        rootURL.appending(path: "tokenizer_config.json")
    }

    public var addedTokensURL: URL {
        rootURL.appending(path: "added_tokens.json")
    }

    // Model config
    public var configURL: URL {
        rootURL.appending(path: "config.json")
    }

    public var generationConfigURL: URL {
        rootURL.appending(path: "generation_config.json")
    }

    // Preprocessor config (for mel spectrogram parameters)
    public var preprocessorConfigURL: URL {
        rootURL.appending(path: "preprocessor_config.json")
    }

    /// Validate that all required files exist
    /// Returns list of missing files
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

        let hasTokenizerJSON = fileManager.fileExists(atPath: tokenizerJSONURL.path)
        let hasVocab = fileManager.fileExists(atPath: vocabJSONURL.path)
        let hasMerges = fileManager.fileExists(atPath: mergesTxtURL.path)
        if !hasTokenizerJSON && !(hasVocab && hasMerges) {
            missing.append(tokenizerJSONURL)
        }

        if !fileManager.fileExists(atPath: tokenizerConfigURL.path) {
            missing.append(tokenizerConfigURL)
        }

        return missing
    }

    public static func resolveNestedIfNeeded(
        base: URL,
        fileManager: FileManager = .default
    ) -> URL {
        let direct = Qwen3ASRResources(rootURL: base)
        if direct.validate(fileManager: fileManager).isEmpty {
            return base
        }

        let nested = base.appendingPathComponent(defaultModelId, isDirectory: true)
        if Qwen3ASRResources(rootURL: nested).validate(fileManager: fileManager).isEmpty {
            return nested
        }

        return base
    }
}
