import Foundation

/// Resource paths and validation for Psi3 (GLM-4.7 Flash) chat models.
public struct Psi3ChatResources: Sendable, Hashable {
    public static let defaultModelId = "text-chat-psi-agent"

    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    // Model configuration
    public var configURL: URL {
        rootURL.appending(path: "config.json")
    }

    // Model weights index (for sharded weights)
    public var modelIndexURL: URL {
        rootURL.appending(path: "model.safetensors.index.json")
    }

    // Single-file weights (fallback)
    public var modelWeightsURL: URL {
        rootURL.appending(path: "model.safetensors")
    }

    // Tokenizer files
    public var tokenizerURL: URL {
        rootURL.appending(path: "tokenizer.json")
    }

    public var tokenizerConfigURL: URL {
        rootURL.appending(path: "tokenizer_config.json")
    }

    /// Validate that required model files exist.
    /// Returns list of missing URLs. Empty list means all required files are present.
    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []

        // Config is required
        if !fileManager.fileExists(atPath: configURL.path) {
            missing.append(configURL)
        }

        // Need either sharded index or single weights file
        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle {
            missing.append(modelIndexURL)
        }

        // Tokenizer is required
        if !fileManager.fileExists(atPath: tokenizerURL.path) {
            missing.append(tokenizerURL)
        }

        return missing
    }
}
