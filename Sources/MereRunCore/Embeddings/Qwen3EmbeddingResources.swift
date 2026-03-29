import Foundation

public enum Qwen3EmbeddingCatalog {
    public static let modelId = "text-embed-qwen3-0.6b"
    public static let modelDirectoryName = "Qwen3-Embedding-0.6B"
    public static let archiveKey = "models/text-embed-qwen3-0.6b.tar.gz"
    public static let archiveSize: Int64 = 0  // Unknown at compile-time.
    public static var archiveURL: URL? {
        MereRunModelSourceConfiguration.publicArchiveURL(for: archiveKey)
    }

    /// Returns path if downloaded, nil otherwise. Usable from both app and CLI.
    /// Also checks the Music Acestep checkpoints location as a compatibility fallback.
    public static func resolveModelPath(fileManager: FileManager = .default) -> String? {
        let directRoot = MereRunModelPaths.resolveModelDir(modelId) { root in
            let direct = Qwen3EmbeddingResources(rootURL: root)
            if direct.validate(fileManager: fileManager).isEmpty {
                return true
            }

            let nested = root.appendingPathComponent(modelDirectoryName, isDirectory: true)
            return Qwen3EmbeddingResources(rootURL: nested).validate(fileManager: fileManager).isEmpty
        }

        let directResources = Qwen3EmbeddingResources(rootURL: directRoot)
        if directResources.validate(fileManager: fileManager).isEmpty {
            return directRoot.path
        }

        let nestedDirect = directRoot.appendingPathComponent(modelDirectoryName, isDirectory: true)
        let nestedDirectResources = Qwen3EmbeddingResources(rootURL: nestedDirect)
        if nestedDirectResources.validate(fileManager: fileManager).isEmpty {
            return nestedDirect.path
        }

        let acestepRoot = MereRunModelPaths.resolveModelDir("music-acestep") { root in
            let nested = root.appendingPathComponent(modelDirectoryName, isDirectory: true)
            return Qwen3EmbeddingResources(rootURL: nested).validate(fileManager: fileManager).isEmpty
        }
        let acestepNested = acestepRoot.appendingPathComponent(modelDirectoryName, isDirectory: true)
        let acestepNestedResources = Qwen3EmbeddingResources(rootURL: acestepNested)
        if acestepNestedResources.validate(fileManager: fileManager).isEmpty {
            return acestepNested.path
        }

        if let explicitRoot = ProcessInfo.processInfo.environment["MERERUN_MUSIC_ACESTEP_ROOT"],
           !explicitRoot.isEmpty {
            let envNested = URL(fileURLWithPath: explicitRoot, isDirectory: true)
                .appendingPathComponent(modelDirectoryName, isDirectory: true)
            let envResources = Qwen3EmbeddingResources(rootURL: envNested)
            if envResources.validate(fileManager: fileManager).isEmpty {
                return envNested.path
            }
        }

        return nil
    }
}

public struct Qwen3EmbeddingResources: Sendable, Hashable {
    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var weightsIndexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }
    public var weightsURL: URL { rootURL.appending(path: "model.safetensors") }
    public var tokenizerConfigURL: URL { rootURL.appending(path: "tokenizer_config.json") }
    public var tokenizerDataURL: URL { rootURL.appending(path: "tokenizer.json") }
    public var vocabURL: URL { rootURL.appending(path: "vocab.json") }
    public var mergesURL: URL { rootURL.appending(path: "merges.txt") }
    public var addedTokensURL: URL { rootURL.appending(path: "added_tokens.json") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var required: [URL] = [
            configURL,
            tokenizerConfigURL,
            addedTokensURL,
        ]

        let weightsOK =
            fileManager.fileExists(atPath: weightsIndexURL.path)
            || fileManager.fileExists(atPath: weightsURL.path)
        if !weightsOK {
            required.append(weightsIndexURL)
        }

        let tokenizerOK =
            fileManager.fileExists(atPath: tokenizerDataURL.path)
            || (fileManager.fileExists(atPath: vocabURL.path) && fileManager.fileExists(atPath: mergesURL.path))
        if !tokenizerOK {
            required.append(tokenizerDataURL)
        }

        return required.filter { !fileManager.fileExists(atPath: $0.path) }
    }
}
