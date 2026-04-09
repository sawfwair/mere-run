import Foundation
import MereRunCore

public struct ParakeetResources: Sendable, Hashable {
    public static let defaultModelId = "speech-asr-parakeet"
    public static let defaultRepoId = "mlx-community/parakeet-tdt-0.6b-v3"
    public static let sampleRate = 16_000
    public static let r2ArchiveKey = "models/asr-parakeet.tar.gz"
    public static let r2ArchiveSize: Int64 = 2_332_340_210

    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var modelIndexURL: URL {
        rootURL.appending(path: "model.safetensors.index.json")
    }

    public var modelWeightsURL: URL {
        rootURL.appending(path: "model.safetensors")
    }

    public var configURL: URL {
        rootURL.appending(path: "config.json")
    }

    public var tokenizerModelURL: URL {
        rootURL.appending(path: "tokenizer.model")
    }

    public var tokenizerVocabURL: URL {
        rootURL.appending(path: "tokenizer.vocab")
    }

    public var vocabTxtURL: URL {
        rootURL.appending(path: "vocab.txt")
    }

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

        let hasTokenizer = fileManager.fileExists(atPath: tokenizerModelURL.path)
            || fileManager.fileExists(atPath: tokenizerVocabURL.path)
            || fileManager.fileExists(atPath: vocabTxtURL.path)
        if !hasTokenizer {
            missing.append(tokenizerModelURL)
        }

        return missing
    }

    public static func resolveLocalModelRoot(
        modelId: String,
        fileManager: FileManager = .default
    ) -> URL {
        MereRunModelPaths.resolveModelDir(modelId) { root in
            let resources = ParakeetResources(rootURL: root)
            if resources.validate(fileManager: fileManager).isEmpty {
                return true
            }

            let nested = root.appendingPathComponent(defaultModelId, isDirectory: true)
            return ParakeetResources(rootURL: nested).validate(fileManager: fileManager).isEmpty
        }
    }

    public static func resolveNestedIfNeeded(
        base: URL,
        fileManager: FileManager = .default
    ) -> URL {
        let direct = ParakeetResources(rootURL: base)
        if direct.validate(fileManager: fileManager).isEmpty {
            return base
        }

        let nested = base.appendingPathComponent(defaultModelId, isDirectory: true)
        if ParakeetResources(rootURL: nested).validate(fileManager: fileManager).isEmpty {
            return nested
        }

        return base
    }
}
