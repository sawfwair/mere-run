import Foundation

public enum OpenAIPrivacyFilterCatalog {
    public static let modelId = "text-anonymize-privacy-filter"
    public static let defaultRepoId = "openai/privacy-filter"
    public static let defaultRevision = "main"

    public static let hubFallbackConfig = HubFallbackConfig(
        repoId: defaultRepoId,
        revision: defaultRevision,
        patterns: [
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "viterbi_calibration.json",
        ]
    )
}

public struct OpenAIPrivacyFilterResources: Sendable, Hashable {
    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var weightsIndexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }
    public var weightsURL: URL { rootURL.appending(path: "model.safetensors") }
    public var tokenizerConfigURL: URL { rootURL.appending(path: "tokenizer_config.json") }
    public var tokenizerDataURL: URL { rootURL.appending(path: "tokenizer.json") }
    public var viterbiCalibrationURL: URL { rootURL.appending(path: "viterbi_calibration.json") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var required: [URL] = [
            configURL,
            tokenizerConfigURL,
            tokenizerDataURL,
        ]

        let weightsOK =
            fileManager.fileExists(atPath: weightsIndexURL.path)
            || fileManager.fileExists(atPath: weightsURL.path)
        if !weightsOK {
            required.append(weightsIndexURL)
        }

        return required.filter { !fileManager.fileExists(atPath: $0.path) }
    }
}
