import Foundation

public struct HiDreamO1Resources: Sendable, Hashable {
    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var generationConfigURL: URL { rootURL.appending(path: "generation_config.json") }
    public var tokenizerJSONURL: URL { rootURL.appending(path: "tokenizer.json") }
    public var tokenizerConfigURL: URL { rootURL.appending(path: "tokenizer_config.json") }
    public var vocabURL: URL { rootURL.appending(path: "vocab.json") }
    public var mergesURL: URL { rootURL.appending(path: "merges.txt") }
    public var preprocessorConfigURL: URL { rootURL.appending(path: "preprocessor_config.json") }
    public var videoPreprocessorConfigURL: URL { rootURL.appending(path: "video_preprocessor_config.json") }
    public var weightsIndexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }
    public var singleWeightsURL: URL { rootURL.appending(path: "model.safetensors") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        if !fileManager.fileExists(atPath: configURL.path) { missing.append(configURL) }
        if !fileManager.fileExists(atPath: tokenizerConfigURL.path) { missing.append(tokenizerConfigURL) }
        if !fileManager.fileExists(atPath: preprocessorConfigURL.path) { missing.append(preprocessorConfigURL) }

        let hasTokenizerJSON = fileManager.fileExists(atPath: tokenizerJSONURL.path)
        let hasBPEPair = fileManager.fileExists(atPath: vocabURL.path) && fileManager.fileExists(atPath: mergesURL.path)
        if !hasTokenizerJSON && !hasBPEPair { missing.append(tokenizerJSONURL) }

        let hasIndexedWeights = fileManager.fileExists(atPath: weightsIndexURL.path)
        let hasSingleWeights = fileManager.fileExists(atPath: singleWeightsURL.path)
        if !hasIndexedWeights && !hasSingleWeights { missing.append(weightsIndexURL) }

        return missing
    }
}
