import Foundation

public struct ACEStep5HzLMResources: Sendable, Hashable {
    public var modelRootURL: URL

    public init(rootURL: URL) {
        self.modelRootURL = rootURL
    }

    public var configURL: URL { modelRootURL.appending(path: "config.json") }
    public var weightsIndexURL: URL { modelRootURL.appending(path: "model.safetensors.index.json") }
    public var weightsURL: URL { modelRootURL.appending(path: "model.safetensors") }

    public var tokenizerConfigURL: URL { modelRootURL.appending(path: "tokenizer_config.json") }
    public var tokenizerDataURL: URL { modelRootURL.appending(path: "tokenizer.json") }
    public var vocabURL: URL { modelRootURL.appending(path: "vocab.json") }
    public var mergesURL: URL { modelRootURL.appending(path: "merges.txt") }
    public var addedTokensURL: URL { modelRootURL.appending(path: "added_tokens.json") }
    public var chatTemplateURL: URL { modelRootURL.appending(path: "chat_template.jinja") }

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

