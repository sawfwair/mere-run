import Foundation

public struct ACEStepResources: Sendable, Hashable {
    public var modelRootURL: URL

    public init(rootURL: URL) {
        self.modelRootURL = rootURL
    }

    public var configURL: URL {
        modelRootURL.appending(path: "config.json")
    }

    public var weightsIndexURL: URL {
        modelRootURL.appending(path: "model.safetensors.index.json")
    }

    public var weightsURL: URL {
        modelRootURL.appending(path: "model.safetensors")
    }

    public var silenceLatentURL: URL {
        modelRootURL.appending(path: "silence_latent.pt")
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var urls: [URL] = [configURL]

        let weightsOK =
            fileManager.fileExists(atPath: weightsIndexURL.path)
            || fileManager.fileExists(atPath: weightsURL.path)
        if !weightsOK {
            urls.append(weightsIndexURL)
        }

        return urls.filter { !fileManager.fileExists(atPath: $0.path) }
    }
}
