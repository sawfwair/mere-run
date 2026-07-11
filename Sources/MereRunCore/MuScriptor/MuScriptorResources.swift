import Foundation

public struct MuScriptorResources: Sendable {
    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var weightsURL: URL { rootURL.appending(path: "model.safetensors") }

    public func configuration(fallback: MuScriptorConfiguration) throws -> MuScriptorConfiguration {
        let config: MuScriptorConfiguration
        if FileManager.default.fileExists(atPath: configURL.path) {
            config = try JSONDecoder().decode(
                MuScriptorConfiguration.self,
                from: Data(contentsOf: configURL)
            )
        } else {
            config = fallback
        }
        try config.validate()
        return config
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        [configURL, weightsURL].filter { !fileManager.fileExists(atPath: $0.path) }
    }
}
