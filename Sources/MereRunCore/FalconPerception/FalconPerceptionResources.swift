import Foundation

public struct FalconPerceptionResources: Sendable, Hashable {
    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL.standardizedFileURL
    }

    public var configURL: URL { rootURL.appendingPathComponent("config.json") }
    public var weightsURL: URL { rootURL.appendingPathComponent("model.safetensors") }
    public var weightsIndexURL: URL { rootURL.appendingPathComponent("model.safetensors.index.json") }

    public var tokenizerRootURL: URL {
        let tokenizerDirectory = rootURL.appendingPathComponent("tokenizer", isDirectory: true)
        if FileManager.default.fileExists(atPath: tokenizerDirectory.path) {
            return tokenizerDirectory
        }
        return rootURL
    }

    public var tokenizerJSONURL: URL { tokenizerRootURL.appendingPathComponent("tokenizer.json") }
    public var tokenizerConfigURL: URL { tokenizerRootURL.appendingPathComponent("tokenizer_config.json") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        if !fileManager.fileExists(atPath: configURL.path) {
            missing.append(configURL)
        }

        let hasWeights = fileManager.fileExists(atPath: weightsURL.path)
            || fileManager.fileExists(atPath: weightsIndexURL.path)
        if !hasWeights {
            missing.append(weightsIndexURL)
        }

        if !fileManager.fileExists(atPath: tokenizerJSONURL.path) {
            missing.append(tokenizerJSONURL)
        }
        if !fileManager.fileExists(atPath: tokenizerConfigURL.path) {
            missing.append(tokenizerConfigURL)
        }

        return missing
    }

    public static func validateRoot(
        _ rootURL: URL,
        fileManager: FileManager = .default
    ) -> [String] {
        FalconPerceptionResources(rootURL: rootURL)
            .validate(fileManager: fileManager)
            .map { "Missing required Falcon Perception file: \($0.lastPathComponent)" }
    }
}
