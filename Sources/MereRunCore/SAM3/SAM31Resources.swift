import Foundation

public struct SAM31Resources: Sendable, Hashable {
    public let modelRootURL: URL
    public let configURL: URL
    public let tokenizerRootURL: URL
    public let weightsURL: URL
    public let weightsIndexURL: URL

    public init(modelRootURL: URL) {
        self.modelRootURL = modelRootURL.standardizedFileURL
        self.configURL = modelRootURL.appendingPathComponent("config.json")
        let tokenizerCandidate = modelRootURL.appendingPathComponent("tokenizer", isDirectory: true)
        self.tokenizerRootURL = FileManager.default.fileExists(atPath: tokenizerCandidate.path)
            ? tokenizerCandidate
            : modelRootURL
        self.weightsURL = modelRootURL.appendingPathComponent("model.safetensors")
        self.weightsIndexURL = modelRootURL.appendingPathComponent("model.safetensors.index.json")
    }

    public func missingRequiredPaths(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        if !fileManager.fileExists(atPath: configURL.path) {
            missing.append(configURL)
        }
        let hasSingleWeights = fileManager.fileExists(atPath: weightsURL.path)
        let hasIndexedWeights = fileManager.fileExists(atPath: weightsIndexURL.path)
        if !hasSingleWeights && !hasIndexedWeights {
            missing.append(weightsURL)
        }

        let tokenizerJSON = tokenizerRootURL.appendingPathComponent("tokenizer.json")
        let tokenizerConfig = tokenizerRootURL.appendingPathComponent("tokenizer_config.json")
        if !fileManager.fileExists(atPath: tokenizerJSON.path) {
            missing.append(tokenizerJSON)
        }
        if !fileManager.fileExists(atPath: tokenizerConfig.path) {
            missing.append(tokenizerConfig)
        }

        return missing
    }

    public static func validateRoot(
        _ rootURL: URL,
        fileManager: FileManager = .default
    ) -> [String] {
        let resources = SAM31Resources(modelRootURL: rootURL)
        let missing = resources.missingRequiredPaths(fileManager: fileManager)
        return missing.map { "Missing required SAM 3.1 file: \($0.lastPathComponent)" }
    }
}
