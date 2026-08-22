import Foundation

public struct SenseNovaU15Resources: Sendable, Hashable {
    public static let modelID = "image-sensenova-u1-5-8b-mot"
    public static let repository = "sensenova/SenseNova-U1.5-8B-MoT"
    public static let revision = "1f6ec60423d29939dde4202fd82ae340b144e280"

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var tokenizerConfigURL: URL { rootURL.appending(path: "tokenizer_config.json") }
    public var tokenizerJSONURL: URL { rootURL.appending(path: "tokenizer.json") }
    public var addedTokensURL: URL { rootURL.appending(path: "added_tokens.json") }
    public var vocabURL: URL { rootURL.appending(path: "vocab.json") }
    public var mergesURL: URL { rootURL.appending(path: "merges.txt") }
    public var weightsIndexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }
    public var singleWeightsURL: URL { rootURL.appending(path: "model.safetensors") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        if !fileManager.fileExists(atPath: configURL.path) { missing.append(configURL) }
        if !fileManager.fileExists(atPath: tokenizerConfigURL.path) { missing.append(tokenizerConfigURL) }
        let hasTokenizer = fileManager.fileExists(atPath: tokenizerJSONURL.path)
            || (fileManager.fileExists(atPath: vocabURL.path) && fileManager.fileExists(atPath: mergesURL.path))
        if !hasTokenizer { missing.append(tokenizerJSONURL) }
        if !fileManager.fileExists(atPath: weightsIndexURL.path)
            && !fileManager.fileExists(atPath: singleWeightsURL.path) {
            missing.append(weightsIndexURL)
        }
        return missing
    }
}

public enum SenseNovaU15Error: LocalizedError, Sendable {
    case invalidConfiguration(String)
    case missingModelFiles([URL])
    case invalidImageSize(width: Int, height: Int)
    case tokenizerTokenMissing(String)
    case imageTokenMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "Invalid SenseNova U1.5 configuration: \(detail)"
        case .missingModelFiles(let files):
            return "Missing SenseNova U1.5 model files: \(files.map(\.path).joined(separator: ", "))"
        case .invalidImageSize(let width, let height):
            return "SenseNova U1.5 output size must be positive and divisible by 32; got \(width)x\(height)."
        case .tokenizerTokenMissing(let token):
            return "SenseNova U1.5 tokenizer is missing required token: \(token)"
        case .imageTokenMismatch(let expected, let actual):
            return "SenseNova U1.5 image token mismatch: expected \(expected), found \(actual)."
        }
    }
}
