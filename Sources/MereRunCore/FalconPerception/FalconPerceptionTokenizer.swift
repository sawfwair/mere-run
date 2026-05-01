import Foundation
@preconcurrency import Hub
@preconcurrency import Tokenizers

public enum FalconPerceptionTokenizerError: LocalizedError {
    case directoryNotFound(URL)
    case fileNotFound(URL)
    case padTokenMissing
    case padTokenNotInVocabulary(String)

    public var errorDescription: String? {
        switch self {
        case .directoryNotFound(let url):
            return "Tokenizer directory not found: \(url.path)"
        case .fileNotFound(let url):
            return "Tokenizer file not found: \(url.path)"
        case .padTokenMissing:
            return "Tokenizer config is missing pad_token."
        case .padTokenNotInVocabulary(let token):
            return "Pad token is not in vocabulary: \(token)"
        }
    }
}

public final class FalconPerceptionTokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizer

    public let padTokenID: Int
    public let eosTokenID: Int?
    public let maxLength: Int

    public init(
        tokenizer: Tokenizer,
        padTokenID: Int,
        eosTokenID: Int?,
        maxLength: Int
    ) {
        self.tokenizer = tokenizer
        self.padTokenID = padTokenID
        self.eosTokenID = eosTokenID
        self.maxLength = maxLength
    }

    public static func load(
        from directory: URL,
        maxLengthOverride: Int? = nil,
        hubAPI: HubApi = .shared
    ) throws -> FalconPerceptionTokenizer {
        let fileManager = FileManager.default
        let tokenizerDirectory = resolveTokenizerDirectory(directory)
        let tokenizerConfigURL = tokenizerDirectory.appendingPathComponent("tokenizer_config.json")
        let tokenizerDataURL = tokenizerDirectory.appendingPathComponent("tokenizer.json")

        guard fileManager.fileExists(atPath: tokenizerDirectory.path) else {
            throw FalconPerceptionTokenizerError.directoryNotFound(tokenizerDirectory)
        }
        guard fileManager.fileExists(atPath: tokenizerConfigURL.path) else {
            throw FalconPerceptionTokenizerError.fileNotFound(tokenizerConfigURL)
        }
        guard fileManager.fileExists(atPath: tokenizerDataURL.path) else {
            throw FalconPerceptionTokenizerError.fileNotFound(tokenizerDataURL)
        }

        let configData = try Data(contentsOf: tokenizerConfigURL)
        let tokenizerConfig = try normalizedTokenizerConfig(
            data: configData,
            url: tokenizerConfigURL,
            overrideTokenizerClass: "PreTrainedTokenizerFast"
        )
        let tokenizerData = try hubAPI.configuration(fileURL: tokenizerDataURL)
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)

        let padTokenNode = tokenizerConfig["pad_token"]
        let padTokenString = padTokenNode.string() ?? padTokenNode["content"].string()
        guard let padToken = padTokenString else {
            throw FalconPerceptionTokenizerError.padTokenMissing
        }
        guard let padTokenID = tokenizer.convertTokenToId(padToken)
            ?? tokenizer.eosTokenId
            ?? tokenizer.bosTokenId
        else {
            throw FalconPerceptionTokenizerError.padTokenNotInVocabulary(padToken)
        }

        let maxLength = maxLengthOverride ?? tokenizerConfig["model_max_length"].integer(or: 8192)
        return FalconPerceptionTokenizer(
            tokenizer: tokenizer,
            padTokenID: padTokenID,
            eosTokenID: tokenizer.eosTokenId,
            maxLength: maxLength
        )
    }

    public func encode(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    public func decode(token: Int) -> String {
        tokenizer.decode(tokens: [token])
    }

    private static func resolveTokenizerDirectory(_ directory: URL) -> URL {
        let tokenizerDir = directory.appendingPathComponent("tokenizer", isDirectory: true)
        if FileManager.default.fileExists(atPath: tokenizerDir.path) {
            return tokenizerDir
        }
        return directory
    }

    private static func normalizedTokenizerConfig(
        data: Data,
        url: URL,
        overrideTokenizerClass: String
    ) throws -> Config {
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard var dictionary = parsed as? [String: Any] else {
            throw FalconPerceptionTokenizerError.fileNotFound(url)
        }

        dictionary["tokenizer_class"] = overrideTokenizerClass
        let nsDictionary: [NSString: Any] = dictionary.reduce(into: [:]) { partialResult, pair in
            partialResult[pair.key as NSString] = pair.value
        }
        return Config(nsDictionary)
    }
}
