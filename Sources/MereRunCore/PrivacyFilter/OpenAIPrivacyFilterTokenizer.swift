import Foundation
@preconcurrency import Hub
@preconcurrency import Tokenizers

public final class OpenAIPrivacyFilterTokenizer {
    private let tokenizer: any Tokenizer

    public let padTokenID: Int
    public let eosTokenID: Int?
    public let maxLength: Int

    public init(tokenizer: any Tokenizer, padTokenID: Int, eosTokenID: Int?, maxLength: Int) {
        self.tokenizer = tokenizer
        self.padTokenID = padTokenID
        self.eosTokenID = eosTokenID
        self.maxLength = maxLength
    }

    public static func load(
        from rootURL: URL,
        config: OpenAIPrivacyFilterConfig,
        hubApi: HubApi = .shared
    ) throws -> OpenAIPrivacyFilterTokenizer {
        let tokenizerConfigURL = rootURL.appending(path: "tokenizer_config.json")
        let tokenizerDataURL = rootURL.appending(path: "tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
            throw OpenAIPrivacyFilterError.fileNotFound(tokenizerConfigURL)
        }
        guard FileManager.default.fileExists(atPath: tokenizerDataURL.path) else {
            throw OpenAIPrivacyFilterError.fileNotFound(tokenizerDataURL)
        }

        let tokenizerConfig = try hubApi.configuration(fileURL: tokenizerConfigURL)
        let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: false)

        let padTokenNode = tokenizerConfig["pad_token"]
        let padTokenString = padTokenNode.string() ?? padTokenNode["content"].string()
        let padID = padTokenString.flatMap { tokenizer.convertTokenToId($0) }
            ?? config.padTokenID
            ?? tokenizer.eosTokenId
            ?? 199_999
        let maxLength = tokenizerConfig["model_max_length"].integer(or: config.defaultContextLength ?? config.maxPositionEmbeddings)

        return OpenAIPrivacyFilterTokenizer(
            tokenizer: tokenizer,
            padTokenID: padID,
            eosTokenID: config.eosTokenID ?? tokenizer.eosTokenId,
            maxLength: maxLength
        )
    }

    public func encode(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(tokens: [Int], skipSpecialTokens: Bool = false) -> String {
        tokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
    }

    func tokenString(for id: Int) -> String? {
        tokenizer.convertIdToToken(id)
    }
}

public enum OpenAIPrivacyFilterError: LocalizedError, Sendable {
    case fileNotFound(URL)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let url):
            return "Privacy filter file not found: \(url.path)"
        }
    }
}
