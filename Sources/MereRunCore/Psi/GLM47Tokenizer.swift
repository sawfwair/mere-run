import Foundation
import MLX
@preconcurrency import Tokenizers
@preconcurrency import Hub

public enum GLM47TokenizerError: Error {
    case directoryNotFound(URL)
    case fileNotFound(URL)
    case padTokenMissing
    case padTokenNotInVocabulary(String)
}

public final class GLM47Tokenizer {
    private let encodeFunction: @Sendable (String) -> [Int]
    private let tokenizer: Tokenizer

    public let padTokenId: Int
    public let eosTokenId: Int?
    public let maxLength: Int

    public init(
        padTokenId: Int,
        eosTokenId: Int?,
        maxLength: Int,
        tokenizer: Tokenizer,
        encode: @escaping @Sendable (String) -> [Int]
    ) {
        self.padTokenId = padTokenId
        self.eosTokenId = eosTokenId
        self.maxLength = maxLength
        self.tokenizer = tokenizer
        self.encodeFunction = encode
    }

    public static func load(
        from directory: URL,
        maxLengthOverride: Int? = nil,
        hubApi: HubApi = .shared
    ) throws -> GLM47Tokenizer {
        let tokenizerConfigURL = directory.appendingPathComponent("tokenizer_config.json")
        let tokenizerDataURL = directory.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw GLM47TokenizerError.directoryNotFound(directory)
        }
        guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
            throw GLM47TokenizerError.fileNotFound(tokenizerConfigURL)
        }

        let tokenizerConfigData = try Data(contentsOf: tokenizerConfigURL)
        let tokenizerConfig = try Self.loadConfig(
            data: tokenizerConfigData,
            url: tokenizerConfigURL,
            overrideTokenizerClass: "PreTrainedTokenizer"
        )

        let tokenizer: Tokenizer
        if FileManager.default.fileExists(atPath: tokenizerDataURL.path) {
            let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)
            tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        } else {
            throw GLM47TokenizerError.fileNotFound(tokenizerDataURL)
        }

        let padTokenNode = tokenizerConfig["pad_token"]
        let padTokenString = padTokenNode.string() ?? padTokenNode["content"].string()
        guard let padToken = padTokenString else {
            throw GLM47TokenizerError.padTokenMissing
        }

        guard let padId = tokenizer.convertTokenToId(padToken) ??
            tokenizer.eosTokenId ??
            tokenizer.bosTokenId
        else {
            throw GLM47TokenizerError.padTokenNotInVocabulary(padToken)
        }

        let eosId = tokenizer.eosTokenId
        let resolvedMaxLength = maxLengthOverride ?? tokenizerConfig["model_max_length"].integer(or: 128_000)

        return GLM47Tokenizer(
            padTokenId: padId,
            eosTokenId: eosId,
            maxLength: resolvedMaxLength,
            tokenizer: tokenizer
        ) { text in
            tokenizer.encode(text: text)
        }
    }

    private static func loadConfig(
        data: Data,
        url: URL,
        overrideTokenizerClass: String
    ) throws -> Config {
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard var dict = parsed as? [String: Any] else {
            throw GLM47TokenizerError.fileNotFound(url)
        }

        if let tokenizerClass = dict["tokenizer_class"] as? String,
           tokenizerClass == "TokenizersBackend" {
            dict["tokenizer_class"] = overrideTokenizerClass
        }

        let nsDict: [NSString: Any] = dict.reduce(into: [:]) { partialResult, pair in
            partialResult[pair.key as NSString] = pair.value
        }
        return Config(nsDict)
    }

    public func encodeText(_ text: String) -> [Int] {
        encodeFunction(text)
    }

    public func encodeChat(
        messages: [ChatMessage],
        tools: [ToolDefinition]? = nil,
        addGenerationPrompt: Bool = true
    ) -> [Int] {
        let prompt = GLM47ChatTemplate.render(
            messages: messages,
            tools: tools,
            addGenerationPrompt: addGenerationPrompt
        )
        return encodeFunction(prompt)
    }

    public func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }
}
