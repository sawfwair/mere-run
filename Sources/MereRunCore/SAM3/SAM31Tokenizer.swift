import Foundation
@preconcurrency import MLX
@preconcurrency import Hub
@preconcurrency import Tokenizers

public enum SAM31TokenizerError: LocalizedError {
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

public struct SAM31TokenBatch: @unchecked Sendable {
    public let inputIDs: MLXArray
    public let attentionMask: MLXArray

    public init(inputIDs: MLXArray, attentionMask: MLXArray) {
        self.inputIDs = inputIDs
        self.attentionMask = attentionMask
    }
}

public final class SAM31Tokenizer {
    private let tokenizer: Tokenizer
    private let encodeFunction: @Sendable (String) -> [Int]

    public let padTokenID: Int
    public let maxLength: Int

    public init(
        tokenizer: Tokenizer,
        padTokenID: Int,
        maxLength: Int,
        encode: @escaping @Sendable (String) -> [Int]
    ) {
        self.tokenizer = tokenizer
        self.padTokenID = padTokenID
        self.maxLength = maxLength
        self.encodeFunction = encode
    }

    public static func load(
        from directory: URL,
        maxLengthOverride: Int? = nil,
        hubAPI: HubApi = .shared
    ) throws -> SAM31Tokenizer {
        let fm = FileManager.default
        let tokenizerDirectory = resolveTokenizerDirectory(directory)
        let tokenizerConfigURL = tokenizerDirectory.appendingPathComponent("tokenizer_config.json")
        let tokenizerDataURL = tokenizerDirectory.appendingPathComponent("tokenizer.json")

        guard fm.fileExists(atPath: tokenizerDirectory.path) else {
            throw SAM31TokenizerError.directoryNotFound(tokenizerDirectory)
        }
        guard fm.fileExists(atPath: tokenizerConfigURL.path) else {
            throw SAM31TokenizerError.fileNotFound(tokenizerConfigURL)
        }
        guard fm.fileExists(atPath: tokenizerDataURL.path) else {
            throw SAM31TokenizerError.fileNotFound(tokenizerDataURL)
        }

        let configData = try Data(contentsOf: tokenizerConfigURL)
        let tokenizerConfig = try normalizedTokenizerConfig(
            data: configData,
            url: tokenizerConfigURL,
            overrideTokenizerClass: "GPT2Tokenizer"
        )
        let tokenizerData = try hubAPI.configuration(fileURL: tokenizerDataURL)
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)

        let padTokenNode = tokenizerConfig["pad_token"]
        let padTokenString = padTokenNode.string() ?? padTokenNode["content"].string()
        guard let padToken = padTokenString else {
            throw SAM31TokenizerError.padTokenMissing
        }
        guard let padTokenID = tokenizer.convertTokenToId(padToken)
            ?? tokenizer.eosTokenId
            ?? tokenizer.bosTokenId
        else {
            throw SAM31TokenizerError.padTokenNotInVocabulary(padToken)
        }

        let maxLength = maxLengthOverride ?? tokenizerConfig["model_max_length"].integer(or: 32)

        return SAM31Tokenizer(
            tokenizer: tokenizer,
            padTokenID: padTokenID,
            maxLength: maxLength
        ) { text in
            tokenizer.encode(text: text, addSpecialTokens: true)
        }
    }

    public func encode(prompts: [String], maxLength: Int? = nil) -> SAM31TokenBatch {
        precondition(!prompts.isEmpty, "At least one prompt must be provided.")

        let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
        var inputSequences: [[Int32]] = []
        var attentionSequences: [[Int32]] = []
        inputSequences.reserveCapacity(prompts.count)
        attentionSequences.reserveCapacity(prompts.count)

        for prompt in prompts {
            var tokens = encodeFunction(prompt)
            if tokens.count > targetLength {
                tokens = Array(tokens.prefix(targetLength))
            }

            var ids = Array(repeating: Int32(padTokenID), count: targetLength)
            var mask = Array(repeating: Int32(0), count: targetLength)
            for (index, token) in tokens.enumerated() {
                ids[index] = Int32(token)
                mask[index] = 1
            }
            inputSequences.append(ids)
            attentionSequences.append(mask)
        }

        let flatIDs = inputSequences.flatMap { $0 }
        let flatMask = attentionSequences.flatMap { $0 }
        let shape = [prompts.count, targetLength]
        return SAM31TokenBatch(
            inputIDs: MLXArray(flatIDs, shape),
            attentionMask: MLXArray(flatMask, shape)
        )
    }

    public func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
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
            throw SAM31TokenizerError.fileNotFound(url)
        }

        if let tokenizerClass = dictionary["tokenizer_class"] as? String {
            if tokenizerClass == "TokenizersBackend" {
                dictionary["tokenizer_class"] = overrideTokenizerClass
            }
        } else {
            dictionary["tokenizer_class"] = overrideTokenizerClass
        }

        let nsDictionary: [NSString: Any] = dictionary.reduce(into: [:]) { partialResult, pair in
            partialResult[pair.key as NSString] = pair.value
        }
        return Config(nsDictionary)
    }
}
