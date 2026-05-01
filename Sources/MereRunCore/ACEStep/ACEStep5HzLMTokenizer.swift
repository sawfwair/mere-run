import Foundation
import MLX
@preconcurrency import Tokenizers
@preconcurrency import Hub

public enum ACEStep5HzLMTokenizerError: Error {
    case directoryNotFound(URL)
    case fileNotFound(URL)
    case padTokenMissing
    case padTokenNotInVocabulary(String)
}

public final class ACEStep5HzLMTokenizer {
    private let tokenizer: Tokenizer

    public let padTokenId: Int
    public let bosTokenId: Int?
    public let eosTokenId: Int?

    public let audioCodeTokenIds: Set<Int>
    public let audioCodeTokenIdToValue: [Int: Int]

    public init(
        tokenizer: Tokenizer,
        padTokenId: Int,
        audioCodeTokenIds: Set<Int>,
        audioCodeTokenIdToValue: [Int: Int]
    ) {
        self.tokenizer = tokenizer
        self.padTokenId = padTokenId
        self.bosTokenId = tokenizer.bosTokenId
        self.eosTokenId = tokenizer.eosTokenId
        self.audioCodeTokenIds = audioCodeTokenIds
        self.audioCodeTokenIdToValue = audioCodeTokenIdToValue
    }

    public static func load(
        from directory: URL,
        hubApi: HubApi = .shared
    ) throws -> ACEStep5HzLMTokenizer {
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw ACEStep5HzLMTokenizerError.directoryNotFound(directory)
        }

        let tokenizerConfigURL = directory.appending(path: "tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
            throw ACEStep5HzLMTokenizerError.fileNotFound(tokenizerConfigURL)
        }

        let tokenizerConfig = try hubApi.configuration(fileURL: tokenizerConfigURL)
        let tokenizerDataURL = directory.appending(path: "tokenizer.json")

        let tokenizer: Tokenizer
        if FileManager.default.fileExists(atPath: tokenizerDataURL.path) {
            let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)
            tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        } else {
            let vocabURL = directory.appending(path: "vocab.json")
            let mergesURL = directory.appending(path: "merges.txt")
            guard FileManager.default.fileExists(atPath: vocabURL.path),
                  FileManager.default.fileExists(atPath: mergesURL.path)
            else {
                throw ACEStep5HzLMTokenizerError.fileNotFound(tokenizerDataURL)
            }

            let tokenizerData = try makeBPETokenizerData(vocabURL: vocabURL, mergesURL: mergesURL)
            tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        }

        let padTokenNode = tokenizerConfig["pad_token"]
        let padTokenString = padTokenNode.string() ?? padTokenNode["content"].string()
        guard let padToken = padTokenString else {
            throw ACEStep5HzLMTokenizerError.padTokenMissing
        }

        guard let padId = tokenizer.convertTokenToId(padToken) ?? tokenizer.eosTokenId ?? tokenizer.bosTokenId else {
            throw ACEStep5HzLMTokenizerError.padTokenNotInVocabulary(padToken)
        }

        let addedTokensURL = directory.appending(path: "added_tokens.json")
        let (audioCodeIds, audioCodeMap) = try loadAudioCodeTokens(from: addedTokensURL)

        return ACEStep5HzLMTokenizer(
            tokenizer: tokenizer,
            padTokenId: padId,
            audioCodeTokenIds: audioCodeIds,
            audioCodeTokenIdToValue: audioCodeMap
        )
    }

    public func encode(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    public func convertTokenToId(_ token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }

    private static func loadAudioCodeTokens(from url: URL) throws -> (Set<Int>, [Int: Int]) {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ACEStep5HzLMTokenizerError.fileNotFound(url)
        }

        let data = try Data(contentsOf: url)
        guard let obj = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
            return ([], [:])
        }

        var ids: Set<Int> = []
        var map: [Int: Int] = [:]
        ids.reserveCapacity(64_000)
        map.reserveCapacity(64_000)

        let prefix = "<|audio_code_"
        let suffix = "|>"

        for (token, rawId) in obj {
            guard token.hasPrefix(prefix), token.hasSuffix(suffix) else { continue }
            guard let id = rawId as? Int else { continue }

            let start = token.index(token.startIndex, offsetBy: prefix.count)
            let end = token.index(token.endIndex, offsetBy: -suffix.count)
            let numberString = String(token[start..<end])
            guard let value = Int(numberString), (0...63_999).contains(value) else { continue }

            ids.insert(id)
            map[id] = value
        }

        return (ids, map)
    }

    private static func makeBPETokenizerData(vocabURL: URL, mergesURL: URL) throws -> Config {
        let vocabData = try Data(contentsOf: vocabURL)
        guard let vocabObject = try JSONSerialization.jsonObject(with: vocabData, options: []) as? [String: Any] else {
            throw ACEStep5HzLMTokenizerError.fileNotFound(vocabURL)
        }

        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(vocabObject.count)
        for (k, v) in vocabObject {
            if let i = v as? Int { vocab[k] = i }
        }

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        let merges: [String] = mergesText
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        let tokenizerDict: [String: Any] = [
            "model": [
                "vocab": vocab,
                "merges": merges,
            ],
            "preTokenizer": [
                "type": "ByteLevel",
                "addPrefixSpace": false,
                "trimOffsets": true,
                "useRegex": true,
            ],
            "decoder": [
                "type": "ByteLevel",
            ],
        ]

        let raw = try JSONSerialization.data(withJSONObject: tokenizerDict, options: [])
        return try JSONDecoder().decode(Config.self, from: raw)
    }
}
