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
        hubApi: HubApi = .shared,
        requireAudioCodeTokens: Bool = true
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
        let (audioCodeIds, audioCodeMap): (Set<Int>, [Int: Int])
        if FileManager.default.fileExists(atPath: addedTokensURL.path) {
            (audioCodeIds, audioCodeMap) = try loadAudioCodeTokens(from: addedTokensURL)
        } else if requireAudioCodeTokens {
            throw ACEStep5HzLMTokenizerError.fileNotFound(addedTokensURL)
        } else {
            audioCodeIds = []
            audioCodeMap = [:]
        }

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
        let obj = (try? JSONDecoder().decode([String: LenientInt].self, from: data)) ?? [:]

        var ids: Set<Int> = []
        var map: [Int: Int] = [:]
        ids.reserveCapacity(64_000)
        map.reserveCapacity(64_000)

        let prefix = "<|audio_code_"
        let suffix = "|>"

        for (token, rawId) in obj {
            guard token.hasPrefix(prefix), token.hasSuffix(suffix) else { continue }
            let id = rawId.value

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
        let decodedVocab = try JSONDecoder().decode([String: LenientInt].self, from: vocabData)
        let vocab = decodedVocab.mapValues(\.value)

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        let merges: [String] = mergesText
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        let raw = try JSONEncoder().encode(ACEStepBPETokenizerConfig(vocab: vocab, merges: merges))
        return try JSONDecoder().decode(Config.self, from: raw)
    }
}

private struct ACEStepBPETokenizerConfig: Encodable {
    let model: Model
    let preTokenizer = ByteLevelPreTokenizer()
    let decoder = ByteLevelDecoder()

    init(vocab: [String: Int], merges: [String]) {
        self.model = Model(vocab: vocab, merges: merges)
    }

    struct Model: Encodable {
        let vocab: [String: Int]
        let merges: [String]
    }

    struct ByteLevelPreTokenizer: Encodable {
        let type = "ByteLevel"
        let addPrefixSpace = false
        let trimOffsets = true
        let useRegex = true
    }

    struct ByteLevelDecoder: Encodable {
        let type = "ByteLevel"
    }
}
