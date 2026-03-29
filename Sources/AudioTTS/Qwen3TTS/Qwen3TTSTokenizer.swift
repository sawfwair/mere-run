import Foundation
import MLX
@preconcurrency import Tokenizers
@preconcurrency import Hub

// MARK: - Qwen3 TTS Tokenizer

public final class Qwen3TTSTokenizer {
    private let tokenizer: Tokenizer

    // Special token IDs for TTS
    public let startOfSpeechId: Int
    public let endOfSpeechId: Int
    public let audioTokensStart: Int
    public let padTokenId: Int
    public let eosTokenId: Int

    // Chat template tokens
    public let imStartId: Int
    public let imEndId: Int

    public init(
        tokenizer: Tokenizer,
        startOfSpeechId: Int = Qwen3TTSResources.ttsBosTokenId,
        endOfSpeechId: Int = Qwen3TTSResources.ttsEosTokenId,
        audioTokensStart: Int = 151679,  // Tokens >= this are audio tokens
        padTokenId: Int = 151643,
        eosTokenId: Int = 151645,
        imStartId: Int = 151644,
        imEndId: Int = 151645
    ) {
        self.tokenizer = tokenizer
        self.startOfSpeechId = startOfSpeechId
        self.endOfSpeechId = endOfSpeechId
        self.audioTokensStart = audioTokensStart
        self.padTokenId = padTokenId
        self.eosTokenId = eosTokenId
        self.imStartId = imStartId
        self.imEndId = imEndId
    }

    /// Load tokenizer from model directory
    public static func load(from directory: URL, hubApi: HubApi = .shared) throws -> Qwen3TTSTokenizer {
        let tokenizerConfigURL = directory.appending(path: "tokenizer_config.json")
        let vocabURL = directory.appending(path: "vocab.json")
        let mergesURL = directory.appending(path: "merges.txt")

        guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
            throw Qwen3TTSTokenizerError.fileNotFound(tokenizerConfigURL)
        }

        let tokenizerConfig = try hubApi.configuration(fileURL: tokenizerConfigURL)

        // Try tokenizer.json first, fall back to vocab.json + merges.txt
        let tokenizerDataURL = directory.appending(path: "tokenizer.json")
        let tokenizer: Tokenizer
        if FileManager.default.fileExists(atPath: tokenizerDataURL.path) {
            let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)
            tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        } else {
            // Build BPE tokenizer from vocab.json + merges.txt
            guard FileManager.default.fileExists(atPath: vocabURL.path),
                  FileManager.default.fileExists(atPath: mergesURL.path) else {
                throw Qwen3TTSTokenizerError.fileNotFound(vocabURL)
            }
            let tokenizerData = try makeBPETokenizerData(
                vocabURL: vocabURL,
                mergesURL: mergesURL,
                tokenizerConfigURL: tokenizerConfigURL,
                directory: directory
            )
            tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        }

        // Get special token IDs from config or use defaults
        let padTokenId = tokenizer.convertTokenToId("<|endoftext|>") ?? 151643
        let eosTokenId = tokenizer.convertTokenToId("<|im_end|>") ?? 151645
        let imStartId = tokenizer.convertTokenToId("<|im_start|>") ?? 151644

        return Qwen3TTSTokenizer(
            tokenizer: tokenizer,
            padTokenId: padTokenId,
            eosTokenId: eosTokenId,
            imStartId: imStartId,
            imEndId: eosTokenId
        )
    }

    /// Build tokenizer data from BPE vocab and merges files
    private static func makeBPETokenizerData(
        vocabURL: URL,
        mergesURL: URL,
        tokenizerConfigURL: URL,
        directory: URL
    ) throws -> Config {
        let vocabData = try Data(contentsOf: vocabURL)
        var vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)

        var addedList: [Qwen3TokenizerAddedTokenEntry] = []
        var addedTokenIds = Set<Int>()
        var addedTokenContents = Set<String>()

        func addAddedToken(
            id: Int,
            content: String,
            lstrip: Bool,
            rstrip: Bool,
            special: Bool,
            singleWord: Bool? = nil,
            normalized: Bool? = nil
        ) {
            guard !addedTokenIds.contains(id) && !addedTokenContents.contains(content) else { return }
            addedList.append(
                Qwen3TokenizerAddedTokenEntry(
                    id: id,
                    content: content,
                    lstrip: lstrip,
                    rstrip: rstrip,
                    special: special,
                    singleWord: singleWord,
                    normalized: normalized
                )
            )
            addedTokenIds.insert(id)
            addedTokenContents.insert(content)
            if vocab[content] == nil {
                vocab[content] = id
            }
        }

        // Load added tokens from tokenizer_config.json (added_tokens_decoder)
        if let configData = try? Data(contentsOf: tokenizerConfigURL),
           let configObject = try? JSONDecoder().decode(Qwen3TokenizerConfigDocument.self, from: configData),
           let addedDecoder = configObject.addedTokensDecoder {
            for (idStr, value) in addedDecoder {
                guard let id = Int(idStr), !value.content.isEmpty else { continue }
                addAddedToken(
                    id: id,
                    content: value.content,
                    lstrip: value.lstrip ?? false,
                    rstrip: value.rstrip ?? false,
                    special: value.special ?? true,
                    singleWord: value.singleWord,
                    normalized: value.normalized
                )
            }
        }

        // Load added_tokens.json if it exists
        let addedTokensURL = directory.appending(path: "added_tokens.json")
        if FileManager.default.fileExists(atPath: addedTokensURL.path) {
            if let addedData = try? Data(contentsOf: addedTokensURL),
               let added = try? JSONDecoder().decode([String: Int].self, from: addedData) {
                for (k, v) in added {
                    addAddedToken(id: v, content: k, lstrip: false, rstrip: false, special: true)
                }
            }
        }

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        let merges: [String] = mergesText
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        let tokenizerDocument = Qwen3TokenizerDocument(
            model: Qwen3TokenizerDocument.Model(vocab: vocab, merges: merges),
            preTokenizer: Qwen3TokenizerDocument.ByteLevelComponent(),
            decoder: Qwen3TokenizerDocument.DecoderComponent(),
            addedTokens: addedList.isEmpty ? nil : addedList
        )
        let data = try JSONEncoder().encode(tokenizerDocument)
        return try JSONDecoder().decode(Config.self, from: data)
    }

    /// Encode text for TTS generation
    /// Format: <|im_start|>user\n<instruct>{voice_description}</instruct>\n<text>{text}</text><|im_end|>\n<|im_start|>assistant\n
    public func encodeTTSPrompt(text: String, voiceDescription: String) -> [Int] {
        let prompt = """
<|im_start|>user
<instruct>\(voiceDescription)</instruct>
<text>\(text)</text><|im_end|>
<|im_start|>assistant
"""
        return tokenizer.encode(text: prompt)
    }

    /// Encode raw text using the underlying tokenizer.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    /// Encode text with SOH format for TTS
    /// This is an alternative format some models use
    public func encodeSOHFormat(text: String) -> [Int] {
        // Format: [SOH] text [EOT EOH]
        // SOH = Start of Header, EOT = End of Text, EOH = End of Header
        let encoded = tokenizer.encode(text: text)
        return encoded
    }

    /// Decode tokens back to text (useful for debugging)
    public func decode(tokens: [Int]) -> String {
        // Filter out audio tokens before decoding
        let textTokens = tokens.filter { $0 < audioTokensStart }
        return tokenizer.decode(tokens: textTokens)
    }

    /// Extract audio tokens from generated sequence
    /// - Parameter tokens: Full generated token sequence
    /// - Returns: Audio tokens with offset subtracted (ready for SNAC decoder)
    public func extractAudioTokens(_ tokens: [Int]) -> [Int] {
        var audioTokens: [Int] = []
        var inSpeech = false

        for token in tokens {
            if token == startOfSpeechId {
                inSpeech = true
                continue
            }

            if token == endOfSpeechId {
                break
            }

            if inSpeech && token >= audioTokensStart {
                // Subtract offset to get raw audio code
                audioTokens.append(token - audioTokensStart)
            }
        }

        return audioTokens
    }

    /// Check if a token is an audio token
    public func isAudioToken(_ token: Int) -> Bool {
        token >= audioTokensStart
    }

    /// Check if a token signals end of speech
    public func isEndOfSpeech(_ token: Int) -> Bool {
        token == endOfSpeechId
    }
}

private struct Qwen3TokenizerConfigDocument: Decodable {
    let addedTokensDecoder: [String: Qwen3TokenizerConfigEntry]?

    enum CodingKeys: String, CodingKey {
        case addedTokensDecoder = "added_tokens_decoder"
    }
}

private struct Qwen3TokenizerConfigEntry: Decodable {
    let content: String
    let lstrip: Bool?
    let rstrip: Bool?
    let special: Bool?
    let singleWord: Bool?
    let normalized: Bool?

    enum CodingKeys: String, CodingKey {
        case content
        case lstrip
        case rstrip
        case special
        case singleWord = "single_word"
        case normalized
    }
}

private struct Qwen3TokenizerAddedTokenEntry: Encodable {
    let id: Int
    let content: String
    let lstrip: Bool
    let rstrip: Bool
    let special: Bool
    let singleWord: Bool?
    let normalized: Bool?

    enum CodingKeys: String, CodingKey {
        case id
        case content
        case lstrip
        case rstrip
        case special
        case singleWord = "single_word"
        case normalized
    }
}

private struct Qwen3TokenizerDocument: Encodable {
    let model: Model
    let preTokenizer: ByteLevelComponent
    let decoder: DecoderComponent
    let addedTokens: [Qwen3TokenizerAddedTokenEntry]?

    enum CodingKeys: String, CodingKey {
        case model
        case preTokenizer = "pre_tokenizer"
        case decoder
        case addedTokens = "added_tokens"
    }

    struct Model: Encodable {
        let vocab: [String: Int]
        let merges: [String]
    }

    struct ByteLevelComponent: Encodable {
        let type: String = "ByteLevel"
        let addPrefixSpace: Bool = false
        let trimOffsets: Bool = true
        let useRegex: Bool = true

        enum CodingKeys: String, CodingKey {
            case type
            case addPrefixSpace = "add_prefix_space"
            case trimOffsets = "trim_offsets"
            case useRegex = "use_regex"
        }
    }

    struct DecoderComponent: Encodable {
        let type: String = "ByteLevel"
    }
}

// MARK: - Errors

public enum Qwen3TTSTokenizerError: LocalizedError {
    case fileNotFound(URL)
    case tokenizationFailed

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let url):
            return "Tokenizer file not found: \(url.path)"
        case .tokenizationFailed:
            return "Failed to tokenize input text"
        }
    }
}
