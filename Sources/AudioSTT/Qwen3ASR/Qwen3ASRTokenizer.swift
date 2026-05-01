import Foundation
import MLX
@preconcurrency import Tokenizers
@preconcurrency import Hub

// MARK: - Qwen3 ASR Tokenizer

/// Tokenizer wrapper for Qwen3-ASR models
public final class Qwen3ASRTokenizer {
    private let tokenizer: Tokenizer

    // Special token IDs
    public let audioTokenId: Int
    public let audioStartTokenId: Int
    public let audioEndTokenId: Int
    public let asrTextTokenId: Int
    public let eosTokenId: Int
    public let padTokenId: Int
    public let imStartId: Int
    public let imEndId: Int

    public init(
        tokenizer: Tokenizer,
        audioTokenId: Int = 151676,
        audioStartTokenId: Int = 151669,
        audioEndTokenId: Int = 151670,
        asrTextTokenId: Int = 151704,
        eosTokenId: Int = 151645,
        padTokenId: Int = 151643,
        imStartId: Int = 151644,
        imEndId: Int = 151645
    ) {
        self.tokenizer = tokenizer
        self.audioTokenId = audioTokenId
        self.audioStartTokenId = audioStartTokenId
        self.audioEndTokenId = audioEndTokenId
        self.asrTextTokenId = asrTextTokenId
        self.eosTokenId = eosTokenId
        self.padTokenId = padTokenId
        self.imStartId = imStartId
        self.imEndId = imEndId
    }

    /// Load tokenizer from model directory
    public static func load(from directory: URL, config: Qwen3ASRModelConfig? = nil, hubApi: HubApi = .shared) throws -> Qwen3ASRTokenizer {
        let tokenizerConfigURL = directory.appending(path: "tokenizer_config.json")
        let vocabURL = directory.appending(path: "vocab.json")
        let mergesURL = directory.appending(path: "merges.txt")

        guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
            throw Qwen3ASRTokenizerError.fileNotFound(tokenizerConfigURL)
        }

        let tokenizerConfig = try hubApi.configuration(fileURL: tokenizerConfigURL)

        // Try tokenizer.json first, fall back to vocab.json + merges.txt
        let tokenizerDataURL = directory.appending(path: "tokenizer.json")
        let tokenizer: Tokenizer
        if FileManager.default.fileExists(atPath: tokenizerDataURL.path) {
            let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)
            tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        } else {
            guard FileManager.default.fileExists(atPath: vocabURL.path),
                  FileManager.default.fileExists(atPath: mergesURL.path) else {
                throw Qwen3ASRTokenizerError.fileNotFound(vocabURL)
            }
            let tokenizerData = try makeBPETokenizerData(
                vocabURL: vocabURL,
                mergesURL: mergesURL,
                tokenizerConfigURL: tokenizerConfigURL,
                directory: directory
            )
            tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        }

        // Get special token IDs from config or from tokenizer
        let padTokenId = tokenizer.convertTokenToId("<|endoftext|>") ?? 151643
        let eosTokenId = tokenizer.convertTokenToId("<|im_end|>") ?? 151645
        let imStartId = tokenizer.convertTokenToId("<|im_start|>") ?? 151644

        // Use config token IDs if available (more reliable)
        let audioTokenId = config?.audioTokenId ?? tokenizer.convertTokenToId("<|audio_pad|>") ?? 151676
        let audioStartTokenId = config?.audioStartTokenId ?? tokenizer.convertTokenToId("<|audio_start|>") ?? 151669
        let audioEndTokenId = config?.audioEndTokenId ?? tokenizer.convertTokenToId("<|audio_end|>") ?? 151670
        let asrTextTokenId = tokenizer.convertTokenToId("<asr_text>") ?? 151704

        return Qwen3ASRTokenizer(
            tokenizer: tokenizer,
            audioTokenId: audioTokenId,
            audioStartTokenId: audioStartTokenId,
            audioEndTokenId: audioEndTokenId,
            asrTextTokenId: asrTextTokenId,
            eosTokenId: eosTokenId,
            padTokenId: padTokenId,
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
        guard let vocabObject = try JSONSerialization.jsonObject(with: vocabData, options: []) as? [String: Any] else {
            throw Qwen3ASRTokenizerError.fileNotFound(vocabURL)
        }
        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(vocabObject.count)
        for (k, v) in vocabObject {
            if let i = v as? Int { vocab[k] = i }
        }

        var addedList: [[String: Any]] = []
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
            var entry: [String: Any] = [
                "id": id,
                "content": content,
                "lstrip": lstrip,
                "rstrip": rstrip,
                "special": special
            ]
            if let singleWord { entry["single_word"] = singleWord }
            if let normalized { entry["normalized"] = normalized }
            addedList.append(entry)
            addedTokenIds.insert(id)
            addedTokenContents.insert(content)
            if vocab[content] == nil {
                vocab[content] = id
            }
        }

        // Load added tokens from tokenizer_config.json
        if let configData = try? Data(contentsOf: tokenizerConfigURL),
           let configObject = try? JSONSerialization.jsonObject(with: configData, options: []) as? [String: Any],
           let addedDecoder = configObject["added_tokens_decoder"] as? [String: Any] {
            for (idStr, value) in addedDecoder {
                guard let id = Int(idStr),
                      let info = value as? [String: Any],
                      let content = info["content"] as? String else { continue }
                let lstrip = (info["lstrip"] as? Bool) ?? false
                let rstrip = (info["rstrip"] as? Bool) ?? false
                let special = (info["special"] as? Bool) ?? true
                let singleWord = info["single_word"] as? Bool
                let normalized = info["normalized"] as? Bool
                addAddedToken(
                    id: id,
                    content: content,
                    lstrip: lstrip,
                    rstrip: rstrip,
                    special: special,
                    singleWord: singleWord,
                    normalized: normalized
                )
            }
        }

        // Load added_tokens.json if it exists
        let addedTokensURL = directory.appending(path: "added_tokens.json")
        if FileManager.default.fileExists(atPath: addedTokensURL.path) {
            if let addedData = try? Data(contentsOf: addedTokensURL),
               let added = try? JSONSerialization.jsonObject(with: addedData, options: []) as? [String: Any] {
                for (k, v) in added {
                    if let i = v as? Int {
                        addAddedToken(id: i, content: k, lstrip: false, rstrip: false, special: true)
                    }
                }
            }
        }

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        let merges: [String] = mergesText
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        var tokenizerDict: [String: Any] = [
            "model": [
                "vocab": vocab,
                "merges": merges
            ],
            "pre_tokenizer": [
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            ],
            "decoder": [
                "type": "ByteLevel"
            ]
        ]

        if !addedList.isEmpty {
            tokenizerDict["added_tokens"] = addedList
        }

        let data = try JSONSerialization.data(withJSONObject: tokenizerDict, options: [])
        return try JSONDecoder().decode(Config.self, from: data)
    }

    /// Encode text to token IDs
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    /// Decode token IDs to text
    public func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    /// Create ASR prompt with audio placeholder
    /// Format: <|im_start|>user\nAudio:<|audio|><|im_end|>\n<|im_start|>assistant\n
    public func createASRPrompt(
        audioPlaceholderCount: Int,
        instruction: String? = nil,
        assistantPrefix: String? = nil,
        includeSystemPrompt: Bool = true,
        systemPrompt: String = "You are a helpful assistant.",
        includeAudioBoundaryTokens: Bool = true,
        includeAsrTextToken: Bool = true
    ) -> [Int] {
        var tokens: [Int] = []

        if includeSystemPrompt {
            tokens.append(imStartId)
            tokens.append(contentsOf: encode("system\n\(systemPrompt)"))
            tokens.append(imEndId)
            tokens.append(contentsOf: encode("\n"))
        }

        tokens.append(imStartId)
        tokens.append(contentsOf: encode("user\n"))
        tokens.append(contentsOf: encode("Audio 1: "))

        if includeAudioBoundaryTokens {
            tokens.append(audioStartTokenId)
        }

        // Audio placeholders
        for _ in 0..<audioPlaceholderCount {
            tokens.append(audioTokenId)
        }

        if includeAudioBoundaryTokens {
            tokens.append(audioEndTokenId)
        }

        if let instruction, !instruction.isEmpty {
            tokens.append(contentsOf: encode("\n\(instruction)"))
        }

        tokens.append(imEndId)
        tokens.append(contentsOf: encode("\n"))
        tokens.append(imStartId)
        tokens.append(contentsOf: encode("assistant\n"))

        if let assistantPrefix, !assistantPrefix.isEmpty {
            tokens.append(contentsOf: encode(assistantPrefix))
        }

        if includeAsrTextToken {
            tokens.append(asrTextTokenId)
        }

        return tokens
    }

    /// Create Qwen3-ASR prompt matching mlx-audio reference format.
    /// Format:
    /// <|im_start|>system\n<|im_end|>\n
    /// <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
    /// <|im_start|>assistant\nlanguage {lang}<asr_text>
    public func createQwen3ASRPrompt(
        audioPlaceholderCount: Int,
        language: String?,
        supportedLanguages: [String]? = nil
    ) -> [Int] {
        let trimmed = language?.trimmingCharacters(in: .whitespacesAndNewlines)
        let fallback = "English"
        let requested = (trimmed?.isEmpty == false) ? trimmed! : fallback

        let langName: String
        if let supportedLanguages, !supportedLanguages.isEmpty {
            let lowerMap = Dictionary(
                uniqueKeysWithValues: supportedLanguages.map { ($0.lowercased(), $0) }
            )
            langName = lowerMap[requested.lowercased()] ?? requested
        } else {
            langName = requested
        }

        var tokens: [Int] = []

        // system (empty)
        tokens.append(imStartId)
        tokens.append(contentsOf: encode("system\n"))
        tokens.append(imEndId)
        tokens.append(contentsOf: encode("\n"))

        // user with audio
        tokens.append(imStartId)
        tokens.append(contentsOf: encode("user\n"))
        tokens.append(audioStartTokenId)
        if audioPlaceholderCount > 0 {
            tokens.append(contentsOf: Array(repeating: audioTokenId, count: audioPlaceholderCount))
        }
        tokens.append(audioEndTokenId)
        tokens.append(imEndId)
        tokens.append(contentsOf: encode("\n"))

        // assistant prefix
        tokens.append(imStartId)
        tokens.append(contentsOf: encode("assistant\nlanguage \(langName)"))
        tokens.append(asrTextTokenId)

        return tokens
    }

    /// Create audio mask for input tokens
    /// Returns 1 for audio token positions, 0 elsewhere
    public func createAudioMask(for tokens: [Int]) -> [Int] {
        tokens.map { $0 == audioTokenId ? 1 : 0 }
    }
}

// MARK: - Errors

public enum Qwen3ASRTokenizerError: LocalizedError {
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
