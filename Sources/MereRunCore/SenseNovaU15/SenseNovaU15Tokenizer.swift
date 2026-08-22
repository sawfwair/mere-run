import Foundation
@preconcurrency import Hub
@preconcurrency import Tokenizers

public final class SenseNovaU15Tokenizer {
    public static let imageStart = "<img>"
    public static let imageEnd = "</img>"
    public static let imageContext = "<IMG_CONTEXT>"

    public struct EncodedPrompt: Sendable, Equatable {
        public let tokenIDs: [Int]
        public let timeIndexes: [Int32]
        public let heightIndexes: [Int32]
        public let widthIndexes: [Int32]
        public let imageContextPositions: [Int]
    }

    private static let systemMessage = """
    You are an image generation and editing assistant that accurately understands and executes user intent.

    You support two modes:

    1. Think Mode:
    If the task requires reasoning, you MUST start with a <think></think> block. Put all reasoning inside the block using plain text. DO NOT include any image tags. Keep it reasonable and directly useful for producing the final image.

    2. Non-Think Mode:
    If no reasoning is needed, directly produce the final image.

    Task Types:

    A. Text-to-Image Generation:
    - Generate a high-quality image based on the user's description.
    - Ensure visual clarity, semantic consistency, and completeness.
    - DO NOT introduce elements that contradict or override the user's intent.

    B. Image Editing:
    - Use the provided image(s) as input or reference for modification or transformation.
    - The result can be an edited image or a new image based on the reference(s).
    - Preserve all unspecified attributes unless explicitly changed.

    General Rules:
    - For any visible text in the image, follow the language specified for the rendered text in the user's description, not the language of the prompt. If no language is specified, use the user's input language.
    """

    public let tokenizer: any Tokenizer
    public let imageStartTokenID: Int
    public let imageEndTokenID: Int
    public let imageContextTokenID: Int

    public init(tokenizer: any Tokenizer) throws {
        self.tokenizer = tokenizer
        guard let imageStartTokenID = tokenizer.convertTokenToId(Self.imageStart),
              let imageEndTokenID = tokenizer.convertTokenToId(Self.imageEnd),
              let imageContextTokenID = tokenizer.convertTokenToId(Self.imageContext) else {
            throw SenseNovaU15Error.tokenizerTokenMissing("<img>, </img>, or <IMG_CONTEXT>")
        }
        self.imageStartTokenID = imageStartTokenID
        self.imageEndTokenID = imageEndTokenID
        self.imageContextTokenID = imageContextTokenID
    }

    public static func load(
        from resources: SenseNovaU15Resources,
        hubApi: HubApi = .shared
    ) async throws -> SenseNovaU15Tokenizer {
        let tokenizer: any Tokenizer
        if FileManager.default.fileExists(atPath: resources.tokenizerJSONURL.path) {
            tokenizer = try await AutoTokenizer.from(modelFolder: resources.rootURL, hubApi: hubApi)
        } else {
            tokenizer = try AutoTokenizer.from(
                tokenizerConfig: hubApi.configuration(fileURL: resources.tokenizerConfigURL),
                tokenizerData: try makeQwenTokenizerData(resources: resources)
            )
        }
        return try SenseNovaU15Tokenizer(tokenizer: tokenizer)
    }

    public func textToImage(_ prompt: String) -> EncodedPrompt {
        encode(query(user: prompt, system: Self.systemMessage, suffix: "<think>\n\n</think>\n\n<img>"), grids: [])
    }

    public func unconditional(_ negativePrompt: String = "") -> EncodedPrompt {
        encode(query(user: negativePrompt, system: nil, suffix: "<img>"), grids: [])
    }

    public func imageEdit(_ prompt: String, grids: [(height: Int, width: Int)]) -> EncodedPrompt {
        var user = prompt
        let existing = user.components(separatedBy: "<image>").count - 1
        if existing == 0 && grids.count > 1 {
            user = grids.indices.map { "Image-\($0 + 1):<image>\n" }.joined() + user
        } else if existing < grids.count {
            user = String(repeating: "<image>\n", count: grids.count - existing) + user
        }
        let expanded = expandImagePlaceholders(in: user, grids: grids)
        return encode(
            query(user: expanded, system: Self.systemMessage, suffix: "<think>\n\n</think>\n\n<img>"),
            grids: grids
        )
    }

    public func imageOnly(grids: [(height: Int, width: Int)]) -> EncodedPrompt {
        let placeholders = String(repeating: "<image>", count: grids.count)
        return encode(
            query(user: expandImagePlaceholders(in: placeholders, grids: grids), system: nil, suffix: "<img>"),
            grids: grids
        )
    }

    private func query(user: String, system: String?, suffix: String) -> String {
        let systemPart = system.map { "<|im_start|>system\n\($0)<|im_end|>\n" } ?? ""
        return systemPart
            + "<|im_start|>user\n\(user)<|im_end|>\n"
            + "<|im_start|>assistant\n\(suffix)"
    }

    private func expandImagePlaceholders(
        in prompt: String,
        grids: [(height: Int, width: Int)]
    ) -> String {
        var result = prompt
        for grid in grids {
            let count = grid.height * grid.width / 4
            let replacement = Self.imageStart
                + String(repeating: Self.imageContext, count: count)
                + Self.imageEnd
            if let range = result.range(of: "<image>") {
                result.replaceSubrange(range, with: replacement)
            }
        }
        return result
    }

    private func encode(
        _ prompt: String,
        grids: [(height: Int, width: Int)]
    ) -> EncodedPrompt {
        let tokenIDs = tokenizer.encode(text: prompt)
        var timeIndexes = [Int32](repeating: 0, count: tokenIDs.count)
        var heightIndexes = [Int32](repeating: 0, count: tokenIDs.count)
        var widthIndexes = [Int32](repeating: 0, count: tokenIDs.count)
        var positions: [Int] = []
        var time = -1
        var startsImage = false
        for (index, tokenID) in tokenIDs.enumerated() {
            if startsImage || tokenID != imageContextTokenID { time += 1 }
            timeIndexes[index] = Int32(time)
            startsImage = tokenID == imageStartTokenID
            if tokenID == imageContextTokenID { positions.append(index) }
        }

        var positionOffset = 0
        for grid in grids {
            let mergedHeight = grid.height / 2
            let mergedWidth = grid.width / 2
            for row in 0..<mergedHeight {
                for column in 0..<mergedWidth {
                    guard positionOffset < positions.count else { break }
                    let tokenPosition = positions[positionOffset]
                    heightIndexes[tokenPosition] = Int32(row)
                    widthIndexes[tokenPosition] = Int32(column)
                    positionOffset += 1
                }
            }
        }
        return EncodedPrompt(
            tokenIDs: tokenIDs,
            timeIndexes: timeIndexes,
            heightIndexes: heightIndexes,
            widthIndexes: widthIndexes,
            imageContextPositions: positions
        )
    }

    private static func makeQwenTokenizerData(resources: SenseNovaU15Resources) throws -> Config {
        let vocabData = try Data(contentsOf: resources.vocabURL)
        let vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)
        let merges = try String(contentsOf: resources.mergesURL, encoding: .utf8)
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
        let tokenizerConfigData = try Data(contentsOf: resources.tokenizerConfigURL)
        let tokenizerConfig = try JSONDecoder().decode(TokenizerConfigFile.self, from: tokenizerConfigData)
        let addedTokens = try tokenizerConfig.addedTokensDecoder.map { id, token -> Config in
            guard let tokenID = Int(id) else {
                throw SenseNovaU15Error.invalidConfiguration("invalid added token id \(id)")
            }
            return Config([
                "id": Config(tokenID),
                "content": Config(token.content),
                "singleWord": Config(token.singleWord),
                "lstrip": Config(token.lstrip),
                "rstrip": Config(token.rstrip),
                "normalized": Config(token.normalized),
                "special": Config(token.special),
            ])
        }.sorted { lhs, rhs in
            lhs.id.integer(or: 0) < rhs.id.integer(or: 0)
        }

        let splitPattern = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        let byteLevel = Config([
            "type": Config("ByteLevel"),
            "addPrefixSpace": Config(false),
            "trimOffsets": Config(false),
            "useRegex": Config(false),
        ])
        let vocabValues = vocab.reduce(into: [String: Config]()) { result, item in
            result[item.key] = Config(item.value)
        }
        let mergeValues = merges.map { Config($0) }
        let model = Config([
            "type": Config("BPE"),
            "vocab": Config(vocabValues),
            "merges": Config(mergeValues),
            "fuseUnk": Config(false),
            "byteFallback": Config(false),
            "ignoreMerges": Config(false),
        ])
        let preTokenizer = Config([
            "type": Config("Sequence"),
            "pretokenizers": Config([
                Config([
                    "type": Config("Split"),
                    "pattern": Config(["Regex": Config(splitPattern)]),
                    "behavior": Config("Isolated"),
                    "invert": Config(false),
                ]),
                byteLevel,
            ]),
        ])
        return Config([
            "version": Config("1.0"),
            "addedTokens": Config(addedTokens),
            "normalizer": Config(["type": Config("NFC")]),
            "preTokenizer": preTokenizer,
            "postProcessor": byteLevel,
            "decoder": byteLevel,
            "model": model,
        ])
    }
}

private struct TokenizerConfigFile: Decodable {
    let addedTokensDecoder: [String: TokenizerAddedToken]

    enum CodingKeys: String, CodingKey {
        case addedTokensDecoder = "added_tokens_decoder"
    }
}

private struct TokenizerAddedToken: Decodable {
    let content: String
    let lstrip: Bool
    let normalized: Bool
    let rstrip: Bool
    let singleWord: Bool
    let special: Bool

    enum CodingKeys: String, CodingKey {
        case content
        case lstrip
        case normalized
        case rstrip
        case singleWord = "single_word"
        case special
    }
}
