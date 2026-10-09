import Foundation
import Hub
import Tokenizers

struct PPLXEmbedV2Sequence {
    let ids: [Int]
    let spans: [Range<Int>]
}

struct PPLXEmbedV2Tokenizer {
    let tokenizer: any Tokenizer
    let plainTokenizer: any Tokenizer
    let queryID: Int
    let documentID: Int
    let skipIDs: Set<Int>
    let normalizesNFC: Bool

    private struct TokenizerContract: Decodable {
        struct Model: Decodable { let type: String }
        struct Normalizer: Decodable { let type: String }
        struct AddedToken: Decodable {
            let id: Int
            let content: String
            let singleWord: Bool
            let lstrip: Bool
            let rstrip: Bool
            let normalized: Bool
            let special: Bool
            enum CodingKeys: String, CodingKey { case id, content, singleWord = "single_word", lstrip, rstrip, normalized, special }
            var nativeConfig: Config {
                Config(["id": Config(id), "content": Config(content), "single_word": Config(singleWord),
                        "lstrip": Config(lstrip), "rstrip": Config(rstrip), "normalized": Config(normalized), "special": Config(special)])
            }
        }
        let model: Model
        let normalizer: Normalizer?
        let addedTokens: [AddedToken]
        enum CodingKeys: String, CodingKey { case model, normalizer, addedTokens = "added_tokens" }
    }

    private struct MaskConfig: Decodable {
        let skiplistWords: [String]
        let skiplistTasks: [String]
        let keepOnlyTokenIDs: [Int]?
        enum CodingKeys: String, CodingKey {
            case skiplistWords = "skiplist_words", skiplistTasks = "skiplist_tasks", keepOnlyTokenIDs = "keep_only_token_ids"
        }
    }

    static func load(root: URL, contextual: Bool) throws -> Self {
        let config = try HubApi.shared.configuration(fileURL: root.appending(path: "tokenizer_config.json"))
        let data = try HubApi.shared.configuration(fileURL: root.appending(path: "tokenizer.json"))
        let contract = try JSONDecoder().decode(TokenizerContract.self, from: Data(contentsOf: root.appending(path: "tokenizer.json")))
        guard contract.model.type == "BPE", contract.normalizer == nil || contract.normalizer?.type == "NFC" else {
            throw PPLXEmbedV2Error.invalidConfiguration("PPLX Embed v2 requires byte-level BPE with optional NFC normalization.")
        }
        var nativeConfig = config.dictionary(or: [:])
        nativeConfig["tokenizer_class"] = Config("Qwen2Tokenizer")
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(nativeConfig), tokenizerData: data)
        guard let queryID = tokenizer.convertTokenToId("[Q] "), let documentID = tokenizer.convertTokenToId("[D] ") else {
            throw PPLXEmbedV2Error.invalidConfiguration("Missing PPLX query/document marker tokens.")
        }
        // Upstream contextual encoding uses split_special_tokens=True. In
        // particular, document prefixes are ordinary BPE, while the real
        // context tokenizer's non-special added chunk separator stays intact.
        var plainData = data.dictionary(or: [:])
        plainData["added_tokens"] = Config(contract.addedTokens.filter { !$0.special }.map(\.nativeConfig))
        nativeConfig["added_tokens_decoder"] = Config([String: Config]())
        let plain = try AutoTokenizer.from(tokenizerConfig: Config(nativeConfig), tokenizerData: Config(plainData))
        var skipIDs: Set<Int> = []
        if !contextual {
            let mask = try JSONDecoder().decode(MaskConfig.self, from: Data(contentsOf: root.appending(path: "2_MultiVectorMask/config.json")))
            guard mask.skiplistTasks == ["document"], mask.keepOnlyTokenIDs == nil else {
                throw PPLXEmbedV2Error.invalidConfiguration("Unsupported PPLX multi-vector mask configuration.")
            }
            skipIDs = Set(mask.skiplistWords.compactMap { tokenizer.convertTokenToId($0) })
        }
        return Self(tokenizer: tokenizer, plainTokenizer: plain, queryID: queryID, documentID: documentID,
                    skipIDs: skipIDs, normalizesNFC: contract.normalizer?.type == "NFC")
    }

    func late(_ text: String, task: PPLXEmbedV2Task, limit: Int) -> PPLXEmbedV2Sequence {
        let marker = task == .query ? queryID : documentID
        return .init(ids: [marker] + tokenizer.encode(text: text, addSpecialTokens: false).prefix(limit - 1), spans: [])
    }

    func contextual(_ chunks: [String], task: PPLXEmbedV2Task, config: PPLXEmbedV2Config, limit: Int) throws -> PPLXEmbedV2Sequence {
        guard task != .query || chunks.count == 1 else {
            throw PPLXEmbedV2Error.invalidInput("Each contextual query must contain exactly one string.")
        }
        if task == .query {
            let ids = [queryID] + plainTokenizer.encode(text: chunks[0], addSpecialTokens: false)
            guard ids.count <= limit else { throw PPLXEmbedV2Error.invalidInput("Contextual query exceeds the \(limit)-token limit.") }
            return .init(ids: ids, spans: [0..<ids.count])
        }
        var text = config.documentPrefix
        var byteSpans: [Range<Int>] = []
        for (index, chunk) in chunks.enumerated() {
            if index > 0 { text += config.boundaryMarker }
            let start = text.utf8.count
            // Chunk separators and the document prefix isolate normalization
            // boundaries. Measure the normalized bytes consumed by BPE, so
            // decomposed accents and Hangul retain the upstream chunk spans.
            text += normalizesNFC ? chunk.precomposedStringWithCanonicalMapping : chunk
            byteSpans.append(start..<text.utf8.count)
        }
        let tokens = plainTokenizer.tokenize(text: text)
        let ids = try tokens.map { token -> Int in
            guard let id = plainTokenizer.convertTokenToId(token) else {
                throw PPLXEmbedV2Error.invalidConfiguration("Unmapped PPLX BPE token.")
            }
            return id
        }
        guard ids.count <= limit else { throw PPLXEmbedV2Error.invalidInput("Contextual document exceeds the \(limit)-token limit; split into context windows.") }
        let offsets = Self.byteOffsets(tokens: tokens)
        guard offsets.last?.upperBound == text.utf8.count else {
            throw PPLXEmbedV2Error.invalidConfiguration("PPLX tokenizer changed the input bytes.")
        }
        return .init(ids: ids, spans: Self.overlappingTokenSpans(offsets: offsets, chunks: byteSpans))
    }

    static func byteOffsets(tokens: [String]) -> [Range<Int>] {
        // Each Unicode scalar in a byte-level BPE token represents one byte,
        // even when a UTF-8 character is split across multiple token IDs.
        var cursor = 0
        return tokens.map { token in
            let start = cursor
            cursor += token.unicodeScalars.count
            return start..<cursor
        }
    }

    static func overlappingTokenSpans(offsets: [Range<Int>], chunks: [Range<Int>]) -> [Range<Int>] {
        chunks.map { chunk in
            guard !chunk.isEmpty else { return 0..<0 }
            let indices = offsets.indices.filter { offsets[$0].lowerBound < chunk.upperBound && offsets[$0].upperBound > chunk.lowerBound }
            guard let first = indices.first, let last = indices.last else { return 0..<0 }
            return first..<(last + 1)
        }
    }
}
