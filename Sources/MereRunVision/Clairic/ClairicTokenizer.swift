import MereRunCore
#if !os(iOS)
import Foundation
import Hub
import Tokenizers

struct ClairicTokenizer {
    private struct Vocabulary: Decodable {
        struct Model: Decodable {
            let type: String
            let vocab: [String: Int]
        }
        struct DecoderKind: Decodable { let type: String }
        let model: Model
        let decoder: DecoderKind
    }

    private let tokenizer: any Tokenizer
    private let pieces: [Int: [UInt8]]
    let buckets: [[Int]]
    let eos: Int
    let null: Int

    init(root: URL, vocabularySize: Int) throws {
        let tokenizerURL = root.appending(path: "tokenizer.json")
        let vocabulary = try JSONDecoder().decode(Vocabulary.self, from: Data(contentsOf: tokenizerURL))
        guard vocabulary.model.type == "BPE", vocabulary.decoder.type == "ByteLevel" else {
            throw ClairicError.invalid("Clairic requires its byte-level BPE tokenizer.")
        }
        let config = try HubApi.shared.configuration(fileURL: root.appending(path: "tokenizer_config.json"))
        var nativeConfig = config.dictionary(or: [:])
        nativeConfig["tokenizer_class"] = Config("Qwen2Tokenizer")
        tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(nativeConfig),
                                           tokenizerData: HubApi.shared.configuration(fileURL: tokenizerURL))
        guard let eos = tokenizer.convertTokenToId("<|im_end|>"),
              let null = vocabulary.model.vocab["null"], (0..<vocabularySize).contains(eos) else {
            throw ClairicError.invalid("Clairic tokenizer is missing its end or null token.")
        }
        self.eos = eos
        self.null = null

        let mapping = Self.byteMapping()
        var pieces: [Int: [UInt8]] = [:]
        var buckets = Array(repeating: [Int](), count: 256)
        for (token, id) in vocabulary.model.vocab {
            let bytes = token.unicodeScalars.compactMap { mapping[$0.value] }
            guard id >= 0, id < vocabularySize, bytes.count == token.unicodeScalars.count, let first = bytes.first else {
                throw ClairicError.invalid("Clairic tokenizer has an invalid byte-level vocabulary entry.")
            }
            guard pieces.updateValue(bytes, forKey: id) == nil else {
                throw ClairicError.invalid("Clairic tokenizer contains duplicate token IDs.")
            }
            buckets[Int(first)].append(id)
        }
        self.pieces = pieces
        self.buckets = buckets
    }

    func bytes(_ id: Int) -> [UInt8] { pieces[id] ?? [] }

    func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }

    /// Process caller text separately so literal role and media markers cannot
    /// be interpreted as checkpoint control tokens.
    func encodeLiteral(_ text: String) throws -> [Int] {
        try encode(text).flatMap { id in
            if pieces[id] != nil { return [id] }
            guard let token = tokenizer.convertIdToToken(id),
                  let first = token.unicodeScalars.first else {
                throw ClairicError.invalid("Cannot encode a literal Clairic input token.")
            }
            return encode(String(first)) + encode(String(token.unicodeScalars.dropFirst()))
        }
    }

    private static func byteMapping() -> [UInt32: UInt8] {
        let direct = Set(Array(33...126) + Array(161...172) + Array(174...255))
        var missing: UInt32 = 0
        return Dictionary(uniqueKeysWithValues: (0..<256).map { byte -> (UInt32, UInt8) in
            if direct.contains(byte) { return (UInt32(byte), UInt8(byte)) }
            defer { missing += 1 }
            return (256 + missing, UInt8(byte))
        })
    }
}
#endif
