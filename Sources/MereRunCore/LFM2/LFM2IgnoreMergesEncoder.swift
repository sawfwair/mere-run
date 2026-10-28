import Foundation
import Tokenizers

/// Honors BPE's whole-piece vocabulary lookup before falling back to merges.
/// The pinned Swift tokenizer supports merge-based BPE but omits ignore_merges.
final class LFM2IgnoreMergesEncoder: @unchecked Sendable {
    private struct Envelope: Decodable {
        struct Model: Decodable {
            let ignoreMerges: Bool?
            enum CodingKeys: String, CodingKey { case ignoreMerges = "ignore_merges" }
        }
        let model: Model
    }

    private struct Contract: Decodable {
        struct Model: Decodable {
            let type: String
            let vocab: [String: Int]
        }
        struct AddedToken: Decodable {
            let content: String
            let id: Int
            let lstrip: Bool
            let rstrip: Bool
            let singleWord: Bool
            enum CodingKeys: String, CodingKey {
                case content, id, lstrip, rstrip
                case singleWord = "single_word"
            }
        }
        struct Processor: Decodable { let type: String }
        struct PreTokenizer: Decodable {
            struct Component: Decodable {
                struct Pattern: Decodable {
                    let regex: String
                    enum CodingKeys: String, CodingKey { case regex = "Regex" }
                }
                let type: String
                let pattern: Pattern?
                let behavior: String?
                let invert: Bool?
                let addPrefixSpace: Bool?
                let useRegex: Bool?
                enum CodingKeys: String, CodingKey {
                    case type, pattern, behavior, invert
                    case addPrefixSpace = "add_prefix_space"
                    case useRegex = "use_regex"
                }
            }
            let type: String
            let pretokenizers: [Component]
        }
        let model: Model
        let addedTokens: [AddedToken]
        let normalizer: Processor?
        let preTokenizer: PreTokenizer
        let postProcessor: Processor?
        enum CodingKeys: String, CodingKey {
            case model, normalizer
            case addedTokens = "added_tokens"
            case preTokenizer = "pre_tokenizer"
            case postProcessor = "post_processor"
        }
    }

    private let vocab: [String: Int]
    private let addedTokens: [String: Int]
    private let addedPattern: NSRegularExpression
    private let splitPattern: NSRegularExpression
    private let fallback: any Tokenizer

    static func load(from root: URL, fallback: any Tokenizer) throws -> LFM2IgnoreMergesEncoder? {
        let data = try Data(contentsOf: root.appending(path: "tokenizer.json"))
        guard try JSONDecoder().decode(Envelope.self, from: data).model.ignoreMerges == true else { return nil }
        let contract = try JSONDecoder().decode(Contract.self, from: data)
        let components = contract.preTokenizer.pretokenizers
        guard contract.model.type == "BPE", contract.normalizer == nil,
              contract.preTokenizer.type == "Sequence", components.count == 2,
              components[0].type == "Split", components[0].behavior == "Isolated",
              components[0].invert == false, let pattern = components[0].pattern?.regex,
              components[1].type == "ByteLevel", components[1].addPrefixSpace == false,
              components[1].useRegex == false,
              contract.postProcessor == nil || contract.postProcessor?.type == "ByteLevel",
              contract.addedTokens.allSatisfy({ !$0.lstrip && !$0.rstrip && !$0.singleWord && !$0.content.isEmpty }) else {
            throw LFM2Error.generationFailed("LFM2 ignore_merges requires the released byte-level tokenizer contract.")
        }
        return try LFM2IgnoreMergesEncoder(
            vocab: contract.model.vocab, pattern: pattern,
            addedTokens: Dictionary(uniqueKeysWithValues: contract.addedTokens.map { ($0.content, $0.id) }),
            fallback: fallback
        )
    }

    init(vocab: [String: Int], pattern: String, addedTokens: [String: Int], fallback: any Tokenizer) throws {
        self.vocab = vocab
        self.addedTokens = addedTokens
        self.fallback = fallback
        self.splitPattern = try NSRegularExpression(pattern: pattern)
        let alternatives = addedTokens.keys.sorted {
            $0.utf8.count == $1.utf8.count ? $0 < $1 : $0.utf8.count > $1.utf8.count
        }.map(NSRegularExpression.escapedPattern(for:))
        self.addedPattern = try NSRegularExpression(pattern: alternatives.isEmpty ? "(?!)" : alternatives.joined(separator: "|"))
    }

    func encode(_ text: String) -> [Int] {
        let source = text as NSString
        var result: [Int] = []
        var offset = 0
        for match in addedPattern.matches(in: text, range: NSRange(location: 0, length: source.length)) {
            if match.range.location > offset {
                encodePlain(source.substring(with: NSRange(location: offset, length: match.range.location - offset)), into: &result)
            }
            let token = source.substring(with: match.range)
            if let id = addedTokens[token] { result.append(id) }
            offset = NSMaxRange(match.range)
        }
        if offset < source.length {
            encodePlain(source.substring(from: offset), into: &result)
        }
        return result
    }

    private func encodePlain(_ text: String, into result: inout [Int]) {
        let source = text as NSString
        for match in splitPattern.matches(in: text, range: NSRange(location: 0, length: source.length)) {
            let piece = source.substring(with: match.range)
            var bytes = ""
            for byte in piece.utf8 { bytes.unicodeScalars.append(Self.byteAlphabet[Int(byte)]) }
            if let id = vocab[bytes] {
                result.append(id)
            } else {
                result.append(contentsOf: fallback.encode(text: piece, addSpecialTokens: false))
            }
        }
    }

    private static let byteAlphabet: [Unicode.Scalar] = {
        let visible = Set(Array(33...126) + Array(161...172) + Array(174...255))
        var extra = 0
        return (0...255).map { byte in
            if visible.contains(byte) { return Unicode.Scalar(UInt8(byte)) }
            defer { extra += 1 }
            return Unicode.Scalar(256 + extra)!
        }
    }()
}
