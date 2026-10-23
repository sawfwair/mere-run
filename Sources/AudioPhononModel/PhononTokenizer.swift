import Foundation

/// Checkpoint-native Unigram Viterbi tokenization with Metaspace and UTF-8 byte fallback.
/// The published tokenizer needs no external SentencePiece binary.
public struct PhononTokenizer: Sendable {
    private struct Document: Decodable {
        struct Model: Decodable {
            let type: String
            let unkId: Int
            let byteFallback: Bool
            let vocab: [Piece]
        }
        struct Piece: Decodable {
            let text: String
            let score: Float
            init(from decoder: any Decoder) throws {
                var values = try decoder.unkeyedContainer()
                text = try values.decode(String.self)
                score = try values.decode(Float.self)
            }
        }
        struct Normalizer: Decodable { let type, prepend: String }
        struct PreTokenizer: Decodable { let type, replacement, prependScheme: String; let split: Bool }
        let model: Model
        let normalizer: Normalizer
        let preTokenizer: PreTokenizer
    }
    private struct Piece: Sendable { let id: Int; let bytes: [UInt8]; let score: Float }
    private let byFirstByte: [UInt8: [Piece]]
    private let byteIDs: [UInt8: Int]
    private let vocabulary: [String]
    private let unknownScore: Float
    public let vocabularySize: Int

    public init(url: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let document = try decoder.decode(Document.self, from: Data(contentsOf: url))
        guard document.model.type == "Unigram", document.model.byteFallback, document.model.unkId == 0,
              document.normalizer.type == "Prepend", document.normalizer.prepend == "▁",
              document.preTokenizer.type == "Metaspace", document.preTokenizer.replacement == "▁",
              document.preTokenizer.prependScheme == "always", document.preTokenizer.split else {
            throw PhononError.invalidAsset("expected Phonon's Unigram/Metaspace tokenizer with byte fallback")
        }
        var grouped: [UInt8: [Piece]] = [:], byteIDs: [UInt8: Int] = [:]
        for (id, piece) in document.model.vocab.enumerated() {
            guard piece.score.isFinite, !piece.text.isEmpty else { throw PhononError.invalidAsset("invalid vocabulary piece") }
            if piece.text.hasPrefix("<0x"), piece.text.hasSuffix(">"), piece.text.count == 6,
               let byte = UInt8(piece.text.dropFirst(3).dropLast(), radix: 16) {
                byteIDs[byte] = id
            }
            let bytes = Array(piece.text.utf8)
            // The checkpoint makes special/control tokens unmatchable by prefixing NUL.
            if let first = bytes.first, first != 0, id != document.model.unkId {
                grouped[first, default: []].append(Piece(id: id, bytes: bytes, score: piece.score))
            }
        }
        guard byteIDs.count == 256 else { throw PhononError.invalidAsset("the tokenizer must contain all 256 byte fallback pieces") }
        vocabulary = document.model.vocab.map(\.text)
        byFirstByte = grouped
        self.byteIDs = byteIDs
        unknownScore = (document.model.vocab.map(\.score).min() ?? 0) - 10
        vocabularySize = document.model.vocab.count
    }

    public func encode(_ text: String) throws -> [Int] {
        guard !text.contains("\0") else { throw PhononError.invalidRequest("Phonon text must not contain NUL control tokens.") }
        // Prepend plus Metaspace's 'always' produces one leading metaspace, including
        // before the first word. Split at each subsequent space as HF Metaspace does.
        let pieces = text.split(whereSeparator: \.isWhitespace).map { "▁" + $0 }
        return try pieces.flatMap { try encodePiece(String($0)) }
    }

    private func encodePiece(_ text: String) throws -> [Int] {
        let bytes = Array(text.utf8), length = text.utf8.count
        var scores = [Float](repeating: -.infinity, count: length + 1)
        var previous = [Int](repeating: 0, count: length + 1)
        var chosen = [[Int]](repeating: [], count: length + 1)
        scores[0] = 0
        var position = 0
        for scalar in text.unicodeScalars {
            let scalarBytes = Array(String(scalar).utf8)
            let scalarEnd = position + scalarBytes.count
            var hasScalar = false
            for piece in byFirstByte[bytes[position]] ?? [] where position + piece.bytes.count <= length {
                let end = position + piece.bytes.count
                if bytes[position..<end].elementsEqual(piece.bytes) {
                    if end == scalarEnd { hasScalar = true }
                    let score = scores[position] + piece.score
                    if score > scores[end] {
                        scores[end] = score; previous[end] = position; chosen[end] = [piece.id]
                    }
                }
            }
            if !hasScalar {
                let score = scores[position] + unknownScore
                if score > scores[scalarEnd] {
                    let fallback = try scalarBytes.map { byte -> Int in
                        guard let id = byteIDs[byte] else { throw PhononError.invalidAsset("missing byte fallback") }
                        return id
                    }
                    scores[scalarEnd] = score; previous[scalarEnd] = position; chosen[scalarEnd] = fallback
                }
            }
            position = scalarEnd
        }
        var result: [[Int]] = [], cursor = length
        while cursor > 0 {
            guard !chosen[cursor].isEmpty, previous[cursor] < cursor else { throw PhononError.invalidAsset("tokenization failed") }
            result.append(chosen[cursor]); cursor = previous[cursor]
        }
        return result.reversed().flatMap { $0 }
    }

    public static func prepare(_ text: String) -> (text: String, tailFrames: Int) {
        let words = text.split(whereSeparator: \.isWhitespace)
        var result = words.joined(separator: " ")
        if let first = result.first { result = first.uppercased() + result.dropFirst() }
        if result.last?.isLetter == true || result.last?.isNumber == true { result += "." }
        return (result, words.count <= 4 ? 3 : 1)
    }
}

extension PhononTokenizer {
    func decode(_ ids: [Int]) throws -> String {
        var output = "", bytes: [UInt8] = [], first = true
        func flush() { output += String(decoding: bytes, as: UTF8.self); bytes.removeAll(keepingCapacity: true) }
        for id in ids {
            guard vocabulary.indices.contains(id) else { throw PhononError.invalidAsset("token id outside vocabulary") }
            var piece = vocabulary[id]
            if piece.first == "\0" { continue }
            if first, piece.hasPrefix("▁") { piece.removeFirst() }
            first = false
            piece = piece.replacingOccurrences(of: "▁", with: " ")
            if piece.hasPrefix("<0x"), piece.hasSuffix(">"), piece.count == 6,
               let byte = UInt8(piece.dropFirst(3).dropLast(), radix: 16) { bytes.append(byte) }
            else { flush(); output += piece }
        }
        flush()
        return output
    }
}
