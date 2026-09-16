import Foundation

/// Checkpoint-native Qwen byte BPE. Prompt markers are inserted as IDs, never parsed from user text.
struct YuE2Tokenizer {
    private let ranks: [Data: Int]
    private let vocabulary: [Data]
    private let pattern: NSRegularExpression

    init(url: URL) throws {
        try self.init(contents: String(contentsOf: url, encoding: .utf8))
    }

    init(contents: String, vocabularySize: Int = YuE2Protocol.endOfText) throws {
        var ranks: [Data: Int] = [:]
        var vocabulary = [Data](repeating: Data(), count: vocabularySize)
        for line in contents.split(whereSeparator: \.isNewline) {
            let fields = line.split(whereSeparator: \.isWhitespace)
            guard fields.count == 2, let bytes = Data(base64Encoded: String(fields[0])), !bytes.isEmpty,
                  let rank = Int(fields[1]), (0..<vocabularySize).contains(rank),
                  vocabulary[rank].isEmpty, ranks[bytes] == nil else {
                throw YuE2Error.invalidConfiguration("Malformed or duplicate qwen.tiktoken entry.")
            }
            vocabulary[rank] = bytes
            ranks[bytes] = rank
        }
        guard ranks.count == vocabularySize,
              (0...255).allSatisfy({ ranks[Data([UInt8($0)])] != nil }) else {
            throw YuE2Error.invalidConfiguration("qwen.tiktoken must cover every rank and every byte.")
        }
        self.ranks = ranks
        self.vocabulary = vocabulary
        pattern = try NSRegularExpression(pattern:
            #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        )
    }

    func encode(_ text: String) throws -> [Int] {
        let normalized = text.precomposedStringWithCanonicalMapping
        let string = normalized as NSString
        return try pattern.matches(in: normalized, range: NSRange(location: 0, length: string.length)).flatMap {
            try encodePiece(Data(string.substring(with: $0.range).utf8))
        }
    }

    func decode(_ ids: [Int]) throws -> String {
        var bytes = Data()
        for id in ids {
            guard vocabulary.indices.contains(id) else {
                throw YuE2Error.invalidRequest("ABC contains a non-text token: \(id).")
            }
            bytes.append(vocabulary[id])
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func encodePiece(_ piece: Data) throws -> [Int] {
        if let rank = ranks[piece] { return [rank] }
        var parts = piece.map { Data([$0]) }
        while parts.count > 1 {
            var bestRank = Int.max
            var bestIndex: Int?
            for index in 0..<(parts.count - 1) {
                if let rank = ranks[parts[index] + parts[index + 1]], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            guard let index = bestIndex else { break }
            parts[index].append(parts[index + 1])
            parts.remove(at: index + 1)
        }
        return try parts.map {
            guard let rank = ranks[$0] else {
                throw YuE2Error.invalidConfiguration("BPE produced a token absent from qwen.tiktoken.")
            }
            return rank
        }
    }
}
