import Foundation

/// Reads only the filterbank and SentencePiece dump; no upstream engine is loaded.
package struct WhistleVocabulary {
    package static let languages = ["en", "de", "fr", "es", "it", "nl", "pl"]
    package let filterbank: [Float]
    let pieces: [String]
    let types: [UInt8]
    let scores: [Float]
    let pieceIDs: [String: Int]
    let byteIDs: [UInt8: Int]
    let markers: [String]

    package init(cact: Data) throws {
        var reader = WhistleBinaryReader(data: cact)
        guard try reader.u32(at: 0) == 0x05E12A84,
              try reader.u32(at: 4) == 681,
              try reader.u32(at: 8) == 28,
              try reader.u32(at: 20) == 8199,
              try reader.u32(at: 28) == 512,
              try reader.u32(at: 40) == 8 else {
            throw WhistleError.invalid("unsupported .cact geometry")
        }
        let geometry: [UInt32] = [0x05E12A84, 681, 28, 0, 8, 8199, 0, 512, 8, 2, 8, 48, 64, 320, 512, 4,
                                  0, 0, 0, 3, 18432, 128, 4, 4, 3, 0, 2, 2, 3, 0, 0, 2, 3, 7] + Array(repeating: 0, count: 14)
        for (index, expected) in geometry.enumerated() {
            guard try reader.u32(at: index * 4) == expected else { throw WhistleError.invalid("unsupported .cact geometry") }
        }
        guard Float(bitPattern: try reader.u32(at: 192)) == 100000 else { throw WhistleError.invalid("unsupported rotary base") }
        let directory = 196 + 28 * 4
        let filter = directory + 679 * 44
        guard try reader.byte(at: filter) == 2, try reader.byte(at: filter + 1) == 2,
              try reader.u32(at: filter + 4) == 257, try reader.u32(at: filter + 8) == 80,
              try reader.u64(at: filter + 28) == 257 * 80 * 4 else {
            throw WhistleError.invalid("missing released audio filterbank")
        }
        let filterOffset = try reader.offset(at: filter + 20)
        guard filterOffset <= cact.count, 257 * 80 * 4 <= cact.count - filterOffset else {
            throw WhistleError.invalid("truncated audio filterbank")
        }
        filterbank = try (0..<(257 * 80)).map { Float(bitPattern: try reader.u32(at: filterOffset + $0 * 4)) }
        guard filterbank.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw WhistleError.invalid("invalid audio filterbank values")
        }
        let tokenizer = directory + 680 * 44
        guard try reader.byte(at: tokenizer) == 4 else { throw WhistleError.invalid("missing tokenizer") }
        let start = try reader.offset(at: tokenizer + 20)
        let size = try reader.offset(at: tokenizer + 28)
        guard start <= cact.count, size <= cact.count - start, size >= 24 else {
            throw WhistleError.invalid("truncated tokenizer attachment")
        }
        // Constrain the reader to the attachment, so a corrupt piece cannot read another tensor.
        reader = WhistleBinaryReader(data: cact.subdata(in: start..<(start + size)))
        guard try reader.u32(at: 0) == 8199, try reader.u32(at: 4) == 0,
              try reader.u32(at: 8) == 1, try reader.u32(at: 12) == 2,
              try reader.byte(at: 20) == 0, try reader.byte(at: 21) == 1 else {
            throw WhistleError.invalid("unsupported tokenizer header")
        }
        var cursor = 24
        var pieces: [String] = []
        var types: [UInt8] = []
        var scores: [Float] = []
        for _ in 0..<8199 {
            let score = Float(bitPattern: try reader.u32(at: cursor))
            guard score.isFinite else { throw WhistleError.invalid("invalid tokenizer score") }
            scores.append(score)
            let type = try reader.byte(at: cursor + 4)
            let length = Int(try reader.u16(at: cursor + 5))
            cursor += 7
            guard type <= 4, cursor <= reader.data.count, length <= reader.data.count - cursor,
                  let piece = String(data: reader.data.subdata(in: cursor..<(cursor + length)), encoding: .utf8) else {
                throw WhistleError.invalid("invalid tokenizer piece")
            }
            if type == 4 {
                guard piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">"),
                      UInt8(piece.dropFirst(3).prefix(2), radix: 16) != nil else {
                    throw WhistleError.invalid("invalid tokenizer byte piece")
                }
            }
            pieces.append(piece)
            types.append(type)
            cursor += length
        }
        guard cursor == reader.data.count else { throw WhistleError.invalid("unexpected tokenizer trailing data") }
        guard Set(pieces).count == pieces.count else { throw WhistleError.invalid("duplicate tokenizer piece") }
        self.pieces = pieces
        self.types = types
        self.scores = scores
        pieceIDs = Dictionary(uniqueKeysWithValues: pieces.enumerated().map { ($0.element, $0.offset) })
        let byteEntries = pieces.indices.filter { types[$0] == 4 }.map {
            (UInt8(pieces[$0].dropFirst(3).prefix(2), radix: 16)!, $0)
        }
        guard byteEntries.count == 256, Set(byteEntries.map { $0.0 }).count == 256 else {
            throw WhistleError.invalid("invalid byte fallback vocabulary")
        }
        byteIDs = Dictionary(uniqueKeysWithValues: byteEntries)
        markers = pieces.indices.filter { types[$0] == 3 }.map { pieces[$0] }.sorted { $0.utf8.count > $1.utf8.count }
    }

    package func startsWord(_ token: Int) -> Bool { pieces[token].hasPrefix("▁") }

    /// SentencePiece BPE with the archive's scores, user-defined symbols and UTF-8 fallback.
    package func encode(_ text: String) -> [Int] {
        let escaped = text.replacingOccurrences(of: " ", with: "▁")
        var remaining = escaped[...]
        var buffer = ""
        var result: [Int] = []
        while !remaining.isEmpty {
            if let marker = markers.first(where: { remaining.hasPrefix($0) }) {
                result += bpe(buffer)
                buffer = ""
                result.append(pieceIDs[marker]!)
                remaining = remaining.dropFirst(marker.count)
            } else {
                let scalar = remaining.unicodeScalars.first!
                buffer.unicodeScalars.append(scalar)
                remaining = remaining[remaining.unicodeScalars.index(after: remaining.unicodeScalars.startIndex)...]
            }
        }
        return result + bpe(buffer)
    }

    private func bpe(_ text: String) -> [Int] {
        var symbols = text.unicodeScalars.map(String.init)
        while symbols.count > 1 {
            var best: (index: Int, score: Float)?
            for index in 0..<(symbols.count - 1) {
                if let id = pieceIDs[symbols[index] + symbols[index + 1]], best == nil || scores[id] > best!.score {
                    best = (index, scores[id])
                }
            }
            guard let best else { break }
            symbols[best.index] += symbols.remove(at: best.index + 1)
        }
        return symbols.flatMap { symbol -> [Int] in
            if let id = pieceIDs[symbol] { return [id] }
            return symbol.utf8.map { byteIDs[$0]! }
        }
    }

    package func decode(_ tokens: [Int]) throws -> String {
        var bytes: [UInt8] = []
        for token in tokens {
            guard pieces.indices.contains(token) else { throw WhistleError.invalid("token outside vocabulary") }
            if types[token] == 4 {
                bytes.append(UInt8(pieces[token].dropFirst(3).prefix(2), radix: 16)!)
            } else if types[token] == 0 || types[token] == 3 {
                bytes.append(contentsOf: pieces[token].utf8)
            }
        }
        return String(decoding: bytes, as: UTF8.self).replacingOccurrences(of: "▁", with: " ")
    }
}

struct WhistleBinaryReader {
    let data: Data

    func byte(at offset: Int) throws -> UInt8 {
        guard offset >= 0, offset < data.count else { throw WhistleError.invalid("truncated .cact attachment") }
        return data[offset]
    }

    func u16(at offset: Int) throws -> UInt16 {
        UInt16(try byte(at: offset)) | (UInt16(try byte(at: offset + 1)) << 8)
    }

    func u32(at offset: Int) throws -> UInt32 {
        UInt32(try u16(at: offset)) | (UInt32(try u16(at: offset + 2)) << 16)
    }

    func u64(at offset: Int) throws -> UInt64 {
        UInt64(try u32(at: offset)) | (UInt64(try u32(at: offset + 4)) << 32)
    }

    func offset(at offset: Int) throws -> Int {
        let value = try u64(at: offset)
        guard value <= UInt64(Int.max) else { throw WhistleError.invalid("attachment offset overflow") }
        return Int(value)
    }
}
