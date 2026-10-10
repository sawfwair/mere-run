import Foundation
import MLX

extension WhistleWeights {
    private struct Mapping: Decodable {
        struct Piece: Decodable { let record: Int; let rows: [Int]? }
        let pieces: [Piece]
        let transpose: Bool
    }

    /// The released archive has a nameless directory. The resource binds its pinned record order
    /// to checkpoint names; every record shape, length, dtype and byte range is checked first.
    package static func load(cact data: Data) throws -> WhistleWeights {
        _ = try WhistleVocabulary(cact: data)
        let reader = WhistleBinaryReader(data: data)
        let url = Bundle.module.url(forResource: "cact-layout", withExtension: "json")!
        let mappings = try JSONDecoder().decode([String: Mapping].self, from: Data(contentsOf: url))
        let layoutURL = Bundle.module.url(forResource: "layout", withExtension: "json")!
        let layout = try JSONDecoder().decode([String: [Int]].self, from: Data(contentsOf: layoutURL))
        var raw: [Int: MLXArray] = [:]
        var quantized: [Int: WhistleCactusQuant] = [:]
        for record in 0..<679 {
            let entry = 308 + record * 44
            let dtype = try reader.byte(at: entry)
            let ndim = Int(try reader.byte(at: entry + 1))
            guard (0...4).contains(ndim) else { throw WhistleError.invalid("invalid tensor rank") }
            let shape = try (0..<ndim).map { Int(try reader.u32(at: entry + 4 + $0 * 4)) }
            guard shape.allSatisfy({ $0 > 0 && $0 <= 100000 }) else { throw WhistleError.invalid("invalid tensor shape") }
            let offset = try reader.offset(at: entry + 20)
            let size = try reader.offset(at: entry + 28)
            guard offset >= 30272, offset <= data.count, size <= data.count - offset else {
                throw WhistleError.invalid("tensor byte range exceeds archive")
            }
            var count = 1
            for dimension in shape {
                guard count <= 20_000_000 / dimension else { throw WhistleError.invalid("tensor exceeds released size limits") }
                count *= dimension
            }
            if dtype == 3 {
                let bits = Int(try reader.u32(at: entry + 40))
                guard shape.count == 2, try reader.u32(at: entry + 36) == 128,
                      bits == 2 || bits == 4, shape[1].isMultiple(of: 128),
                      size == count * bits / 8 + count / 128 * 2 else {
                    throw WhistleError.invalid("unsupported CQ record")
                }
                let packedCount = count * bits / 8
                let bytes = Array(data[offset..<(offset + packedCount)])
                let norms = try (0..<(count / 128)).map { Float(Float16(bitPattern: try reader.u16(at: offset + packedCount + $0 * 2))) }
                guard norms.allSatisfy({ $0.isFinite && $0 >= 0 }) else { throw WhistleError.invalid("invalid CQ norms") }
                let bookStart = 196 + (bits == 2 ? 0 : 12) * 4
                let book = try (0..<(1 << bits)).map { Float(bitPattern: try reader.u32(at: bookStart + $0 * 4)) }
                guard book.allSatisfy(\.isFinite) else { throw WhistleError.invalid("invalid CQ codebook") }
                quantized[record] = WhistleCactusQuant(packed: MLXArray(bytes), norms: MLXArray(norms),
                                                     codebook: MLXArray(book), columns: shape[1], rows: shape[0], bits: bits)
            } else {
                guard (dtype == 1 || dtype == 2), size == count * (dtype == 1 ? 2 : 4) else {
                    throw WhistleError.invalid("invalid floating point record")
                }
                let values = try (0..<count).map { index in
                    if dtype == 1 { return Float(Float16(bitPattern: try reader.u16(at: offset + index * 2))) }
                    return Float(bitPattern: try reader.u32(at: offset + index * 4))
                }
                guard values.allSatisfy(\.isFinite) else { throw WhistleError.invalid("non-finite tensor") }
                raw[record] = MLXArray(values, shape)
            }
        }
        var arrays: [String: MLXArray] = [:]
        var packed: [String: [WhistleCactusQuant]] = [:]
        for (name, mapping) in mappings {
            let shape = layout[name]!
            if let first = quantized[mapping.pieces[0].record] {
                if mapping.pieces.count == 1, shape.count == 3, name.contains("/mhc_phi_") {
                    let perLayer = shape[2]
                    guard first.rows == 8 * perLayer, first.columns == shape[1] else { throw WhistleError.invalid("invalid lane map") }
                    packed[name] = (0..<8).map { first.slice(($0 * perLayer)..<(($0 + 1) * perLayer)) }
                } else {
                    let matrices = try mapping.pieces.map { piece -> WhistleCactusQuant in
                        guard let matrix = quantized[piece.record], piece.rows == [0, matrix.rows] else {
                            throw WhistleError.invalid("CQ mapping disagrees with tensor directory")
                        }
                        return matrix
                    }
                    let matrixShape = Array(shape.suffix(2))
                    let expected = mapping.transpose ? matrixShape.reversed().map { $0 } : matrixShape
                    // Engram tables flatten their first two dimensions into rows.
                    let actual = name.hasSuffix("/embedding") && name.hasPrefix("engrams_")
                        ? [shape[0] * shape[1], shape[2]] : expected
                    guard matrices.allSatisfy({ [$0.rows, $0.columns] == actual }) else { throw WhistleError.invalid("invalid CQ shape for \(name)") }
                    packed[name] = matrices
                }
            } else {
                let pieces = try mapping.pieces.map { piece -> MLXArray in
                    guard let value = raw[piece.record] else { throw WhistleError.invalid("missing raw record") }
                    return value
                }
                let value = pieces.count == 1 ? pieces[0] : stacked(pieces)
                let canonical = mapping.transpose ? value.swappedAxes(-1, -2) : value
                guard canonical.size == shape.reduce(1, *) else { throw WhistleError.invalid("invalid raw shape for \(name)") }
                arrays[name] = canonical.reshaped(shape)
            }
        }
        return try WhistleWeights(arrays: arrays, packed: packed)
    }
}
