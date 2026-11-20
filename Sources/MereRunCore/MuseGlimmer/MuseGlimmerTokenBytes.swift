import Foundation

/// The released tokenizer uses GPT-style ByteLevel BPE. Added protocol tokens
/// retain their literal spelling. Fuse is also supported for synthetic fixtures.
struct MuseGlimmerTokenBytes: Sendable {
    struct Metadata: Decodable {
        struct Decoder: Decodable { let type: String }
        struct Added: Decodable { let id: Int }
        let decoder: Decoder?
        let added_tokens: [Added]?
    }
    let byteLevel: Bool
    let added: Set<Int>
    init?(data: Data) throws {
        let metadata = try JSONDecoder().decode(Metadata.self, from: data)
        guard let type = metadata.decoder?.type, ["ByteLevel", "Fuse"].contains(type) else { return nil }
        byteLevel = type == "ByteLevel"
        added = Set((metadata.added_tokens ?? []).map(\.id))
    }
    func bytes(id: Int, spelling: String?) -> [UInt8] {
        guard let spelling else { return [] }
        if !byteLevel || added.contains(id) { return Array(spelling.utf8) }
        var result: [UInt8] = []
        for scalar in spelling.unicodeScalars {
            guard let byte = Self.inverse[scalar] else { return [] }
            result.append(byte)
        }
        return result
    }
    private static let inverse: [Unicode.Scalar: UInt8] = {
        let printable = Array(33...126) + Array(161...172) + Array(174...255)
        let retained = Set(printable)
        var result: [Unicode.Scalar: UInt8] = [:]
        var extra = 256
        for byte in 0...255 {
            let scalar = retained.contains(byte) ? byte : extra
            if !retained.contains(byte) { extra += 1 }
            result[Unicode.Scalar(scalar)!] = UInt8(byte)
        }
        return result
    }()
}
