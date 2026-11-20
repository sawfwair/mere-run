import Foundation

/// Strict incremental UTF-8. An unfinished scalar is retained as bytes; invalid
/// bytes are never repaired to U+FFFD or silently discarded by the matcher.
enum ChatUTF8Prefix {
    static func split(_ bytes: [UInt8]) -> (text: String, pending: [UInt8])? {
        var index = 0
        while index < bytes.count {
            let lead = bytes[index]
            let width: Int
            switch lead {
            case 0...0x7f: width = 1
            case 0xc2...0xdf: width = 2
            case 0xe0...0xef: width = 3
            case 0xf0...0xf4: width = 4
            default: return nil
            }
            let available = min(width, bytes.count - index)
            for offset in 1..<available {
                let byte = bytes[index + offset]
                guard (0x80...0xbf).contains(byte) else { return nil }
                if offset == 1 {
                    if lead == 0xe0 && byte < 0xa0 { return nil }
                    if lead == 0xed && byte > 0x9f { return nil }
                    if lead == 0xf0 && byte < 0x90 { return nil }
                    if lead == 0xf4 && byte > 0x8f { return nil }
                }
            }
            if available < width {
                guard let text = String(bytes: bytes[..<index], encoding: .utf8) else { return nil }
                return (text, Array(bytes[index...]))
            }
            index += width
        }
        guard let text = String(bytes: bytes, encoding: .utf8) else { return nil }
        return (text, [])
    }

    static func scalarRange(_ prefix: [UInt8]) -> ClosedRange<UInt32>? {
        guard let first = prefix.first, let decoded = Self.split(prefix), decoded.text.isEmpty, !decoded.pending.isEmpty else { return nil }
        let width = first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4
        var value = UInt32(first & (width == 2 ? 0x1f : width == 3 ? 0x0f : 0x07))
        for byte in prefix.dropFirst() { value = (value << 6) | UInt32(byte & 0x3f) }
        let bits = 6 * (width - prefix.count)
        let lower = max(value << bits, width == 2 ? 0x80 : width == 3 ? 0x800 : 0x10000)
        let upper = min((value << bits) + (UInt32(1) << bits) - 1, 0x10ffff)
        return lower <= upper ? lower...upper : nil
    }
}

extension ChatSchemaGrammar {
    func acceptsUTF8Prefix(_ prefix: [UInt8]) -> Bool {
        guard let range = ChatUTF8Prefix.scalarRange(prefix) else { return false }
        return nextTerminals.contains { symbol in
            switch symbol {
            case .character(let scalar): return Array(String(scalar).utf8).starts(with: prefix)
            case .stringCharacter, .xmlCharacter: return true // Valid non-ASCII scalars.
            case .except(let excluded):
                var value = range.lowerBound
                for _ in 0...excluded.count {
                    if (0xd800...0xdfff).contains(value) { value = 0xe000 }
                    guard value <= range.upperBound else { return false }
                    if let scalar = Unicode.Scalar(value), !excluded.contains(scalar) { return true }
                    value += 1
                }
                return false
            case .rule, .hex, .digit, .nonzero: return false
            }
        }
    }
}
