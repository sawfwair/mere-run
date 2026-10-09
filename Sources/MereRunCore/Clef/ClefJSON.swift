#if !os(iOS)
import Foundation
import MereRunQwenModel

/// Ordered, typed JSON. JSONDecoder's keyed containers do not retain schema field order.
public indirect enum ClefJSON: Sendable, Encodable {
    case string(String), number(String), bool(Bool), null
    case array([ClefJSON]), object([Field])

    public struct Field: Sendable {
        public let key: String
        public let value: ClefJSON
    }

    public var string: String? { if case .string(let value) = self { value } else { nil } }
    public var array: [ClefJSON]? { if case .array(let value) = self { value } else { nil } }
    public var fields: [Field]? { if case .object(let value) = self { value } else { nil } }
    public subscript(_ key: String) -> ClefJSON? { fields?.first { $0.key == key }?.value }

    var isTruthy: Bool {
        switch self {
        case .null: false
        case .bool(let value): value
        case .string(let value): !value.isEmpty
        case .number(let value): Double(value) != 0
        case .array(let value): !value.isEmpty
        case .object(let value): !value.isEmpty
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .number(let value):
            if let integer = Int64(value) { try container.encode(integer) }
            else { try container.encode(JSONDecoder().decode(Double.self, from: Data(value.utf8))) }
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        case .array(let value): try container.encode(value)
        case .object(let fields): try container.encode(Dictionary(uniqueKeysWithValues: fields.map { ($0.key, $0.value) }))
        }
    }

    /// Python json.dumps(..., ensure_ascii=False, separators=(",", ":"), sort_keys=True).
    func canonical() throws -> String {
        func quote(_ text: String) throws -> String {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.withoutEscapingSlashes]
            return String(decoding: try encoder.encode(text), as: UTF8.self)
        }
        switch self {
        case .string(let value): return try quote(value)
        case .number(let value):
            if !value.contains(where: { ".eE".contains($0) }) { return Int64(value).map(String.init) ?? value }
            let number = try JSONDecoder().decode(Double.self, from: Data(value.utf8))
            guard number.isFinite else { throw ClefError.invalidInput("Clef JSON numbers must be finite.") }
            return String(number)
        case .bool(let value): return value ? "true" : "false"
        case .null: return "null"
        case .array(let values): return "[" + (try values.map { try $0.canonical() }).joined(separator: ",") + "]"
        case .object(let fields):
            return "{" + (try fields.sorted { Self.keyOrder($0.key, $1.key) }.map {
                try quote($0.key) + ":" + $0.value.canonical()
            }).joined(separator: ",") + "}"
        }
    }

    func rendered() throws -> String { try string ?? canonical() }
    static func keyOrder(_ lhs: String, _ rhs: String) -> Bool { lhs.utf8.lexicographicallyPrecedes(rhs.utf8) }

    public static func parse(_ data: Data) throws -> ClefJSON {
        var parser = Parser(bytes: Array(data))
        let value = try parser.value(depth: 0)
        parser.whitespace()
        guard parser.index == parser.bytes.count else { throw ClefError.invalidInput("Trailing data in Clef JSON request.") }
        return value
    }

    private struct Parser {
        let bytes: [UInt8]
        var index = 0
        mutating func whitespace() { while index < bytes.count, [9, 10, 13, 32].contains(bytes[index]) { index += 1 } }
        mutating func consume(_ byte: UInt8) -> Bool {
            whitespace()
            guard index < bytes.count, bytes[index] == byte else { return false }
            index += 1
            return true
        }
        mutating func text() throws -> String {
            whitespace()
            let start = index
            guard consume(34) else { throw ClefError.invalidInput("Expected JSON string in Clef request.") }
            while index < bytes.count {
                let byte = bytes[index]
                index += 1
                if byte == 34 { return try JSONDecoder().decode(String.self, from: Data(bytes[start..<index])) }
                if byte == 92 { index += 1 }
            }
            throw ClefError.invalidInput("Unterminated JSON string in Clef request.")
        }
        mutating func value(depth: Int) throws -> ClefJSON {
            whitespace()
            guard depth <= 64, index < bytes.count else { throw ClefError.invalidInput("Invalid or too deeply nested Clef JSON.") }
            if bytes[index] == 34 { return .string(try text()) }
            if consume(123) {
                var fields: [Field] = []
                var keys = Set<String>()
                if consume(125) { return .object(fields) }
                repeat {
                    let key = try text()
                    guard keys.insert(key).inserted, consume(58) else {
                        throw ClefError.invalidInput("Duplicate key or missing colon in Clef JSON.")
                    }
                    fields.append(Field(key: key, value: try value(depth: depth + 1)))
                    if consume(125) { return .object(fields) }
                } while consume(44)
                throw ClefError.invalidInput("Invalid Clef JSON object.")
            }
            if consume(91) {
                var values: [ClefJSON] = []
                if consume(93) { return .array(values) }
                repeat {
                    values.append(try value(depth: depth + 1))
                    if consume(93) { return .array(values) }
                } while consume(44)
                throw ClefError.invalidInput("Invalid Clef JSON array.")
            }
            let start = index
            while index < bytes.count, ![9, 10, 13, 32, 44, 93, 125].contains(bytes[index]) { index += 1 }
            let token = String(decoding: bytes[start..<index], as: UTF8.self)
            switch token {
            case "true": return .bool(true)
            case "false": return .bool(false)
            case "null": return .null
            default:
                let number = try JSONDecoder().decode(Double.self, from: Data(token.utf8))
                guard number.isFinite else { throw ClefError.invalidInput("Clef JSON numbers must be finite.") }
                return .number(token)
            }
        }
    }
}
#endif
