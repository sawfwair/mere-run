import Foundation

/// Parses the function syntax emitted by the LFM2.5 chat template:
/// `<|tool_call_start|>[name(key='value'), other()]<|tool_call_end|>`.
public enum LFM2ToolParser {
    public static func renderToolCalls(_ calls: [ChatMessageToolCall]) throws -> String {
        let rendered = try calls.map { call in
            let arguments = try call.arguments.sorted { $0.key < $1.key }.map { key, value in
                return "\(key)=\(try renderArgument(value))"
            }.joined(separator: ", ")
            return "\(call.name)(\(arguments))"
        }.joined(separator: ", ")
        return "<|tool_call_start|>[\(rendered)]<|tool_call_end|>"
    }

    private static func renderArgument(_ value: OpenAIJSONValue) throws -> String {
        if case .string(let string) = value {
            let escaped = string
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "'", with: "\\'")
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\r", with: "\\r")
            return "'\(escaped)'"
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }

    public static func parseToolCalls(_ text: String) -> [ToolCall] {
        guard let start = text.range(of: "<|tool_call_start|>") else { return [] }
        var parser = Parser(characters: Array(text[start.upperBound...]))
        return parser.parse() ?? []
    }

    private struct Parser {
        let characters: [Character]
        var position = 0

        mutating func parse() -> [ToolCall]? {
            skipWhitespace()
            guard take("[") else { return nil }
            var calls: [ToolCall] = []
            repeat {
                skipWhitespace()
                guard let name = identifier(), take("(") else { return nil }
                var arguments: [String: String] = [:]
                skipWhitespace()
                if !take(")") {
                    repeat {
                        skipWhitespace()
                        guard let key = identifier(), take("=") else { return nil }
                        skipWhitespace()
                        guard let value = value() else { return nil }
                        arguments[key] = value
                        skipWhitespace()
                        if take(")") { break }
                    } while take(",")
                    guard characters[position - 1] == ")" else { return nil }
                }
                calls.append(ToolCall(name: name, arguments: arguments))
                skipWhitespace()
                if take("]") { return calls }
            } while take(",")
            return nil
        }

        mutating func value() -> String? {
            guard position < characters.count else { return nil }
            if characters[position] == "'" || characters[position] == "\"" {
                let quote = characters[position]
                position += 1
                var result = ""
                while position < characters.count {
                    let character = characters[position]
                    position += 1
                    if character == quote { return result }
                    if character == "\\" {
                        guard position < characters.count else { return nil }
                        let escaped = characters[position]
                        position += 1
                        switch escaped {
                        case "n": result.append("\n")
                        case "r": result.append("\r")
                        case "t": result.append("\t")
                        default: result.append(escaped)
                        }
                    } else {
                        result.append(character)
                    }
                }
                return nil
            }
            let start = position
            var depth = 0
            while position < characters.count {
                let character = characters[position]
                if depth == 0 && (character == "," || character == ")") { break }
                if character == "[" || character == "{" { depth += 1 }
                if character == "]" || character == "}" { depth -= 1 }
                if depth < 0 { return nil }
                position += 1
            }
            guard depth == 0 else { return nil }
            let result = String(characters[start..<position]).trimmingCharacters(in: .whitespacesAndNewlines)
            return result.isEmpty ? nil : result
        }

        mutating func identifier() -> String? {
            skipWhitespace()
            let start = position
            guard position < characters.count,
                  characters[position].isLetter || characters[position] == "_" else { return nil }
            position += 1
            while position < characters.count,
                  characters[position].isLetter || characters[position].isNumber || characters[position] == "_" {
                position += 1
            }
            return String(characters[start..<position])
        }

        mutating func skipWhitespace() {
            while position < characters.count && characters[position].isWhitespace {
                position += 1
            }
        }

        mutating func take(_ character: Character) -> Bool {
            skipWhitespace()
            guard position < characters.count, characters[position] == character else { return false }
            position += 1
            return true
        }
    }
}
