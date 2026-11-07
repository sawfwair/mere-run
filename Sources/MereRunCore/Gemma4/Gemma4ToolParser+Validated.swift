import Foundation

extension Gemma4ToolParser {
    static func validatedToolCalls(_ text: String, tools: [ToolDefinition], parallel: Bool) throws -> [ToolCall] {
        let opening = "<|tool_call>"
        var cursor = text.startIndex
        var calls: [ToolCall] = []
        while let range = text.range(of: opening, range: cursor..<text.endIndex) {
            if !calls.isEmpty, !text[cursor..<range.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                throw ChatRequestIssue("tool_use", "unexpected content between tool calls")
            }
            var parser = GemmaToolValueParser(text: String(text[range.upperBound...]))
            let (name, arguments, consumed) = try parser.call()
            guard let tool = tools.first(where: { $0.name == name }) else {
                throw ChatRequestIssue("tool_use.name", "model generated an undeclared tool '\(name)'")
            }
            try ChatToolValidation.validate(.object(arguments), schema: tool.parametersJSONSchema)
            calls.append(ToolCall(name: name, jsonArguments: arguments))
            cursor = text.index(range.upperBound, offsetBy: consumed)
        }
        if !calls.isEmpty, !text[cursor...].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw ChatRequestIssue("tool_use", "unexpected or incomplete content after a tool call")
        }
        guard parallel || calls.count <= 1 else { throw ChatRequestIssue("tool_use", "model generated parallel calls when disabled") }
        return calls
    }
}

/// Gemma's quoted-string delimiter is distinct from JSON. Parse it recursively
/// instead of replacing delimiters inside arbitrary user strings.
private struct GemmaToolValueParser {
    let text: String
    private var position: String.Index
    private let stringDelimiter = "<|\"|>"

    init(text: String) { self.text = text; position = text.startIndex }

    mutating func call() throws -> (String, [String: ChatJSONValue], Int) {
        whitespace()
        guard consume("call:") else { throw malformed() }
        let start = position
        while position < text.endIndex, text[position] != "{" { position = text.index(after: position) }
        let name = text[start..<position].trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, case .object(let arguments) = try value(depth: 0) else { throw malformed() }
        whitespace()
        // The native closing delimiter is an EOS token and may be consumed by
        // the decoder. The complete argument object is still mandatory.
        _ = consume("<tool_call|>")
        return (name, arguments, text.distance(from: text.startIndex, to: position))
    }

    private mutating func value(depth: Int) throws -> ChatJSONValue {
        guard depth < 64 else { throw malformed() }
        whitespace()
        if consume(stringDelimiter) {
            guard let end = text.range(of: stringDelimiter, range: position..<text.endIndex) else { throw malformed() }
            let string = String(text[position..<end.lowerBound]); position = end.upperBound
            return .string(string)
        }
        guard position < text.endIndex else { throw malformed() }
        if text[position] == "\"" { return .string(try jsonString()) }
        if consume("{") {
            var object: [String: ChatJSONValue] = [:]
            whitespace()
            if consume("}") { return .object(object) }
            while true {
                whitespace()
                let key: String
                if position < text.endIndex, text[position] == "\"" { key = try jsonString() }
                else {
                    let start = position
                    while position < text.endIndex, ![":", ",", "{", "}", "[", "]"].contains(text[position]) {
                        position = text.index(after: position)
                    }
                    key = text[start..<position].trimmingCharacters(in: .whitespacesAndNewlines)
                }
                whitespace()
                guard !key.isEmpty, object[key] == nil, consume(":") else { throw malformed() }
                object[key] = try value(depth: depth + 1)
                whitespace()
                if consume("}") { return .object(object) }
                guard consume(",") else { throw malformed() }
            }
        }
        if consume("[") {
            var array: [ChatJSONValue] = []
            whitespace()
            if consume("]") { return .array(array) }
            while true {
                array.append(try value(depth: depth + 1)); whitespace()
                if consume("]") { return .array(array) }
                guard consume(",") else { throw malformed() }
            }
        }
        let start = position
        while position < text.endIndex, ![",", "}", "]"].contains(text[position]) { position = text.index(after: position) }
        let raw = text[start..<position].trimmingCharacters(in: .whitespacesAndNewlines)
        guard let scalar = try? JSONDecoder().decode(ChatJSONValue.self, from: Data(raw.utf8)),
              scalar.objectValue == nil, scalar.arrayValue == nil else { throw malformed() }
        return scalar
    }

    private mutating func jsonString() throws -> String {
        let start = position
        position = text.index(after: position)
        var escaped = false
        while position < text.endIndex {
            let character = text[position]; position = text.index(after: position)
            if !escaped, character == "\"" {
                do { return try JSONDecoder().decode(String.self, from: Data(text[start..<position].utf8)) }
                catch { throw malformed() }
            }
            if character == "\\", !escaped { escaped = true } else { escaped = false }
        }
        throw malformed()
    }
    private mutating func consume(_ value: String) -> Bool {
        guard text[position...].hasPrefix(value) else { return false }
        position = text.index(position, offsetBy: value.count); return true
    }
    private mutating func whitespace() {
        while position < text.endIndex, text[position].isWhitespace { position = text.index(after: position) }
    }
    private func malformed() -> ChatRequestIssue { ChatRequestIssue("tool_use.input", "model generated malformed Gemma tool arguments") }
}
