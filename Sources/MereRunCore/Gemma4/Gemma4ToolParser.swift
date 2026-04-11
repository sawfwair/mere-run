import Foundation

/// Parses Gemma4 tool calls from generated text.
///
/// Gemma4 format:
///   `<|tool_call>call:functionName{key1:value1,key2:value2}<tool_call|>`
///
/// String values may use `<|"|>` delimiters.
public enum Gemma4ToolParser {
    private static let toolCallStart = "<|tool_call>"
    private static let toolCallEnd = "<tool_call|>"

    public static func parseToolCalls(_ text: String) -> [ToolCall] {
        var results: [ToolCall] = []
        var searchRange = text.startIndex..<text.endIndex

        while let startRange = text.range(of: toolCallStart, range: searchRange) {
            let afterStart = startRange.upperBound
            let payloadEnd: String.Index
            if let endRange = text.range(of: toolCallEnd, range: afterStart..<text.endIndex) {
                payloadEnd = endRange.lowerBound
                searchRange = endRange.upperBound..<text.endIndex
            } else {
                // No closing tag — generation stopped on the <tool_call|> EOS token,
                // so it was consumed and not included in the decoded text.
                payloadEnd = text.endIndex
                searchRange = text.endIndex..<text.endIndex
            }
            let payload = String(text[afterStart..<payloadEnd]).trimmingCharacters(in: .whitespacesAndNewlines)
            if let call = parseSingleCall(payload) {
                results.append(call)
            }
        }

        return results
    }

    /// Parse: `call:functionName{key1:value1,key2:value2}`
    private static func parseSingleCall(_ payload: String) -> ToolCall? {
        guard payload.hasPrefix("call:") else { return nil }
        let afterCall = payload.dropFirst("call:".count)

        guard let braceIndex = afterCall.firstIndex(of: "{") else {
            // No arguments: `call:functionName`
            let name = String(afterCall).trimmingCharacters(in: .whitespacesAndNewlines)
            return name.isEmpty ? nil : ToolCall(name: name, arguments: [:])
        }

        let name = String(afterCall[afterCall.startIndex..<braceIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }

        // Extract the content between outermost { }
        let argsStart = afterCall.index(after: braceIndex)
        guard let closingBrace = findMatchingClosingBrace(String(afterCall), from: afterCall.distance(from: afterCall.startIndex, to: braceIndex)) else {
            return ToolCall(name: name, arguments: [:])
        }

        let argsString = String(afterCall[argsStart..<afterCall.index(afterCall.startIndex, offsetBy: closingBrace)])
        let arguments = parseArguments(argsString)
        return ToolCall(name: name, arguments: arguments)
    }

    /// Find the matching `}` for the `{` at position `openPos`, respecting nesting.
    private static func findMatchingClosingBrace(_ text: String, from openPos: Int) -> Int? {
        var depth = 0
        var inGemmaString = false
        let chars = Array(text)
        var i = openPos

        while i < chars.count {
            // Check for <|"|> string delimiter
            if i + 4 < chars.count,
               chars[i] == "<", chars[i+1] == "|", chars[i+2] == "\"", chars[i+3] == "|", chars[i+4] == ">" {
                inGemmaString.toggle()
                i += 5
                continue
            }

            if !inGemmaString {
                if chars[i] == "{" { depth += 1 }
                else if chars[i] == "}" {
                    depth -= 1
                    if depth == 0 { return i }
                }
            }
            i += 1
        }
        return nil
    }

    /// Parse `key1:value1,key2:value2` into a dictionary.
    /// Handles `<|"|>` delimited strings and nested structures.
    private static func parseArguments(_ text: String) -> [String: String] {
        var args: [String: String] = [:]
        var i = text.startIndex

        while i < text.endIndex {
            // Skip whitespace and commas
            while i < text.endIndex && (text[i] == " " || text[i] == "," || text[i] == "\n") {
                i = text.index(after: i)
            }
            guard i < text.endIndex else { break }

            // Parse key (up to `:`)
            let keyStart = i
            while i < text.endIndex && text[i] != ":" {
                i = text.index(after: i)
            }
            guard i < text.endIndex else { break }
            let key = String(text[keyStart..<i]).trimmingCharacters(in: .whitespacesAndNewlines)
            i = text.index(after: i) // skip `:`

            // Parse value
            let value = parseValue(text, from: &i)
            if !key.isEmpty {
                args[key] = unescapeGemmaString(value)
            }
        }

        return args
    }

    /// Parse a value starting at position `i`, advancing `i` past the value.
    private static func parseValue(_ text: String, from i: inout String.Index) -> String {
        guard i < text.endIndex else { return "" }

        // Check for <|"|> delimited string
        let remaining = text[i...]
        if remaining.hasPrefix("<|\"") {
            let delimLen = "<|\"|>".count
            if remaining.hasPrefix("<|\"|>") {
                let valueStart = text.index(i, offsetBy: delimLen)
                if let endDelim = text.range(of: "<|\"|>", range: valueStart..<text.endIndex) {
                    let value = String(text[valueStart..<endDelim.lowerBound])
                    i = endDelim.upperBound
                    return value
                }
            }
        }

        // Check for nested object
        if remaining.hasPrefix("{") {
            let startPos = text.distance(from: text.startIndex, to: i)
            if let closePos = findMatchingClosingBrace(String(text), from: startPos) {
                let endIdx = text.index(text.startIndex, offsetBy: closePos + 1)
                let value = String(text[i..<endIdx])
                i = endIdx
                return value
            }
        }

        // Check for array
        if remaining.hasPrefix("[") {
            var depth = 0
            let start = i
            while i < text.endIndex {
                if text[i] == "[" { depth += 1 }
                else if text[i] == "]" {
                    depth -= 1
                    if depth == 0 {
                        i = text.index(after: i)
                        return String(text[start..<i])
                    }
                }
                i = text.index(after: i)
            }
            return String(text[start..<text.endIndex])
        }

        // Plain value (until next comma or end)
        let start = i
        var depth = 0
        while i < text.endIndex {
            if text[i] == "{" { depth += 1 }
            else if text[i] == "}" { depth -= 1; if depth < 0 { break } }
            else if text[i] == "," && depth == 0 { break }
            i = text.index(after: i)
        }
        return String(text[start..<i]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Remove `<|"|>` delimiters from a string value.
    private static func unescapeGemmaString(_ text: String) -> String {
        text.replacingOccurrences(of: "<|\"|>", with: "")
    }
}
