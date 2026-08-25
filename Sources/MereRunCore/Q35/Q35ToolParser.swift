import Foundation

/// Parses the native Qwen3.5/Qwen3.8 function-call protocol emitted by the
/// checkpoint chat template.
///
/// Qwen-family format:
/// ```
/// <tool_call>
/// <function=tool_name>
/// <parameter=argument_name>
/// argument value
/// </parameter>
/// </function>
/// </tool_call>
/// ```
public enum Q35ToolParser {
    public static let closingTag = "</tool_call>"

    private static let openingTag = "<tool_call>"
    private static let functionOpeningTag = "<function="
    private static let functionClosingTag = "</function>"
    private static let parameterOpeningTag = "<parameter="
    private static let parameterClosingTag = "</parameter>"

    /// Incremental completion detector for token streaming.
    ///
    /// Most tokens cannot complete a tool call, so structural parsing only runs
    /// after a newly streamed closing marker. The number of reparses is bounded
    /// to keep attacker-controlled output from creating quadratic work. If the
    /// bound is exhausted, generation continues to EOS and the final response
    /// still goes through the unbounded, linear structural parser.
    public struct StreamingCompletionDetector: Sendable {
        private static let maximumStructuralChecks = 64

        private var bufferedText = ""
        private var markerTail = ""
        private var structuralChecks = 0

        public init() {}

        public mutating func feed(_ piece: String) -> Bool {
            guard !piece.isEmpty else { return false }
            bufferedText += piece

            let markerWindow = markerTail + piece
            let newMarkerCount = Q35ToolParser.occurrenceCount(
                of: Q35ToolParser.closingTag,
                in: markerWindow
            )
            markerTail = String(markerWindow.suffix(max(0, Q35ToolParser.closingTag.count - 1)))

            guard newMarkerCount > 0,
                  structuralChecks < Self.maximumStructuralChecks else {
                return false
            }
            structuralChecks += newMarkerCount
            return Q35ToolParser.containsCompletedToolCall(bufferedText)
        }
    }

    /// Streams only user-visible prose while a tool-capable decode is active.
    ///
    /// The checkpoint contract permits prose before a tool call but forbids a
    /// suffix after the opening envelope. Keep a short marker-sized tail so an
    /// opening tag split across tokenizer pieces is never exposed, then suppress
    /// the protocol envelope once it begins.
    public struct StreamingVisibleTextFilter: Sendable {
        private var pendingText = ""
        private var isSuppressingToolCall = false

        public init() {}

        public mutating func feed(_ piece: String) -> String {
            guard !piece.isEmpty, !isSuppressingToolCall else { return "" }
            pendingText += piece

            if let openingRange = pendingText.range(of: Q35ToolParser.openingTag) {
                let visible = String(pendingText[..<openingRange.lowerBound])
                pendingText = ""
                isSuppressingToolCall = true
                return visible
            }

            let retainedCount = max(0, Q35ToolParser.openingTag.count - 1)
            let emittedCount = max(0, pendingText.count - retainedCount)
            guard emittedCount > 0 else { return "" }
            let emittedEnd = pendingText.index(pendingText.startIndex, offsetBy: emittedCount)
            let visible = String(pendingText[..<emittedEnd])
            pendingText.removeSubrange(..<emittedEnd)
            return visible
        }

        public mutating func finish() -> String {
            guard !isSuppressingToolCall else { return "" }
            defer { pendingText = "" }
            return pendingText
        }
    }

    public static func parseToolCalls(_ text: String) -> [ToolCall] {
        parsedEnvelopes(in: text).compactMap(\.toolCall)
    }

    /// Removes structurally valid tool-call envelopes while preserving any
    /// allowed prose that precedes them. Malformed marker-shaped text remains
    /// visible and is never promoted into a typed call.
    public static func visibleText(_ text: String) -> String {
        var visible = text
        for envelope in parsedEnvelopes(in: text).reversed() {
            visible.removeSubrange(envelope.startIndex..<envelope.endIndex)
        }
        return visible.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func containsCompletedToolCall(_ text: String) -> Bool {
        firstParsedEnvelope(in: text, startingAt: text.startIndex) != nil
    }

    private struct ParsedEnvelope {
        let startIndex: String.Index
        let endIndex: String.Index
        let toolCall: ToolCall?
    }

    private static func parsedEnvelopes(in text: String) -> [ParsedEnvelope] {
        var parsed: [ParsedEnvelope] = []
        var cursor = text.startIndex

        while cursor < text.endIndex,
              let openingRange = text.range(
                  of: openingTag,
                  range: cursor..<text.endIndex
              ) {
            if let envelope = parseEnvelope(in: text, openingRange: openingRange) {
                parsed.append(envelope)
                cursor = envelope.endIndex
            } else {
                cursor = openingRange.upperBound
            }
        }

        return parsed
    }

    private static func firstParsedEnvelope(
        in text: String,
        startingAt cursor: String.Index
    ) -> ParsedEnvelope? {
        var searchCursor = cursor
        while searchCursor < text.endIndex,
              let openingRange = text.range(
                  of: openingTag,
                  range: searchCursor..<text.endIndex
              ) {
            if let envelope = parseEnvelope(in: text, openingRange: openingRange) {
                return envelope
            }
            searchCursor = openingRange.upperBound
        }
        return nil
    }

    /// Parse one envelope by following sibling parameter elements. A parameter
    /// close is structural only when the next non-space token is another
    /// parameter or the enclosing function close. Marker-shaped text inside a
    /// value is therefore preserved instead of truncating the call.
    private static func parseEnvelope(
        in text: String,
        openingRange: Range<String.Index>
    ) -> ParsedEnvelope? {
        var cursor = skipWhitespace(in: text, from: openingRange.upperBound)
        guard text[cursor...].hasPrefix(functionOpeningTag) else { return nil }

        let nameStart = text.index(cursor, offsetBy: functionOpeningTag.count)
        guard let nameEnd = elementOpeningTagEnd(in: text, from: nameStart) else {
            return nil
        }
        let name = text[nameStart..<nameEnd]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        cursor = text.index(after: nameEnd)

        var arguments: [String: String] = [:]
        while true {
            cursor = skipWhitespace(in: text, from: cursor)

            if text[cursor...].hasPrefix(functionClosingTag) {
                let functionEnd = text.index(cursor, offsetBy: functionClosingTag.count)
                let callClose = skipWhitespace(in: text, from: functionEnd)
                guard text[callClose...].hasPrefix(closingTag) else { return nil }
                let envelopeEnd = text.index(callClose, offsetBy: closingTag.count)
                let toolCall = name.isEmpty ? nil : ToolCall(name: name, arguments: arguments)
                return ParsedEnvelope(
                    startIndex: openingRange.lowerBound,
                    endIndex: envelopeEnd,
                    toolCall: toolCall
                )
            }

            guard text[cursor...].hasPrefix(parameterOpeningTag) else { return nil }
            let keyStart = text.index(cursor, offsetBy: parameterOpeningTag.count)
            guard let keyEnd = elementOpeningTagEnd(in: text, from: keyStart) else {
                return nil
            }
            let key = text[keyStart..<keyEnd]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let valueStart = text.index(after: keyEnd)
            guard let valueClose = structuralParameterClose(
                in: text,
                valueStart: valueStart
            ) else {
                return nil
            }

            if !key.isEmpty {
                arguments[key] = text[valueStart..<valueClose.lowerBound]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            cursor = valueClose.upperBound
        }
    }

    private static func elementOpeningTagEnd(
        in text: String,
        from start: String.Index
    ) -> String.Index? {
        guard let end = text[start...].firstIndex(of: ">") else { return nil }
        guard !text[start..<end].contains(where: { $0.isNewline }) else { return nil }
        return end
    }

    private static func structuralParameterClose(
        in text: String,
        valueStart: String.Index
    ) -> Range<String.Index>? {
        var searchCursor = valueStart
        while searchCursor < text.endIndex,
              let closeRange = text.range(
                  of: parameterClosingTag,
                  range: searchCursor..<text.endIndex
              ) {
            let nextToken = skipWhitespace(in: text, from: closeRange.upperBound)
            if text[nextToken...].hasPrefix(parameterOpeningTag)
                || text[nextToken...].hasPrefix(functionClosingTag) {
                return closeRange
            }
            searchCursor = closeRange.upperBound
        }
        return nil
    }

    private static func skipWhitespace(
        in text: String,
        from start: String.Index
    ) -> String.Index {
        var cursor = start
        while cursor < text.endIndex, text[cursor].isWhitespace {
            cursor = text.index(after: cursor)
        }
        return cursor
    }

    private static func occurrenceCount(of needle: String, in text: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        var count = 0
        var cursor = text.startIndex
        while cursor < text.endIndex,
              let range = text.range(of: needle, range: cursor..<text.endIndex) {
            count += 1
            cursor = range.upperBound
        }
        return count
    }
}
