import Foundation

struct MuseGlimmerParsedOutput {
    let visible: String
    let reasoning: String?
    let toolCalls: [ToolCall]
}

enum MuseGlimmerOutputParser {
    static func parse(_ raw: String) -> MuseGlimmerParsedOutput {
        let toolCalls = MuseGlimmerToolParser.parseToolCalls(raw)
        var parser = MuseGlimmerContentStream()
        _ = parser.append(raw, final: true)
        let visible = parser.visible.trimmingCharacters(in: .whitespacesAndNewlines)
        let reasoning = parser.thinking.trimmingCharacters(in: .whitespacesAndNewlines)
        return MuseGlimmerParsedOutput(visible: visible, reasoning: reasoning.isEmpty ? nil : reasoning, toolCalls: toolCalls)
    }
}
