import Foundation

struct MuseGlimmerParsedOutput {
    let visible: String
    let reasoning: String?
    let toolCalls: [ToolCall]
}

enum MuseGlimmerOutputParser {
    static func parse(_ raw: String) -> MuseGlimmerParsedOutput {
        let toolCalls = MuseGlimmerToolParser.parseToolCalls(raw)
        var visible = MuseGlimmerToolParser.visibleText(raw)
        var reasoning: String?

        let reasoningMarkers = ["to=self<|message|>", " to=self<|message|>"]
        if let marker = reasoningMarkers.first(where: { visible.contains($0) }),
           let range = visible.range(of: marker) {
            let remainder = String(visible[range.upperBound...])
            if let finalRange = remainder.range(of: "<|start|>assistant to=user<|message|>") {
                reasoning = String(remainder[..<finalRange.lowerBound])
                    .replacingOccurrences(of: "<|eom|>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                visible = String(remainder[finalRange.upperBound...])
            } else {
                reasoning = remainder
                    .replacingOccurrences(of: "<|eom|>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                visible = ""
            }
        }

        for prefix in [
            " to=user<|message|>",
            "to=user<|message|>",
            "<|start|>assistant to=user<|message|>",
        ] where visible.hasPrefix(prefix) {
            visible.removeFirst(prefix.count)
        }
        visible = visible
            .replacingOccurrences(of: "<|eot|>", with: "")
            .replacingOccurrences(of: "<|eom|>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return MuseGlimmerParsedOutput(visible: visible, reasoning: reasoning, toolCalls: toolCalls)
    }
}
