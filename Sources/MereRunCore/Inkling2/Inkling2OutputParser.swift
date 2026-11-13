import Foundation

enum Inkling2OutputParser {
    static func parse(_ text: String, tools: [ToolDefinition] = [], parallel: Bool = true) -> InklingParsedOutput {
        let markers = ["<|content_text|>", "<|content_thinking|>", "<|content_invoke_tool_json|>"]
        var visible: [String] = [], reasoning: [String] = [], calls: [ToolCall] = []
        for message in text.components(separatedBy: "<|end_message|>") {
            let first = markers.compactMap { marker in message.range(of: marker).map { (marker, $0) } }
                .min { $0.1.lowerBound < $1.1.lowerBound }
            guard let first else { continue }
            let content = String(message[first.1.upperBound...])
                .replacingOccurrences(of: "<|content_model_end_sampling|>", with: "")
            switch first.0 {
            case markers[0]: visible.append(content)
            case markers[1]: reasoning.append(content)
            default:
                // Parse only the message's declared channel. Invocation-shaped text
                // inside a thinking or text message is never an executable call.
                calls += InklingOutputParser.parse(first.0 + content + "<|end_message|>").toolCalls
            }
        }
        let thought = reasoning.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        return InklingParsedOutput(
            visible: visible.joined(separator: "\n"),
            reasoning: thought.isEmpty ? nil : thought,
            toolCalls: ToolCallPolicy.validatedCalls(calls, tools: tools, parallelToolCalls: parallel)
        )
    }
}
