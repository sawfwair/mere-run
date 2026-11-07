import Foundation

/// Keeps checkpoint rendering for tools, reasoning and media while inserting
/// real ChatML system turns at their original conversation boundaries.
enum Q35SystemPrompt {
    static let toolParameterEncoding = """
    Tool parameter encoding: inside each <parameter=...> element, write string values as raw text, without JSON quotes. \
    Encode every non-string value as valid JSON matching its parameter schema. Boolean values must be lowercase true or false \
    without quotes; null must be lowercase null. Encode numbers, arrays and objects as JSON, including lowercase booleans and \
    null in nested values. Do not use Python True, False or None, Markdown fences, or XML type annotations.
    """

    static func encode(messages: [ChatMessage], addGenerationPrompt: Bool,
                       hasTools: Bool = false,
                       render: ([ChatMessage], Bool) throws -> [Int],
                       encodeText: (String) -> [Int]) throws -> [Int] {
        let messages = hasTools ? messagesWithToolEncoding(messages) : messages
        var base: [ChatMessage] = []
        var insertions: [(index: Int, content: String)] = []
        var leadingSystems: [ChatMessage] = []
        for message in messages {
            if message.role == .system, base.isEmpty {
                leadingSystems.append(message)
                continue
            }
            if !leadingSystems.isEmpty {
                if leadingSystems.count == 1 { base.append(leadingSystems[0]) }
                else { base.append(ChatMessage(role: .system, content: leadingSystems.map(NativeChatTextContent.render).joined(separator: "\n"))) }
                leadingSystems = []
            }
            if message.role == .system { insertions.append((base.count, NativeChatTextContent.render(message))) }
            else { base.append(message) }
        }
        if !leadingSystems.isEmpty {
            base.append(ChatMessage(role: .system, content: leadingSystems.map(NativeChatTextContent.render).joined(separator: "\n")))
        }
        guard !insertions.isEmpty else { return try render(base, addGenerationPrompt) }
        let tokens = try render(base, addGenerationPrompt)
        var result: [Int] = []
        var offset = 0
        for insertion in insertions {
            let prefix = try render(Array(base.prefix(insertion.index)), false)
            guard prefix.count >= offset, tokens.starts(with: prefix) else {
                throw ChatRequestIssue("messages", "the Qwen template cannot preserve this system-message boundary; use a compatible adapter")
            }
            result += tokens[offset..<prefix.count]
            result += encodeText("<|im_start|>system\n" + insertion.content + "<|im_end|>\n")
            offset = prefix.count
        }
        result += tokens[offset...]
        return result
    }

    static func messagesWithToolEncoding(_ messages: [ChatMessage]) -> [ChatMessage] {
        // Keep the native wire-format reminder next to generation even with a
        // large client preamble, without rewriting any supplied conversation turn.
        messages + [ChatMessage(role: .system, content: toolParameterEncoding)]
    }
}
