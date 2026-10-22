import Foundation
@preconcurrency import Tokenizers

enum FlintPromptMessages {
    static func render(_ messages: [ChatMessage]) -> [Message] {
        let sanitized = messages.map { message in
            var message = message
            // Swift Jinja does not interpret the template's Python \x00 string escape.
            // Apply the same NUL removal before rendering non-assistant content.
            if message.role != .assistant {
                message.content = message.content.replacingOccurrences(of: "\u{0000}", with: "")
            }
            return message
        }
        return LagunaTokenizerAndTemplate.renderMessages(sanitized)
    }
}
