import Foundation

public struct Q35TokenizerAndTemplate {
    public let tokenizer: QwenTokenizer

    public init(tokenizer: QwenTokenizer) {
        self.tokenizer = tokenizer
    }

    public static func load(from rootURL: URL, maxLengthOverride: Int? = nil) throws -> Q35TokenizerAndTemplate {
        let tokenizer = try QwenTokenizer.load(from: rootURL, maxLengthOverride: maxLengthOverride)
        return Q35TokenizerAndTemplate(tokenizer: tokenizer)
    }

    public func encodeForGeneration(
        messages: [ChatMessage],
        tools: [[String: Any]]? = nil,
        addGenerationPrompt: Bool = true,
        includeThinking: Bool = true,
        maxLength: Int
    ) -> [Int] {
        let rendered = render(
            messages: messages,
            tools: tools,
            addGenerationPrompt: addGenerationPrompt,
            includeThinking: includeThinking
        )
        let encoded = tokenizer.encodeText(rendered)
        if encoded.count <= maxLength {
            return encoded
        }
        return Array(encoded.suffix(maxLength))
    }

    public func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    public func decode(token: Int) -> String {
        tokenizer.decode(tokens: [token])
    }

    public var eosTokenId: Int? {
        tokenizer.eosTokenId
    }

    public func render(
        messages: [ChatMessage],
        tools: [[String: Any]]? = nil,
        addGenerationPrompt: Bool = true,
        includeThinking: Bool = true
    ) -> String {
        var prompt = ""

        if let tools, !tools.isEmpty {
            prompt += "<|im_start|>system\n"
            prompt += "# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>"
            for tool in tools {
                if let data = try? JSONSerialization.data(withJSONObject: tool, options: []),
                   let json = String(data: data, encoding: .utf8) {
                    prompt += "\\n\(json)"
                }
            }
            prompt += "\\n</tools>"
            prompt += "\\n\\nIf you choose to call a function ONLY reply in the XML tool_call format."
            prompt += "<|im_end|>\\n"
        }

        for message in messages {
            let role = message.role.rawValue
            prompt += "<|im_start|>\(role)\\n"

            if message.role != .system, let imageURL = message.imageUrl, !imageURL.isEmpty {
                prompt += "<|vision_start|><|image_pad|><|vision_end|>\\n"
            }

            if !message.content.isEmpty {
                prompt += message.content
            }

            prompt += "<|im_end|>\\n"
        }

        if addGenerationPrompt {
            if includeThinking {
                prompt += "<|im_start|>assistant\\n<think>\\n"
            } else {
                prompt += "<|im_start|>assistant\\n"
            }
        }

        return prompt
    }
}
