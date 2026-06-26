import Foundation
@preconcurrency import Tokenizers

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
        tools: [ToolDefinition]? = nil,
        addGenerationPrompt: Bool = true,
        includeThinking: Bool = true,
        maxLength: Int,
        imageTokenCounts: [Int] = []
    ) throws -> [Int] {
        if !imageTokenCounts.isEmpty {
            var encoded = tokenizer.encodeText(
                Self.renderPrompt(
                    messages: messages,
                    tools: tools,
                    addGenerationPrompt: addGenerationPrompt,
                    includeThinking: includeThinking,
                    imageTokenCounts: imageTokenCounts
                )
            )
            let targetLength = min(maxLength, tokenizer.maxLength)
            if encoded.count > targetLength {
                encoded = Array(encoded.suffix(targetLength))
            }
            return encoded
        }

        let toolSpecs: [ToolSpec]? = tools?.isEmpty == false ? tools!.map { $0.toToolSpec() } : nil
        return try tokenizer.encodeChatTemplate(
            messages: Self.renderMessages(messages),
            tools: toolSpecs,
            addGenerationPrompt: addGenerationPrompt,
            includeThinking: includeThinking,
            maxLength: maxLength
        )
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
        tools: [ToolDefinition]? = nil,
        addGenerationPrompt: Bool = true,
        includeThinking: Bool = true
    ) -> String {
        Self.renderPrompt(
            messages: messages,
            tools: tools,
            addGenerationPrompt: addGenerationPrompt,
            includeThinking: includeThinking
        )
    }

    static func renderMessages(_ messages: [ChatMessage]) -> [Message] {
        messages.map(renderMessage)
    }

    private static func renderMessage(_ message: ChatMessage) -> Message {
        var rendered: Message = [
            "role": message.role.rawValue,
        ]

        if message.role != .system, let imageURL = message.imageUrl, !imageURL.isEmpty {
            let content: [[String: String]] = [
                ["type": "image", "image_url": imageURL],
                ["type": "text", "text": message.content],
            ]
            rendered["content"] = content
        } else {
            rendered["content"] = message.content
        }

        return rendered
    }

    static func renderPrompt(
        messages: [ChatMessage],
        tools: [ToolDefinition]? = nil,
        addGenerationPrompt: Bool = true,
        includeThinking: Bool = true,
        imageTokenCounts: [Int] = []
    ) -> String {
        var prompt = ""
        var imageIndex = 0

        if let tools, !tools.isEmpty {
            prompt += "<|im_start|>system\n"
            prompt += "# Tools\n\nYou have access to the following functions:\n\n<tools>"
            for tool in tools {
                if let json = try? tool.promptSchemaJSONString() {
                    prompt += "\n\(json)"
                }
            }
            prompt += "\n</tools>"
            prompt += "\n\nIf you choose to call a function ONLY reply in the XML tool_call format."
            prompt += "<|im_end|>\n"
        }

        for message in messages {
            let role = message.role.rawValue
            prompt += "<|im_start|>\(role)\n"

            if message.role != .system, let imageURL = message.imageUrl, !imageURL.isEmpty {
                let imageTokenCount = imageIndex < imageTokenCounts.count ? max(1, imageTokenCounts[imageIndex]) : 1
                prompt += "<|vision_start|>"
                prompt += String(repeating: "<|image_pad|>", count: imageTokenCount)
                prompt += "<|vision_end|>"
                imageIndex += 1
            }

            if !message.content.isEmpty {
                prompt += message.content
            }

            prompt += "<|im_end|>\n"
        }

        if addGenerationPrompt {
            if includeThinking {
                prompt += "<|im_start|>assistant\n<think>\n"
            } else {
                prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            }
        }

        return prompt
    }
}
