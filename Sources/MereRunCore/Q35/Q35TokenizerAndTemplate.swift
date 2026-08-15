import Foundation
@preconcurrency import Tokenizers

enum Q35TokenizerAndTemplateError: LocalizedError {
    case missingImageToken
    case imageTokenCountMismatch(expected: Int, actual: Int)

    var errorDescription: String? {
        switch self {
        case .missingImageToken:
            "Qwen-family tokenizer is missing the image placeholder token."
        case .imageTokenCountMismatch(let expected, let actual):
            "Qwen-family prompt rendered \(actual) image placeholders for \(expected) encoded images."
        }
    }
}

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
        let toolSpecs: [ToolSpec]? = tools?.isEmpty == false ? tools!.map { $0.toToolSpec() } : nil
        var encoded = try tokenizer.encodeChatTemplate(
            messages: Self.renderMessages(messages),
            tools: toolSpecs,
            addGenerationPrompt: addGenerationPrompt,
            includeThinking: includeThinking,
            maxLength: tokenizer.maxLength
        )
        if !imageTokenCounts.isEmpty {
            guard let imageTokenId = tokenizer.imageTokenId else {
                throw Q35TokenizerAndTemplateError.missingImageToken
            }
            encoded = try Self.expandingImageTokenIds(
                encoded,
                imageTokenId: imageTokenId,
                imageTokenCounts: imageTokenCounts
            )
        }

        let targetLength = min(maxLength, tokenizer.maxLength)
        if encoded.count > targetLength {
            encoded = Array(encoded.suffix(targetLength))
        }
        return encoded
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

        if let reasoning = message.reasoningContent, !reasoning.isEmpty {
            rendered["reasoning_content"] = reasoning
        }
        if let name = message.name, !name.isEmpty {
            rendered["name"] = name
        }
        if let toolCallID = message.toolCallID, !toolCallID.isEmpty {
            rendered["tool_call_id"] = toolCallID
        }
        if let calls = message.toolCalls, !calls.isEmpty {
            rendered["tool_calls"] = calls.map { call -> [String: any Sendable] in
                var result: [String: any Sendable] = [
                    "function": [
                        "name": call.name,
                        "arguments": call.arguments.mapValues(renderJSONValue),
                    ] as [String: any Sendable],
                ]
                if let id = call.id, !id.isEmpty {
                    result["id"] = id
                }
                return result
            }
        }

        return rendered
    }

    static func expandingImageTokenIds(
        _ tokenIds: [Int],
        imageTokenId: Int,
        imageTokenCounts: [Int]
    ) throws -> [Int] {
        var expanded: [Int] = []
        expanded.reserveCapacity(tokenIds.count + imageTokenCounts.reduce(0, +))
        var imageIndex = 0

        for tokenId in tokenIds {
            guard tokenId == imageTokenId else {
                expanded.append(tokenId)
                continue
            }
            guard imageIndex < imageTokenCounts.count else {
                throw Q35TokenizerAndTemplateError.imageTokenCountMismatch(
                    expected: imageTokenCounts.count,
                    actual: imageIndex + 1
                )
            }
            expanded.append(
                contentsOf: repeatElement(imageTokenId, count: max(1, imageTokenCounts[imageIndex]))
            )
            imageIndex += 1
        }

        guard imageIndex == imageTokenCounts.count else {
            throw Q35TokenizerAndTemplateError.imageTokenCountMismatch(
                expected: imageTokenCounts.count,
                actual: imageIndex
            )
        }
        return expanded
    }

    private static func renderJSONValue(_ value: OpenAIJSONValue) -> any Sendable {
        switch value {
        case .string(let value): value
        case .number(let value): value
        case .bool(let value): value
        case .object(let value): value.mapValues(renderJSONValue)
        case .array(let value): value.map(renderJSONValue)
        case .null: Optional<String>.none
        }
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
