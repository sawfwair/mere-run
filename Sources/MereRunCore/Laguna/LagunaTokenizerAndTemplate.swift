import Foundation
@preconcurrency import Hub
@preconcurrency import Tokenizers

public final class LagunaTokenizerAndTemplate: @unchecked Sendable {
    public let tokenizer: any Tokenizer
    public let maxLength: Int
    public let eosTokenID: Int?
    public let assistantEndTokenID: Int?

    init(
        tokenizer: any Tokenizer,
        maxLength: Int,
        eosTokenID: Int?,
        assistantEndTokenID: Int?
    ) {
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.eosTokenID = eosTokenID
        self.assistantEndTokenID = assistantEndTokenID
    }

    public static func load(
        from rootURL: URL,
        maxLength: Int,
        hubApi: HubApi = .shared
    ) async throws -> LagunaTokenizerAndTemplate {
        let tokenizer = try await AutoTokenizer.from(modelFolder: rootURL, hubApi: hubApi)
        return LagunaTokenizerAndTemplate(
            tokenizer: tokenizer,
            maxLength: maxLength,
            eosTokenID: tokenizer.eosTokenId,
            assistantEndTokenID: tokenizer.convertTokenToId("</assistant>")
        )
    }

    public func encodeForGeneration(
        messages: [ChatMessage],
        tools: [ToolDefinition]? = nil,
        addGenerationPrompt: Bool = true,
        includeThinking: Bool,
        maxLength: Int
    ) throws -> [Int] {
        let toolSpecs = tools?.isEmpty == false ? tools!.map { $0.toToolSpec() } : nil
        var encoded = try tokenizer.applyChatTemplate(
            messages: Self.renderMessages(messages),
            chatTemplate: nil,
            addGenerationPrompt: addGenerationPrompt,
            truncation: false,
            maxLength: nil,
            tools: toolSpecs,
            additionalContext: [
                "enable_thinking": includeThinking,
                "preserve_thinking": includeThinking,
            ]
        )
        let limit = min(maxLength, self.maxLength)
        if encoded.count > limit {
            encoded = Array(encoded.suffix(limit))
        }
        return encoded
    }

    public func encodeRaw(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    public func decode(token: Int) -> String {
        tokenizer.decode(tokens: [token])
    }

    public var stopTokenIDs: [Int] {
        Array(Set([eosTokenID, assistantEndTokenID].compactMap { $0 }))
    }

    static func renderMessages(_ messages: [ChatMessage]) -> [Message] {
        messages.map(renderMessage)
    }

    private static func renderMessage(_ message: ChatMessage) -> Message {
        var rendered: Message = [
            "role": message.role.rawValue,
            "content": message.content,
        ]
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
}
