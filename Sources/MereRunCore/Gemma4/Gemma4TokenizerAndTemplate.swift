import Foundation
@preconcurrency import Hub
@preconcurrency import Tokenizers

public final class Gemma4TokenizerAndTemplate: @unchecked Sendable {
    public let tokenizer: any Tokenizer
    public let maxLength: Int
    public let eosTokenId: Int?
    public let turnTokenId: Int?
    public let toolCallEndTokenId: Int?

    public init(
        tokenizer: any Tokenizer,
        maxLength: Int,
        eosTokenId: Int?,
        turnTokenId: Int?,
        toolCallEndTokenId: Int? = nil
    ) {
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.eosTokenId = eosTokenId
        self.turnTokenId = turnTokenId
        self.toolCallEndTokenId = toolCallEndTokenId
    }

    public static func load(
        from rootURL: URL,
        maxLengthOverride: Int? = nil,
        hubApi: HubApi = .shared
    ) async throws -> Gemma4TokenizerAndTemplate {
        let tokenizer = try await AutoTokenizer.from(modelFolder: rootURL, hubApi: hubApi)
        let tokenizerConfigURL = rootURL.appending(path: "tokenizer_config.json")

        let maxLength: Int
        if FileManager.default.fileExists(atPath: tokenizerConfigURL.path),
           let data = try? Data(contentsOf: tokenizerConfigURL),
           let config = try? JSONDecoder().decode(Gemma4TokenizerConfig.self, from: data),
           let configured = config.modelMaxLength {
            let raw = configured.value
            maxLength = maxLengthOverride ?? (raw > 0 ? raw : 131_072)
        } else {
            maxLength = maxLengthOverride ?? 131_072
        }

        return Gemma4TokenizerAndTemplate(
            tokenizer: tokenizer,
            maxLength: maxLength,
            eosTokenId: tokenizer.eosTokenId,
            turnTokenId: tokenizer.convertTokenToId("<turn|>"),
            toolCallEndTokenId: tokenizer.convertTokenToId("<tool_call|>")
        )
    }

    public func encodeForGeneration(
        messages: [ChatMessage],
        tools: [ToolDefinition]? = nil,
        addGenerationPrompt: Bool = true,
        includeThinking: Bool,
        maxLength: Int
    ) throws -> [Int] {
        let renderedMessages = Self.renderMessages(messages)

        let toolSpecs: [ToolSpec]? = tools?.isEmpty == false ? tools!.map { $0.toToolSpec() } : nil

        var encoded = try tokenizer.applyChatTemplate(
            messages: renderedMessages,
            chatTemplate: nil,
            addGenerationPrompt: addGenerationPrompt,
            truncation: false,
            maxLength: nil,
            tools: toolSpecs,
            additionalContext: ["enable_thinking": includeThinking]
        )

        let targetLength = min(maxLength, self.maxLength)
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

    public var stopTokenIds: [Int] {
        [eosTokenId, turnTokenId].compactMap { $0 }
    }

    public func stopTokenIds(withTools: Bool) -> [Int] {
        var ids = stopTokenIds
        if withTools, let tcEnd = toolCallEndTokenId {
            ids.append(tcEnd)
        }
        return ids
    }

    static func renderMessages(_ messages: [ChatMessage]) -> [Message] {
        messages.map(renderMessage)
    }

    private static func renderMessage(_ message: ChatMessage) -> Message {
        var rendered: Message = [
            "role": message.role.rawValue,
        ]

        switch message.role {
        case .assistant:
            let toolCalls = Gemma4ToolParser.parseToolCalls(message.content)
            if !toolCalls.isEmpty {
                let renderedToolCalls: [[String: any Sendable]] = toolCalls.map { call in
                    [
                        "function": [
                            "name": call.name,
                            "arguments": call.arguments,
                        ] as [String: any Sendable],
                    ]
                }
                rendered["tool_calls"] = renderedToolCalls

                let content = stripToolCalls(from: message.content)
                if !content.isEmpty {
                    rendered["content"] = content
                }
            } else {
                rendered["content"] = renderContent(for: message)
            }
        case .tool:
            let toolResponses: [[String: any Sendable]] = [[
                "name": "tool",
                "response": message.content,
            ]]
            rendered["tool_responses"] = toolResponses
        default:
            rendered["content"] = renderContent(for: message)
        }

        return rendered
    }

    private static func renderContent(for message: ChatMessage) -> any Sendable {
        if let imageURL = message.imageUrl, !imageURL.isEmpty {
            return [
                ["type": "text", "text": message.content],
                ["type": "image", "image_url": imageURL],
            ] as [[String: String]]
        }
        return message.content
    }

    private static func stripToolCalls(from text: String) -> String {
        text
            .replacingOccurrences(
                of: "(?s)<\\|tool_call>.*?(?:<tool_call\\|>|$)",
                with: "",
                options: .regularExpression
            )
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private struct Gemma4TokenizerConfig: Decodable {
    let modelMaxLength: LenientInt?

    enum CodingKeys: String, CodingKey {
        case modelMaxLength = "model_max_length"
    }
}
