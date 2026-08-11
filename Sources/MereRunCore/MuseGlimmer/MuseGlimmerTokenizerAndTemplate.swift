import Foundation
@preconcurrency import Hub
@preconcurrency import Tokenizers

public final class MuseGlimmerTokenizerAndTemplate: @unchecked Sendable {
    public let tokenizer: any Tokenizer
    public let maxLength: Int
    public let eosTokenIds: [Int]
    public let imageTokenId: Int
    public let imageStartTokenId: Int
    public let imageEndTokenId: Int

    private init(
        tokenizer: any Tokenizer,
        maxLength: Int,
        eosTokenIds: [Int],
        imageTokenId: Int,
        imageStartTokenId: Int,
        imageEndTokenId: Int
    ) {
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.eosTokenIds = eosTokenIds
        self.imageTokenId = imageTokenId
        self.imageStartTokenId = imageStartTokenId
        self.imageEndTokenId = imageEndTokenId
    }

    static func load(
        from rootURL: URL,
        generationConfig: MuseGlimmerGenerationConfig,
        maxLengthOverride: Int? = nil,
        hubApi: HubApi = .shared
    ) async throws -> MuseGlimmerTokenizerAndTemplate {
        let tokenizer = try await AutoTokenizer.from(modelFolder: rootURL, hubApi: hubApi)
        guard let imageTokenId = tokenizer.convertTokenToId("<|patch|>"),
              let imageStartTokenId = tokenizer.convertTokenToId("<|image_start|>"),
              let imageEndTokenId = tokenizer.convertTokenToId("<|image_end|>") else {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Muse Glimmer tokenizer is missing one or more required image tokens."
            )
        }
        let configuredMaxLength = generationConfig.maxLength ?? MuseGlimmerResources.defaultContextLength
        let stopIds = Array(Set(generationConfig.eosTokenIds + [tokenizer.eosTokenId].compactMap { $0 })).sorted()
        return MuseGlimmerTokenizerAndTemplate(
            tokenizer: tokenizer,
            maxLength: maxLengthOverride ?? configuredMaxLength,
            eosTokenIds: stopIds,
            imageTokenId: imageTokenId,
            imageStartTokenId: imageStartTokenId,
            imageEndTokenId: imageEndTokenId
        )
    }

    public func encodeForGeneration(
        messages: [ChatMessage],
        tools: [ToolDefinition]? = nil,
        reasoningStrength: String,
        currentDate: String,
        maxLength: Int
    ) throws -> [Int] {
        let toolSpecs: [ToolSpec]? = tools?.isEmpty == false ? tools?.map { $0.toToolSpec() } : nil
        var encoded = try tokenizer.applyChatTemplate(
            messages: Self.renderMessages(messages),
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: toolSpecs,
            additionalContext: [
                "reasoning_strength": reasoningStrength,
                "current_date": currentDate,
            ]
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

    static func renderMessages(_ messages: [ChatMessage]) -> [Message] {
        messages.map { message in
            var rendered: Message = ["role": message.role.rawValue]
            if let image = message.imageUrl, !image.isEmpty {
                rendered["content"] = [
                    ["type": "image", "image_url": image],
                    ["type": "text", "text": message.content],
                ] as [[String: String]]
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
