import Foundation
@preconcurrency import Hub
@preconcurrency import Tokenizers

public final class LFM2TokenizerAndTemplate: @unchecked Sendable {
    public let tokenizer: any Tokenizer
    public let maxLength: Int
    public let eosTokenId: Int?
    public let imEndTokenId: Int?
    public let toolCallEndTokenId: Int?
    public let generationPromptSuffix: String

    public init(
        tokenizer: any Tokenizer,
        maxLength: Int,
        eosTokenId: Int?,
        imEndTokenId: Int?,
        toolCallEndTokenId: Int?,
        generationPromptSuffix: String
    ) {
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.eosTokenId = eosTokenId
        self.imEndTokenId = imEndTokenId
        self.toolCallEndTokenId = toolCallEndTokenId
        self.generationPromptSuffix = generationPromptSuffix
    }

    public static func load(
        from rootURL: URL,
        maxLengthOverride: Int? = nil,
        hubApi: HubApi = .shared
    ) async throws -> LFM2TokenizerAndTemplate {
        let tokenizer = try await AutoTokenizer.from(modelFolder: rootURL, hubApi: hubApi)
        let chatTemplateURL = rootURL.appendingPathComponent("chat_template.jinja")
        let chatTemplate = try? String(contentsOf: chatTemplateURL, encoding: .utf8)
        let generationPromptSuffix = chatTemplate?.contains("<|im_start|>assistant\\n<think>") == true
            ? "<think>"
            : ""
        return LFM2TokenizerAndTemplate(
            tokenizer: tokenizer,
            maxLength: maxLengthOverride ?? LFM2Resources.defaultContextLength,
            eosTokenId: tokenizer.eosTokenId,
            imEndTokenId: tokenizer.convertTokenToId("<|im_end|>"),
            toolCallEndTokenId: tokenizer.convertTokenToId("<|tool_call_end|>"),
            generationPromptSuffix: generationPromptSuffix
        )
    }

    public func encodeForGeneration(
        messages: [ChatMessage],
        tools: [ToolDefinition]? = nil,
        addGenerationPrompt: Bool = true,
        includeThinking: Bool,
        maxLength: Int
    ) throws -> [Int] {
        var encoded = tokenizer.encode(
            text: try Self.renderPrompt(
                messages: messages,
                tools: tools,
                addGenerationPrompt: addGenerationPrompt,
                includeThinking: includeThinking,
                generationPromptSuffix: generationPromptSuffix
            ),
            addSpecialTokens: false
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

    public func stopTokenIds(withTools: Bool) -> [Int] {
        var ids = [eosTokenId, imEndTokenId].compactMap { $0 }
        if withTools, let toolCallEndTokenId {
            ids.append(toolCallEndTokenId)
        }
        return Array(Set(ids))
    }

    static func renderMessages(_ messages: [ChatMessage]) -> [Message] {
        messages.map { message in
            var rendered: Message = [
                "role": message.role.rawValue,
            ]
            if let imageURL = message.imageUrl, !imageURL.isEmpty {
                rendered["content"] = [
                    ["type": "text", "text": message.content],
                    ["type": "image", "image": imageURL],
                ] as [[String: String]]
            } else {
                rendered["content"] = message.content
            }
            return rendered
        }
    }

    public static func renderPrompt(
        messages: [ChatMessage],
        tools: [ToolDefinition]? = nil,
        addGenerationPrompt: Bool = true,
        includeThinking: Bool,
        generationPromptSuffix: String = ""
    ) throws -> String {
        var prompt = "<|startoftext|>"
        var remaining = messages
        var systemPrompt = ""

        if let first = remaining.first, first.role == .system {
            systemPrompt = renderContent(for: first)
            remaining.removeFirst()
        }

        if let tools, !tools.isEmpty {
            if !systemPrompt.isEmpty {
                systemPrompt += "\n"
            }
            systemPrompt += "List of tools: ["
            systemPrompt += try tools.map { try $0.promptSchemaJSONString() }.joined(separator: ", ")
            systemPrompt += "]"
        }

        if !systemPrompt.isEmpty {
            prompt += "<|im_start|>system\n"
            prompt += systemPrompt
            prompt += "<|im_end|>\n"
        }

        let lastUserIndex = remaining.lastIndex { $0.role == .user } ?? -1

        for (index, message) in remaining.enumerated() {
            prompt += "<|im_start|>\(message.role.rawValue)\n"

            if message.role == .assistant {
                var content = renderContent(for: message)
                if !includeThinking, index <= lastUserIndex, let range = content.range(of: "</think>") {
                    content = String(content[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                }
                prompt += content
            } else {
                prompt += renderContent(for: message)
            }

            prompt += "<|im_end|>\n"
        }

        if addGenerationPrompt {
            prompt += "<|im_start|>assistant\n"
            prompt += generationPromptSuffix
        }

        return prompt
    }

    private static func renderContent(for message: ChatMessage) -> String {
        if let imageURL = message.imageUrl, !imageURL.isEmpty {
            return "\(message.content)<image>"
        }
        return message.content
    }
}
