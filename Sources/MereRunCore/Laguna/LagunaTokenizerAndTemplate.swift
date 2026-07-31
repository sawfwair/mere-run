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
            messages: messages.map(Self.renderMessage),
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

    private static func renderMessage(_ message: ChatMessage) -> Message {
        [
            "role": message.role.rawValue,
            "content": message.content,
        ]
    }
}
