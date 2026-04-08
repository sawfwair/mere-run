import Foundation
@preconcurrency import Hub
@preconcurrency import Tokenizers

public final class Gemma4TokenizerAndTemplate {
    public let tokenizer: any Tokenizer
    public let maxLength: Int
    public let eosTokenId: Int?
    public let turnTokenId: Int?

    public init(
        tokenizer: any Tokenizer,
        maxLength: Int,
        eosTokenId: Int?,
        turnTokenId: Int?
    ) {
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.eosTokenId = eosTokenId
        self.turnTokenId = turnTokenId
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
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let configured = object["model_max_length"] as? NSNumber {
            let raw = configured.intValue
            maxLength = maxLengthOverride ?? (raw > 0 ? raw : 131_072)
        } else {
            maxLength = maxLengthOverride ?? 131_072
        }

        return Gemma4TokenizerAndTemplate(
            tokenizer: tokenizer,
            maxLength: maxLength,
            eosTokenId: tokenizer.eosTokenId,
            turnTokenId: tokenizer.convertTokenToId("<turn|>")
        )
    }

    public func encodeForGeneration(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true,
        includeThinking: Bool,
        maxLength: Int
    ) throws -> [Int] {
        let renderedMessages: [Message] = messages.map { message in
            var rendered: Message = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
            if let imageURL = message.imageUrl, !imageURL.isEmpty {
                rendered["content"] = [
                    ["type": "text", "text": message.content],
                    ["type": "image", "image_url": imageURL],
                ]
            }
            return rendered
        }

        var encoded = try tokenizer.applyChatTemplate(
            messages: renderedMessages,
            chatTemplate: nil,
            addGenerationPrompt: addGenerationPrompt,
            truncation: false,
            maxLength: nil,
            tools: nil,
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
}
