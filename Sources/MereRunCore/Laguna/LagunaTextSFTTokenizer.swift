import Foundation

public enum LagunaTextSFTTokenizer {
    public static func tokenize(
        _ examples: [TextSFTExample],
        tokenizerAndTemplate: LagunaTokenizerAndTemplate,
        maxSequenceLength: Int
    ) throws -> [TextSFTTokenizedExample] {
        try examples.map { example in
            guard let assistant = example.messages.last,
                  assistant.role == .assistant else {
                throw LagunaTextSFTTokenizerError.lastMessageMustBeAssistant
            }
            let prefixTokens = try tokenizerAndTemplate.encodeForGeneration(
                messages: Array(example.messages.dropLast()),
                tools: example.tools,
                addGenerationPrompt: true,
                includeThinking: false,
                maxLength: tokenizerAndTemplate.maxLength
            )
            let targetTokens: [Int]
            if assistant.toolCalls?.isEmpty == false {
                let fullTokens = try tokenizerAndTemplate.encodeForGeneration(
                    messages: example.messages,
                    tools: example.tools,
                    addGenerationPrompt: false,
                    includeThinking: false,
                    maxLength: tokenizerAndTemplate.maxLength
                )
                targetTokens = try TextSFTTrainingBatchBuilder.nativeAssistantTarget(
                    prefixTokenIds: prefixTokens,
                    fullConversationTokenIds: fullTokens
                )
            } else {
                var contentTokens = tokenizerAndTemplate.encodeRaw(
                    assistantTargetText(assistant.content),
                    addSpecialTokens: false
                )
                if let assistantEndTokenID = tokenizerAndTemplate.assistantEndTokenID {
                    contentTokens.append(assistantEndTokenID)
                } else if let eosTokenID = tokenizerAndTemplate.eosTokenID {
                    contentTokens.append(eosTokenID)
                }
                targetTokens = contentTokens
            }
            return try TextSFTTrainingBatchBuilder.shiftedTargetExample(
                prefixTokenIds: prefixTokens,
                targetTokenIds: targetTokens,
                maxSequenceLength: maxSequenceLength
            )
        }
    }

    static func assistantTargetText(_ content: String) -> String {
        content.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

public enum LagunaTextSFTTokenizerError: Error, LocalizedError, Sendable {
    case lastMessageMustBeAssistant

    public var errorDescription: String? {
        switch self {
        case .lastMessageMustBeAssistant:
            return "Laguna text SFT tokenization requires the last message to be assistant."
        }
    }
}
