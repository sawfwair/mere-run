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
                addGenerationPrompt: true,
                includeThinking: false,
                maxLength: tokenizerAndTemplate.maxLength
            )
            var targetTokens = tokenizerAndTemplate.encodeRaw(
                assistantTargetText(assistant.content),
                addSpecialTokens: false
            )
            if let assistantEndTokenID = tokenizerAndTemplate.assistantEndTokenID {
                targetTokens.append(assistantEndTokenID)
            } else if let eosTokenID = tokenizerAndTemplate.eosTokenID {
                targetTokens.append(eosTokenID)
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
