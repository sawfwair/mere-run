import Foundation

public enum InklingTextSFTTokenizer {
    public static func tokenize(
        _ examples: [TextSFTExample],
        tokenizerAndTemplate: InklingTokenizerAndTemplate,
        maxSequenceLength: Int,
        reasoningEffort: Double = 0.9
    ) throws -> [TextSFTTokenizedExample] {
        try examples.map { example in
            guard let assistant = example.messages.last,
                  assistant.role == .assistant else {
                throw InklingTextSFTTokenizerError.lastMessageMustBeAssistant
            }
            let prefixTokens = try tokenizerAndTemplate.encodeForGeneration(
                messages: Array(example.messages.dropLast()),
                tools: example.tools,
                addGenerationPrompt: true,
                reasoningEffort: reasoningEffort,
                maxLength: tokenizerAndTemplate.maxLength
            )
            let targetTokens: [Int]
            if assistant.toolCalls?.isEmpty == false {
                let fullTokens = try tokenizerAndTemplate.encodeForGeneration(
                    messages: example.messages,
                    tools: example.tools,
                    addGenerationPrompt: false,
                    reasoningEffort: reasoningEffort,
                    maxLength: tokenizerAndTemplate.maxLength
                )
                targetTokens = try TextSFTTrainingBatchBuilder.nativeAssistantTarget(
                    prefixTokenIds: prefixTokens,
                    fullConversationTokenIds: fullTokens
                )
            } else {
                targetTokens = tokenizerAndTemplate.encodeRaw(
                    assistantTargetText(assistant.content),
                    addSpecialTokens: false
                )
            }
            return try TextSFTTrainingBatchBuilder.shiftedTargetExample(
                prefixTokenIds: prefixTokens,
                targetTokenIds: targetTokens,
                maxSequenceLength: maxSequenceLength
            )
        }
    }

    static func assistantTargetText(_ content: String) -> String {
        let visible = content.trimmingCharacters(in: .whitespacesAndNewlines)
        return "<|content_text|>\(visible)<|end_message|><|content_model_end_sampling|>"
    }
}

public enum InklingTextSFTTokenizerError: Error, LocalizedError, Sendable {
    case lastMessageMustBeAssistant

    public var errorDescription: String? {
        switch self {
        case .lastMessageMustBeAssistant:
            return "Inkling text SFT tokenization requires the last message to be assistant."
        }
    }
}
