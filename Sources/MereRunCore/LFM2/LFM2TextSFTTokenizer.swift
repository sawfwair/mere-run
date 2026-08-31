import Foundation

public enum LFM2TextSFTTokenizer {
    public static func tokenize(
        _ examples: [TextSFTExample],
        tokenizerAndTemplate: LFM2TokenizerAndTemplate,
        maxSequenceLength: Int
    ) throws -> [TextSFTTokenizedExample] {
        try examples.map { example in
            guard let assistant = example.messages.last,
                  assistant.role == .assistant else {
                throw LFM2TextSFTTokenizerError.lastMessageMustBeAssistant
            }
            guard assistant.toolCalls?.isEmpty != false else {
                throw LFM2TextSFTTokenizerError.assistantToolCallsUnsupported
            }
            let prefixTokens = try tokenizerAndTemplate.encodeForGeneration(
                messages: Array(example.messages.dropLast()),
                tools: example.tools,
                addGenerationPrompt: true,
                includeThinking: false,
                maxLength: tokenizerAndTemplate.maxLength
            )
            var targetTokens = tokenizerAndTemplate.encodeRaw(
                assistantTargetText(
                    content: assistant.content,
                    reasoningContent: assistant.reasoningContent,
                    generationPromptSuffix: tokenizerAndTemplate.generationPromptSuffix
                ),
                addSpecialTokens: false
            )
            if let imEndTokenId = tokenizerAndTemplate.imEndTokenId {
                targetTokens.append(imEndTokenId)
            } else if let eosTokenId = tokenizerAndTemplate.eosTokenId {
                targetTokens.append(eosTokenId)
            }
            return try TextSFTTrainingBatchBuilder.shiftedTargetExample(
                prefixTokenIds: prefixTokens,
                targetTokenIds: targetTokens,
                maxSequenceLength: maxSequenceLength
            )
        }
    }

    static func assistantTargetText(
        content: String,
        reasoningContent: String?,
        generationPromptSuffix: String
    ) -> String {
        var visible = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard generationPromptSuffix == "<think>" else { return visible }

        if visible.lowercased().hasPrefix("<think>") {
            visible = String(visible.dropFirst("<think>".count))
        }
        if visible.localizedCaseInsensitiveContains("</think>") {
            return visible
        }
        if let reasoning = reasoningContent?.trimmingCharacters(in: .whitespacesAndNewlines),
           !reasoning.isEmpty {
            return "\(reasoning)</think>\n\(visible)"
        }
        return "</think>\n\(visible)"
    }
}

public enum LFM2TextSFTTokenizerError: Error, LocalizedError, Sendable {
    case lastMessageMustBeAssistant
    case assistantToolCallsUnsupported

    public var errorDescription: String? {
        switch self {
        case .lastMessageMustBeAssistant:
            return "LFM2 text SFT tokenization requires the last message to be assistant."
        case .assistantToolCallsUnsupported:
            return "LFM2 text LoRA training v1 supports assistant text targets, not assistant tool-call targets."
        }
    }
}
