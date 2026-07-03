import Foundation

public enum Gemma4TextSFTTokenizer {
    public static func tokenize(
        _ examples: [TextSFTExample],
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        maxSequenceLength: Int
    ) throws -> [TextSFTTokenizedExample] {
        try examples.map { example in
            guard let assistant = example.messages.last, assistant.role == .assistant else {
                throw Gemma4TextSFTTokenizerError.lastMessageMustBeAssistant
            }
            let prefixMessages = Array(example.messages.dropLast())
            let prefixTokens = try tokenizerAndTemplate.encodeForGeneration(
                messages: prefixMessages,
                addGenerationPrompt: true,
                includeThinking: false,
                maxLength: tokenizerAndTemplate.maxLength
            )

            var targetTokens = tokenizerAndTemplate.encodeRaw(
                assistantTargetText(assistant.content),
                addSpecialTokens: false
            )
            if let turnTokenId = tokenizerAndTemplate.turnTokenId {
                targetTokens.append(turnTokenId)
            }

            return try TextSFTTrainingBatchBuilder.shiftedTargetExample(
                prefixTokenIds: prefixTokens,
                targetTokenIds: targetTokens,
                maxSequenceLength: maxSequenceLength
            )
        }
    }

    static func assistantTargetText(_ content: String) -> String {
        stripThinking(from: content).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func stripThinking(from text: String) -> String {
        let marker = "<channel|>"
        guard text.contains(marker) else { return text }
        return text
            .components(separatedBy: marker)
            .map { part -> String in
                guard let range = part.range(of: "<|channel>") else { return part }
                return String(part[..<range.lowerBound])
            }
            .joined()
    }

    public static func messageTokenSpans(
        messages: [ChatMessage],
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        maxSequenceLength: Int
    ) throws -> [[Int]] {
        try legacyMessageTokenSpans(
            messages: messages,
            tokenizerAndTemplate: tokenizerAndTemplate,
            maxSequenceLength: maxSequenceLength
        )
    }

    static func legacyTokenize(
        _ examples: [TextSFTExample],
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        maxSequenceLength: Int
    ) throws -> [TextSFTTokenizedExample] {
        try examples.map { example in
            let spans = try legacyMessageTokenSpans(
                messages: example.messages,
                tokenizerAndTemplate: tokenizerAndTemplate,
                maxSequenceLength: maxSequenceLength
            )
            return try TextSFTTrainingBatchBuilder.shiftedExample(
                messageTokenIds: spans,
                messageRoles: example.messages.map(\.role),
                maxSequenceLength: maxSequenceLength
            )
        }
    }

    private static func legacyMessageTokenSpans(
        messages: [ChatMessage],
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        maxSequenceLength: Int
    ) throws -> [[Int]] {
        guard !messages.isEmpty else {
            throw Gemma4TextSFTTokenizerError.emptyMessages
        }

        var spans: [[Int]] = []
        var previous: [Int] = []
        for endIndex in messages.indices {
            let prefix = Array(messages[...endIndex])
            let encoded = try tokenizerAndTemplate.encodeForGeneration(
                messages: prefix,
                addGenerationPrompt: false,
                includeThinking: false,
                maxLength: tokenizerAndTemplate.maxLength
            )
            let common = commonPrefixLength(previous, encoded)
            let span = Array(encoded.dropFirst(common))
            guard !span.isEmpty else {
                throw Gemma4TextSFTTokenizerError.emptyMessageSpan(endIndex)
            }
            spans.append(span)
            previous = encoded
        }

        let totalCount = spans.reduce(0) { $0 + $1.count }
        guard totalCount >= 2 else {
            throw Gemma4TextSFTTokenizerError.tooFewTokens
        }
        _ = maxSequenceLength
        return spans
    }

    static func commonPrefixLength(_ lhs: [Int], _ rhs: [Int]) -> Int {
        let limit = min(lhs.count, rhs.count)
        var index = 0
        while index < limit, lhs[index] == rhs[index] {
            index += 1
        }
        return index
    }
}

public enum Gemma4TextSFTTokenizerError: Error, LocalizedError, Sendable {
    case emptyMessages
    case emptyMessageSpan(Int)
    case tooFewTokens
    case lastMessageMustBeAssistant

    public var errorDescription: String? {
        switch self {
        case .emptyMessages:
            return "Gemma4 text SFT tokenization requires at least one message."
        case .emptyMessageSpan(let index):
            return "Gemma4 text SFT tokenization produced no tokens for message index \(index)."
        case .tooFewTokens:
            return "Gemma4 text SFT tokenization produced fewer than two tokens."
        case .lastMessageMustBeAssistant:
            return "Gemma4 text SFT tokenization requires the last message to be assistant."
        }
    }
}
