import Foundation

public enum Gemma4VLMSFTTokenizer {
    public static func tokenize(
        _ examples: [TextSFTExample],
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        config: Gemma4Config,
        maxSequenceLength: Int,
        expectedImageDigestsByPath: [String: String] = [:]
    ) throws -> [TextSFTTokenizedExample] {
        guard let visionConfig = config.visionConfig,
              let imageTokenId = config.imageTokenId,
              let boiTokenId = config.boiTokenId,
              let eoiTokenId = config.eoiTokenId else {
            throw Gemma4VLMSFTTokenizerError.missingVisionConfiguration
        }

        return try examples.enumerated().map { exampleIndex, example in
            guard let assistant = example.messages.last, assistant.role == .assistant else {
                throw Gemma4VLMSFTTokenizerError.lastMessageMustBeAssistant(exampleIndex + 1)
            }
            let prefixMessages = Array(example.messages.dropLast())
            let imageReferences = prefixMessages.compactMap { message -> String? in
                guard let reference = message.imageUrl?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                    !reference.isEmpty else {
                    return nil
                }
                return reference
            }
            guard imageReferences.count == 1 else {
                throw Gemma4VLMSFTTokenizerError.singleImageRequired(exampleIndex + 1)
            }
            let imageSHA256 = try imageReferences.map { reference in
                let digest = try TextSFTDataset.fileDigest(
                    URL(fileURLWithPath: reference).standardizedFileURL
                )
                if !expectedImageDigestsByPath.isEmpty,
                   expectedImageDigestsByPath[reference] != digest {
                    throw Gemma4VLMSFTTokenizerError.imageChanged(exampleIndex + 1)
                }
                return digest
            }

            let softTokenCounts = try Gemma4UnifiedImageProcessor.softTokenCounts(
                imageReferences: imageReferences,
                visionConfig: visionConfig
            )
            let rawPrefixTokens = try tokenizerAndTemplate.encodeForGeneration(
                messages: prefixMessages,
                tools: example.tools,
                addGenerationPrompt: true,
                includeThinking: false,
                maxLength: tokenizerAndTemplate.maxLength
            )
            let expandedPrefixTokens = try Gemma4UnifiedImageProcessor.expandedPromptTokens(
                rawPrefixTokens,
                softTokenCounts: softTokenCounts,
                imageTokenId: imageTokenId,
                boiTokenId: boiTokenId,
                eoiTokenId: eoiTokenId
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
                    prefixTokenIds: rawPrefixTokens,
                    fullConversationTokenIds: fullTokens
                )
            } else {
                var contentTokens = tokenizerAndTemplate.encodeRaw(
                    Gemma4TextSFTTokenizer.assistantTargetText(assistant.content),
                    addSpecialTokens: false
                )
                if let turnTokenId = tokenizerAndTemplate.turnTokenId {
                    contentTokens.append(turnTokenId)
                }
                targetTokens = contentTokens
            }

            let tokenized = try TextSFTTrainingBatchBuilder.shiftedTargetExample(
                prefixTokenIds: expandedPrefixTokens,
                targetTokenIds: targetTokens,
                maxSequenceLength: maxSequenceLength
            )
            let expectedImageTokenCount = softTokenCounts.reduce(0, +)
            let actualImageTokenCount = tokenized.inputTokenIds.filter { $0 == imageTokenId }.count
            guard actualImageTokenCount == expectedImageTokenCount else {
                throw Gemma4VLMSFTTokenizerError.imageTruncated(
                    line: exampleIndex + 1,
                    maxSequenceLength: maxSequenceLength
                )
            }

            let mmTokenTypeIds = tokenized.inputTokenIds.map { token -> Int32 in
                token == imageTokenId ? 1 : 0
            }
            return TextSFTTokenizedExample(
                inputTokenIds: tokenized.inputTokenIds,
                labelTokenIds: tokenized.labelTokenIds,
                lossMask: tokenized.lossMask,
                multimodalInputs: TextSFTMultimodalInputs(
                    imageReferences: imageReferences,
                    imageSHA256: imageSHA256,
                    softTokenCounts: softTokenCounts,
                    mmTokenTypeIds: mmTokenTypeIds,
                    mmTokenTypeShape: [1, mmTokenTypeIds.count]
                )
            )
        }
    }
}

public enum Gemma4VLMSFTTokenizerError: Error, LocalizedError, Sendable {
    case missingVisionConfiguration
    case lastMessageMustBeAssistant(Int)
    case singleImageRequired(Int)
    case imageChanged(Int)
    case imageTruncated(line: Int, maxSequenceLength: Int)

    public var errorDescription: String? {
        switch self {
        case .missingVisionConfiguration:
            return "Gemma4 VLM SFT requires vision and image-token configuration."
        case .lastMessageMustBeAssistant(let line):
            return "Gemma4 VLM SFT example at line \(line) must end with an assistant message."
        case .singleImageRequired(let line):
            return "Gemma4 VLM SFT example at line \(line) must include exactly one image."
        case .imageChanged(let line):
            return "Gemma4 VLM SFT image at line \(line) changed after dataset validation."
        case .imageTruncated(let line, let maxSequenceLength):
            return "Gemma4 VLM SFT example at line \(line) loses image tokens at max sequence length \(maxSequenceLength); increase --max-sequence-length or shorten the conversation."
        }
    }
}
