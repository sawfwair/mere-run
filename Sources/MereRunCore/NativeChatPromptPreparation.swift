import Foundation

extension Q35Generator {
    public func preparePrompt(_ request: ChatRequest, modelPath: String?) async throws -> ChatPreparedPrompt {
        guard request.messages.allSatisfy({ $0.imageUrl == nil && $0.audioUrl == nil && $0.videoUrl == nil }) else {
            throw ChatRequestIssue("count_tokens", "Qwen media expansion is not qualified for Messages counting")
        }
        try await prepare(modelPath: modelPath)
        guard let tokenizerAndTemplate, let loadedConfig else { throw Q35Error.modelNotLoaded }
        let context = min(request.maxContextTokens ?? loadedConfig.textConfig.maxPositionEmbeddings,
                          loadedConfig.textConfig.maxPositionEmbeddings, tokenizerAndTemplate.tokenizer.maxLength)
        let tokens = try tokenizerAndTemplate.encodeForGeneration(messages: NativeChatToolPolicy.requestMessages(request), tools: request.tools,
            includeThinking: request.showThinking && !request.requiresJSON,
            reasoningEffort: Q35Resources.isQ38ModelId(modelId) ? Q35Resources.q38ReasoningEffortLabel(for: request.reasoningEffort) : nil,
            maxLength: context, truncate: false)
        return try ChatPreparedPrompt(tokens: tokens, contextLimit: context)
    }
}

extension Gemma4Generator {
    public func preparePrompt(_ request: ChatRequest, modelPath: String?) async throws -> ChatPreparedPrompt {
        guard request.messages.allSatisfy({ $0.imageUrl == nil && $0.audioUrl == nil && $0.videoUrl == nil }) else {
            throw ChatRequestIssue("count_tokens", "Gemma media expansion is not qualified for Messages counting")
        }
        try await prepare(modelPath: modelPath)
        guard let tokenizerAndTemplate, let loadedConfig else { throw Gemma4Error.modelNotLoaded }
        let context = min(request.maxContextTokens ?? loadedConfig.textConfig.maxPositionEmbeddings,
                          loadedConfig.textConfig.maxPositionEmbeddings, tokenizerAndTemplate.maxLength)
        let tokens = try tokenizerAndTemplate.encodeForGeneration(messages: NativeChatToolPolicy.requestMessages(request), tools: request.tools,
            includeThinking: request.showThinking && !request.requiresJSON, maxLength: context, truncate: false)
        return try ChatPreparedPrompt(tokens: tokens, contextLimit: context)
    }
}
