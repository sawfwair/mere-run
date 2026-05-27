import Foundation
import MLX

extension Flux2KleinGenerator {

    // MARK: - Chat Generation

    public func prepareChat(
        modelPath: String,
        standalone: Bool,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading model"))
        if loadedModelPath == modelPath, textEncoder != nil, tokenizer != nil {
            return
        }
        if standalone {
            try await loadStandaloneChatModel(from: modelPath, progressHandler: progressHandler)
        } else {
            try await loadTextEncoderOnly(from: modelPath, progressHandler: progressHandler)
        }
    }


    public func chat(
        _ request: ChatRequest,
        modelPath: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        let debugLog = MereRunRuntimeDebug.logger(keys: ["MERERUN_FLUX2_DEBUG"], prefix: "[Flux2KleinGenerator]")
        debugLog?("chat: starting")
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading model"))

        // Load model if needed (only text encoder + tokenizer for chat)
        if loadedModelPath != modelPath {
            debugLog?("chat: loading model from \(modelPath)")
            try await loadTextEncoderOnly(from: modelPath, progressHandler: progressHandler)
        }
        debugLog?("chat: model loaded")

        guard let textEncoder = textEncoder,
              let tokenizer = tokenizer else {
            throw Flux2Error.modelsNotLoaded
        }

        // Apply text LoRA (game cartridge) if specified
        try await applyTextLoRAIfNeeded(request.lora, modelPath: modelPath, progressHandler: progressHandler)

        debugLog?("chat: encoding messages")
        progressHandler?(ChatProgress(stage: .encoding, message: "Encoding messages"))

        // Convert ChatMessage array to tokenizer format
        let messages: [[String: Any]] = request.messages.map { msg in
            var dict: [String: Any] = [
                "role": msg.role.rawValue,
                "content": msg.content
            ]
            if let imageUrl = msg.imageUrl {
                dict["image_url"] = imageUrl
            }
            return dict
        }

        // Encode messages for generation
        debugLog?("chat: tokenizing \(messages.count) messages")
        let tokens = try tokenizer.encodeChatForGeneration(
            messages: messages,
            maxLength: tokenizer.maxLength
        )
        debugLog?("chat: got \(tokens.count) tokens")
        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)

        debugLog?("chat: starting generation")
        progressHandler?(ChatProgress(stage: .generating, message: "Generating response"))

        // Configure generation
        let config = PromptEnhanceConfig(
            maxNewTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topP: 0.9,
            repetitionPenalty: 1.05,
            repetitionContextSize: 20,
            eosTokenId: tokenizer.eosTokenId ?? 151645,
            stopTokenIds: Set([tokenizer.eosTokenId ?? 151645, 151643])
        )

        // Retry loop for JSON validation
        let maxRetries = request.requiresJSON ? 3 : 1
        var lastResponse: String = ""

        for attempt in 1...maxRetries {
        // Generate response
        debugLog?("chat: calling generate with maxTokens=\(request.maxTokens) (attempt \(attempt)/\(maxRetries))")
        lastResponse = ""
        let generatedTokens = textEncoder.generate(inputIds: inputIds, config: config) { token in
            let piece = tokenizer.decode(tokens: [token])
            if !piece.isEmpty {
                lastResponse += piece
                progressHandler?(ChatProgress(stage: .generating, message: piece))
            }
            return true
        }
        debugLog?("chat: generated \(generatedTokens.count) tokens")

        // Preserve previous behavior of normalizing response whitespace.
        lastResponse = lastResponse.trimmingCharacters(in: .whitespacesAndNewlines)
            debugLog?("chat: decoded response length=\(lastResponse.count)")
            debugLog?("chat: response=\(lastResponse)")

            // If JSON is required, validate it
            if request.requiresJSON {
                if isValidJSON(lastResponse) {
                    debugLog?("chat: JSON validation passed")
                    return ChatResponse(
                        response: lastResponse,
                        tokensGenerated: generatedTokens.count
                    )
                } else {
                    debugLog?("chat: JSON validation failed, attempt \(attempt)/\(maxRetries)")
                    if attempt < maxRetries {
                        progressHandler?(ChatProgress(stage: .generating, message: "Retrying generation"))
                    }
                }
            } else {
                return ChatResponse(
                    response: lastResponse,
                    tokensGenerated: generatedTokens.count
                )
            }
        }

        // All retries exhausted - return last response anyway
        debugLog?("chat: JSON validation failed after \(maxRetries) attempts, returning last response")
        return ChatResponse(
            response: lastResponse,
            tokensGenerated: 0
        )
    }

    /// Chat using a standalone Qwen3-Instruct model (no FLUX text encoder).
    public func chatStandalone(
        _ request: ChatRequest,
        modelPath: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        let debugLog = MereRunRuntimeDebug.logger(keys: ["MERERUN_FLUX2_DEBUG"], prefix: "[Flux2KleinGenerator]")
        debugLog?("chatStandalone: starting")
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading model"))

        if loadedModelPath != modelPath {
            debugLog?("chatStandalone: loading standalone model from \(modelPath)")
            try await loadStandaloneChatModel(from: modelPath, progressHandler: progressHandler)
        }

        guard let textEncoder = textEncoder,
              let tokenizer = tokenizer else {
            throw Flux2Error.modelsNotLoaded
        }

        try await applyTextLoRAIfNeeded(request.lora, modelPath: modelPath, progressHandler: progressHandler)

        debugLog?("chatStandalone: encoding messages")
        progressHandler?(ChatProgress(stage: .encoding, message: "Encoding messages"))

        let messages: [[String: Any]] = request.messages.map { msg in
            var dict: [String: Any] = [
                "role": msg.role.rawValue,
                "content": msg.content
            ]
            if let imageUrl = msg.imageUrl {
                dict["image_url"] = imageUrl
            }
            return dict
        }

        let tokens = try tokenizer.encodeChatForGeneration(
            messages: messages,
            maxLength: tokenizer.maxLength
        )
        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)

        progressHandler?(ChatProgress(stage: .generating, message: "Generating response"))

        let config = PromptEnhanceConfig(
            maxNewTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topP: 0.9,
            repetitionPenalty: 1.05,
            repetitionContextSize: 20,
            eosTokenId: tokenizer.eosTokenId ?? 151645,
            stopTokenIds: Set([tokenizer.eosTokenId ?? 151645, 151643])
        )

        let maxRetries = request.requiresJSON ? 3 : 1
        var lastResponse: String = ""

        for attempt in 1...maxRetries {
            lastResponse = ""
            let generatedTokens = textEncoder.generate(inputIds: inputIds, config: config) { token in
                let piece = tokenizer.decode(tokens: [token])
                if !piece.isEmpty {
                    lastResponse += piece
                    progressHandler?(ChatProgress(stage: .generating, message: piece))
                }
                return true
            }

            lastResponse = lastResponse.trimmingCharacters(in: .whitespacesAndNewlines)

            if request.requiresJSON {
                if isValidJSON(lastResponse) {
                    return ChatResponse(
                        response: lastResponse,
                        tokensGenerated: generatedTokens.count
                    )
                } else if attempt < maxRetries {
                    progressHandler?(ChatProgress(stage: .generating, message: "Retrying generation"))
                }
            } else {
                return ChatResponse(
                    response: lastResponse,
                    tokensGenerated: generatedTokens.count
                )
            }
        }

        return ChatResponse(
            response: lastResponse,
            tokensGenerated: 0
        )
    }

    /// Check if the response is valid JSON (possibly wrapped in markdown code blocks)
    private func isValidJSON(_ response: String) -> Bool {
        guard let jsonString = extractJSON(from: response),
              let data = jsonString.data(using: .utf8) else {
            return false
        }

        do {
            _ = try JSONSerialization.jsonObject(with: data)
            return true
        } catch {
            return false
        }
    }

    /// Extract JSON from response, handling markdown blocks, think tags, and template artifacts
    private func extractJSON(from response: String) -> String? {
        var text = response.trimmingCharacters(in: .whitespacesAndNewlines)

        // Remove markdown code blocks if present
        if text.hasPrefix("```json") {
            text = String(text.dropFirst(7))
        } else if text.hasPrefix("```") {
            text = String(text.dropFirst(3))
        }
        if text.hasSuffix("```") {
            text = String(text.dropLast(3))
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Handle <think>...</think> tags from Qwen3
        if let thinkEnd = text.range(of: "</think>") {
            text = String(text[thinkEnd.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Strip "assistant" prefix if model leaked template
        if text.hasPrefix("assistant") {
            text = String(text.dropFirst(9)).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Find first '{' - skip any garbage before JSON
        guard let jsonStart = text.firstIndex(of: "{") else {
            return nil
        }
        text = String(text[jsonStart...])

        return text
    }


}
