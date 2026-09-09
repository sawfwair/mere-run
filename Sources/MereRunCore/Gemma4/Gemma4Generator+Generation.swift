import Foundation
import MLX
import MLXNN

extension Gemma4Generator {
    func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        maxContextLength: Int
    ) async throws -> ChatResponse {
        guard let model, let tokenizerAndTemplate, let loadedConfig else {
            throw Gemma4Error.modelNotLoaded
        }
        let fallbackKVCacheQuantization = try self.kvCacheQuantization.validated()

        let requestedContextLength = request.maxContextTokens ?? maxContextLength
        guard requestedContextLength > 0 else {
            throw Gemma4Error.unsupportedConfiguration("maxContextTokens must be greater than zero.")
        }
        if let noRepeatNgramSize = request.noRepeatNgramSize, noRepeatNgramSize < 1 {
            throw Gemma4Error.unsupportedConfiguration("noRepeatNgramSize must be greater than zero.")
        }
        let effectiveContext = min(
            maxContextLength,
            requestedContextLength,
            loadedConfig.textConfig.maxPositionEmbeddings
        )
        let prefillStart = Date()
        let messages = request.messages
        let imageReferences = messages.compactMap { message -> String? in
            guard let imageURL = message.imageUrl?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !imageURL.isEmpty else {
                return nil
            }
            return imageURL
        }
        let imageBatch: Gemma4UnifiedImageBatch?
        if imageReferences.isEmpty {
            imageBatch = nil
        } else {
            guard let visionConfig = loadedConfig.visionConfig,
                  loadedConfig.imageTokenId != nil,
                  loadedConfig.boiTokenId != nil,
                  loadedConfig.eoiTokenId != nil else {
                throw Gemma4Error.unsupportedConfiguration("This Gemma4 model does not support image inputs.")
            }
            guard model is Gemma4UnifiedCausalLM else {
                throw Gemma4Error.unsupportedConfiguration("Image inputs require \(Gemma4Resources.visionTwelveBModelId).")
            }
            imageBatch = try Gemma4UnifiedImageProcessor.makeBatch(
                imageReferences: imageReferences,
                visionConfig: visionConfig
            )
        }
        var promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: messages,
            tools: request.tools,
            addGenerationPrompt: true,
            includeThinking: request.showThinking,
            maxLength: effectiveContext
        )
        if let imageBatch {
            guard let imageTokenId = loadedConfig.imageTokenId,
                  let boiTokenId = loadedConfig.boiTokenId,
                  let eoiTokenId = loadedConfig.eoiTokenId else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 unified prompt expansion requires image token IDs.")
            }
            promptTokens = try Gemma4UnifiedImageProcessor.expandedPromptTokens(
                promptTokens,
                softTokenCounts: imageBatch.softTokenCounts,
                imageTokenId: imageTokenId,
                boiTokenId: boiTokenId,
                eoiTokenId: eoiTokenId
            )
        }
        if promptTokens.count > effectiveContext {
            promptTokens = Array(promptTokens.suffix(effectiveContext))
        }
        let effectiveKVCacheMode = request.kvCacheMode ?? .default
        let kvCacheQuantization = try effectiveKVCacheMode.gemma4Quantization(
            fallback: fallbackKVCacheQuantization,
            promptTokenCount: promptTokens.count
        ).validated()
        let prefillKVCacheQuantization = prefillQuantization(for: kvCacheQuantization)

        let hasTools = request.tools?.isEmpty == false
        let eosSet = request.stopOnEOS
            ? Set(loadedConfig.eosTokenIds + tokenizerAndTemplate.stopTokenIds(withTools: hasTools))
            : []
        var generationConfig = GenerationConfig(
            maxTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topP: Float(request.topP),
            minP: Float(request.minP),
            repetitionPenalty: 1.05,
            repetitionContextSize: 64
        )
        // Multimodal marker IDs are prompt-only control tokens. They can appear
        // in VLM prefill, but assistant decode must never emit them as content.
        generationConfig.bannedTokens = Self.multimodalDecodeBannedTokens(
            imageTokenId: loadedConfig.imageTokenId,
            audioTokenId: loadedConfig.audioTokenId,
            videoTokenId: loadedConfig.videoTokenId,
            boiTokenId: loadedConfig.boiTokenId,
            boaTokenId: loadedConfig.boaTokenId,
            eoiTokenId: loadedConfig.eoiTokenId,
            eoaTokenId: loadedConfig.eoaTokenId,
            excluding: eosSet
        )

        let usePrefixKVCache = imageBatch == nil
        let prefixSeed = usePrefixKVCache ? prefixKVCacheSeed(
            modelPath: loadedModelPath ?? "",
            quantization: kvCacheQuantization,
            promptTokens: promptTokens
        ) : nil
        let prefixCheckpoints = usePrefixKVCache ? semanticPrefixCheckpoints(
            tokenizerAndTemplate: tokenizerAndTemplate,
            messages: messages,
            tools: request.tools,
            includeThinking: request.showThinking,
            promptTokens: promptTokens,
            maxContextLength: effectiveContext
        ) : []
        var mtpReason = Gemma4MTPPolicy.activationReason(
            assistant: mtpModel,
            promptTokenCount: promptTokens.count,
            generationConfig: generationConfig,
            prefixSeedWasUsed: prefixSeed != nil
        )
        if continuousBatchingEnabled, mtpReason == nil {
            mtpReason = "continuous batching"
        }
        if request.requiresJSON, mtpReason == nil {
            // Speculative drafting verifies tokens against the unconstrained
            // distribution; JSON-constrained decoding must stay on the serial path.
            mtpReason = "json constrained decoding"
        }
        if request.noRepeatNgramSize != nil, mtpReason == nil {
            mtpReason = "no-repeat n-gram decoding"
        }
        let useMTP = mtpReason == nil
        let layerCaches = try prefixSeed?.caches ?? makeLayerCaches(
            model: model,
            quantization: prefillKVCacheQuantization
        )
        let prefillResult: Gemma4PrefillResult
        if let imageBatch {
            guard let unifiedModel = model as? Gemma4UnifiedCausalLM,
                  let imageTokenId = loadedConfig.imageTokenId else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 unified prefill requires a unified runtime model.")
            }
            prefillResult = try await unifiedPrefill(
                model: unifiedModel,
                promptTokens: promptTokens,
                imageBatch: imageBatch,
                imageTokenId: imageTokenId,
                cache: layerCaches,
                captureSpeculation: useMTP,
                progressHandler: progressHandler
            )
        } else {
            prefillResult = try await chunkedPrefill(
                model: model,
                promptTokens: promptTokens,
                cache: layerCaches,
                startIndex: prefixSeed?.tokenCount ?? 0,
                existingLogits: prefixSeed?.logits,
                modelPath: loadedModelPath ?? "",
                quantization: kvCacheQuantization,
                checkpointTokenCounts: prefixCheckpoints,
                captureSpeculation: useMTP,
                progressHandler: progressHandler
            )
        }

        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        let preparedCaches = try prepareLayerCachesForDecode(
            layerCaches,
            quantization: kvCacheQuantization,
            progressHandler: progressHandler
        )
        let tokenBudget = max(0, min(request.maxTokens, effectiveContext - promptTokens.count))
        let mtpSharedKVStates = useMTP
            ? collectSharedKVStatesForMTP(config: loadedConfig.textConfig, caches: preparedCaches.caches)
            : [:]
        if useMTP, mtpSharedKVStates.isEmpty {
            mtpReason = "shared KV unavailable"
        }
        let mtpTemplate = Gemma4MTPStats(
            available: mtpModel != nil,
            enabled: Gemma4MTPPolicy.enabled(),
            active: useMTP && mtpReason == nil,
            assistantModelPath: loadedMTPModelPath,
            reason: mtpReason,
            blockSize: mtpModel.map { Gemma4MTPPolicy.blockSize(configured: $0.config.blockSize) } ?? 0,
            threshold: Gemma4MTPPolicy.promptThreshold()
        )
        lastMTPStats = mtpTemplate

        progressHandler?(ChatProgress(stage: .generating, message: ""))

        let decodeResult = try await decodeTokens(
            model: model,
            tokenizerAndTemplate: tokenizerAndTemplate,
            initialLogits: prefillResult.logits,
            layerCaches: preparedCaches.caches,
            eosSet: eosSet,
            generationConfig: generationConfig,
            tokenBudget: tokenBudget,
            promptTokens: promptTokens,
            mtpModel: mtpTemplate.active ? mtpModel : nil,
            prefillHidden: mtpTemplate.active ? prefillResult.hidden : nil,
            sharedKVStates: mtpTemplate.active ? mtpSharedKVStates : [:],
            prefillTokenCount: promptTokens.count,
            mtpStatsTemplate: mtpTemplate,
            jsonConstrained: request.requiresJSON,
            noRepeatNgramSize: request.noRepeatNgramSize,
            progressHandler: progressHandler
        )
        lastMTPStats = decodeResult.mtpStats ?? mtpTemplate
        let generated = decodeResult.generatedTokens
        let decodedRaw = tokenizerAndTemplate.decode(tokens: generated)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let decoded = Self.cleanedResponse(
            decodedRaw,
            showThinking: request.showThinking
        )
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let reasoningSplit = ChatReasoningMarkup.splitThinkBlocks(in: decodedRaw)

        let toolCalls: [ToolCall]? = hasTools ? {
            let parsed = Gemma4ToolParser.parseToolCalls(decoded)
            return parsed.isEmpty ? nil : parsed
        }() : nil

        return ChatResponse(
            response: decoded,
            tokensGenerated: generated.count,
            timing: ChatTiming(
                loadSeconds: 0,
                prefillSeconds: prefillSeconds,
                cacheConversionSeconds: preparedCaches.conversionSeconds,
                decodeSeconds: decodeResult.decodeSeconds,
                firstTokenSeconds: decodeResult.firstTokenSeconds,
                kvCacheMode: effectiveKVCacheMode,
                prefillKVCache: prefillKVCacheQuantization.statusDescription,
                decodeKVCache: kvCacheQuantization.statusDescription,
                prefillTokensPerSecond: prefillSeconds > 0
                    ? Double(max(0, promptTokens.count - (prefixSeed?.tokenCount ?? 0))) / prefillSeconds
                    : nil,
                decodeTokensPerSecond: decodeResult.decodeSeconds > 0
                    ? Double(generated.count) / decodeResult.decodeSeconds
                    : nil
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count,
            reasoningContent: reasoningSplit.reasoningContent,
            hasIncompleteReasoning: reasoningSplit.hasIncompleteReasoning
        )
    }
}
