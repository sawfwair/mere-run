import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

extension Q35Generator {
    func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        maxContextLength: Int
    ) async throws -> ChatResponse {
        try await Q35Sampling.withRequestState(seed: request.seed) {
            try await generateWithRequestState(
                request, progressHandler: progressHandler, maxContextLength: maxContextLength
            )
        }
    }

    private func generateWithRequestState(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        maxContextLength: Int
    ) async throws -> ChatResponse {
        guard let model,
              let tokenizerAndTemplate,
              let loadedConfig else {
            throw Q35Error.modelNotLoaded
        }
        // An exact prefix-cache hit skips the prefill loop entirely. Reclaim
        // disposable buffers from the previous request before forking its KV.
        clearMLXCacheUnderPressureIfNeeded()

        let messages = request.messages
        let jsonConstrained = request.requiresJSON
        let includeThinking = request.showThinking && !jsonConstrained
        let nativeReasoningEffort = Q35Resources.isQ38ModelId(modelId)
            ? Q35Resources.q38ReasoningEffortLabel(for: request.reasoningEffort)
            : nil
        let requestedContextLength = request.maxContextTokens ?? maxContextLength
        guard requestedContextLength > 0 else {
            throw Q35Error.generationFailed("maxContextTokens must be greater than zero.")
        }
        let effectiveContext = min(
            maxContextLength,
            requestedContextLength,
            loadedConfig.textConfig.maxPositionEmbeddings
        )
        let prefillStart = Date()
        let imageURLs = collectImageURLs(from: messages)
        var visionReplacements: [Q35VisionReplacement] = []
        var visionTokenLimitPerImage: Int?

        if !imageURLs.isEmpty {
            progressHandler?(ChatProgress(stage: .encoding, message: "Encoding images"))
            guard visionTower != nil else {
                throw Q35Error.generationFailed("Model \(modelId) does not include a vision tower; use text-only prompts.")
            }
            guard let imageTokenId = loadedConfig.imageTokenId ?? tokenizerAndTemplate.tokenizer.imageTokenId else {
                throw Q35Error.generationFailed("Qwen-family tokenizer is missing the image placeholder token.")
            }
            let promptWithSingleImageTokens = try tokenizerAndTemplate.encodeForGeneration(
                messages: messages,
                tools: request.tools,
                addGenerationPrompt: true,
                includeThinking: includeThinking,
                reasoningEffort: nativeReasoningEffort,
                maxLength: loadedConfig.textConfig.maxPositionEmbeddings,
                imageTokenCounts: Array(repeating: 1, count: imageURLs.count)
            )
            let placeholderCount = promptWithSingleImageTokens.filter { $0 == imageTokenId }.count
            guard placeholderCount == imageURLs.count else {
                throw Q35Error.generationFailed(
                    "Qwen-family prompt did not preserve one placeholder for each encoded image."
                )
            }
            let nonVisionPromptTokenCount = promptWithSingleImageTokens.count - placeholderCount
            guard let perImageLimit = Self.visionTokenLimitPerImage(
                contextLength: effectiveContext,
                generationTokenCount: request.maxTokens,
                nonVisionPromptTokenCount: nonVisionPromptTokenCount,
                imageCount: imageURLs.count
            ) else {
                throw Q35Error.generationFailed(
                    "Qwen-family prompt and requested generation leave no context for \(imageURLs.count) image(s); "
                        + "increase maxContextTokens or lower maxTokens."
                )
            }
            visionTokenLimitPerImage = perImageLimit
            try ensureVisionWeightsLoaded(progressHandler: progressHandler)
            if let visionTower {
                visionReplacements = try buildVisionReplacements(
                    imageURLs: imageURLs,
                    visionTower: visionTower,
                    maximumTokensPerImage: perImageLimit
                )
            }
        }

        var promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: messages,
            tools: request.tools,
            addGenerationPrompt: true,
            includeThinking: includeThinking,
            reasoningEffort: nativeReasoningEffort,
            maxLength: imageURLs.isEmpty
                ? effectiveContext
                : loadedConfig.textConfig.maxPositionEmbeddings,
            imageTokenCounts: visionReplacements.map { max(1, $0.embeddings.dim(0)) }
        )
        if visionTokenLimitPerImage != nil, promptTokens.count > effectiveContext {
            throw Q35Error.generationFailed(
                "Qwen-family vision prompt exceeded maxContextTokens after context-aware image resizing."
            )
        } else if promptTokens.count > effectiveContext {
            promptTokens = Array(promptTokens.suffix(effectiveContext))
        }
        if let dumpPath = ProcessInfo.processInfo.environment["MERERUN_Q35_DEBUG_PROMPT_TOKENS"] {
            try? promptTokens.map(String.init).joined(separator: ",")
                .write(toFile: dumpPath, atomically: true, encoding: .utf8)
        }

        let eosSet = request.stopOnEOS
            ? Set(
                loadedConfig.eosTokenIds
                    + loadedGenerationEOSTokenIds
                    + [tokenizerAndTemplate.eosTokenId].compactMap { $0 }
            )
            : Set<Int>()
        let generationConfig = Q35Sampling.generationConfig(for: request, promptTokenCount: promptTokens.count)
        let mtpSpeculationEligible = !request.logprobCapture.isEnabled
            && !generationConfig.hasActivePenalties
            && !jsonConstrained
            && request.tools?.isEmpty != false
            && imageURLs.isEmpty
            && Self.shouldSpeculate(
                modelId: modelId,
                usesMoE: loadedConfig.textConfig.usesMoE,
                promptTokenCount: promptTokens.count,
                maxContextTokens: effectiveContext
            )
            && mtpModel != nil
        let historyMode = Self.mtpHistoryMode(
            modelId: modelId,
            isQwen4Exp: loadedConfig.textConfig.isQwen4Exp,
            usesMoE: loadedConfig.textConfig.usesMoE,
            speculationEligible: mtpSpeculationEligible,
            greedy: generationConfig.temperature == 0,
            promptTokenCount: promptTokens.count
        )
        let retainMTPPromptHistory = historyMode != .none
        let streamMTPHistory = historyMode == .streaming
        let retainPrefillHidden = prefixKVCacheEnabled || mtpSpeculationEligible
        let effectiveKVCacheMode: RuntimeKVCacheMode
        switch request.kvCacheMode {
        case .affine4:
            effectiveKVCacheMode = .affine4
        case .affine8:
            effectiveKVCacheMode = .affine8
        default:
            effectiveKVCacheMode = .default
        }

        var layerCaches = makeLayerCaches(config: loadedConfig, kvCacheMode: effectiveKVCacheMode)
        let promptInput = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, promptTokens.count)

        var prefillOutput: Q35PrefillOutput
        var prefillLength = promptTokens.count
        var mropeRopeDelta: Int?

        if imageURLs.isEmpty {
            let prefixSeed = retainMTPPromptHistory && !streamMTPHistory ? nil : prefixKVCacheSeed(
                modelPath: loadedModelPath ?? "",
                promptTokens: promptTokens,
                cacheMode: effectiveKVCacheMode,
                requiresMTPSession: streamMTPHistory
            )
            let prefixCheckpoints = semanticPrefixCheckpoints(
                tokenizerAndTemplate: tokenizerAndTemplate,
                messages: messages,
                tools: request.tools,
                includeThinking: includeThinking,
                reasoningEffort: nativeReasoningEffort,
                promptTokens: promptTokens,
                maxContextLength: effectiveContext
            )
            if let prefixSeed {
                layerCaches = prefixSeed.caches
            }
            prefillOutput = try await chunkedPrefill(
                model: model,
                promptTokens: promptTokens,
                cache: layerCaches,
                startIndex: prefixSeed?.tokenCount ?? 0,
                existingLogits: prefixSeed?.logits,
                existingHidden: prefixSeed?.hidden,
                modelPath: loadedModelPath ?? "",
                checkpointTokenCounts: prefixCheckpoints,
                retainHidden: retainPrefillHidden,
                retainMTPHistory: retainMTPPromptHistory,
                mtpSession: streamMTPHistory
                    ? (prefixSeed?.mtpSession ?? Q35MTPDraftSession(
                        historyCache: loadedConfig.textConfig.isQwen4Exp ? Q38QSACache() : KVCacheSimple()
                    )) : nil,
                prefillMTPModel: streamMTPHistory ? mtpModel : nil,
                progressHandler: progressHandler
            )
        } else {
            if let imageTokenId = loadedConfig.imageTokenId ?? tokenizerAndTemplate.tokenizer.imageTokenId {
                if visionReplacements.isEmpty {
                    prefillOutput = try await chunkedPrefill(
                        model: model,
                        promptTokens: promptTokens,
                        cache: layerCaches,
                        retainHidden: retainPrefillHidden,
                        progressHandler: progressHandler
                    )
                } else {
                    var promptEmbeddings = model.embeddings(for: promptInput)
                    promptEmbeddings = insertVisionEmbeddings(
                        hiddenStates: promptEmbeddings,
                        inputIds: promptInput,
                        imageTokenId: imageTokenId,
                        replacements: visionReplacements
                    )
                    let positionData = try buildMRoPEPositionData(
                        inputIds: promptInput,
                        imageTokenId: imageTokenId,
                        replacements: visionReplacements,
                        spatialMergeSize: visionTower?.spatialMergeSize ?? 1
                    )
                    var positionIds = positionData?.positionIds

                    if promptEmbeddings.dim(1) > effectiveContext {
                        promptEmbeddings = promptEmbeddings[0..., (promptEmbeddings.dim(1) - effectiveContext)..., 0...]
                        if let currentPositionIds = positionIds {
                            positionIds = currentPositionIds[0..., 0..., (currentPositionIds.dim(2) - effectiveContext)...]
                        }
                    }
                    prefillLength = promptEmbeddings.dim(1)
                    mropeRopeDelta = positionData?.ropeDelta
                    prefillOutput = try await chunkedPrefillEmbeddings(
                        model: model,
                        inputIds: promptInput,
                        inputEmbeddings: promptEmbeddings,
                        cache: layerCaches,
                        positionIds: positionIds,
                        retainHidden: retainPrefillHidden,
                        progressHandler: progressHandler
                    )
                }
            } else {
                prefillOutput = try await chunkedPrefill(
                    model: model,
                    promptTokens: promptTokens,
                    cache: layerCaches,
                    retainHidden: retainPrefillHidden,
                    progressHandler: progressHandler
                )
            }
        }
        let prefillSeconds = Date().timeIntervalSince(prefillStart)

        let tokenBudget = max(0, min(request.maxTokens, effectiveContext - prefillLength))

        progressHandler?(ChatProgress(stage: .generating, message: ""))

        let decodeResult = try await decodeTokens(
            model: model,
            tokenizerAndTemplate: tokenizerAndTemplate,
            initialLogits: prefillOutput.logits,
            initialHidden: prefillOutput.hidden,
            prefillMTPHidden: prefillOutput.mtpHistoryHidden,
            prefillMTPSession: prefillOutput.mtpSession,
            layerCaches: layerCaches,
            eosSet: eosSet,
            generationConfig: generationConfig,
            tokenBudget: tokenBudget,
            prefillTokenCount: prefillLength,
            mropeRopeDelta: mropeRopeDelta,
            promptTokens: promptTokens,
            maxContextTokens: effectiveContext,
            jsonConstrained: jsonConstrained,
            stopAtCompletedToolCall: request.tools?.isEmpty == false && !request.parallelToolCalls,
            logprobCapture: request.logprobCapture,
            logprobRegion: request.logprobRegionHint ?? .visible,
            progressHandler: progressHandler
        )

        let stopSequences = TextGenerationStopSequences.merged(request.stopSequences)
        let decodedRaw = tokenizerAndTemplate.decode(tokens: decodeResult.generatedTokens)
        let trimmed = TextGenerationStopSequences.trimming(decodedRaw, sequences: stopSequences)
        let decoded = trimmed.text
        let finishReason = Self.finishReason(
            generatedTokenCount: decodeResult.generatedTokens.count,
            tokenBudget: tokenBudget,
            matchedStopSequence: trimmed.matchedSequence != nil
        )
        let parsedToolCalls = request.tools?.isEmpty == false
            ? Q35ToolParser.parseToolCalls(decoded)
            : []
        let toolCalls: [ToolCall]? = request.tools.flatMap { tools -> [ToolCall]? in
            let validated = ToolCallPolicy.validatedCalls(
                parsedToolCalls,
                tools: tools,
                parallelToolCalls: request.parallelToolCalls
            )
            return validated.isEmpty ? nil : validated
        }
        let visibleText = parsedToolCalls.isEmpty
            ? decoded
            : Q35ToolParser.visibleText(decoded)

        return ChatResponse(
            generatedText: visibleText,
            tokensGenerated: decodeResult.generatedTokens.count,
            showThinking: includeThinking,
            timing: ChatTiming(
                loadSeconds: 0,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decodeResult.decodeSeconds,
                firstTokenSeconds: decodeResult.firstTokenSeconds,
                kvCacheMode: effectiveKVCacheMode,
                prefillKVCache: effectiveKVCacheMode.genericCacheLabel,
                decodeKVCache: effectiveKVCacheMode.genericCacheLabel
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count,
            finishReason: finishReason,
            logprobs: decodeResult.logprobs,
            acceleration: decodeResult.acceleration
        )
    }
}
