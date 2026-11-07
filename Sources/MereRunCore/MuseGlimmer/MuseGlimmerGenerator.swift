import Foundation
import MLX

private struct MuseGlimmerDecodeResult {
    let tokens: [Int]
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
}

enum MuseGlimmerWeightKeys {
    static func normalized(_ key: String) -> String {
        if key.hasPrefix("language_model.model.") {
            return "model.language_model.\(key.dropFirst("language_model.model.".count))"
        }
        if key.hasPrefix("language_model.lm_head") {
            return "lm_head\(key.dropFirst("language_model.lm_head".count))"
        }
        if key.hasPrefix("vision_tower.") {
            return "model.vision_tower.\(key.dropFirst("vision_tower.".count))"
        }
        if key.hasPrefix("vision_adapter.") {
            return "model.vision_adapter.\(key.dropFirst("vision_adapter.".count))"
        }
        if key.hasPrefix("vision_projection") {
            return "model.vision_projection\(key.dropFirst("vision_projection".count))"
        }
        return key
    }

    static func mapped(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        [(normalized(key), value)]
    }
}

public actor MuseGlimmerGenerator: ChatGenerator {
    private static let prefillChunkSize = 512

    private let modelID: String
    private var model: MuseGlimmerModel?
    var tokenizerAndTemplate: MuseGlimmerTokenizerAndTemplate?
    private var loadedConfig: MuseGlimmerConfig?
    var loadedModelPath: String?
    private var assistantModel: MuseGlimmerAssistantModel?
    private var loadedAssistantPath: String?
    private var warmupSignature: String?
    private var lastDFlashStats = MuseGlimmerDFlashStats()
    public static let maximumCachedPrefixTokens = 32768
    var prefixCache = ChatNativePrefixCache<MuseGlimmerCachedPrefix>(maximumTokens: MuseGlimmerGenerator.maximumCachedPrefixTokens)

    public init(modelID: String = MuseGlimmerResources.modelId) {
        self.modelID = modelID
    }

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        try await chat(request, modelPath: nil, progressHandler: progressHandler)
    }

    public func chat(
        _ request: ChatRequest,
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        try await Stream.withNewDefaultStream {
            let root = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
            let loadStart = Date()
            try await ensureLoaded(rootURL: root, progressHandler: progressHandler)
            let loadSeconds = Date().timeIntervalSince(loadStart)
            var response = try generate(request, progressHandler: progressHandler)
            if var timing = response.timing {
                timing.loadSeconds = loadSeconds
                response.timing = timing
            }
            return response
        }
    }

    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        try await Stream.withNewDefaultStream {
            let root = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
            try await ensureLoaded(rootURL: root, progressHandler: progressHandler, warmup: false)
        }
    }

    public func unload() {
        prefixCache.removeAll()
        model = nil
        tokenizerAndTemplate = nil
        loadedConfig = nil
        loadedModelPath = nil
        assistantModel = nil
        loadedAssistantPath = nil
        warmupSignature = nil
        lastDFlashStats = MuseGlimmerDFlashStats()
        Memory.clearCache()
    }

    public func dflashStats() -> MuseGlimmerDFlashStats {
        lastDFlashStats
    }

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        warmup: Bool = true
    ) async throws {
        let root = rootURL.standardizedFileURL
        if loadedModelPath == root.path, model != nil, tokenizerAndTemplate != nil {
            if assistantModel == nil, MuseGlimmerDFlashPolicy.enabled(), let loadedConfig {
                try loadAssistantIfAvailable(
                    targetConfig: loadedConfig,
                    progressHandler: progressHandler
                )
            }
            if warmup { try warmLoadedModelsIfNeeded(progressHandler: progressHandler) }
            return
        }
        let resources = MuseGlimmerResources(rootURL: root)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw MuseGlimmerError.missingFiles(missing.map(\.path))
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Muse Glimmer config"))
        let config = try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(contentsOf: resources.configURL))
        guard config.modelType == "muse_glimmer",
              config.textConfig.modelType == "muse_glimmer_text",
              config.visionConfig.modelType == "muse_glimmer_vision" else {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Expected muse_glimmer, muse_glimmer_text, and muse_glimmer_vision config types."
            )
        }
        let generationConfig = try JSONDecoder().decode(
            MuseGlimmerGenerationConfig.self,
            from: Data(contentsOf: resources.generationConfigURL)
        )
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Muse Glimmer tokenizer"))
        let tokenizer = try await MuseGlimmerTokenizerAndTemplate.load(
            from: root,
            generationConfig: generationConfig,
            maxLengthOverride: min(
                generationConfig.maxLength ?? MuseGlimmerResources.defaultContextLength,
                config.textConfig.maxPositionEmbeddings
            )
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Muse Glimmer weights"))
        let loadedModel = MuseGlimmerModel(config: config)
        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            if let quantization = config.quantization {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: resources.modelIndexURL,
                    to: loadedModel,
                    groupSize: quantization.groupSize,
                    bits: quantization.bits,
                    keyMapper: MuseGlimmerWeightKeys.normalized,
                    mapper: MuseGlimmerWeightKeys.mapped,
                    progressHandler: { progress in
                        progressHandler?(ChatProgress(
                            stage: .loadingModel,
                            message: "Loading Muse Glimmer shard \(progress.shardIndex + 1)/\(progress.shardCount)"
                        ))
                    }
                )
            } else {
                try HFSafetensorsWeightsLoader.applyShardedWeights(
                    indexURL: resources.modelIndexURL,
                    to: loadedModel,
                    mapper: MuseGlimmerWeightKeys.mapped,
                    progressHandler: { progress in
                        progressHandler?(ChatProgress(
                            stage: .loadingModel,
                            message: "Loading Muse Glimmer shard \(progress.shardIndex + 1)/\(progress.shardCount)"
                        ))
                    }
                )
            }
        } else if let quantization = config.quantization {
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                try MLX.loadArrays(url: resources.modelWeightsURL),
                to: loadedModel,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                keyMapper: MuseGlimmerWeightKeys.normalized,
                mapper: MuseGlimmerWeightKeys.mapped
            )
        } else {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: resources.modelWeightsURL,
                to: loadedModel,
                mapper: MuseGlimmerWeightKeys.mapped
            )
        }
        try Task.checkCancellation()
        model = loadedModel
        tokenizerAndTemplate = tokenizer
        loadedConfig = config
        prefixCache.removeAll()
        loadedModelPath = root.path
        try loadAssistantIfAvailable(targetConfig: config, progressHandler: progressHandler)
        if warmup { try warmLoadedModelsIfNeeded(progressHandler: progressHandler) }
    }

    private func warmLoadedModelsIfNeeded(
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws {
        guard let model, let loadedModelPath else {
            throw MuseGlimmerError.modelNotLoaded
        }
        let speculativeTokens = assistantModel.map {
            MuseGlimmerDFlashPolicy.speculativeTokens(maximum: $0.config.blockSize - 1)
        }
        let signature = [
            loadedModelPath,
            loadedAssistantPath ?? "target-only",
            speculativeTokens.map(String.init) ?? "serial",
        ].joined(separator: "|")
        guard signature != warmupSignature else { return }

        let mode = assistantModel == nil ? "target" : "target and DFlash"
        progressHandler?(ChatProgress(
            stage: .loadingModel,
            message: "Warming Muse Glimmer \(mode) inference"
        ))
        try Task.checkCancellation()
        _ = try Self.warmUp(
            model: model,
            assistant: assistantModel,
            assistantModelPath: loadedAssistantPath,
            speculativeTokens: speculativeTokens,
            checkCancellation: { try Task.checkCancellation() }
        )
        try Task.checkCancellation()
        warmupSignature = signature
    }

    /// Materializes the serial target cache path and the production DFlash
    /// verification shape before the first user-visible decode. MLX compiles
    /// lazily, so evaluating only the loaded weights leaves that cost and cache
    /// lifecycle on the first request.
    static func warmUp(
        model: MuseGlimmerModel,
        assistant: MuseGlimmerAssistantModel?,
        assistantModelPath: String? = nil,
        speculativeTokens requestedSpeculativeTokens: Int? = nil,
        checkCancellation: (() throws -> Void)? = nil
    ) throws -> MuseGlimmerDFlashStats? {
        try checkCancellation?()
        let token = MLXArray([Int32(0)]).reshaped(1, 1)
        let captureLayers = Set(assistant?.config.targetLayerIds ?? [])
        let targetCache = model.makeCache()
        let output = model.forward(
            token,
            cache: targetCache,
            captureLayerIndices: captureLayers
        )
        MLX.eval([output.logits] + Array(output.capturedHiddenStates.values))
        evaluateGemma4CacheStorage(targetCache)

        // Compile and materialize the one-token serial decode independently of
        // whether DFlash is attached. This is also the adaptive fallback path.
        let serialCache = targetCache.map { $0.fork() }
        let serialLogits = model(token, cache: serialCache)
        MLX.eval(serialLogits)
        evaluateGemma4CacheStorage(serialCache)

        guard let assistant else { return nil }
        let assistantCache = assistant.makeCache()
        assistant.appendTargetContext(output.capturedHiddenStates, cache: assistantCache)
        evaluateGemma4CacheStorage(assistantCache)
        let speculativeTokens = requestedSpeculativeTokens
            ?? MuseGlimmerDFlashPolicy.speculativeTokens(maximum: assistant.config.blockSize - 1)
        let generation = GenerationConfig(
            maxTokens: 8,
            temperature: 0,
            topK: MuseGlimmerResources.recommendedTopK,
            topP: 0.95,
            minP: 0,
            repetitionPenalty: 1,
            repetitionContextSize: 64
        )
        let result = try MuseGlimmerDFlashDecoder.decode(
            initialLogits: output.logits,
            target: model,
            targetCache: targetCache,
            assistant: assistant,
            assistantCache: assistantCache,
            generationConfig: generation,
            eosTokens: [],
            tokenBudget: generation.maxTokens,
            historySeedTokens: [0],
            speculativeTokens: speculativeTokens,
            assistantModelPath: assistantModelPath,
            checkCancellation: checkCancellation
        )
        return result.stats
    }

    private func loadAssistantIfAvailable(
        targetConfig: MuseGlimmerConfig,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws {
        assistantModel = nil
        loadedAssistantPath = nil
        guard MuseGlimmerDFlashPolicy.enabled() else {
            lastDFlashStats = MuseGlimmerDFlashStats(
                reason: "disabled by MERERUN_MUSE_GLIMMER_DFLASH"
            )
            return
        }
        guard let root = resolveAssistantRoot() else {
            lastDFlashStats = MuseGlimmerDFlashStats(
                enabled: true,
                reason: "assistant companion is not installed"
            )
            return
        }
        let missing = MuseGlimmerResources.validateAssistant(rootURL: root)
        guard missing.isEmpty else {
            throw MuseGlimmerError.missingFiles(missing.map(\.path))
        }
        progressHandler?(ChatProgress(
            stage: .loadingModel,
            message: "Loading Muse Glimmer DFlash assistant"
        ))
        let config = try JSONDecoder().decode(
            MuseGlimmerAssistantConfig.self,
            from: Data(contentsOf: root.appending(path: "config.json"))
        )
        try validateAssistantCompatibility(config, target: targetConfig.textConfig)
        let assistant = MuseGlimmerAssistantModel(config: config)
        let index = root.appending(path: "model.safetensors.index.json")
        let weights = root.appending(path: "model.safetensors")
        if FileManager.default.fileExists(atPath: index.path) {
            if let quantization = config.quantization {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: index,
                    to: assistant,
                    groupSize: quantization.groupSize,
                    bits: quantization.bits,
                    mapper: MuseGlimmerDFlash2WeightKeys.mapped
                )
            } else {
                try HFSafetensorsWeightsLoader.applyShardedWeights(
                    indexURL: index,
                    to: assistant,
                    mapper: MuseGlimmerDFlash2WeightKeys.mapped
                )
            }
        } else if let quantization = config.quantization {
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                try MLX.loadArrays(url: weights),
                to: assistant,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                mapper: MuseGlimmerDFlash2WeightKeys.mapped
            )
        } else {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: weights,
                to: assistant,
                mapper: MuseGlimmerDFlash2WeightKeys.mapped
            )
        }
        assistantModel = assistant
        loadedAssistantPath = root.path
        lastDFlashStats = MuseGlimmerDFlashStats(
            enabled: true,
            active: true,
            assistantModelPath: root.path,
            speculativeTokens: MuseGlimmerDFlashPolicy.speculativeTokens(
                maximum: config.blockSize - 1
            )
        )
    }

    private func resolveAssistantRoot() -> URL? {
        let environment = ProcessInfo.processInfo.environment
        let configured = environment["MERERUN_MUSE_GLIMMER_DFLASH_PATH"]
            .map { URL(fileURLWithPath: $0).standardizedFileURL }
        let quantized = MereRunModelPaths.modelsDir
            .appendingPathComponent(MuseGlimmerResources.assistantQuantizedModelId, isDirectory: true)
            .standardizedFileURL
        let dflash2 = ManagedModelResolver.resolveInstalledModel(
            id: MuseGlimmerResources.dflash2ModelId
        )?.standardizedFileURL
        let legacy = ManagedModelResolver.resolveInstalledModel(
            id: MuseGlimmerResources.assistantModelId
        )?.standardizedFileURL
        return [configured, dflash2, legacy, quantized]
            .compactMap { $0 }
            .first { MuseGlimmerResources.validateAssistant(rootURL: $0).isEmpty }
    }

    private func validateAssistantCompatibility(
        _ assistant: MuseGlimmerAssistantConfig,
        target: MuseGlimmerTextConfig
    ) throws {
        let supportedModelType = assistant.modelType == "muse_glimmer_assistant"
            || (assistant.isDFlash2 && assistant.modelType == "qwen3")
        guard supportedModelType else {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Expected a Muse Glimmer DFlash or DFlash2 companion."
            )
        }
        if assistant.isDFlash2,
           assistant.isCausal != false {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Muse DFlash2 requires non-causal block attention."
            )
        }
        guard assistant.hiddenSize == target.hiddenSize else {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Muse DFlash hidden size \(assistant.hiddenSize) does not match target \(target.hiddenSize)."
            )
        }
        guard assistant.targetLayerIds.allSatisfy({ (0..<target.numHiddenLayers).contains($0) }) else {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Muse DFlash target_layer_ids are outside the target layer range."
            )
        }
        if let vocabularySize = assistant.vocabSize,
           vocabularySize != target.vocabSize {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Muse DFlash vocabulary size \(vocabularySize) does not match target \(target.vocabSize)."
            )
        }
        if let dflash2 = assistant.dflash2,
           dflash2.outputMultiplier != target.outputMultiplier
            || dflash2.finalLogitSoftcapping != target.finalLogitSoftcapping {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Muse DFlash2 logit scaling does not match the target."
            )
        }
        guard assistant.hiddenActivation == "silu",
              assistant.attentionDropout == 0 else {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Muse DFlash requires SiLU and zero attention dropout."
            )
        }
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> ChatResponse {
        guard let model, let tokenizerAndTemplate, loadedConfig != nil else {
            throw MuseGlimmerError.modelNotLoaded
        }
        guard request.lora == nil else {
            throw MuseGlimmerError.generationFailed("Muse Glimmer LoRA loading is not yet supported.")
        }
        guard !request.requiresJSON else {
            throw MuseGlimmerError.generationFailed(
                "Muse Glimmer constrained JSON generation is not yet supported; use tool schemas for structured actions."
            )
        }
        guard request.kvCacheMode == nil || request.kvCacheMode == .default else {
            throw MuseGlimmerError.generationFailed("Muse Glimmer currently supports the native BF16 KV cache mode.")
        }
        let (prepared, imageBatch) = try preparedInput(request)
        let promptTokens = prepared.tokens
        let effectiveContext = prepared.contextLimit
        guard promptTokens.count <= effectiveContext else { throw ChatRequestIssue("messages", "prompt is too long for the configured context") }
        let tokenBudget = request.cacheWarmOnly ? 0 : max(0, min(request.maxTokens, effectiveContext - promptTokens.count))
        let identity = prefixIdentity(request)
        let checkpoints = imageBatch == nil ? try prefixCheckpoints(request, tokens: promptTokens) : [:]
        if request.cacheWarmOnly, checkpoints.isEmpty { throw ChatRequestIssue("cache_control", "no reusable text prefix fits the native cache limit of \(Self.maximumCachedPrefixTokens) retained tokens; image cache warming is unsupported") }
        let seed = request.prefixCachePolicy != nil && imageBatch == nil && !checkpoints.isEmpty
            ? prefixCache.lookup(identity: identity, tokens: promptTokens, ttls: checkpoints) : nil
        var cacheWrites = 0
        let minimumDFlashOutput = MuseGlimmerDFlashPolicy.minimumOutputTokens()
        let useDFlash = assistantModel != nil && tokenBudget >= minimumDFlashOutput && request.prefixCachePolicy == nil
        let assistantCache = useDFlash ? assistantModel?.makeCache() : nil
        let prefillStart = Date()
        let cache = seed?.value.cache.map { $0.fork() } ?? model.makeCache()
        let prefilled = try chunkedPrefill(
            model: model,
            tokens: promptTokens,
            imageBatch: imageBatch,
            cache: cache,
            assistant: useDFlash ? assistantModel : nil,
            assistantCache: assistantCache,
            progressHandler: progressHandler,
            offset: seed?.tokens.count ?? 0,
            initialLogits: seed?.value.logits,
            checkpoints: checkpoints,
            cacheIdentity: identity
        )
        let initialLogits = prefilled.logits
        cacheWrites = prefilled.cacheWrites
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        progressHandler?(.init(stage: .encoding, event: .usage(input: promptTokens.count, output: 0,
            cacheRead: seed?.tokens.count ?? 0, cacheWrite: cacheWrites)))
        if request.cacheWarmOnly {
            var response = ChatResponse(response: "", tokensGenerated: 0, promptTokens: promptTokens.count, finishReason: .stop)
            response.cacheReadTokens = seed?.tokens.count ?? 0; response.cacheCreationTokens = cacheWrites
            return response
        }
        let sampling = GenerationConfig(
            maxTokens: tokenBudget,
            temperature: Float(request.temperature),
            topK: request.topK ?? MuseGlimmerResources.recommendedTopK,
            topP: Float(request.topP),
            minP: Float(request.minP),
            repetitionPenalty: 1,
            repetitionContextSize: 64
        )
        progressHandler?(ChatProgress(stage: .generating, message: ""))
        let decoded: MuseGlimmerDecodeResult
        var contentStream = MuseGlimmerContentStream(stops: request.stopSequences)
        var contentEvents: [ChatGenerationEvent] = []
        let emit: (String) -> Void = { piece in
            for event in contentStream.append(piece) {
                contentEvents.append(event)
                progressHandler?(ChatProgress(stage: .generating, event: event))
            }
        }
        if let assistant = useDFlash ? assistantModel : nil,
           let assistantCache {
            progressHandler?(ChatProgress(
                stage: .generating,
                message: "Muse DFlash active"
            ))
            let dflash = try MuseGlimmerDFlashDecoder.decode(
                initialLogits: initialLogits,
                target: model,
                targetCache: cache,
                assistant: assistant,
                assistantCache: assistantCache,
                generationConfig: sampling,
                eosTokens: Set(tokenizerAndTemplate.eosTokenIds),
                tokenBudget: tokenBudget,
                historySeedTokens: promptTokens,
                speculativeTokens: MuseGlimmerDFlashPolicy.speculativeTokens(
                    maximum: assistant.config.blockSize - 1
                ),
                assistantModelPath: loadedAssistantPath,
                decodeToken: { tokenizerAndTemplate.decode(token: $0) },
                emitPiece: { _, piece in emit(piece) },
                decodeTokens: { tokenizerAndTemplate.decode(tokens: $0) },
                shouldContinue: { !contentStream.didStop },
                checkCancellation: { try Task.checkCancellation() }
            )
            lastDFlashStats = dflash.stats
            decoded = MuseGlimmerDecodeResult(
                tokens: dflash.generatedTokens,
                decodeSeconds: dflash.decodeSeconds,
                firstTokenSeconds: dflash.firstTokenSeconds
            )
        } else {
            if assistantModel != nil {
                lastDFlashStats = MuseGlimmerDFlashStats(
                    enabled: true,
                    active: false,
                    assistantModelPath: loadedAssistantPath,
                    speculativeTokens: assistantModel.map {
                        MuseGlimmerDFlashPolicy.speculativeTokens(maximum: $0.config.blockSize - 1)
                    } ?? 0,
                    reason: request.prefixCachePolicy != nil ? "prefix cache uses target-only decoding" : "output budget is below \(minimumDFlashOutput) tokens"
                )
            }
            decoded = try decode(
                model: model,
                tokenizer: tokenizerAndTemplate,
                initialLogits: initialLogits,
                cache: cache,
                sampling: sampling,
                tokenBudget: tokenBudget,
                promptTokens: promptTokens,
                emitPiece: emit,
                shouldContinue: { !contentStream.didStop }
            )
        }
        for event in contentStream.append("", final: true) {
            contentEvents.append(event)
            progressHandler?(ChatProgress(stage: .generating, event: event))
        }
        var rawText = tokenizerAndTemplate.decode(tokens: decoded.tokens)
        let matchedStop = TextGenerationStopSequences.firstMatch(in: rawText, sequences: request.stopSequences)?.sequence
        rawText = Self.truncate(rawText, at: request.stopSequences)
        let parsed = MuseGlimmerOutputParser.parse(rawText)
        let validatedCalls = try MuseGlimmerToolParser.validatedToolCalls(
            rawText, tools: request.tools ?? [], parallel: request.parallelToolCalls
        )
        let response: String
        if request.showThinking, let reasoning = parsed.reasoning, !reasoning.isEmpty {
            response = "<think>\n\(reasoning)\n</think>\n\(parsed.visible)"
        } else {
            response = parsed.visible
        }
        let trimmedResponse = response.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedResponse.isEmpty {
            // Muse's recipient protocol emits private reasoning before the user-facing
            // answer. Buffer decoding so streaming never leaks that hidden channel.
            progressHandler?(ChatProgress(stage: .generating, message: trimmedResponse))
        }
        var final = ChatResponse(
            response: trimmedResponse,
            tokensGenerated: decoded.tokens.count,
            timing: ChatTiming(
                loadSeconds: 0,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decoded.decodeSeconds,
                firstTokenSeconds: decoded.firstTokenSeconds,
                kvCacheMode: .default,
                prefillKVCache: "native-bf16",
                decodeKVCache: "native-bf16"
            ),
            toolCalls: validatedCalls.isEmpty ? nil : validatedCalls,
            promptTokens: promptTokens.count,
            finishReason: matchedStop != nil ? .stopSequence : decoded.tokens.count >= tokenBudget ? .length : .stop,
            reasoningContent: parsed.reasoning,
            reasoningBlockCount: parsed.reasoning == nil ? 0 : 1
        )
        final.generationEvents = contentEvents
        final.matchedStopSequence = matchedStop
        final.cacheReadTokens = seed?.tokens.count ?? 0; final.cacheCreationTokens = cacheWrites
        return final
    }

    private func preparedInput(_ request: ChatRequest) throws -> (ChatPreparedPrompt, MuseGlimmerImageBatch?) {
        guard let tokenizerAndTemplate, let loadedConfig else { throw MuseGlimmerError.modelNotLoaded }
        let requestedContext = request.maxContextTokens ?? MuseGlimmerResources.defaultContextLength
        guard requestedContext > 0 else {
            throw MuseGlimmerError.generationFailed("maxContextTokens must be greater than zero.")
        }
        let effectiveContext = min(
            requestedContext,
            loadedConfig.textConfig.maxPositionEmbeddings,
            tokenizerAndTemplate.maxLength
        )
        let imageReferences = request.messages.flatMap { message -> [String] in
            if let blocks = message.contentBlocks {
                return blocks.compactMap { if case .image(let reference, _) = $0 { return reference }; return nil }
            }
            return message.imageUrl.map { [$0] } ?? []
        }
        let imageBatch = try imageReferences.isEmpty ? nil : MuseGlimmerImageProcessor.makeBatch(
            imageReferences: imageReferences,
            config: loadedConfig.visionConfig
        )
        if let prepared = request.preparedPrompt { return (prepared, imageBatch) }
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyy-MM-dd"
        var promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: MuseGlimmerTokenizerAndTemplate.requestMessages(request),
            tools: request.tools,
            reasoningStrength: Self.reasoningStrength(request.reasoningEffort),
            currentDate: formatter.string(from: Date()),
            maxLength: effectiveContext,
            truncate: false
        )
        if let imageBatch {
            promptTokens = try MuseGlimmerImageProcessor.expandedPromptTokens(
                promptTokens,
                tokenCounts: imageBatch.tokenCounts,
                imageTokenId: tokenizerAndTemplate.imageTokenId,
                imageStartTokenId: tokenizerAndTemplate.imageStartTokenId,
                imageEndTokenId: tokenizerAndTemplate.imageEndTokenId
            )
        }
        guard !promptTokens.isEmpty else {
            throw MuseGlimmerError.generationFailed("Muse Glimmer prompt tokenization produced no tokens.")
        }

        return (try ChatPreparedPrompt(tokens: promptTokens, contextLimit: effectiveContext), imageBatch)
    }

    public func preparePrompt(_ request: ChatRequest, modelPath: String? = nil) async throws -> ChatPreparedPrompt {
        try await prepare(modelPath: modelPath)
        return try preparedInput(request).0
    }

    private func chunkedPrefill(
        model: MuseGlimmerModel,
        tokens: [Int],
        imageBatch: MuseGlimmerImageBatch?,
        cache: [Gemma4AttentionCache],
        assistant: MuseGlimmerAssistantModel?,
        assistantCache: [Gemma4AttentionCache]?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        offset initialOffset: Int = 0,
        initialLogits: MLXArray? = nil,
        checkpoints: [Int: Int] = [:],
        cacheIdentity: String
    ) throws -> (logits: MLXArray, cacheWrites: Int) {
        try Task.checkCancellation()
        let embeddings = try imageBatch.map {
            try model.inputEmbeddings(
                for: MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count), imageBatch: $0
            )
        }
        if let embeddings { MLX.eval(embeddings) }
        var cacheWrites = 0
        var offset = initialOffset
        var logits: MLXArray? = initialLogits
        while offset < tokens.count {
            try Task.checkCancellation()
            let nextCheckpoint = checkpoints.keys.filter { $0 > offset }.min() ?? tokens.count
            let end = min(tokens.count, offset + Self.prefillChunkSize, nextCheckpoint)
            let output: MuseGlimmerForwardOutput
            let capturedLayers = Set(assistant?.config.targetLayerIds ?? [])
            if let embeddings {
                output = model.forwardPrefillDetailed(
                    embeddings: embeddings[0..., offset..<end, 0...],
                    cache: cache, captureLayerIndices: capturedLayers
                )
            } else {
                let chunk = Array(tokens[offset..<end])
                output = try model.forwardPrefillDetailed(
                    inputIds: MLXArray(chunk.map(Int32.init)).reshaped(1, chunk.count),
                    imageBatch: nil, cache: cache, captureLayerIndices: capturedLayers
                )
            }
            if let assistant, let assistantCache {
                assistant.appendTargetContext(output.capturedHiddenStates, cache: assistantCache)
                evaluateGemma4CacheStorage(assistantCache)
            }
            MLX.eval(output.logits)
            evaluateGemma4CacheStorage(cache)
            logits = output.logits
            offset = end
            if let ttl = checkpoints[end] {
                let value = MuseGlimmerCachedPrefix(cache: cache.map { $0.fork() }, logits: output.logits)
                if prefixCache.store(identity: cacheIdentity, tokens: Array(tokens.prefix(end)), value: value, ttl: ttl) {
                    cacheWrites = max(cacheWrites, end - initialOffset)
                }
            }
            progressHandler?(ChatProgress(stage: .encoding, message: "Prefilled \(offset)/\(tokens.count) tokens"))
        }
        guard let logits else {
            throw MuseGlimmerError.generationFailed("Muse Glimmer prefill produced no logits.")
        }
        return (logits, cacheWrites)
    }

    private func decode(
        model: MuseGlimmerModel,
        tokenizer: MuseGlimmerTokenizerAndTemplate,
        initialLogits: MLXArray,
        cache: [Gemma4AttentionCache],
        sampling: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        emitPiece: @escaping (String) -> Void,
        shouldContinue: @escaping () -> Bool
    ) throws -> MuseGlimmerDecodeResult {
        let result = try AutoregressiveDecodeEngine.decode(
            AutoregressiveDecodeRequest(
                initialLogits: initialLogits,
                generationConfig: sampling,
                eosTokens: Set(tokenizer.eosTokenIds),
                tokenBudget: tokenBudget,
                historySeedTokens: promptTokens
            ),
            stepForward: { token in model(token, cache: cache) },
            decodeToken: { tokenizer.decode(token: $0) },
            decodeTokens: { tokenizer.decode(tokens: $0) },
            emitPiece: { _, piece in emitPiece(piece) },
            shouldContinue: { _, _ in shouldContinue() },
            checkCancellation: { try Task.checkCancellation() }
        )
        return MuseGlimmerDecodeResult(
            tokens: result.generatedTokens,
            decodeSeconds: result.decodeSeconds,
            firstTokenSeconds: result.firstTokenSeconds
        )
    }

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        if let modelPath = modelPath?.trimmingCharacters(in: .whitespacesAndNewlines), !modelPath.isEmpty {
            let root = URL(fileURLWithPath: modelPath).standardizedFileURL
            guard FileManager.default.fileExists(atPath: root.path) else {
                throw MuseGlimmerError.unsupportedModelLocation(root.path)
            }
            return root
        }
        guard MuseGlimmerResources.handles(modelSpec: modelID) else {
            throw MuseGlimmerError.unsupportedModelID(modelID)
        }
        do {
            let resolution = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelID,
                defaultModelID: MuseGlimmerResources.modelId,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(ChatProgress(
                            stage: .loadingModel,
                            message: "Downloading Muse Glimmer... \(percent)%"
                        ))
                    case .extracting:
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Extracting Muse Glimmer..."))
                    }
                }
            )
            return resolution.url.standardizedFileURL
        } catch let error as ManagedModelResolver.ResolverError {
            throw MuseGlimmerError.downloadFailed(error.localizedDescription)
        }
    }

    static func reasoningStrength(_ effort: Double?) -> String {
        guard let effort else { return "high" }
        switch effort {
        case ..<0.34: return "low"
        case ..<0.67: return "medium"
        case ..<0.9: return "high"
        default: return "xhigh"
        }
    }

    private static func truncate(_ text: String, at stops: [String]) -> String {
        let ranges = stops.filter { !$0.isEmpty }.compactMap { text.range(of: $0) }
        guard let first = ranges.min(by: { $0.lowerBound < $1.lowerBound }) else { return text }
        return String(text[..<first.lowerBound])
    }
}
