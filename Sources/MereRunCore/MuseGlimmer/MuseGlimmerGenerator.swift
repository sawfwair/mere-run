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
    private static let textPrefillChunkSize = 512

    private let modelID: String
    private var model: MuseGlimmerModel?
    private var tokenizerAndTemplate: MuseGlimmerTokenizerAndTemplate?
    private var loadedConfig: MuseGlimmerConfig?
    private var loadedModelPath: String?
    private var assistantModel: MuseGlimmerAssistantModel?
    private var loadedAssistantPath: String?
    private var warmupSignature: String?
    private var lastDFlashStats = MuseGlimmerDFlashStats()

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
            try await ensureLoaded(rootURL: root, progressHandler: progressHandler)
        }
    }

    public func unload() {
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
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let root = rootURL.standardizedFileURL
        if loadedModelPath == root.path, model != nil, tokenizerAndTemplate != nil {
            if assistantModel == nil, MuseGlimmerDFlashPolicy.enabled(), let loadedConfig {
                try loadAssistantIfAvailable(
                    targetConfig: loadedConfig,
                    progressHandler: progressHandler
                )
            }
            try warmLoadedModelsIfNeeded(progressHandler: progressHandler)
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
        loadedModelPath = root.path
        try loadAssistantIfAvailable(targetConfig: config, progressHandler: progressHandler)
        try warmLoadedModelsIfNeeded(progressHandler: progressHandler)
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
                    bits: quantization.bits
                )
            } else {
                try HFSafetensorsWeightsLoader.applyShardedWeights(indexURL: index, to: assistant)
            }
        } else if let quantization = config.quantization {
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                try MLX.loadArrays(url: weights),
                to: assistant,
                groupSize: quantization.groupSize,
                bits: quantization.bits
            )
        } else {
            try HFSafetensorsWeightsLoader.applyWeights(url: weights, to: assistant)
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
        let managed = ManagedModelResolver.resolveInstalledModel(
            id: MuseGlimmerResources.assistantModelId
        )?.standardizedFileURL
        return [configured, managed, quantized]
            .compactMap { $0 }
            .first { MuseGlimmerResources.validateAssistant(rootURL: $0).isEmpty }
    }

    private func validateAssistantCompatibility(
        _ assistant: MuseGlimmerAssistantConfig,
        target: MuseGlimmerTextConfig
    ) throws {
        guard assistant.modelType == "muse_glimmer_assistant" else {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Expected a muse_glimmer_assistant companion."
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
        guard let model, let tokenizerAndTemplate, let loadedConfig else {
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
        let requestedContext = request.maxContextTokens ?? MuseGlimmerResources.defaultContextLength
        guard requestedContext > 0 else {
            throw MuseGlimmerError.generationFailed("maxContextTokens must be greater than zero.")
        }
        let effectiveContext = min(
            requestedContext,
            loadedConfig.textConfig.maxPositionEmbeddings,
            tokenizerAndTemplate.maxLength
        )
        let imageReferences = request.messages.compactMap { message -> String? in
            guard let reference = message.imageUrl, !reference.isEmpty else { return nil }
            return reference
        }
        let imageBatch = try imageReferences.isEmpty ? nil : MuseGlimmerImageProcessor.makeBatch(
            imageReferences: imageReferences,
            config: loadedConfig.visionConfig
        )
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyy-MM-dd"
        var promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: request.messages,
            tools: request.tools,
            reasoningStrength: Self.reasoningStrength(request.reasoningEffort),
            currentDate: formatter.string(from: Date()),
            maxLength: effectiveContext
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
        if promptTokens.count > effectiveContext {
            promptTokens = Array(promptTokens.suffix(effectiveContext))
        }
        guard !promptTokens.isEmpty else {
            throw MuseGlimmerError.generationFailed("Muse Glimmer prompt tokenization produced no tokens.")
        }

        let tokenBudget = max(0, min(request.maxTokens, effectiveContext - promptTokens.count))
        let minimumDFlashOutput = MuseGlimmerDFlashPolicy.minimumOutputTokens()
        let useDFlash = assistantModel != nil && tokenBudget >= minimumDFlashOutput
        let assistantCache = useDFlash ? assistantModel?.makeCache() : nil
        let prefillStart = Date()
        let cache = model.makeCache()
        let initialLogits: MLXArray
        if imageBatch == nil {
            initialLogits = try chunkedTextPrefill(
                model: model,
                tokens: promptTokens,
                cache: cache,
                assistant: useDFlash ? assistantModel : nil,
                assistantCache: assistantCache,
                progressHandler: progressHandler
            )
        } else {
            let output = try model.forwardPrefillDetailed(
                inputIds: MLXArray(promptTokens.map(Int32.init)).reshaped(1, promptTokens.count),
                imageBatch: imageBatch,
                cache: cache,
                captureLayerIndices: Set(useDFlash ? assistantModel?.config.targetLayerIds ?? [] : [])
            )
            initialLogits = output.logits
            if let assistant = useDFlash ? assistantModel : nil,
               let assistantCache {
                assistant.appendTargetContext(output.capturedHiddenStates, cache: assistantCache)
                evaluateGemma4CacheStorage(assistantCache)
            }
            MLX.eval(output.logits)
            evaluateGemma4CacheStorage(cache)
        }
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
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
                    reason: "output budget is below \(minimumDFlashOutput) tokens"
                )
            }
            decoded = try decode(
                model: model,
                tokenizer: tokenizerAndTemplate,
                initialLogits: initialLogits,
                cache: cache,
                sampling: sampling,
                tokenBudget: tokenBudget,
                promptTokens: promptTokens
            )
        }
        var rawText = tokenizerAndTemplate.decode(tokens: decoded.tokens)
        rawText = Self.truncate(rawText, at: request.stopSequences)
        let parsed = MuseGlimmerOutputParser.parse(rawText)
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
        return ChatResponse(
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
            toolCalls: parsed.toolCalls.isEmpty ? nil : parsed.toolCalls,
            promptTokens: promptTokens.count,
            finishReason: decoded.tokens.count >= tokenBudget ? .length : .stop,
            reasoningContent: parsed.reasoning,
            reasoningBlockCount: parsed.reasoning == nil ? 0 : 1
        )
    }

    private func chunkedTextPrefill(
        model: MuseGlimmerModel,
        tokens: [Int],
        cache: [Gemma4AttentionCache],
        assistant: MuseGlimmerAssistantModel?,
        assistantCache: [Gemma4AttentionCache]?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> MLXArray {
        var offset = 0
        var logits: MLXArray?
        while offset < tokens.count {
            try Task.checkCancellation()
            let end = min(tokens.count, offset + Self.textPrefillChunkSize)
            let chunk = Array(tokens[offset..<end])
            let output = try model.forwardPrefillDetailed(
                inputIds: MLXArray(chunk.map(Int32.init)).reshaped(1, chunk.count),
                imageBatch: nil,
                cache: cache,
                captureLayerIndices: Set(assistant?.config.targetLayerIds ?? [])
            )
            if let assistant, let assistantCache {
                assistant.appendTargetContext(output.capturedHiddenStates, cache: assistantCache)
                evaluateGemma4CacheStorage(assistantCache)
            }
            MLX.eval(output.logits)
            evaluateGemma4CacheStorage(cache)
            logits = output.logits
            offset = end
            progressHandler?(ChatProgress(stage: .encoding, message: "Prefilled \(offset)/\(tokens.count) tokens"))
        }
        guard let logits else {
            throw MuseGlimmerError.generationFailed("Muse Glimmer prefill produced no logits.")
        }
        return logits
    }

    private func decode(
        model: MuseGlimmerModel,
        tokenizer: MuseGlimmerTokenizerAndTemplate,
        initialLogits: MLXArray,
        cache: [Gemma4AttentionCache],
        sampling: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int]
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
            emitPiece: nil,
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

    private static func reasoningStrength(_ effort: Double?) -> String {
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
