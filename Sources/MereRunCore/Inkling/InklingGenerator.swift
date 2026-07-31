import Foundation
import MLX

private struct InklingDecodeResult {
    let tokens: [Int]
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
}

public actor InklingGenerator: ChatGenerator {
    // The relative-position mask is [heads, query, key]; a conservative chunk
    // keeps its transient footprint bounded as global layers grow.
    private static let prefillChunkSize = 64

    private let modelID: String
    private var model: InklingLanguageModel?
    private var tokenizerAndTemplate: InklingTokenizerAndTemplate?
    private var loadedConfig: InklingConfig?
    private var loadedModelPath: String?

    public init(modelID: String = InklingResources.modelID) {
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
            var response = try await generate(request, progressHandler: progressHandler)
            if var timing = response.timing {
                timing.loadSeconds = loadSeconds
                response.timing = timing
            } else {
                response.timing = ChatTiming(loadSeconds: loadSeconds)
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
        Memory.clearCache()
    }

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let normalized = InklingResources.normalizedRootURL(rootURL)
        if loadedModelPath == normalized.path, model != nil, tokenizerAndTemplate != nil {
            return
        }
        let missing = InklingResources.validate(rootURL: normalized)
        guard missing.isEmpty else {
            throw InklingError.missingFiles(missing.map(\.path))
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Inkling config"))
        let config = try JSONDecoder().decode(
            InklingConfig.self,
            from: Data(contentsOf: normalized.appendingPathComponent("config.json"))
        )
        guard config.quantization?.bits == InklingResources.quantizationBits,
              config.quantization?.groupSize == InklingResources.quantizationGroupSize,
              config.quantization?.mode == InklingResources.quantizationMode,
              config.quantization?.scope == InklingResources.quantizationScope else {
            throw InklingError.generationFailed(
                "Inkling artifact must use affine 2-bit/group-128 routed experts with BF16 non-routed weights."
            )
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Inkling tokenizer"))
        let tokenizer = try await InklingTokenizerAndTemplate.load(
            from: normalized,
            maxLengthOverride: min(
                InklingResources.defaultContextLength,
                config.textConfig.modelMaxLength
            )
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Inkling weights"))
        let languageModel = InklingLanguageModel(config: config)
        let index = normalized.appendingPathComponent("model.safetensors.index.json")
        let single = normalized.appendingPathComponent("model.safetensors")
        let groupSize = config.quantization?.groupSize ?? InklingResources.quantizationGroupSize
        let bits = config.quantization?.bits ?? InklingResources.quantizationBits
        if FileManager.default.fileExists(atPath: index.path) {
            try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                indexURL: index,
                to: languageModel,
                groupSize: groupSize,
                bits: bits,
                keyMapper: InklingResources.mapWeightKey,
                mapper: InklingResources.mapWeight(key:value:),
                progressHandler: { progress in
                    progressHandler?(ChatProgress(
                        stage: .loadingModel,
                        message: "Loading Inkling shard \(progress.shardIndex + 1)/\(progress.shardCount)"
                    ))
                }
            )
        } else {
            let arrays = try MLX.loadArrays(url: single)
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                arrays,
                to: languageModel,
                groupSize: groupSize,
                bits: bits,
                keyMapper: InklingResources.mapWeightKey,
                mapper: InklingResources.mapWeight(key:value:)
            )
        }

        try Task.checkCancellation()
        model = languageModel
        tokenizerAndTemplate = tokenizer
        loadedConfig = config
        loadedModelPath = normalized.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        guard let model, let tokenizerAndTemplate, let loadedConfig else {
            throw InklingError.modelNotLoaded
        }
        guard request.lora == nil else {
            throw InklingError.generationFailed("Inkling LoRA loading is not yet supported.")
        }
        guard !request.requiresJSON else {
            throw InklingError.generationFailed("Inkling constrained JSON generation is not yet supported.")
        }

        let requestedContext = request.maxContextTokens ?? InklingResources.defaultContextLength
        guard requestedContext > 0 else {
            throw InklingError.generationFailed("maxContextTokens must be greater than zero.")
        }
        let effectiveContext = min(
            requestedContext,
            InklingResources.maximumContextLength,
            loadedConfig.textConfig.modelMaxLength
        )

        let prefillStart = Date()
        var promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: request.messages,
            tools: request.tools,
            addGenerationPrompt: true,
            reasoningEffort: 0.9,
            maxLength: effectiveContext
        )
        if promptTokens.count > effectiveContext {
            promptTokens = Array(promptTokens.suffix(effectiveContext))
        }
        guard !promptTokens.isEmpty else {
            throw InklingError.generationFailed("Inkling prompt tokenization produced no tokens.")
        }

        let effectiveKVMode: RuntimeKVCacheMode
        switch request.kvCacheMode {
        case .affine4:
            effectiveKVMode = .affine4
        case .affine8:
            effectiveKVMode = .affine8
        default:
            effectiveKVMode = .default
        }
        let caches = makeCaches(config: loadedConfig, mode: effectiveKVMode)
        let initialLogits = try await chunkedPrefill(
            model: model,
            tokens: promptTokens,
            cache: caches,
            progressHandler: progressHandler
        )
        let prefillSeconds = Date().timeIntervalSince(prefillStart)

        let tokenBudget = max(0, min(request.maxTokens, effectiveContext - promptTokens.count))
        progressHandler?(ChatProgress(stage: .generating, message: ""))
        let generationConfig = GenerationConfig(
            maxTokens: tokenBudget,
            temperature: Float(request.temperature),
            topK: request.topK ?? 0,
            topP: Float(request.topP),
            minP: Float(request.minP),
            repetitionPenalty: 1.05,
            repetitionContextSize: 64
        )
        let decode = try decode(
            model: model,
            tokenizer: tokenizerAndTemplate,
            initialLogits: initialLogits,
            cache: caches,
            eosTokens: Set(loadedConfig.eosTokenIDs + tokenizerAndTemplate.stopTokenIDs),
            generationConfig: generationConfig,
            tokenBudget: tokenBudget,
            promptTokens: promptTokens,
            showThinking: request.showThinking,
            progressHandler: progressHandler
        )

        let parsed = InklingOutputParser.parse(tokenizerAndTemplate.decode(tokens: decode.tokens))
        let visible: String
        if request.showThinking, let reasoning = parsed.reasoning {
            visible = "<think>\n\(reasoning)\n</think>\n\(parsed.visible)"
        } else {
            visible = parsed.visible
        }
        return ChatResponse(
            response: visible.trimmingCharacters(in: .whitespacesAndNewlines),
            tokensGenerated: decode.tokens.count,
            timing: ChatTiming(
                loadSeconds: 0,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decode.decodeSeconds,
                firstTokenSeconds: decode.firstTokenSeconds,
                kvCacheMode: effectiveKVMode,
                prefillKVCache: effectiveKVMode.genericCacheLabel,
                decodeKVCache: effectiveKVMode.genericCacheLabel
            ),
            toolCalls: parsed.toolCalls.isEmpty ? nil : parsed.toolCalls,
            promptTokens: promptTokens.count,
            finishReason: decode.tokens.count >= tokenBudget ? .length : .stop,
            reasoningContent: parsed.reasoning,
            reasoningBlockCount: parsed.reasoning == nil ? 0 : 1
        )
    }

    private func chunkedPrefill(
        model: InklingLanguageModel,
        tokens: [Int],
        cache: [InklingLayerCache],
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> MLXArray {
        var offset = 0
        var logits: MLXArray?
        while offset < tokens.count {
            try Task.checkCancellation()
            let end = min(tokens.count, offset + Self.prefillChunkSize)
            let chunk = Array(tokens[offset..<end])
            let output = model.forwardPrefill(
                MLXArray(chunk.map(Int32.init)).reshaped(1, chunk.count),
                cache: cache
            )
            MLX.eval(output.logits)
            logits = output.logits
            offset = end
            progressHandler?(ChatProgress(
                stage: .encoding,
                message: "Prefilled \(offset)/\(tokens.count) tokens"
            ))
            await Task.yield()
        }
        guard let logits else {
            throw InklingError.generationFailed("Inkling prefill produced no logits.")
        }
        return logits
    }

    private func decode(
        model: InklingLanguageModel,
        tokenizer: InklingTokenizerAndTemplate,
        initialLogits: MLXArray,
        cache: [InklingLayerCache],
        eosTokens: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        showThinking: Bool,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> InklingDecodeResult {
        guard tokenBudget > 0 else {
            return InklingDecodeResult(tokens: [], decodeSeconds: 0, firstTokenSeconds: nil)
        }
        let result = try AutoregressiveDecodeEngine.decode(
            AutoregressiveDecodeRequest(
                initialLogits: initialLogits,
                generationConfig: generationConfig,
                eosTokens: eosTokens,
                tokenBudget: tokenBudget,
                historySeedTokens: promptTokens
            ),
            stepForward: { token in model(token, cache: cache) },
            decodeToken: { tokenizer.decode(token: $0) },
            emitPiece: { _, piece in
                // Hidden-reasoning mode buffers output until channel parsing can
                // separate thinking from the visible answer.
                if showThinking {
                    progressHandler?(ChatProgress(stage: .generating, message: piece))
                }
            },
            checkCancellation: { try Task.checkCancellation() }
        )
        return InklingDecodeResult(
            tokens: result.generatedTokens,
            decodeSeconds: result.decodeSeconds,
            firstTokenSeconds: result.firstTokenSeconds
        )
    }

    private func makeCaches(
        config: InklingConfig,
        mode: RuntimeKVCacheMode
    ) -> [InklingLayerCache] {
        (0..<config.textConfig.numHiddenLayers).map { _ in
            let attention: KVCache
            if mode == .affine4 || mode == .affine8 {
                attention = AffineQuantizedKVCache(
                    groupSize: 64,
                    bits: mode == .affine4 ? 4 : 8,
                    step: 256
                )
            } else {
                attention = KVCacheSimple(step: 256)
            }
            return InklingLayerCache(attention: attention)
        }
    }

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        let requested = modelPath ?? modelID
        guard InklingResources.handles(modelSpec: requested)
            || requested.hasPrefix("/")
            || requested.hasPrefix("~")
            || requested.hasPrefix(".") else {
            throw InklingError.unsupportedModelID(requested)
        }
        do {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: requested,
                defaultModelID: InklingResources.modelID,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(ChatProgress(
                            stage: .loadingModel,
                            message: "Downloading Inkling... \(percent)%"
                        ))
                    case .extracting:
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Extracting Inkling..."))
                    }
                }
            )
            return InklingResources.normalizedRootURL(resolved.url)
        } catch let error as ManagedModelResolver.ResolverError {
            throw InklingError.downloadFailed(error.localizedDescription)
        }
    }
}
