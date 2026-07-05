import Foundation
import MLX

private struct LFM2PrefillOutput {
    let logits: MLXArray
    let hidden: MLXArray
}

private struct LFM2DecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
}

private struct LFM2PrefixKVCacheKey: Hashable {
    let modelPath: String
    let tokens: [Int]
}

private struct LFM2PrefixKVCacheEntry {
    let caches: [LFM2LayerCache?]
    let logits: MLXArray
    let priority: RuntimePrefixCacheEntryPriority
    var lastAccess: Date
}

public actor LFM2Generator: ChatGenerator {
    private static let prefillChunkSize = 512
    private static let prefixKVCacheMaxEntries = 4

    /// In-memory prompt-prefix reuse, mirroring the Qwen-family
    /// implementation: forked layer caches (both attention KV and conv states
    /// support forking) are stored at prefill chunk boundaries and the
    /// longest matching token prefix seeds the next request. Chunk-boundary
    /// checkpoints only for now — the Gemma4-style semantic chat-template
    /// checkpoints are not yet derived for LFM2. The serve pool passes
    /// default-on (MERERUN_LFM2_PREFIX_KV_CACHE=0 opts out, matching the
    /// Gemma4/Q35 pattern); one-shot CLI processes keep it off since a
    /// prefix cache cannot outlive the process.
    private let prefixKVCacheEnabled: Bool

    private var prefixKVCache: [LFM2PrefixKVCacheKey: LFM2PrefixKVCacheEntry] = [:]
    private var prefixKVCacheHits = 0
    private var prefixKVCacheMisses = 0
    private var prefixKVCacheStores = 0
    private var prefixKVCacheReusedTokens = 0

    private var model: LFM2Model?
    private var tokenizerAndTemplate: LFM2TokenizerAndTemplate?
    private var loadedModelPath: String?
    private var loadedConfig: LFM2Config?

    private let modelId: String

    public init(
        modelId: String = LFM2Resources.defaultModelId,
        prefixKVCacheEnabled: Bool =
            ProcessInfo.processInfo.environment["MERERUN_LFM2_PREFIX_KV_CACHE"] == "1"
    ) {
        self.modelId = modelId
        self.prefixKVCacheEnabled = prefixKVCacheEnabled
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
        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
        let loadStart = Date()
        try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
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

    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
        try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
    }

    public func unload() {
        model = nil
        tokenizerAndTemplate = nil
        loadedModelPath = nil
        loadedConfig = nil
        Memory.clearCache()
    }

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let normalizedRoot = LFM2Resources.normalizedRootURL(rootURL)
        if loadedModelPath == normalizedRoot.path, model != nil, tokenizerAndTemplate != nil {
            return
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2 config"))
        let configData = try Data(contentsOf: normalizedRoot.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(LFM2Config.self, from: configData)

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2 tokenizer"))
        let tokenizer = try await LFM2TokenizerAndTemplate.load(
            from: normalizedRoot,
            maxLengthOverride: min(LFM2Resources.defaultContextLength, config.maxPositionEmbeddings)
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2 weights"))
        let lfm2Model = LFM2Model(config: config)
        let resources = LFM2Resources(rootURL: normalizedRoot)
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 8

        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                indexURL: resources.modelIndexURL,
                to: lfm2Model,
                groupSize: groupSize,
                bits: bits,
                mapper: LFM2Resources.mapWeight(key:value:),
                progressHandler: { progress in
                    progressHandler?(ChatProgress(
                        stage: .loadingModel,
                        message: "Loading LFM2 shard \(progress.shardIndex + 1)/\(progress.shardCount)"
                    ))
                }
            )
        } else {
            let arrays = try MLX.loadArrays(url: resources.modelWeightsURL)
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                arrays,
                to: lfm2Model,
                groupSize: groupSize,
                bits: bits,
                mapper: LFM2Resources.mapWeight(key:value:)
            )
        }

        self.model = lfm2Model
        self.tokenizerAndTemplate = tokenizer
        self.loadedConfig = config
        self.loadedModelPath = normalizedRoot.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        guard let model,
              let tokenizerAndTemplate,
              let loadedConfig else {
            throw LFM2Error.modelNotLoaded
        }

        let requestedContextLength = request.maxContextTokens ?? LFM2Resources.defaultContextLength
        guard requestedContextLength > 0 else {
            throw LFM2Error.generationFailed("maxContextTokens must be greater than zero.")
        }
        let effectiveContext = min(
            LFM2Resources.defaultContextLength,
            requestedContextLength,
            loadedConfig.maxPositionEmbeddings
        )

        let prefillStart = Date()
        var promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: request.messages,
            tools: request.tools,
            addGenerationPrompt: true,
            includeThinking: request.showThinking,
            maxLength: effectiveContext
        )
        if promptTokens.count > effectiveContext {
            promptTokens = Array(promptTokens.suffix(effectiveContext))
        }

        let eosSet = Set(
            loadedConfig.eosTokenIds
                + tokenizerAndTemplate.stopTokenIds(withTools: request.tools?.isEmpty == false)
        )
        let generationConfig = GenerationConfig(
            maxTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topP: Float(request.topP),
            repetitionPenalty: 1.05,
            repetitionContextSize: 64
        )

        var layerCaches = makeLayerCaches(config: loadedConfig)
        var prefillStartIndex = 0
        var prefillExistingLogits: MLXArray?
        if let seed = prefixKVCacheSeed(modelPath: loadedModelPath, promptTokens: promptTokens) {
            layerCaches = seed.caches
            prefillStartIndex = seed.tokenCount
            prefillExistingLogits = seed.logits
            progressHandler?(ChatProgress(stage: .encoding, message: "Reusing \(seed.tokenCount) prompt KV tokens"))
        }
        let prefillOutput = try await chunkedPrefill(
            model: model,
            promptTokens: promptTokens,
            cache: layerCaches,
            modelPath: loadedModelPath,
            startIndex: prefillStartIndex,
            existingLogits: prefillExistingLogits,
            progressHandler: progressHandler
        )
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        let tokenBudget = max(0, min(request.maxTokens, effectiveContext - promptTokens.count))

        progressHandler?(ChatProgress(stage: .generating, message: ""))
        let decodeResult = try await decodeTokensSerially(
            model: model,
            tokenizerAndTemplate: tokenizerAndTemplate,
            initialLogits: prefillOutput.logits,
            layerCaches: layerCaches,
            eosSet: eosSet,
            generationConfig: generationConfig,
            tokenBudget: tokenBudget,
            promptTokens: promptTokens,
            progressHandler: progressHandler
        )

        let decoded = tokenizerAndTemplate.decode(tokens: decodeResult.generatedTokens)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let toolCalls: [ToolCall]? = request.tools?.isEmpty == false ? {
            let parsed = Gemma4ToolParser.parseToolCalls(decoded)
            return parsed.isEmpty ? nil : parsed
        }() : nil

        return ChatResponse(
            generatedText: decoded,
            tokensGenerated: decodeResult.generatedTokens.count,
            showThinking: request.showThinking,
            timing: ChatTiming(
                loadSeconds: 0,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decodeResult.decodeSeconds
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count
        )
    }

    private func decodeTokensSerially(
        model: LFM2Model,
        tokenizerAndTemplate: LFM2TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [LFM2LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> LFM2DecodeResult {
        guard tokenBudget > 0 else {
            return LFM2DecodeResult(generatedTokens: [], decodeSeconds: 0)
        }

        var logits = initialLogits
        var generated: [Int] = []
        generated.reserveCapacity(tokenBudget)
        var repetitionHistory = promptTokens
        var pendingProgressWhitespace = ""
        let decodeStart = Date()

        func emit(_ token: Int) {
            generated.append(token)
            repetitionHistory.append(token)
            let piece = tokenizerAndTemplate.decode(token: token)
            guard !piece.isEmpty else { return }
            if piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                pendingProgressWhitespace += piece
                return
            }
            let visiblePiece = pendingProgressWhitespace.isEmpty
                ? piece
                : pendingProgressWhitespace + piece
            pendingProgressWhitespace = ""
            progressHandler?(ChatProgress(stage: .generating, message: visiblePiece))
        }

        while generated.count < tokenBudget {
            try Task.checkCancellation()
            let next = sampleToken(
                logits: logits[0, -1, 0...],
                config: generationConfig,
                previousTokens: repetitionHistory
            )
            if eosSet.contains(next) {
                break
            }
            emit(next)

            guard generated.count < tokenBudget else {
                break
            }
            let nextInput = MLXArray([Int32(next)]).reshaped(1, 1)
            logits = model(nextInput, cache: layerCaches)
            MLX.eval(logits)
        }

        return LFM2DecodeResult(
            generatedTokens: generated,
            decodeSeconds: Date().timeIntervalSince(decodeStart)
        )
    }

    private func chunkedPrefill(
        model: LFM2Model,
        promptTokens: [Int],
        cache: [LFM2LayerCache?],
        modelPath: String? = nil,
        startIndex: Int = 0,
        existingLogits: MLXArray? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> LFM2PrefillOutput {
        guard !promptTokens.isEmpty else {
            throw LFM2Error.generationFailed("Prompt tokenization produced no tokens.")
        }

        var offset = startIndex
        var logits = existingLogits
        var hidden: MLXArray?
        while offset < promptTokens.count {
            try Task.checkCancellation()
            let end = min(promptTokens.count, offset + Self.prefillChunkSize)
            let chunk = Array(promptTokens[offset..<end])
            let input = MLXArray(chunk.map(Int32.init)).reshaped(1, chunk.count)
            let output = model.forwardPrefill(input, cache: cache)
            MLX.eval(output.logits)
            MLX.eval(output.hidden)
            logits = output.logits
            hidden = output.hidden
            offset = end
            if let modelPath {
                storePrefixKVCache(
                    modelPath: modelPath,
                    promptTokens: promptTokens,
                    tokenCount: end,
                    cache: cache,
                    logits: output.logits
                )
            }
            progressHandler?(ChatProgress(
                stage: .encoding,
                message: "Prefilled \(offset)/\(promptTokens.count) tokens"
            ))
            await Task.yield()
        }

        guard let logits else {
            throw LFM2Error.generationFailed("LFM2 prefill produced no logits.")
        }
        return LFM2PrefillOutput(logits: logits, hidden: hidden ?? logits)
    }

    // MARK: - Prefix KV cache

    public func prefixKVCacheStats() -> PrefixKVCacheStats {
        PrefixKVCacheStats(
            enabled: prefixKVCacheEnabled,
            entries: prefixKVCache.count,
            maxEntries: Self.prefixKVCacheMaxEntries,
            hits: prefixKVCacheHits,
            misses: prefixKVCacheMisses,
            storedPrefixes: prefixKVCacheStores,
            reusedTokens: prefixKVCacheReusedTokens,
            storedTokens: prefixKVCache.keys.reduce(0) { $0 + $1.tokens.count }
        )
    }

    private func prefixKVCacheSeed(
        modelPath: String?,
        promptTokens: [Int]
    ) -> (tokenCount: Int, caches: [LFM2LayerCache?], logits: MLXArray)? {
        guard prefixKVCacheEnabled, let modelPath else { return nil }
        let matchingKey = prefixKVCache.keys
            .filter { key in
                key.modelPath == modelPath
                    && key.tokens.count <= promptTokens.count
                    && promptTokens.starts(with: key.tokens)
            }
            .max { $0.tokens.count < $1.tokens.count }

        guard let matchingKey, var entry = prefixKVCache[matchingKey] else {
            prefixKVCacheMisses += 1
            return nil
        }

        entry.lastAccess = Date()
        prefixKVCache[matchingKey] = entry
        prefixKVCacheHits += 1
        prefixKVCacheReusedTokens += matchingKey.tokens.count
        return (
            matchingKey.tokens.count,
            entry.caches.map { $0?.fork() },
            entry.logits
        )
    }

    private func storePrefixKVCache(
        modelPath: String,
        promptTokens: [Int],
        tokenCount: Int,
        cache: [LFM2LayerCache?],
        logits: MLXArray
    ) {
        guard prefixKVCacheEnabled, tokenCount > 0 else { return }
        let key = LFM2PrefixKVCacheKey(
            modelPath: modelPath,
            tokens: Array(promptTokens.prefix(tokenCount))
        )
        prefixKVCache[key] = LFM2PrefixKVCacheEntry(
            caches: cache.map { $0?.fork() },
            logits: logits,
            priority: .chunk,
            lastAccess: Date()
        )
        prefixKVCacheStores += 1
        while prefixKVCache.count > Self.prefixKVCacheMaxEntries {
            let metadata = prefixKVCache.mapValues {
                RuntimePrefixCacheRetentionMetadata(
                    priority: $0.priority,
                    lastAccess: $0.lastAccess
                )
            }
            guard let oldest = RuntimePrefixCacheRetentionPlanner.keyToPrune(entries: metadata) else {
                return
            }
            prefixKVCache.removeValue(forKey: oldest)
        }
    }

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        let requestedModel = modelPath ?? modelId
        guard requestedModel == LFM2Resources.defaultModelId
            || requestedModel.caseInsensitiveCompare(LFM2Resources.upstreamRepoId) == .orderedSame
            || requestedModel.hasPrefix("/")
            || requestedModel.hasPrefix("~")
            || requestedModel.hasPrefix(".") else {
            throw LFM2Error.unsupportedModelId(requestedModel)
        }

        do {
            let root = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: requestedModel,
                defaultModelID: LFM2Resources.defaultModelId,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Downloading model... \(percent)%"))
                    case .extracting:
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Extracting model..."))
                    }
                }
            )
            return LFM2Resources.normalizedRootURL(root.url)
        } catch let error as ManagedModelResolver.ResolverError {
            throw LFM2Error.downloadFailed(error.localizedDescription)
        }
    }

    private func makeLayerCaches(config: LFM2Config) -> [LFM2LayerCache?] {
        let attentionLayers = config.fullAttentionLayerIndexes
        return (0..<config.numHiddenLayers).map { layerIndex in
            if attentionLayers.contains(layerIndex) {
                return .attention(KVCacheSimple(step: 256))
            }
            return .conv(LFM2ConvCache())
        }
    }
}
