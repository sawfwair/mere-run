import Foundation
import MLX
import MLXNN

public typealias Gemma4PrefixKVCacheStats = PrefixKVCacheStats
public typealias Gemma4ContinuousBatchingStats = RuntimeDecodeBatchingStats

/// Set MERERUN_GEMMA4_DECODE_TRACE=1 to log the per-token split between graph
/// construction (CPU) and evaluation waits (GPU) to stderr after each decode.
enum Gemma4DecodeTrace {
    static let enabled = ProcessInfo.processInfo.environment["MERERUN_GEMMA4_DECODE_TRACE"] == "1"

    static func emit(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }
}

private struct Gemma4PrefixKVCacheKey: Hashable {
    let modelPath: String
    let quantization: Gemma4KVCacheQuantization
    let tokens: [Int]
}

private struct Gemma4PrefixKVCacheEntry {
    let caches: [Gemma4AttentionCache]
    let logits: MLXArray
    let priority: RuntimePrefixCacheEntryPriority
    var lastAccess: Date
}

private struct Gemma4BatchedDecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    let mtpStats: Gemma4MTPStats?
}

private struct Gemma4PrefillResult {
    let logits: MLXArray
    let hidden: MLXArray?
    let sharedKVStates: [String: Gemma4SharedKVState]
}

private final class Gemma4BatchedDecodeRow: @unchecked Sendable {
    let id: UUID
    let eosSet: Set<Int>
    let generationConfig: GenerationConfig
    let tokenBudget: Int
    let progressHandler: (@Sendable (ChatProgress) -> Void)?
    let decodeStart: Date
    let continuation: CheckedContinuation<Gemma4BatchedDecodeResult, Error>

    var logits: MLXArray
    var layerCaches: [Gemma4AttentionCache]
    var generatedTokens: [Int]
    var repetitionHistory: [Int]
    var firstTokenSeconds: Double?
    var stopped = false

    init(
        id: UUID,
        logits: MLXArray,
        layerCaches: [Gemma4AttentionCache],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        repetitionHistory: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        continuation: CheckedContinuation<Gemma4BatchedDecodeResult, Error>
    ) {
        self.id = id
        self.logits = logits
        self.layerCaches = layerCaches
        self.eosSet = eosSet
        self.generationConfig = generationConfig
        self.tokenBudget = tokenBudget
        self.repetitionHistory = repetitionHistory
        self.progressHandler = progressHandler
        self.continuation = continuation
        self.decodeStart = Date()
        self.generatedTokens = []
        self.generatedTokens.reserveCapacity(tokenBudget)
    }

    var needsDecodeStep: Bool {
        !stopped && generatedTokens.count < tokenBudget
    }

    func finish() {
        continuation.resume(
            returning: Gemma4BatchedDecodeResult(
                generatedTokens: generatedTokens,
                decodeSeconds: Date().timeIntervalSince(decodeStart),
                firstTokenSeconds: firstTokenSeconds,
                mtpStats: nil
            )
        )
    }

    func fail(_ error: Error) {
        continuation.resume(throwing: error)
    }
}

public actor Gemma4Generator: ChatGenerator {
    private static let prefillChunkSize = 512
    private static let prefixKVCacheMaxEntries = 4

    private var model: (any Gemma4CausalModel)?
    private var mtpModel: Gemma4AssistantDraftModel?
    private var loadedMTPModelPath: String?
    private var tokenizerAndTemplate: Gemma4TokenizerAndTemplate?
    private var loadedModelPath: String?
    private var loadedTextLoRASignature: String?
    private var loadedConfig: Gemma4Config?
    private var lastMTPStats = Gemma4MTPStats()

    private let modelId: String
    private let kvCacheQuantization: Gemma4KVCacheQuantization
    private let prefixKVCacheEnabled: Bool
    private let continuousBatchingEnabled: Bool

    private var prefixKVCache: [Gemma4PrefixKVCacheKey: Gemma4PrefixKVCacheEntry] = [:]
    private var prefixKVCacheHits = 0
    private var prefixKVCacheMisses = 0
    private var prefixKVCacheStores = 0
    private var prefixKVCacheReusedTokens = 0

    private var decodeQueue: [Gemma4BatchedDecodeRow] = []
    private var activeDecodeRows: [Gemma4BatchedDecodeRow] = []
    private var decodeLoopRunning = false
    private var batchedDecodeSteps = 0
    private var samePositionBatchedSteps = 0
    private var variablePositionBatchedSteps = 0
    private var singleDecodeSteps = 0
    private var totalBatchedRows = 0
    private var maxObservedBatchSize = 0

    public init(
        modelId: String = Gemma4Resources.defaultModelId,
        kvCacheQuantization: Gemma4KVCacheQuantization = Gemma4KVCacheQuantization(),
        prefixKVCacheEnabled: Bool = ProcessInfo.processInfo.environment["MERERUN_GEMMA4_PREFIX_KV_CACHE"] == "1",
        continuousBatchingEnabled: Bool = ProcessInfo.processInfo.environment["MERERUN_GEMMA4_CONTINUOUS_BATCHING"] == "1"
    ) {
        self.modelId = modelId
        self.kvCacheQuantization = kvCacheQuantization
        self.prefixKVCacheEnabled = prefixKVCacheEnabled
        self.continuousBatchingEnabled = continuousBatchingEnabled
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
        let requestedLoRASignature = Self.loraSignature(request.lora)
        if loadedTextLoRASignature != requestedLoRASignature {
            resetLoadedModel()
        }
        try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
        try await applyTextLoRAIfNeeded(request.lora, progressHandler: progressHandler)
        let loadSeconds = Date().timeIntervalSince(loadStart)

        var response = try await generate(
            request,
            progressHandler: progressHandler,
            maxContextLength: request.maxContextTokens ?? Gemma4Resources.defaultContextLength
        )
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
        failQueuedDecodeRows(CancellationError())
        resetPrefixKVCache()
        model = nil
        mtpModel = nil
        loadedMTPModelPath = nil
        tokenizerAndTemplate = nil
        loadedModelPath = nil
        loadedTextLoRASignature = nil
        loadedConfig = nil
        lastMTPStats = Gemma4MTPStats()
        Memory.clearCache()
    }

    private func resetLoadedModel() {
        failQueuedDecodeRows(CancellationError())
        resetPrefixKVCache()
        model = nil
        mtpModel = nil
        loadedMTPModelPath = nil
        tokenizerAndTemplate = nil
        loadedModelPath = nil
        loadedTextLoRASignature = nil
        loadedConfig = nil
        lastMTPStats = Gemma4MTPStats()
        Memory.clearCache()
    }

    public func prefixKVCacheStats() -> Gemma4PrefixKVCacheStats {
        Gemma4PrefixKVCacheStats(
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

    public func continuousBatchingStats() -> Gemma4ContinuousBatchingStats {
        Gemma4ContinuousBatchingStats(
            enabled: continuousBatchingEnabled,
            activeRows: activeDecodeRows.count,
            queuedRows: decodeQueue.count,
            batchedDecodeSteps: batchedDecodeSteps,
            samePositionBatchedSteps: samePositionBatchedSteps,
            variablePositionBatchedSteps: variablePositionBatchedSteps,
            singleDecodeSteps: singleDecodeSteps,
            totalBatchedRows: totalBatchedRows,
            maxBatchSize: maxObservedBatchSize
        )
    }

    public func mtpStats() -> Gemma4MTPStats {
        lastMTPStats
    }

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let normalizedRoot = Gemma4Resources.normalizedRootURL(rootURL)
        if loadedModelPath == normalizedRoot.path, model != nil, tokenizerAndTemplate != nil {
            return
        }
        resetPrefixKVCache()

        let resources = Gemma4Resources(rootURL: normalizedRoot)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw Gemma4Error.missingFiles(missing.map(\.lastPathComponent))
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 config"))
        let configData = try Data(contentsOf: resources.configURL)
        let config = try JSONDecoder().decode(Gemma4Config.self, from: configData)

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 tokenizer"))
        let tokenizer = try await Gemma4TokenizerAndTemplate.load(
            from: normalizedRoot,
            maxLengthOverride: min(Gemma4Resources.defaultContextLength, config.textConfig.maxPositionEmbeddings)
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 weights"))
        if Gemma4Resources.supportsVision(modelSpec: modelId) {
            let unifiedModel = try Gemma4UnifiedCausalLM(config: config)
            try loadWeights(into: unifiedModel, from: resources, config: config)
            model = unifiedModel
        } else {
            let textModel = Gemma4TextCausalLM(config: config.textConfig)
            try loadWeights(into: textModel, from: resources, config: config)
            model = textModel
        }
        let loadedMTP = try loadMTPAssistantIfAvailable(
            baseModelRoot: normalizedRoot,
            config: config,
            progressHandler: progressHandler
        )
        mtpModel = loadedMTP.model
        loadedMTPModelPath = loadedMTP.path
        tokenizerAndTemplate = tokenizer
        loadedConfig = config
        loadedModelPath = normalizedRoot.path
    }

    private func applyTextLoRAIfNeeded(
        _ lora: LoRA?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        guard let lora else {
            loadedTextLoRASignature = nil
            return
        }
        let signature = Self.loraSignature(lora)
        guard loadedTextLoRASignature != signature else { return }
        guard let model else {
            throw Gemma4Error.modelNotLoaded
        }
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 text LoRA"))
        _ = try await Gemma4TextLoRAAdapter.apply(lora, to: model)
        loadedTextLoRASignature = signature
        resetPrefixKVCache()
    }

    private static func loraSignature(_ lora: LoRA?) -> String? {
        guard let lora else { return nil }
        switch lora {
        case .local(let path, let scale):
            return "local:\(URL(fileURLWithPath: path).standardizedFileURL.path):\(scale)"
        case .remote(let reference, let scale):
            return "remote:\(reference):\(scale)"
        }
    }

    private func generate(
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
                decodeKVCache: kvCacheQuantization.statusDescription
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count,
            reasoningContent: reasoningSplit.reasoningContent,
            hasIncompleteReasoning: reasoningSplit.hasIncompleteReasoning
        )
    }

    private func prefillQuantization(for quantization: Gemma4KVCacheQuantization) -> Gemma4KVCacheQuantization {
        guard shouldDeferQuantizationUntilDecode(quantization) else {
            return quantization
        }

        if Gemma4Resources.usesTurboDefaults(modelSpec: modelId)
            && Gemma4Resources.supportsDefaultTurboKVQuantization {
            return Gemma4KVCacheQuantization(
                bits: Gemma4Resources.defaultTurboKVBits,
                scheme: Gemma4Resources.defaultTurboKVQuantizationScheme,
                groupSize: quantization.groupSize,
                quantizedStart: Gemma4Resources.defaultTurboQuantizedKVStart
            )
        }

        return Gemma4KVCacheQuantization()
    }

    nonisolated static func multimodalDecodeBannedTokens(
        imageTokenId: Int?,
        audioTokenId: Int?,
        videoTokenId: Int?,
        boiTokenId: Int?,
        boaTokenId: Int?,
        eoiTokenId: Int?,
        eoaTokenId: Int?,
        excluding excludedTokens: Set<Int>
    ) -> [Int] {
        var seen = Set<Int>()
        var result: [Int] = []
        for token in [
            imageTokenId,
            audioTokenId,
            videoTokenId,
            boiTokenId,
            boaTokenId,
            eoiTokenId,
            eoaTokenId,
        ].compactMap({ $0 }) {
            guard !excludedTokens.contains(token), seen.insert(token).inserted else { continue }
            result.append(token)
        }
        return result
    }

    private func shouldDeferQuantizationUntilDecode(_ quantization: Gemma4KVCacheQuantization) -> Bool {
        quantization.isEnabled && quantization.scheme == .polar
    }

    static func cleanedResponse(_ response: String, showThinking: Bool) -> String {
        guard !showThinking else { return response }

        if let finalRange = response.range(
            of: #"(?is)<\|channel>final\s*"#,
            options: .regularExpression
        ) {
            return String(response[finalRange.upperBound...])
                .replacingOccurrences(
                    of: #"(?is)<\|channel>[a-z_]+\s*"#,
                    with: "",
                    options: .regularExpression
                )
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let thoughtCloseRange = response.range(
            of: #"(?is)<\|channel>thought\b.*?<channel\|>\s*"#,
            options: .regularExpression
        ) {
            return String(response[thoughtCloseRange.upperBound...])
                .replacingOccurrences(
                    of: #"(?is)<\|channel>[a-z_]+\s*|<channel\|>"#,
                    with: "",
                    options: .regularExpression
                )
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        var cleaned = response.replacingOccurrences(
            of: #"(?is)<\|channel>thought\b.*\z"#,
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?is)<think>.*?</think>",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?is)<think>.*\\z",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?i)</think>",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: #"(?is)<\|channel>[a-z_]+\s*|<channel\|>"#,
            with: "",
            options: .regularExpression
        )
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func prepareLayerCachesForDecode(
        _ layerCaches: [Gemma4AttentionCache],
        quantization: Gemma4KVCacheQuantization,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> (caches: [Gemma4AttentionCache], conversionSeconds: Double?) {
        guard shouldDeferQuantizationUntilDecode(quantization) else {
            return (layerCaches, nil)
        }

        progressHandler?(ChatProgress(stage: .encoding, message: "Packing KV cache for decode"))
        let start = Date()
        let converted = try layerCaches.map { cache -> Gemma4AttentionCache in
            guard let reencoded = cache.reencoded(quantization: quantization) else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 could not reencode the prefill KV cache for decode.")
            }
            reencoded.evaluateStorage()
            return reencoded
        }
        return (converted, Date().timeIntervalSince(start))
    }

    private func decodeTokens(
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [Gemma4AttentionCache],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        mtpModel: Gemma4AssistantDraftModel?,
        prefillHidden: MLXArray?,
        sharedKVStates: [String: Gemma4SharedKVState],
        prefillTokenCount: Int,
        mtpStatsTemplate: Gemma4MTPStats,
        jsonConstrained: Bool = false,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Gemma4BatchedDecodeResult {
        guard tokenBudget > 0 else {
            return Gemma4BatchedDecodeResult(
                generatedTokens: [],
                decodeSeconds: 0,
                firstTokenSeconds: nil,
                mtpStats: mtpStatsTemplate
            )
        }
        // JSON-constrained requests always decode serially: the batched rows share
        // one sampling path and cannot carry per-request scanner state.
        guard continuousBatchingEnabled, !jsonConstrained else {
            return try await decodeTokensSerially(
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate,
                initialLogits: initialLogits,
                layerCaches: layerCaches,
                eosSet: eosSet,
                generationConfig: generationConfig,
                tokenBudget: tokenBudget,
                promptTokens: promptTokens,
                mtpModel: mtpModel,
                prefillHidden: prefillHidden,
                sharedKVStates: sharedKVStates,
                prefillTokenCount: prefillTokenCount,
                mtpStatsTemplate: mtpStatsTemplate,
                jsonConstrained: jsonConstrained,
                progressHandler: progressHandler
            )
        }

        let rowID = UUID()
        let initialLogitsBox = RuntimeUncheckedSendable(initialLogits)
        let layerCachesBox = RuntimeUncheckedSendable(layerCaches)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let row = Gemma4BatchedDecodeRow(
                    id: rowID,
                    logits: initialLogitsBox.value,
                    layerCaches: layerCachesBox.value,
                    eosSet: eosSet,
                    generationConfig: generationConfig,
                    tokenBudget: tokenBudget,
                    repetitionHistory: promptTokens,
                    progressHandler: progressHandler,
                    continuation: continuation
                )
                enqueueDecodeRow(row, model: model, tokenizerAndTemplate: tokenizerAndTemplate)
            }
        } onCancel: { [weak self] in
            guard let self else { return }
            Task {
                await self.cancelDecodeRow(id: rowID)
            }
        }
    }

    private func decodeTokensSerially(
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [Gemma4AttentionCache],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        mtpModel: Gemma4AssistantDraftModel?,
        prefillHidden: MLXArray?,
        sharedKVStates: [String: Gemma4SharedKVState],
        prefillTokenCount: Int,
        mtpStatsTemplate: Gemma4MTPStats,
        jsonConstrained: Bool = false,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Gemma4BatchedDecodeResult {
        if canUsePipelinedDecode(
            generationConfig,
            mtpModel: mtpModel,
            jsonConstrained: jsonConstrained
        ) {
            return try await decodeTokensPipelined(
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate,
                initialLogits: initialLogits,
                layerCaches: layerCaches,
                eosSet: eosSet,
                generationConfig: generationConfig,
                tokenBudget: tokenBudget,
                promptTokens: promptTokens,
                mtpStatsTemplate: mtpStatsTemplate,
                progressHandler: progressHandler
            )
        }

        var logits = initialLogits
        var layerCaches = layerCaches
        var generated: [Int] = []
        generated.reserveCapacity(tokenBudget)
        var repetitionHistory = promptTokens
        var previousHidden = prefillHidden.map { model.speculativeDraftHidden(lastTokenHidden($0)) }
        var currentSharedKVStates = sharedKVStates
        var mtpStats = mtpStatsTemplate
        var firstTokenSeconds: Double?
        var pendingSampledToken: Int?
        var jsonScanner = JSONPrefixScanner()
        let decodeStart = Date()
        let traceEnabled = Gemma4DecodeTrace.enabled
        var traceSampleSeconds = 0.0
        var traceForwardSeconds = 0.0

        while generated.count < tokenBudget {
            try Task.checkCancellation()
            var next: Int
            if let pending = pendingSampledToken {
                next = pending
                pendingSampledToken = nil
            } else {
                let sampleStart = CFAbsoluteTimeGetCurrent()
                next = sampleToken(
                    logits: logits[0, -1, 0...],
                    config: generationConfig,
                    previousTokens: repetitionHistory
                )
                if traceEnabled {
                    traceSampleSeconds += CFAbsoluteTimeGetCurrent() - sampleStart
                }
            }
            if jsonConstrained {
                guard let constrained = jsonConstrainedToken(
                    initial: next,
                    logits: logits[0, -1, 0...],
                    config: generationConfig,
                    previousTokens: repetitionHistory,
                    eosSet: eosSet,
                    scanner: &jsonScanner,
                    decode: { tokenizerAndTemplate.decode(token: $0) }
                ) else {
                    break
                }
                next = constrained
            }

            if eosSet.contains(next) {
                break
            }

            generated.append(next)
            if firstTokenSeconds == nil {
                firstTokenSeconds = Date().timeIntervalSince(decodeStart)
            }
            repetitionHistory.append(next)
            if let progressHandler {
                let piece = tokenizerAndTemplate.decode(token: next)
                if !piece.isEmpty {
                    progressHandler(ChatProgress(stage: .generating, message: piece))
                }
            }

            if jsonConstrained, jsonScanner.isComplete {
                break
            }

            guard generated.count < tokenBudget else {
                break
            }

            if let mtpModel, let hidden = previousHidden, !jsonConstrained, !currentSharedKVStates.isEmpty {
                let baseCaches = layerCaches
                let blockSize = min(
                    mtpStats.blockSize,
                    max(2, tokenBudget - generated.count + 1)
                )
                let positionOffset = prefillTokenCount + generated.count - 1
                // For sampled requests the drafts are generated greedily: the
                // verify loop samples the target either way, so correctness is
                // unaffected, and matching the target's argmax maximizes the
                // acceptance rate (sampled drafts collapse it to sum(p*q)).
                var draftConfig = generationConfig
                draftConfig.temperature = 0
                let draft = try mtpModel.draftBlock(
                    lastToken: next,
                    hidden: hidden,
                    sharedKVStates: currentSharedKVStates,
                    positionOffset: positionOffset,
                    blockSize: blockSize,
                    baseModel: model,
                    generationConfig: draftConfig,
                    repetitionHistory: repetitionHistory
                )
                if !draft.tokens.isEmpty {
                    mtpStats.rounds += 1
                    mtpStats.draftedTokens += draft.tokens.count
                    let candidateCaches = forkLayerCaches(baseCaches)
                    let candidateInput = MLXArray(([next] + draft.tokens).map(Int32.init))
                        .reshaped(1, draft.tokens.count + 1)
                    let candidate = model.forwardForSpeculation(
                        inputIds: candidateInput,
                        cache: candidateCaches
                    )

                    // Sample every verify position — the draft checks plus the
                    // bonus token after a full accept — in one graph with one
                    // readback, instead of a blocking sample per position.
                    // Histories are prospective: position i verifies against
                    // history + draft[0..<i], which matches the serial loop for
                    // every position at or before the first mismatch (later
                    // samples go unused).
                    let verifyBanMask = tokenBanMask(
                        vocabularySize: candidate.logits.dim(-1),
                        dtype: candidate.logits.dtype,
                        tokens: generationConfig.bannedTokens
                    )
                    var verifySampleArrays: [MLXArray] = []
                    verifySampleArrays.reserveCapacity(draft.tokens.count + 1)
                    var prospectiveHistory = repetitionHistory
                    for index in 0...draft.tokens.count {
                        verifySampleArrays.append(sampledTokenArray(
                            logits: candidate.logits[0, index, 0...],
                            config: generationConfig,
                            previousTokenIndices: repetitionHistoryArray(
                                promptTokens: prospectiveHistory,
                                config: generationConfig
                            ),
                            banMask: verifyBanMask
                        ))
                        if index < draft.tokens.count {
                            prospectiveHistory.append(draft.tokens[index])
                        }
                    }
                    let stackedVerify = MLX.stacked(verifySampleArrays)
                    MLX.eval(stackedVerify, candidate.hidden)
                    let verifySamples = stackedVerify.asArray(Int32.self).map(Int.init)

                    var accepted = 0
                    var replacement: Int?
                    for (index, draftToken) in draft.tokens.enumerated() {
                        guard verifySamples[index] == draftToken else {
                            replacement = verifySamples[index]
                            mtpStats.rejectedTokens += 1
                            break
                        }
                        accepted += 1
                    }

                    if accepted == draft.tokens.count {
                        var hitEOS = false
                        for token in draft.tokens {
                            if eosSet.contains(token) {
                                hitEOS = true
                                break
                            }
                            generated.append(token)
                            repetitionHistory.append(token)
                            mtpStats.acceptedTokens += 1
                            if let progressHandler {
                                let tokenPiece = tokenizerAndTemplate.decode(token: token)
                                if !tokenPiece.isEmpty {
                                    progressHandler(ChatProgress(stage: .generating, message: tokenPiece))
                                }
                            }
                        }
                        layerCaches = candidateCaches
                        logits = lastTokenLogits(candidate.logits)
                        previousHidden = model.speculativeDraftHidden(lastTokenHidden(candidate.hidden))
                        currentSharedKVStates = candidate.sharedKVStates
                        if hitEOS || generated.count >= tokenBudget {
                            break
                        }
                        // The bonus token was already sampled in the batched
                        // verify pass (last position, full-draft history).
                        pendingSampledToken = verifySamples[draft.tokens.count]
                        if let previousHidden {
                            MLX.eval(previousHidden)
                        }
                        continue
                    }

                    let acceptedPrefix = Array(draft.tokens.prefix(accepted))
                    var hitEOS = false
                    for token in acceptedPrefix {
                        if eosSet.contains(token) {
                            hitEOS = true
                            break
                        }
                        generated.append(token)
                        repetitionHistory.append(token)
                        mtpStats.acceptedTokens += 1
                        if let progressHandler {
                            let tokenPiece = tokenizerAndTemplate.decode(token: token)
                            if !tokenPiece.isEmpty {
                                progressHandler(ChatProgress(stage: .generating, message: tokenPiece))
                            }
                        }
                    }
                    if hitEOS || generated.count >= tokenBudget {
                        break
                    }

                    guard let replacement else {
                        continue
                    }
                    if eosSet.contains(replacement) {
                        break
                    }
                    generated.append(replacement)
                    repetitionHistory.append(replacement)
                    if let progressHandler {
                        let replacementPiece = tokenizerAndTemplate.decode(token: replacement)
                        if !replacementPiece.isEmpty {
                            progressHandler(ChatProgress(stage: .generating, message: replacementPiece))
                        }
                    }

                    let replacementCaches = forkLayerCaches(baseCaches)
                    let replacementInput = MLXArray(([next] + acceptedPrefix + [replacement]).map(Int32.init))
                        .reshaped(1, acceptedPrefix.count + 2)
                    let replacementForward = model.forwardForSpeculation(
                        inputIds: replacementInput,
                        cache: replacementCaches
                    )
                    MLX.eval(replacementForward.logits, replacementForward.hidden)
                    layerCaches = replacementCaches
                    logits = lastTokenLogits(replacementForward.logits)
                    previousHidden = model.speculativeDraftHidden(lastTokenHidden(replacementForward.hidden))
                    currentSharedKVStates = replacementForward.sharedKVStates
                    if let previousHidden {
                        MLX.eval(logits, previousHidden)
                    }
                    continue
                }
            }

            let forwardStart = CFAbsoluteTimeGetCurrent()
            let nextInput = MLXArray([Int32(next)]).reshaped(1, 1)
            if mtpModel != nil {
                let output = model.forwardForSpeculation(inputIds: nextInput, cache: layerCaches)
                logits = output.logits
                previousHidden = model.speculativeDraftHidden(lastTokenHidden(output.hidden))
                currentSharedKVStates = output.sharedKVStates
                MLX.eval(logits, previousHidden!)
            } else {
                logits = model.forward(inputIds: nextInput, cache: layerCaches)
                MLX.eval(logits)
            }
            if traceEnabled {
                traceForwardSeconds += CFAbsoluteTimeGetCurrent() - forwardStart
            }
        }

        if traceEnabled, !generated.isEmpty {
            let count = Double(generated.count)
            Gemma4DecodeTrace.emit(String(
                format: "[gemma4-decode-trace] mode=serial mtp=%d tokens=%d sample=%.2fms/tok forward=%.2fms/tok wall=%.2fms/tok",
                mtpModel == nil ? 0 : 1,
                generated.count,
                traceSampleSeconds / count * 1000,
                traceForwardSeconds / count * 1000,
                Date().timeIntervalSince(decodeStart) / count * 1000
            ))
        }

        return Gemma4BatchedDecodeResult(
            generatedTokens: generated,
            decodeSeconds: Date().timeIntervalSince(decodeStart),
            firstTokenSeconds: firstTokenSeconds,
            mtpStats: mtpStats
        )
    }

    private func canUsePipelinedDecode(
        _ config: GenerationConfig,
        mtpModel: Gemma4AssistantDraftModel?,
        jsonConstrained: Bool
    ) -> Bool {
        // MTP verification and JSON-constrained decoding both need each token on
        // the CPU before the next forward; everything else can pipeline, sampling
        // included — the token stays on the GPU and only the previous step's
        // readback blocks.
        mtpModel == nil && !jsonConstrained
    }

    private func decodeTokensPipelined(
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [Gemma4AttentionCache],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        mtpStatsTemplate: Gemma4MTPStats,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Gemma4BatchedDecodeResult {
        let layerCaches = layerCaches
        var generated: [Int] = []
        generated.reserveCapacity(tokenBudget)
        var repetitionHistory = greedyRepetitionHistoryArray(
            promptTokens: promptTokens,
            config: generationConfig
        )
        let banMask = tokenBanMask(
            vocabularySize: initialLogits.dim(-1),
            dtype: initialLogits.dtype,
            tokens: generationConfig.bannedTokens
        )
        var firstTokenSeconds: Double?
        var pendingToken = sampledTokenArray(
            logits: initialLogits[0, -1, 0...],
            config: generationConfig,
            previousTokenIndices: repetitionHistory,
            banMask: banMask
        )
        MLX.asyncEval(pendingToken)
        let decodeStart = Date()
        let traceEnabled = Gemma4DecodeTrace.enabled
        var traceBuildSeconds = 0.0
        var traceWaitSeconds = 0.0
        var traceForwardSeconds = 0.0
        var traceSampleSeconds = 0.0
        var traceScheduleSeconds = 0.0

        while generated.count < tokenBudget {
            try Task.checkCancellation()

            let buildStart = CFAbsoluteTimeGetCurrent()
            var forwardEnd = buildStart
            var sampleEnd = buildStart
            let scheduled: (token: MLXArray, history: MLXArray?)?
            if generated.count + 1 < tokenBudget {
                let nextInput = pendingToken.asType(.int32).reshaped(1, 1)
                let nextLogits = model.forward(inputIds: nextInput, cache: layerCaches)
                forwardEnd = CFAbsoluteTimeGetCurrent()
                let nextHistory = appendingGreedyRepetitionHistory(
                    repetitionHistory,
                    token: pendingToken,
                    config: generationConfig
                )
                let nextToken = sampledTokenArray(
                    logits: nextLogits[0, -1, 0...],
                    config: generationConfig,
                    previousTokenIndices: nextHistory,
                    banMask: banMask
                )
                sampleEnd = CFAbsoluteTimeGetCurrent()
                MLX.asyncEval(nextToken, nextLogits)
                scheduled = (nextToken, nextHistory)
            } else {
                scheduled = nil
            }
            let buildEnd = CFAbsoluteTimeGetCurrent()

            let next = pendingToken.item(Int.self)
            if traceEnabled {
                traceForwardSeconds += forwardEnd - buildStart
                traceSampleSeconds += sampleEnd - forwardEnd
                traceScheduleSeconds += buildEnd - sampleEnd
                traceBuildSeconds += buildEnd - buildStart
                traceWaitSeconds += CFAbsoluteTimeGetCurrent() - buildEnd
            }
            if eosSet.contains(next) {
                break
            }

            generated.append(next)
            if firstTokenSeconds == nil {
                firstTokenSeconds = Date().timeIntervalSince(decodeStart)
            }
            if let progressHandler {
                let piece = tokenizerAndTemplate.decode(token: next)
                if !piece.isEmpty {
                    progressHandler(ChatProgress(stage: .generating, message: piece))
                }
            }

            guard let scheduled, generated.count < tokenBudget else {
                break
            }
            pendingToken = scheduled.token
            repetitionHistory = scheduled.history
        }

        if traceEnabled, !generated.isEmpty {
            let count = Double(generated.count)
            Gemma4DecodeTrace.emit(String(
                format: "[gemma4-decode-trace] mode=pipelined temp=\(generationConfig.temperature) tokens=%d build=%.2fms/tok (forward=%.2f sample=%.2f schedule=%.2f) wait=%.2fms/tok wall=%.2fms/tok",
                generated.count,
                traceBuildSeconds / count * 1000,
                traceForwardSeconds / count * 1000,
                traceSampleSeconds / count * 1000,
                traceScheduleSeconds / count * 1000,
                traceWaitSeconds / count * 1000,
                Date().timeIntervalSince(decodeStart) / count * 1000
            ))
        }

        return Gemma4BatchedDecodeResult(
            generatedTokens: generated,
            decodeSeconds: Date().timeIntervalSince(decodeStart),
            firstTokenSeconds: firstTokenSeconds,
            mtpStats: mtpStatsTemplate
        )
    }

    private func greedyRepetitionHistoryArray(
        promptTokens: [Int],
        config: GenerationConfig
    ) -> MLXArray? {
        repetitionHistoryArray(promptTokens: promptTokens, config: config)
    }

    private func appendingGreedyRepetitionHistory(
        _ history: MLXArray?,
        token: MLXArray,
        config: GenerationConfig
    ) -> MLXArray? {
        appendingRepetitionHistory(history, token: token, config: config)
    }

    private func enqueueDecodeRow(
        _ row: Gemma4BatchedDecodeRow,
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate
    ) {
        decodeQueue.append(row)
        startDecodeLoopIfNeeded(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
    }

    private func cancelDecodeRow(id: UUID) {
        if let index = decodeQueue.firstIndex(where: { $0.id == id }) {
            let row = decodeQueue.remove(at: index)
            row.fail(CancellationError())
            return
        }
        if let index = activeDecodeRows.firstIndex(where: { $0.id == id }) {
            let row = activeDecodeRows.remove(at: index)
            row.fail(CancellationError())
        }
    }

    private func startDecodeLoopIfNeeded(
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate
    ) {
        guard !decodeLoopRunning else { return }
        decodeLoopRunning = true
        Task {
            await runDecodeLoop(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
        }
    }

    private func runDecodeLoop(
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate
    ) async {
        defer {
            decodeLoopRunning = false
            if !decodeQueue.isEmpty || !activeDecodeRows.isEmpty {
                startDecodeLoopIfNeeded(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
            }
        }

        while !decodeQueue.isEmpty || !activeDecodeRows.isEmpty {
            if activeDecodeRows.isEmpty {
                try? await Task.sleep(nanoseconds: 1_000_000)
            }
            activateQueuedDecodeRows()
            guard !activeDecodeRows.isEmpty else {
                continue
            }

            let rows = selectDecodeRows()
            do {
                try decodeOneStep(
                    rows: rows,
                    model: model,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
            } catch {
                failRows(rows, with: error)
            }
            finishCompletedDecodeRows()
            await Task.yield()
        }
    }

    private func activateQueuedDecodeRows() {
        guard !decodeQueue.isEmpty else { return }
        activeDecodeRows.append(contentsOf: decodeQueue)
        decodeQueue.removeAll(keepingCapacity: true)
    }

    private func selectDecodeRows() -> [Gemma4BatchedDecodeRow] {
        let eligible = activeDecodeRows.filter(\.needsDecodeStep)
        let selectedIDs = Set(RuntimeDecodeBatchPlanner.selectRows(
            eligible.map { row in
                RuntimeDecodeBatchRowMetadata(
                    row: row.id,
                    signature: decodeBatchSignature(for: row),
                    position: decodePosition(row)
                )
            }
        ))
        return eligible.filter { selectedIDs.contains($0.id) }
    }

    private func decodeBatchSignature(for row: Gemma4BatchedDecodeRow) -> String {
        row.layerCaches
            .map { "\(String(describing: type(of: $0))):\($0.offset)" }
            .joined(separator: "|")
    }

    private func decodePosition(_ row: Gemma4BatchedDecodeRow) -> Int {
        row.layerCaches.map(\.offset).min() ?? 0
    }

    private func decodeOneStep(
        rows: [Gemma4BatchedDecodeRow],
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate
    ) throws {
        let sampledRows = rows.filter(\.needsDecodeStep)
        guard !sampledRows.isEmpty else { return }

        for row in sampledRows {
            let next = sampleToken(
                logits: row.logits[0, -1, 0...],
                config: row.generationConfig,
                previousTokens: row.repetitionHistory
            )
            guard !row.eosSet.contains(next) else {
                row.stopped = true
                continue
            }
            row.generatedTokens.append(next)
            if row.firstTokenSeconds == nil {
                row.firstTokenSeconds = Date().timeIntervalSince(row.decodeStart)
            }
            row.repetitionHistory.append(next)
            if let progressHandler = row.progressHandler {
                let piece = tokenizerAndTemplate.decode(token: next)
                if !piece.isEmpty {
                    progressHandler(ChatProgress(stage: .generating, message: piece))
                }
            }
        }

        let continuingRows = sampledRows.filter(\.needsDecodeStep)
        guard !continuingRows.isEmpty else { return }

        if continuingRows.count > 1,
           let batchedCaches = makeBatchedLayerCaches(continuingRows.map(\.layerCaches)) {
            let nextInput = MLXArray(continuingRows.compactMap { $0.generatedTokens.last }.map(Int32.init))
                .reshaped(continuingRows.count, 1)
            let batchedLogits = model.forward(inputIds: nextInput, cache: batchedCaches)
            MLX.eval(batchedLogits)
            guard let splitCaches = splitBatchedLayerCaches(batchedCaches, rowCount: continuingRows.count) else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 batched decode could not split merged KV cache rows.")
            }
            for (index, row) in continuingRows.enumerated() {
                row.layerCaches = splitCaches[index]
                row.logits = batchedLogits[index..<(index + 1), 0..., 0...]
            }
            batchedDecodeSteps += 1
            if RuntimeDecodeBatchPositionKind.variablePositionBatchCount(continuingRows.map(decodePosition)) > 0 {
                variablePositionBatchedSteps += 1
            } else {
                samePositionBatchedSteps += 1
            }
            totalBatchedRows += continuingRows.count
            maxObservedBatchSize = max(maxObservedBatchSize, continuingRows.count)
            return
        }

        for row in continuingRows {
            guard let next = row.generatedTokens.last else { continue }
            let nextInput = MLXArray([Int32(next)]).reshaped(1, 1)
            row.logits = model.forward(inputIds: nextInput, cache: row.layerCaches)
            MLX.eval(row.logits)
            singleDecodeSteps += 1
        }
    }

    private func makeBatchedLayerCaches(_ rowCaches: [[Gemma4AttentionCache]]) -> [Gemma4AttentionCache]? {
        guard let first = rowCaches.first, !first.isEmpty else { return nil }
        guard rowCaches.allSatisfy({ $0.count == first.count }) else { return nil }

        var result: [Gemma4AttentionCache] = []
        result.reserveCapacity(first.count)
        for layerIndex in first.indices {
            let layerCaches = rowCaches.map { $0[layerIndex] }
            guard let batched = layerCaches[0].batched(with: layerCaches) else {
                return nil
            }
            result.append(batched)
        }
        return result
    }

    private func splitBatchedLayerCaches(
        _ caches: [Gemma4AttentionCache],
        rowCount: Int
    ) -> [[Gemma4AttentionCache]]? {
        guard rowCount > 0 else { return nil }
        var rows = Array(repeating: [Gemma4AttentionCache](), count: rowCount)
        for cache in caches {
            guard let split = cache.unbatchedRows(count: rowCount), split.count == rowCount else {
                return nil
            }
            for index in 0..<rowCount {
                rows[index].append(split[index])
            }
        }
        return rows
    }

    private func forkLayerCaches(_ caches: [Gemma4AttentionCache]) -> [Gemma4AttentionCache] {
        caches.map { $0.fork() }
    }

    private func lastTokenLogits(_ logits: MLXArray) -> MLXArray {
        logits[0..., (logits.dim(1) - 1)..., 0...]
    }

    private func lastTokenHidden(_ hidden: MLXArray) -> MLXArray {
        hidden[0..., (hidden.dim(1) - 1)..., 0...]
    }

    private func finishCompletedDecodeRows() {
        var remaining: [Gemma4BatchedDecodeRow] = []
        remaining.reserveCapacity(activeDecodeRows.count)
        for row in activeDecodeRows {
            if row.needsDecodeStep {
                remaining.append(row)
            } else {
                row.finish()
            }
        }
        activeDecodeRows = remaining
    }

    private func failRows(_ rows: [Gemma4BatchedDecodeRow], with error: Error) {
        let ids = Set(rows.map(\.id))
        activeDecodeRows.removeAll { row in
            guard ids.contains(row.id) else { return false }
            row.fail(error)
            return true
        }
    }

    private func failQueuedDecodeRows(_ error: Error) {
        for row in decodeQueue {
            row.fail(error)
        }
        for row in activeDecodeRows {
            row.fail(error)
        }
        decodeQueue.removeAll()
        activeDecodeRows.removeAll()
    }

    private func chunkedPrefill(
        model: any Gemma4CausalModel,
        promptTokens: [Int],
        cache: [Gemma4AttentionCache],
        startIndex: Int,
        existingLogits: MLXArray?,
        modelPath: String,
        quantization: Gemma4KVCacheQuantization,
        checkpointTokenCounts: Set<Int> = [],
        captureSpeculation: Bool,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Gemma4PrefillResult {
        guard !promptTokens.isEmpty else {
            throw Gemma4Error.unsupportedConfiguration("Prompt is empty after tokenization.")
        }

        var processed = startIndex
        var logits = existingLogits
        var hidden: MLXArray?
        var sharedKVStates: [String: Gemma4SharedKVState] = [:]
        if processed > 0, processed < promptTokens.count {
            progressHandler?(ChatProgress(stage: .encoding, message: "Reusing \(processed) prompt KV tokens"))
        }
        while processed < promptTokens.count {
            try Task.checkCancellation()
            let end = RuntimePrefillCheckpointPlanner.nextEnd(
                processed: processed,
                total: promptTokens.count,
                chunkSize: Self.prefillChunkSize,
                checkpoints: checkpointTokenCounts
            )
            if promptTokens.count > Self.prefillChunkSize {
                progressHandler?(ChatProgress(stage: .encoding, message: "Prefilling \(end)/\(promptTokens.count) tokens"))
            }
            let chunk = MLXArray(promptTokens[processed..<end].map(Int32.init))
                .reshaped(1, end - processed)
            let chunkLogits: MLXArray
            if captureSpeculation, end == promptTokens.count {
                let output = model.forwardForSpeculation(inputIds: chunk, cache: cache)
                chunkLogits = output.logits
                hidden = output.hidden
                sharedKVStates = output.sharedKVStates
                MLX.eval(chunkLogits, output.hidden)
            } else {
                chunkLogits = model.prefillStep(inputIds: chunk, cache: cache)
                MLX.eval(chunkLogits)
            }
            logits = chunkLogits
            processed = end
            storePrefixKVCache(
                modelPath: modelPath,
                quantization: quantization,
                promptTokens: promptTokens,
                tokenCount: processed,
                cache: cache,
                logits: chunkLogits,
                priority: checkpointTokenCounts.contains(processed) ? .semantic : .chunk
            )
            await Task.yield()
        }

        guard let logits else {
            throw Gemma4Error.unsupportedConfiguration("Prefill did not produce logits.")
        }
        return Gemma4PrefillResult(logits: logits, hidden: hidden, sharedKVStates: sharedKVStates)
    }

    private func unifiedPrefill(
        model: Gemma4UnifiedCausalLM,
        promptTokens: [Int],
        imageBatch: Gemma4UnifiedImageBatch,
        imageTokenId: Int,
        cache: [Gemma4AttentionCache],
        captureSpeculation: Bool,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Gemma4PrefillResult {
        guard !promptTokens.isEmpty else {
            throw Gemma4Error.unsupportedConfiguration("Prompt is empty after tokenization.")
        }

        progressHandler?(ChatProgress(
            stage: .encoding,
            message: "Prefilling \(promptTokens.count) multimodal tokens"
        ))
        let inputIds = MLXArray(promptTokens.map(Int32.init))
            .reshaped(1, promptTokens.count)
        let mmTokenTypeIds = Gemma4UnifiedImageProcessor.mmTokenTypeIds(
            tokens: promptTokens,
            imageTokenId: imageTokenId
        )
        let result: Gemma4PrefillResult
        if captureSpeculation {
            let output = try model.forwardForSpeculation(
                inputIds: inputIds,
                pixelValues: imageBatch.pixelValues,
                imagePositionIds: imageBatch.imagePositionIds,
                mmTokenTypeIds: mmTokenTypeIds,
                cache: cache
            )
            MLX.eval(output.logits, output.hidden)
            result = Gemma4PrefillResult(
                logits: output.logits,
                hidden: output.hidden,
                sharedKVStates: output.sharedKVStates
            )
        } else {
            let logits = try model.forward(
                inputIds: inputIds,
                pixelValues: imageBatch.pixelValues,
                imagePositionIds: imageBatch.imagePositionIds,
                mmTokenTypeIds: mmTokenTypeIds,
                cache: cache
            )
            MLX.eval(logits)
            result = Gemma4PrefillResult(logits: logits, hidden: nil, sharedKVStates: [:])
        }
        await Task.yield()
        return result
    }

    private func makeLayerCaches(
        model: any Gemma4CausalModel,
        quantization: Gemma4KVCacheQuantization
    ) throws -> [Gemma4AttentionCache] {
        model.makeAttentionCache(quantization: quantization)
    }

    private func collectSharedKVStatesForMTP(
        config: Gemma4TextConfig,
        caches: [Gemma4AttentionCache]
    ) -> [String: Gemma4SharedKVState] {
        guard !caches.isEmpty else { return [:] }
        let firstSharedLayerIndex = max(0, config.numHiddenLayers - config.numKVSharedLayers)
        var cacheMap: [Int] = Array(0..<firstSharedLayerIndex)
        if firstSharedLayerIndex < config.numHiddenLayers {
            let concreteLayerTypes = Array(config.layerTypes.prefix(firstSharedLayerIndex))
            let sharedFullIndex = concreteLayerTypes.lastIndex(of: "full_attention") ?? 0
            let sharedSlidingIndex = concreteLayerTypes.lastIndex(of: "sliding_attention") ?? 0
            for index in firstSharedLayerIndex..<config.numHiddenLayers {
                if config.layerTypes[index] == "full_attention" {
                    cacheMap.append(sharedFullIndex)
                } else {
                    cacheMap.append(sharedSlidingIndex)
                }
            }
        }

        var states: [String: Gemma4SharedKVState] = [:]
        for index in 0..<min(config.numHiddenLayers, cacheMap.count) {
            let cacheIndex = cacheMap[index]
            guard cacheIndex < caches.count,
                  index < config.layerTypes.count,
                  let state = caches[cacheIndex].currentState() else {
                continue
            }
            let maxSize = (caches[cacheIndex] as? Gemma4SlidingKVCache)?.configuredMaxSize
            states[config.layerTypes[index]] = Gemma4SharedKVState(
                keys: state.0,
                values: state.1,
                offset: caches[cacheIndex].offset,
                maxSize: maxSize
            )
        }
        return states
    }

    private func semanticPrefixCheckpoints(
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        messages: [ChatMessage],
        tools: [ToolDefinition]?,
        includeThinking: Bool,
        promptTokens: [Int],
        maxContextLength: Int
    ) -> Set<Int> {
        guard prefixKVCacheEnabled, messages.count > 1 else {
            return []
        }
        let prefixMessages = Array(messages.dropLast())
        guard !prefixMessages.isEmpty,
              let prefixTokens = try? tokenizerAndTemplate.encodeForGeneration(
                  messages: prefixMessages,
                  tools: tools,
                  addGenerationPrompt: false,
                  includeThinking: includeThinking,
                  maxLength: maxContextLength
              ),
              promptTokens.starts(with: prefixTokens) else {
            return []
        }
        return RuntimePrefillCheckpointPlanner.normalizedCheckpoints(
            [prefixTokens.count],
            total: promptTokens.count
        )
    }

    private func prefixKVCacheSeed(
        modelPath: String,
        quantization: Gemma4KVCacheQuantization,
        promptTokens: [Int]
    ) -> (tokenCount: Int, caches: [Gemma4AttentionCache], logits: MLXArray)? {
        guard prefixKVCacheEnabled else { return nil }
        let matchingKey = prefixKVCache.keys
            .filter { key in
                key.modelPath == modelPath
                    && key.quantization == quantization
                    && key.tokens.count <= promptTokens.count
                    && promptTokens.starts(with: key.tokens)
            }
            .max { $0.tokens.count < $1.tokens.count }

        guard let matchingKey,
              var entry = prefixKVCache[matchingKey] else {
            prefixKVCacheMisses += 1
            return nil
        }

        entry.lastAccess = Date()
        prefixKVCache[matchingKey] = entry
        prefixKVCacheHits += 1
        prefixKVCacheReusedTokens += matchingKey.tokens.count
        return (
            matchingKey.tokens.count,
            entry.caches.map { $0.fork() },
            entry.logits
        )
    }

    private func storePrefixKVCache(
        modelPath: String,
        quantization: Gemma4KVCacheQuantization,
        promptTokens: [Int],
        tokenCount: Int,
        cache: [Gemma4AttentionCache],
        logits: MLXArray,
        priority: RuntimePrefixCacheEntryPriority
    ) {
        guard prefixKVCacheEnabled, tokenCount > 0 else { return }
        let key = Gemma4PrefixKVCacheKey(
            modelPath: modelPath,
            quantization: quantization,
            tokens: Array(promptTokens.prefix(tokenCount))
        )
        prefixKVCache[key] = Gemma4PrefixKVCacheEntry(
            caches: cache.map { $0.fork() },
            logits: logits,
            priority: priority,
            lastAccess: Date()
        )
        prefixKVCacheStores += 1
        prunePrefixKVCache()
    }

    private func prunePrefixKVCache() {
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

    private func resetPrefixKVCache() {
        prefixKVCache.removeAll()
    }

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        if let explicit = modelPath?.trimmingCharacters(in: .whitespacesAndNewlines), !explicit.isEmpty {
            return try await resolveModelLocation(explicit, progressHandler: progressHandler)
        }

        let trimmedModelId = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        let requestedModelId = trimmedModelId.isEmpty ? Gemma4Resources.defaultModelId : trimmedModelId

        if let modelID = ModelResolver.ModelID(rawValue: requestedModelId),
           let resolved = ModelResolver().resolveIfPresent(modelID) {
            return resolved.rootURL
        }

        if let fallback = resolveInstalledGemmaRoot(for: requestedModelId) {
            return fallback
        }

        if requestedModelId != Gemma4Resources.defaultModelId {
            return try await resolveModelLocation(requestedModelId, progressHandler: progressHandler)
        }

        return try await resolveHubSnapshot(
            repoId: Gemma4Resources.defaultUpstreamModelId,
            progressHandler: progressHandler
        )
    }

    private func resolveInstalledGemmaRoot(for requestedModelId: String) -> URL? {
        func existingModelDir(_ id: String) -> URL? {
            let url = MereRunModelPaths.modelDir(id)
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
                return nil
            }
            guard (try? FileManager.default.contentsOfDirectory(atPath: url.path))?.isEmpty == false else {
                return nil
            }
            return url.standardizedFileURL
        }

        if requestedModelId == Gemma4Resources.defaultModelId {
            return existingModelDir(Gemma4Resources.defaultModelId)
                ?? existingModelDir(Gemma4Resources.maxModelId)
                ?? existingModelDir(Gemma4Resources.nanoModelId)
        }

        return existingModelDir(requestedModelId)
    }

    private func resolveModelLocation(
        _ location: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        if let modelID = ModelResolver.ModelID(rawValue: location),
           Gemma4Resources.handles(modelSpec: modelID.rawValue) {
            if let resolved = ModelResolver().resolveIfPresent(modelID) {
                return resolved.rootURL
            }
            do {
                let resolution = try await ManagedModelResolver.resolveForRuntime(
                    requestedModel: modelID.rawValue,
                    defaultModelID: modelID.rawValue,
                    progress: { event in
                        switch event {
                        case .downloading(let percent):
                            progressHandler?(ChatProgress(stage: .loadingModel, message: "Downloading Gemma4... \(percent)%"))
                        case .extracting:
                            progressHandler?(ChatProgress(stage: .loadingModel, message: "Extracting Gemma4..."))
                        }
                    }
                )
                return Gemma4Resources.normalizedRootURL(resolution.url)
            } catch {
                throw Gemma4Error.downloadFailed(error.localizedDescription)
            }
        }

        let fileURL = URL(fileURLWithPath: location).standardizedFileURL
        if FileManager.default.fileExists(atPath: fileURL.path) {
            return Gemma4Resources.normalizedRootURL(fileURL)
        }
        if Gemma4Resources.isLikelyHubRepoID(location) {
            return try await resolveHubSnapshot(repoId: location, progressHandler: progressHandler)
        }
        throw Gemma4Error.unsupportedModelLocation(location)
    }

    private func resolveHubSnapshot(
        repoId: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        let snapshot = try HubSnapshot(options: HubSnapshotOptions(
            repoId: repoId,
            patterns: Gemma4Resources.snapshotPatterns
        ))
        do {
            return try await snapshot.prepare { progress in
                let percent = Int((progress.fractionCompleted * 100).rounded())
                progressHandler?(ChatProgress(stage: .loadingModel, message: "Downloading Gemma4... \(percent)%"))
            }
        } catch {
            throw Gemma4Error.downloadFailed(error.localizedDescription)
        }
    }

    private func loadMTPAssistantIfAvailable(
        baseModelRoot: URL,
        config: Gemma4Config,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> (model: Gemma4AssistantDraftModel?, path: String?) {
        guard Gemma4MTPPolicy.enabled() else {
            return (nil, nil)
        }
        guard supportsManagedMTPAssistant(baseModelRoot: baseModelRoot) else {
            return (nil, nil)
        }
        guard let assistantRoot = ManagedModelResolver.resolveInstalledModel(id: Gemma4MTPResources.modelId) else {
            return (nil, nil)
        }

        let resources = Gemma4MTPResources(rootURL: assistantRoot)
        let missing = resources.validate()
        guard missing.isEmpty else {
            return (nil, nil)
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Gemma4 MTP assistant"))
        let configData = try Data(contentsOf: resources.configURL)
        let assistantConfig = try JSONDecoder().decode(Gemma4AssistantConfig.self, from: configData)
        guard assistantConfig.backboneHiddenSize == config.textConfig.hiddenSize else {
            throw Gemma4Error.unsupportedConfiguration(
                "Gemma4 MTP assistant hidden size \(assistantConfig.backboneHiddenSize) does not match target hidden size \(config.textConfig.hiddenSize)."
            )
        }
        let assistant = try Gemma4AssistantDraftModel(config: assistantConfig)
        try assistant.loadWeights(from: resources)
        return (assistant, assistantRoot.path)
    }

    private func supportsManagedMTPAssistant(baseModelRoot: URL) -> Bool {
        let normalizedModelId = modelId.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalizedModelId == Gemma4Resources.twelveBModelId
            || normalizedModelId == Gemma4Resources.twelveB4BitModelId
            || normalizedModelId == Gemma4Resources.visionTwelveBModelId {
            return true
        }
        let path = baseModelRoot.standardizedFileURL.path
        let textManagedRoot = MereRunModelPaths.modelDir(Gemma4Resources.twelveBModelId).standardizedFileURL.path
        let text4BitManagedRoot = MereRunModelPaths.modelDir(Gemma4Resources.twelveB4BitModelId).standardizedFileURL.path
        let visionManagedRoot = MereRunModelPaths.modelDir(Gemma4Resources.visionTwelveBModelId).standardizedFileURL.path
        return path == textManagedRoot || path == text4BitManagedRoot || path == visionManagedRoot
    }

    private func loadWeights(
        into model: Gemma4TextCausalLM,
        from resources: Gemma4Resources,
        config: Gemma4Config
    ) throws {
        let include: (String) -> Bool = { key in
            key.hasPrefix("model.language_model.") || key.hasPrefix("language_model.")
        }
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            if key.hasPrefix("model.language_model.") {
                return [(String(key.dropFirst("model.".count)), value)]
            }
            if key.hasPrefix("language_model.model.") {
                return [("language_model.\(key.dropFirst("language_model.model.".count))", value)]
            }
            if key.hasPrefix("language_model.") {
                return [(key, value)]
            }
            return []
        }
        let keyMapper: (String) -> String = { key in
            if key.hasPrefix("model.language_model.") {
                return String(key.dropFirst("model.".count))
            }
            if key.hasPrefix("language_model.model.") {
                return "language_model.\(key.dropFirst("language_model.model.".count))"
            }
            if key.hasPrefix("language_model.") {
                return key
            }
            return "__unused__.\(key)"
        }
        let quantizedMapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            key.hasPrefix("__unused__.") ? [] : [(key, value)]
        }
        let quantizedModuleResolver: HFSafetensorsWeightsLoader.QuantizedModuleResolver = { _, _, _, _, biases, fallbackGroupSize, fallbackBits in
            if biases != nil || fallbackBits > 4 {
                return (groupSize: fallbackGroupSize, bits: fallbackBits, mode: QuantizationMode.affine)
            }
            return (groupSize: fallbackGroupSize, bits: fallbackBits, mode: QuantizationMode.nvfp4)
        }

        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            if try Self.indexContainsQuantizedWeights(resources.modelIndexURL) {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: resources.modelIndexURL,
                    to: model,
                    groupSize: config.textConfig.enableMoEBlock ? 16 : 64,
                    bits: 4,
                    quantizedModuleResolver: quantizedModuleResolver,
                    keyMapper: keyMapper,
                    mapper: quantizedMapper
                )
            } else {
                try HFSafetensorsWeightsLoader.applyShardedWeights(
                    indexURL: resources.modelIndexURL,
                    to: model,
                    dtype: .bfloat16,
                    verify: .none,
                    mapper: mapper
                )
            }
        } else if FileManager.default.fileExists(atPath: resources.modelWeightsURL.path) {
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: resources.modelWeightsURL,
                to: model,
                dtype: .bfloat16,
                verify: .none,
                include: include,
                mapper: mapper,
                batchSize: 24
            )
        } else {
            throw Gemma4Error.missingFiles([resources.modelIndexURL.lastPathComponent, resources.modelWeightsURL.lastPathComponent])
        }
    }

    private static func indexContainsQuantizedWeights(_ indexURL: URL) throws -> Bool {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
        return index.weightMap.keys.contains { $0.hasSuffix(".scales") }
    }

    private func loadWeights(
        into model: Gemma4UnifiedCausalLM,
        from resources: Gemma4Resources,
        config: Gemma4Config
    ) throws {
        let include: (String) -> Bool = { key in
            Self.normalizedUnifiedWeightKey(key) != nil
        }
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            guard let mapped = Self.normalizedUnifiedWeightKey(key) else {
                return []
            }
            return [(mapped, value)]
        }

        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            try HFSafetensorsWeightsLoader.applyShardedWeights(
                indexURL: resources.modelIndexURL,
                to: model,
                dtype: .bfloat16,
                verify: .none,
                mapper: mapper
            )
        } else if FileManager.default.fileExists(atPath: resources.modelWeightsURL.path) {
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: resources.modelWeightsURL,
                to: model,
                dtype: .bfloat16,
                verify: .none,
                include: include,
                mapper: mapper,
                batchSize: 24
            )
        } else {
            throw Gemma4Error.missingFiles([resources.modelIndexURL.lastPathComponent, resources.modelWeightsURL.lastPathComponent])
        }
        _ = config
    }

    private static func normalizedUnifiedWeightKey(_ key: String) -> String? {
        guard !key.contains("rotary_emb"),
              key != "lm_head.weight",
              !key.contains("embed_audio") else {
            return nil
        }

        let withoutModelPrefix = key.hasPrefix("model.")
            ? String(key.dropFirst("model.".count))
            : key

        if withoutModelPrefix.hasPrefix("language_model.model.") {
            return "language_model.\(withoutModelPrefix.dropFirst("language_model.model.".count))"
        }
        if withoutModelPrefix.hasPrefix("language_model.") {
            return withoutModelPrefix
        }
        if withoutModelPrefix.hasPrefix("vision_embedder.")
            || withoutModelPrefix.hasPrefix("embed_vision.") {
            return withoutModelPrefix
        }
        return nil
    }
}
