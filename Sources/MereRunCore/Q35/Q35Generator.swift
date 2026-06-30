import Foundation
import MediaIO
import MLX

public typealias Q35ContinuousBatchingStats = RuntimeDecodeBatchingStats

private struct Q35PrefixKVCacheKey: Hashable {
    let modelPath: String
    let tokens: [Int]
}

private struct Q35PrefixKVCacheEntry {
    let caches: [Q35LayerCache?]
    let logits: MLXArray
    let hidden: MLXArray
    let priority: RuntimePrefixCacheEntryPriority
    var lastAccess: Date
}

private struct Q35PrefillOutput {
    let logits: MLXArray
    let hidden: MLXArray?
}

private struct Q35BatchedDecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
}

private struct Q35VisionReplacement {
    let embeddings: MLXArray
    let gridTHW: (Int, Int, Int)
}

private struct Q35MRoPEPositionData {
    let positionIds: MLXArray
    let ropeDelta: Int
}

private final class Q35BatchedDecodeRow: @unchecked Sendable {
    let id: UUID
    let eosSet: Set<Int>
    let generationConfig: GenerationConfig
    let tokenBudget: Int
    let progressHandler: (@Sendable (ChatProgress) -> Void)?
    let decodeStart: Date
    let prefillTokenCount: Int
    let mropeRopeDelta: Int?
    let continuation: CheckedContinuation<Q35BatchedDecodeResult, Error>

    var logits: MLXArray
    var layerCaches: [Q35LayerCache?]
    var generatedTokens: [Int]
    var repetitionHistory: [Int]
    var stopped = false

    init(
        id: UUID,
        logits: MLXArray,
        layerCaches: [Q35LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        prefillTokenCount: Int,
        mropeRopeDelta: Int?,
        repetitionHistory: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        continuation: CheckedContinuation<Q35BatchedDecodeResult, Error>
    ) {
        self.id = id
        self.logits = logits
        self.layerCaches = layerCaches
        self.eosSet = eosSet
        self.generationConfig = generationConfig
        self.tokenBudget = tokenBudget
        self.prefillTokenCount = prefillTokenCount
        self.mropeRopeDelta = mropeRopeDelta
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
            returning: Q35BatchedDecodeResult(
                generatedTokens: generatedTokens,
                decodeSeconds: Date().timeIntervalSince(decodeStart)
            )
        )
    }

    func fail(_ error: Error) {
        continuation.resume(throwing: error)
    }
}

public actor Q35Generator: ChatGenerator {
    private static let prefillChunkSize = 512
    private static let prefixKVCacheMaxEntries = 4
    private static let defaultMTPBlockSize = 4
    public static let qwen3VLMinPixels = 2_048
    public static let qwen3VLMaxPixels = 16_777_216

    private var model: Q35Model?
    private var tokenizerAndTemplate: Q35TokenizerAndTemplate?
    private var visionTower: Q35VisionTower?
    private var mtpModel: Q35MTPModel?
    private var loadedModelPath: String?
    private var loadedConfig: Q35Config?
    private var loadedResources: Q35Resources?

    private let modelId: String
    private let prefixKVCacheEnabled: Bool
    private let continuousBatchingEnabled: Bool
    private let visionMinPixels: Int
    private let visionMaxPixels: Int

    private var prefixKVCache: [Q35PrefixKVCacheKey: Q35PrefixKVCacheEntry] = [:]
    private var prefixKVCacheHits = 0
    private var prefixKVCacheMisses = 0
    private var prefixKVCacheStores = 0
    private var prefixKVCacheReusedTokens = 0

    private var decodeQueue: [Q35BatchedDecodeRow] = []
    private var activeDecodeRows: [Q35BatchedDecodeRow] = []
    private var decodeLoopRunning = false
    private var batchedDecodeSteps = 0
    private var samePositionBatchedSteps = 0
    private var variablePositionBatchedSteps = 0
    private var singleDecodeSteps = 0
    private var totalBatchedRows = 0
    private var maxObservedBatchSize = 0

    public init(
        modelId: String = Q35Resources.defaultModelId,
        prefixKVCacheEnabled: Bool = ProcessInfo.processInfo.environment["MERERUN_Q35_PREFIX_KV_CACHE"] == "1",
        continuousBatchingEnabled: Bool = ProcessInfo.processInfo.environment["MERERUN_Q35_CONTINUOUS_BATCHING"] == "1",
        visionMinPixels: Int = Q35Generator.qwen3VLMinPixels,
        visionMaxPixels: Int = Q35Generator.qwen3VLMaxPixels
    ) {
        self.modelId = modelId
        self.prefixKVCacheEnabled = prefixKVCacheEnabled
        self.continuousBatchingEnabled = continuousBatchingEnabled
        self.visionMinPixels = visionMinPixels
        self.visionMaxPixels = visionMaxPixels
    }

    static func qwen3VLTargetSize(
        originalWidth width: Int,
        originalHeight height: Int,
        patchSize: Int,
        temporalPatchSize: Int,
        spatialMergeSize: Int,
        minPixels: Int = Q35Generator.qwen3VLMinPixels,
        maxPixels: Int = Q35Generator.qwen3VLMaxPixels
    ) -> (width: Int, height: Int) {
        let factor = max(1, patchSize * max(1, spatialMergeSize))
        let temporalFactor = max(1, temporalPatchSize)
        let frames = 1

        func roundedToFactor(_ value: Int) -> Int {
            max(factor, Int((Double(value) / Double(factor)).rounded()) * factor)
        }

        var targetHeight = roundedToFactor(height)
        var targetWidth = roundedToFactor(width)
        let temporalPaddedFrames = Int(ceil(Double(frames) / Double(temporalFactor))) * temporalFactor

        if temporalPaddedFrames * targetHeight * targetWidth > maxPixels {
            let beta = sqrt(Double(frames * height * width) / Double(maxPixels))
            targetHeight = max(factor, Int(floor(Double(height) / beta / Double(factor))) * factor)
            targetWidth = max(factor, Int(floor(Double(width) / beta / Double(factor))) * factor)
        } else if temporalPaddedFrames * targetHeight * targetWidth < minPixels {
            let beta = sqrt(Double(minPixels) / Double(frames * height * width))
            targetHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
            targetWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }

        return (targetWidth, targetHeight)
    }

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        let rootURL = try await resolveModelRoot(modelPath: nil, progressHandler: progressHandler)
        let loadStart = Date()
        try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
        let loadSeconds = Date().timeIntervalSince(loadStart)

        var response = try await generate(
            request,
            progressHandler: progressHandler,
            maxContextLength: request.maxContextTokens ?? Q35Resources.defaultContextLength
        )
        if var timing = response.timing {
            timing.loadSeconds = loadSeconds
            response.timing = timing
        } else {
            response.timing = ChatTiming(loadSeconds: loadSeconds)
        }
        return response
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

        var response = try await generate(
            request,
            progressHandler: progressHandler,
            maxContextLength: request.maxContextTokens ?? Q35Resources.defaultContextLength
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
        tokenizerAndTemplate = nil
        visionTower = nil
        mtpModel = nil
        loadedModelPath = nil
        loadedConfig = nil
        loadedResources = nil
        Memory.clearCache()
    }

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

    public func continuousBatchingStats() -> Q35ContinuousBatchingStats {
        Q35ContinuousBatchingStats(
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

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let normalizedRoot = Q35Resources.normalizedRootURL(rootURL)
        if loadedModelPath == normalizedRoot.path, model != nil, tokenizerAndTemplate != nil {
            return
        }
        resetPrefixKVCache()

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family config"))
        let configData = try Data(contentsOf: normalizedRoot.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Q35Config.self, from: configData)

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family tokenizer"))
        let tokenizer = try Q35TokenizerAndTemplate.load(
            from: normalizedRoot,
            maxLengthOverride: min(Q35Resources.defaultContextLength, config.textConfig.maxPositionEmbeddings)
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family weights"))
        let q35Model = Q35Model(config: config)
        let resources = Q35Resources(rootURL: normalizedRoot)

        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4

        try loadTextWeights(
            into: q35Model,
            from: resources,
            groupSize: groupSize,
            bits: bits
        )

        let tower = config.visionConfig == nil ? nil : Q35VisionTower(config: config)
        let mtpURL = normalizedRoot.appendingPathComponent("mtp.safetensors")
        // Load the MTP draft head whenever it ships with the model and isn't
        // explicitly disabled. Whether speculation is actually USED is decided
        // per request by prompt length (see Self.shouldSpeculate): MTP speculative
        // decode regresses at short context (measured ~-20-30%) but is a large win
        // at long context (~+1.5-2.5x past ~6-8K tokens) on both Metal and CUDA.
        let mtpExplicitlyDisabled = {
            guard let raw = ProcessInfo.processInfo.environment["MERERUN_Q35_MTP_SPECULATION"]?.lowercased()
            else { return false }
            return raw == "0" || raw == "false" || raw == "no"
        }()
        let loadedMTP: Q35MTPModel?
        if !mtpExplicitlyDisabled, FileManager.default.fileExists(atPath: mtpURL.path) {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family MTP weights"))
            let mtp = Q35MTPModel(config: config)
            let arrays = try MLX.loadArrays(url: mtpURL)
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                arrays,
                to: mtp,
                groupSize: groupSize,
                bits: bits,
                keyMapper: { key in
                    if key.hasPrefix("mtp.") {
                        return String(key.dropFirst("mtp.".count))
                    }
                    return "__unused__.\(key)"
                }
            )
            loadedMTP = mtp
        } else {
            loadedMTP = nil
        }

        model = q35Model
        tokenizerAndTemplate = tokenizer
        visionTower = tower
        mtpModel = loadedMTP
        loadedConfig = config
        loadedResources = resources
        loadedModelPath = normalizedRoot.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        maxContextLength: Int
    ) async throws -> ChatResponse {
        guard let model,
              let tokenizerAndTemplate,
              let loadedConfig else {
            throw Q35Error.modelNotLoaded
        }

        let messages = request.messages
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

        if !imageURLs.isEmpty {
            progressHandler?(ChatProgress(stage: .encoding, message: "Encoding images"))
            guard visionTower != nil else {
                throw Q35Error.generationFailed("Model \(modelId) does not include a vision tower; use text-only prompts.")
            }
            try ensureVisionWeightsLoaded(progressHandler: progressHandler)
            if let visionTower {
                visionReplacements = try buildVisionReplacements(
                    imageURLs: imageURLs,
                    visionTower: visionTower
                )
            }
        }

        var promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: messages,
            tools: request.tools,
            addGenerationPrompt: true,
            includeThinking: request.showThinking,
            maxLength: effectiveContext,
            imageTokenCounts: visionReplacements.map { max(1, $0.embeddings.dim(0)) }
        )
        if promptTokens.count > effectiveContext {
            promptTokens = Array(promptTokens.suffix(effectiveContext))
        }

        let eosSet = Set(loadedConfig.eosTokenIds + [tokenizerAndTemplate.eosTokenId].compactMap { $0 })
        let generationConfig = GenerationConfig(
            maxTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topP: Float(request.topP),
            repetitionPenalty: nil,
            repetitionContextSize: 64
        )
        let retainPrefillHidden =
            prefixKVCacheEnabled
            || (Self.shouldSpeculate(promptTokenCount: promptTokens.count, maxContextTokens: effectiveContext)
                && mtpModel != nil)

        var layerCaches = makeLayerCaches(config: loadedConfig)
        let promptInput = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, promptTokens.count)

        var prefillOutput: Q35PrefillOutput
        var prefillLength = promptTokens.count
        var mropeRopeDelta: Int?

        if imageURLs.isEmpty {
            let prefixSeed = prefixKVCacheSeed(
                modelPath: loadedModelPath ?? "",
                promptTokens: promptTokens
            )
            let prefixCheckpoints = semanticPrefixCheckpoints(
                tokenizerAndTemplate: tokenizerAndTemplate,
                messages: messages,
                tools: request.tools,
                includeThinking: request.showThinking,
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
            layerCaches: layerCaches,
            eosSet: eosSet,
            generationConfig: generationConfig,
            tokenBudget: tokenBudget,
            prefillTokenCount: prefillLength,
            mropeRopeDelta: mropeRopeDelta,
            promptTokens: promptTokens,
            maxContextTokens: effectiveContext,
            progressHandler: progressHandler
        )

        let stopSequences = TextGenerationStopSequences.merged(request.stopSequences)
        let decodedRaw = tokenizerAndTemplate.decode(tokens: decodeResult.generatedTokens)
        let trimmed = TextGenerationStopSequences.trimming(decodedRaw, sequences: stopSequences)
        let decoded = trimmed.text
        let finishReason: ChatFinishReason = {
            if trimmed.matchedSequence != nil {
                return .stopSequence
            }
            return decodeResult.generatedTokens.count >= tokenBudget ? .length : .stop
        }()
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
            promptTokens: promptTokens.count,
            finishReason: finishReason
        )
    }

    /// Decide whether to use MTP speculative decode for a request.
    ///
    /// Speculative decode only pays off at long context: each main-model pass gets
    /// more expensive as the KV cache grows, so verifying several drafted tokens per
    /// pass amortizes — but at short prompts the draft-head overhead dominates.
    /// Measured (Qwen3.6-35B-A3B OptiQ-4bit, M4 Max): ~20-tok ctx -31%, ~4K -22%,
    /// ~12K +1.5-2.5x. Default to speculating only when the prompt and request
    /// context are long; MERERUN_Q35_MTP_SPECULATION can enable/disable it, and
    /// MERERUN_Q35_MTP_MIN_PROMPT_TOKENS tunes the threshold.
    static func shouldSpeculate(promptTokenCount: Int, maxContextTokens: Int? = nil) -> Bool {
        shouldSpeculate(
            promptTokenCount: promptTokenCount,
            maxContextTokens: maxContextTokens,
            environment: ProcessInfo.processInfo.environment
        )
    }

    static func shouldSpeculate(
        promptTokenCount: Int,
        maxContextTokens: Int?,
        environment env: [String: String]
    ) -> Bool {
        let threshold = env["MERERUN_Q35_MTP_MIN_PROMPT_TOKENS"].flatMap { Int($0) } ?? 6144
        let contextAllowsSpeculation = maxContextTokens.map { $0 >= threshold } ?? true
        if let raw = env["MERERUN_Q35_MTP_SPECULATION"]?.lowercased() {
            if raw == "0" || raw == "false" || raw == "no" { return false }
            if raw == "1" || raw == "true" || raw == "yes" || raw == "on" {
                return contextAllowsSpeculation
            }
            // any other value (e.g. "auto") falls through to the adaptive threshold
        }
        if !contextAllowsSpeculation {
            return false
        }
        return promptTokenCount >= threshold
    }

    static func mtpBlockSize(environment env: [String: String] = ProcessInfo.processInfo.environment) -> Int {
        guard let raw = env["MERERUN_Q35_MTP_BLOCK_SIZE"],
              let value = Int(raw), value >= 2 else {
            return defaultMTPBlockSize
        }
        return min(16, value)
    }

    private func decodeTokens(
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate,
        initialLogits: MLXArray,
        initialHidden: MLXArray?,
        layerCaches: [Q35LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        prefillTokenCount: Int,
        mropeRopeDelta: Int?,
        promptTokens: [Int],
        maxContextTokens: Int,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35BatchedDecodeResult {
        guard tokenBudget > 0 else {
            return Q35BatchedDecodeResult(generatedTokens: [], decodeSeconds: 0)
        }
        guard continuousBatchingEnabled else {
            // Use the loaded MTP head only when the prompt is long enough for
            // speculation to pay off; otherwise decode without it (nil).
            let speculationMTP = Self.shouldSpeculate(
                promptTokenCount: promptTokens.count,
                maxContextTokens: maxContextTokens
            ) && mropeRopeDelta == nil ? mtpModel : nil
            return try await decodeTokensSerially(
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate,
                initialLogits: initialLogits,
                initialHidden: initialHidden,
                mtpModel: speculationMTP,
                layerCaches: layerCaches,
                eosSet: eosSet,
                generationConfig: generationConfig,
                tokenBudget: tokenBudget,
                prefillTokenCount: prefillTokenCount,
                mropeRopeDelta: mropeRopeDelta,
                promptTokens: promptTokens,
                progressHandler: progressHandler
            )
        }

        let rowID = UUID()
        let initialLogitsBox = RuntimeUncheckedSendable(initialLogits)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let row = Q35BatchedDecodeRow(
                    id: rowID,
                    logits: initialLogitsBox.value,
                    layerCaches: layerCaches,
                    eosSet: eosSet,
                    generationConfig: generationConfig,
                    tokenBudget: tokenBudget,
                    prefillTokenCount: prefillTokenCount,
                    mropeRopeDelta: mropeRopeDelta,
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
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate,
        initialLogits: MLXArray,
        initialHidden: MLXArray?,
        mtpModel: Q35MTPModel?,
        layerCaches: [Q35LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        prefillTokenCount: Int,
        mropeRopeDelta: Int?,
        promptTokens: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35BatchedDecodeResult {
        var logits = initialLogits
        var layerCaches = layerCaches
        let retainHidden = mtpModel != nil
        var previousHidden = retainHidden ? initialHidden.map(lastTokenHidden) : nil
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

            let visiblePiece: String
            if pendingProgressWhitespace.isEmpty {
                visiblePiece = piece
            } else {
                visiblePiece = pendingProgressWhitespace + piece
                pendingProgressWhitespace = ""
            }
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

            if let mtpModel, let hidden = previousHidden {
                let positionOffset = prefillTokenCount + generated.count - 1
                if generationConfig.temperature == 0 {
                    let blockSize = min(
                        Self.mtpBlockSize(),
                        max(2, tokenBudget - generated.count + 1)
                    )
                    let draftTokens = mtpModel.draftBlock(
                        lastToken: next,
                        hidden: hidden,
                        positionOffset: positionOffset,
                        blockSize: blockSize,
                        baseModel: model
                    )
                    guard !draftTokens.isEmpty else {
                        continue
                    }

                    let candidateCaches = forkLayerCaches(layerCaches)
                    let candidateInput = MLXArray(([next] + draftTokens).map(Int32.init))
                        .reshaped(1, draftTokens.count + 1)
                    let candidate = model.forward(candidateInput, cache: candidateCaches)
                    MLX.eval(candidate.logits)
                    MLX.eval(candidate.hidden)

                    var accepted = 0
                    var verificationHistory = repetitionHistory
                    var replacement: Int?
                    for (index, draftToken) in draftTokens.enumerated() {
                        let targetToken = sampleToken(
                            logits: candidate.logits[0, index, 0...],
                            config: generationConfig,
                            previousTokens: verificationHistory
                        )
                        guard targetToken == draftToken else {
                            replacement = targetToken
                            break
                        }
                        accepted += 1
                        verificationHistory.append(draftToken)
                    }

                    if accepted == draftTokens.count {
                        var hitEOS = false
                        for token in draftTokens {
                            if eosSet.contains(token) {
                                hitEOS = true
                                break
                            }
                            emit(token)
                        }
                        layerCaches = candidateCaches
                        logits = lastTokenLogits(candidate.logits)
                        previousHidden = lastTokenHidden(candidate.hidden)
                        if hitEOS || generated.count >= tokenBudget {
                            break
                        }
                        continue
                    }

                    let acceptedPrefix = Array(draftTokens.prefix(accepted))
                    var hitEOS = false
                    for token in acceptedPrefix {
                        if eosSet.contains(token) {
                            hitEOS = true
                            break
                        }
                        emit(token)
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

                    let replacementCaches = forkLayerCaches(layerCaches)
                    let replacementInput = MLXArray(([next] + acceptedPrefix + [replacement]).map(Int32.init))
                        .reshaped(1, acceptedPrefix.count + 2)
                    let replacementForward = model.forward(replacementInput, cache: replacementCaches)
                    MLX.eval(replacementForward.logits)
                    MLX.eval(replacementForward.hidden)
                    emit(replacement)
                    layerCaches = replacementCaches
                    logits = lastTokenLogits(replacementForward.logits)
                    previousHidden = lastTokenHidden(replacementForward.hidden)
                    continue
                }

                let draftLogits = mtpModel.draftLogits(
                    token: next,
                    previousHidden: hidden,
                    positionOffset: positionOffset,
                    baseModel: model
                )
                MLX.eval(draftLogits)

                let draftProbs = samplingProbabilities(
                    logits: draftLogits[0, -1, 0...],
                    config: generationConfig,
                    previousTokens: repetitionHistory
                )
                let draft = sampleToken(probabilities: draftProbs)

                let candidateCaches = forkLayerCaches(layerCaches)
                let candidateInput = MLXArray([Int32(next), Int32(draft)]).reshaped(1, 2)
                let candidate = model.forward(candidateInput, cache: candidateCaches)
                MLX.eval(candidate.logits)
                MLX.eval(candidate.hidden)

                let targetProbs = samplingProbabilities(
                    logits: candidate.logits[0, 0, 0...],
                    config: generationConfig,
                    previousTokens: repetitionHistory
                )
                let draftProb = max(draftProbs[draft].item(Float.self), Float.leastNonzeroMagnitude)
                let targetProb = targetProbs[draft].item(Float.self)
                let acceptProbability = min(1.0, targetProb / draftProb)

                if Float.random(in: 0..<1) <= acceptProbability {
                    if eosSet.contains(draft) {
                        break
                    }
                    emit(draft)
                    layerCaches = candidateCaches
                    logits = lastTokenLogits(candidate.logits)
                    previousHidden = lastTokenHidden(candidate.hidden)
                    continue
                }

                let residualProbs = MLX.maximum(targetProbs - draftProbs, MLXArray(0.0))
                let residualMass = residualProbs.sum().item(Float.self)
                let replacement = residualMass > 1e-6
                    ? sampleToken(probabilities: residualProbs / residualProbs.sum())
                    : sampleToken(probabilities: targetProbs)
                if eosSet.contains(replacement) {
                    break
                }

                let replacementCaches = forkLayerCaches(layerCaches)
                let replacementInput = MLXArray([Int32(next), Int32(replacement)]).reshaped(1, 2)
                let replacementForward = model.forward(replacementInput, cache: replacementCaches)
                MLX.eval(replacementForward.logits)
                MLX.eval(replacementForward.hidden)
                emit(replacement)
                layerCaches = replacementCaches
                logits = lastTokenLogits(replacementForward.logits)
                previousHidden = lastTokenHidden(replacementForward.hidden)
                continue
            }

            let nextInput = MLXArray([Int32(next)]).reshaped(1, 1)
            let positionIds = decodePositionIds(layerCaches: layerCaches, tokenCount: 1, ropeDelta: mropeRopeDelta)
            if retainHidden {
                let output = model.forward(
                    nextInput,
                    cache: layerCaches,
                    positionIds: positionIds
                )
                logits = output.logits
                previousHidden = lastTokenHidden(output.hidden)
                MLX.eval(logits)
                MLX.eval(previousHidden!)
            } else {
                logits = model(
                    nextInput,
                    cache: layerCaches,
                    positionIds: positionIds
                )
                MLX.eval(logits)
            }
        }

        return Q35BatchedDecodeResult(
            generatedTokens: generated,
            decodeSeconds: Date().timeIntervalSince(decodeStart)
        )
    }

    private func enqueueDecodeRow(
        _ row: Q35BatchedDecodeRow,
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate
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
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate
    ) {
        guard !decodeLoopRunning else { return }
        decodeLoopRunning = true
        Task {
            await runDecodeLoop(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
        }
    }

    private func runDecodeLoop(
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate
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

    private func selectDecodeRows() -> [Q35BatchedDecodeRow] {
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

    private func decodeBatchSignature(for row: Q35BatchedDecodeRow) -> String {
        row.layerCaches
            .map { cache in
                switch cache {
                case .full(let kv):
                    if kv.supportsVariablePositionBatching {
                        return "full:variable"
                    }
                    return "full:\(kv.offset)"
                case .linear(let linear):
                    return "linear:\(Q35Generator.linearCacheSignature(linear))"
                case nil:
                    return "nil"
                }
            }
            .joined(separator: "|")
    }

    private static func linearCacheSignature(_ cache: Q35LinearCache) -> String {
        let convShape = cache.convState?.shape.map(String.init).joined(separator: "x") ?? "nil"
        let recurrentShape = cache.recurrentState?.shape.map(String.init).joined(separator: "x") ?? "nil"
        return "\(convShape):\(recurrentShape)"
    }

    private func decodePosition(_ row: Q35BatchedDecodeRow) -> Int {
        row.layerCaches.compactMap { cache in
            if case .full(let kv)? = cache {
                return kv.offset
            }
            return nil
        }.min() ?? (row.prefillTokenCount + row.generatedTokens.count)
    }

    private func decodePositionIds(
        layerCaches: [Q35LayerCache?],
        tokenCount: Int,
        ropeDelta: Int?
    ) -> MLXArray? {
        guard let ropeDelta, tokenCount > 0 else { return nil }
        let offset = layerCaches.compactMap { cache in
            if case .full(let kv)? = cache {
                return kv.offset
            }
            return nil
        }.min() ?? 0
        let positions = (0..<tokenCount).map { Int32(offset + ropeDelta + $0) }
        let values = positions + positions + positions
        return MLXArray(values, [3, 1, tokenCount])
    }

    private func batchedDecodePositionIds(rows: [Q35BatchedDecodeRow]) -> MLXArray? {
        guard rows.contains(where: { $0.mropeRopeDelta != nil }) else { return nil }
        var values: [Int32] = []
        values.reserveCapacity(rows.count * 3)
        for _ in 0..<3 {
            for row in rows {
                values.append(Int32(decodePosition(row) + (row.mropeRopeDelta ?? 0)))
            }
        }
        return MLXArray(values, [3, rows.count, 1])
    }

    private func decodeOneStep(
        rows: [Q35BatchedDecodeRow],
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate
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
            row.repetitionHistory.append(next)
            let piece = tokenizerAndTemplate.decode(token: next)
            if !piece.isEmpty {
                row.progressHandler?(ChatProgress(stage: .generating, message: piece))
            }
        }

        let continuingRows = sampledRows.filter(\.needsDecodeStep)
        guard !continuingRows.isEmpty else { return }

        if continuingRows.count > 1,
           let batchedCaches = makeBatchedLayerCaches(continuingRows.map(\.layerCaches)) {
            let nextInput = MLXArray(continuingRows.compactMap { $0.generatedTokens.last }.map(Int32.init))
                .reshaped(continuingRows.count, 1)
            let batchedLogits = model(
                nextInput,
                cache: batchedCaches,
                positionIds: batchedDecodePositionIds(rows: continuingRows)
            )
            MLX.eval(batchedLogits)
            guard let splitCaches = splitBatchedLayerCaches(batchedCaches, rowCount: continuingRows.count) else {
                throw Q35Error.generationFailed("Qwen-family batched decode could not split merged cache rows.")
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
            row.logits = model(
                nextInput,
                cache: row.layerCaches,
                positionIds: decodePositionIds(
                    layerCaches: row.layerCaches,
                    tokenCount: 1,
                    ropeDelta: row.mropeRopeDelta
                )
            )
            MLX.eval(row.logits)
            singleDecodeSteps += 1
        }
    }

    private func makeBatchedLayerCaches(_ rowCaches: [[Q35LayerCache?]]) -> [Q35LayerCache?]? {
        guard let first = rowCaches.first, !first.isEmpty else { return nil }
        guard rowCaches.allSatisfy({ $0.count == first.count }) else { return nil }

        var result: [Q35LayerCache?] = []
        result.reserveCapacity(first.count)
        for layerIndex in first.indices {
            let layerCaches = rowCaches.map { $0[layerIndex] }
            if layerCaches.allSatisfy({ $0 == nil }) {
                result.append(nil)
                continue
            }
            let nonNil = layerCaches.compactMap { $0 }
            guard nonNil.count == layerCaches.count,
                  let batched = nonNil[0].batched(with: nonNil) else {
                return nil
            }
            result.append(batched)
        }
        return result
    }

    private func splitBatchedLayerCaches(
        _ caches: [Q35LayerCache?],
        rowCount: Int
    ) -> [[Q35LayerCache?]]? {
        guard rowCount > 0 else { return nil }
        var rows = Array(repeating: [Q35LayerCache?](), count: rowCount)
        for cache in caches {
            guard let cache else {
                for index in 0..<rowCount {
                    rows[index].append(nil)
                }
                continue
            }
            guard let split = cache.unbatchedRows(count: rowCount), split.count == rowCount else {
                return nil
            }
            for index in 0..<rowCount {
                rows[index].append(split[index])
            }
        }
        return rows
    }

    private func finishCompletedDecodeRows() {
        var remaining: [Q35BatchedDecodeRow] = []
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

    private func failRows(_ rows: [Q35BatchedDecodeRow], with error: Error) {
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

    private func lastTokenHidden(_ hidden: MLXArray) -> MLXArray {
        let start = max(0, hidden.dim(1) - 1)
        return hidden[0..., start..<(start + 1), 0...]
    }

    private func lastTokenLogits(_ logits: MLXArray) -> MLXArray {
        let start = max(0, logits.dim(1) - 1)
        return logits[0..., start..<(start + 1), 0...]
    }

    private func chunkedPrefill(
        model: Q35Model,
        promptTokens: [Int],
        cache: [Q35LayerCache?],
        startIndex: Int = 0,
        existingLogits: MLXArray? = nil,
        existingHidden: MLXArray? = nil,
        modelPath: String? = nil,
        checkpointTokenCounts: Set<Int> = [],
        retainHidden: Bool = true,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35PrefillOutput {
        guard !promptTokens.isEmpty else {
            throw Q35Error.generationFailed("Prompt is empty after tokenization.")
        }

        var processed = startIndex
        var logits = existingLogits
        var hidden = existingHidden
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
            if retainHidden {
                let output = model.forward(chunk, cache: cache)
                MLX.eval(output.logits)
                MLX.eval(output.hidden)
                logits = output.logits
                hidden = output.hidden
                if let modelPath {
                    storePrefixKVCache(
                        modelPath: modelPath,
                        promptTokens: promptTokens,
                        tokenCount: end,
                        cache: cache,
                        logits: output.logits,
                        hidden: output.hidden,
                        priority: checkpointTokenCounts.contains(end) ? .semantic : .chunk
                    )
                }
            } else {
                let output = model(chunk, cache: cache)
                MLX.eval(output)
                logits = output
                hidden = nil
            }
            processed = end
            await Task.yield()
        }

        guard let logits else {
            throw Q35Error.generationFailed("Prefill did not produce logits.")
        }
        return Q35PrefillOutput(logits: logits, hidden: hidden)
    }

    private func chunkedPrefillEmbeddings(
        model: Q35Model,
        inputEmbeddings: MLXArray,
        cache: [Q35LayerCache?],
        positionIds: MLXArray? = nil,
        retainHidden: Bool = true,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35PrefillOutput {
        let tokenCount = inputEmbeddings.dim(1)
        guard tokenCount > 0 else {
            throw Q35Error.generationFailed("Prompt embeddings are empty after tokenization.")
        }

        var processed = 0
        var logits: MLXArray?
        var hidden: MLXArray?
        while processed < tokenCount {
            try Task.checkCancellation()
            let end = min(processed + Self.prefillChunkSize, tokenCount)
            if tokenCount > Self.prefillChunkSize {
                progressHandler?(ChatProgress(stage: .encoding, message: "Prefilling \(end)/\(tokenCount) tokens"))
            }
            let chunkEmbeddings = inputEmbeddings[0..., processed..<end, 0...]
            let chunkInput = MLXArray.zeros([1, end - processed], dtype: .int32)
            let chunkPositionIds = positionIds?[0..., 0..., processed..<end]
            if retainHidden {
                let output = model.forward(
                    chunkInput,
                    cache: cache,
                    inputEmbeddings: chunkEmbeddings,
                    positionIds: chunkPositionIds
                )
                MLX.eval(output.logits)
                MLX.eval(output.hidden)
                logits = output.logits
                hidden = output.hidden
            } else {
                let output = model(
                    chunkInput,
                    cache: cache,
                    inputEmbeddings: chunkEmbeddings,
                    positionIds: chunkPositionIds
                )
                MLX.eval(output)
                logits = output
                hidden = nil
            }
            processed = end
            await Task.yield()
        }

        guard let logits else {
            throw Q35Error.generationFailed("Prefill did not produce logits.")
        }
        return Q35PrefillOutput(logits: logits, hidden: hidden)
    }

    private func semanticPrefixCheckpoints(
        tokenizerAndTemplate: Q35TokenizerAndTemplate,
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
        guard !prefixMessages.isEmpty else {
            return []
        }
        guard let prefixTokens = try? tokenizerAndTemplate.encodeForGeneration(
            messages: prefixMessages,
            tools: tools,
            addGenerationPrompt: false,
            includeThinking: includeThinking,
            maxLength: maxContextLength
        ) else {
            return []
        }
        guard promptTokens.starts(with: prefixTokens) else {
            return []
        }
        return RuntimePrefillCheckpointPlanner.normalizedCheckpoints(
            [prefixTokens.count],
            total: promptTokens.count
        )
    }

    private func prefixKVCacheSeed(
        modelPath: String,
        promptTokens: [Int]
    ) -> (tokenCount: Int, caches: [Q35LayerCache?], logits: MLXArray, hidden: MLXArray)? {
        guard prefixKVCacheEnabled else { return nil }
        let matchingKey = prefixKVCache.keys
            .filter { key in
                key.modelPath == modelPath
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
            forkLayerCaches(entry.caches),
            entry.logits,
            entry.hidden
        )
    }

    private func storePrefixKVCache(
        modelPath: String,
        promptTokens: [Int],
        tokenCount: Int,
        cache: [Q35LayerCache?],
        logits: MLXArray,
        hidden: MLXArray,
        priority: RuntimePrefixCacheEntryPriority
    ) {
        guard prefixKVCacheEnabled, tokenCount > 0 else { return }
        let key = Q35PrefixKVCacheKey(
            modelPath: modelPath,
            tokens: Array(promptTokens.prefix(tokenCount))
        )
        prefixKVCache[key] = Q35PrefixKVCacheEntry(
            caches: forkLayerCaches(cache),
            logits: logits,
            hidden: hidden,
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

    private func forkLayerCaches(_ caches: [Q35LayerCache?]) -> [Q35LayerCache?] {
        caches.map { $0?.fork() }
    }

    private func loadTextWeights(
        into q35Model: Q35Model,
        from resources: Q35Resources,
        groupSize: Int,
        bits: Int
    ) throws {
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            guard let mapped = Self.mapTextWeightKey(key) else { return [] }
            if q35Model.config.tieWordEmbeddings, mapped == "lm_head.weight" {
                return []
            }
            if let splitExperts = Self.splitMappedExpertGateUpWeight(mapped, value) {
                return splitExperts
            }
            let normalizedMapped = Self.normalizeMappedExpertWeightKey(mapped)
            if normalizedMapped.hasSuffix(".linear_attn.conv1d.weight"), value.ndim == 3 {
                return [(normalizedMapped, Self.normalizedLinearAttentionConv1DWeight(value))]
            }
            if Self.isOffsetRMSNormWeight(normalizedMapped) {
                return [(normalizedMapped, value - MLXArray(1.0).asType(value.dtype))]
            }
            return [(normalizedMapped, value)]
        }
        let keyMapper: (String) -> String = { key in
            guard let mapped = Self.mapTextWeightKey(key) else { return "__unused__.\(key)" }
            if q35Model.config.tieWordEmbeddings, mapped == "lm_head.weight" {
                return "__unused__.\(key)"
            }
            return Self.normalizeMappedExpertWeightKey(mapped)
        }
        let quantizedMapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            guard !key.hasPrefix("__unused__.") else { return [] }
            if key.hasSuffix(".linear_attn.conv1d.weight"), value.ndim == 3 {
                return [(key, Self.normalizedLinearAttentionConv1DWeight(value))]
            }
            if Self.isOffsetRMSNormWeight(key) {
                return [(key, value - MLXArray(1.0).asType(value.dtype))]
            }
            return [(key, value)]
        }

        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            if try Self.indexContainsQuantizedWeights(resources.modelIndexURL) {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: resources.modelIndexURL,
                    to: q35Model,
                    groupSize: groupSize,
                    bits: bits,
                    keyMapper: keyMapper,
                    mapper: quantizedMapper
                )
            } else {
                try HFSafetensorsWeightsLoader.applyShardedWeights(
                    indexURL: resources.modelIndexURL,
                    to: q35Model,
                    dtype: .bfloat16,
                    verify: .none,
                    mapper: mapper
                )
            }
            return
        }

        let arrays = try MLX.loadArrays(url: resources.modelWeightsURL)
        if HFSafetensorsWeightsLoader.isQuantized(arrays) {
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                arrays,
                to: q35Model,
                groupSize: groupSize,
                bits: bits,
                keyMapper: keyMapper,
                mapper: quantizedMapper
            )
        } else {
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: resources.modelWeightsURL,
                to: q35Model,
                dtype: .bfloat16,
                verify: .none,
                include: { Self.mapTextWeightKey($0) != nil },
                mapper: mapper,
                batchSize: 32
            )
        }
    }

    private static func mapTextWeightKey(_ key: String) -> String? {
        if key.hasPrefix("lm_head.") {
            return key
        }
        if key.hasPrefix("model.language_model.") {
            return mapLanguageModelWeightSuffix(String(key.dropFirst("model.language_model.".count)))
        }
        if key.hasPrefix("language_model.") {
            return mapLanguageModelWeightSuffix(String(key.dropFirst("language_model.".count)))
        }
        return nil
    }

    private static func mapLanguageModelWeightSuffix(_ suffix: String) -> String {
        if suffix.hasPrefix("model.") || suffix.hasPrefix("lm_head.") {
            return suffix
        }
        return "model.\(suffix)"
    }

    private static func normalizeMappedExpertWeightKey(_ key: String) -> String {
        let expertDownSuffix = ".mlp.experts.down_proj"
        if key.hasSuffix(expertDownSuffix) {
            return String(key.dropLast(expertDownSuffix.count)) + ".mlp.switch_mlp.down_proj.weight"
        }
        return key
    }

    static func normalizedLinearAttentionConv1DWeight(_ value: MLXArray) -> MLXArray {
        guard value.ndim == 3, value.dim(1) == 1, value.dim(2) > 1 else {
            return value
        }
        let transposed = value.transposed(0, 2, 1)
        return transposed.reshaped(-1).reshaped(transposed.shape)
    }

    static func isOffsetRMSNormWeight(_ key: String) -> Bool {
        key.hasSuffix(".input_layernorm.weight")
            || key.hasSuffix(".post_attention_layernorm.weight")
            || key.hasSuffix(".self_attn.q_norm.weight")
            || key.hasSuffix(".self_attn.k_norm.weight")
            || key == "model.norm.weight"
    }

    private static func splitMappedExpertGateUpWeight(_ key: String, _ value: MLXArray) -> [(String, MLXArray)]? {
        let expertGateUpSuffix = ".mlp.experts.gate_up_proj"
        guard key.hasSuffix(expertGateUpSuffix), value.ndim == 3 else {
            return nil
        }

        let fusedDim = value.dim(1)
        guard fusedDim > 0, fusedDim.isMultiple(of: 2) else {
            return nil
        }

        let intermediate = fusedDim / 2
        let base = String(key.dropLast(expertGateUpSuffix.count)) + ".mlp.switch_mlp"
        return [
            ("\(base).gate_proj.weight", value[0..., 0..<intermediate, 0...]),
            ("\(base).up_proj.weight", value[0..., intermediate..., 0...]),
        ]
    }

    private static func indexContainsQuantizedWeights(_ indexURL: URL) throws -> Bool {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
        return index.weightMap.keys.contains { $0.hasSuffix(".scales") }
    }

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        guard let profile = Q35Resources.profile(for: modelId) else {
            throw Q35Error.unsupportedModelId(modelId)
        }

        do {
            let root = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelPath ?? modelId,
                defaultModelID: profile.modelId,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Downloading model... \(percent)%"))
                    case .extracting:
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Extracting model..."))
                    }
                }
            )
            return Q35Resources.normalizedRootURL(root.url)
        } catch let error as ManagedModelResolver.ResolverError {
            throw Q35Error.downloadFailed(error.localizedDescription)
        }
    }

    private func mapLoaderError(_ error: PretrainedModelLoader.LoadError) -> Q35Error {
        switch error {
        case .unsupportedModelId(let modelId):
            return .unsupportedModelId(modelId)
        case .missingFiles(let files):
            return .missingFiles(files)
        case .downloadFailed(let message):
            return .downloadFailed(message)
        }
    }

    private func makeLayerCaches(config: Q35Config) -> [Q35LayerCache?] {
        let text = config.textConfig
        let mlpOnly = Set(text.mlpOnlyLayers)
        return (0..<text.numHiddenLayers).map { layerIndex in
            if mlpOnly.contains(layerIndex) {
                return nil
            }
            let layerType = layerIndex < text.layerTypes.count ? text.layerTypes[layerIndex] : "linear_attention"
            if layerType == "full_attention" {
                return .full(KVCacheSimple(step: 256))
            }
            return .linear(Q35LinearCache())
        }
    }

    private func collectImageURLs(from messages: [ChatMessage]) -> [String] {
        messages.compactMap { message in
            guard message.role != .system else { return nil }
            guard let url = message.imageUrl?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !url.isEmpty else { return nil }
            return url
        }
    }

    private func ensureVisionWeightsLoaded(
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws {
        guard let visionTower else { return }
        guard !visionTower.isLoaded else { return }
        guard let loadedResources else { throw Q35Error.modelNotLoaded }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family vision tower"))
        try visionTower.loadWeights(from: loadedResources)
    }

    private func buildVisionReplacements(
        imageURLs: [String],
        visionTower: Q35VisionTower
    ) throws -> [Q35VisionReplacement] {
        var replacements: [Q35VisionReplacement] = []
        replacements.reserveCapacity(imageURLs.count)

        for imageURL in imageURLs {
            let prepared = try loadImageTensor(
                from: imageURL,
                patchSize: visionTower.patchSize,
                spatialMergeSize: visionTower.spatialMergeSize
            )
            let embeds = try visionTower.encodeImage(
                pixelValues: prepared.tensor,
                gridTHW: prepared.gridTHW
            )
            replacements.append(Q35VisionReplacement(embeddings: embeds, gridTHW: prepared.gridTHW))
        }

        return replacements
    }

    private func buildMRoPEPositionData(
        inputIds: MLXArray,
        imageTokenId: Int,
        replacements: [Q35VisionReplacement],
        spatialMergeSize: Int
    ) throws -> Q35MRoPEPositionData? {
        guard !replacements.isEmpty else { return nil }

        let seqLen = inputIds.dim(1)
        let tokenArray = inputIds.asType(.int32)
        MLX.eval(tokenArray)
        let tokenValues = tokenArray.asArray(Int32.self)

        var axes = Array(repeating: [Int](), count: 3)
        for axis in axes.indices {
            axes[axis].reserveCapacity(seqLen)
        }

        let mergeSize = max(1, spatialMergeSize)
        var cursor = 0
        var currentPosition = 0
        var replacementIndex = 0

        func appendText(count: Int) {
            guard count > 0 else { return }
            for offset in 0..<count {
                let position = currentPosition + offset
                for axis in axes.indices {
                    axes[axis].append(position)
                }
            }
            currentPosition += count
        }

        func appendVision(startPosition: Int, gridTHW: (Int, Int, Int)) -> Int {
            let gridT = max(1, gridTHW.0)
            let gridH = max(1, gridTHW.1 / mergeSize)
            let gridW = max(1, gridTHW.2 / mergeSize)
            for t in 0..<gridT {
                for h in 0..<gridH {
                    for w in 0..<gridW {
                        axes[0].append(startPosition + t)
                        axes[1].append(startPosition + h)
                        axes[2].append(startPosition + w)
                    }
                }
            }
            return max(gridH, gridW)
        }

        while cursor < seqLen {
            if tokenValues[cursor] != Int32(imageTokenId) {
                let textStart = cursor
                while cursor < seqLen, tokenValues[cursor] != Int32(imageTokenId) {
                    cursor += 1
                }
                appendText(count: cursor - textStart)
                continue
            }

            let runStart = cursor
            while cursor < seqLen, tokenValues[cursor] == Int32(imageTokenId) {
                cursor += 1
            }
            let runLength = cursor - runStart
            guard replacementIndex < replacements.count else {
                throw Q35Error.generationFailed("Qwen-family M-RoPE found more image-token runs than encoded images.")
            }

            let replacement = replacements[replacementIndex]
            guard runLength == replacement.embeddings.dim(0) else {
                throw Q35Error.generationFailed(
                    "Qwen-family image-token span mismatch: prompt has \(runLength) placeholders, vision tower produced \(replacement.embeddings.dim(0)) tokens."
                )
            }

            currentPosition += appendVision(
                startPosition: currentPosition,
                gridTHW: replacement.gridTHW
            )
            replacementIndex += 1
        }

        guard replacementIndex == replacements.count else {
            throw Q35Error.generationFailed("Qwen-family M-RoPE received encoded images without matching image-token runs.")
        }
        guard axes.allSatisfy({ $0.count == seqLen }) else {
            throw Q35Error.generationFailed("Qwen-family M-RoPE position length did not match prompt length.")
        }

        let maxPosition = axes.flatMap { $0 }.max() ?? (seqLen - 1)
        let ropeDelta = maxPosition + 1 - seqLen
        let values = axes.flatMap { $0.map(Int32.init) }
        return Q35MRoPEPositionData(
            positionIds: MLXArray(values, [3, 1, seqLen]),
            ropeDelta: ropeDelta
        )
    }

    private func insertVisionEmbeddings(
        hiddenStates: MLXArray,
        inputIds: MLXArray,
        imageTokenId: Int,
        replacements: [Q35VisionReplacement]
    ) -> MLXArray {
        guard !replacements.isEmpty else { return hiddenStates }

        let seqLen = hiddenStates.dim(1)
        let tokenArray = inputIds.asType(.int32)
        MLX.eval(tokenArray)
        let tokenValues = tokenArray.asArray(Int32.self)

        var positions: [Int] = []
        positions.reserveCapacity(replacements.count)
        for index in 0..<seqLen where tokenValues[index] == Int32(imageTokenId) {
            positions.append(index)
        }

        guard !positions.isEmpty else { return hiddenStates }

        var runs: [(start: Int, end: Int)] = []
        for position in positions {
            if let last = runs.last, last.end == position {
                runs[runs.count - 1] = (start: last.start, end: position + 1)
            } else {
                runs.append((start: position, end: position + 1))
            }
        }

        let pairCount = min(runs.count, replacements.count)

        var parts: [MLXArray] = []
        parts.reserveCapacity(pairCount * 2 + 1)

        var cursor = 0
        for pairIndex in 0..<pairCount {
            let run = runs[pairIndex]
            if run.start > cursor {
                parts.append(hiddenStates[0..., cursor..<run.start, 0...])
            }

            var replacement = replacements[pairIndex].embeddings
            if replacement.dtype != hiddenStates.dtype {
                replacement = replacement.asType(hiddenStates.dtype)
            }
            parts.append(replacement.expandedDimensions(axis: 0))

            cursor = run.end
        }

        if cursor < seqLen {
            parts.append(hiddenStates[0..., cursor..., 0...])
        }

        if parts.isEmpty {
            return hiddenStates
        }
        if parts.count == 1 {
            return parts[0]
        }
        return MLX.concatenated(parts, axis: 1)
    }

    private func loadImageTensor(
        from imageRef: String,
        patchSize: Int,
        spatialMergeSize: Int
    ) throws -> (tensor: MLXArray, gridTHW: (Int, Int, Int)) {
        let image = try loadImage(from: imageRef)
        let target = Self.qwen3VLTargetSize(
            originalWidth: image.width,
            originalHeight: image.height,
            patchSize: patchSize,
            temporalPatchSize: visionTower?.temporalPatchSize ?? 2,
            spatialMergeSize: spatialMergeSize,
            minPixels: visionMinPixels,
            maxPixels: visionMaxPixels
        )
        let resized = try MediaImageIO.resized(
            image,
            width: target.width,
            height: target.height
        )
        let floats = MediaImageIO.rgbCHWFloat(resized, normalizedToMinusOneToOne: true)
        let pixels = MLXArray(
            floats,
            [1, 3, resized.height, resized.width]
        )
        let gridTHW = (
            1,
            resized.height / patchSize,
            resized.width / patchSize
        )
        return (pixels, gridTHW)
    }

    private func loadImage(from imageRef: String) throws -> MediaImage {
        if let remoteURL = URL(string: imageRef),
           let scheme = remoteURL.scheme?.lowercased(),
           scheme == "http" || scheme == "https" {
            let data = try Data(contentsOf: remoteURL)
            do {
                return try MediaImageIO.decode(data: data)
            } catch {
                throw NSError(
                    domain: "Q35Generator",
                    code: 1002,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to decode image URL: \(imageRef)"]
                )
            }
        }

        let localURL: URL
        if imageRef.hasPrefix("file://"), let parsed = URL(string: imageRef) {
            localURL = parsed
        } else {
            localURL = URL(fileURLWithPath: imageRef)
        }
        guard FileManager.default.fileExists(atPath: localURL.path) else {
            throw NSError(
                domain: "Q35Generator",
                code: 1003,
                userInfo: [NSLocalizedDescriptionKey: "Image file not found: \(imageRef)"]
            )
        }
        do {
            return try MediaImageIO.decode(localURL)
        } catch {
            throw NSError(
                domain: "Q35Generator",
                code: 1004,
                userInfo: [NSLocalizedDescriptionKey: "Failed to decode image file: \(imageRef)"]
            )
        }
    }
}
