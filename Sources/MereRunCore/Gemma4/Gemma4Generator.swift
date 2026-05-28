import Foundation
import MLX

public typealias Gemma4PrefixKVCacheStats = PrefixKVCacheStats
public typealias Gemma4ContinuousBatchingStats = RuntimeDecodeBatchingStats

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
                firstTokenSeconds: firstTokenSeconds
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

    private var model: Gemma4TextCausalLM?
    private var tokenizerAndTemplate: Gemma4TokenizerAndTemplate?
    private var loadedModelPath: String?
    private var loadedConfig: Gemma4Config?

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
        try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
        let loadSeconds = Date().timeIntervalSince(loadStart)

        var response = try await generate(
            request,
            progressHandler: progressHandler,
            maxContextLength: Gemma4Resources.defaultContextLength
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
        loadedModelPath = nil
        loadedConfig = nil
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
        let textModel = Gemma4TextCausalLM(config: config.textConfig)
        try loadWeights(into: textModel, from: resources, config: config)

        model = textModel
        tokenizerAndTemplate = tokenizer
        loadedConfig = config
        loadedModelPath = normalizedRoot.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        maxContextLength: Int
    ) async throws -> ChatResponse {
        guard let model, let tokenizerAndTemplate, let loadedConfig else {
            throw Gemma4Error.modelNotLoaded
        }
        let kvCacheQuantization = try self.kvCacheQuantization.validated()
        let prefillKVCacheQuantization = prefillQuantization(for: kvCacheQuantization)

        let effectiveContext = min(maxContextLength, loadedConfig.textConfig.maxPositionEmbeddings)
        let prefillStart = Date()
        let messages = request.messages
        var promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: messages,
            tools: request.tools,
            addGenerationPrompt: true,
            includeThinking: request.showThinking,
            maxLength: effectiveContext
        )
        if promptTokens.count > effectiveContext {
            promptTokens = Array(promptTokens.suffix(effectiveContext))
        }

        let hasTools = request.tools?.isEmpty == false
        let eosSet = request.stopOnEOS
            ? Set(loadedConfig.eosTokenIds + tokenizerAndTemplate.stopTokenIds(withTools: hasTools))
            : []
        let generationConfig = GenerationConfig(
            maxTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topP: Float(request.topP),
            repetitionPenalty: 1.05,
            repetitionContextSize: 64
        )

        let prefixSeed = prefixKVCacheSeed(
            modelPath: loadedModelPath ?? "",
            quantization: kvCacheQuantization,
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
        let layerCaches = try prefixSeed?.caches ?? makeLayerCaches(
            model: model,
            quantization: prefillKVCacheQuantization
        )
        let logits = try await chunkedPrefill(
            model: model,
            promptTokens: promptTokens,
            cache: layerCaches,
            startIndex: prefixSeed?.tokenCount ?? 0,
            existingLogits: prefixSeed?.logits,
            modelPath: loadedModelPath ?? "",
            quantization: kvCacheQuantization,
            checkpointTokenCounts: prefixCheckpoints,
            progressHandler: progressHandler
        )

        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        let preparedCaches = try prepareLayerCachesForDecode(
            layerCaches,
            quantization: kvCacheQuantization,
            progressHandler: progressHandler
        )
        let tokenBudget = max(0, min(request.maxTokens, effectiveContext - promptTokens.count))

        progressHandler?(ChatProgress(stage: .generating, message: ""))

        let decodeResult = try await decodeTokens(
            model: model,
            tokenizerAndTemplate: tokenizerAndTemplate,
            initialLogits: logits,
            layerCaches: preparedCaches.caches,
            eosSet: eosSet,
            generationConfig: generationConfig,
            tokenBudget: tokenBudget,
            promptTokens: promptTokens,
            progressHandler: progressHandler
        )
        let generated = decodeResult.generatedTokens
        let decoded = tokenizerAndTemplate.decode(tokens: generated)
            .trimmingCharacters(in: .whitespacesAndNewlines)

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
                firstTokenSeconds: decodeResult.firstTokenSeconds
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count
        )
    }

    private func prefillQuantization(for quantization: Gemma4KVCacheQuantization) -> Gemma4KVCacheQuantization {
        guard shouldDeferQuantizationUntilDecode(quantization) else {
            return quantization
        }

        if Gemma4Resources.usesTurboDefaults(modelSpec: modelId) {
            return Gemma4KVCacheQuantization(
                bits: Gemma4Resources.defaultTurboKVBits,
                scheme: Gemma4Resources.defaultTurboKVQuantizationScheme,
                groupSize: quantization.groupSize,
                quantizedStart: Gemma4Resources.defaultTurboQuantizedKVStart
            )
        }

        return Gemma4KVCacheQuantization()
    }

    private func shouldDeferQuantizationUntilDecode(_ quantization: Gemma4KVCacheQuantization) -> Bool {
        quantization.isEnabled && quantization.scheme == .polar
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
        model: Gemma4TextCausalLM,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [Gemma4AttentionCache],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Gemma4BatchedDecodeResult {
        guard tokenBudget > 0 else {
            return Gemma4BatchedDecodeResult(generatedTokens: [], decodeSeconds: 0, firstTokenSeconds: nil)
        }
        guard continuousBatchingEnabled else {
            return try await decodeTokensSerially(
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate,
                initialLogits: initialLogits,
                layerCaches: layerCaches,
                eosSet: eosSet,
                generationConfig: generationConfig,
                tokenBudget: tokenBudget,
                promptTokens: promptTokens,
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
        model: Gemma4TextCausalLM,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [Gemma4AttentionCache],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Gemma4BatchedDecodeResult {
        var logits = initialLogits
        var generated: [Int] = []
        generated.reserveCapacity(tokenBudget)
        var repetitionHistory = promptTokens
        var firstTokenSeconds: Double?
        let decodeStart = Date()

        for _ in 0..<tokenBudget {
            try Task.checkCancellation()
            let next = sampleToken(
                logits: logits[0, -1, 0...],
                config: generationConfig,
                previousTokens: repetitionHistory
            )

            if eosSet.contains(next) {
                break
            }

            generated.append(next)
            if firstTokenSeconds == nil {
                firstTokenSeconds = Date().timeIntervalSince(decodeStart)
            }
            repetitionHistory.append(next)
            let piece = tokenizerAndTemplate.decode(token: next)
            if !piece.isEmpty {
                progressHandler?(ChatProgress(stage: .generating, message: piece))
            }

            guard generated.count < tokenBudget else {
                break
            }
            let nextInput = MLXArray([Int32(next)]).reshaped(1, 1)
            logits = model(nextInput, cache: layerCaches as [AnyObject])
            MLX.eval(logits)
        }

        return Gemma4BatchedDecodeResult(
            generatedTokens: generated,
            decodeSeconds: Date().timeIntervalSince(decodeStart),
            firstTokenSeconds: firstTokenSeconds
        )
    }

    private func enqueueDecodeRow(
        _ row: Gemma4BatchedDecodeRow,
        model: Gemma4TextCausalLM,
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
        model: Gemma4TextCausalLM,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate
    ) {
        guard !decodeLoopRunning else { return }
        decodeLoopRunning = true
        Task {
            await runDecodeLoop(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
        }
    }

    private func runDecodeLoop(
        model: Gemma4TextCausalLM,
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
        model: Gemma4TextCausalLM,
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
            let batchedLogits = model(nextInput, cache: batchedCaches as [AnyObject])
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
            row.logits = model(nextInput, cache: row.layerCaches as [AnyObject])
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
        model: Gemma4TextCausalLM,
        promptTokens: [Int],
        cache: [Gemma4AttentionCache],
        startIndex: Int,
        existingLogits: MLXArray?,
        modelPath: String,
        quantization: Gemma4KVCacheQuantization,
        checkpointTokenCounts: Set<Int> = [],
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> MLXArray {
        guard !promptTokens.isEmpty else {
            throw Gemma4Error.unsupportedConfiguration("Prompt is empty after tokenization.")
        }

        var processed = startIndex
        var logits = existingLogits
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
            let chunkLogits = model(chunk, cache: cache as [AnyObject])
            MLX.eval(chunkLogits)
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
        return logits
    }

    private func makeLayerCaches(
        model: Gemma4TextCausalLM,
        quantization: Gemma4KVCacheQuantization
    ) throws -> [Gemma4AttentionCache] {
        guard let caches = model.makeCache(quantization: quantization) as? [Gemma4AttentionCache] else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 cache construction returned an incompatible cache type.")
        }
        return caches
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
            if config.textConfig.enableMoEBlock {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: resources.modelIndexURL,
                    to: model,
                    groupSize: 16,
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
}
