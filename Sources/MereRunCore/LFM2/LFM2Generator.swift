import Foundation
import MLX

public typealias LFM2ContinuousBatchingStats = RuntimeDecodeBatchingStats

private struct LFM2PrefillOutput {
    let logits: MLXArray
    let hidden: MLXArray
}

private struct LFM2DecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
    var firstTokenSeconds: Double? = nil
}

private struct LFM2PrefixKVCacheKey: Hashable {
    let modelPath: String
    let cacheMode: RuntimeKVCacheMode
    let tokens: [Int]
}

private struct LFM2PrefixKVCacheEntry {
    let caches: [LFM2LayerCache?]
    let logits: MLXArray
    let priority: RuntimePrefixCacheEntryPriority
    var lastAccess: Date
}

private final class LFM2BatchedDecodeRow: @unchecked Sendable {
    let id: UUID
    let eosSet: Set<Int>
    let generationConfig: GenerationConfig
    let tokenBudget: Int
    let prefillTokenCount: Int
    let progressHandler: (@Sendable (ChatProgress) -> Void)?
    let decodeStart: Date
    let continuation: CheckedContinuation<LFM2DecodeResult, Error>

    var logits: MLXArray
    var layerCaches: [LFM2LayerCache?]
    var generatedTokens: [Int] = []
    var repetitionHistory: [Int]
    var repetitionHistoryGPU: MLXArray?
    var repetitionHistoryGPUSeeded = false
    var firstTokenSeconds: Double?
    var stopped = false

    init(
        id: UUID,
        logits: MLXArray,
        layerCaches: [LFM2LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        prefillTokenCount: Int,
        repetitionHistory: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        continuation: CheckedContinuation<LFM2DecodeResult, Error>
    ) {
        self.id = id
        self.logits = logits
        self.layerCaches = layerCaches
        self.eosSet = eosSet
        self.generationConfig = generationConfig
        self.tokenBudget = tokenBudget
        self.prefillTokenCount = prefillTokenCount
        self.repetitionHistory = repetitionHistory
        self.progressHandler = progressHandler
        self.continuation = continuation
        self.decodeStart = Date()
        self.generatedTokens.reserveCapacity(tokenBudget)
    }

    var needsDecodeStep: Bool {
        !stopped && generatedTokens.count < tokenBudget
    }

    func finish() {
        continuation.resume(returning: LFM2DecodeResult(
            generatedTokens: generatedTokens,
            decodeSeconds: Date().timeIntervalSince(decodeStart),
            firstTokenSeconds: firstTokenSeconds
        ))
    }

    func fail(_ error: Error) {
        continuation.resume(throwing: error)
    }
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
    private let continuousBatchingEnabled: Bool

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

    private var decodeQueue: [LFM2BatchedDecodeRow] = []
    private var activeDecodeRows: [LFM2BatchedDecodeRow] = []
    private var decodeLoopRunning = false
    private var batchedDecodeSteps = 0
    private var samePositionBatchedSteps = 0
    private var variablePositionBatchedSteps = 0
    private var singleDecodeSteps = 0
    private var totalBatchedRows = 0
    private var maxObservedBatchSize = 0

    public init(
        modelId: String = LFM2Resources.defaultModelId,
        prefixKVCacheEnabled: Bool =
            ProcessInfo.processInfo.environment["MERERUN_LFM2_PREFIX_KV_CACHE"] == "1",
        continuousBatchingEnabled: Bool =
            ProcessInfo.processInfo.environment["MERERUN_LFM2_CONTINUOUS_BATCHING"] == "1"
    ) {
        self.modelId = modelId
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
        failQueuedDecodeRows(CancellationError())
        model = nil
        tokenizerAndTemplate = nil
        loadedModelPath = nil
        loadedConfig = nil
        Memory.clearCache()
    }

    public func continuousBatchingStats() -> LFM2ContinuousBatchingStats {
        LFM2ContinuousBatchingStats(
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

        let effectiveKVCacheMode: RuntimeKVCacheMode = request.kvCacheMode == .affine8
            ? .affine8
            : .default
        var layerCaches = makeLayerCaches(config: loadedConfig, kvCacheMode: effectiveKVCacheMode)
        var prefillStartIndex = 0
        var prefillExistingLogits: MLXArray?
        if let seed = prefixKVCacheSeed(
            modelPath: loadedModelPath,
            promptTokens: promptTokens,
            cacheMode: effectiveKVCacheMode
        ) {
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
        let decodeResult = try await decodeTokens(
            model: model,
            tokenizerAndTemplate: tokenizerAndTemplate,
            initialLogits: prefillOutput.logits,
            layerCaches: layerCaches,
            eosSet: eosSet,
            generationConfig: generationConfig,
            tokenBudget: tokenBudget,
            prefillTokenCount: promptTokens.count,
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
                decodeSeconds: decodeResult.decodeSeconds,
                firstTokenSeconds: decodeResult.firstTokenSeconds,
                kvCacheMode: effectiveKVCacheMode,
                prefillKVCache: effectiveKVCacheMode.genericCacheLabel,
                decodeKVCache: effectiveKVCacheMode.genericCacheLabel
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count
        )
    }

    private func decodeTokens(
        model: LFM2Model,
        tokenizerAndTemplate: LFM2TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [LFM2LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        prefillTokenCount: Int,
        promptTokens: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> LFM2DecodeResult {
        guard tokenBudget > 0 else {
            return LFM2DecodeResult(generatedTokens: [], decodeSeconds: 0)
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
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let row = LFM2BatchedDecodeRow(
                    id: rowID,
                    logits: initialLogitsBox.value,
                    layerCaches: layerCaches,
                    eosSet: eosSet,
                    generationConfig: generationConfig,
                    tokenBudget: tokenBudget,
                    prefillTokenCount: prefillTokenCount,
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

        // Shared pipelined decode: GPU-side sampling with an on-GPU
        // repetition window and a depth-1 lagged readback. The previous
        // loop sampled on the host and blocked on eval twice per token.
        let result = try AutoregressiveDecodeEngine.decode(
            AutoregressiveDecodeRequest(
                initialLogits: initialLogits,
                generationConfig: generationConfig,
                eosTokens: eosSet,
                tokenBudget: tokenBudget,
                historySeedTokens: promptTokens
            ),
            stepForward: { token in model(token, cache: layerCaches) },
            decodeToken: { tokenizerAndTemplate.decode(token: $0) },
            emitPiece: { _, piece in
                progressHandler?(ChatProgress(stage: .generating, message: piece))
            },
            checkCancellation: { try Task.checkCancellation() }
        )
        return LFM2DecodeResult(
            generatedTokens: result.generatedTokens,
            decodeSeconds: result.decodeSeconds,
            firstTokenSeconds: result.firstTokenSeconds
        )
    }

    private func enqueueDecodeRow(
        _ row: LFM2BatchedDecodeRow,
        model: LFM2Model,
        tokenizerAndTemplate: LFM2TokenizerAndTemplate
    ) {
        decodeQueue.append(row)
        startDecodeLoopIfNeeded(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
    }

    private func cancelDecodeRow(id: UUID) {
        if let index = decodeQueue.firstIndex(where: { $0.id == id }) {
            decodeQueue.remove(at: index).fail(CancellationError())
            return
        }
        if let index = activeDecodeRows.firstIndex(where: { $0.id == id }) {
            activeDecodeRows.remove(at: index).fail(CancellationError())
        }
    }

    private func startDecodeLoopIfNeeded(
        model: LFM2Model,
        tokenizerAndTemplate: LFM2TokenizerAndTemplate
    ) {
        guard !decodeLoopRunning else { return }
        decodeLoopRunning = true
        Task {
            await runDecodeLoop(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
        }
    }

    private func runDecodeLoop(
        model: LFM2Model,
        tokenizerAndTemplate: LFM2TokenizerAndTemplate
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
            guard !activeDecodeRows.isEmpty else { continue }

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

    private func selectDecodeRows() -> [LFM2BatchedDecodeRow] {
        let eligible = activeDecodeRows.filter(\.needsDecodeStep)
        let selectedIDs = Set(RuntimeDecodeBatchPlanner.selectRows(
            eligible.map { row in
                RuntimeDecodeBatchRowMetadata(
                    row: row.id,
                    signature: row.layerCaches.map { $0?.batchSignature ?? "nil" }.joined(separator: "|"),
                    position: decodePosition(row)
                )
            }
        ))
        return eligible.filter { selectedIDs.contains($0.id) }
    }

    private func decodePosition(_ row: LFM2BatchedDecodeRow) -> Int {
        row.layerCaches.compactMap { $0?.offset }.min()
            ?? (row.prefillTokenCount + row.generatedTokens.count)
    }

    private func decodeOneStep(
        rows: [LFM2BatchedDecodeRow],
        model: LFM2Model,
        tokenizerAndTemplate: LFM2TokenizerAndTemplate
    ) throws {
        let sampledRows = rows.filter(\.needsDecodeStep)
        guard !sampledRows.isEmpty else { return }

        var tokenArrays: [MLXArray] = []
        tokenArrays.reserveCapacity(sampledRows.count)
        for row in sampledRows {
            if !row.repetitionHistoryGPUSeeded {
                row.repetitionHistoryGPU = repetitionHistoryArray(
                    promptTokens: row.repetitionHistory,
                    config: row.generationConfig
                )
                row.repetitionHistoryGPUSeeded = true
            }
            let token = sampledTokenArray(
                logits: row.logits[0, -1, 0...],
                config: row.generationConfig,
                previousTokenIndices: row.repetitionHistoryGPU,
                banMask: nil
            )
            row.repetitionHistoryGPU = appendingRepetitionHistory(
                row.repetitionHistoryGPU,
                token: token,
                config: row.generationConfig
            )
            tokenArrays.append(token.reshaped(1))
        }
        let sampledValues = concatenated(tokenArrays, axis: 0).asArray(Int32.self)
        for (index, row) in sampledRows.enumerated() {
            let next = Int(sampledValues[index])
            guard !row.eosSet.contains(next) else {
                row.stopped = true
                continue
            }
            row.generatedTokens.append(next)
            row.repetitionHistory.append(next)
            if row.firstTokenSeconds == nil {
                row.firstTokenSeconds = Date().timeIntervalSince(row.decodeStart)
            }
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
            let positionsBeforeStep = continuingRows.map(decodePosition)
            let nextInput = MLXArray(continuingRows.compactMap { $0.generatedTokens.last }.map(Int32.init))
                .reshaped(continuingRows.count, 1)
            let batchedLogits = model(nextInput, cache: batchedCaches)
            MLX.eval(batchedLogits)
            guard let splitCaches = splitBatchedLayerCaches(batchedCaches, rowCount: continuingRows.count) else {
                throw LFM2Error.generationFailed("LFM2 batched decode could not split merged cache rows.")
            }
            for (index, row) in continuingRows.enumerated() {
                row.layerCaches = splitCaches[index]
                row.logits = batchedLogits[index..<(index + 1), 0..., 0...]
            }
            batchedDecodeSteps += 1
            if RuntimeDecodeBatchPositionKind.variablePositionBatchCount(positionsBeforeStep) > 0 {
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
            row.logits = model(MLXArray([Int32(next)]).reshaped(1, 1), cache: row.layerCaches)
            MLX.eval(row.logits)
            singleDecodeSteps += 1
        }
    }

    private func makeBatchedLayerCaches(_ rowCaches: [[LFM2LayerCache?]]) -> [LFM2LayerCache?]? {
        guard let first = rowCaches.first, !first.isEmpty,
              rowCaches.allSatisfy({ $0.count == first.count }) else {
            return nil
        }
        var result: [LFM2LayerCache?] = []
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
        _ caches: [LFM2LayerCache?],
        rowCount: Int
    ) -> [[LFM2LayerCache?]]? {
        guard rowCount > 0 else { return nil }
        var rows = Array(repeating: [LFM2LayerCache?](), count: rowCount)
        for cache in caches {
            guard let cache else {
                for index in rows.indices {
                    rows[index].append(nil)
                }
                continue
            }
            guard let split = cache.unbatchedRows(count: rowCount), split.count == rowCount else {
                return nil
            }
            for index in rows.indices {
                rows[index].append(split[index])
            }
        }
        return rows
    }

    private func finishCompletedDecodeRows() {
        var remaining: [LFM2BatchedDecodeRow] = []
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

    private func failRows(_ rows: [LFM2BatchedDecodeRow], with error: Error) {
        let ids = Set(rows.map(\.id))
        activeDecodeRows.removeAll { row in
            guard ids.contains(row.id) else { return false }
            row.fail(error)
            return true
        }
    }

    private func failQueuedDecodeRows(_ error: Error) {
        decodeQueue.forEach { $0.fail(error) }
        activeDecodeRows.forEach { $0.fail(error) }
        decodeQueue.removeAll()
        activeDecodeRows.removeAll()
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
        promptTokens: [Int],
        cacheMode: RuntimeKVCacheMode
    ) -> (tokenCount: Int, caches: [LFM2LayerCache?], logits: MLXArray)? {
        guard prefixKVCacheEnabled, let modelPath else { return nil }
        let matchingKey = prefixKVCache.keys
            .filter { key in
                key.modelPath == modelPath
                    && key.cacheMode == cacheMode
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
            cacheMode: cacheMode(for: cache),
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

    private func makeLayerCaches(
        config: LFM2Config,
        kvCacheMode: RuntimeKVCacheMode = .default
    ) -> [LFM2LayerCache?] {
        let attentionLayers = config.fullAttentionLayerIndexes
        return (0..<config.numHiddenLayers).map { layerIndex in
            if attentionLayers.contains(layerIndex) {
                if kvCacheMode == .affine8 {
                    return .attention(AffineQuantizedKVCache(
                        groupSize: Self.affineKVGroupSize(headDimension: config.headDim),
                        step: 256
                    ))
                }
                return .attention(KVCacheSimple(step: 256))
            }
            return .conv(LFM2ConvCache())
        }
    }

    private func cacheMode(for caches: [LFM2LayerCache?]) -> RuntimeKVCacheMode {
        caches.contains { entry in
            guard case .attention(let cache)? = entry else { return false }
            return cache is AffineQuantizedKVCache
        } ? .affine8 : .default
    }

    private static func affineKVGroupSize(headDimension: Int) -> Int {
        [64, 32, 16, 8].first { headDimension % $0 == 0 } ?? 1
    }
}
