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

private final class Q35BatchedDecodeRow: @unchecked Sendable {
    let id: UUID
    let eosSet: Set<Int>
    let generationConfig: GenerationConfig
    let tokenBudget: Int
    let progressHandler: (@Sendable (ChatProgress) -> Void)?
    let decodeStart: Date
    let prefillTokenCount: Int
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
        continuousBatchingEnabled: Bool = ProcessInfo.processInfo.environment["MERERUN_Q35_CONTINUOUS_BATCHING"] == "1"
    ) {
        self.modelId = modelId
        self.prefixKVCacheEnabled = prefixKVCacheEnabled
        self.continuousBatchingEnabled = continuousBatchingEnabled
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
            maxContextLength: Q35Resources.defaultContextLength
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
            maxContextLength: Q35Resources.defaultContextLength
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

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Q35 config"))
        let configData = try Data(contentsOf: normalizedRoot.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Q35Config.self, from: configData)

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Q35 tokenizer"))
        let tokenizer = try Q35TokenizerAndTemplate.load(
            from: normalizedRoot,
            maxLengthOverride: min(Q35Resources.defaultContextLength, config.textConfig.maxPositionEmbeddings)
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Q35 weights"))
        let q35Model = Q35Model(config: config)
        let resources = Q35Resources(rootURL: normalizedRoot)

        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4

        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                indexURL: resources.modelIndexURL,
                to: q35Model,
                groupSize: groupSize,
                bits: bits,
                keyMapper: { key in
                    if key.hasPrefix("language_model.") {
                        return String(key.dropFirst("language_model.".count))
                    }
                    return "__unused__.\(key)"
                }
            )
        } else {
            let arrays = try MLX.loadArrays(url: resources.modelWeightsURL)
            let filtered = arrays.filter { $0.key.hasPrefix("language_model.") }
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                filtered,
                to: q35Model,
                groupSize: groupSize,
                bits: bits,
                keyMapper: { key in
                    String(key.dropFirst("language_model.".count))
                }
            )
        }

        let tower = config.visionConfig == nil ? nil : Q35VisionTower(config: config)
        let mtpURL = normalizedRoot.appendingPathComponent("mtp.safetensors")
        let mtpDisabled = {
            guard let raw = ProcessInfo.processInfo.environment["MERERUN_Q35_MTP_SPECULATION"]?.lowercased() else {
                return false
            }
            return raw == "0" || raw == "false" || raw == "no"
        }()
        let loadedMTP: Q35MTPModel?
        if !mtpDisabled, FileManager.default.fileExists(atPath: mtpURL.path) {
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Q35 MTP weights"))
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
        let effectiveContext = min(maxContextLength, loadedConfig.textConfig.maxPositionEmbeddings)
        let prefillStart = Date()

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

        let eosSet = Set(loadedConfig.eosTokenIds + [tokenizerAndTemplate.eosTokenId].compactMap { $0 })
        let generationConfig = GenerationConfig(
            maxTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topP: Float(request.topP),
            repetitionPenalty: 1.05,
            repetitionContextSize: 64
        )

        var layerCaches = makeLayerCaches(config: loadedConfig)
        let promptInput = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, promptTokens.count)

        let imageURLs = collectImageURLs(from: messages)
        var prefillOutput: Q35PrefillOutput
        var prefillLength = promptTokens.count

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
                progressHandler: progressHandler
            )
        } else {
            progressHandler?(ChatProgress(stage: .encoding, message: "Encoding images"))
            guard visionTower != nil else {
                throw Q35Error.generationFailed("Model \(modelId) does not include a vision tower; use text-only prompts.")
            }
            try ensureVisionWeightsLoaded(progressHandler: progressHandler)

            if let visionTower,
               let imageTokenId = loadedConfig.imageTokenId ?? tokenizerAndTemplate.tokenizer.imageTokenId {
                let replacements = try buildVisionReplacements(
                    imageURLs: imageURLs,
                    visionTower: visionTower
                )

                if replacements.isEmpty {
                    prefillOutput = try await chunkedPrefill(
                        model: model,
                        promptTokens: promptTokens,
                        cache: layerCaches,
                        progressHandler: progressHandler
                    )
                } else {
                    var promptEmbeddings = model.embeddings(for: promptInput)
                    promptEmbeddings = insertVisionEmbeddings(
                        hiddenStates: promptEmbeddings,
                        inputIds: promptInput,
                        imageTokenId: imageTokenId,
                        replacements: replacements
                    )

                    if promptEmbeddings.dim(1) > effectiveContext {
                        promptEmbeddings = promptEmbeddings[0..., (promptEmbeddings.dim(1) - effectiveContext)..., 0...]
                    }
                    prefillLength = promptEmbeddings.dim(1)
                    prefillOutput = try await chunkedPrefillEmbeddings(
                        model: model,
                        inputEmbeddings: promptEmbeddings,
                        cache: layerCaches,
                        progressHandler: progressHandler
                    )
                }
            } else {
                prefillOutput = try await chunkedPrefill(
                    model: model,
                    promptTokens: promptTokens,
                    cache: layerCaches,
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
            response: decoded,
            tokensGenerated: decodeResult.generatedTokens.count,
            timing: ChatTiming(
                loadSeconds: 0,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decodeResult.decodeSeconds
            ),
            toolCalls: toolCalls
        )
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
        promptTokens: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35BatchedDecodeResult {
        guard tokenBudget > 0 else {
            return Q35BatchedDecodeResult(generatedTokens: [], decodeSeconds: 0)
        }
        guard continuousBatchingEnabled else {
            return try await decodeTokensSerially(
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate,
                initialLogits: initialLogits,
                initialHidden: initialHidden,
                mtpModel: mtpModel,
                layerCaches: layerCaches,
                eosSet: eosSet,
                generationConfig: generationConfig,
                tokenBudget: tokenBudget,
                prefillTokenCount: prefillTokenCount,
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
        promptTokens: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35BatchedDecodeResult {
        var logits = initialLogits
        var layerCaches = layerCaches
        var previousHidden = initialHidden.map(lastTokenHidden)
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
            let output = model.forward(nextInput, cache: layerCaches)
            logits = output.logits
            previousHidden = lastTokenHidden(output.hidden)
            MLX.eval(logits)
            MLX.eval(previousHidden!)
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
            let batchedLogits = model(nextInput, cache: batchedCaches)
            MLX.eval(batchedLogits)
            guard let splitCaches = splitBatchedLayerCaches(batchedCaches, rowCount: continuingRows.count) else {
                throw Q35Error.generationFailed("Q35 batched decode could not split merged cache rows.")
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
            row.logits = model(nextInput, cache: row.layerCaches)
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
            let output = model.forward(chunk, cache: cache)
            MLX.eval(output.logits)
            MLX.eval(output.hidden)
            logits = output.logits
            hidden = output.hidden
            processed = end
            if let modelPath {
                storePrefixKVCache(
                    modelPath: modelPath,
                    promptTokens: promptTokens,
                    tokenCount: processed,
                    cache: cache,
                    logits: output.logits,
                    hidden: output.hidden,
                    priority: checkpointTokenCounts.contains(processed) ? .semantic : .chunk
                )
            }
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
            let output = model.forward(chunkInput, cache: cache, inputEmbeddings: chunkEmbeddings)
            MLX.eval(output.logits)
            MLX.eval(output.hidden)
            logits = output.logits
            hidden = output.hidden
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

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Q35 vision tower"))
        try visionTower.loadWeights(from: loadedResources)
    }

    private func buildVisionReplacements(
        imageURLs: [String],
        visionTower: Q35VisionTower
    ) throws -> [MLXArray] {
        var replacements: [MLXArray] = []
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
            replacements.append(embeds)
        }

        return replacements
    }

    private func insertVisionEmbeddings(
        hiddenStates: MLXArray,
        inputIds: MLXArray,
        imageTokenId: Int,
        replacements: [MLXArray]
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
        let pairCount = min(positions.count, replacements.count)

        var parts: [MLXArray] = []
        parts.reserveCapacity(pairCount * 2 + 1)

        var cursor = 0
        for pairIndex in 0..<pairCount {
            let position = positions[pairIndex]
            if position > cursor {
                parts.append(hiddenStates[0..., cursor..<position, 0...])
            }

            var replacement = replacements[pairIndex]
            if replacement.dtype != hiddenStates.dtype {
                replacement = replacement.asType(hiddenStates.dtype)
            }
            parts.append(replacement.expandedDimensions(axis: 0))

            cursor = position + 1
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
        let divisor = max(1, patchSize * max(1, spatialMergeSize))

        let targetWidth = max(divisor, (image.width / divisor) * divisor)
        let targetHeight = max(divisor, (image.height / divisor) * divisor)

        let pixels = try QwenImageIO.resizedPixelArray(
            from: image,
            width: targetWidth,
            height: targetHeight,
            addBatchDimension: true,
            dtype: .float16
        )
        let normalized = (pixels - 0.5) / 0.5
        let gridTHW = (1, targetHeight / patchSize, targetWidth / patchSize)
        return (normalized, gridTHW)
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
