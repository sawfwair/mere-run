import Foundation
import MLX

public enum LagunaError: LocalizedError {
    case modelPathRequired
    case missingFiles([String])
    case dflashIncompatible(String)
    case modelNotLoaded
    case adapterSwitchDuringActiveGeneration
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelPathRequired:
            return "Laguna requires an installed managed model or an explicit local MLX checkpoint path."
        case .missingFiles(let files):
            return "Laguna checkpoint is missing required files: \(files.joined(separator: ", "))."
        case .dflashIncompatible(let message):
            return "Laguna DFlash checkpoint is incompatible: \(message)"
        case .modelNotLoaded:
            return "Laguna model is not loaded."
        case .adapterSwitchDuringActiveGeneration:
            return "Laguna cannot switch text LoRA adapters while batched generation is active."
        case .generationFailed(let message):
            return "Laguna generation failed: \(message)"
        }
    }
}

public typealias LagunaContinuousBatchingStats = RuntimeDecodeBatchingStats

private struct LagunaDecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    var logprobs: ChatLogprobDiagnostics? = nil
    var acceleration: ChatAccelerationDiagnostics? = nil
}

private struct LagunaPrefillResult {
    let logits: MLXArray
    let dflashCache: [Gemma4AttentionCache]?
}

private final class LagunaBatchedDecodeRow: @unchecked Sendable {
    let id: UUID
    let eosTokens: Set<Int>
    let generationConfig: GenerationConfig
    let tokenBudget: Int
    let progressHandler: (@Sendable (ChatProgress) -> Void)?
    let decodeStart = Date()
    let continuation: CheckedContinuation<LagunaDecodeResult, Error>

    var logits: MLXArray
    var caches: [Gemma4AttentionCache]
    var generatedTokens: [Int] = []
    var repetitionHistory: [Int]
    var firstTokenSeconds: Double?
    var pendingProgressWhitespace = ""
    var progressDecoder = IncrementalTokenTextDecoder()
    var stopped = false

    init(
        id: UUID,
        logits: MLXArray,
        caches: [Gemma4AttentionCache],
        eosTokens: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        repetitionHistory: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        continuation: CheckedContinuation<LagunaDecodeResult, Error>
    ) {
        self.id = id
        self.logits = logits
        self.caches = caches
        self.eosTokens = eosTokens
        self.generationConfig = generationConfig
        self.tokenBudget = tokenBudget
        self.repetitionHistory = repetitionHistory
        self.progressHandler = progressHandler
        self.continuation = continuation
        generatedTokens.reserveCapacity(tokenBudget)
    }

    var needsDecodeStep: Bool {
        !stopped && generatedTokens.count < tokenBudget
    }

    func finish() {
        continuation.resume(returning: LagunaDecodeResult(
            generatedTokens: generatedTokens,
            decodeSeconds: Date().timeIntervalSince(decodeStart),
            firstTokenSeconds: firstTokenSeconds,
            acceleration: ChatAccelerationDiagnostics(route: "continuous-batched")
        ))
    }

    func fail(_ error: Error) {
        continuation.resume(throwing: error)
    }
}

private final class LagunaDFlashBatchedDecodeRow: @unchecked Sendable {
    let id: UUID
    let eosTokens: Set<Int>
    let generationConfig: GenerationConfig
    let tokenBudget: Int
    let progressHandler: (@Sendable (ChatProgress) -> Void)?
    let decodeStart = Date()
    let continuation: CheckedContinuation<LagunaDecodeResult, Error>

    var logits: MLXArray
    var targetCaches: [Gemma4AttentionCache]
    let draftCaches: [Gemma4AttentionCache]
    var generatedTokens: [Int] = []
    var repetitionHistory: [Int]
    var firstTokenSeconds: Double?
    var pendingProgressWhitespace = ""
    var progressDecoder = IncrementalTokenTextDecoder()
    var stopped = false

    init(
        id: UUID,
        logits: MLXArray,
        targetCaches: [Gemma4AttentionCache],
        draftCaches: [Gemma4AttentionCache],
        eosTokens: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        repetitionHistory: [Int],
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        continuation: CheckedContinuation<LagunaDecodeResult, Error>
    ) {
        self.id = id
        self.logits = logits
        self.targetCaches = targetCaches
        self.draftCaches = draftCaches
        self.eosTokens = eosTokens
        self.generationConfig = generationConfig
        self.tokenBudget = tokenBudget
        self.repetitionHistory = repetitionHistory
        self.progressHandler = progressHandler
        self.continuation = continuation
        generatedTokens.reserveCapacity(tokenBudget)
    }

    var needsDecodeRound: Bool {
        !stopped && generatedTokens.count < tokenBudget
    }

    func finish() {
        continuation.resume(returning: LagunaDecodeResult(
            generatedTokens: generatedTokens,
            decodeSeconds: Date().timeIntervalSince(decodeStart),
            firstTokenSeconds: firstTokenSeconds,
            acceleration: ChatAccelerationDiagnostics(
                route: "dflash-continuous-batched",
                draftModel: LagunaResources.dflashModelID
            )
        ))
    }

    func fail(_ error: Error) {
        continuation.resume(throwing: error)
    }
}

private struct LagunaDFlashRecovery {
    let row: LagunaDFlashBatchedDecodeRow
    let candidateHiddenStates: [Int: MLXArray]
    let committedCandidateTokenCount: Int
    let replacement: Int
    let recoveryCache: [Gemma4AttentionCache]
}

public actor LagunaGenerator: ChatGenerator {
    private static let prefillChunkSize = 512
    private static let prefillChunkingThreshold = 4_096

    private var model: LagunaCausalLM?
    private var tokenizerAndTemplate: LagunaTokenizerAndTemplate?
    private var config: LagunaConfig?
    private var dflashModel: LagunaDFlashModel?
    private var dflashConfig: LagunaDFlashConfig?
    private var loadedModelPath: String?
    private var loadedDFlashPath: String?
    private var loadedTextLoRASignature: String?
    private let continuousBatchingEnabled: Bool
    private let configuredDFlashPath: String?
    private let dflashSpeculativeTokens: Int
    private let dflashMinimumOutputTokens: Int

    private var decodeQueue: [LagunaBatchedDecodeRow] = []
    private var activeDecodeRows: [LagunaBatchedDecodeRow] = []
    private var decodeLoopRunning = false
    private var dflashDecodeQueue: [LagunaDFlashBatchedDecodeRow] = []
    private var activeDFlashDecodeRows: [LagunaDFlashBatchedDecodeRow] = []
    private var dflashDecodeLoopRunning = false
    private var batchedDecodeSteps = 0
    private var samePositionBatchedSteps = 0
    private var variablePositionBatchedSteps = 0
    private var singleDecodeSteps = 0
    private var totalBatchedRows = 0
    private var maxObservedBatchSize = 0
    private var dflashRounds = 0
    private var dflashDraftedTokens = 0
    private var dflashAcceptedDraftTokens = 0
    private var dflashRejectedDraftTokens = 0
    private var dflashFullAcceptanceRounds = 0
    private var dflashTargetVerificationForwards = 0
    private var dflashTargetRecoveryForwards = 0
    private var dflashTargetFallbackForwards = 0
    private var dflashAdaptiveFallbacks = 0
    private var dflashRoutedRequests = 0
    private var dflashBypassedRequests = 0

    public init(
        continuousBatchingEnabled: Bool =
            ProcessInfo.processInfo.environment["MERERUN_LAGUNA_CONTINUOUS_BATCHING"] == "1",
        dflashModelPath: String? =
            ProcessInfo.processInfo.environment["MERERUN_LAGUNA_DFLASH_PATH"],
        dflashSpeculativeTokens: Int =
            Int(ProcessInfo.processInfo.environment["MERERUN_LAGUNA_DFLASH_TOKENS"] ?? "")
                ?? LagunaDFlashRouting.defaultSpeculativeTokens,
        dflashMinimumOutputTokens: Int =
            Int(ProcessInfo.processInfo.environment[
                "MERERUN_LAGUNA_DFLASH_MIN_TOKENS"
            ] ?? "") ?? LagunaDFlashRouting.defaultMinimumOutputTokens
    ) {
        self.continuousBatchingEnabled = continuousBatchingEnabled
        self.configuredDFlashPath = dflashModelPath
        self.dflashSpeculativeTokens = max(1, dflashSpeculativeTokens)
        self.dflashMinimumOutputTokens = max(1, dflashMinimumOutputTokens)
    }

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        throw LagunaError.modelPathRequired
    }

    public func chat(
        _ request: ChatRequest,
        modelPath: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> ChatResponse {
        try await chat(
            request,
            modelPath: modelPath,
            dflashRouting: .automatic,
            progressHandler: progressHandler
        )
    }

    public func chat(
        _ request: ChatRequest,
        modelPath: String,
        dflashRouting: LagunaDFlashRoutingMode,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> ChatResponse {
        try await Stream.withNewDefaultStream {
            let rootURL = URL(fileURLWithPath: modelPath).standardizedFileURL
            let loadStart = Date()
            let requestedLoRASignature = Self.loraSignature(request.lora)
            if loadedTextLoRASignature != requestedLoRASignature {
                guard !hasActiveGeneration else {
                    throw LagunaError.adapterSwitchDuringActiveGeneration
                }
                resetLoadedModel()
            }
            try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
            try await applyTextLoRAIfNeeded(
                request.lora,
                progressHandler: progressHandler
            )
            let loadSeconds = Date().timeIntervalSince(loadStart)

            var response = try await generate(
                request,
                dflashRouting: dflashRouting,
                progressHandler: progressHandler
            )
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
        modelPath: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        try await Stream.withNewDefaultStream {
            try await ensureLoaded(
                rootURL: URL(fileURLWithPath: modelPath).standardizedFileURL,
                progressHandler: progressHandler
            )
            guard let model else {
                throw LagunaError.modelNotLoaded
            }
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Warming Laguna inference"
            ))
            warmUp(model: model, dflash: dflashModel)
        }
    }

    public func unload() {
        guard !hasActiveGeneration else { return }
        resetLoadedModel()
    }

    private func resetLoadedModel() {
        model = nil
        tokenizerAndTemplate = nil
        config = nil
        dflashModel = nil
        dflashConfig = nil
        loadedModelPath = nil
        loadedDFlashPath = nil
        loadedTextLoRASignature = nil
        Memory.clearCache()
    }

    private var hasActiveGeneration: Bool {
        decodeLoopRunning
            || dflashDecodeLoopRunning
            || !decodeQueue.isEmpty
            || !activeDecodeRows.isEmpty
            || !dflashDecodeQueue.isEmpty
            || !activeDFlashDecodeRows.isEmpty
    }

    public func continuousBatchingStats() -> LagunaContinuousBatchingStats {
        LagunaContinuousBatchingStats(
            enabled: continuousBatchingEnabled,
            activeRows: activeDecodeRows.count + activeDFlashDecodeRows.count,
            queuedRows: decodeQueue.count + dflashDecodeQueue.count,
            batchedDecodeSteps: batchedDecodeSteps,
            samePositionBatchedSteps: samePositionBatchedSteps,
            variablePositionBatchedSteps: variablePositionBatchedSteps,
            singleDecodeSteps: singleDecodeSteps,
            totalBatchedRows: totalBatchedRows,
            maxBatchSize: maxObservedBatchSize
        )
    }

    public func dflashStats() -> LagunaDFlashStats {
        LagunaDFlashStats(
            enabled: dflashModel != nil,
            speculativeTokens: dflashSpeculativeTokens,
            minimumOutputTokens: dflashMinimumOutputTokens,
            routedRequests: dflashRoutedRequests,
            bypassedRequests: dflashBypassedRequests,
            rounds: dflashRounds,
            draftedTokens: dflashDraftedTokens,
            acceptedDraftTokens: dflashAcceptedDraftTokens,
            rejectedDraftTokens: dflashRejectedDraftTokens,
            fullAcceptanceRounds: dflashFullAcceptanceRounds,
            targetVerificationForwards: dflashTargetVerificationForwards,
            targetRecoveryForwards: dflashTargetRecoveryForwards,
            targetFallbackForwards: dflashTargetFallbackForwards,
            adaptiveFallbacks: dflashAdaptiveFallbacks
        )
    }

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let requestedDFlashPath = configuredDFlashPath.map {
            URL(fileURLWithPath: $0).standardizedFileURL.path
        }
        if loadedModelPath == rootURL.path,
           loadedDFlashPath == requestedDFlashPath,
           model != nil,
           tokenizerAndTemplate != nil {
            return
        }

        let missingFiles = LagunaResources.validate(rootURL: rootURL)
        guard missingFiles.isEmpty else {
            throw LagunaError.missingFiles(missingFiles)
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Laguna config"))
        let configData = try Data(contentsOf: rootURL.appending(path: "config.json"))
        let config = try JSONDecoder().decode(LagunaConfig.self, from: configData)
        let weightsIndexURL = rootURL.appending(path: "model.safetensors.index.json")
        let weightsIndex = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: weightsIndexURL)
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Laguna tokenizer"))
        let tokenizer = try await LagunaTokenizerAndTemplate.load(
            from: rootURL,
            maxLength: config.maxPositionEmbeddings
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Laguna weights"))
        let model = LagunaCausalLM(
            config: config,
            quantizedSharedExperts: LagunaResources.hasQuantizedSharedExperts(weightsIndex)
        )
        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: weightsIndexURL,
            to: model,
            dtype: nil,
            verify: .shapeMismatch,
            progressHandler: { progress in
                progressHandler?(ChatProgress(
                    stage: .loadingModel,
                    message: "Loading Laguna shard \(progress.shardIndex + 1)/\(progress.shardCount)"
                ))
            }
        )
        let runtimeAccelerationArrays = model.prepareRuntimeAcceleration()
        if !runtimeAccelerationArrays.isEmpty {
            MLX.eval(runtimeAccelerationArrays)
        }

        self.model = model
        self.tokenizerAndTemplate = tokenizer
        self.config = config
        self.loadedModelPath = rootURL.path
        self.loadedTextLoRASignature = nil

        if let configuredDFlashPath {
            let dflashRootURL = URL(fileURLWithPath: configuredDFlashPath)
                .standardizedFileURL
            let missingDFlashFiles = LagunaResources.validateDFlash(
                rootURL: dflashRootURL
            )
            guard missingDFlashFiles.isEmpty else {
                throw LagunaError.missingFiles(
                    missingDFlashFiles.map { "DFlash/\($0)" }
                )
            }
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Loading Laguna DFlash config"
            ))
            let dflashConfig = try JSONDecoder().decode(
                LagunaDFlashConfig.self,
                from: Data(contentsOf: dflashRootURL.appending(path: "config.json"))
            )
            try validateDFlashCompatibility(
                target: config,
                dflash: dflashConfig
            )
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Loading Laguna DFlash weights"
            ))
            let dflashModel = LagunaDFlashModel(config: dflashConfig)
            try HFSafetensorsWeightsLoader.applyWeights(
                url: dflashRootURL.appending(path: "model.safetensors"),
                to: dflashModel,
                dtype: nil,
                verify: .shapeMismatch
            )
            self.dflashModel = dflashModel
            self.dflashConfig = dflashConfig
            self.loadedDFlashPath = dflashRootURL.path
        } else {
            dflashModel = nil
            dflashConfig = nil
            loadedDFlashPath = nil
        }
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
            throw LagunaError.modelNotLoaded
        }
        progressHandler?(ChatProgress(
            stage: .loadingModel,
            message: "Loading Laguna text LoRA"
        ))
        _ = try await LagunaTextLoRAAdapter.apply(lora, to: model)
        loadedTextLoRASignature = signature
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

    private func warmUp(
        model: LagunaCausalLM,
        dflash: LagunaDFlashModel?
    ) {
        let token = MLXArray([Int32(0)]).reshaped(1, 1)
        let captureLayerIndices = dflash.map {
            Set($0.config.dflash.targetLayerIDs)
        } ?? []
        let targetCache = model.makeCache()
        let output = model.forward(
            token,
            cache: targetCache,
            captureLayerIndices: captureLayerIndices,
            lastPositionOnly: true
        )
        guard let dflash else {
            MLX.eval(output.logits)
            targetCache.forEach { $0.evaluateStorage() }
            return
        }

        let draftCache = dflash.makeCache()
        dflash.appendTargetContext(
            dflash.combineTargetHiddenStates(output.capturedHiddenStates),
            cache: draftCache
        )
        let draftLogits = dflash.draftLogits(
            anchorTokens: token,
            speculativeTokenCount: 1,
            cache: draftCache,
            target: model
        )
        MLX.eval(output.logits, draftLogits)
        targetCache.forEach { $0.evaluateStorage() }
        draftCache.forEach { $0.evaluateStorage() }
    }

    private func validateDFlashCompatibility(
        target: LagunaConfig,
        dflash: LagunaDFlashConfig
    ) throws {
        guard dflash.vocabSize == target.vocabSize else {
            throw LagunaError.dflashIncompatible(
                "draft vocabulary \(dflash.vocabSize) does not match target \(target.vocabSize)."
            )
        }
        guard dflash.hiddenSize == target.hiddenSize else {
            throw LagunaError.dflashIncompatible(
                "draft hidden size \(dflash.hiddenSize) does not match target \(target.hiddenSize)."
            )
        }
        guard dflash.dflash.numTargetLayers == target.numHiddenLayers else {
            throw LagunaError.dflashIncompatible(
                "draft expects \(dflash.dflash.numTargetLayers) target layers, found \(target.numHiddenLayers)."
            )
        }
        guard dflash.dflash.targetLayerIDs.allSatisfy({
            target.modelType == "laguna" && target.numHiddenLayers > $0
        }) else {
            throw LagunaError.dflashIncompatible(
                "target_layer_ids reference layers outside the target model."
            )
        }
    }

    private func generate(
        _ request: ChatRequest,
        dflashRouting: LagunaDFlashRoutingMode,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        guard let model, let tokenizerAndTemplate, let config else {
            throw LagunaError.modelNotLoaded
        }

        let requestedContext = request.maxContextTokens ?? LagunaResources.defaultContextLength
        guard requestedContext > 0 else {
            throw LagunaError.generationFailed("maxContextTokens must be greater than zero.")
        }
        let effectiveContext = min(requestedContext, config.maxPositionEmbeddings)

        progressHandler?(ChatProgress(stage: .encoding, message: "Encoding Laguna prompt"))
        let prefillStart = Date()
        let promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: request.messages,
            tools: request.tools,
            includeThinking: request.showThinking,
            maxLength: effectiveContext
        )
        guard !promptTokens.isEmpty else {
            throw LagunaError.generationFailed("The rendered prompt contained no tokens.")
        }

        let tokenBudget = max(0, min(
            request.maxTokens,
            effectiveContext - promptTokens.count
        ))
        let activeDFlash = request.logprobCapture.isEnabled ? nil : dflashModel.flatMap { dflash in
            let enabled = switch dflashRouting {
            case .automatic:
                LagunaDFlashRouting.shouldUseDFlash(
                    tokenBudget: tokenBudget,
                    minimumOutputTokens: dflashMinimumOutputTokens
                )
            case .targetOnly:
                false
            case .dflash:
                true
            }
            return enabled ? dflash : nil
        }
        if dflashModel != nil {
            if activeDFlash == nil {
                dflashBypassedRequests += 1
            } else {
                dflashRoutedRequests += 1
            }
        }

        let cache = model.makeCache()
        let prefill = try prefill(
            model: model,
            promptTokens: promptTokens,
            cache: cache,
            dflash: activeDFlash,
            progressHandler: progressHandler
        )
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        let generationConfig = GenerationConfig(
            maxTokens: tokenBudget,
            temperature: Float(request.temperature),
            topK: request.topK ?? 0,
            topP: Float(request.topP),
            minP: Float(request.minP),
            repetitionPenalty: nil,
            repetitionContextSize: 64
        )
        let eosTokens = Self.resolvedEOSTokens(
            modelTokenIDs: config.eosTokenIDs,
            templateTokenIDs: tokenizerAndTemplate.stopTokenIDs,
            stopOnEOS: request.stopOnEOS
        )

        progressHandler?(ChatProgress(stage: .generating, message: ""))
        let decode = try await decodeTokens(
            model: model,
            tokenizerAndTemplate: tokenizerAndTemplate,
            initialLogits: prefill.logits,
            caches: cache,
            dflash: activeDFlash,
            dflashCache: prefill.dflashCache,
            eosTokens: eosTokens,
            generationConfig: generationConfig,
            tokenBudget: tokenBudget,
            promptTokens: promptTokens,
            dflashRouting: dflashRouting,
            logprobCapture: request.logprobCapture,
            logprobRegion: request.logprobRegionHint ?? .visible,
            progressHandler: progressHandler
        )

        let decoded = tokenizerAndTemplate.decode(tokens: decode.generatedTokens)
        let trimmed = TextGenerationStopSequences.trimming(
            decoded,
            sequences: request.stopSequences
        )
        let toolCalls: [ToolCall]? = request.tools?.isEmpty == false ? {
            let parsed = LagunaToolParser.parseToolCalls(trimmed.text)
            return parsed.isEmpty ? nil : parsed
        }() : nil
        let finishReason: ChatFinishReason
        if trimmed.matchedSequence != nil {
            finishReason = .stopSequence
        } else if decode.generatedTokens.count >= tokenBudget, tokenBudget > 0 {
            finishReason = .length
        } else {
            finishReason = .stop
        }

        return ChatResponse(
            generatedText: trimmed.text,
            tokensGenerated: decode.generatedTokens.count,
            showThinking: request.showThinking,
            timing: ChatTiming(
                loadSeconds: 0,
                prefillSeconds: prefillSeconds,
                decodeSeconds: decode.decodeSeconds,
                firstTokenSeconds: decode.firstTokenSeconds,
                kvCacheMode: .default,
                prefillKVCache: "bf16",
                decodeKVCache: "bf16"
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count,
            finishReason: finishReason,
            logprobs: decode.logprobs,
            acceleration: decode.acceleration
        )
    }

    static func resolvedEOSTokens(
        modelTokenIDs: [Int],
        templateTokenIDs: [Int],
        stopOnEOS: Bool
    ) -> Set<Int> {
        guard stopOnEOS else { return [] }
        return Set(modelTokenIDs + templateTokenIDs)
    }

    private func decodeTokens(
        model: LagunaCausalLM,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate,
        initialLogits: MLXArray,
        caches: [Gemma4AttentionCache],
        dflash: LagunaDFlashModel?,
        dflashCache: [Gemma4AttentionCache]?,
        eosTokens: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        dflashRouting: LagunaDFlashRoutingMode,
        logprobCapture: ChatLogprobCapture,
        logprobRegion: ChatLogprobRegion,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> LagunaDecodeResult {
        if let dflash, let dflashCache {
            if continuousBatchingEnabled, dflashRouting == .dflash {
                guard tokenBudget > 0 else {
                    return LagunaDecodeResult(
                        generatedTokens: [],
                        decodeSeconds: 0,
                        firstTokenSeconds: nil
                    )
                }
                let rowID = UUID()
                // The row and model stay confined to this generator actor. Swift
                // 6.0's targeted-concurrency checker still treats the continuation
                // closure as a send boundary, so make that ownership transfer
                // explicit just as the other continuous-batching runtimes do.
                let initialLogitsBox = RuntimeUncheckedSendable(initialLogits)
                let targetCachesBox = RuntimeUncheckedSendable(caches)
                let draftCachesBox = RuntimeUncheckedSendable(dflashCache)
                let modelBox = RuntimeUncheckedSendable(model)
                let dflashBox = RuntimeUncheckedSendable(dflash)
                return try await withTaskCancellationHandler {
                    try await withCheckedThrowingContinuation { continuation in
                        enqueueDFlashDecodeRow(
                            LagunaDFlashBatchedDecodeRow(
                                id: rowID,
                                logits: initialLogitsBox.value,
                                targetCaches: targetCachesBox.value,
                                draftCaches: draftCachesBox.value,
                                eosTokens: eosTokens,
                                generationConfig: generationConfig,
                                tokenBudget: tokenBudget,
                                repetitionHistory: promptTokens,
                                progressHandler: progressHandler,
                                continuation: continuation
                            ),
                            model: modelBox.value,
                            dflash: dflashBox.value,
                            tokenizerAndTemplate: tokenizerAndTemplate
                        )
                    }
                } onCancel: { [weak self] in
                    guard let self else { return }
                    Task {
                        await self.cancelDFlashDecodeRow(id: rowID)
                    }
                }
            }

            let result = try LagunaDFlashDecoder.decode(
                initialLogits: initialLogits,
                target: model,
                targetCache: caches,
                dflash: dflash,
                draftCache: dflashCache,
                generationConfig: generationConfig,
                eosTokens: eosTokens,
                tokenBudget: tokenBudget,
                historySeedTokens: promptTokens,
                speculativeTokens: dflashSpeculativeTokens,
                adaptiveMinimumAcceptanceRate: dflashRouting == .automatic
                    ? LagunaDFlashRouting.defaultMinimumAcceptanceRate
                    : nil,
                decodeToken: { tokenizerAndTemplate.decode(token: $0) },
                decodeTokens: { tokenizerAndTemplate.decode(tokens: $0) },
                emitPiece: { _, piece in
                    progressHandler?(ChatProgress(stage: .generating, message: piece))
                },
                checkCancellation: { try Task.checkCancellation() }
            )
            accumulateDFlashStats(result.stats)
            return LagunaDecodeResult(
                generatedTokens: result.generatedTokens,
                decodeSeconds: result.decodeSeconds,
                firstTokenSeconds: result.firstTokenSeconds,
                acceleration: ChatAccelerationDiagnostics(
                    route: "dflash-speculative",
                    draftModel: LagunaResources.dflashModelID,
                    rounds: result.stats.rounds,
                    draftedTokens: result.stats.draftedTokens,
                    acceptedDraftTokens: result.stats.acceptedDraftTokens
                )
            )
        }

        guard continuousBatchingEnabled && !logprobCapture.isEnabled else {
            let result = try AutoregressiveDecodeEngine.decode(
                AutoregressiveDecodeRequest(
                    initialLogits: initialLogits,
                    generationConfig: generationConfig,
                    eosTokens: eosTokens,
                    tokenBudget: tokenBudget,
                    historySeedTokens: promptTokens,
                    logprobCapture: logprobCapture,
                    logprobRegion: logprobRegion
                ),
                stepForward: { token in
                    model.lastPositionLogits(token, cache: caches)
                },
                decodeToken: { tokenizerAndTemplate.decode(token: $0) },
                decodeTokens: { tokenizerAndTemplate.decode(tokens: $0) },
                emitPiece: { _, piece in
                    progressHandler?(ChatProgress(stage: .generating, message: piece))
                },
                checkCancellation: { try Task.checkCancellation() }
            )
            return LagunaDecodeResult(
                generatedTokens: result.generatedTokens,
                decodeSeconds: result.decodeSeconds,
                firstTokenSeconds: result.firstTokenSeconds,
                logprobs: result.logprobs,
                acceleration: ChatAccelerationDiagnostics(route: "final-target-pipelined")
            )
        }

        guard tokenBudget > 0 else {
            return LagunaDecodeResult(
                generatedTokens: [],
                decodeSeconds: 0,
                firstTokenSeconds: nil
            )
        }

        let rowID = UUID()
        let initialLogitsBox = RuntimeUncheckedSendable(initialLogits)
        let cachesBox = RuntimeUncheckedSendable(caches)
        let modelBox = RuntimeUncheckedSendable(model)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                enqueueDecodeRow(
                    LagunaBatchedDecodeRow(
                        id: rowID,
                        logits: initialLogitsBox.value,
                        caches: cachesBox.value,
                        eosTokens: eosTokens,
                        generationConfig: generationConfig,
                        tokenBudget: tokenBudget,
                        repetitionHistory: promptTokens,
                        progressHandler: progressHandler,
                        continuation: continuation
                    ),
                    model: modelBox.value,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
            }
        } onCancel: { [weak self] in
            guard let self else { return }
            Task {
                await self.cancelDecodeRow(id: rowID)
            }
        }
    }

    private func prefill(
        model: LagunaCausalLM,
        promptTokens: [Int],
        cache: [Gemma4AttentionCache],
        dflash: LagunaDFlashModel?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> LagunaPrefillResult {
        var processed = 0
        var logits: MLXArray?
        let dflashContextStart = dflash.map {
            max(0, promptTokens.count - $0.config.slidingWindow)
        }
        let dflashCache: [Gemma4AttentionCache]?
        if let dflash, let dflashContextStart {
            dflashCache = dflash.makeCache(initialOffset: dflashContextStart)
        } else {
            dflashCache = nil
        }
        let prefillChunkSize = promptTokens.count > Self.prefillChunkingThreshold
            ? Self.prefillChunkSize
            : promptTokens.count
        while processed < promptTokens.count {
            try Task.checkCancellation()
            let end = min(processed + prefillChunkSize, promptTokens.count)
            if promptTokens.count > Self.prefillChunkingThreshold {
                progressHandler?(ChatProgress(
                    stage: .encoding,
                    message: "Prefilling \(end)/\(promptTokens.count) Laguna tokens"
                ))
            }
            let chunk = MLXArray(promptTokens[processed..<end].map(Int32.init))
                .reshaped(1, end - processed)
            let chunkLogits: MLXArray
            if let dflash, let dflashCache, let dflashContextStart {
                let capturesDraftContext = end > dflashContextStart
                let output = model.forward(
                    chunk,
                    cache: cache,
                    captureLayerIndices: capturesDraftContext
                        ? Set(dflash.config.dflash.targetLayerIDs)
                        : [],
                    lastPositionOnly: true
                )
                if capturesDraftContext {
                    let retainedStart = max(processed, dflashContextStart) - processed
                    let retainedHiddenStates = output.capturedHiddenStates.mapValues {
                        $0[0..., retainedStart..., 0...]
                    }
                    let combined = dflash.combineTargetHiddenStates(
                        retainedHiddenStates
                    )
                    dflash.appendTargetContext(combined, cache: dflashCache)
                    dflashCache.forEach { $0.evaluateStorage() }
                }
                chunkLogits = output.logits
            } else {
                chunkLogits = model.lastPositionLogits(chunk, cache: cache)
            }
            MLX.eval(chunkLogits)
            logits = chunkLogits
            processed = end
        }
        guard let logits else {
            throw LagunaError.generationFailed("Laguna prefill produced no logits.")
        }
        return LagunaPrefillResult(logits: logits, dflashCache: dflashCache)
    }

    private func accumulateDFlashStats(_ stats: LagunaDFlashStats) {
        dflashRounds += stats.rounds
        dflashDraftedTokens += stats.draftedTokens
        dflashAcceptedDraftTokens += stats.acceptedDraftTokens
        dflashRejectedDraftTokens += stats.rejectedDraftTokens
        dflashFullAcceptanceRounds += stats.fullAcceptanceRounds
        dflashTargetVerificationForwards += stats.targetVerificationForwards
        dflashTargetRecoveryForwards += stats.targetRecoveryForwards
        dflashTargetFallbackForwards += stats.targetFallbackForwards
        dflashAdaptiveFallbacks += stats.adaptiveFallbacks
    }

    private func enqueueDecodeRow(
        _ row: LagunaBatchedDecodeRow,
        model: LagunaCausalLM,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
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
        model: LagunaCausalLM,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) {
        guard !decodeLoopRunning else { return }
        decodeLoopRunning = true
        Task {
            await runDecodeLoop(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
        }
    }

    private func runDecodeLoop(
        model: LagunaCausalLM,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) async {
        defer {
            decodeLoopRunning = false
            if !decodeQueue.isEmpty || !activeDecodeRows.isEmpty {
                startDecodeLoopIfNeeded(
                    model: model,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
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
        activeDecodeRows.append(contentsOf: decodeQueue)
        decodeQueue.removeAll(keepingCapacity: true)
    }

    private func selectDecodeRows() -> [LagunaBatchedDecodeRow] {
        let eligible = activeDecodeRows.filter(\.needsDecodeStep)
        let selectedIDs = Set(RuntimeDecodeBatchPlanner.selectRows(
            eligible.map { row in
                RuntimeDecodeBatchRowMetadata(
                    row: row.id,
                    signature: row.caches
                        .map { String(describing: type(of: $0)) }
                        .joined(separator: "|"),
                    position: row.caches.map(\.offset).min() ?? 0
                )
            }
        ))
        return eligible.filter { selectedIDs.contains($0.id) }
    }

    private func decodeOneStep(
        rows: [LagunaBatchedDecodeRow],
        model: LagunaCausalLM,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) throws {
        let sampledRows = rows.filter(\.needsDecodeStep)
        guard !sampledRows.isEmpty else { return }

        for row in sampledRows {
            let token = sampleToken(
                logits: row.logits[0, -1, 0...],
                config: row.generationConfig,
                previousTokens: row.repetitionHistory
            )
            guard !row.eosTokens.contains(token) else {
                row.stopped = true
                continue
            }

            row.generatedTokens.append(token)
            row.repetitionHistory.append(token)
            if row.firstTokenSeconds == nil {
                row.firstTokenSeconds = Date().timeIntervalSince(row.decodeStart)
            }
            if let progressHandler = row.progressHandler {
                let piece = row.progressDecoder.append(
                    decodedText: tokenizerAndTemplate.decode(tokens: row.generatedTokens)
                )
                if piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    row.pendingProgressWhitespace += piece
                } else if !piece.isEmpty {
                    progressHandler(ChatProgress(
                        stage: .generating,
                        message: row.pendingProgressWhitespace + piece
                    ))
                    row.pendingProgressWhitespace = ""
                }
            }
        }

        let continuingRows = sampledRows.filter(\.needsDecodeStep)
        guard !continuingRows.isEmpty else { return }

        if continuingRows.count > 1,
           let batchedCaches = makeBatchedCaches(continuingRows.map(\.caches)) {
            let positions = continuingRows.map { $0.caches.map(\.offset).min() ?? 0 }
            let input = MLXArray(
                continuingRows.compactMap(\.generatedTokens.last).map(Int32.init)
            ).reshaped(continuingRows.count, 1)
            let logits = model.lastPositionLogits(input, cache: batchedCaches)
            MLX.eval(logits)
            guard let splitCaches = splitBatchedCaches(
                batchedCaches,
                rowCount: continuingRows.count
            ) else {
                throw LagunaError.generationFailed(
                    "Laguna could not split ragged decode cache rows."
                )
            }
            for (index, row) in continuingRows.enumerated() {
                row.caches = splitCaches[index]
                row.logits = logits[index..<(index + 1), 0..., 0...]
            }
            batchedDecodeSteps += 1
            if RuntimeDecodeBatchPositionKind.variablePositionBatchCount(positions) > 0 {
                variablePositionBatchedSteps += 1
            } else {
                samePositionBatchedSteps += 1
            }
            totalBatchedRows += continuingRows.count
            maxObservedBatchSize = max(maxObservedBatchSize, continuingRows.count)
            return
        }

        for row in continuingRows {
            guard let token = row.generatedTokens.last else { continue }
            row.logits = model.lastPositionLogits(
                MLXArray([Int32(token)]).reshaped(1, 1),
                cache: row.caches
            )
            MLX.eval(row.logits)
            singleDecodeSteps += 1
        }
    }

    private func makeBatchedCaches(
        _ rowCaches: [[Gemma4AttentionCache]]
    ) -> [Gemma4AttentionCache]? {
        guard let first = rowCaches.first, !first.isEmpty else { return nil }
        guard rowCaches.allSatisfy({ $0.count == first.count }) else { return nil }

        var result: [Gemma4AttentionCache] = []
        result.reserveCapacity(first.count)
        for layerIndex in first.indices {
            guard let cache = LagunaRaggedKVCache(
                rows: rowCaches.map { $0[layerIndex] }
            ) else {
                return nil
            }
            result.append(cache)
        }
        return result
    }

    private func splitBatchedCaches(
        _ caches: [Gemma4AttentionCache],
        rowCount: Int
    ) -> [[Gemma4AttentionCache]]? {
        var rows = Array(repeating: [Gemma4AttentionCache](), count: rowCount)
        for cache in caches {
            guard let split = cache.unbatchedRows(count: rowCount) else {
                return nil
            }
            for index in 0..<rowCount {
                rows[index].append(split[index])
            }
        }
        return rows
    }

    private func finishCompletedDecodeRows() {
        var remaining: [LagunaBatchedDecodeRow] = []
        for row in activeDecodeRows {
            if row.needsDecodeStep {
                remaining.append(row)
            } else {
                row.finish()
            }
        }
        activeDecodeRows = remaining
    }

    private func failRows(_ rows: [LagunaBatchedDecodeRow], with error: Error) {
        let failedIDs = Set(rows.map(\.id))
        for row in rows {
            row.fail(error)
        }
        activeDecodeRows.removeAll { failedIDs.contains($0.id) }
    }

    private func enqueueDFlashDecodeRow(
        _ row: LagunaDFlashBatchedDecodeRow,
        model: LagunaCausalLM,
        dflash: LagunaDFlashModel,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) {
        dflashDecodeQueue.append(row)
        startDFlashDecodeLoopIfNeeded(
            model: model,
            dflash: dflash,
            tokenizerAndTemplate: tokenizerAndTemplate
        )
    }

    private func cancelDFlashDecodeRow(id: UUID) {
        if let index = dflashDecodeQueue.firstIndex(where: { $0.id == id }) {
            dflashDecodeQueue.remove(at: index).fail(CancellationError())
            return
        }
        if let index = activeDFlashDecodeRows.firstIndex(where: { $0.id == id }) {
            activeDFlashDecodeRows.remove(at: index).fail(CancellationError())
        }
    }

    private func startDFlashDecodeLoopIfNeeded(
        model: LagunaCausalLM,
        dflash: LagunaDFlashModel,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) {
        guard !dflashDecodeLoopRunning else { return }
        dflashDecodeLoopRunning = true
        Task {
            await runDFlashDecodeLoop(
                model: model,
                dflash: dflash,
                tokenizerAndTemplate: tokenizerAndTemplate
            )
        }
    }

    private func runDFlashDecodeLoop(
        model: LagunaCausalLM,
        dflash: LagunaDFlashModel,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) async {
        defer {
            dflashDecodeLoopRunning = false
            if !dflashDecodeQueue.isEmpty || !activeDFlashDecodeRows.isEmpty {
                startDFlashDecodeLoopIfNeeded(
                    model: model,
                    dflash: dflash,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
            }
        }

        while !dflashDecodeQueue.isEmpty || !activeDFlashDecodeRows.isEmpty {
            if activeDFlashDecodeRows.isEmpty {
                try? await Task.sleep(nanoseconds: 1_000_000)
            }
            activeDFlashDecodeRows.append(contentsOf: dflashDecodeQueue)
            dflashDecodeQueue.removeAll(keepingCapacity: true)
            let rows = activeDFlashDecodeRows.filter(\.needsDecodeRound)
            guard !rows.isEmpty else {
                finishCompletedDFlashDecodeRows()
                continue
            }
            do {
                try decodeOneDFlashRound(
                    rows: rows,
                    model: model,
                    dflash: dflash,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
            } catch {
                failDFlashRows(rows, with: error)
            }
            finishCompletedDFlashDecodeRows()
            await Task.yield()
        }
    }

    private func decodeOneDFlashRound(
        rows: [LagunaDFlashBatchedDecodeRow],
        model: LagunaCausalLM,
        dflash: LagunaDFlashModel,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) throws {
        for row in rows {
            let anchor = sampleToken(
                logits: row.logits[0, -1, 0...],
                config: row.generationConfig,
                previousTokens: row.repetitionHistory
            )
            if row.eosTokens.contains(anchor) {
                row.stopped = true
            } else {
                emitDFlashToken(
                    anchor,
                    row: row,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
            }
        }

        let continuingRows = rows.filter(\.needsDecodeRound)
        guard !continuingRows.isEmpty else { return }
        let draftCount = min(
            dflashSpeculativeTokens,
            min(
                dflash.config.dflash.blockSize - 1,
                continuingRows.map {
                    $0.tokenBudget - $0.generatedTokens.count
                }.min()!
            )
        )
        precondition(draftCount > 0)

        let draftCandidateRows = continuingRows.map {
            $0.draftCaches.map { $0.fork() }
        }
        guard let batchedDraftCaches = makeBatchedCaches(draftCandidateRows) else {
            throw LagunaError.generationFailed(
                "Laguna could not batch DFlash draft caches."
            )
        }
        let anchorTokens = MLXArray(
            continuingRows.map { Int32($0.generatedTokens.last!) }
        ).reshaped(continuingRows.count, 1)
        let draftLogits = dflash.draftLogits(
            anchorTokens: anchorTokens,
            speculativeTokenCount: draftCount,
            cache: batchedDraftCaches,
            target: model
        )
        MLX.eval(draftLogits)

        var proposals: [[Int]] = []
        var proposalProbabilities: [[MLXArray]] = []
        proposals.reserveCapacity(continuingRows.count)
        proposalProbabilities.reserveCapacity(continuingRows.count)
        for (rowIndex, row) in continuingRows.enumerated() {
            var rowProposals: [Int] = []
            var rowProbabilities: [MLXArray] = []
            var history = row.repetitionHistory
            for draftIndex in 0..<draftCount {
                let proposalLogits = draftLogits[rowIndex, draftIndex, 0...]
                if row.generationConfig.temperature == 0 {
                    let token = sampleToken(
                        logits: proposalLogits,
                        config: row.generationConfig,
                        previousTokens: history
                    )
                    rowProposals.append(token)
                    history.append(token)
                } else {
                    let probabilities = samplingProbabilities(
                        logits: proposalLogits,
                        config: row.generationConfig,
                        previousTokens: history
                    )
                    let token = sampleToken(probabilities: probabilities)
                    rowProposals.append(token)
                    rowProbabilities.append(probabilities)
                    history.append(token)
                }
            }
            proposals.append(rowProposals)
            proposalProbabilities.append(rowProbabilities)
        }

        let targetCandidateRows = continuingRows.map {
            $0.targetCaches.map { $0.fork() }
        }
        guard let batchedTargetCaches = makeBatchedCaches(targetCandidateRows) else {
            throw LagunaError.generationFailed(
                "Laguna could not batch DFlash verification caches."
            )
        }
        let candidateInput = MLXArray(
            zip(continuingRows, proposals).flatMap { row, rowProposals in
                [Int32(row.generatedTokens.last!)] + rowProposals.map(Int32.init)
            }
        ).reshaped(continuingRows.count, draftCount + 1)
        let candidate = model.forward(
            candidateInput,
            cache: batchedTargetCaches,
            captureLayerIndices: Set(dflash.config.dflash.targetLayerIDs)
        )
        MLX.eval([candidate.logits] + Array(candidate.capturedHiddenStates.values))
        guard let candidateCaches = splitBatchedCaches(
            batchedTargetCaches,
            rowCount: continuingRows.count
        ) else {
            throw LagunaError.generationFailed(
                "Laguna could not split DFlash verification caches."
            )
        }

        dflashRounds += continuingRows.count
        dflashDraftedTokens += continuingRows.count * draftCount
        dflashTargetVerificationForwards += 1
        recordBatchedForward(
            positions: continuingRows.map {
                $0.targetCaches.map(\.offset).min() ?? 0
            }
        )

        var recoveries: [LagunaDFlashRecovery] = []
        for (rowIndex, row) in continuingRows.enumerated() {
            let rowProposals = proposals[rowIndex]
            var accepted = 0
            var replacement: Int?
            var history = row.repetitionHistory
            for (draftIndex, proposal) in rowProposals.enumerated() {
                let targetLogits = candidate.logits[rowIndex, draftIndex, 0...]
                if row.generationConfig.temperature == 0 {
                    let targetToken = sampleToken(
                        logits: targetLogits,
                        config: row.generationConfig,
                        previousTokens: history
                    )
                    guard targetToken == proposal else {
                        replacement = targetToken
                        break
                    }
                } else {
                    let targetProbabilities = samplingProbabilities(
                        logits: targetLogits,
                        config: row.generationConfig,
                        previousTokens: history
                    )
                    let draftProbabilities = proposalProbabilities[rowIndex][draftIndex]
                    let draftProbability = max(
                        draftProbabilities[proposal].item(Float.self),
                        Float.leastNonzeroMagnitude
                    )
                    let targetProbability = targetProbabilities[proposal]
                        .item(Float.self)
                    guard Float.random(in: 0..<1)
                        <= min(1, targetProbability / draftProbability) else {
                        replacement = sampleToken(probabilities:
                            LagunaDFlashDecoder.rejectionDistribution(
                                target: targetProbabilities,
                                draft: draftProbabilities
                            )
                        )
                        break
                    }
                }
                accepted += 1
                history.append(proposal)
            }
            dflashAcceptedDraftTokens += accepted

            let rowHiddenStates = Dictionary(
                uniqueKeysWithValues: dflash.config.dflash.targetLayerIDs.map { layerID in
                    (
                        layerID,
                        candidate.capturedHiddenStates[layerID]![
                            rowIndex..<(rowIndex + 1),
                            0...,
                            0...
                        ]
                    )
                }
            )
            if accepted == rowProposals.count {
                dflashFullAcceptanceRounds += 1
                for proposal in rowProposals {
                    if row.eosTokens.contains(proposal) {
                        row.stopped = true
                        break
                    }
                    emitDFlashToken(
                        proposal,
                        row: row,
                        tokenizerAndTemplate: tokenizerAndTemplate
                    )
                    if !row.needsDecodeRound {
                        break
                    }
                }
                guard row.needsDecodeRound else { continue }
                row.targetCaches = candidateCaches[rowIndex]
                row.logits = candidate.logits[
                    rowIndex..<(rowIndex + 1),
                    (candidate.logits.dim(1) - 1)...,
                    0...
                ]
                dflash.appendTargetContext(
                    dflash.combineTargetHiddenStates(rowHiddenStates),
                    cache: row.draftCaches
                )
                evaluateGemma4CacheStorage(row.draftCaches)
                continue
            }

            dflashRejectedDraftTokens += 1
            for proposal in rowProposals.prefix(accepted) {
                if row.eosTokens.contains(proposal) {
                    row.stopped = true
                    break
                }
                emitDFlashToken(
                    proposal,
                    row: row,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
                if !row.needsDecodeRound {
                    break
                }
            }
            guard row.needsDecodeRound,
                  let replacement,
                  !row.eosTokens.contains(replacement) else {
                row.stopped = row.stopped || replacement.map(row.eosTokens.contains) == true
                continue
            }
            let committedCandidateTokenCount = accepted + 1
            recoveries.append(LagunaDFlashRecovery(
                row: row,
                candidateHiddenStates: rowHiddenStates,
                committedCandidateTokenCount: committedCandidateTokenCount,
                replacement: replacement,
                recoveryCache: LagunaDFlashDecoder.commitCandidatePrefix(
                    base: row.targetCaches,
                    candidate: candidateCaches[rowIndex],
                    tokenCount: committedCandidateTokenCount
                )
            ))
        }

        guard !recoveries.isEmpty else { return }
        guard let batchedRecoveryCaches = makeBatchedCaches(
            recoveries.map(\.recoveryCache)
        ) else {
            throw LagunaError.generationFailed(
                "Laguna could not batch DFlash recovery caches."
            )
        }
        let recoveryInput = MLXArray(
            recoveries.map { Int32($0.replacement) }
        ).reshaped(recoveries.count, 1)
        let recovery = model.forward(
            recoveryInput,
            cache: batchedRecoveryCaches,
            captureLayerIndices: Set(dflash.config.dflash.targetLayerIDs)
        )
        MLX.eval([recovery.logits] + Array(recovery.capturedHiddenStates.values))
        guard let recoveryCaches = splitBatchedCaches(
            batchedRecoveryCaches,
            rowCount: recoveries.count
        ) else {
            throw LagunaError.generationFailed(
                "Laguna could not split DFlash recovery caches."
            )
        }
        dflashTargetRecoveryForwards += 1
        recordBatchedForward(
            positions: recoveries.map {
                $0.recoveryCache.map(\.offset).min() ?? 0
            }
        )

        for (recoveryIndex, pending) in recoveries.enumerated() {
            let row = pending.row
            emitDFlashToken(
                pending.replacement,
                row: row,
                tokenizerAndTemplate: tokenizerAndTemplate
            )
            row.targetCaches = recoveryCaches[recoveryIndex]
            row.logits = recovery.logits[
                recoveryIndex..<(recoveryIndex + 1),
                (recovery.logits.dim(1) - 1)...,
                0...
            ]
            let committedHiddenStates = Dictionary(
                uniqueKeysWithValues: dflash.config.dflash.targetLayerIDs.map { layerID in
                    let candidatePrefix = pending.candidateHiddenStates[layerID]![
                        0...,
                        ..<pending.committedCandidateTokenCount,
                        0...
                    ]
                    return (
                        layerID,
                        concatenated(
                            [
                                candidatePrefix,
                                recovery.capturedHiddenStates[layerID]![
                                    recoveryIndex..<(recoveryIndex + 1),
                                    0...,
                                    0...
                                ],
                            ],
                            axis: 1
                        )
                    )
                }
            )
            dflash.appendTargetContext(
                dflash.combineTargetHiddenStates(committedHiddenStates),
                cache: row.draftCaches
            )
            evaluateGemma4CacheStorage(row.draftCaches)
        }
    }

    private func emitDFlashToken(
        _ token: Int,
        row: LagunaDFlashBatchedDecodeRow,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) {
        row.generatedTokens.append(token)
        row.repetitionHistory.append(token)
        if row.firstTokenSeconds == nil {
            row.firstTokenSeconds = Date().timeIntervalSince(row.decodeStart)
        }
        guard let progressHandler = row.progressHandler else { return }
        let piece = row.progressDecoder.append(
            decodedText: tokenizerAndTemplate.decode(tokens: row.generatedTokens)
        )
        if piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            row.pendingProgressWhitespace += piece
        } else if !piece.isEmpty {
            progressHandler(ChatProgress(
                stage: .generating,
                message: row.pendingProgressWhitespace + piece
            ))
            row.pendingProgressWhitespace = ""
        }
    }

    private func recordBatchedForward(positions: [Int]) {
        if positions.count > 1 {
            batchedDecodeSteps += 1
            if RuntimeDecodeBatchPositionKind.variablePositionBatchCount(positions) > 0 {
                variablePositionBatchedSteps += 1
            } else {
                samePositionBatchedSteps += 1
            }
            totalBatchedRows += positions.count
            maxObservedBatchSize = max(maxObservedBatchSize, positions.count)
        } else {
            singleDecodeSteps += 1
        }
    }

    private func finishCompletedDFlashDecodeRows() {
        var remaining: [LagunaDFlashBatchedDecodeRow] = []
        for row in activeDFlashDecodeRows {
            if row.needsDecodeRound {
                remaining.append(row)
            } else {
                row.finish()
            }
        }
        activeDFlashDecodeRows = remaining
    }

    private func failDFlashRows(
        _ rows: [LagunaDFlashBatchedDecodeRow],
        with error: Error
    ) {
        let failedIDs = Set(rows.map(\.id))
        for row in rows {
            row.fail(error)
        }
        activeDFlashDecodeRows.removeAll { failedIDs.contains($0.id) }
    }
}
