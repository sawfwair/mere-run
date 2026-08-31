import Foundation
import MLX
import MLXNN

public typealias LFM2ContinuousBatchingStats = RuntimeDecodeBatchingStats

private struct LFM2PrefillOutput {
    let logits: MLXArray
    let hidden: MLXArray
}

private struct LFM2ModelTypeEnvelope: Decodable {
    let modelType: String

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
    }
}

private struct LFM2DecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
    var firstTokenSeconds: Double? = nil
    var acceleration: ChatAccelerationDiagnostics? = nil
}

struct LFM2DecodeLoopEpochState: Equatable, Sendable {
    private(set) var residencyEpoch: UInt64 = 0
    private(set) var runningEpoch: UInt64?

    @discardableResult
    mutating func beginResidencyTransition() -> UInt64 {
        residencyEpoch &+= 1
        return residencyEpoch
    }

    mutating func startLoopIfCurrent(epoch: UInt64) -> Bool {
        guard epoch == residencyEpoch, runningEpoch != epoch else {
            return false
        }
        runningEpoch = epoch
        return true
    }

    mutating func finishLoop(epoch: UInt64) {
        guard runningEpoch == epoch else { return }
        runningEpoch = nil
    }

    func isCurrent(_ epoch: UInt64) -> Bool {
        epoch == residencyEpoch
    }
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
    let residencyEpoch: UInt64
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
        residencyEpoch: UInt64,
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
        self.residencyEpoch = residencyEpoch
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

    static func prefillTokensPerSecond(
        promptTokenCount: Int,
        prefillSeconds: Double
    ) -> Double? {
        guard promptTokenCount > 0, prefillSeconds > 0 else { return nil }
        return Double(promptTokenCount) / prefillSeconds
    }

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
    private var visionModel: LFM2VLModel?
    private var dspark: LFM2DSparkModel?
    private var tokenizerAndTemplate: LFM2TokenizerAndTemplate?
    private var loadedModelPath: String?
    private var loadedConfig: LFM2Config?
    private var loadedVisionConfig: LFM2VLConfig?
    private var loadedVisionProcessorConfig: LFM2VLProcessorConfig?
    private var loadedTextLoRASignature: String?

    private let modelId: String

    private var decodeQueue: [LFM2BatchedDecodeRow] = []
    private var activeDecodeRows: [LFM2BatchedDecodeRow] = []
    private var decodeLoopEpochState = LFM2DecodeLoopEpochState()
    private var batchedDecodeSteps = 0
    private var samePositionBatchedSteps = 0
    private var variablePositionBatchedSteps = 0
    private var singleDecodeSteps = 0
    private var totalBatchedRows = 0
    private var maxObservedBatchSize = 0
    private var latestDSparkStats = LFM2DSparkStats()

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
        try await Stream.withNewDefaultStream {
            let rootURL = try await resolveModelRoot(
                modelPath: modelPath,
                progressHandler: progressHandler
            )
            let loadStart = Date()
            let requestedLoRASignature = Self.loraSignature(request.lora)
            if loadedTextLoRASignature != requestedLoRASignature {
                guard !hasActiveGeneration else {
                    throw LFM2Error.adapterSwitchDuringActiveGeneration
                }
                beginResidencyTransition()
            }
            try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
            try await applyTextLoRAIfNeeded(request.lora, progressHandler: progressHandler)
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
            let rootURL = try await resolveModelRoot(
                modelPath: modelPath,
                progressHandler: progressHandler
            )
            try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
        }
    }

    public func unload() {
        beginResidencyTransition()
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

    public func dsparkStats() -> LFM2DSparkStats {
        latestDSparkStats
    }

    #if DEBUG
    func decodeLoopEpochStateForTesting() -> LFM2DecodeLoopEpochState {
        decodeLoopEpochState
    }
    #endif

    private func ensureLoaded(
        rootURL: URL,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        let normalizedRoot = LFM2Resources.normalizedRootURL(rootURL)
        if loadedModelPath == normalizedRoot.path, model != nil, tokenizerAndTemplate != nil {
            return
        }
        let loadEpoch = beginResidencyTransition()

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2 config"))
        let configData = try Data(contentsOf: normalizedRoot.appendingPathComponent("config.json"))
        let modelType = try JSONDecoder().decode(LFM2ModelTypeEnvelope.self, from: configData).modelType
        let config: LFM2Config
        let visionConfig: LFM2VLConfig?
        if modelType == "lfm2_vl" {
            let decoded = try JSONDecoder().decode(LFM2VLConfig.self, from: configData)
            config = decoded.textConfig
            visionConfig = decoded
        } else {
            config = try JSONDecoder().decode(LFM2Config.self, from: configData)
            visionConfig = nil
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2 tokenizer"))
        let tokenizer = try await LFM2TokenizerAndTemplate.load(
            from: normalizedRoot,
            maxLengthOverride: min(LFM2Resources.defaultContextLength, config.maxPositionEmbeddings)
        )
        try Task.checkCancellation()
        try requireCurrentResidency(loadEpoch)

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2 weights"))
        let resources = LFM2Resources(rootURL: normalizedRoot)
        let groupSize = visionConfig?.quantization?.groupSize ?? config.quantization?.groupSize ?? 64
        let bits = visionConfig?.quantization?.bits ?? config.quantization?.bits ?? 8
        let quantized = (visionConfig?.quantization ?? config.quantization) != nil
        let lfm2Model: LFM2Model
        let lfm2VisionModel: LFM2VLModel?
        if let visionConfig {
            let composite = LFM2VLModel(config: visionConfig)
            try loadWeights(
                into: composite,
                resources: resources,
                groupSize: groupSize,
                bits: bits,
                quantized: quantized,
                progressHandler: progressHandler
            )
            lfm2Model = composite.languageModel
            lfm2VisionModel = composite
        } else {
            let language = LFM2Model(config: config)
            try loadWeights(
                into: language,
                resources: resources,
                groupSize: groupSize,
                bits: bits,
                quantized: quantized,
                progressHandler: progressHandler
            )
            lfm2Model = language
            lfm2VisionModel = nil
        }

        let lfm2DSparkModel: LFM2DSparkModel?
        if visionConfig == nil,
           let dsparkPath = LFM2Resources.installedDSparkPath(
               for: modelId,
               config: config
           ) {
            let dsparkRoot = URL(fileURLWithPath: dsparkPath).standardizedFileURL
            let missing = LFM2Resources.missingDSparkFiles(rootURL: dsparkRoot)
            guard missing.isEmpty else {
                throw LFM2Error.missingFiles(missing.map { "DSpark/\($0.lastPathComponent)" })
            }
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading LFM2.5 DSpark"))
            let dsparkConfig = try JSONDecoder().decode(
                LFM2DSparkConfig.self,
                from: Data(contentsOf: dsparkRoot.appendingPathComponent("config.json"))
            )
            try Self.validateDSparkCompatibility(target: config, dspark: dsparkConfig)
            let loaded = LFM2DSparkModel(config: dsparkConfig)
            try HFSafetensorsWeightsLoader.applyWeights(
                url: dsparkRoot.appendingPathComponent("model.safetensors"),
                to: loaded,
                dtype: nil,
                verify: .shapeMismatch
            )
            lfm2DSparkModel = loaded
        } else {
            lfm2DSparkModel = nil
        }

        try Task.checkCancellation()
        try requireCurrentResidency(loadEpoch)
        self.model = lfm2Model
        self.visionModel = lfm2VisionModel
        self.dspark = lfm2DSparkModel
        self.tokenizerAndTemplate = tokenizer
        self.loadedConfig = config
        self.loadedVisionConfig = visionConfig
        if visionConfig != nil,
           let data = try? Data(contentsOf: resources.processorConfigURL) {
            self.loadedVisionProcessorConfig = try JSONDecoder().decode(
                LFM2VLProcessorConfig.self,
                from: data
            )
        }
        self.loadedModelPath = normalizedRoot.path
    }

    static func validateDSparkCompatibility(
        target: LFM2Config,
        dspark: LFM2DSparkConfig
    ) throws {
        guard target.vocabSize == dspark.vocabularySize else {
            throw LFM2Error.generationFailed("LFM2.5 DSpark vocabulary does not match its target.")
        }
        guard target.hiddenSize == dspark.hiddenSize else {
            throw LFM2Error.generationFailed("LFM2.5 DSpark hidden size does not match its target.")
        }
        guard target.numHiddenLayers == dspark.features.targetLayerCount,
              dspark.features.targetLayerIDs.allSatisfy({
                  $0 >= 0 && $0 < target.numHiddenLayers
              }) else {
            throw LFM2Error.generationFailed("LFM2.5 DSpark target-layer contract is incompatible.")
        }
    }

    private func loadWeights(
        into model: Module,
        resources: LFM2Resources,
        groupSize: Int,
        bits: Int,
        quantized: Bool,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws {
        if !quantized {
            if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
                try HFSafetensorsWeightsLoader.applyShardedWeights(
                    indexURL: resources.modelIndexURL,
                    to: model,
                    dtype: nil,
                    verify: .shapeMismatch,
                    mapper: LFM2Resources.mapWeight(key:value:),
                    progressHandler: { progress in
                        progressHandler?(ChatProgress(
                            stage: .loadingModel,
                            message: "Loading LFM2 shard \(progress.shardIndex + 1)/\(progress.shardCount)"
                        ))
                    }
                )
            } else {
                try HFSafetensorsWeightsLoader.applyWeights(
                    url: resources.modelWeightsURL,
                    to: model,
                    dtype: nil,
                    verify: .shapeMismatch,
                    mapper: LFM2Resources.mapWeight(key:value:)
                )
            }
            return
        }
        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                indexURL: resources.modelIndexURL,
                to: model,
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
                to: model,
                groupSize: groupSize,
                bits: bits,
                mapper: LFM2Resources.mapWeight(key:value:)
            )
        }
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
        let residencyEpoch = decodeLoopEpochState.residencyEpoch

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
        let imageReferences = request.messages.compactMap { message -> String? in
            guard let value = message.imageUrl?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !value.isEmpty else { return nil }
            return value
        }
        let inputEmbeddings: MLXArray?
        if imageReferences.isEmpty {
            inputEmbeddings = nil
        } else {
            guard let visionModel,
                  let loadedVisionConfig,
                  let imageStartTokenId = tokenizerAndTemplate.tokenizer.convertTokenToId("<|image_start|>"),
                  let imageEndTokenId = tokenizerAndTemplate.tokenizer.convertTokenToId("<|image_end|>") else {
                throw LFM2Error.generationFailed(
                    "This LFM2 checkpoint does not provide the LFM2-VL vision tower and image tokens."
                )
            }
            progressHandler?(ChatProgress(stage: .encoding, message: "Encoding \(imageReferences.count) image(s)"))
            let batch = try LFM2VLImageProcessor.makeBatch(
                imageReferences: imageReferences,
                config: loadedVisionConfig,
                processorConfig: loadedVisionProcessorConfig
            )
            promptTokens = try LFM2VLImageProcessor.expandedPromptTokens(
                promptTokens,
                grids: batch.grids,
                downsampleFactor: loadedVisionConfig.downsampleFactor,
                imageTokenId: loadedVisionConfig.imageTokenId,
                imageStartTokenId: imageStartTokenId,
                imageEndTokenId: imageEndTokenId
            )
            guard promptTokens.count <= effectiveContext else {
                throw LFM2Error.generationFailed(
                    "LFM2-VL prompt requires \(promptTokens.count) tokens after image expansion; context limit is \(effectiveContext)."
                )
            }
            inputEmbeddings = try visionModel.inputEmbeddings(
                inputTokens: promptTokens,
                pixelValues: batch.pixelValues,
                grids: batch.grids
            )
        }
        if imageReferences.isEmpty, promptTokens.count > effectiveContext {
            promptTokens = Array(promptTokens.suffix(effectiveContext))
        }

        let eosSet = Set(
            loadedConfig.eosTokenIds
                + tokenizerAndTemplate.stopTokenIds(withTools: request.tools?.isEmpty == false)
        )
        let generationConfig = GenerationConfig(
            maxTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topK: request.topK ?? (loadedVisionConfig == nil ? 0 : 50),
            topP: Float(request.topP),
            minP: Float(request.minP),
            repetitionPenalty: loadedVisionConfig == nil ? 1.05 : 1.0,
            repetitionContextSize: 64
        )

        let effectiveKVCacheMode: RuntimeKVCacheMode
        switch request.kvCacheMode {
        case .affine4:
            effectiveKVCacheMode = .affine4
        case .affine8:
            effectiveKVCacheMode = .affine8
        default:
            effectiveKVCacheMode = .default
        }
        var layerCaches = makeLayerCaches(config: loadedConfig, kvCacheMode: effectiveKVCacheMode)
        let draftCache = imageReferences.isEmpty ? dspark?.makeCache() : nil
        let prefixCheckpoints = imageReferences.isEmpty
            ? semanticPrefixCheckpoints(
                tokenizerAndTemplate: tokenizerAndTemplate,
                messages: request.messages,
                tools: request.tools,
                includeThinking: request.showThinking,
                promptTokens: promptTokens,
                maxContextLength: effectiveContext
            )
            : []
        var prefillStartIndex = 0
        var prefillExistingLogits: MLXArray?
        if imageReferences.isEmpty, dspark == nil, let seed = prefixKVCacheSeed(
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
            inputEmbeddings: inputEmbeddings,
            modelPath: imageReferences.isEmpty ? loadedModelPath : nil,
            startIndex: prefillStartIndex,
            existingLogits: prefillExistingLogits,
            checkpointTokenCounts: prefixCheckpoints,
            dspark: dspark,
            draftCache: draftCache,
            residencyEpoch: residencyEpoch,
            progressHandler: progressHandler
        )
        try requireCurrentResidency(residencyEpoch)
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        let tokenBudget = max(0, min(request.maxTokens, effectiveContext - promptTokens.count))

        progressHandler?(ChatProgress(stage: .generating, message: ""))
        let decodeResult: LFM2DecodeResult
        if let dspark,
           let draftCache,
           effectiveKVCacheMode == .default,
           !continuousBatchingEnabled,
           !request.logprobCapture.isEnabled,
           LFM2DSparkPolicy.enabled(),
           tokenBudget >= LFM2DSparkPolicy.minimumOutputTokens() {
            let result = try LFM2DSparkDecoder.decode(
                initialLogits: prefillOutput.logits,
                target: model,
                targetCache: layerCaches,
                dspark: dspark,
                draftCache: draftCache,
                generationConfig: generationConfig,
                eosTokens: eosSet,
                tokenBudget: tokenBudget,
                historySeedTokens: promptTokens,
                decodeToken: { tokenizerAndTemplate.decode(token: $0) },
                emitPiece: { _, piece in
                    progressHandler?(ChatProgress(stage: .generating, message: piece))
                },
                checkCancellation: { try Task.checkCancellation() }
            )
            latestDSparkStats = result.stats
            decodeResult = LFM2DecodeResult(
                generatedTokens: result.generatedTokens,
                decodeSeconds: result.decodeSeconds,
                firstTokenSeconds: result.firstTokenSeconds,
                acceleration: ChatAccelerationDiagnostics(
                    route: "dspark-speculative",
                    draftModel: LFM2Resources.dsparkModelID(
                        for: modelId,
                        config: loadedConfig
                    ),
                    rounds: result.stats.rounds,
                    draftedTokens: result.stats.draftedTokens,
                    acceptedDraftTokens: result.stats.acceptedDraftTokens
                )
            )
        } else {
            let reason: String?
            if dspark == nil {
                reason = "companion model is not installed"
            } else if effectiveKVCacheMode != .default {
                reason = "DSpark requires the default KV cache"
            } else if continuousBatchingEnabled {
                reason = "continuous batching uses final-target decode"
            } else if request.logprobCapture.isEnabled {
                reason = "logprob capture requires final-target decode"
            } else if !LFM2DSparkPolicy.enabled() {
                reason = "disabled by MERERUN_LFM25_DSPARK"
            } else if tokenBudget < LFM2DSparkPolicy.minimumOutputTokens() {
                reason = "output budget is below the DSpark break-even gate"
            } else {
                reason = nil
            }
            latestDSparkStats = LFM2DSparkStats(
                enabled: dspark != nil,
                active: false,
                speculativeTokens: dspark?.config.blockSize ?? 0,
                reason: reason
            )
            decodeResult = try await decodeTokens(
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate,
                initialLogits: prefillOutput.logits,
                layerCaches: layerCaches,
                eosSet: eosSet,
                generationConfig: generationConfig,
                tokenBudget: tokenBudget,
                prefillTokenCount: promptTokens.count,
                promptTokens: promptTokens,
                residencyEpoch: residencyEpoch,
                progressHandler: progressHandler
            )
        }

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
                decodeKVCache: effectiveKVCacheMode.genericCacheLabel,
                prefillTokensPerSecond: Self.prefillTokensPerSecond(
                    promptTokenCount: promptTokens.count,
                    prefillSeconds: prefillSeconds
                )
            ),
            toolCalls: toolCalls,
            promptTokens: promptTokens.count,
            acceleration: decodeResult.acceleration
        )
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
        guard let model, let loadedConfig else {
            throw LFM2Error.modelNotLoaded
        }
        guard loadedVisionConfig == nil,
              loadedConfig.modelType == "lfm2_moe",
              loadedConfig.quantization?.bits == 8 else {
            throw LFM2Error.generationFailed(
                "Native LFM2 text LoRA adapters v1 require the affine 8-bit LFM2.5 A1B text runtime."
            )
        }
        progressHandler?(ChatProgress(
            stage: .loadingModel,
            message: "Loading LFM2 text LoRA"
        ))
        _ = try await LFM2TextLoRAAdapter.apply(lora, to: model)
        loadedTextLoRASignature = signature
        prefixKVCache.removeAll(keepingCapacity: false)
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
        residencyEpoch: UInt64,
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
        try Task.checkCancellation()
        try requireCurrentResidency(residencyEpoch)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let row = LFM2BatchedDecodeRow(
                    id: rowID,
                    residencyEpoch: residencyEpoch,
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
        guard decodeLoopEpochState.isCurrent(row.residencyEpoch) else {
            row.fail(CancellationError())
            return
        }
        decodeQueue.append(row)
        startDecodeLoopIfNeeded(
            epoch: row.residencyEpoch,
            model: model,
            tokenizerAndTemplate: tokenizerAndTemplate
        )
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
        epoch: UInt64,
        model: LFM2Model,
        tokenizerAndTemplate: LFM2TokenizerAndTemplate
    ) {
        guard decodeLoopEpochState.startLoopIfCurrent(epoch: epoch) else { return }
        Task {
            await runDecodeLoop(
                epoch: epoch,
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate
            )
        }
    }

    private func runDecodeLoop(
        epoch: UInt64,
        model: LFM2Model,
        tokenizerAndTemplate: LFM2TokenizerAndTemplate
    ) async {
        defer {
            decodeLoopEpochState.finishLoop(epoch: epoch)
            if decodeLoopEpochState.isCurrent(epoch),
               (!decodeQueue.isEmpty || !activeDecodeRows.isEmpty),
               let currentModel = self.model,
               let currentTokenizer = self.tokenizerAndTemplate {
                startDecodeLoopIfNeeded(
                    epoch: epoch,
                    model: currentModel,
                    tokenizerAndTemplate: currentTokenizer
                )
            }
        }

        while !decodeQueue.isEmpty || !activeDecodeRows.isEmpty {
            guard decodeLoopEpochState.isCurrent(epoch) else { return }
            if activeDecodeRows.isEmpty {
                try? await Task.sleep(nanoseconds: 1_000_000)
                guard decodeLoopEpochState.isCurrent(epoch) else { return }
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
        inputEmbeddings: MLXArray? = nil,
        modelPath: String? = nil,
        startIndex: Int = 0,
        existingLogits: MLXArray? = nil,
        checkpointTokenCounts: Set<Int> = [],
        dspark: LFM2DSparkModel? = nil,
        draftCache: [Gemma4AttentionCache]? = nil,
        residencyEpoch: UInt64,
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
            try requireCurrentResidency(residencyEpoch)
            let end = RuntimePrefillCheckpointPlanner.nextEnd(
                processed: offset,
                total: promptTokens.count,
                chunkSize: Self.prefillChunkSize,
                checkpoints: checkpointTokenCounts
            )
            let chunk = Array(promptTokens[offset..<end])
            let input = MLXArray(chunk.map(Int32.init)).reshaped(1, chunk.count)
            let chunkEmbeddings = inputEmbeddings?[0..., offset..<end, 0...]
            let output = model.forwardPrefill(
                input,
                cache: cache,
                inputEmbeddings: chunkEmbeddings,
                captureLayerIndices: dspark.map { Set($0.config.features.targetLayerIDs) } ?? []
            )
            if let dspark, let draftCache {
                dspark.appendTargetContext(
                    dspark.combineTargetHiddenStates(output.capturedHiddenStates),
                    cache: draftCache
                )
            }
            MLX.eval(
                [output.logits, output.hidden]
                    + Array(output.capturedHiddenStates.values)
                    + (draftCache?.flatMap { $0.storageArraysForEvaluation() } ?? [])
            )
            logits = output.logits
            hidden = output.hidden
            offset = end
            if let modelPath,
               let priority = RuntimePrefillCheckpointPlanner.storagePriority(
                   tokenCount: end,
                   total: promptTokens.count,
                   semanticCheckpoints: checkpointTokenCounts
               ) {
                storePrefixKVCache(
                    modelPath: modelPath,
                    promptTokens: promptTokens,
                    tokenCount: end,
                    cache: cache,
                    logits: output.logits,
                    priority: priority
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

    private func semanticPrefixCheckpoints(
        tokenizerAndTemplate: LFM2TokenizerAndTemplate,
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
        logits: MLXArray,
        priority: RuntimePrefixCacheEntryPriority
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
            priority: priority,
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
        let normalizedModel = requestedModel.lowercased()
        guard LFM2Resources.managedModelIds.contains(where: { $0.lowercased() == normalizedModel })
            || LFM2Resources.upstreamRepoIds.contains(where: { $0.lowercased() == normalizedModel })
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
                if kvCacheMode == .affine4 || kvCacheMode == .affine8 {
                    return .attention(AffineQuantizedKVCache(
                        groupSize: Self.affineKVGroupSize(headDimension: config.headDim),
                        bits: kvCacheMode == .affine4 ? 4 : 8,
                        step: 256
                    ))
                }
                return .attention(KVCacheSimple(step: 256))
            }
            return .conv(LFM2ConvCache())
        }
    }

    private var hasActiveGeneration: Bool {
        decodeLoopEpochState.runningEpoch != nil
            || !decodeQueue.isEmpty
            || !activeDecodeRows.isEmpty
    }

    private func cacheMode(for caches: [LFM2LayerCache?]) -> RuntimeKVCacheMode {
        caches.contains { entry in
            guard case .attention(let cache)? = entry else { return false }
            guard let affine = cache as? AffineQuantizedKVCache else { return false }
            return affine.bitWidth == 4
        } ? .affine4 : caches.contains { entry in
            guard case .attention(let cache)? = entry else { return false }
            return cache is AffineQuantizedKVCache
        } ? .affine8 : .default
    }

    private static func affineKVGroupSize(headDimension: Int) -> Int {
        [64, 32, 16, 8].first { headDimension % $0 == 0 } ?? 1
    }

    @discardableResult
    private func beginResidencyTransition() -> UInt64 {
        let epoch = decodeLoopEpochState.beginResidencyTransition()
        failQueuedDecodeRows(CancellationError())
        model = nil
        visionModel = nil
        dspark = nil
        latestDSparkStats = LFM2DSparkStats()
        tokenizerAndTemplate = nil
        loadedModelPath = nil
        loadedConfig = nil
        loadedVisionConfig = nil
        loadedVisionProcessorConfig = nil
        loadedTextLoRASignature = nil
        prefixKVCache.removeAll(keepingCapacity: false)
        Memory.clearCache()
        return epoch
    }

    private func requireCurrentResidency(_ epoch: UInt64) throws {
        guard decodeLoopEpochState.isCurrent(epoch) else {
            throw CancellationError()
        }
    }
}
