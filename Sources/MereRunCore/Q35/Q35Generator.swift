import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

public typealias Q35ContinuousBatchingStats = RuntimeDecodeBatchingStats

private struct Q35PrefixKVCacheKey: Hashable {
    let modelPath: String
    let cacheMode: RuntimeKVCacheMode
    let tokens: [Int]
}

private struct Q35PrefixKVCacheEntry {
    let caches: [Q35LayerCache?]
    let logits: MLXArray
    let hidden: MLXArray
    let mtpSession: Q35MTPDraftSession?
    let priority: RuntimePrefixCacheEntryPriority
    var lastAccess: Date
}

struct Q35PrefillOutput {
    let logits: MLXArray
    let hidden: MLXArray?
    let mtpHistoryHidden: MLXArray?
    let mtpSession: Q35MTPDraftSession?
}

@_spi(Benchmark)
public struct Q35VerificationFrontierResult: Encodable, Sendable {
    public let width: Int
    public let trial: Int
    public let verifiedTokens: Int
    public let verificationPasses: Int
    public let verificationSeconds: Double
    public let verifiedTokensPerSecond: Double
    public let greedyOutputParity: Bool
    public let activeMemoryBytes: Int
    public let cacheMemoryBytes: Int
}

private struct Q35BatchedDecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
    var firstTokenSeconds: Double? = nil
    var logprobs: ChatLogprobDiagnostics? = nil
    var acceleration: ChatAccelerationDiagnostics? = nil
}

enum Q35DecodePath: Equatable {
    case jsonConstrainedSerial
    case continuousBatched
    case mtpSpeculativeSerial
    case pipelined
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
    /// GPU-resident repetition window for batched GPU-side sampling; seeded
    /// lazily from `repetitionHistory` and appended without host readbacks.
    var repetitionHistoryGPU: MLXArray?
    var repetitionHistoryGPUSeeded = false
    var firstTokenSeconds: Double?
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
                decodeSeconds: Date().timeIntervalSince(decodeStart),
                firstTokenSeconds: firstTokenSeconds,
                acceleration: ChatAccelerationDiagnostics(route: "continuous-batched")
            )
        )
    }

    func fail(_ error: Error) {
        continuation.resume(throwing: error)
    }
}

public actor Q35Generator: ChatGenerator {
    private static let contendedPrefillChunkSize = 512
    private static let q38LowPrefillHeadroomBytes = UInt64(16) * 1_024 * 1_024 * 1_024
    private static let minimumReclaimableCacheBytes = 256 * 1_024 * 1_024
    private static let flashNextReusableCacheBytes = 4 * 1_024 * 1_024 * 1_024
    private static let prefixKVCacheMaxEntries = 4
    private static let defaultMTPBlockSize = 4
    /// One committed token plus up to seven Qwen3.8 proposal tokens. Wider
    /// target verification is split only at SDPA, preserving one weight pass.
    private static let q38MTPBlockSize = 8

    /// Continuous-batching decode samples every row on GPU (the same sampler
    /// the serial pipelined path uses) and reads the whole batch back in one
    /// sync per step; the legacy path performed one blocking readback per row
    /// per step. MERERUN_Q35_BATCHED_GPU_SAMPLING=0 restores per-row host
    /// sampling.
    private static let batchedGPUSamplingEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_Q35_BATCHED_GPU_SAMPLING"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }()
    public static let qwen3VLMinPixels = 2_048
    public static let qwen3VLMaxPixels = 16_777_216

    private var model: Q35Model?
    private var tokenizerAndTemplate: Q35TokenizerAndTemplate?
    private var visionTower: Q35VisionTower?
    private var mtpModel: (any Q35MTPDraftModel)?
    private var loadedModelPath: String?
    private var loadedConfig: Q35Config?
    private var loadedGenerationEOSTokenIds: [Int] = []
    private var loadedResources: Q35Resources?
    private var loadedVisionResources: Q35Resources?

    #if DEBUG
    private var checkpointTransformForTesting: (@Sendable (Q35Model) throws -> Void)?

    /// Installed-checkpoint experiments run before fusion releases source weights.
    /// This hook is absent from production builds and never changes source files.
    func setCheckpointTransformForTesting(_ transform: @escaping @Sendable (Q35Model) throws -> Void) {
        precondition(model == nil, "Install checkpoint transforms before loading the model")
        checkpointTransformForTesting = transform
    }

    #endif

    /// Measures target-only linear verification against an oracle sequence
    /// produced by the same loaded target. This benchmark-only API excludes
    /// draft cost and does not implement tree-shaped state.
    @_spi(Benchmark)
    public func benchmarkVerificationFrontier(
        messages: [ChatMessage],
        tokenCount: Int,
        widths: [Int],
        trials: Int
    ) throws -> [Q35VerificationFrontierResult] {
        precondition(tokenCount > 0 && !widths.isEmpty && trials > 0)
        guard let model, let tokenizerAndTemplate else {
            throw Q35Error.modelNotLoaded
        }
        let promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: messages,
            includeThinking: false,
            maxLength: Q35Resources.defaultContextLength
        )
        let promptInput = MLXArray(promptTokens.map(Int32.init)).reshaped(1, promptTokens.count)
        let serialCaches = makeLayerCaches(config: model.config)
        var serial = model.forward(promptInput, cache: serialCaches)
        MLX.eval(serial.logits)
        var oracleTokens: [Int] = []
        oracleTokens.reserveCapacity(tokenCount)
        for _ in 0..<tokenCount {
            let token = MLX.argMax(serial.logits[0, -1, 0...]).item(Int32.self)
            oracleTokens.append(Int(token))
            let input = MLXArray([token]).reshaped(1, 1)
            serial = model.forward(input, cache: serialCaches)
            MLX.eval(serial.logits)
        }

        var results: [Q35VerificationFrontierResult] = []
        for trial in 0..<max(1, trials) {
            let orderedWidths = trial.isMultiple(of: 2) ? widths : Array(widths.reversed())
            for width in orderedWidths {
                let effectiveWidth = max(1, min(width, tokenCount))
                let caches = makeLayerCaches(config: model.config)
                var output = model.forward(promptInput, cache: caches)
                MLX.eval(output.logits)
                var index = 0
                var passes = 0
                var seconds = 0.0
                var parity = MLX.argMax(output.logits[0, -1, 0...]).item(Int32.self)
                    == Int32(oracleTokens[0])
                while index < oracleTokens.count {
                    let end = min(index + effectiveWidth, oracleTokens.count)
                    let values = oracleTokens[index..<end].map(Int32.init)
                    let input = MLXArray(values).reshaped(1, values.count)
                    let started = Date()
                    output = model.forward(
                        input,
                        cache: caches,
                        targetVerify: values.count > 1
                    )
                    MLX.eval(output.logits)
                    seconds += Date().timeIntervalSince(started)
                    passes += 1

                    if values.count > 1 {
                        commitVerificationCaches(caches)
                    }
                    for offset in 0..<values.count where index + offset + 1 < oracleTokens.count {
                        let predicted = MLX.argMax(output.logits[0, offset, 0...]).item(Int32.self)
                        parity = parity && predicted == Int32(oracleTokens[index + offset + 1])
                    }
                    index = end
                }
                results.append(Q35VerificationFrontierResult(
                    width: effectiveWidth,
                    trial: trial,
                    verifiedTokens: oracleTokens.count,
                    verificationPasses: passes,
                    verificationSeconds: seconds,
                    verifiedTokensPerSecond: Double(oracleTokens.count) / seconds,
                    greedyOutputParity: parity,
                    activeMemoryBytes: Memory.activeMemory,
                    cacheMemoryBytes: Memory.cacheMemory
                ))
            }
        }
        return results
    }

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
    private var activeChatRequestCount = 0
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
        visionMinPixels: Int? = nil,
        visionMaxPixels: Int? = nil
    ) {
        let visionPixelBounds = Q35Resources.visionPixelBounds(forModelId: modelId)
        self.modelId = modelId
        self.prefixKVCacheEnabled = prefixKVCacheEnabled
        self.continuousBatchingEnabled = continuousBatchingEnabled
        self.visionMinPixels = visionMinPixels ?? visionPixelBounds.minimum
        self.visionMaxPixels = visionMaxPixels ?? visionPixelBounds.maximum
    }

    /// Select a prefill width from the model, live machine headroom, and current
    /// scheduler contention. Q38's dense BF16 path must not infer activation
    /// headroom from total physical memory: model residency and concurrent work
    /// can consume most unified memory before prefill starts. The environment
    /// override remains an upper bound, not permission to ignore live pressure.
    static func prefillChunkSize(
        modelId: String,
        availableMemory: UInt64? = currentHostAvailableMemoryBytes(),
        activeRequestCount: Int = 1,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        let configured = environment["MERERUN_Q35_PREFILL_CHUNK_TOKENS"]
            .flatMap(Int.init)
            .flatMap { (64...8192).contains($0) ? $0 : nil }
        let base = configured ?? 1_024
        if activeRequestCount > 1 {
            return min(base, contendedPrefillChunkSize)
        }
        if Q35Resources.isQ38ModelId(modelId),
           let availableMemory,
           availableMemory < q38LowPrefillHeadroomBytes {
            return min(base, contendedPrefillChunkSize)
        }
        return base
    }

    static func currentHostAvailableMemoryBytes() -> UInt64? {
        #if canImport(Darwin)
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size) / 4
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        let pageSize = UInt64(getpagesize())
        let availablePages = UInt64(stats.free_count) + UInt64(stats.inactive_count)
        let availableBytes = availablePages.multipliedReportingOverflow(by: pageSize)
        return availableBytes.overflow ? UInt64.max : availableBytes.partialValue
        #else
        return nil
        #endif
    }

    static func shouldClearMLXCache(
        activeMemory: Int,
        cacheMemory: Int,
        memoryLimit: Int,
        isFlashNext: Bool = false
    ) -> Bool {
        guard activeMemory >= 0,
              cacheMemory >= minimumReclaimableCacheBytes,
              memoryLimit > 0 else {
            return false
        }
        // Growing QSA shapes leave buffers that later chunks cannot reuse.
        // The device-wide limit does not reserve headroom for other processes;
        // reclaim disposable buffers without evicting live prefix/KV state.
        if isFlashNext, cacheMemory >= flashNextReusableCacheBytes {
            return true
        }
        let total = activeMemory.addingReportingOverflow(cacheMemory)
        guard !total.overflow else { return true }
        return total.partialValue >= memoryLimit * 9 / 10
    }

    private func clearMLXCacheUnderPressureIfNeeded() {
        let snapshot = Memory.snapshot()
        if Self.shouldClearMLXCache(
            activeMemory: snapshot.activeMemory,
            cacheMemory: snapshot.cacheMemory,
            memoryLimit: Memory.memoryLimit,
            isFlashNext: loadedConfig?.textConfig.isQwen4Exp == true
        ) {
            Memory.clearCache()
        }
    }

    static func qwen3VLTargetSize(
        originalWidth width: Int,
        originalHeight height: Int,
        patchSize: Int,
        spatialMergeSize: Int,
        minPixels: Int = Q35Generator.qwen3VLMinPixels,
        maxPixels: Int = Q35Generator.qwen3VLMaxPixels
    ) throws -> (width: Int, height: Int) {
        let aspectRatio = Double(max(width, height)) / Double(min(width, height))
        guard aspectRatio <= 200 else {
            throw Q35Error.generationFailed(
                "Qwen-family image aspect ratio must not exceed 200; received \(aspectRatio)."
            )
        }
        let factor = max(1, patchSize * max(1, spatialMergeSize))

        func roundedToFactor(_ value: Int) -> Int {
            max(factor, Int((Double(value) / Double(factor)).rounded(.toNearestOrEven)) * factor)
        }

        var targetHeight = roundedToFactor(height)
        var targetWidth = roundedToFactor(width)

        if targetHeight * targetWidth > maxPixels {
            let beta = sqrt(Double(height * width) / Double(maxPixels))
            targetHeight = max(factor, Int(floor(Double(height) / beta / Double(factor))) * factor)
            targetWidth = max(factor, Int(floor(Double(width) / beta / Double(factor))) * factor)
        } else if targetHeight * targetWidth < minPixels {
            let beta = sqrt(Double(minPixels) / Double(height * width))
            targetHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
            targetWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }

        return (targetWidth, targetHeight)
    }

    static func visionTokenLimitPerImage(
        contextLength: Int,
        generationTokenCount: Int,
        nonVisionPromptTokenCount: Int,
        imageCount: Int
    ) -> Int? {
        guard contextLength > 0, imageCount > 0 else { return nil }
        let reservedGenerationTokens = min(contextLength, max(0, generationTokenCount))
        let availableVisionTokens = contextLength
            - reservedGenerationTokens
            - max(0, nonVisionPromptTokenCount)
        guard availableVisionTokens >= imageCount else { return nil }
        return availableVisionTokens / imageCount
    }

    static func visionPixelLimit(
        tokenLimit: Int,
        patchSize: Int,
        spatialMergeSize: Int,
        configuredMaximum: Int
    ) -> Int {
        let factor = max(1, patchSize * max(1, spatialMergeSize))
        let factorArea = factor * factor
        let requestedArea = max(1, tokenLimit).multipliedReportingOverflow(by: factorArea)
        let contextMaximum = requestedArea.overflow ? Int.max : requestedArea.partialValue
        return min(configuredMaximum, contextMaximum)
    }

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        activeChatRequestCount += 1
        defer { activeChatRequestCount = max(0, activeChatRequestCount - 1) }
        return try await Q35CompiledOperations.withNewDefaultStream(
            scoped: Q35RuntimeTuning.isEnabled(.scopedCompilation, modelID: modelId)
        ) {
            let rootURL = try await resolveModelRoot(modelPath: nil, progressHandler: progressHandler)
            let loadStart = Date()
            try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
            let loadSeconds = Date().timeIntervalSince(loadStart)

            var response = try await generate(
                request,
                progressHandler: progressHandler,
                maxContextLength: request.maxContextTokens
                    ?? Q35Resources.defaultContextLength(forModelId: modelId)
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

    public func chat(
        _ request: ChatRequest,
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        activeChatRequestCount += 1
        defer { activeChatRequestCount = max(0, activeChatRequestCount - 1) }
        return try await Q35CompiledOperations.withNewDefaultStream(
            scoped: Q35RuntimeTuning.isEnabled(.scopedCompilation, modelID: modelId)
        ) {
            let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
            let loadStart = Date()
            try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
            let loadSeconds = Date().timeIntervalSince(loadStart)

            var response = try await generate(
                request,
                progressHandler: progressHandler,
                maxContextLength: request.maxContextTokens
                    ?? Q35Resources.defaultContextLength(forModelId: modelId)
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
        modelPath: String? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        try await Q35CompiledOperations.withNewDefaultStream(
            scoped: Q35RuntimeTuning.isEnabled(.scopedCompilation, modelID: modelId)
        ) {
            let rootURL = try await resolveModelRoot(
                modelPath: modelPath,
                progressHandler: progressHandler
            )
            try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
        }
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
        loadedGenerationEOSTokenIds = []
        loadedResources = nil
        loadedVisionResources = nil
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
        let generationConfigURL = normalizedRoot.appendingPathComponent("generation_config.json")
        let generationEOSTokenIds: [Int]
        if FileManager.default.fileExists(atPath: generationConfigURL.path) {
            let generationConfigData = try Data(contentsOf: generationConfigURL)
            generationEOSTokenIds = try JSONDecoder()
                .decode(Q35GenerationConfig.self, from: generationConfigData)
                .eosTokenIds
        } else {
            generationEOSTokenIds = []
        }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family tokenizer"))
        let tokenizer = try Q35TokenizerAndTemplate.load(
            from: normalizedRoot,
            maxLengthOverride: config.textConfig.maxPositionEmbeddings
        )

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family weights"))
        let q35Model = Q35Model(
            config: config,
            asynchronousDecodeBlocks: Q35RuntimeTuning.isEnabled(.asynchronousDecode, modelID: modelId)
        )
        let resources = Q35Resources(rootURL: normalizedRoot)

        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4

        try loadTextWeights(
            into: q35Model,
            from: resources,
            groupSize: groupSize,
            bits: bits
        )
        if config.textConfig.isQwen4Exp {
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Loading Qwen4Exp n-gram embedding shards"
            ))
            try loadQ38NGramEmbeddings(
                into: q35Model,
                from: resources,
                progressHandler: progressHandler
            )
        }
        #if DEBUG
        try checkpointTransformForTesting?(q35Model)
        #endif
        if Q35FusedSwitchGLUPolicy.enabled, config.textConfig.usesMoE {
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Preparing Qwen-family fused MoE weights"
            ))
            _ = q35Model.prepareFusedSwitchGLU()
        }

        let primaryResources = Q35Resources(rootURL: normalizedRoot)
        let visionConfigAndResources: (Q35Config, Q35Resources)?
        if config.visionConfig != nil {
            visionConfigAndResources = (config, primaryResources)
        } else if modelId == Q35Resources.q38TwentySevenB4BitModelId {
            let companionResources = primaryResources.q38VisionComponentResources
            let companionConfigData = try Data(contentsOf: companionResources.configURL)
            let companionConfig = try JSONDecoder().decode(Q35Config.self, from: companionConfigData)
            guard companionConfig.visionConfig != nil else {
                throw Q35Error.generationFailed("Qwen3.8 4-bit vision companion does not include a vision config.")
            }
            visionConfigAndResources = (companionConfig, companionResources)
        } else if modelId == Q35Resources.ornith35BMLX4BitModelId {
            let companionResources = primaryResources.ornithVisionComponentResources
            let companionConfigData = try Data(contentsOf: companionResources.configURL)
            let companionConfig = try JSONDecoder().decode(Q35Config.self, from: companionConfigData)
            guard companionConfig.visionConfig != nil else {
                throw Q35Error.generationFailed("Ornith 4-bit vision companion does not include a vision config.")
            }
            visionConfigAndResources = (companionConfig, companionResources)
        } else {
            visionConfigAndResources = nil
        }
        let tower = visionConfigAndResources.map { Q35VisionTower(config: $0.0) }
        // Load the MTP draft head whenever it ships with the model and isn't
        // explicitly disabled. Whether speculation is actually USED is decided
        // by the model-specific policy in Self.shouldSpeculate. BF16 Qwen3.8
        // remains opt-in, Q4 Qwen3.8 and Ornith use their short-context
        // paths, and Qwen3.6 hybrid MoE retains the measured long-context threshold.
        let mtpPolicy = ProcessInfo.processInfo.environment["MERERUN_Q35_MTP_SPECULATION"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let mtpExplicitlyDisabled = mtpPolicy == "0" || mtpPolicy == "false" || mtpPolicy == "no"
        let mtpExplicitlyEnabled = mtpPolicy == "1" || mtpPolicy == "true"
            || mtpPolicy == "yes" || mtpPolicy == "on"
        let q38Q4 = modelId == Q35Resources.q38TwentySevenB4BitModelId
        let shouldLoadMTP = !mtpExplicitlyDisabled
            && (config.textConfig.usesMoE || q38Q4 || mtpExplicitlyEnabled)
        let loadedMTP: (any Q35MTPDraftModel)?
        let ornithMTPCompanionRoot = Q35Resources.isOrnith35BMLXModelId(modelId)
            ? ManagedModelResolver.resolveInstalledModel(id: Q35Resources.ornith35BMTPModelId)
            : nil
        if shouldLoadMTP,
           let mtpResources = Self.mtpResources(
               primary: resources,
               companionRootURL: ornithMTPCompanionRoot
           ) {
            if config.textConfig.isQwen4Exp {
                guard config.textConfig.mtp?.numHiddenLayers == 1 else {
                    throw Q35Error.generationFailed(
                        "Qwen4Exp MTP requires the published one-layer draft configuration."
                    )
                }
                progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen4Exp MTP weights"))
                let mtp = Q38MTPModel(config: config)
                try loadQ38MTPWeights(
                    into: mtp,
                    from: mtpResources,
                    groupSize: groupSize,
                    bits: bits
                )
                if Q35FusedSwitchGLUPolicy.enabled {
                    progressHandler?(ChatProgress(
                        stage: .loadingModel,
                        message: "Preparing Qwen4Exp MTP fused MoE weights"
                    ))
                    _ = mtp.prepareFusedSwitchGLU()
                }
                loadedMTP = mtp
            } else {
                progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family MTP weights"))
                let mtp = Q35MTPModel(config: config)
                try loadMTPWeights(
                    into: mtp,
                    baseModel: q35Model,
                    from: mtpResources,
                    groupSize: groupSize,
                    bits: bits
                )
                loadedMTP = mtp
            }
            progressHandler?(ChatProgress(stage: .loadingModel, message: "Preparing Qwen-family MTP drafting"))
            q35Model.prepareGreedyMTPDrafting()
        } else {
            loadedMTP = nil
        }

        model = q35Model
        tokenizerAndTemplate = tokenizer
        visionTower = tower
        mtpModel = loadedMTP
        loadedConfig = config
        loadedGenerationEOSTokenIds = generationEOSTokenIds
        loadedResources = resources
        loadedVisionResources = visionConfigAndResources?.1
        loadedModelPath = normalizedRoot.path
    }

    private func generate(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        maxContextLength: Int
    ) async throws -> ChatResponse {
        try await Q35Sampling.withRequestState(seed: request.seed) {
            try await generateWithRequestState(
                request, progressHandler: progressHandler, maxContextLength: maxContextLength
            )
        }
    }

    private func generateWithRequestState(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?,
        maxContextLength: Int
    ) async throws -> ChatResponse {
        guard let model,
              let tokenizerAndTemplate,
              let loadedConfig else {
            throw Q35Error.modelNotLoaded
        }
        // An exact prefix-cache hit skips the prefill loop entirely. Reclaim
        // disposable buffers from the previous request before forking its KV.
        if loadedConfig.textConfig.isQwen4Exp {
            clearMLXCacheUnderPressureIfNeeded()
        }

        let messages = request.messages
        let jsonConstrained = request.requiresJSON
        let includeThinking = request.showThinking && !jsonConstrained
        let nativeReasoningEffort = Q35Resources.isQ38ModelId(modelId)
            ? Q35Resources.q38ReasoningEffortLabel(for: request.reasoningEffort)
            : nil
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
        var visionTokenLimitPerImage: Int?

        if !imageURLs.isEmpty {
            progressHandler?(ChatProgress(stage: .encoding, message: "Encoding images"))
            guard visionTower != nil else {
                throw Q35Error.generationFailed("Model \(modelId) does not include a vision tower; use text-only prompts.")
            }
            guard let imageTokenId = loadedConfig.imageTokenId ?? tokenizerAndTemplate.tokenizer.imageTokenId else {
                throw Q35Error.generationFailed("Qwen-family tokenizer is missing the image placeholder token.")
            }
            let promptWithSingleImageTokens = try tokenizerAndTemplate.encodeForGeneration(
                messages: messages,
                tools: request.tools,
                addGenerationPrompt: true,
                includeThinking: includeThinking,
                reasoningEffort: nativeReasoningEffort,
                maxLength: loadedConfig.textConfig.maxPositionEmbeddings,
                imageTokenCounts: Array(repeating: 1, count: imageURLs.count)
            )
            let placeholderCount = promptWithSingleImageTokens.filter { $0 == imageTokenId }.count
            guard placeholderCount == imageURLs.count else {
                throw Q35Error.generationFailed(
                    "Qwen-family prompt did not preserve one placeholder for each encoded image."
                )
            }
            let nonVisionPromptTokenCount = promptWithSingleImageTokens.count - placeholderCount
            guard let perImageLimit = Self.visionTokenLimitPerImage(
                contextLength: effectiveContext,
                generationTokenCount: request.maxTokens,
                nonVisionPromptTokenCount: nonVisionPromptTokenCount,
                imageCount: imageURLs.count
            ) else {
                throw Q35Error.generationFailed(
                    "Qwen-family prompt and requested generation leave no context for \(imageURLs.count) image(s); "
                        + "increase maxContextTokens or lower maxTokens."
                )
            }
            visionTokenLimitPerImage = perImageLimit
            try ensureVisionWeightsLoaded(progressHandler: progressHandler)
            if let visionTower {
                visionReplacements = try buildVisionReplacements(
                    imageURLs: imageURLs,
                    visionTower: visionTower,
                    maximumTokensPerImage: perImageLimit
                )
            }
        }

        var promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: messages,
            tools: request.tools,
            addGenerationPrompt: true,
            includeThinking: includeThinking,
            reasoningEffort: nativeReasoningEffort,
            maxLength: imageURLs.isEmpty
                ? effectiveContext
                : loadedConfig.textConfig.maxPositionEmbeddings,
            imageTokenCounts: visionReplacements.map { max(1, $0.embeddings.dim(0)) }
        )
        if visionTokenLimitPerImage != nil, promptTokens.count > effectiveContext {
            throw Q35Error.generationFailed(
                "Qwen-family vision prompt exceeded maxContextTokens after context-aware image resizing."
            )
        } else if promptTokens.count > effectiveContext {
            promptTokens = Array(promptTokens.suffix(effectiveContext))
        }
        if let dumpPath = ProcessInfo.processInfo.environment["MERERUN_Q35_DEBUG_PROMPT_TOKENS"] {
            try? promptTokens.map(String.init).joined(separator: ",")
                .write(toFile: dumpPath, atomically: true, encoding: .utf8)
        }

        let eosSet = request.stopOnEOS
            ? Set(
                loadedConfig.eosTokenIds
                    + loadedGenerationEOSTokenIds
                    + [tokenizerAndTemplate.eosTokenId].compactMap { $0 }
            )
            : Set<Int>()
        let generationConfig = GenerationConfig(
            maxTokens: request.maxTokens,
            temperature: Float(request.temperature),
            topK: request.topK ?? 0,
            topP: Float(request.topP),
            minP: Float(request.minP),
            repetitionPenalty: nil,
            repetitionContextSize: 64
        )
        let mtpSpeculationEligible = !request.logprobCapture.isEnabled
            && !jsonConstrained
            && request.tools?.isEmpty != false
            && imageURLs.isEmpty
            && Self.shouldSpeculate(
                modelId: modelId,
                usesMoE: loadedConfig.textConfig.usesMoE,
                promptTokenCount: promptTokens.count,
                maxContextTokens: effectiveContext
            )
            && mtpModel != nil
        let historyMode = Self.mtpHistoryMode(
            modelId: modelId,
            isQwen4Exp: loadedConfig.textConfig.isQwen4Exp,
            usesMoE: loadedConfig.textConfig.usesMoE,
            speculationEligible: mtpSpeculationEligible,
            greedy: generationConfig.temperature == 0,
            promptTokenCount: promptTokens.count
        )
        let retainMTPPromptHistory = historyMode != .none
        let streamMTPHistory = historyMode == .streaming
        let retainPrefillHidden = prefixKVCacheEnabled || mtpSpeculationEligible
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
        let promptInput = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, promptTokens.count)

        var prefillOutput: Q35PrefillOutput
        var prefillLength = promptTokens.count
        var mropeRopeDelta: Int?

        if imageURLs.isEmpty {
            let prefixSeed = retainMTPPromptHistory && !streamMTPHistory ? nil : prefixKVCacheSeed(
                modelPath: loadedModelPath ?? "",
                promptTokens: promptTokens,
                cacheMode: effectiveKVCacheMode,
                requiresMTPSession: streamMTPHistory
            )
            let prefixCheckpoints = semanticPrefixCheckpoints(
                tokenizerAndTemplate: tokenizerAndTemplate,
                messages: messages,
                tools: request.tools,
                includeThinking: includeThinking,
                reasoningEffort: nativeReasoningEffort,
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
                retainMTPHistory: retainMTPPromptHistory,
                mtpSession: streamMTPHistory
                    ? (prefixSeed?.mtpSession ?? Q35MTPDraftSession(
                        historyCache: loadedConfig.textConfig.isQwen4Exp ? Q38QSACache() : KVCacheSimple()
                    )) : nil,
                prefillMTPModel: streamMTPHistory ? mtpModel : nil,
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
                        inputIds: promptInput,
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
            prefillMTPHidden: prefillOutput.mtpHistoryHidden,
            prefillMTPSession: prefillOutput.mtpSession,
            layerCaches: layerCaches,
            eosSet: eosSet,
            generationConfig: generationConfig,
            tokenBudget: tokenBudget,
            prefillTokenCount: prefillLength,
            mropeRopeDelta: mropeRopeDelta,
            promptTokens: promptTokens,
            maxContextTokens: effectiveContext,
            jsonConstrained: jsonConstrained,
            stopAtCompletedToolCall: request.tools?.isEmpty == false && !request.parallelToolCalls,
            logprobCapture: request.logprobCapture,
            logprobRegion: request.logprobRegionHint ?? .visible,
            progressHandler: progressHandler
        )

        let stopSequences = TextGenerationStopSequences.merged(request.stopSequences)
        let decodedRaw = tokenizerAndTemplate.decode(tokens: decodeResult.generatedTokens)
        let trimmed = TextGenerationStopSequences.trimming(decodedRaw, sequences: stopSequences)
        let decoded = trimmed.text
        let finishReason = Self.finishReason(
            generatedTokenCount: decodeResult.generatedTokens.count,
            tokenBudget: tokenBudget,
            matchedStopSequence: trimmed.matchedSequence != nil
        )
        let parsedToolCalls = request.tools?.isEmpty == false
            ? Q35ToolParser.parseToolCalls(decoded)
            : []
        let toolCalls: [ToolCall]? = request.tools.flatMap { tools -> [ToolCall]? in
            let validated = ToolCallPolicy.validatedCalls(
                parsedToolCalls,
                tools: tools,
                parallelToolCalls: request.parallelToolCalls
            )
            return validated.isEmpty ? nil : validated
        }
        let visibleText = parsedToolCalls.isEmpty
            ? decoded
            : Q35ToolParser.visibleText(decoded)

        return ChatResponse(
            generatedText: visibleText,
            tokensGenerated: decodeResult.generatedTokens.count,
            showThinking: includeThinking,
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
            promptTokens: promptTokens.count,
            finishReason: finishReason,
            logprobs: decodeResult.logprobs,
            acceleration: decodeResult.acceleration
        )
    }

    static func finishReason(
        generatedTokenCount: Int,
        tokenBudget: Int,
        matchedStopSequence: Bool
    ) -> ChatFinishReason {
        if matchedStopSequence { return .stopSequence }
        return generatedTokenCount >= tokenBudget ? .length : .stop
    }

    static func decodePath(
        jsonConstrained: Bool,
        continuousBatchingEnabled: Bool,
        mtpSpeculationEnabled: Bool,
        schedulerContended: Bool = false,
        stopAtCompletedToolCall: Bool = false
    ) -> Q35DecodePath {
        if stopAtCompletedToolCall { return .pipelined }
        if jsonConstrained { return .jsonConstrainedSerial }
        if mtpSpeculationEnabled, !schedulerContended { return .mtpSpeculativeSerial }
        if continuousBatchingEnabled { return .continuousBatched }
        if mtpSpeculationEnabled { return .mtpSpeculativeSerial }
        return .pipelined
    }

    /// Decide whether to use MTP speculative decode for a request.
    ///
    /// Select the model-specific MTP break-even point. Qwen3.6 hybrid MoE uses
    /// the measured long-context threshold (~20-token context -31%, ~4K -22%,
    /// ~12K +1.5-2.5x on M4 Max). Ornith 1.5 uses its validated MTP companion
    /// from short prompts. Quantized Qwen3.8 uses serial-exact small-batch Q4
    /// verification from short prompts; its BF16 sibling remains opt-in.
    /// Qwen4Exp uses its exact verified inline head from short prompts.
    /// MERERUN_Q35_MTP_SPECULATION can enable/disable the path and
    /// MERERUN_Q35_MTP_MIN_PROMPT_TOKENS overrides either default.
    static func shouldSpeculate(
        promptTokenCount: Int,
        maxContextTokens: Int? = nil,
        defaultMinimumPromptTokens: Int = 6144,
        enabledByDefault: Bool = true
    ) -> Bool {
        shouldSpeculate(
            promptTokenCount: promptTokenCount,
            maxContextTokens: maxContextTokens,
            defaultMinimumPromptTokens: defaultMinimumPromptTokens,
            enabledByDefault: enabledByDefault,
            environment: ProcessInfo.processInfo.environment
        )
    }

    static func shouldSpeculate(
        promptTokenCount: Int,
        maxContextTokens: Int?,
        defaultMinimumPromptTokens: Int = 6144,
        enabledByDefault: Bool = true,
        environment env: [String: String]
    ) -> Bool {
        let rawPolicy = env["MERERUN_Q35_MTP_SPECULATION"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if rawPolicy == "0" || rawPolicy == "false" || rawPolicy == "no" {
            return false
        }
        let explicitlyEnabled = rawPolicy == "1" || rawPolicy == "true"
            || rawPolicy == "yes" || rawPolicy == "on"
        if !enabledByDefault, !explicitlyEnabled {
            return false
        }

        let threshold = env["MERERUN_Q35_MTP_MIN_PROMPT_TOKENS"].flatMap { Int($0) }
            ?? defaultMinimumPromptTokens
        let contextAllowsSpeculation = maxContextTokens.map { $0 >= threshold } ?? true
        if explicitlyEnabled {
            return contextAllowsSpeculation
        }
        if !contextAllowsSpeculation {
            return false
        }
        return promptTokenCount >= threshold
    }

    static func defaultMTPMinimumPromptTokens(modelId: String, usesMoE: Bool) -> Int {
        if Q35Resources.isOrnith35BModelId(modelId) {
            return 0
        }
        if modelId == Q35Resources.q38FlashNextMixedModelId
            || modelId == Q35Resources.q38FlashNext3BitModelId
            || modelId == Q35Resources.q38FlashNext3BitNativePLEModelId
            || modelId == Q35Resources.q38FlashNext4BitModelId {
            return 0
        }
        return usesMoE ? 6144 : 0
    }

    static func mtpBlockSize(
        modelId: String = Q35Resources.q36NanoModelId,
        environment env: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        let denseQ38 = modelId == Q35Resources.q38TwentySevenBModelId
            || modelId == Q35Resources.q38TwentySevenB4BitModelId
        let modelDefault = denseQ38
            ? q38MTPBlockSize
            : defaultMTPBlockSize
        guard let raw = env["MERERUN_Q35_MTP_BLOCK_SIZE"],
              let value = Int(raw), value >= 2 else {
            return modelDefault
        }
        let maximum = modelId == Q35Resources.q38TwentySevenB4BitModelId ? 9 : 16
        return min(maximum, value)
    }

    private func decodeTokens(
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate,
        initialLogits: MLXArray,
        initialHidden: MLXArray?,
        prefillMTPHidden: MLXArray?,
        prefillMTPSession: Q35MTPDraftSession?,
        layerCaches: [Q35LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        prefillTokenCount: Int,
        mropeRopeDelta: Int?,
        promptTokens: [Int],
        maxContextTokens: Int,
        jsonConstrained: Bool,
        stopAtCompletedToolCall: Bool,
        logprobCapture: ChatLogprobCapture,
        logprobRegion: ChatLogprobRegion,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35BatchedDecodeResult {
        guard tokenBudget > 0 else {
            return Q35BatchedDecodeResult(generatedTokens: [], decodeSeconds: 0)
        }
        let speculationMTP = !logprobCapture.isEnabled
            && !jsonConstrained && !stopAtCompletedToolCall && Self.shouldSpeculate(
            modelId: modelId,
            usesMoE: model.config.textConfig.usesMoE,
            promptTokenCount: promptTokens.count,
            maxContextTokens: maxContextTokens
        ) && mropeRopeDelta == nil ? mtpModel : nil
        let decodePath = Self.decodePath(
            jsonConstrained: jsonConstrained,
            continuousBatchingEnabled: continuousBatchingEnabled && !logprobCapture.isEnabled,
            mtpSpeculationEnabled: speculationMTP != nil,
            schedulerContended: activeChatRequestCount > 1
                || !decodeQueue.isEmpty
                || !activeDecodeRows.isEmpty,
            stopAtCompletedToolCall: stopAtCompletedToolCall
        )

        // JSON mode owns mutable prefix-grammar state and must validate every
        // token before it is streamed. Its plan therefore cannot select continuous
        // batching, MTP speculation, or the shared pipelined decoder.
        if decodePath != .continuousBatched {
            return try await decodeTokensSerially(
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate,
                initialLogits: initialLogits,
                initialHidden: initialHidden,
                prefillMTPHidden: prefillMTPHidden,
                prefillMTPSession: prefillMTPSession,
                mtpModel: decodePath == .mtpSpeculativeSerial ? speculationMTP : nil,
                layerCaches: layerCaches,
                eosSet: eosSet,
                generationConfig: generationConfig,
                tokenBudget: tokenBudget,
                prefillTokenCount: prefillTokenCount,
                mropeRopeDelta: mropeRopeDelta,
                promptTokens: promptTokens,
                jsonConstrained: decodePath == .jsonConstrainedSerial,
                stopAtCompletedToolCall: stopAtCompletedToolCall,
                logprobCapture: logprobCapture,
                logprobRegion: logprobRegion,
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
        prefillMTPHidden: MLXArray?,
        prefillMTPSession: Q35MTPDraftSession?,
        mtpModel: (any Q35MTPDraftModel)?,
        layerCaches: [Q35LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        prefillTokenCount: Int,
        mropeRopeDelta: Int?,
        promptTokens: [Int],
        jsonConstrained: Bool,
        stopAtCompletedToolCall: Bool,
        logprobCapture: ChatLogprobCapture,
        logprobRegion: ChatLogprobRegion,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35BatchedDecodeResult {
        if mtpModel == nil, !jsonConstrained {
            return try await decodeTokensPipelined(
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate,
                initialLogits: initialLogits,
                layerCaches: layerCaches,
                eosSet: eosSet,
                generationConfig: generationConfig,
                tokenBudget: tokenBudget,
                mropeRopeDelta: mropeRopeDelta,
                promptTokens: promptTokens,
                stopAtCompletedToolCall: stopAtCompletedToolCall,
                logprobCapture: logprobCapture,
                logprobRegion: logprobRegion,
                progressHandler: progressHandler
            )
        }

        var logits = initialLogits
        var layerCaches = layerCaches
        let retainHidden = mtpModel != nil
        var previousHidden = retainHidden ? initialHidden.map(lastTokenHidden) : nil
        var generated: [Int] = []
        generated.reserveCapacity(tokenBudget)
        var repetitionHistory = promptTokens
        var pendingProgressWhitespace = ""
        var streamedJSONText = ""
        var firstTokenSeconds: Double?
        var jsonGrammar = JSONObjectPrefixGrammar()
        var mtpDraftedTokens = 0
        var mtpAcceptedTokens = 0
        var mtpVerificationPasses = 0
        var mtpReplacementPasses = 0
        var mtpNonDraftingRounds = 0
        var usePipelinedFallback = false
        let supportsPipelinedFallback = modelId == Q35Resources.q38TwentySevenB4BitModelId
            || modelId == Q35Resources.ornith35BMLX4BitModelId
        let pipelinedFallbackEnabled = Q35RuntimeTuning.isEnabled(.pipelinedFallback, modelID: modelId)
        let mtpBlockSize = Self.mtpBlockSize(modelId: modelId)
        var mtpAdaptivePolicy = Q35MTPAdaptivePolicy(
            maxDraftDepth: mtpBlockSize - 1, headStepCostRatio: Q35MTPAdaptivePolicy.configuredCostRatio()
        )
        let mtpDraftSession = prefillMTPSession ?? Q35MTPDraftSession(
            promptTokens: promptTokens,
            promptHidden: prefillMTPHidden,
            historyCache: model.config.textConfig.isQwen4Exp ? Q38QSACache() : KVCacheSimple()
        )
        let draftHistoryTokens = mtpDraftSession.committedHistoryCount
        let mtpProfile = generationConfig.temperature == 0 && mtpModel != nil ? Q35MTPProfile.make() : nil
        let decodeStart = Date()

        func emit(_ token: Int) {
            generated.append(token)
            repetitionHistory.append(token)
            if firstTokenSeconds == nil {
                firstTokenSeconds = Date().timeIntervalSince(decodeStart)
            }
            guard let progressHandler else { return }
            if jsonConstrained {
                // Byte-fallback BPE tokens can decode individually as U+FFFD even
                // though the cumulative token sequence decodes to valid Unicode.
                // Stream only the stable cumulative prefix and wait for trailing
                // replacement scalars to resolve before exposing them.
                var stableText = tokenizerAndTemplate.decode(tokens: generated)
                while stableText.last == "\u{FFFD}" {
                    stableText.removeLast()
                }
                guard stableText.hasPrefix(streamedJSONText) else { return }
                let deltaStart = stableText.index(
                    stableText.startIndex,
                    offsetBy: streamedJSONText.count
                )
                let delta = String(stableText[deltaStart...])
                streamedJSONText = stableText
                if !delta.isEmpty {
                    progressHandler(ChatProgress(stage: .generating, message: delta))
                }
                return
            }
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
            progressHandler(ChatProgress(stage: .generating, message: visiblePiece))
        }

        while generated.count < tokenBudget {
            try Task.checkCancellation()
            if model.config.textConfig.isQwen4Exp {
                clearMLXCacheUnderPressureIfNeeded()
            }
            let samplingStart = mtpProfile?.clock()
            var next = sampleToken(
                logits: logits[0, -1, 0...],
                config: generationConfig,
                previousTokens: repetitionHistory
            )

            mtpProfile?.recordSampling(since: samplingStart)
            if jsonConstrained {
                guard let constrained = jsonConstrainedToken(
                    initial: next,
                    logits: logits[0, -1, 0...],
                    config: generationConfig,
                    eosSet: eosSet,
                    grammar: &jsonGrammar,
                    decode: { tokenizerAndTemplate.decode(token: $0) }
                ) else {
                    break
                }
                next = constrained
            }

            if eosSet.contains(next) {
                break
            }

            emit(next)

            if jsonConstrained, jsonGrammar.isComplete {
                break
            }

            guard generated.count < tokenBudget else {
                break
            }

            if let mtpModel, let hidden = previousHidden {
                let positionOffset = prefillTokenCount + generated.count - 1
                if generationConfig.temperature == 0 {
                    let offeredDepth = min(
                        mtpBlockSize - 1,
                        tokenBudget - generated.count
                    )
                    let draftDepth = mtpAdaptivePolicy.draftDepth(offeredDepth: offeredDepth)
                    if draftDepth > 0 {
                        mtpProfile?.beginRound(depth: draftDepth)
                        defer { mtpProfile?.finishRound() }
                        let draftBlock = mtpModel.draftBlock(
                            lastToken: next,
                            hidden: hidden,
                            blockSize: draftDepth + 1,
                            session: mtpDraftSession,
                            baseModel: model
                        )
                        mtpProfile?.finishDraft(tokens: draftBlock.tokenIDs)
                        mtpDraftedTokens += draftBlock.count

                        let candidateCaches = forkLayerCaches(layerCaches)
                        let nextToken = MLXArray([Int32(next)]).reshaped(1, 1)
                        let candidateInput = MLX.concatenated(
                            [nextToken, draftBlock.tokenIDs],
                            axis: 1
                        )
                        let candidate = model.forward(
                            candidateInput,
                            cache: candidateCaches,
                            targetVerify: true
                        )
                        let candidateMTPHidden = candidate.mtpHidden ?? candidate.hidden
                        mtpProfile?.verificationSubmitted()
                        MLX.eval(candidate.logits, candidateMTPHidden, draftBlock.tokenIDs)
                        mtpProfile?.verificationCompleted()
                        mtpVerificationPasses += 1
                        let draftTokens = draftBlock.tokens

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
                        mtpProfile?.accepted(accepted)
                        mtpAdaptivePolicy.record(
                            acceptedDrafts: accepted,
                            drafted: draftTokens.count
                        )

                        if accepted == draftTokens.count {
                            mtpAcceptedTokens += accepted
                            mtpDraftSession.recordCommittedTransitions(
                                hiddenStates: candidateMTPHidden,
                                nextTokens: draftTokens
                            )
                            var hitEOS = false
                            for token in draftTokens {
                                if eosSet.contains(token) {
                                    hitEOS = true
                                    break
                                }
                                emit(token)
                            }
                            commitVerificationCaches(candidateCaches)
                            layerCaches = candidateCaches
                            logits = lastTokenLogits(candidate.logits)
                            previousHidden = lastTokenHidden(candidateMTPHidden)
                            if hitEOS || generated.count >= tokenBudget {
                                break
                            }
                            continue
                        }

                        let acceptedPrefix = Array(draftTokens.prefix(accepted))
                        mtpAcceptedTokens += accepted
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

                        let committedTokenCount = accepted + 1
                        if restoreVerificationCaches(
                            candidateCaches,
                            totalTokens: draftTokens.count + 1,
                            tokenCount: committedTokenCount
                        ) {
                            let restored = mtpDraftSession.restoredVerificationState(
                                from: candidate,
                                acceptedTokens: acceptedPrefix
                            )
                            layerCaches = candidateCaches
                            logits = restored.logits
                            previousHidden = restored.hidden
                            continue
                        }

                        let replacementCaches = forkLayerCaches(layerCaches)
                        let replacementTokens = [next] + acceptedPrefix + [replacement]
                        let replacementInputValues = replacementTokens.map { Int32($0) }
                        let replacementInput = MLXArray(replacementInputValues)
                            .reshaped(1, acceptedPrefix.count + 2)
                        let replacementForward = model.forward(replacementInput, cache: replacementCaches)
                        let replacementMTPHidden = replacementForward.mtpHidden ?? replacementForward.hidden
                        MLX.eval(replacementForward.logits)
                        MLX.eval(replacementMTPHidden)
                        mtpReplacementPasses += 1
                        mtpDraftSession.recordCommittedTransitions(
                            hiddenStates: replacementMTPHidden,
                            nextTokens: acceptedPrefix + [replacement]
                        )
                        emit(replacement)
                        layerCaches = replacementCaches
                        logits = lastTokenLogits(replacementForward.logits)
                        previousHidden = lastTokenHidden(replacementMTPHidden)
                        continue
                    }

                    mtpNonDraftingRounds += 1
                    // Once the policy reaches zero, no more acceptance evidence
                    // can change it. Finish through the existing pipelined
                    // target decoder after consuming this emitted token.
                    usePipelinedFallback = supportsPipelinedFallback && pipelinedFallbackEnabled
                    mtpDraftSession.recordCommittedTransitions(
                        hiddenStates: hidden,
                        nextTokens: [next]
                    )
                } else {
                    let draftLogits = mtpModel.draftLogits(
                        token: next,
                        previousHidden: hidden,
                        positionOffset: positionOffset,
                        baseModel: model
                    )
                    MLX.eval(draftLogits)
                    mtpDraftedTokens += 1

                    let draftProbs = samplingProbabilities(
                        logits: draftLogits[0, -1, 0...],
                        config: generationConfig,
                        previousTokens: repetitionHistory
                    )
                    let draft = sampleToken(probabilities: draftProbs)

                    let candidateCaches = forkLayerCaches(layerCaches)
                    let candidateInput = MLXArray([Int32(next), Int32(draft)]).reshaped(1, 2)
                    let candidate = model.forward(
                        candidateInput,
                        cache: candidateCaches,
                        targetVerify: true
                    )
                    let candidateMTPHidden = candidate.mtpHidden ?? candidate.hidden
                    MLX.eval(candidate.logits)
                    MLX.eval(candidateMTPHidden)
                    mtpVerificationPasses += 1

                    let targetProbs = samplingProbabilities(
                        logits: candidate.logits[0, 0, 0...],
                        config: generationConfig,
                        previousTokens: repetitionHistory
                    )
                    let draftProb = max(draftProbs[draft].item(Float.self), Float.leastNonzeroMagnitude)
                    let targetProb = targetProbs[draft].item(Float.self)
                    let acceptProbability = min(1.0, targetProb / draftProb)

                    if Q35Sampling.acceptsDraft(probability: acceptProbability) {
                        mtpAcceptedTokens += 1
                        if eosSet.contains(draft) {
                            break
                        }
                        emit(draft)
                        commitVerificationCaches(candidateCaches)
                        layerCaches = candidateCaches
                        logits = lastTokenLogits(candidate.logits)
                        previousHidden = lastTokenHidden(candidateMTPHidden)
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
                    let replacementMTPHidden = replacementForward.mtpHidden ?? replacementForward.hidden
                    MLX.eval(replacementForward.logits)
                    MLX.eval(replacementMTPHidden)
                    mtpReplacementPasses += 1
                    emit(replacement)
                    layerCaches = replacementCaches
                    logits = lastTokenLogits(replacementForward.logits)
                    previousHidden = lastTokenHidden(replacementMTPHidden)
                    continue
                }
            }

            let serialStart = mtpProfile?.clock()
            let nextInput = MLXArray([Int32(next)]).reshaped(1, 1)
            let positionIds = decodePositionIds(layerCaches: layerCaches, tokenCount: 1, ropeDelta: mropeRopeDelta)
            if retainHidden {
                let output = model.forward(
                    nextInput,
                    cache: layerCaches,
                    positionIds: positionIds
                )
                let outputMTPHidden = output.mtpHidden ?? output.hidden
                logits = output.logits
                previousHidden = lastTokenHidden(outputMTPHidden)
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
            mtpProfile?.recordSerial(since: serialStart)
            if usePipelinedFallback {
                if !pendingProgressWhitespace.isEmpty {
                    progressHandler?(ChatProgress(stage: .generating, message: pendingProgressWhitespace))
                }
                let tail = try await decodeTokensPipelined(
                    model: model, tokenizerAndTemplate: tokenizerAndTemplate,
                    initialLogits: logits, layerCaches: layerCaches, eosSet: eosSet,
                    generationConfig: generationConfig, tokenBudget: tokenBudget - generated.count,
                    mropeRopeDelta: mropeRopeDelta, promptTokens: repetitionHistory,
                    stopAtCompletedToolCall: false, logprobCapture: logprobCapture,
                    logprobRegion: logprobRegion, progressHandler: progressHandler
                )
                generated.append(contentsOf: tail.generatedTokens)
                mtpNonDraftingRounds += tail.generatedTokens.count
                mtpProfile?.recordPipeline(seconds: tail.decodeSeconds, tokens: tail.generatedTokens.count)
                break
            }
        }

        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        if Gemma4DecodeTrace.enabled, mtpModel != nil {
            let acceptance = mtpDraftedTokens > 0
                ? Double(mtpAcceptedTokens) / Double(mtpDraftedTokens) * 100
                : 0
            Gemma4DecodeTrace.emit(String(
                format: "[q35-decode-trace] mode=mtp tokens=%d drafted=%d accepted=%d acceptance=%.1f%% verify=%d replacement=%d serial=%d wall=%.2fms/tok",
                generated.count,
                mtpDraftedTokens,
                mtpAcceptedTokens,
                acceptance,
                mtpVerificationPasses,
                mtpReplacementPasses,
                mtpNonDraftingRounds,
                decodeSeconds / Double(max(1, generated.count)) * 1000
            ))
        }
        return Q35BatchedDecodeResult(
            generatedTokens: generated,
            decodeSeconds: decodeSeconds,
            firstTokenSeconds: firstTokenSeconds,
            acceleration: mtpModel.map { _ in
                ChatAccelerationDiagnostics(
                    route: "mtp-speculative",
                    draftModel: mtpModel?.diagnosticsID ?? "qwen-mtp",
                    rounds: mtpVerificationPasses,
                    draftedTokens: mtpDraftedTokens,
                    acceptedDraftTokens: mtpAcceptedTokens,
                    draftHistoryTokens: draftHistoryTokens,
                    speculationProfile: mtpProfile?.result
                )
            } ?? ChatAccelerationDiagnostics(
                route: jsonConstrained ? "json-constrained-serial" : "serial"
            )
        )
    }

    private func decodeTokensPipelined(
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [Q35LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        mropeRopeDelta: Int?,
        promptTokens: [Int],
        stopAtCompletedToolCall: Bool,
        logprobCapture: ChatLogprobCapture,
        logprobRegion: ChatLogprobRegion,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35BatchedDecodeResult {
        let layerCaches = layerCaches
        let banMask = tokenBanMask(
            vocabularySize: initialLogits.dim(-1),
            dtype: initialLogits.dtype,
            tokens: generationConfig.bannedTokens
        )

        // Shared pipelined decode; mRoPE positions derive from the cache
        // offset, so the step closure computes them per forward.
        var toolCallCompletionDetector = Q35ToolParser.StreamingCompletionDetector()
        let result = try AutoregressiveDecodeEngine.decode(
            AutoregressiveDecodeRequest(
                initialLogits: initialLogits,
                generationConfig: generationConfig,
                eosTokens: eosSet,
                tokenBudget: tokenBudget,
                historySeedTokens: promptTokens,
                banMask: banMask,
                logprobCapture: logprobCapture,
                logprobRegion: logprobRegion
            ),
            stepForward: { token in
                if model.config.textConfig.isQwen4Exp {
                    clearMLXCacheUnderPressureIfNeeded()
                }
                return model(
                    token,
                    cache: layerCaches,
                    positionIds: decodePositionIds(layerCaches: layerCaches, tokenCount: 1, ropeDelta: mropeRopeDelta)
                )
            },
            decodeToken: { tokenizerAndTemplate.decode(token: $0) },
            emitPiece: { _, piece in
                progressHandler?(ChatProgress(stage: .generating, message: piece))
            },
            shouldContinue: { _, piece in
                guard stopAtCompletedToolCall else { return true }
                return !toolCallCompletionDetector.feed(piece)
            },
            checkCancellation: { try Task.checkCancellation() }
        )

        if Gemma4DecodeTrace.enabled, !result.generatedTokens.isEmpty {
            let count = Double(result.generatedTokens.count)
            Gemma4DecodeTrace.emit(String(
                format: "[q35-decode-trace] mode=pipelined temp=\(generationConfig.temperature) tokens=%d build=%.2fms/tok wait=%.2fms/tok wall=%.2fms/tok",
                result.generatedTokens.count,
                result.buildSeconds / count * 1000,
                result.waitSeconds / count * 1000,
                result.decodeSeconds / count * 1000
            ))
        }

        return Q35BatchedDecodeResult(
            generatedTokens: result.generatedTokens,
            decodeSeconds: result.decodeSeconds,
            firstTokenSeconds: result.firstTokenSeconds,
            logprobs: result.logprobs,
            acceleration: ChatAccelerationDiagnostics(route: "final-target-pipelined")
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
            if model.config.textConfig.isQwen4Exp {
                clearMLXCacheUnderPressureIfNeeded()
            }
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

        if Self.batchedGPUSamplingEnabled {
            // Sample every row on GPU (same sampler the serial pipelined
            // decode uses), then read the whole batch back in one sync —
            // the legacy path performed one blocking readback per row per
            // step, which scales linearly with serve concurrency.
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
                let tokenArray = sampledTokenArray(
                    logits: row.logits[0, -1, 0...],
                    config: row.generationConfig,
                    previousTokenIndices: row.repetitionHistoryGPU,
                    banMask: nil
                )
                row.repetitionHistoryGPU = appendingRepetitionHistory(
                    row.repetitionHistoryGPU,
                    token: tokenArray,
                    config: row.generationConfig
                )
                tokenArrays.append(tokenArray.reshaped(1))
            }
            let values = MLX.concatenated(tokenArrays, axis: 0).asArray(Int32.self)
            for (index, row) in sampledRows.enumerated() {
                let next = Int(values[index])
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
        } else {
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

    private func greedyTokenArray(from logits: MLXArray) -> MLXArray {
        argMax(logits[0, -1, 0...], axis: -1).asType(.int32)
    }

    func chunkedPrefill(
        model: Q35Model,
        promptTokens: [Int],
        cache: [Q35LayerCache?],
        startIndex: Int = 0,
        existingLogits: MLXArray? = nil,
        existingHidden: MLXArray? = nil,
        modelPath: String? = nil,
        checkpointTokenCounts: Set<Int> = [],
        retainHidden: Bool = true,
        retainMTPHistory: Bool = false,
        mtpSession: Q35MTPDraftSession? = nil,
        prefillMTPModel: (any Q35MTPDraftModel)? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35PrefillOutput {
        guard !promptTokens.isEmpty else {
            throw Q35Error.generationFailed("Prompt is empty after tokenization.")
        }

        var processed = startIndex
        var logits = existingLogits
        var hidden = existingHidden
        var mtpHistoryChunks: [MLXArray] = []
        if processed > 0, processed < promptTokens.count {
            progressHandler?(ChatProgress(stage: .encoding, message: "Reusing \(processed) prompt KV tokens"))
        }
        while processed < promptTokens.count {
            try Task.checkCancellation()
            let chunkSize = Self.prefillChunkSize(
                modelId: modelId,
                activeRequestCount: activeChatRequestCount
            )
            let end = RuntimePrefillCheckpointPlanner.nextEnd(
                processed: processed,
                total: promptTokens.count,
                chunkSize: chunkSize,
                checkpoints: checkpointTokenCounts
            )
            if promptTokens.count > chunkSize {
                progressHandler?(ChatProgress(stage: .encoding, message: "Prefilling \(end)/\(promptTokens.count) tokens"))
            }
            let chunk = MLXArray(promptTokens[processed..<end].map(Int32.init))
                .reshaped(1, end - processed)
            if retainHidden {
                let output = model.forwardPrefill(
                    chunk,
                    cache: cache,
                    retainAllHidden: retainMTPHistory
                )
                let outputMTPHidden = output.mtpHidden ?? output.hidden
                MLX.eval(output.logits)
                MLX.eval(outputMTPHidden)
                logits = output.logits
                if let mtpSession, let prefillMTPModel {
                    if let hidden, processed > 0 {
                        mtpSession.recordCommittedTransitions(
                            hiddenStates: hidden, nextTokens: [promptTokens[processed]]
                        )
                    }
                    mtpSession.recordCommittedTransitions(
                        hiddenStates: outputMTPHidden,
                        nextTokens: Array(promptTokens[(processed + 1)..<end])
                    )
                    mtpSession.primeCommittedHistory(mtpModel: prefillMTPModel, baseModel: model)
                } else if retainMTPHistory {
                    mtpHistoryChunks.append(outputMTPHidden)
                }
                hidden = lastTokenHidden(outputMTPHidden)
                let priority: RuntimePrefixCacheEntryPriority? = mtpSession != nil || model.config.textConfig.isQwen4Exp
                    ? RuntimePrefillCheckpointPlanner.storagePriority(
                        tokenCount: end, total: promptTokens.count,
                        semanticCheckpoints: checkpointTokenCounts
                    )
                    : (checkpointTokenCounts.contains(end) ? .semantic : .chunk)
                if let modelPath, let priority {
                    storePrefixKVCache(
                        modelPath: modelPath,
                        promptTokens: promptTokens,
                        tokenCount: end,
                        cache: cache,
                        logits: output.logits,
                        hidden: lastTokenHidden(outputMTPHidden),
                        mtpSession: mtpSession,
                        priority: priority
                    )
                }
            } else {
                let output = model.forwardPrefill(chunk, cache: cache)
                MLX.eval(output.logits)
                logits = output.logits
                hidden = nil
            }
            clearMLXCacheUnderPressureIfNeeded()
            processed = end
            await Task.yield()
        }

        guard let logits else {
            throw Q35Error.generationFailed("Prefill did not produce logits.")
        }
        let mtpHistoryHidden = mtpHistoryChunks.isEmpty
            ? nil
            : MLX.concatenated(mtpHistoryChunks, axis: 1)
        if let mtpHistoryHidden {
            MLX.eval(mtpHistoryHidden)
        }
        return Q35PrefillOutput(
            logits: logits,
            hidden: hidden,
            mtpHistoryHidden: mtpHistoryHidden,
            mtpSession: mtpSession
        )
    }

    func chunkedPrefillEmbeddings(
        model: Q35Model,
        inputIds: MLXArray,
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
            let chunkSize = Self.prefillChunkSize(
                modelId: modelId,
                activeRequestCount: activeChatRequestCount
            )
            let end = min(processed + chunkSize, tokenCount)
            if tokenCount > chunkSize {
                progressHandler?(ChatProgress(stage: .encoding, message: "Prefilling \(end)/\(tokenCount) tokens"))
            }
            let chunkEmbeddings = inputEmbeddings[0..., processed..<end, 0...]
            // Flash-Next PLE still consumes the original token IDs after
            // image embeddings replace the multimodal placeholder vectors.
            let chunkInput = inputIds[0..., processed..<end]
            let chunkPositionIds = positionIds?[0..., 0..., processed..<end]
            if retainHidden {
                let output = model.forwardPrefill(
                    chunkInput,
                    cache: cache,
                    inputEmbeddings: chunkEmbeddings,
                    positionIds: chunkPositionIds
                )
                let outputMTPHidden = output.mtpHidden ?? output.hidden
                MLX.eval(output.logits)
                MLX.eval(outputMTPHidden)
                logits = output.logits
                hidden = outputMTPHidden
            } else {
                let output = model.forwardPrefill(
                    chunkInput,
                    cache: cache,
                    inputEmbeddings: chunkEmbeddings,
                    positionIds: chunkPositionIds
                )
                MLX.eval(output.logits)
                logits = output.logits
                hidden = nil
            }
            clearMLXCacheUnderPressureIfNeeded()
            processed = end
            await Task.yield()
        }

        guard let logits else {
            throw Q35Error.generationFailed("Prefill did not produce logits.")
        }
        return Q35PrefillOutput(
            logits: logits,
            hidden: hidden,
            mtpHistoryHidden: nil,
            mtpSession: nil
        )
    }

    private func semanticPrefixCheckpoints(
        tokenizerAndTemplate: Q35TokenizerAndTemplate,
        messages: [ChatMessage],
        tools: [ToolDefinition]?,
        includeThinking: Bool,
        reasoningEffort: String?,
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
            reasoningEffort: reasoningEffort,
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

    func prefixKVCacheSeed(
        modelPath: String,
        promptTokens: [Int],
        cacheMode: RuntimeKVCacheMode,
        requiresMTPSession: Bool = false
    ) -> (tokenCount: Int, caches: [Q35LayerCache?], logits: MLXArray,
          hidden: MLXArray, mtpSession: Q35MTPDraftSession?)? {
        guard prefixKVCacheEnabled else { return nil }
        let matchingKey = prefixKVCache.keys
            .filter { key in
                key.modelPath == modelPath
                    && key.cacheMode == cacheMode
                    && key.tokens.count <= promptTokens.count
                    && promptTokens.starts(with: key.tokens)
                    && (!requiresMTPSession || prefixKVCache[key]?.mtpSession != nil)
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
            entry.hidden,
            entry.mtpSession?.fork()
        )
    }

    private func storePrefixKVCache(
        modelPath: String,
        promptTokens: [Int],
        tokenCount: Int,
        cache: [Q35LayerCache?],
        logits: MLXArray,
        hidden: MLXArray,
        mtpSession: Q35MTPDraftSession? = nil,
        priority: RuntimePrefixCacheEntryPriority
    ) {
        guard prefixKVCacheEnabled, tokenCount > 0 else { return }
        let key = Q35PrefixKVCacheKey(
            modelPath: modelPath,
            cacheMode: cacheMode(for: cache),
            tokens: Array(promptTokens.prefix(tokenCount))
        )
        prefixKVCache[key] = Q35PrefixKVCacheEntry(
            caches: forkLayerCaches(cache),
            logits: logits,
            hidden: hidden,
            mtpSession: mtpSession?.fork(),
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

    private func restoreVerificationCaches(
        _ caches: [Q35LayerCache?],
        totalTokens: Int,
        tokenCount: Int
    ) -> Bool {
        let presentCaches = caches.compactMap { $0 }
        guard presentCaches.allSatisfy({ cache in
            cache.canRestoreVerificationPrefix(
                totalTokens: totalTokens,
                tokenCount: tokenCount
            )
        }) else {
            return false
        }
        for cache in presentCaches {
            guard cache.restoreVerificationPrefix(
                totalTokens: totalTokens,
                tokenCount: tokenCount
            ) else {
                return false
            }
        }
        return true
    }

    private func commitVerificationCaches(_ caches: [Q35LayerCache?]) {
        for cache in caches.compactMap({ $0 }) {
            cache.commitVerification()
        }
    }

    private func loadTextWeights(
        into q35Model: Q35Model,
        from resources: Q35Resources,
        groupSize: Int,
        bits: Int
    ) throws {
        let checkpointUsesZeroCenteredNorms = try Self.checkpointUsesZeroCenteredRMSNorm(from: resources)
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            guard let mapped = Self.mapTextWeightKey(key) else { return [] }
            if q35Model.config.tieWordEmbeddings, mapped == "lm_head.weight" {
                return []
            }
            if let splitExperts = Self.splitMappedExpertGateUpWeight(mapped, value) {
                return splitExperts
            }
            let normalizedMapped = Self.normalizeMappedExpertWeightKey(mapped)
            if normalizedMapped.hasSuffix(".conv1d.weight"), value.ndim == 3 {
                return [(normalizedMapped, Self.normalizedLinearAttentionConv1DWeight(value))]
            }
            if Self.isOffsetRMSNormWeight(normalizedMapped) {
                return [(
                    normalizedMapped,
                    Self.normalizedRMSNormWeight(
                        value,
                        checkpointUsesZeroCenteredNorms: checkpointUsesZeroCenteredNorms
                    )
                )]
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
            if key.hasSuffix(".conv1d.weight"), value.ndim == 3 {
                return [(key, Self.normalizedLinearAttentionConv1DWeight(value))]
            }
            if Self.isOffsetRMSNormWeight(key) {
                return [(
                    key,
                    Self.normalizedRMSNormWeight(
                        value,
                        checkpointUsesZeroCenteredNorms: checkpointUsesZeroCenteredNorms
                    )
                )]
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

    private func loadQ38NGramEmbeddings(
        into model: Q35Model,
        from resources: Q35Resources,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws {
        let targets = model.q38NGramEmbeddings
        guard !targets.isEmpty else { return }

        let placement = try Q38PLEPlacement.resolve(
            rootURL: resources.rootURL,
            progressHandler: { message in
                progressHandler?(ChatProgress(stage: .loadingModel, message: message))
            }
        )

        for (pleLayerIndex, target) in targets.enumerated() {
            let layerIndex = model.config.textConfig.pleLayerIds[pleLayerIndex] - 1
            let base = "language_model.model.layers.\(layerIndex).ple.ple_embedding.ngram_embedding"
            let dimensions = model.config.textConfig.pleEmbeddingDimensions
                / ((model.config.textConfig.ngramSize - 1) * model.config.textConfig.headsPerNgram)
            target.installDiskTable(try Q38DiskNGramTable(
                indexURL: placement?.indexURL ?? resources.modelIndexURL,
                base: base,
                shardCount: model.config.textConfig.splitNgramParts,
                dimensions: dimensions,
                minimumRowCount: target.minimumRowCount
            ))
        }
    }

    private static func hasMTPWeights(resources: Q35Resources) -> Bool {
        if standaloneMTPWeightsURL(resources: resources) != nil {
            return true
        }
        guard FileManager.default.fileExists(atPath: resources.modelIndexURL.path),
              let data = try? Data(contentsOf: resources.modelIndexURL),
              let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data) else {
            return false
        }
        return index.weightMap.keys.contains { $0.hasPrefix("mtp.") }
    }

    static func mtpResources(
        primary resources: Q35Resources,
        companionRootURL: URL? = nil
    ) -> Q35Resources? {
        if hasMTPWeights(resources: resources) {
            return resources
        }
        let mounted = Q35Resources(
            rootURL: resources.rootURL.appendingPathComponent(
                Q35Resources.q38MTPComponentPath,
                isDirectory: true
            )
        )
        if hasMTPWeights(resources: mounted) {
            return mounted
        }
        if let companionRootURL {
            let companion = Q35Resources(rootURL: companionRootURL)
            if hasMTPWeights(resources: companion) {
                return companion
            }
        }
        return nil
    }

    private static func standaloneMTPWeightsURL(resources: Q35Resources) -> URL? {
        let explicit = resources.rootURL.appendingPathComponent("mtp.safetensors")
        if FileManager.default.fileExists(atPath: explicit.path) {
            return explicit
        }
        guard resources.rootURL.lastPathComponent == Q35Resources.q38MTPComponentPath,
              FileManager.default.fileExists(atPath: resources.modelWeightsURL.path) else {
            return nil
        }
        return resources.modelWeightsURL
    }

    private func loadQ38MTPWeights(
        into mtp: Q38MTPModel,
        from resources: Q35Resources,
        groupSize: Int,
        bits: Int
    ) throws {
        guard FileManager.default.fileExists(atPath: resources.modelIndexURL.path) else {
            throw Q35Error.missingFiles([resources.modelIndexURL.lastPathComponent])
        }
        let data = try Data(contentsOf: resources.modelIndexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
        let filenames = Self.embeddedMTPShardFilenames(weightMap: index.weightMap)
        guard !filenames.isEmpty else {
            throw Q35Error.missingFiles(["mtp.*"])
        }

        var arrays: [String: MLXArray] = [:]
        for filename in filenames {
            let shard = try SafetensorsStreamingLoader.loadArrays(
                url: resources.rootURL.appendingPathComponent(filename),
                where: { $0.hasPrefix("mtp.") }
            )
            arrays.merge(shard) { _, replacement in replacement }
        }

        let required = [
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.fc_embedding.weight",
            "mtp.fc_hidden.weight",
            "mtp.hyper_connection_mixer.hc_norm.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
        ]
        let missing = required.filter { arrays[$0] == nil }
        guard missing.isEmpty else {
            throw Q35Error.missingFiles(missing)
        }

        try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
            arrays,
            to: mtp,
            groupSize: groupSize,
            bits: bits,
            keyMapper: { key in
                Self.mapQ38MTPWeightKey(key) ?? "__unused__.\(key)"
            },
            mapper: { key, value in
                key.hasPrefix("__unused__.") ? [] : [(key, value)]
            }
        )
        Memory.clearCache()
    }

    static func mapQ38MTPWeightKey(_ key: String) -> String? {
        guard key.hasPrefix("mtp.") else { return nil }
        return String(key.dropFirst("mtp.".count))
    }

    private func loadMTPWeights(
        into mtp: Q35MTPModel,
        baseModel: Q35Model,
        from resources: Q35Resources,
        groupSize: Int,
        bits: Int
    ) throws {
        if let standalone = Self.standaloneMTPWeightsURL(resources: resources) {
            let metadata = try SafetensorsStreamingLoader.metadata(url: standalone)
            if metadata.keys.contains(where: { $0.hasSuffix(".scales") }) {
                let arrays = try MLX.loadArrays(url: standalone)
                if let weight = arrays["draft_lm_head.weight"],
                   let scales = arrays["draft_lm_head.scales"],
                   let biases = arrays["draft_lm_head.biases"] {
                    baseModel.installCoarseDraftHead(
                        weight: weight,
                        scales: scales,
                        biases: biases
                    )
                }
                try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                    arrays,
                    to: mtp,
                    groupSize: groupSize,
                    bits: bits,
                    keyMapper: { key in
                        Self.mapMTPWeightKey(key, standalone: true) ?? "__unused__.\(key)"
                    },
                    mapper: Self.mapMTPWeight
                )
            } else {
                try SafetensorsStreamingLoader.applyWeightsStreaming(
                    url: standalone,
                    to: mtp,
                    dtype: .bfloat16,
                    verify: .none,
                    include: { Self.mapMTPWeightKey($0, standalone: true) != nil },
                    mapper: { key, value in
                        guard let mapped = Self.mapMTPWeightKey(key, standalone: true) else { return [] }
                        return Self.mapMTPWeight(mapped, value)
                    },
                    batchSize: 32
                )
            }
            return
        }

        let data = try Data(contentsOf: resources.modelIndexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
        let shardFilenames = Self.embeddedMTPShardFilenames(weightMap: index.weightMap)
        let checkpointUsesZeroCenteredNorms = try Self.checkpointUsesZeroCenteredRMSNorm(
            from: resources
        )
        for filename in shardFilenames {
            let arrays = try SafetensorsStreamingLoader.loadArrays(
                url: resources.rootURL.appendingPathComponent(filename),
                where: { Self.mapMTPWeightKey($0) != nil },
                dtype: .bfloat16
            )
            let updates = try Self.mappedMTPUpdates(
                arrays: arrays,
                expertCount: mtp.layers.first?.mlp.experts?.expertCount ?? 0,
                checkpointUsesZeroCenteredNorms: checkpointUsesZeroCenteredNorms
            )
            try mtp.update(parameters: ModuleParameters.unflattened(updates), verify: .none)
            MLX.eval(updates.map(\.1))
            Memory.clearCache()
        }
    }

    static func embeddedMTPShardFilenames(weightMap: [String: String]) -> [String] {
        Array(Set(weightMap.compactMap { key, filename in
            key.hasPrefix("mtp.") ? filename : nil
        })).sorted()
    }

    static func mapMTPWeightKey(_ key: String, standalone: Bool = false) -> String? {
        if key.hasPrefix("mtp.") {
            return String(key.dropFirst("mtp.".count))
        }
        guard standalone else { return nil }
        let barePrefixes = [
            "fc.",
            "layers.",
            "norm.",
            "pre_fc_norm_embedding.",
            "pre_fc_norm_hidden.",
        ]
        return barePrefixes.contains(where: { key.hasPrefix($0) }) ? key : nil
    }

    static func isMTPRMSNormWeight(_ key: String) -> Bool {
        key == "norm.weight"
            || key == "pre_fc_norm_embedding.weight"
            || key == "pre_fc_norm_hidden.weight"
            || key.hasSuffix(".input_layernorm.weight")
            || key.hasSuffix(".post_attention_layernorm.weight")
            || key.hasSuffix(".self_attn.q_norm.weight")
            || key.hasSuffix(".self_attn.k_norm.weight")
    }

    private static func mapMTPWeight(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        guard !key.hasPrefix("__unused__.") else { return [] }
        if isMTPRMSNormWeight(key) {
            return [(key, value - MLXArray(1.0).asType(value.dtype))]
        }
        return [(key, value)]
    }

    static func mappedMTPUpdates(
        arrays: [String: MLXArray],
        expertCount: Int,
        checkpointUsesZeroCenteredNorms: Bool = false
    ) throws -> [(String, MLXArray)] {
        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(arrays.count)
        let individualExpertMarker = ".mlp.experts."

        for (key, value) in arrays {
            guard let mapped = mapMTPWeightKey(key) else { continue }
            if expertCount > 0,
               mapped.contains(individualExpertMarker),
               mapped.contains(".weight") {
                continue
            }
            if isMTPRMSNormWeight(mapped) {
                updates.append((
                    mapped,
                    normalizedRMSNormWeight(
                        value,
                        checkpointUsesZeroCenteredNorms: checkpointUsesZeroCenteredNorms
                    )
                ))
            } else {
                updates.append(contentsOf: mapMTPWeight(mapped, value))
            }
        }

        guard expertCount > 0 else { return updates }
        let prefix = "mtp.layers.0.mlp.experts"
        func required(_ expert: Int, _ projection: String) throws -> MLXArray {
            let key = "\(prefix).\(expert).\(projection).weight"
            guard let value = arrays[key] else {
                throw Q35Error.missingFiles([key])
            }
            return value
        }

        var gateUpExperts: [MLXArray] = []
        var downExperts: [MLXArray] = []
        gateUpExperts.reserveCapacity(expertCount)
        downExperts.reserveCapacity(expertCount)
        for expert in 0..<expertCount {
            let gate = try required(expert, "gate_proj")
            let up = try required(expert, "up_proj")
            gateUpExperts.append(MLX.concatenated([gate, up], axis: 0))
            downExperts.append(try required(expert, "down_proj"))
        }

        let gateUp = MLX.stacked(gateUpExperts, axis: 0)
        let down = MLX.stacked(downExperts, axis: 0)
        MLX.eval(gateUp, down)
        updates.append(("layers.0.mlp.experts.gate_up_proj", gateUp))
        updates.append(("layers.0.mlp.experts.down_proj", down))
        return updates
    }

    private static func mapTextWeightKey(_ key: String) -> String? {
        // PLE tables are mapped separately and never installed as MLX weights.
        if key.contains(".ple.ple_embedding.ngram_embedding.") { return nil }
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

    static func normalizedRMSNormWeight(
        _ value: MLXArray,
        checkpointUsesZeroCenteredNorms: Bool
    ) -> MLXArray {
        if checkpointUsesZeroCenteredNorms {
            return value
        }
        return value - MLXArray(1.0).asType(value.dtype)
    }

    static func checkpointUsesZeroCenteredRMSNorm(from resources: Q35Resources) throws -> Bool {
        if FileManager.default.fileExists(atPath: resources.modelIndexURL.path) {
            let data = try Data(contentsOf: resources.modelIndexURL)
            let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
            let weightKeys = Array(index.weightMap.keys)
            if checkpointUsesZeroCenteredRMSNorm(weightKeys: weightKeys, tensorShapes: [:]) {
                return true
            }
            guard let convEntry = index.weightMap.first(where: { key, _ in
                key.hasSuffix(".linear_attn.conv1d.weight")
            }) else {
                return false
            }
            let shardURL = resources.rootURL.appending(path: convEntry.value)
            let metadata = try SafetensorsStreamingLoader.metadata(url: shardURL)
            return checkpointUsesZeroCenteredRMSNorm(
                weightKeys: weightKeys,
                tensorShapes: metadata.mapValues(\.shape)
            )
        }

        let metadata = try SafetensorsStreamingLoader.metadata(url: resources.modelWeightsURL)
        return checkpointUsesZeroCenteredRMSNorm(
            weightKeys: Array(metadata.keys),
            tensorShapes: metadata.mapValues(\.shape)
        )
    }

    static func checkpointUsesZeroCenteredRMSNorm(
        weightKeys: [String],
        tensorShapes: [String: [Int]]
    ) -> Bool {
        if weightKeys.contains(where: { $0.hasPrefix("mtp.") || $0.contains(".mtp.") }) {
            return true
        }
        return tensorShapes.contains { key, shape in
            key.hasSuffix(".linear_attn.conv1d.weight")
                && shape.count == 3
                && shape.last != 1
        }
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

    private func makeLayerCaches(
        config: Q35Config,
        kvCacheMode: RuntimeKVCacheMode = .default
    ) -> [Q35LayerCache?] {
        let text = config.textConfig
        let mlpOnly = Set(text.mlpOnlyLayers)
        return (0..<text.numHiddenLayers).map { layerIndex in
            if mlpOnly.contains(layerIndex) {
                return nil
            }
            let layerType = layerIndex < text.layerTypes.count ? text.layerTypes[layerIndex] : "linear_attention"
            if layerType == "full_attention" {
                let attention: KVCache
                if kvCacheMode == .affine4 || kvCacheMode == .affine8 {
                    attention = AffineQuantizedKVCache(
                        groupSize: Self.affineKVGroupSize(headDimension: text.headDim),
                        bits: kvCacheMode == .affine4 ? 4 : 8,
                        step: 256
                    )
                } else {
                    attention = KVCacheSimple(step: 256)
                }
                return .full(text.isQwen4Exp ? Q38QSACache(attention: attention) : attention)
            }
            return .linear(Q35LinearCache())
        }
    }

    private func cacheMode(for caches: [Q35LayerCache?]) -> RuntimeKVCacheMode {
        caches.contains { entry in
            guard case .full(let cache)? = entry else { return false }
            let main = (cache as? Q38QSACache)?.attention ?? cache
            guard let affine = main as? AffineQuantizedKVCache else { return false }
            return affine.bitWidth == 4
        } ? .affine4 : caches.contains { entry in
            guard case .full(let cache)? = entry else { return false }
            return ((cache as? Q38QSACache)?.attention ?? cache) is AffineQuantizedKVCache
        } ? .affine8 : .default
    }

    private static func affineKVGroupSize(headDimension: Int) -> Int {
        [64, 32, 16, 8].first { headDimension % $0 == 0 } ?? 1
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
        guard let loadedVisionResources else { throw Q35Error.modelNotLoaded }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family vision tower"))
        try visionTower.loadWeights(from: loadedVisionResources)
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loaded Qwen-family vision tower"))
    }

    private func buildVisionReplacements(
        imageURLs: [String],
        visionTower: Q35VisionTower,
        maximumTokensPerImage: Int
    ) throws -> [Q35VisionReplacement] {
        var replacements: [Q35VisionReplacement] = []
        replacements.reserveCapacity(imageURLs.count)

        for imageURL in imageURLs {
            let prepared = try loadImageTensor(
                from: imageURL,
                patchSize: visionTower.patchSize,
                spatialMergeSize: visionTower.spatialMergeSize,
                maximumVisionTokens: maximumTokensPerImage
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
        spatialMergeSize: Int,
        maximumVisionTokens: Int
    ) throws -> (tensor: MLXArray, gridTHW: (Int, Int, Int)) {
        let image = try loadImage(from: imageRef)
        let contextMaximumPixels = Self.visionPixelLimit(
            tokenLimit: maximumVisionTokens,
            patchSize: patchSize,
            spatialMergeSize: spatialMergeSize,
            configuredMaximum: visionMaxPixels
        )
        let target = try Self.qwen3VLTargetSize(
            originalWidth: image.width,
            originalHeight: image.height,
            patchSize: patchSize,
            spatialMergeSize: spatialMergeSize,
            minPixels: min(visionMinPixels, contextMaximumPixels),
            maxPixels: contextMaximumPixels
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
