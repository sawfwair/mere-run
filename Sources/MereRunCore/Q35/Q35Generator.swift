import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

public actor Q35Generator: ChatGenerator {
    static let contendedPrefillChunkSize = 512
    static let q38LowPrefillHeadroomBytes = UInt64(16) * 1_024 * 1_024 * 1_024
    static let minimumReclaimableCacheBytes = 256 * 1_024 * 1_024
    static let maximumReusableCacheBytes = 4 * 1_024 * 1_024 * 1_024
    static let prefixKVCacheMaxEntries = 4
    static let defaultMTPBlockSize = 4
    /// One committed token plus up to seven Qwen3.8 proposal tokens. Wider
    /// target verification is split only at SDPA, preserving one weight pass.
    static let q38MTPBlockSize = 8

    /// Continuous-batching decode samples every row on GPU (the same sampler
    /// the serial pipelined path uses) and reads the whole batch back in one
    /// sync per step; the legacy path performed one blocking readback per row
    /// per step. MERERUN_Q35_BATCHED_GPU_SAMPLING=0 restores per-row host
    /// sampling.
    static let batchedGPUSamplingEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_Q35_BATCHED_GPU_SAMPLING"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }()
    public static let qwen3VLMinPixels = 2_048
    public static let qwen3VLMaxPixels = 16_777_216

    var model: Q35Model?
    var tokenizerAndTemplate: Q35TokenizerAndTemplate?
    var visionTower: Q35VisionTower?
    var mtpModel: (any Q35MTPDraftModel)?
    var loadedModelPath: String?
    var loadedConfig: Q35Config?
    var loadedGenerationEOSTokenIds: [Int] = []
    var loadedResources: Q35Resources?
    var loadedVisionResources: Q35Resources?

    #if DEBUG
    var checkpointTransformForTesting: (@Sendable (Q35Model) throws -> Void)?

    /// Installed-checkpoint experiments run before fusion releases source weights.
    /// This hook is absent from production builds and never changes source files.
    func setCheckpointTransformForTesting(_ transform: @escaping @Sendable (Q35Model) throws -> Void) {
        precondition(model == nil, "Install checkpoint transforms before loading the model")
        checkpointTransformForTesting = transform
    }

    #endif
    let modelId: String
    let prefixKVCacheEnabled: Bool
    let continuousBatchingEnabled: Bool
    let visionMinPixels: Int
    let visionMaxPixels: Int

    var prefixKVCache: [Q35PrefixKVCacheKey: Q35PrefixKVCacheEntry] = [:]
    var prefixKVCacheHits = 0
    var prefixKVCacheMisses = 0
    var prefixKVCacheStores = 0
    var prefixKVCacheReusedTokens = 0

    var decodeQueue: [Q35BatchedDecodeRow] = []
    var activeDecodeRows: [Q35BatchedDecodeRow] = []
    var decodeLoopRunning = false
    var activeChatRequestCount = 0
    private var availableStreamContexts: [MLX.Stream.Context] = []
    var batchedDecodeSteps = 0
    var samePositionBatchedSteps = 0
    var variablePositionBatchedSteps = 0
    var singleDecodeSteps = 0
    var totalBatchedRows = 0
    var maxObservedBatchSize = 0

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
    /// Lease a context until the request ends. Actor reentrancy can admit another
    /// request while this one is suspended, so active requests need distinct
    /// contexts. Completed requests reuse them instead of accumulating MLX
    /// backend streams, which live until process exit.
    func withRequestStream<Result>(
        _ operation: () async throws -> Result
    ) async rethrows -> Result {
        let context = availableStreamContexts.popLast() ?? MLX.Stream.Context()
        defer {
            context.synchronize()
            clearMLXCacheUnderPressureIfNeeded()
            Q35MemoryTrace.record(modelID: modelId)
            availableStreamContexts.append(context)
        }
        return try await Q35CompiledOperations.withDefaultStream(
            context,
            scoped: Q35RuntimeTuning.isEnabled(.scopedCompilation, modelID: modelId),
            operation
        )
    }

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        activeChatRequestCount += 1
        defer { activeChatRequestCount = max(0, activeChatRequestCount - 1) }
        return try await withRequestStream {
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
        return try await withRequestStream {
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
        try await withRequestStream {
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
}
