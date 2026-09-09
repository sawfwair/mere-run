import Foundation
import MLX
import MLXNN

public actor Gemma4Generator: ChatGenerator {
    /// Draft-model-free speculation for greedy decode when no MTP assistant
    /// is loaded: drafts are the continuation of the most recent earlier
    /// occurrence of the current token suffix in the context, verified in
    /// bursts inside the pipelined loop. A no-match token costs one host-side
    /// scan (microseconds) and no GPU work, so this is enabled by default;
    /// MERERUN_GEMMA4_PROMPT_LOOKUP=0 disables.
    static let promptLookupSpeculationEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_GEMMA4_PROMPT_LOOKUP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }()

    /// Draft length for prompt-lookup speculation
    /// (MERERUN_GEMMA4_PROMPT_LOOKUP_BLOCK, default 8). Lookup drafts cost
    /// nothing to produce, so they run longer than assistant-model blocks.
    static let promptLookupBlockSize: Int = {
        if let raw = ProcessInfo.processInfo.environment["MERERUN_GEMMA4_PROMPT_LOOKUP_BLOCK"],
           let value = Int(raw), value >= 1, value <= 64 {
            return value
        }
        return 8
    }()

    /// Finds the most recent earlier occurrence of the context's trailing
    /// 3-gram (falling back to 2-gram) and proposes the tokens that followed
    /// it. Host-side scan over Ints — microseconds at chat context lengths.
    static func promptLookupDraft(
        context: [Int],
        blockSize: Int,
        maxMatchLength: Int = 3,
        minMatchLength: Int = 2
    ) -> [Int] {
        guard blockSize >= 1, context.count >= minMatchLength + 2 else { return [] }
        for matchLength in stride(from: maxMatchLength, through: minMatchLength, by: -1) {
            guard context.count > matchLength else { continue }
            let suffixStart = context.count - matchLength
            var start = suffixStart - 1
            while start >= 0 {
                var matched = true
                for offset in 0..<matchLength where context[start + offset] != context[suffixStart + offset] {
                    matched = false
                    break
                }
                if matched {
                    let followStart = start + matchLength
                    let followEnd = min(followStart + blockSize, context.count)
                    if followStart < followEnd {
                        return Array(context[followStart..<followEnd])
                    }
                    return []
                }
                start -= 1
            }
        }
        return []
    }

    static let prefillChunkSize = 512
    static let prefixKVCacheMaxEntries = 4

    var model: (any Gemma4CausalModel)?
    var mtpModel: Gemma4AssistantDraftModel?
    var loadedMTPModelPath: String?
    var tokenizerAndTemplate: Gemma4TokenizerAndTemplate?
    var loadedModelPath: String?
    var loadedTextLoRASignature: String?
    var loadedConfig: Gemma4Config?
    var lastMTPStats = Gemma4MTPStats()

    let modelId: String
    let kvCacheQuantization: Gemma4KVCacheQuantization
    let prefixKVCacheEnabled: Bool
    let continuousBatchingEnabled: Bool

    var prefixKVCache: [Gemma4PrefixKVCacheKey: Gemma4PrefixKVCacheEntry] = [:]
    var prefixKVCacheHits = 0
    var prefixKVCacheMisses = 0
    var prefixKVCacheStores = 0
    var prefixKVCacheReusedTokens = 0

    var decodeQueue: [Gemma4BatchedDecodeRow] = []
    var activeDecodeRows: [Gemma4BatchedDecodeRow] = []
    var decodeLoopRunning = false
    var batchedDecodeSteps = 0
    var samePositionBatchedSteps = 0
    var variablePositionBatchedSteps = 0
    var singleDecodeSteps = 0
    var totalBatchedRows = 0
    var maxObservedBatchSize = 0

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
        try await Stream.withNewDefaultStream {
            let rootURL = try await resolveModelRoot(
                modelPath: modelPath,
                progressHandler: progressHandler
            )
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

    func resetLoadedModel() {
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
}
