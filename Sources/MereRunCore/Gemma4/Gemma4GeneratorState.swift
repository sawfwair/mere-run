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

struct Gemma4PrefixKVCacheKey: Hashable {
    let modelPath: String
    let quantization: Gemma4KVCacheQuantization
    let tokens: [Int]
}

struct Gemma4PrefixKVCacheEntry {
    let caches: [Gemma4AttentionCache]
    let logits: MLXArray
    let priority: RuntimePrefixCacheEntryPriority
    var lastAccess: Date
}

struct Gemma4BatchedDecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    let mtpStats: Gemma4MTPStats?
}

struct Gemma4PrefillResult {
    let logits: MLXArray
    let hidden: MLXArray?
    let sharedKVStates: [String: Gemma4SharedKVState]
}

final class Gemma4BatchedDecodeRow: @unchecked Sendable {
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
