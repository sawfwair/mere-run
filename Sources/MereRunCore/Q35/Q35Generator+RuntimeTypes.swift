import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

public typealias Q35ContinuousBatchingStats = RuntimeDecodeBatchingStats

struct Q35PrefixKVCacheKey: Hashable {
    let modelPath: String
    let cacheMode: RuntimeKVCacheMode
    let tokens: [Int]
}

struct Q35PrefixKVCacheEntry {
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

struct Q35BatchedDecodeResult {
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

struct Q35VisionReplacement {
    let embeddings: MLXArray
    let gridTHW: (Int, Int, Int)
}

struct Q35MRoPEPositionData {
    let positionIds: MLXArray
    let ropeDelta: Int
}

final class Q35BatchedDecodeRow: @unchecked Sendable {
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
