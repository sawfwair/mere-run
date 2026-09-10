import Foundation
import MLX

struct LagunaDecodeResult {
    let generatedTokens: [Int]
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    var logprobs: ChatLogprobDiagnostics? = nil
    var acceleration: ChatAccelerationDiagnostics? = nil
}

struct LagunaPrefillResult {
    let logits: MLXArray
    let dflashCache: [Gemma4AttentionCache]?
}

final class LagunaBatchedDecodeRow: @unchecked Sendable {
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

final class LagunaDFlashBatchedDecodeRow: @unchecked Sendable {
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

struct LagunaDFlashRecovery {
    let row: LagunaDFlashBatchedDecodeRow
    let candidateHiddenStates: [Int: MLXArray]
    let committedCandidateTokenCount: Int
    let replacement: Int
    let recoveryCache: [Gemma4AttentionCache]
}
