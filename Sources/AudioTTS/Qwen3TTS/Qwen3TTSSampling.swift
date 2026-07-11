import Foundation
import MLX

/// Environment gates for the Qwen3 TTS talker decode path.
enum Qwen3TTSEnvironment {
    /// GPU-side sampling plus a depth-1 pipelined talker loop (default on).
    /// The legacy loop performs roughly nine GPU→CPU round trips per emitted
    /// frame (one per sampled code) and re-uploads the ~1k-entry suppress
    /// list and the host-side repetition history every token.
    /// MERERUN_TTS_PIPELINED_DECODE=0 restores the legacy synchronous loop.
    static let pipelinedDecodeEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_TTS_PIPELINED_DECODE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }()
}

/// Generation-constant sampling state for the talker loop, kept on GPU so a
/// sampling step schedules pure GPU work and returns a `[1, 1]` token array
/// without a host readback. Matches `sampleToken`'s semantics exactly:
/// suppress bias and repetition penalty apply before the greedy early-out,
/// and the EOS logit is exempt from top-k/top-p truncation.
struct Qwen3TTSSamplerContext {
    let temperature: Float
    let topK: Int
    let topP: Float
    let repetitionPenalty: Float
    /// `[vocab]` additive bias: `-inf` at suppressed ids, `0` elsewhere.
    let suppressBias: MLXArray?
    /// `[1]` int32 index of the EOS token, pre-uploaded.
    let eosIndex: MLXArray?
    /// Growing `[n]` int32 vector of previously sampled first tokens. The
    /// pipelined loop appends the sampled token as an array, so the penalty
    /// for step N+1 sees token N without ever reading it back to the host.
    private(set) var history: MLXArray?

    init(
        vocabSize: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float,
        eosTokenId: Int?,
        suppressTokens: [Int]?
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        if let suppressTokens, !suppressTokens.isEmpty {
            let bias = MLXArray.zeros([vocabSize], dtype: .float32)
            bias[MLXArray(suppressTokens.map { Int32($0) })] = MLXArray(
                Array(repeating: -Float.infinity, count: suppressTokens.count)
            )
            self.suppressBias = bias
        } else {
            self.suppressBias = nil
        }
        if let eosTokenId, eosTokenId < vocabSize {
            self.eosIndex = MLXArray([Int32(eosTokenId)])
        } else {
            self.eosIndex = nil
        }
        self.history = nil
    }

    mutating func appendHistory(_ token: MLXArray) {
        let flat = token.reshaped(1).asType(.int32)
        if let history {
            self.history = MLX.concatenated([history, flat], axis: 0)
        } else {
            self.history = flat
        }
    }
}

/// GPU-side equivalent of `sampleToken`: same masking order, same greedy
/// early-out, same EOS exemption from top-k/top-p, but every step stays on
/// GPU and the result is a `[1, 1]` int32 array.
func sampleTokenArrayTTS(
    logits: MLXArray,
    context: Qwen3TTSSamplerContext
) -> MLXArray {
    let lastLogits = logits[0..., (logits.dim(1) - 1), 0...]
    var scores = lastLogits.squeezed(axis: 0)
    if scores.dtype != .float32 {
        scores = scores.asType(.float32)
    }

    if let bias = context.suppressBias {
        scores = scores + bias
    }

    if let history = context.history, context.repetitionPenalty != 1.0 {
        let membership = MLXArray.zeros([scores.dim(0)], dtype: .float32)
        membership[history] = MLXArray.ones([history.dim(0)], dtype: .float32)
        let inHistory = membership .> 0
        let penalized = MLX.where(
            scores .< 0,
            scores * context.repetitionPenalty,
            scores / context.repetitionPenalty
        )
        scores = MLX.where(inHistory, penalized, scores)
    }

    if context.temperature <= 0 {
        return argMax(scores, axis: -1).asType(.int32).reshaped(1, 1)
    }

    scores = scores / context.temperature
    var eosLogit: MLXArray?
    if let eosIndex = context.eosIndex {
        eosLogit = scores.take(eosIndex, axis: -1)
    }

    let vocab = scores.dim(0)
    if context.topK > 0 && context.topK < vocab {
        // k-th largest value as the cutoff — identical threshold to the
        // legacy full argSort, without sorting the whole vocabulary.
        let kth = vocab - context.topK
        let partition = argPartition(scores, kth: kth, axis: -1)
        let threshold = scores.take(partition[kth..<(kth + 1)], axis: -1)
        scores = MLX.where(scores .< threshold, MLXArray(-Float.infinity), scores)
    }

    if context.topP < 1.0 {
        let probs = softmax(scores, axis: -1)
        let sortedIndices = argSort(probs, axis: -1)
        let sortedProbs = probs.take(sortedIndices, axis: -1)
        let cumulative = cumsum(sortedProbs, axis: -1)
        let cutoffMask = cumulative .> (1.0 - context.topP)
        let shifted = MLX.concatenated([
            MLX.zeros([1], dtype: .bool),
            cutoffMask[0..<(cutoffMask.dim(0) - 1)],
        ], axis: -1)
        let cutoffIndex = argMax(shifted.asType(.int32), axis: -1)
        let threshold = sortedProbs.take(cutoffIndex.reshaped(1), axis: -1)
        scores = MLX.where(probs .< threshold, MLXArray(-Float.infinity), scores)
    }

    if let eosIndex = context.eosIndex, let eosLogit {
        scores[eosIndex] = eosLogit.reshaped(1)
    }

    return categorical(scores).asType(.int32).reshaped(1, 1)
}
