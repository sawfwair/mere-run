// Adapted from mlx-swift-lm MLXLMCommon/Evaluate.swift

import Foundation
import MLX
import MLXRandom

public struct GenerationConfig: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var repetitionPenalty: Float?
    public var repetitionContextSize: Int
    /// Token ids that must never be sampled. Applied as a -inf logit mask.
    public var bannedTokens: [Int]

    public init(
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topK: Int = 0,
        topP: Float = 0.9,
        repetitionPenalty: Float? = 1.05,
        repetitionContextSize: Int = 20,
        bannedTokens: [Int] = []
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.bannedTokens = bannedTokens
    }
}

public func applyTokenBan(logits: MLXArray, tokens: [Int]) -> MLXArray {
    guard !tokens.isEmpty else { return logits }
    let vocabularySize = logits.dim(-1)
    let valid = tokens.filter { $0 >= 0 && $0 < vocabularySize }
    guard !valid.isEmpty else { return logits }
    // Build the ban as an additive mask on a fresh array — MLXArray is a reference
    // type, so writing into `logits` directly would mutate the caller's tensor.
    let indices = MLXArray(valid.map(Int32.init))
    let mask = MLXArray.zeros(like: logits)
    mask[indices] = MLXArray(Array(repeating: -Float.infinity, count: valid.count)).asType(logits.dtype)
    return logits + mask
}

/// Precomputes the additive token-ban mask so per-token sampling reuses one
/// materialized array instead of rebuilding a vocab-sized scatter every step.
public func tokenBanMask(vocabularySize: Int, dtype: DType, tokens: [Int]) -> MLXArray? {
    let valid = tokens.filter { $0 >= 0 && $0 < vocabularySize }
    guard !valid.isEmpty else { return nil }
    let indices = MLXArray(valid.map(Int32.init))
    let mask = MLXArray.zeros([vocabularySize], dtype: dtype)
    mask[indices] = MLXArray(Array(repeating: -Float.infinity, count: valid.count)).asType(dtype)
    return mask
}

/// Seeds the repetition-penalty history for pipelined decode loops that keep
/// the window on the GPU instead of re-uploading previousTokens each step.
public func repetitionHistoryArray(
    promptTokens: [Int],
    config: GenerationConfig
) -> MLXArray? {
    guard config.repetitionPenalty != nil,
          config.repetitionContextSize > 0,
          !promptTokens.isEmpty else {
        return nil
    }
    return MLXArray(promptTokens.suffix(config.repetitionContextSize).map { Int32($0) })
}

/// Appends a still-on-GPU token to the repetition window, trimming to
/// `repetitionContextSize`.
public func appendingRepetitionHistory(
    _ history: MLXArray?,
    token: MLXArray,
    config: GenerationConfig
) -> MLXArray? {
    guard config.repetitionPenalty != nil, config.repetitionContextSize > 0 else {
        return nil
    }

    let nextToken = token.asType(.int32).reshaped(1)
    guard let history else {
        return nextToken
    }

    let combined = concatenated([history, nextToken], axis: 0)
    let overflow = combined.dim(0) - config.repetitionContextSize
    guard overflow > 0 else {
        return combined
    }
    return combined[overflow..<combined.dim(0)]
}

enum SamplerPolicy {
    /// Top-p sampling prefilters to this many top-logit candidates via
    /// argPartition before the softmax/sort/cumsum chain, replacing a
    /// full-vocabulary sort (262k for Gemma) with a tiny one. The top-p
    /// nucleus at practical temperatures lives entirely inside the top few
    /// dozen tokens, so the truncation is a no-op in practice; it only bends
    /// the distribution when the nucleus would exceed this many candidates.
    /// MERERUN_SAMPLER_TOP_P_PREFILTER overrides; 0 disables (exact full sort).
    static let topPPrefilter: Int = {
        if let raw = ProcessInfo.processInfo.environment["MERERUN_SAMPLER_TOP_P_PREFILTER"],
           let value = Int(raw), value >= 0 {
            return value
        }
        return 256
    }()
}

/// Samples a token entirely on the GPU and returns it as a 0-d array, so decode
/// loops can pipeline the readback instead of blocking on `.item()` per token.
/// Matches `sampleToken(logits:config:previousTokens:)` distribution semantics;
/// intentional differences: top-k thresholding happens after the float32
/// upcast rather than in bfloat16, and top-p runs on an argPartition prefilter
/// of the top logits (see `SamplerPolicy.topPPrefilter`).
public func sampledTokenArray(
    logits: MLXArray,
    config: GenerationConfig,
    previousTokenIndices: MLXArray?,
    banMask: MLXArray?
) -> MLXArray {
    var logits = logits
    if let banMask {
        logits = logits + banMask
    }

    if let penalty = config.repetitionPenalty,
       let previousTokenIndices,
       previousTokenIndices.dim(0) > 0 {
        logits = applyRepetitionPenalty(
            logits: logits,
            tokenIndices: previousTokenIndices,
            penalty: penalty
        )
    }

    if config.temperature == 0 {
        return argMax(logits, axis: -1).asType(.int32)
    }

    let useTopK = config.topK > 0 && config.topK < logits.dim(-1)
    let useTopP = config.topP > 0 && config.topP < 1

    if useTopK || useTopP, logits.dtype == .bfloat16 {
        logits = logits.asType(.float32)
    }

    if useTopK {
        let sortedIndices = argSort(logits, axis: -1)
        let sortedLogits = logits.take(sortedIndices, axis: -1)
        let threshold = sortedLogits[sortedLogits.dim(-1) - config.topK]
        logits = MLX.where(logits .< threshold, MLXArray(-Float.infinity), logits)
    }

    if useTopP {
        // Restrict the top-p math to the highest-logit candidates. Skipped
        // when an explicit top-k already thresholded the logits (masked
        // entries would defeat the partition) or when disabled.
        var candidateIndices: MLXArray?
        var workingLogits = logits
        let prefilter = SamplerPolicy.topPPrefilter
        if !useTopK, prefilter > 0, logits.dim(-1) > prefilter * 4 {
            let vocabulary = logits.dim(-1)
            let partitioned = argPartition(logits, kth: vocabulary - prefilter, axis: -1)
            let top = partitioned[(vocabulary - prefilter)...]
            candidateIndices = top
            workingLogits = logits[top]
        }

        let probs = softmax(workingLogits / config.temperature, axis: -1)
        let sortedIndices = argSort(probs, axis: -1)
        let sortedProbs = probs.take(sortedIndices, axis: -1)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)
        let topProbs = MLX.where(
            cumulativeProbs .> (1 - config.topP),
            sortedProbs,
            MLXArray.zeros(like: sortedProbs)
        )
        let sortedToken = categorical(MLX.log(topProbs + 1e-10))
        let chosen = sortedIndices[sortedToken]
        if let candidateIndices {
            return candidateIndices[chosen].asType(.int32)
        }
        return chosen.asType(.int32)
    }

    return categorical(logits / config.temperature).asType(.int32)
}

public func argMaxSample(logits: MLXArray) -> Int {
    argMax(logits, axis: -1).item(Int.self)
}

public func greedySampleTokenArray(
    logits: MLXArray,
    config: GenerationConfig,
    previousTokens: [Int]
) -> MLXArray {
    let contextTokens = previousTokens.isEmpty || config.repetitionPenalty == nil
        ? nil
        : MLXArray(previousTokens.suffix(config.repetitionContextSize).map { Int32($0) })
    return greedySampleTokenArray(
        logits: logits,
        config: config,
        previousTokenIndices: contextTokens
    )
}

public func greedySampleTokenArray(
    logits: MLXArray,
    config: GenerationConfig,
    previousTokenIndices: MLXArray?
) -> MLXArray {
    var logits = logits

    logits = applyTokenBan(logits: logits, tokens: config.bannedTokens)

    if let penalty = config.repetitionPenalty,
       let previousTokenIndices,
       previousTokenIndices.dim(0) > 0 {
        logits = applyRepetitionPenalty(
            logits: logits,
            tokenIndices: previousTokenIndices,
            penalty: penalty
        )
    }

    return argMax(logits, axis: -1).asType(.int32)
}

public func topPSample(logits: MLXArray, temperature: Float, topP: Float) -> Int {
    var logits = logits
    if logits.dtype == .bfloat16 {
        logits = logits.asType(.float32)
    }

    let probs = softmax(logits / temperature, axis: -1)
    let sortedIndices = argSort(probs, axis: -1)
    let sortedProbs = probs.take(sortedIndices, axis: -1)
    let cumulativeProbs = cumsum(sortedProbs, axis: -1)

    let topProbs = MLX.where(
        cumulativeProbs .> (1 - topP),
        sortedProbs,
        MLXArray.zeros(like: sortedProbs)
    )

    let sortedToken = categorical(MLX.log(topProbs + 1e-10))
    return sortedIndices[sortedToken].item(Int.self)
}

public func categoricalSample(logits: MLXArray, temperature: Float) -> Int {
    categorical(logits / temperature).item(Int.self)
}

public func topKSample(logits: MLXArray, temperature: Float, topK: Int) -> Int {
    var logits = logits
    if logits.dtype == .bfloat16 {
        logits = logits.asType(.float32)
    }

    if topK > 0 && topK < logits.dim(-1) {
        let sortedIndices = argSort(logits, axis: -1)
        let sortedLogits = logits.take(sortedIndices, axis: -1)
        let threshold = sortedLogits[sortedLogits.dim(-1) - topK]
        logits = MLX.where(logits .< threshold, MLXArray(-Float.infinity), logits)
    }

    let probs = categorical(logits / temperature)
    return probs.item(Int.self)
}

public func applyRepetitionPenalty(
    logits: MLXArray,
    tokens: [Int],
    penalty: Float
) -> MLXArray {
    guard !tokens.isEmpty, penalty != 1.0 else { return logits }

    let indices = MLXArray(tokens.map { Int32($0) })
    let selectedLogits = logits[indices]

    let penalized = MLX.where(
        selectedLogits .< 0,
        selectedLogits * penalty,
        selectedLogits / penalty
    )

    let result = logits
    result[indices] = penalized
    return result
}

public func applyRepetitionPenalty(
    logits: MLXArray,
    tokenIndices: MLXArray,
    penalty: Float
) -> MLXArray {
    guard tokenIndices.dim(0) > 0, penalty != 1.0 else { return logits }

    let selectedLogits = logits[tokenIndices]

    let penalized = MLX.where(
        selectedLogits .< 0,
        selectedLogits * penalty,
        selectedLogits / penalty
    )

    let result = logits
    result[tokenIndices] = penalized
    return result
}

public func sampleToken(
    logits: MLXArray,
    config: GenerationConfig,
    previousTokens: [Int]
) -> Int {
    var logits = logits

    logits = applyTokenBan(logits: logits, tokens: config.bannedTokens)

    if let penalty = config.repetitionPenalty, !previousTokens.isEmpty {
        let contextTokens = Array(previousTokens.suffix(config.repetitionContextSize))
        logits = applyRepetitionPenalty(logits: logits, tokens: contextTokens, penalty: penalty)
    }

    if config.temperature == 0 {
        return argMaxSample(logits: logits)
    }

    let useTopK = config.topK > 0
    let useTopP = config.topP > 0 && config.topP < 1

    if useTopK {
        let sortedIndices = argSort(logits, axis: -1)
        let sortedLogits = logits.take(sortedIndices, axis: -1)

        if useTopP {
            if useTopK && config.topK < sortedLogits.dim(-1) {
                let threshold = sortedLogits[sortedLogits.dim(-1) - config.topK]
                logits = MLX.where(logits .< threshold, MLXArray(-Float.infinity), logits)
            }
            return topPSample(logits: logits, temperature: config.temperature, topP: config.topP)
        } else {
            return topKSample(logits: logits, temperature: config.temperature, topK: config.topK)
        }
    } else if useTopP {
        return topPSample(logits: logits, temperature: config.temperature, topP: config.topP)
    } else {
        return categoricalSample(logits: logits, temperature: config.temperature)
    }
}

public func samplingProbabilities(
    logits: MLXArray,
    config: GenerationConfig,
    previousTokens: [Int]
) -> MLXArray {
    var logits = logits

    logits = applyTokenBan(logits: logits, tokens: config.bannedTokens)

    if let penalty = config.repetitionPenalty, !previousTokens.isEmpty {
        let contextTokens = Array(previousTokens.suffix(config.repetitionContextSize))
        logits = applyRepetitionPenalty(logits: logits, tokens: contextTokens, penalty: penalty)
    }

    if logits.dtype == .bfloat16 {
        logits = logits.asType(.float32)
    }

    if config.topK > 0 && config.topK < logits.dim(-1) {
        let sortedIndices = argSort(logits, axis: -1)
        let sortedLogits = logits.take(sortedIndices, axis: -1)
        let threshold = sortedLogits[sortedLogits.dim(-1) - config.topK]
        logits = MLX.where(logits .< threshold, MLXArray(-Float.infinity), logits)
    }

    if config.temperature == 0 {
        let token = argMaxSample(logits: logits)
        let result = MLXArray.zeros(like: logits)
        result[token] = MLXArray(1.0)
        return result
    }

    var probs = softmax(logits / config.temperature, axis: -1)
    if config.topP > 0 && config.topP < 1 {
        let sortedIndices = argSort(probs, axis: -1)
        let sortedProbs = probs.take(sortedIndices, axis: -1)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)
        let kept = MLX.where(
            cumulativeProbs .> (1 - config.topP),
            sortedProbs,
            MLXArray(Float.infinity)
        )
        let threshold = kept.min(axis: -1, keepDims: true)
        probs = MLX.where(probs .>= threshold, probs, MLXArray.zeros(like: probs))
        probs = probs / probs.sum(axis: -1, keepDims: true)
    }
    return probs
}

public func sampleToken(probabilities: MLXArray) -> Int {
    categorical(MLX.log(probabilities + 1e-10)).item(Int.self)
}
