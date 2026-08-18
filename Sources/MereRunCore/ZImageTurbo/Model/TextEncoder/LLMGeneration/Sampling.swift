// Adapted from mlx-swift-lm MLXLMCommon/Evaluate.swift

import Foundation
import MLX
import MLXRandom

public struct GenerationConfig: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    /// Minimum token probability relative to the most likely token. Zero disables it.
    public var minP: Float
    public var repetitionPenalty: Float?
    public var repetitionContextSize: Int
    /// Token ids that must never be sampled. Applied as a -inf logit mask.
    public var bannedTokens: [Int]
    /// Per-request top-p candidate limit. Nil uses the process policy; zero
    /// requests exact full-vocabulary top-p sampling.
    public var topPPrefilter: Int?

    public init(
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topK: Int = 0,
        topP: Float = 0.9,
        minP: Float = 0,
        repetitionPenalty: Float? = 1.05,
        repetitionContextSize: Int = 20,
        bannedTokens: [Int] = [],
        topPPrefilter: Int? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.bannedTokens = bannedTokens
        self.topPPrefilter = topPPrefilter
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

/// Applies an exact top-k threshold without sorting the full vocabulary.
/// `argPartition` is linear-time on the selection axis and the selected
/// candidates are only reduced to their minimum; their order is irrelevant.
func applyingTopK(_ logits: MLXArray, topK: Int) -> MLXArray {
    let vocabulary = logits.dim(-1)
    guard topK > 0, topK < vocabulary else { return logits }
    let firstTopIndex = vocabulary - topK
    let partitioned = argPartition(logits, kth: firstTopIndex, axis: -1)
    let topIndices = partitioned[firstTopIndex...]
    let threshold = logits.take(topIndices, axis: -1).min()
    return MLX.where(logits .< threshold, MLXArray(-Float.infinity), logits)
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
    let useMinP = config.minP > 0 && config.minP <= 1

    if useTopK || useTopP || useMinP, logits.dtype == .bfloat16 {
        logits = logits.asType(.float32)
    }

    if useTopK {
        logits = applyingTopK(logits, topK: config.topK)
    }

    if useTopP {
        // Restrict the top-p math to the highest-logit candidates. Skipped
        // when an explicit top-k already thresholded the logits (masked
        // entries would defeat the partition) or when disabled.
        var candidateIndices: MLXArray?
        var workingLogits = logits
        let prefilter = max(0, config.topPPrefilter ?? SamplerPolicy.topPPrefilter)
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
        var topProbs = MLX.where(
            cumulativeProbs .> (1 - config.topP),
            sortedProbs,
            MLXArray.zeros(like: sortedProbs)
        )
        if useMinP {
            topProbs = applyingMinP(
                probabilities: topProbs,
                minP: config.minP
            )
        }
        let sortedToken = categorical(categoricalLogits(probabilities: topProbs))
        let chosen = sortedIndices[sortedToken]
        if let candidateIndices {
            return candidateIndices[chosen].asType(.int32)
        }
        return chosen.asType(.int32)
    }

    if useMinP {
        let scaledLogits = logits / config.temperature
        let threshold = scaledLogits.max(axis: -1, keepDims: true)
            + MLXArray(log(config.minP))
        let filtered = MLX.where(
            scaledLogits .>= threshold,
            scaledLogits,
            MLXArray(-Float.infinity)
        )
        return categorical(filtered).asType(.int32)
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

public func topPSample(
    logits: MLXArray,
    temperature: Float,
    topP: Float,
    minP: Float = 0
) -> Int {
    var logits = logits
    if logits.dtype == .bfloat16 {
        logits = logits.asType(.float32)
    }

    let probs = softmax(logits / temperature, axis: -1)
    let sortedIndices = argSort(probs, axis: -1)
    let sortedProbs = probs.take(sortedIndices, axis: -1)
    let cumulativeProbs = cumsum(sortedProbs, axis: -1)

    var topProbs = MLX.where(
        cumulativeProbs .> (1 - topP),
        sortedProbs,
        MLXArray.zeros(like: sortedProbs)
    )
    if minP > 0 && minP <= 1 {
        topProbs = applyingMinP(probabilities: topProbs, minP: minP)
    }

    let sortedToken = categorical(categoricalLogits(probabilities: topProbs))
    return sortedIndices[sortedToken].item(Int.self)
}

/// Keeps tokens whose probability is at least `minP` times the most likely
/// token's probability. The caller owns normalization, so this composes with
/// top-p without changing the already-selected nucleus.
func applyingMinP(probabilities: MLXArray, minP: Float) -> MLXArray {
    guard minP > 0, minP <= 1 else { return probabilities }
    let threshold = probabilities.max(axis: -1, keepDims: true) * minP
    return MLX.where(
        probabilities .>= threshold,
        probabilities,
        MLXArray.zeros(like: probabilities)
    )
}

public func minPSample(logits: MLXArray, temperature: Float, minP: Float) -> Int {
    var logits = logits
    if logits.dtype == .bfloat16 {
        logits = logits.asType(.float32)
    }
    let probs = softmax(logits / temperature, axis: -1)
    return sampleToken(probabilities: applyingMinP(probabilities: probs, minP: minP))
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
        logits = applyingTopK(logits, topK: topK)
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
    let useMinP = config.minP > 0 && config.minP <= 1

    if useTopK {
        // Preserve the legacy top-k + top-p boundary in the source dtype;
        // topPSample performs the float32 promotion after thresholding. The
        // top-k-only path historically promoted before choosing its boundary.
        if !useTopP, logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }
        logits = applyingTopK(logits, topK: config.topK)
        if useTopP {
            return topPSample(
                logits: logits,
                temperature: config.temperature,
                topP: config.topP,
                minP: config.minP
            )
        } else if useMinP {
            return minPSample(
                logits: logits,
                temperature: config.temperature,
                minP: config.minP
            )
        } else {
            return categoricalSample(logits: logits, temperature: config.temperature)
        }
    } else if useTopP {
        return topPSample(
            logits: logits,
            temperature: config.temperature,
            topP: config.topP,
            minP: config.minP
        )
    } else if useMinP {
        return minPSample(
            logits: logits,
            temperature: config.temperature,
            minP: config.minP
        )
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
        logits = applyingTopK(logits, topK: config.topK)
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
        let keptSortedProbs = MLX.where(
            cumulativeProbs .> (1 - config.topP),
            sortedProbs,
            MLXArray.zeros(like: sortedProbs)
        )
        let keptProbs = MLXArray.zeros(like: probs)
        keptProbs[sortedIndices] = keptSortedProbs
        probs = keptProbs
        probs = probs / probs.sum(axis: -1, keepDims: true)
    }
    if config.minP > 0 && config.minP <= 1 {
        probs = applyingMinP(probabilities: probs, minP: config.minP)
        probs = probs / probs.sum(axis: -1, keepDims: true)
    }
    return probs
}

/// Measures the chosen token against both the unmodified model distribution
/// and the exact distribution produced by the active sampling policy. This is
/// intentionally host-reading diagnostic work and must stay opt-in.
public func tokenLogprobMeasurement(
    logits: MLXArray,
    selectedToken: Int,
    config: GenerationConfig,
    previousTokens: [Int],
    topLogprobs: Int = 0
) -> ChatTokenLogprob {
    let rawProbabilities = softmax(logits.asType(.float32), axis: -1)
    let policyProbabilities = samplingProbabilities(
        logits: logits,
        config: config,
        previousTokens: previousTokens
    )
    let rawEntropyArray = distributionEntropy(rawProbabilities)
    let policyEntropyArray = distributionEntropy(policyProbabilities)
    let candidateCount = min(
        max(2, topLogprobs),
        policyProbabilities.dim(-1)
    )
    let rawTopIndices = argPartition(
        -rawProbabilities,
        kth: candidateCount - 1,
        axis: -1
    )[..<candidateCount].asType(.int32)
    let policyTopIndices = argPartition(
        -policyProbabilities,
        kth: candidateCount - 1,
        axis: -1
    )[..<candidateCount].asType(.int32)

    MLX.eval(
        rawProbabilities,
        policyProbabilities,
        rawEntropyArray,
        policyEntropyArray,
        rawTopIndices,
        policyTopIndices
    )

    let rawTop = rankedProbabilities(rawProbabilities, indices: rawTopIndices)
    let policyTop = rankedProbabilities(policyProbabilities, indices: policyTopIndices)
    let requestedPolicyTop = Array(policyTop.prefix(max(0, topLogprobs)))
    let candidates = requestedPolicyTop.map { candidate in
        ChatTopLogprob(
            tokenID: candidate.tokenID,
            rawLogprob: stableLogProbability(rawProbabilities[candidate.tokenID].item(Float.self)),
            policyLogprob: stableLogProbability(candidate.probability)
        )
    }

    return ChatTokenLogprob(
        tokenID: selectedToken,
        rawLogprob: stableLogProbability(rawProbabilities[selectedToken].item(Float.self)),
        policyLogprob: stableLogProbability(policyProbabilities[selectedToken].item(Float.self)),
        rawEntropy: Double(rawEntropyArray.item(Float.self)),
        policyEntropy: Double(policyEntropyArray.item(Float.self)),
        rawTop1Top2Margin: topLogprobMargin(rawTop),
        policyTop1Top2Margin: topLogprobMargin(policyTop),
        topLogprobs: candidates
    )
}

private func distributionEntropy(_ probabilities: MLXArray) -> MLXArray {
    let contributions = MLX.where(
        probabilities .> 0,
        -probabilities * MLX.log(probabilities),
        MLXArray.zeros(like: probabilities)
    )
    return contributions.sum(axis: -1)
}

private func rankedProbabilities(
    _ probabilities: MLXArray,
    indices: MLXArray
) -> [(tokenID: Int, probability: Float)] {
    let tokenIDs = indices.asArray(Int32.self).map(Int.init)
    return tokenIDs
        .map { tokenID in
            (tokenID, probabilities[tokenID].item(Float.self))
        }
        .sorted { lhs, rhs in lhs.probability > rhs.probability }
}

private func stableLogProbability(_ probability: Float) -> Double {
    Double(log(max(probability, Float.leastNonzeroMagnitude)))
}

private func topLogprobMargin(
    _ probabilities: [(tokenID: Int, probability: Float)]
) -> Double {
    guard probabilities.count >= 2 else { return 0 }
    return stableLogProbability(probabilities[0].probability)
        - stableLogProbability(probabilities[1].probability)
}

public func sampleToken(probabilities: MLXArray) -> Int {
    categorical(categoricalLogits(probabilities: probabilities)).item(Int.self)
}

/// Converts a normalized distribution to categorical logits without assigning
/// any mass to filtered tokens. At least one caller-provided probability must
/// be positive.
func categoricalLogits(probabilities: MLXArray) -> MLXArray {
    MLX.where(
        probabilities .> 0,
        MLX.log(probabilities),
        MLXArray(-Float.infinity)
    )
}
