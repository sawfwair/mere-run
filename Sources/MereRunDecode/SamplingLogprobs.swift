import Foundation
import MLX

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
