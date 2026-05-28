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

    public init(
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topK: Int = 0,
        topP: Float = 0.9,
        repetitionPenalty: Float? = 1.05,
        repetitionContextSize: Int = 20
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }
}

public func argMaxSample(logits: MLXArray) -> Int {
    argMax(logits, axis: -1).item(Int.self)
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

public func sampleToken(
    logits: MLXArray,
    config: GenerationConfig,
    previousTokens: [Int]
) -> Int {
    var logits = logits

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
