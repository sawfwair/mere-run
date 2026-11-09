import Foundation
import MLX

/// Distribution of `sampledTokenArray`, including its top-p prefilter policy.
/// Speculative verification needs the normalized mass, rather than a token draw.
/// History and logits remain on the request's stream.
package func samplingDistribution(
    logits: MLXArray, config: GenerationConfig, previousTokenIndices: MLXArray?, banMask: MLXArray?
) -> MLXArray {
    var scores = logits
    if let banMask { scores = scores + banMask }
    scores = applyingSamplingPenalties(scores, config: config, history: previousTokenIndices)
    if config.temperature == 0 {
        let result = MLXArray.zeros(scores.shape, dtype: .float32)
        result[argMax(scores)] = MLXArray(Float(1))
        return result
    }
    let useTopK = config.topK > 0 && config.topK < scores.dim(-1)
    let useTopP = config.topP > 0 && config.topP < 1
    let useMinP = config.minP > 0 && config.minP <= 1
    if useTopK || useTopP || useMinP, scores.dtype == .bfloat16 {
        scores = scores.asType(.float32)
    }
    if useTopK { scores = applyingTopK(scores, topK: config.topK) }
    if useTopP {
        var candidates: MLXArray?
        var working = scores
        let prefilter = max(0, config.topPPrefilter ?? SamplerPolicy.topPPrefilter)
        if !useTopK, prefilter > 0, scores.dim(-1) > prefilter * 4 {
            let vocabulary = scores.dim(-1)
            let partitioned = argPartition(scores, kth: vocabulary - prefilter, axis: -1)
            let top = partitioned[(vocabulary - prefilter)...]
            candidates = top
            working = scores[top]
        }
        let probabilities = softmax(working / config.temperature, axis: -1)
        let order = argSort(probabilities, axis: -1)
        let sorted = probabilities[order]
        var kept = MLX.where(
            cumsum(sorted, axis: -1) .> (1 - config.topP), sorted, MLXArray.zeros(like: sorted)
        )
        if useMinP { kept = applyingMinP(probabilities: kept, minP: config.minP) }
        let result = MLXArray.zeros(scores.shape, dtype: .float32)
        result[candidates.map { $0[order] } ?? order] = kept.asType(.float32)
        return result / result.sum()
    }
    var scaled = scores / config.temperature
    if useMinP {
        let threshold = scaled.max() + MLXArray(log(config.minP))
        scaled = MLX.where(scaled .>= threshold, scaled, MLXArray(-Float.infinity))
    }
    // categorical promotes its scaled logits before adding FP32 Gumbel noise.
    // Match that boundary rather than rounding softmax probabilities to BF16.
    return softmax(scaled.asType(.float32), axis: -1)
}
