import MLX

enum Flux2KleinBatchNorm {
    static func normalizePackedLatents(_ latents: MLXArray, mean: MLXArray, variance: MLXArray) -> MLXArray {
        let stats = reshapedStats(mean: mean, variance: variance)
        return (latents.asType(.float32) - stats.mean) / stats.std
    }

    static func denormalizePackedLatents(_ latents: MLXArray, mean: MLXArray, variance: MLXArray) -> MLXArray {
        let stats = reshapedStats(mean: mean, variance: variance)
        return latents.asType(.float32) * stats.std + stats.mean
    }

    static func std(mean: MLXArray, variance: MLXArray) -> MLXArray {
        reshapedStats(mean: mean, variance: variance).std
    }

    private static func reshapedStats(mean: MLXArray, variance: MLXArray) -> (mean: MLXArray, std: MLXArray) {
        let reshapedMean = mean.asType(.float32).reshaped([1, -1, 1, 1])
        let reshapedVariance = variance.asType(.float32).reshaped([1, -1, 1, 1])
        let eps = MLXArray(Float(1e-4))
        return (reshapedMean, MLX.sqrt(reshapedVariance + eps))
    }
}
