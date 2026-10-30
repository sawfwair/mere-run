import MLX

extension IdolConditioning {
    /// FLUX.2 stochastic posterior sample, packed in channel/row/column order.
    /// BatchNorm arithmetic runs in the posterior's BF16 dtype before conversion to FP32.
    public static func packVAE(_ sample: MLXArray, mean: MLXArray, variance: MLXArray,
                               epsilon: Float = 1e-4) throws -> MLXArray {
        guard sample.ndim == 4, sample.dim(0) == 1, sample.dim(1) == 32,
              sample.dim(2) > 0, sample.dim(3) > 0, sample.dim(2) % 2 == 0, sample.dim(3) % 2 == 0,
              mean.shape == [128], variance.shape == [128], epsilon.isFinite, epsilon > 0 else {
            throw IdolError.invalidInput("Invalid VAE posterior or BatchNorm statistics.")
        }
        let height = sample.dim(2) / 2, width = sample.dim(3) / 2
        let packed = sample.reshaped(1, 32, height, 2, width, 2).transposed(0, 1, 3, 5, 2, 4)
            .reshaped(1, 128, height, width)
        let runningMean = mean.reshaped(1, 128, 1, 1).asType(sample.dtype)
        let standardDeviation = sqrt(variance.reshaped(1, 128, 1, 1).asType(sample.dtype) + epsilon).asType(sample.dtype)
        let tokens = ((packed - runningMean) / standardDeviation).asType(.float32)
            .reshaped(1, 128, height * width).transposed(0, 2, 1)
        return concatenated([MLX.zeros([1, 5, 128]), tokens], axis: 1)
    }
}
