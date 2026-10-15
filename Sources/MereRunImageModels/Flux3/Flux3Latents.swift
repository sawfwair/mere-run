import MLX

public enum Flux3Latents {
    /// Channel-major 2x2 patch order: channel, patch row, patch column.
    public static func pack(_ mean: MLXArray, runningMean: MLXArray, runningVariance: MLXArray) -> MLXArray {
        precondition(mean.ndim == 4 && mean.dim(0) == 1 && mean.dim(1) == 32)
        precondition(mean.dim(2).isMultiple(of: 2) && mean.dim(3).isMultiple(of: 2))
        let height = mean.dim(2) / 2, width = mean.dim(3) / 2
        let packed = mean.reshaped(1, 32, height, 2, width, 2).transposed(0, 2, 4, 1, 3, 5).reshaped(height * width, 128)
        return (packed.asType(.float32) - runningMean.reshaped(1, 128)) / sqrt(runningVariance.reshaped(1, 128) + 1e-4)
    }

    public static func unpack(_ tokens: MLXArray, height: Int, width: Int,
                              runningMean: MLXArray, runningVariance: MLXArray) -> MLXArray {
        precondition(tokens.shape == [height * width, 128])
        let values = tokens.asType(.float32) * sqrt(runningVariance.reshaped(1, 128) + 1e-4) + runningMean.reshaped(1, 128)
        return values.reshaped(1, height, width, 32, 2, 2).transposed(0, 3, 1, 4, 2, 5).reshaped(1, 32, height * 2, width * 2)
    }

    /// Image RoPE uses half-grid spatial coordinates. References differ on the last axis.
    public static func imageIDs(height: Int, width: Int, referenceIndex: Int? = nil) -> MLXArray {
        precondition(height > 0 && width > 0)
        let last = referenceIndex.map { -Float(1 + $0) } ?? 0
        let values = (0..<(height * width)).flatMap { [Float(0), Float($0 / width) * 0.5, Float($0 % width) * 0.5, last] }
        return MLXArray(values).reshaped(height * width, 4)
    }

    public static func textIDs(length: Int) -> MLXArray {
        Flux2PosEmbed.prepareTextIds(seqLen: length)
    }
}
