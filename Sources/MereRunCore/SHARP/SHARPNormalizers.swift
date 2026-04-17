import MLX

public protocol SharpImageNormalizer {
    func callAsFunction(_ image: MLXArray) -> MLXArray
}

public final class SharpMeanStdNormalizer: SharpImageNormalizer, @unchecked Sendable {
    public let mean: MLXArray
    public let stdInv: MLXArray

    public init(mean: [Float], std: [Float]) {
        precondition(mean.count == std.count, "mean/std channel count mismatch.")
        self.mean = MLXArray(mean, [1, mean.count, 1, 1]).asType(.float32)
        self.stdInv = (1.0 / MLXArray(std, [1, std.count, 1, 1]).asType(.float32))
    }

    public func callAsFunction(_ image: MLXArray) -> MLXArray {
        (image - mean) * stdInv
    }
}

public class SharpAffineRangeNormalizer: SharpImageNormalizer, @unchecked Sendable {
    public let scale: Float
    public let bias: Float

    public init(inputRange: (Float, Float), outputRange: (Float, Float) = (0, 1)) {
        let inputMin = inputRange.0
        let inputMax = inputRange.1
        let outputMin = outputRange.0
        let outputMax = outputRange.1
        precondition(inputMax > inputMin, "Invalid inputRange.")
        precondition(outputMax > outputMin, "Invalid outputRange.")

        self.scale = (outputMax - outputMin) / (inputMax - inputMin)
        self.bias = outputMin - (inputMin * scale)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        if scale != 1.0 {
            y = y * scale
        }
        if bias != 0.0 {
            y = y + bias
        }
        return y
    }
}

public final class SharpMobileNetNormalizer: SharpAffineRangeNormalizer, @unchecked Sendable {
    public init(inputRange: (Float, Float) = (0, 1)) {
        super.init(inputRange: inputRange, outputRange: (-1, 1))
    }
}
