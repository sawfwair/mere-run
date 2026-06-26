import Foundation
import MLX
import MLXNN

final class Q35RMSNorm: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray

    private let eps: Float

    init(dimensions: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let x32 = x.asType(.float32)
        let variance = MLX.mean(x32 * x32, axis: -1, keepDims: true)
        let normalized = x32 * rsqrt(variance + MLXArray(eps))
        let scale = weight.asType(.float32) + MLXArray(1.0)
        let output = normalized * scale.reshaped(Array(repeating: 1, count: x.ndim - 1) + [scale.dim(0)])
        return output.asType(dtype)
    }
}
