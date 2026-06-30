import Foundation
import MLX
import MLXFast
import MLXNN

final class Q35RMSNorm: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray

    private let eps: Float
    private var effectiveWeight: MLXArray?

    init(dimensions: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if effectiveWeight == nil {
            let scale = weight + MLXArray(1.0).asType(weight.dtype)
            MLX.eval(scale)
            effectiveWeight = scale
        }
        return MLXFast.rmsNorm(x, weight: effectiveWeight ?? weight, eps: eps)
    }
}
