import Foundation
import MLX
import MLXFast
import MLXNN

final class Q35RMSNorm: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray

    private let eps: Float
    private let groupSize: Int?
    private let zeroCenteredWeight: Bool
    private let useFloat32Scale: Bool
    private var effectiveWeight: MLXArray?

    init(
        dimensions: Int, eps: Float, groupSize: Int? = nil, zeroCenteredWeight: Bool = true,
        useFloat32Scale: Bool = false
    ) {
        self._weight.wrappedValue = zeroCenteredWeight
            ? MLXArray.zeros([dimensions])
            : MLXArray.ones([dimensions])
        self.eps = eps
        self.groupSize = groupSize
        self.useFloat32Scale = useFloat32Scale
        self.zeroCenteredWeight = zeroCenteredWeight
        super.init()
    }

    func resolvedWeight() -> MLXArray {
        if effectiveWeight == nil {
            let scale = zeroCenteredWeight
                ? weight + MLXArray(1.0).asType(weight.dtype)
                : weight
            MLX.eval(scale)
            effectiveWeight = scale
        }
        return effectiveWeight ?? weight
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scale = resolvedWeight()
        if useFloat32Scale {
            return MLXFast.rmsNorm(x.asType(.float32), weight: scale, eps: eps).asType(x.dtype)
        }
        guard let groupSize else {
            return MLXFast.rmsNorm(x, weight: scale, eps: eps)
        }
        precondition(x.dim(-1).isMultiple(of: groupSize))
        let groupCount = x.dim(-1) / groupSize
        let grouped = x.reshaped(Array(x.shape.dropLast()) + [groupCount, groupSize])
        let groupedFloat = grouped.asType(.float32)
        let variance = MLX.square(groupedFloat).mean(axis: -1, keepDims: true)
        let normalized = groupedFloat * MLX.rsqrt(
            variance + MLXArray(eps).asType(.float32)
        )
        let groupedWeight = scale
            .asType(.float32)
            .reshaped(groupCount, groupSize)
        return (normalized * groupedWeight).asType(x.dtype).reshaped(x.shape)
    }
}
