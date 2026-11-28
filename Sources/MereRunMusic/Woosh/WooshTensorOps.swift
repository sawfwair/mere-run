import Foundation
import MLX
import MLXFast

enum WooshTensorOps {
    static func geluTanh(_ x: MLXArray) -> MLXArray {
        let c = MLXArray(0.7978845608028654).asType(x.dtype)
        let k = MLXArray(0.044715).asType(x.dtype)
        return MLXArray(0.5).asType(x.dtype) * x * (MLXArray(1).asType(x.dtype) + MLX.tanh(c * (x + k * x * x * x)))
    }

    static func softplus(_ x: MLXArray) -> MLXArray {
        MLX.log(MLXArray(1).asType(x.dtype) + MLX.exp(-MLX.abs(x))) + MLX.maximum(x, MLXArray(0).asType(x.dtype))
    }

    static func layerNormNoAffine(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        MLXFast.layerNorm(x, eps: eps)
    }

    static func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float = 1e-6) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }

    static func l2Normalize(_ x: MLXArray, eps: Float = 1e-12) -> MLXArray {
        let norm = MLX.sqrt(MLX.sum(x * x, axis: -1, keepDims: true) + MLXArray(eps).asType(x.dtype))
        return x / norm
    }

    static func conv1dWeightOIToOKI(_ value: MLXArray) -> MLXArray {
        guard value.ndim == 3 else { return value }
        let transposed = value.transposed(0, 2, 1)
        return transposed.reshaped(-1).reshaped(transposed.shape)
    }

    static func boolMaskToAttentionBias(_ mask: MLXArray, dtype: DType, queryLength: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let keyMask = mask.asType(dtype).expandedDimensions(axes: [1, 2])
        let zeros = MLXArray.zeros([mask.dim(0), 1, queryLength, mask.dim(1)], dtype: dtype)
        let negative = zeros + MLXArray(-1e9).asType(dtype)
        return .array(MLX.where(keyMask .> MLXArray(0).asType(dtype), zeros, negative))
    }
}
