import MLX

/// Keep checkpoint normalization in FP32, including the learned scale and
/// optional centered offset. Cast once after normalization and multiplication.
/// This helper is outside Module, so prepared scales do not enter checkpoints.
package final class FusedRMSNorm {
    private let scale: MLXArray
    private let eps: Float

    package init(weight: MLXArray, eps: Float, centered: Bool = false) {
        let promoted = weight.asType(.float32)
        scale = centered ? 1 + promoted : promoted
        self.eps = eps
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: scale, eps: eps).asType(x.dtype)
    }
}
