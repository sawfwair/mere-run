import Foundation
import MLX
import MLXFast
import MLXNN

/// Proportional RoPE for Gemma 4 full-attention layers.
///
/// Frequencies are computed relative to the **full** head dimension (not just the
/// rotated portion), and rotation is applied to the first `rotatedDims/2`
/// elements of each half of the head — matching HF's rotate_half convention.
package final class Gemma4ProportionalRoPE: Module, OffsetLayer {
    private let dims: Int
    private let rotatedDims: Int
    private let traditional: Bool
    private let freqs: MLXArray?

    package init(dims: Int, traditional: Bool = false, base: Float = 10_000, partialRotaryFactor: Float = 1.0, factor: Float = 1.0) {
        self.dims = dims
        self.traditional = traditional
        let ropeAngles = Int(partialRotaryFactor * Float(dims / 2))
        self.rotatedDims = 2 * ropeAngles

        if rotatedDims > 0 {
            let exponents = MLXArray(stride(from: Float(0), to: Float(rotatedDims), by: 2))
                / MLXArray(Float(dims))
            self.freqs = MLXArray(factor) * pow(MLXArray(base), exponents)
        } else {
            self.freqs = nil
        }
        super.init()
    }

    package func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        guard rotatedDims > 0 else { return x }

        let head = x[0..., 0..., 0..., ..<dims]
        let half = dims / 2
        let rotHalf = rotatedDims / 2

        let left = head[0..., 0..., 0..., ..<half]
        let right = head[0..., 0..., 0..., half...]

        let toRotate = concatenated(
            [left[0..., 0..., 0..., ..<rotHalf], right[0..., 0..., 0..., ..<rotHalf]],
            axis: -1
        )
        let rotated = MLXFast.RoPE(
            toRotate,
            dimensions: rotatedDims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )

        let newLeft = concatenated(
            [rotated[0..., 0..., 0..., ..<rotHalf], left[0..., 0..., 0..., rotHalf...]],
            axis: -1
        )
        let newRight = concatenated(
            [rotated[0..., 0..., 0..., rotHalf...], right[0..., 0..., 0..., rotHalf...]],
            axis: -1
        )
        let newHead = concatenated([newLeft, newRight], axis: -1)

        if x.dim(-1) > dims {
            return concatenated([newHead, x[0..., 0..., 0..., dims...]], axis: -1)
        }
        return newHead
    }
}
