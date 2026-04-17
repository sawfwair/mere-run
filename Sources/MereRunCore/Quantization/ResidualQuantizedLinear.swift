import Foundation
import MLX
import MLXNN

/// QuantizedLinear with an optional low-rank residual (SVD-style) term.
/// This enables weight-only quantization with a low-rank correction:
///   y = x * Wq^T + x * (R)^T, where R = U * V
public final class ResidualQuantizedLinear: QuantizedLinear {
    public let residualDown: MLXArray?
    public let residualUp: MLXArray?

    public init(
        weight: MLXArray,
        bias: MLXArray? = nil,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine,
        residualDown: MLXArray?,
        residualUp: MLXArray?
    ) {
        let outDim = weight.shape2.0
        let inDim = weight.shape2.1 * 32 / bits
        let aligned = ResidualQuantizedLinear.alignResidual(
            residualDown: residualDown,
            residualUp: residualUp,
            outDim: outDim,
            inDim: inDim
        )
        self.residualDown = aligned?.down
        self.residualUp = aligned?.up
        super.init(
            weight: weight,
            bias: bias,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        self.freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = super.callAsFunction(x)
        guard let residualDown, let residualUp else { return out }

        // Residual is computed in fp32 for stability, then cast back.
        let xF = x.asType(.float32)
        let residual = MLX.matmul(MLX.matmul(xF, residualDown.T), residualUp.T)
        out = out + residual.asType(out.dtype)
        return out
    }

    private struct AlignedResidual {
        let down: MLXArray
        let up: MLXArray
    }

    private static func alignResidual(
        residualDown: MLXArray?,
        residualUp: MLXArray?,
        outDim: Int,
        inDim: Int
    ) -> AlignedResidual? {
        guard let residualDown, let residualUp else { return nil }
        guard residualDown.ndim == 2, residualUp.ndim == 2 else { return nil }

        var down = residualDown
        var up = residualUp

        // Expect down: [rank, inDim], up: [outDim, rank]
        if down.dim(1) != inDim, down.dim(0) == inDim {
            down = down.T
        }
        if up.dim(0) != outDim, up.dim(1) == outDim {
            up = up.T
        }

        let rank = down.dim(0)
        guard down.dim(1) == inDim, up.dim(0) == outDim, up.dim(1) == rank else {
            return nil
        }

        return AlignedResidual(down: down, up: up)
    }
}
