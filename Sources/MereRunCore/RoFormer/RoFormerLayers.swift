import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

final class RoFormerRMSNorm: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dimensions: Int) {
        self._gamma.wrappedValue = MLX.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let meanSquare = MLX.mean(input.square(), axis: -1, keepDims: true)
        return input * MLX.rsqrt(meanSquare + MLXArray(1e-12).asType(input.dtype))
            * gamma.asType(input.dtype)
    }
}

final class RoFormerRotaryEmbedding: Module {
    @ParameterInfo(key: "freqs") var frequencies: MLXArray

    init(dimensions: Int) {
        self._frequencies.wrappedValue = MLX.zeros([dimensions / 2])
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let heads = input.dim(2)
        let dimensions = input.dim(3)
        let positions = MLXArray((0..<sequence).map(Float.init))
        let angles = positions.expandedDimensions(axis: -1) * frequencies.expandedDimensions(axis: 0)
        let cosine = MLX.cos(angles).reshaped(1, sequence, 1, dimensions / 2).asType(input.dtype)
        let sine = MLX.sin(angles).reshaped(1, sequence, 1, dimensions / 2).asType(input.dtype)
        let paired = input.reshaped(batch, sequence, heads, dimensions / 2, 2)
        let even = paired[.ellipsis, 0]
        let odd = paired[.ellipsis, 1]
        return MLX.stacked([
            even * cosine - odd * sine,
            odd * cosine + even * sine,
        ], axis: -1).reshaped(batch, sequence, heads, dimensions)
    }
}

final class RoFormerAttentionOutput: Module {
    @ModuleInfo(key: "0") var projection: Linear

    init(inputDimensions: Int, outputDimensions: Int) {
        self._projection.wrappedValue = Linear(inputDimensions, outputDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { projection(input) }
}

final class RoFormerAttention: Module {
    @ModuleInfo(key: "norm") var norm: RoFormerRMSNorm
    @ModuleInfo(key: "rotary_embed") var rotary: RoFormerRotaryEmbedding
    @ModuleInfo(key: "to_qkv") var qkv: Linear
    @ModuleInfo(key: "to_gates") var gates: Linear
    @ModuleInfo(key: "to_out") var output: RoFormerAttentionOutput

    private let heads: Int
    private let headDimension: Int
    private let scale: Float

    init(dimensions: Int, heads: Int, headDimension: Int) {
        self.heads = heads
        self.headDimension = headDimension
        self.scale = 1 / sqrt(Float(headDimension))
        self._norm.wrappedValue = RoFormerRMSNorm(dimensions: dimensions)
        self._rotary.wrappedValue = RoFormerRotaryEmbedding(dimensions: headDimension)
        self._qkv.wrappedValue = Linear(dimensions, 3 * heads * headDimension, bias: false)
        self._gates.wrappedValue = Linear(dimensions, heads, bias: true)
        self._output.wrappedValue = RoFormerAttentionOutput(
            inputDimensions: heads * headDimension,
            outputDimensions: dimensions
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let normalized = norm(input)
        let projected = qkv(normalized).reshaped(batch, sequence, 3, heads, headDimension)
        let parts = MLX.split(projected, parts: 3, axis: 2).map { $0.squeezed(axis: 2) }
        let queries = rotary(parts[0]).transposed(0, 2, 1, 3)
        let keys = rotary(parts[1]).transposed(0, 2, 1, 3)
        let values = parts[2].transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        ).transposed(0, 2, 1, 3)
        let gated = attended * MLX.sigmoid(gates(normalized)).expandedDimensions(axis: -1)
        return output(gated.reshaped(batch, sequence, heads * headDimension))
    }
}

final class RoFormerFeedForwardNet: Module {
    @ModuleInfo(key: "0") var norm: RoFormerRMSNorm
    @ModuleInfo(key: "1") var inputProjection: Linear
    @ModuleInfo(key: "4") var outputProjection: Linear

    init(dimensions: Int, expansionFactor: Int) {
        self._norm.wrappedValue = RoFormerRMSNorm(dimensions: dimensions)
        self._inputProjection.wrappedValue = Linear(dimensions, dimensions * expansionFactor, bias: true)
        self._outputProjection.wrappedValue = Linear(dimensions * expansionFactor, dimensions, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let hidden = inputProjection(norm(input))
        let exactGELU = hidden * MLXArray(0.5).asType(hidden.dtype)
            * (MLXArray(1).asType(hidden.dtype) + MLX.erf(hidden / sqrt(Float(2))))
        return outputProjection(exactGELU)
    }
}

final class RoFormerFeedForward: Module {
    @ModuleInfo(key: "net") var net: RoFormerFeedForwardNet

    init(dimensions: Int, expansionFactor: Int) {
        self._net.wrappedValue = RoFormerFeedForwardNet(
            dimensions: dimensions,
            expansionFactor: expansionFactor
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { net(input) }
}
