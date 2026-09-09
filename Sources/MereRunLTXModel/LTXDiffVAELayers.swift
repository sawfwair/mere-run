import Foundation
import MLX
import MLXFast
import MLXNN

final class LTXDiffVAEDeterministicStage: Module {
    @ModuleInfo(key: "blocks") var blocks: [LTXDiffVAEDeterministicBlock]

    init(channels: Int, depth: Int, kernel: (Int, Int, Int)) {
        self._blocks.wrappedValue = (0..<depth).map { _ in
            LTXDiffVAEDeterministicBlock(channels: channels, kernel: kernel)
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        blocks.reduce(input) { hidden, block in block(hidden) }
    }
}

final class LTXDiffVAEDeterministicBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "attn") var attention: LTXDiffVAENeighborhoodAttention3D
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: LTXDiffVAESwiGLU

    init(channels: Int, kernel: (Int, Int, Int)) {
        self._norm1.wrappedValue = RMSNorm(dimensions: channels, eps: 1e-6)
        self._attention.wrappedValue = LTXDiffVAENeighborhoodAttention3D(
            channels: channels,
            kernel: kernel
        )
        self._norm2.wrappedValue = RMSNorm(dimensions: channels, eps: 1e-6)
        self._mlp.wrappedValue = LTXDiffVAESwiGLU(channels: channels)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = input + attention(norm1(input))
        hidden = hidden + mlp(norm2(hidden))
        return hidden
    }
}

final class LTXDiffVAELinearUpsampler: Module {
    let stride: (Int, Int, Int)
    @ModuleInfo(key: "proj") var projection: Linear

    init(channels: Int, stride: (Int, Int, Int), reduction: Int) {
        self.stride = stride
        self._projection.wrappedValue = Linear(
            channels,
            channels * stride.0 * stride.1 * stride.2 / reduction,
            bias: true
        )
    }

    func callAsFunction(_ input: MLXArray, dropLeadingFrame: Bool) -> MLXArray {
        let batch = input.dim(0)
        let frames = input.dim(1)
        let height = input.dim(2)
        let width = input.dim(3)
        let expanded = projection(input)
        let channelsAfterShuffle = expanded.dim(4) / (stride.0 * stride.1 * stride.2)
        var output = expanded
            .reshaped(batch, frames, height, width, channelsAfterShuffle, stride.0, stride.1, stride.2)
            .transposed(0, 1, 5, 2, 6, 3, 7, 4)
            .reshaped(
                batch,
                frames * stride.0,
                height * stride.1,
                width * stride.2,
                channelsAfterShuffle
            )
        if stride.0 == 2, dropLeadingFrame {
            output = output[0..., 1..., 0..., 0..., 0...]
        }
        return output
    }
}

final class LTXDiffVAESharedAdaLN: Module {
    @ModuleInfo(key: "proj") var projection: Linear

    override init() {
        self._projection.wrappedValue = Linear(384, 7 * 256, bias: true)
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        projection(ltxDiffVAESiLU(timestep)).reshaped(timestep.dim(0), 7, 256)
    }
}

final class LTXDiffVAETimestepEmbedder: Module {
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear

    override init() {
        self._first.wrappedValue = Linear(256, 384, bias: true)
        self._second.wrappedValue = Linear(384, 384, bias: true)
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let projected = ltxDiffVAETimestepEmbedding(timestep * MLXArray(Float(1_000)))
            .asType(timestep.dtype)
        return second(ltxDiffVAESiLU(first(projected)))
    }
}

final class LTXDiffVAEDiffusionBlock: Module {
    @ModuleInfo(key: "context_proj") var contextProjection: Linear
    @ModuleInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "attn") var attention: LTXDiffVAENeighborhoodAttention3D
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: LTXDiffVAESwiGLU

    init(channels: Int, contextChannels: Int, kernel: (Int, Int, Int)) {
        self._contextProjection.wrappedValue = Linear(contextChannels, channels, bias: true)
        self._scaleShiftTable.wrappedValue = MLX.zeros([7, channels], dtype: .float32)
        self._norm1.wrappedValue = RMSNorm(dimensions: channels, eps: 1e-6)
        self._attention.wrappedValue = LTXDiffVAENeighborhoodAttention3D(
            channels: channels,
            kernel: kernel
        )
        self._norm2.wrappedValue = RMSNorm(dimensions: channels, eps: 1e-6)
        self._mlp.wrappedValue = LTXDiffVAESwiGLU(channels: channels)
    }

    func callAsFunction(
        _ input: MLXArray,
        context: MLXArray,
        modulation: MLXArray
    ) -> MLXArray {
        let batch = input.dim(0)
        let combined = modulation + scaleShiftTable
            .asType(modulation.dtype)
            .reshaped(1, 7, input.dim(4))
        let scaleAttention = combined[0..., 0, 0...].reshaped(batch, 1, 1, 1, input.dim(4))
        let shiftAttention = combined[0..., 1, 0...].reshaped(batch, 1, 1, 1, input.dim(4))
        let scaleMLP = combined[0..., 3, 0...].reshaped(batch, 1, 1, 1, input.dim(4))
        let shiftMLP = combined[0..., 4, 0...].reshaped(batch, 1, 1, 1, input.dim(4))
        var hidden = input + contextProjection(context)
        let normalizedAttention = norm1(hidden) * (MLXArray(1).asType(hidden.dtype) + scaleAttention)
            + shiftAttention
        hidden = hidden + attention(normalizedAttention)
        let normalizedMLP = norm2(hidden) * (MLXArray(1).asType(hidden.dtype) + scaleMLP)
            + shiftMLP
        return hidden + mlp(normalizedMLP)
    }
}

final class LTXDiffVAESwiGLU: Module {
    @ModuleInfo(key: "w_up") var up: Linear
    @ModuleInfo(key: "w_gate") var gate: Linear
    @ModuleInfo(key: "w_down") var down: Linear

    init(channels: Int) {
        let hidden = ((channels * 4 + 15) / 16) * 16
        self._up.wrappedValue = Linear(channels, hidden, bias: false)
        self._gate.wrappedValue = Linear(channels, hidden, bias: false)
        self._down.wrappedValue = Linear(hidden, channels, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let tokenCount = input.dim(0) * input.dim(1) * input.dim(2) * input.dim(3)
        let flat = input.reshaped(tokenCount, input.dim(4))
        var chunks: [MLXArray] = []
        chunks.reserveCapacity((tokenCount + 16_383) / 16_384)
        for start in stride(from: 0, to: tokenCount, by: 16_384) {
            let end = min(start + 16_384, tokenCount)
            let tile = flat[start..<end]
            chunks.append(down(ltxDiffVAESiLU(gate(tile)) * up(tile)))
        }
        return MLX.concatenated(chunks, axis: 0).reshaped(input.shape)
    }
}

final class LTXDiffVAENeighborhoodAttention3D: Module {
    let channels: Int
    let headDimension = 64
    let kernel: (Int, Int, Int)
    let scoreBudget: Int

    @ModuleInfo(key: "to_q") var queryProjection: Linear
    @ModuleInfo(key: "to_k") var keyProjection: Linear
    @ModuleInfo(key: "to_v") var valueProjection: Linear
    @ModuleInfo(key: "proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    init(
        channels: Int,
        kernel: (Int, Int, Int),
        scoreBudget: Int = 1 << 25
    ) {
        precondition(channels % headDimension == 0)
        self.channels = channels
        self.kernel = kernel
        self.scoreBudget = scoreBudget
        self._queryProjection.wrappedValue = Linear(channels, channels, bias: true)
        self._keyProjection.wrappedValue = Linear(channels, channels, bias: true)
        self._valueProjection.wrappedValue = Linear(channels, channels, bias: true)
        self._outputProjection.wrappedValue = Linear(channels, channels, bias: true)
        self._queryNorm.wrappedValue = RMSNorm(dimensions: headDimension, eps: 1e-6)
        self._keyNorm.wrappedValue = RMSNorm(dimensions: headDimension, eps: 1e-6)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let frames = input.dim(1)
        let height = input.dim(2)
        let width = input.dim(3)
        precondition(frames >= kernel.0 && height >= kernel.1 && width >= kernel.2)
        let heads = channels / headDimension
        let headShape = [batch, frames, height, width, heads, headDimension]
        var query = queryNorm(queryProjection(input).reshaped(headShape))
            * MLXArray(1 / Float(headDimension).squareRoot()).asType(input.dtype)
        var key = keyNorm(keyProjection(input).reshaped(headShape))
        let value = valueProjection(input).reshaped(headShape)
        query = ltxDiffVAEAbsoluteRoPE(query)
        key = ltxDiffVAEAbsoluteRoPE(key)
        let attended = LTXDiffVAEMetalNeighborhoodAttention.apply(
            query: query,
            key: key,
            value: value,
            kernel: kernel
        ) ?? ltxDiffVAENeighborhoodAttention(
            query: query,
            key: key,
            value: value,
            kernel: kernel,
            scoreBudget: scoreBudget
        )
        return outputProjection(attended.reshaped(batch, frames, height, width, channels))
    }
}

func ltxDiffVAEWindowBounds(length: Int, kernel: Int) -> (starts: [Int], ends: [Int]) {
    let effectiveKernel = min(length, kernel)
    let maximumStart = length - effectiveKernel
    let half = effectiveKernel / 2
    let starts = (0..<length).map { min(max($0 - half, 0), maximumStart) }
    return (starts, starts.map { $0 + effectiveKernel })
}
