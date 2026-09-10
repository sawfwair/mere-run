import Foundation
import MLX
import MLXFast
import MLXNN

public struct MiniMaxH3VideoDecoderConfiguration: Hashable, Sendable {
    public let latentChannels: Int
    public let outputChannels: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let layerCount: Int
    public let headCount: Int
    public let headDimension: Int
    public let registerTokenCount: Int
    public let feedForwardMultiplier: Int
    public let rotaryDimensionRatio: Float
    public let rotaryTheta: Float

    public init(
        latentChannels: Int = 24,
        outputChannels: Int = 3,
        patchSize: Int = 16,
        temporalPatchSize: Int = 4,
        layerCount: Int = 36,
        headCount: Int = 32,
        headDimension: Int = 64,
        registerTokenCount: Int = 4,
        feedForwardMultiplier: Int = 4,
        rotaryDimensionRatio: Float = 0.75,
        rotaryTheta: Float = 100
    ) {
        self.latentChannels = latentChannels
        self.outputChannels = outputChannels
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.headDimension = headDimension
        self.registerTokenCount = registerTokenCount
        self.feedForwardMultiplier = feedForwardMultiplier
        self.rotaryDimensionRatio = rotaryDimensionRatio
        self.rotaryTheta = rotaryTheta
    }

    package var hiddenSize: Int { headCount * headDimension }
    package var rotaryDimension: Int { Int(Float(headDimension) * rotaryDimensionRatio) }
}

package final class MiniMaxH3VideoDecoderAttention: Module {
    package let headCount: Int
    package let headDimension: Int
    package let rotaryDimension: Int

    @ModuleInfo(key: "to_qkv") package var queryKeyValue: Linear
    @ModuleInfo(key: "to_out") package var output: Linear

    package init(configuration: MiniMaxH3VideoDecoderConfiguration) {
        headCount = configuration.headCount
        headDimension = configuration.headDimension
        rotaryDimension = configuration.rotaryDimension
        let hidden = configuration.hiddenSize
        _queryKeyValue.wrappedValue = Linear(hidden, 3 * hidden, bias: true)
        _output.wrappedValue = Linear(hidden, hidden, bias: true)
    }

    package func callAsFunction(_ input: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let projected = MLX.split(queryKeyValue(input), parts: 3, axis: -1)
        var q = projected[0].reshaped(batch, sequence, headCount, headDimension)
        var k = projected[1].reshaped(batch, sequence, headCount, headDimension)
        let v = projected[2].reshaped(batch, sequence, headCount, headDimension)

        q = MiniMaxH3VideoDecoderAttention.rmsNormalizeHeads(q)
        k = MiniMaxH3VideoDecoderAttention.rmsNormalizeHeads(k)
        q = applyRotary(q, cosine: cosine, sine: sine)
        k = applyRotary(k, cosine: cosine, sine: sine)

        let attended = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3),
            keys: k.transposed(0, 2, 1, 3),
            values: v.transposed(0, 2, 1, 3),
            scale: 1 / sqrt(Float(headDimension)),
            mask: .none
        )
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, -1))
    }

    private static func rmsNormalizeHeads(_ value: MLXArray) -> MLXArray {
        let value32 = value.asType(.float32)
        let denominator = MLX.sqrt(MLX.mean(value32 * value32, axis: -1, keepDims: true) + 1e-5)
        return (value32 / denominator).asType(value.dtype)
    }

    private func applyRotary(_ value: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let rotary = value[0..., 0..., 0..., 0..<rotaryDimension]
        let pass = value[0..., 0..., 0..., rotaryDimension...]
        let halves = MLX.split(rotary, parts: 2, axis: -1)
        let rotated = MLX.concatenated([-halves[1], halves[0]], axis: -1)
        return MLX.concatenated([rotary * cosine + rotated * sine, pass], axis: -1)
    }
}

package final class MiniMaxH3VideoDecoderFeedForward: Module {
    @ModuleInfo(key: "linear_in") package var input: Linear
    @ModuleInfo(key: "linear_out") package var output: Linear

    package init(configuration: MiniMaxH3VideoDecoderConfiguration) {
        let hidden = configuration.hiddenSize
        let intermediate = hidden * configuration.feedForwardMultiplier
        _input.wrappedValue = Linear(hidden, intermediate * 2, bias: true)
        _output.wrappedValue = Linear(intermediate, hidden, bias: true)
    }

    package func callAsFunction(_ value: MLXArray) -> MLXArray {
        let parts = MLX.split(input(value), parts: 2, axis: -1)
        return output(MLXNN.silu(parts[0]) * parts[1])
    }
}

package final class MiniMaxH3VideoDecoderBlock: Module {
    @ModuleInfo(key: "norm1") package var attentionNorm: RMSNorm
    @ModuleInfo(key: "attn") package var attention: MiniMaxH3VideoDecoderAttention
    @ParameterInfo(key: "scale1") package var attentionScale: MLXArray
    @ModuleInfo(key: "norm2") package var feedForwardNorm: RMSNorm
    @ModuleInfo(key: "ff") package var feedForward: MiniMaxH3VideoDecoderFeedForward
    @ParameterInfo(key: "scale2") package var feedForwardScale: MLXArray

    package init(configuration: MiniMaxH3VideoDecoderConfiguration) {
        _attentionNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: 1e-5)
        _attention.wrappedValue = MiniMaxH3VideoDecoderAttention(configuration: configuration)
        _attentionScale.wrappedValue = MLXArray.zeros([configuration.hiddenSize])
        _feedForwardNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: 1e-5)
        _feedForward.wrappedValue = MiniMaxH3VideoDecoderFeedForward(configuration: configuration)
        _feedForwardScale.wrappedValue = MLXArray.zeros([configuration.hiddenSize])
    }

    package func callAsFunction(_ value: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        var hidden = value
        hidden = hidden + attention(attentionNorm(hidden), cosine: cosine, sine: sine) * attentionScale
        return hidden + feedForward(feedForwardNorm(hidden)) * feedForwardScale
    }
}

public final class MiniMaxH3VideoDecoder: Module {
    public let configuration: MiniMaxH3VideoDecoderConfiguration

    @ModuleInfo(key: "proj_in") package var input: Linear
    @ParameterInfo(key: "register_tokens") package var registerTokens: MLXArray
    @ModuleInfo(key: "transformer_blocks") package var blocks: [MiniMaxH3VideoDecoderBlock]
    @ModuleInfo(key: "norm_out") package var outputNorm: LayerNorm
    @ModuleInfo(key: "proj_out") package var output: Linear

    public init(configuration: MiniMaxH3VideoDecoderConfiguration = .init()) {
        self.configuration = configuration
        _input.wrappedValue = Linear(configuration.latentChannels, configuration.hiddenSize, bias: true)
        _registerTokens.wrappedValue = MLXArray.zeros([
            1, configuration.registerTokenCount, configuration.hiddenSize,
        ])
        _blocks.wrappedValue = (0..<configuration.layerCount).map { _ in
            MiniMaxH3VideoDecoderBlock(configuration: configuration)
        }
        _outputNorm.wrappedValue = LayerNorm(dimensions: configuration.hiddenSize, eps: 1e-5)
        _output.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.outputChannels * configuration.temporalPatchSize
                * configuration.patchSize * configuration.patchSize,
            bias: true
        )
    }

    /// Decodes denormalized `[B, C, T, H, W]` H3 latents to ImageNet-normalized RGB.
    public func callAsFunction(_ latents: MLXArray) -> MLXArray {
        precondition(latents.ndim == 5 && latents.dim(1) == configuration.latentChannels)
        let batch = latents.dim(0)
        let frames = latents.dim(2)
        let height = latents.dim(3)
        let width = latents.dim(4)
        var hidden = input(
            latents.transposed(0, 2, 3, 4, 1)
                .reshaped(batch, frames * height * width, configuration.latentChannels)
        )
        let patchCount = hidden.dim(1)
        let registers = MLX.tiled(registerTokens, repetitions: [batch, 1, 1])
        let zeroToken = MLXArray.zeros([batch, 1, configuration.hiddenSize], dtype: hidden.dtype)
        hidden = MLX.concatenated([hidden, registers.asType(hidden.dtype), zeroToken], axis: 1)
        let (cosine, sine) = rotaryEmbedding(
            batch: batch,
            frames: frames,
            height: height,
            width: width,
            dtype: hidden.dtype
        )
        for block in blocks {
            hidden = block(hidden, cosine: cosine, sine: sine)
        }
        hidden = output(outputNorm(hidden))[0..., 0..<patchCount, 0...]
        let patch = configuration.patchSize
        let temporalPatch = configuration.temporalPatchSize
        return hidden
            .reshaped(batch, frames, height, width, configuration.outputChannels, temporalPatch, patch, patch)
            .transposed(0, 4, 1, 5, 2, 6, 3, 7)
            .reshaped(batch, configuration.outputChannels, frames * temporalPatch, height * patch, width * patch)
    }

    private func rotaryEmbedding(
        batch: Int,
        frames: Int,
        height: Int,
        width: Int,
        dtype: DType
    ) -> (MLXArray, MLXArray) {
        let axisWidth = configuration.rotaryDimension / 6
        let inverse = MLX.pow(
            MLXArray(configuration.rotaryTheta),
            -MLXArray(stride(from: 0, to: axisWidth * 2, by: 2).map(Float.init))
                / Float(axisWidth * 2)
        )
        func normalized(_ count: Int) -> MLXArray {
            (2 * ((MLXArray(0..<count).asType(.float32) + 0.5) / Float(count)) - 1) * (2 * Float.pi)
        }
        let t = normalized(frames).reshaped(frames, 1, 1, 1) * inverse
        let h = normalized(height).reshaped(1, height, 1, 1) * inverse
        let w = normalized(width).reshaped(1, 1, width, 1) * inverse
        var angles = MLX.concatenated([
            MLX.broadcast(t, to: [frames, height, width, axisWidth]),
            MLX.broadcast(h, to: [frames, height, width, axisWidth]),
            MLX.broadcast(w, to: [frames, height, width, axisWidth]),
        ], axis: -1).reshaped(1, frames * height * width, 1, configuration.rotaryDimension / 2)
        angles = MLX.concatenated([angles, angles], axis: -1)
        let suffix = MLXArray.zeros([
            1, configuration.registerTokenCount + 1, 1, configuration.rotaryDimension,
        ])
        angles = MLX.tiled(MLX.concatenated([angles, suffix], axis: 1), repetitions: [batch, 1, configuration.headCount, 1])
        return (MLX.cos(angles).asType(dtype), MLX.sin(angles).asType(dtype))
    }
}
