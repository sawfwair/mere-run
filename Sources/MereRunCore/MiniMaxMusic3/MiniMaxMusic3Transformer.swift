import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

final class MiniMaxMusic3FourierEmbedding: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int) {
        self._weight.wrappedValue = MLXRandom.normal([dimensions / 2, 1])
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let angles = MLXArray(2 * Float.pi).asType(timestep.dtype)
            * MLX.matmul(timestep.reshaped(-1, 1), weight.transposed())
        return MLX.concatenated([MLX.cos(angles), MLX.sin(angles)], axis: -1)
    }
}

final class MiniMaxMusic3TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear

    init(inputDimensions: Int, outputDimensions: Int) {
        self._first.wrappedValue = Linear(inputDimensions, outputDimensions)
        self._second.wrappedValue = Linear(outputDimensions, outputDimensions)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        second(MLXNN.silu(first(input)))
    }
}

enum MiniMaxMusic3RotaryEmbedding {
    static func cache(length: Int, dimensions: Int, theta: Float, dtype: DType) -> (MLXArray, MLXArray) {
        var inverse: [Float] = []
        inverse.reserveCapacity(dimensions / 2)
        for index in 0..<(dimensions / 2) {
            let exponent = -Double(index * 2) / Double(dimensions)
            inverse.append(Float(Foundation.pow(Double(theta), exponent)))
        }
        let positions = MLXArray((0..<length).map(Float.init)).reshaped(length, 1)
        let frequencies = positions * MLXArray(inverse).reshaped(1, -1)
        let doubled = MLX.concatenated([frequencies, frequencies], axis: -1)
        return (MLX.cos(doubled).asType(dtype), MLX.sin(doubled).asType(dtype))
    }

    static func apply(_ input: MLXArray, cos: MLXArray, sin: MLXArray, dimensions: Int) -> MLXArray {
        let rotary = input[0..., 0..., 0..., 0..<dimensions]
        let halves = MLX.split(rotary, parts: 2, axis: -1)
        let rotatedHalf = MLX.concatenated([-halves[1], halves[0]], axis: -1)
        let shapedCos = cos.reshaped(1, cos.dim(0), 1, dimensions)
        let shapedSin = sin.reshaped(1, sin.dim(0), 1, dimensions)
        let rotated = rotary * shapedCos + rotatedHalf * shapedSin
        guard dimensions < input.dim(-1) else { return rotated }
        return MLX.concatenated([rotated, input[0..., 0..., 0..., dimensions...]], axis: -1)
    }
}

final class MiniMaxMusic3FlowAttention: Module {
    let heads: Int
    let headDimension: Int
    let rotaryDimension: Int

    @ModuleInfo(key: "to_q") var query: Linear
    @ModuleInfo(key: "to_k") var key: Linear
    @ModuleInfo(key: "to_v") var value: Linear
    @ModuleInfo(key: "to_out") var output: [Linear]

    private var usesFusedProjections = false

    init(dimensions: Int, heads: Int, headDimension: Int, rotaryDimension: Int) {
        self.heads = heads
        self.headDimension = headDimension
        self.rotaryDimension = rotaryDimension
        let inner = heads * headDimension
        self._query.wrappedValue = Linear(dimensions, inner, bias: false)
        self._key.wrappedValue = Linear(dimensions, inner, bias: false)
        self._value.wrappedValue = Linear(dimensions, inner, bias: false)
        self._output.wrappedValue = [Linear(inner, dimensions, bias: false)]
    }

    func callAsFunction(_ hidden: MLXArray, rotary: (MLXArray, MLXArray)) -> MLXArray {
        if !usesFusedProjections {
            return reference(hidden, rotary: rotary)
        }
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let projections: [MLXArray]
        if usesFusedProjections {
            projections = MLX.split(query(hidden), parts: 3, axis: -1)
        } else {
            projections = [query(hidden), key(hidden), value(hidden)]
        }
        var q = projections[0].reshaped(batch, length, heads, headDimension)
        var k = projections[1].reshaped(batch, length, heads, headDimension)
        let v = projections[2].reshaped(batch, length, heads, headDimension)
        q = MiniMaxMusic3RotaryEmbedding.apply(q, cos: rotary.0, sin: rotary.1, dimensions: rotaryDimension)
        k = MiniMaxMusic3RotaryEmbedding.apply(k, cos: rotary.0, sin: rotary.1, dimensions: rotaryDimension)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3).asType(.float32),
            keys: k.transposed(0, 2, 1, 3).asType(.float32),
            values: v.transposed(0, 2, 1, 3).asType(.float32),
            scale: 1 / Float(headDimension).squareRoot(),
            mask: .none
        )
        let merged = attended.asType(q.dtype).transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return output[0](merged)
    }

    private func reference(
        _ hidden: MLXArray,
        rotary: (MLXArray, MLXArray)
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        var q = query(hidden).reshaped(batch, length, heads, headDimension)
        var k = key(hidden).reshaped(batch, length, heads, headDimension)
        let v = value(hidden).reshaped(batch, length, heads, headDimension)
        q = MiniMaxMusic3RotaryEmbedding.apply(
            q,
            cos: rotary.0,
            sin: rotary.1,
            dimensions: rotaryDimension
        )
        k = MiniMaxMusic3RotaryEmbedding.apply(
            k,
            cos: rotary.0,
            sin: rotary.1,
            dimensions: rotaryDimension
        )
        let attended = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3).asType(.float32),
            keys: k.transposed(0, 2, 1, 3).asType(.float32),
            values: v.transposed(0, 2, 1, 3).asType(.float32),
            scale: 1 / Float(headDimension).squareRoot(),
            mask: .none
        )
        let merged = attended.asType(q.dtype).transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)
        return output[0](merged)
    }

    func prepareFusedProjections() {
        guard !usesFusedProjections else { return }
        let fused = MLX.concatenated([query.weight, key.weight, value.weight], axis: 0)
        MLX.eval(fused)
        update(modules: ModuleChildren.unflattened([
            ("to_q", Linear(weight: fused, bias: nil)),
            ("to_k", Linear(weight: MLXArray.zeros([1, 1], dtype: fused.dtype), bias: nil)),
            ("to_v", Linear(weight: MLXArray.zeros([1, 1], dtype: fused.dtype), bias: nil)),
        ]))
        usesFusedProjections = true
    }
}

final class MiniMaxMusic3TransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attention: MiniMaxMusic3FlowAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ff_in") var feedForwardInput: Linear
    @ModuleInfo(key: "ff_out") var feedForwardOutput: Linear

    init(configuration: MiniMaxMusic3TransformerConfiguration) {
        let inner = configuration.numAttentionHeads * configuration.attentionHeadDim
        self._norm1.wrappedValue = LayerNorm(dimensions: inner)
        self._attention.wrappedValue = MiniMaxMusic3FlowAttention(
            dimensions: inner,
            heads: configuration.numAttentionHeads,
            headDimension: configuration.attentionHeadDim,
            rotaryDimension: configuration.rotaryDim
        )
        self._norm2.wrappedValue = LayerNorm(dimensions: inner)
        self._feedForwardInput.wrappedValue = Linear(inner, configuration.ffInnerDim * 2)
        self._feedForwardOutput.wrappedValue = Linear(configuration.ffInnerDim, inner)
    }

    func callAsFunction(_ input: MLXArray, rotary: (MLXArray, MLXArray)) -> MLXArray {
        let attended = input + attention(norm1(input), rotary: rotary)
        let parts = MLX.split(feedForwardInput(norm2(attended)), parts: 2, axis: -1)
        return attended + feedForwardOutput(parts[0] * MLXNN.silu(parts[1]))
    }

    func prepareFusedProjections() {
        attention.prepareFusedProjections()
    }
}

public final class MiniMaxMusic3Transformer: Module {
    public let configuration: MiniMaxMusic3TransformerConfiguration

    @ModuleInfo(key: "time_proj") var timeProjection: MiniMaxMusic3FourierEmbedding
    @ModuleInfo(key: "time_embed") var timeEmbedding: MiniMaxMusic3TimestepEmbedding
    @ModuleInfo(key: "preprocess_conv") var preprocessConvolution: Conv1d
    @ModuleInfo(key: "proj_in") var inputProjection: Linear
    @ModuleInfo(key: "transformer_blocks") var blocks: [MiniMaxMusic3TransformerBlock]
    @ModuleInfo(key: "proj_out") var outputProjection: Linear
    @ModuleInfo(key: "postprocess_conv") var postprocessConvolution: Conv1d

    public init(configuration: MiniMaxMusic3TransformerConfiguration) {
        self.configuration = configuration
        let inner = configuration.numAttentionHeads * configuration.attentionHeadDim
        let concatenated = 2 * configuration.inChannels + configuration.conditionDim
        self._timeProjection.wrappedValue = MiniMaxMusic3FourierEmbedding(
            dimensions: configuration.fourierEmbeddingDim
        )
        self._timeEmbedding.wrappedValue = MiniMaxMusic3TimestepEmbedding(
            inputDimensions: configuration.fourierEmbeddingDim,
            outputDimensions: inner
        )
        self._preprocessConvolution.wrappedValue = Conv1d(
            inputChannels: concatenated,
            outputChannels: concatenated,
            kernelSize: 1,
            bias: false
        )
        self._inputProjection.wrappedValue = Linear(concatenated, inner, bias: false)
        self._blocks.wrappedValue = (0..<configuration.numLayers).map { _ in
            MiniMaxMusic3TransformerBlock(configuration: configuration)
        }
        self._outputProjection.wrappedValue = Linear(inner, configuration.inChannels, bias: false)
        self._postprocessConvolution.wrappedValue = Conv1d(
            inputChannels: configuration.inChannels,
            outputChannels: configuration.inChannels,
            kernelSize: 1,
            bias: false
        )
    }

    public func callAsFunction(
        latents: MLXArray,
        timestep: MLXArray,
        condition: MLXArray
    ) -> MLXArray {
        let latentNLC = latents.transposed(0, 2, 1)
        let zeros = MLXArray.zeros(latentNLC.shape, dtype: latentNLC.dtype)
        var hidden = MLX.concatenated([latentNLC, zeros, condition], axis: -1)
        hidden = preprocessConvolution(hidden) + hidden
        hidden = inputProjection(hidden)
        let time = timeEmbedding(timeProjection(timestep)).expandedDimensions(axis: 1)
        hidden = MLX.concatenated([time, hidden], axis: 1)
        let rotary = MiniMaxMusic3RotaryEmbedding.cache(
            length: hidden.dim(1),
            dimensions: configuration.rotaryDim,
            theta: 10_000,
            dtype: hidden.dtype
        )
        for block in blocks {
            hidden = block(hidden, rotary: rotary)
        }
        hidden = outputProjection(hidden[0..., 1..., 0...])
        let postprocessed = postprocessConvolution(hidden) + hidden
        return postprocessed.transposed(0, 2, 1)
    }

    func rotaryCache(
        latentLength: Int,
        dtype: DType
    ) -> (MLXArray, MLXArray) {
        MiniMaxMusic3RotaryEmbedding.cache(
            length: latentLength + 1,
            dimensions: configuration.rotaryDim,
            theta: 10_000,
            dtype: dtype
        )
    }

    func callAsFunction(
        latents: MLXArray,
        timestep: MLXArray,
        condition: MLXArray,
        rotary: (MLXArray, MLXArray)
    ) -> MLXArray {
        let latentNLC = latents.transposed(0, 2, 1)
        let zeros = MLXArray.zeros(latentNLC.shape, dtype: latentNLC.dtype)
        var hidden = MLX.concatenated([latentNLC, zeros, condition], axis: -1)
        hidden = preprocessConvolution(hidden) + hidden
        hidden = inputProjection(hidden)
        let time = timeEmbedding(timeProjection(timestep)).expandedDimensions(axis: 1)
        hidden = MLX.concatenated([time, hidden], axis: 1)
        for block in blocks {
            hidden = block(hidden, rotary: rotary)
        }
        hidden = outputProjection(hidden[0..., 1..., 0...])
        let postprocessed = postprocessConvolution(hidden) + hidden
        return postprocessed.transposed(0, 2, 1)
    }

    public func prepareFusedProjections() {
        for block in blocks {
            block.prepareFusedProjections()
        }
        MLX.Memory.clearCache()
    }

    static func mapWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key == "preprocess_conv.weight" || key == "postprocess_conv.weight" {
            return [(key, value.transposed(0, 2, 1))]
        }
        return [(key, value)]
    }
}
