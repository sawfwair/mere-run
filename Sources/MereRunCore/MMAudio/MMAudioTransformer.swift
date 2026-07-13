import Foundation
import MLX
import MLXNN
import MLXRandom

final class MMAudioAdaLN: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    private let chunks: Int

    init(dimensions: Int, chunks: Int) {
        self.chunks = chunks
        self._linear.wrappedValue = Linear(dimensions, dimensions * chunks, bias: true)
    }

    func callAsFunction(_ condition: MLXArray) -> [MLXArray] {
        MLX.split(linear(MLXNN.silu(condition)), parts: chunks, axis: -1)
    }
}

struct MMAudioPostAttentionModulation {
    let gateAttention: MLXArray?
    let shiftMLP: MLXArray?
    let scaleMLP: MLXArray?
    let gateMLP: MLXArray?
}

struct MMAudioAttentionProjection {
    let query: MLXArray
    let key: MLXArray
    let value: MLXArray
    let modulation: MMAudioPostAttentionModulation
}

final class MMAudioSingleBlock: Module {
    @ModuleInfo(key: "attn") var attention: MMAudioSelfAttention
    @ModuleInfo(key: "linear1") var linear1: MMAudioDenseOrConv1D?
    @ModuleInfo(key: "ffn") var feedForward: MMAudioConvMLP?
    @ModuleInfo(key: "adaLN_modulation") var adaLN: MMAudioAdaLN

    private let preOnly: Bool

    init(dimensions: Int, heads: Int, preOnly: Bool, kernelSize: Int, padding: Int) {
        self.preOnly = preOnly
        self._attention.wrappedValue = MMAudioSelfAttention(dimensions: dimensions, heads: heads)
        self._adaLN.wrappedValue = MMAudioAdaLN(dimensions: dimensions, chunks: preOnly ? 2 : 6)
        self._linear1.wrappedValue = preOnly ? nil : MMAudioDenseOrConv1D(
            inputChannels: dimensions,
            outputChannels: dimensions,
            kernelSize: kernelSize,
            padding: padding,
            bias: true
        )
        self._feedForward.wrappedValue = preOnly ? nil : MMAudioConvMLP(
            dimensions: dimensions,
            requestedHiddenDimensions: dimensions * 4,
            kernelSize: kernelSize,
            padding: padding
        )
    }

    func projectAttention(_ x: MLXArray, condition: MLXArray, rope: MMAudioRoPE?) -> MMAudioAttentionProjection {
        let modulation = adaLN(condition)
        let shiftAttention = modulation[0]
        let scaleAttention = modulation[1]
        let normalized = MMAudioTensorOps.layerNorm(x)
        let modulated = normalized * (1 + scaleAttention) + shiftAttention
        let projected = attention.project(modulated, rope: rope)
        return MMAudioAttentionProjection(
            query: projected.0,
            key: projected.1,
            value: projected.2,
            modulation: MMAudioPostAttentionModulation(
                gateAttention: preOnly ? nil : modulation[2],
                shiftMLP: preOnly ? nil : modulation[3],
                scaleMLP: preOnly ? nil : modulation[4],
                gateMLP: preOnly ? nil : modulation[5]
            )
        )
    }

    func finishAttention(
        _ residual: MLXArray,
        attended: MLXArray,
        modulation: MMAudioPostAttentionModulation
    ) -> MLXArray {
        if preOnly {
            return residual
        }
        let gateAttention = modulation.gateAttention!
        let shiftMLP = modulation.shiftMLP!
        let scaleMLP = modulation.scaleMLP!
        let gateMLP = modulation.gateMLP!
        var output = residual + linear1!(attended) * gateAttention
        let normalized = MMAudioTensorOps.layerNorm(output)
        let modulated = normalized * (1 + scaleMLP) + shiftMLP
        output = output + feedForward!(modulated) * gateMLP
        return output
    }

    func callAsFunction(_ x: MLXArray, condition: MLXArray, rope: MMAudioRoPE?) -> MLXArray {
        let projection = projectAttention(x, condition: condition, rope: rope)
        let attended = attention.attend(
            query: projection.query,
            key: projection.key,
            value: projection.value
        )
        return finishAttention(x, attended: attended, modulation: projection.modulation)
    }
}

final class MMAudioJointBlock: Module {
    @ModuleInfo(key: "latent_block") var latentBlock: MMAudioSingleBlock
    @ModuleInfo(key: "clip_block") var clipBlock: MMAudioSingleBlock
    @ModuleInfo(key: "text_block") var textBlock: MMAudioSingleBlock

    private let preOnly: Bool

    init(dimensions: Int, heads: Int, preOnly: Bool) {
        self.preOnly = preOnly
        self._latentBlock.wrappedValue = MMAudioSingleBlock(
            dimensions: dimensions,
            heads: heads,
            preOnly: false,
            kernelSize: 3,
            padding: 1
        )
        self._clipBlock.wrappedValue = MMAudioSingleBlock(
            dimensions: dimensions,
            heads: heads,
            preOnly: preOnly,
            kernelSize: 3,
            padding: 1
        )
        self._textBlock.wrappedValue = MMAudioSingleBlock(
            dimensions: dimensions,
            heads: heads,
            preOnly: preOnly,
            kernelSize: 1,
            padding: 0
        )
    }

    func callAsFunction(
        latent: MLXArray,
        clip: MLXArray,
        text: MLXArray,
        globalCondition: MLXArray,
        extendedCondition: MLXArray,
        latentRoPE: MMAudioRoPE,
        clipRoPE: MMAudioRoPE
    ) -> (MLXArray, MLXArray, MLXArray) {
        let latentProjection = latentBlock.projectAttention(
            latent,
            condition: extendedCondition,
            rope: latentRoPE
        )
        let clipProjection = clipBlock.projectAttention(
            clip,
            condition: globalCondition,
            rope: clipRoPE
        )
        let textProjection = textBlock.projectAttention(
            text,
            condition: globalCondition,
            rope: nil
        )

        let query = MLX.concatenated([
            latentProjection.query,
            clipProjection.query,
            textProjection.query,
        ], axis: 2)
        let key = MLX.concatenated([
            latentProjection.key,
            clipProjection.key,
            textProjection.key,
        ], axis: 2)
        let value = MLX.concatenated([
            latentProjection.value,
            clipProjection.value,
            textProjection.value,
        ], axis: 2)
        let attended = latentBlock.attention.attend(query: query, key: key, value: value)

        let latentLength = latent.dim(1)
        let clipLength = clip.dim(1)
        let latentAttended = attended[0..., 0..<latentLength, 0...]
        let clipAttended = attended[0..., latentLength..<(latentLength + clipLength), 0...]
        let textAttended = attended[0..., (latentLength + clipLength)..., 0...]
        let nextLatent = latentBlock.finishAttention(
            latent,
            attended: latentAttended,
            modulation: latentProjection.modulation
        )
        guard !preOnly else {
            return (nextLatent, clip, text)
        }
        return (
            nextLatent,
            clipBlock.finishAttention(clip, attended: clipAttended, modulation: clipProjection.modulation),
            textBlock.finishAttention(text, attended: textAttended, modulation: textProjection.modulation)
        )
    }
}

final class MMAudioFinalBlock: Module {
    @ModuleInfo(key: "adaLN_modulation") var adaLN: MMAudioAdaLN
    @ModuleInfo(key: "conv") var conv: MMAudioDenseOrConv1D

    init(dimensions: Int, outputDimensions: Int) {
        self._adaLN.wrappedValue = MMAudioAdaLN(dimensions: dimensions, chunks: 2)
        self._conv.wrappedValue = MMAudioDenseOrConv1D(
            inputChannels: dimensions,
            outputChannels: outputDimensions,
            kernelSize: 7,
            padding: 3,
            bias: true
        )
    }

    func callAsFunction(_ latent: MLXArray, condition: MLXArray) -> MLXArray {
        let modulation = adaLN(condition)
        let normalized = MMAudioTensorOps.layerNorm(latent)
        return conv(normalized * (1 + modulation[1]) + modulation[0])
    }
}

final class MMAudioTimestepEmbedder: Module {
    @ModuleInfo(key: "mlp") var mlp: MMAudioTimestepMLP
    @ParameterInfo(key: "freqs") var frequencies: MLXArray

    init(dimensions: Int) {
        let half = dimensions / 2
        self._frequencies.wrappedValue = MLXArray((0..<half).map { index in
            10_000 * pow(10_000, -Float(index * 2) / Float(dimensions))
        })
        self._mlp.wrappedValue = MMAudioTimestepMLP(dimensions: dimensions)
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let angles = timestep.asType(.float32).expandedDimensions(axis: 1) * frequencies.expandedDimensions(axis: 0)
        let embedding = MLX.concatenated([MLX.cos(angles), MLX.sin(angles)], axis: -1).asType(timestep.dtype)
        return mlp(embedding)
    }
}

final class MMAudioTimestepMLP: Module {
    @ModuleInfo(key: "first") var first: Linear
    @ModuleInfo(key: "second") var second: Linear

    init(dimensions: Int) {
        self._first.wrappedValue = Linear(dimensions, dimensions, bias: true)
        self._second.wrappedValue = Linear(dimensions, dimensions, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        second(MLXNN.silu(first(x)))
    }
}
