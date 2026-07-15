import Foundation
import MLX
import MLXFast
import MLXNN

@inline(__always)
private func krea2RMSNorm(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
    let dtype = x.dtype
    let xFloat = x.asType(.float32)
    let variance = (xFloat * xFloat).mean(axis: -1, keepDims: true)
    var normalized = xFloat / MLX.sqrt(variance + MLXArray(eps))
    let scale = (weight.asType(.float32) + MLXArray(1.0)).reshaped(
        Array(repeating: 1, count: max(0, x.ndim - 1)) + [weight.dim(0)]
    )
    normalized = normalized * scale
    return normalized.asType(dtype)
}

final class Krea2RMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    private let eps: Float

    init(dimensions: Int, eps: Float = 1e-5) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        krea2RMSNorm(x, weight: weight, eps: eps)
    }
}

final class Krea2SwiGLU: Module {
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "up") var up: Linear
    @ModuleInfo(key: "down") var down: Linear

    init(dim: Int, hiddenDim: Int, bias: Bool = false) {
        self._gate.wrappedValue = Linear(dim, hiddenDim, bias: bias)
        self._up.wrappedValue = Linear(dim, hiddenDim, bias: bias)
        self._down.wrappedValue = Linear(hiddenDim, dim, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(MLXNN.silu(gate(x)) * up(x))
    }
}

final class Krea2Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let hiddenSize: Int
    let scale: Float

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_gate") var toGate: Linear
    @ModuleInfo(key: "norm_q") var normQ: Krea2RMSNorm
    @ModuleInfo(key: "norm_k") var normK: Krea2RMSNorm
    @ModuleInfo(key: "to_out") var toOut: [Linear]

    init(dim: Int, heads: Int, kvHeads: Int, eps: Float, bias: Bool = false) {
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = dim / heads
        self.hiddenSize = dim
        self.scale = 1.0 / sqrt(Float(headDim))
        self._toQ.wrappedValue = Linear(dim, headDim * heads, bias: bias)
        self._toK.wrappedValue = Linear(dim, headDim * kvHeads, bias: bias)
        self._toV.wrappedValue = Linear(dim, headDim * kvHeads, bias: bias)
        self._toGate.wrappedValue = Linear(dim, dim, bias: bias)
        self._normQ.wrappedValue = Krea2RMSNorm(dimensions: headDim, eps: eps)
        self._normK.wrappedValue = Krea2RMSNorm(dimensions: headDim, eps: eps)
        self._toOut.wrappedValue = [Linear(dim, dim, bias: bias)]
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        rotary: (cos: MLXArray, sin: MLXArray)? = nil,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        var q = toQ(x).reshaped(batch, sequence, heads, headDim).transposed(0, 2, 1, 3)
        var k = toK(x).reshaped(batch, sequence, kvHeads, headDim).transposed(0, 2, 1, 3)
        var v = toV(x).reshaped(batch, sequence, kvHeads, headDim).transposed(0, 2, 1, 3)
        let gate = MLX.sigmoid(toGate(x))

        q = normQ(q)
        k = normK(k)
        if let rotary {
            q = Krea2PositionalEncoding.applyRotary(q, rotary: rotary)
            k = Krea2PositionalEncoding.applyRotary(k, rotary: rotary)
        }
        if kvHeads < heads {
            let repeats = heads / kvHeads
            k = MLX.repeated(k, count: repeats, axis: 1)
            v = MLX.repeated(v, count: repeats, axis: 1)
        }

        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )
        let output = attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, hiddenSize)
        return toOut[0](output * gate)
    }
}

final class Krea2TextFusionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: Krea2RMSNorm
    @ModuleInfo(key: "norm2") var norm2: Krea2RMSNorm
    @ModuleInfo(key: "attn") var attn: Krea2Attention
    @ModuleInfo(key: "ff") var ff: Krea2SwiGLU

    init(dim: Int, heads: Int, kvHeads: Int, hiddenDim: Int, eps: Float) {
        self._norm1.wrappedValue = Krea2RMSNorm(dimensions: dim, eps: eps)
        self._norm2.wrappedValue = Krea2RMSNorm(dimensions: dim, eps: eps)
        self._attn.wrappedValue = Krea2Attention(dim: dim, heads: heads, kvHeads: kvHeads, eps: eps)
        self._ff.wrappedValue = Krea2SwiGLU(dim: dim, hiddenDim: hiddenDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var out = x + attn(norm1(x), mask: mask)
        out = out + ff(norm2(out))
        return out
    }

    private var checkpointedWithoutMask: (([MLXArray]) -> [MLXArray])?
    private var checkpointedWithMask: (([MLXArray]) -> [MLXArray])?

    func checkpointed(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        if let mask {
            if checkpointedWithMask == nil {
                checkpointedWithMask = checkpoint(model: self) { block, inputs in
                    [block(inputs[0], mask: stopGradient(inputs[1]))]
                }
            }
            guard let checkpointedWithMask else {
                preconditionFailure("Checkpointed Krea 2 text block was not initialized.")
            }
            return checkpointedWithMask([x, mask])[0]
        }

        if checkpointedWithoutMask == nil {
            checkpointedWithoutMask = checkpoint(model: self) { block, inputs in
                [block(inputs[0])]
            }
        }
        guard let checkpointedWithoutMask else {
            preconditionFailure("Checkpointed Krea 2 text block was not initialized.")
        }
        return checkpointedWithoutMask([x])[0]
    }
}

final class Krea2TextFusionTransformer: Module {
    @ModuleInfo(key: "layerwise_blocks") var layerwiseBlocks: [Krea2TextFusionBlock]
    @ModuleInfo(key: "projector") var projector: Linear
    @ModuleInfo(key: "refiner_blocks") var refinerBlocks: [Krea2TextFusionBlock]

    init(configuration: Krea2TransformerConfiguration) {
        self._layerwiseBlocks.wrappedValue = (0..<configuration.numLayerwiseTextBlocks).map { _ in
            Krea2TextFusionBlock(
                dim: configuration.textHiddenDim,
                heads: configuration.textNumAttentionHeads,
                kvHeads: configuration.textNumKeyValueHeads,
                hiddenDim: configuration.textIntermediateSize,
                eps: configuration.normEps
            )
        }
        self._projector.wrappedValue = Linear(configuration.numTextLayers, 1, bias: false)
        self._refinerBlocks.wrappedValue = (0..<configuration.numRefinerTextBlocks).map { _ in
            Krea2TextFusionBlock(
                dim: configuration.textHiddenDim,
                heads: configuration.textNumAttentionHeads,
                kvHeads: configuration.textNumKeyValueHeads,
                hiddenDim: configuration.textIntermediateSize,
                eps: configuration.normEps
            )
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        gradientCheckpointing: Bool = false
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let layerCount = x.dim(2)
        let dim = x.dim(3)
        var hidden = x.reshaped(batch * sequence, layerCount, dim)
        for block in layerwiseBlocks {
            hidden = gradientCheckpointing ? block.checkpointed(hidden) : block(hidden)
        }
        hidden = hidden.reshaped(batch, sequence, layerCount, dim)
            .transposed(0, 1, 3, 2)
        hidden = projector(hidden).squeezed(axis: -1)
        for block in refinerBlocks {
            hidden = gradientCheckpointing
                ? block.checkpointed(hidden, mask: mask)
                : block(hidden, mask: mask)
        }
        return hidden
    }
}

final class Krea2TextProjection: Module {
    @ModuleInfo(key: "norm") var norm: Krea2RMSNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(textDim: Int, hiddenSize: Int, eps: Float) {
        self._norm.wrappedValue = Krea2RMSNorm(dimensions: textDim, eps: eps)
        self._linear1.wrappedValue = Linear(textDim, hiddenSize, bias: true)
        self._linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(MLXNN.geluApproximate(linear1(norm(x))))
    }
}

final class Krea2TimeEmbed: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let embeddingDim: Int

    init(embeddingDim: Int, hiddenSize: Int) {
        self.embeddingDim = embeddingDim
        self._linear1.wrappedValue = Linear(embeddingDim, hiddenSize, bias: true)
        self._linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray, dtype: DType) -> MLXArray {
        let projected = Self.timestepEmbedding(timestep, dim: embeddingDim).asType(dtype)
        return linear2(MLXNN.geluApproximate(linear1(projected)))
    }

    static func timestepEmbedding(_ timestep: MLXArray, dim: Int) -> MLXArray {
        let half = dim / 2
        let freqs = MLX.exp(
            MLXArray(0..<half).asType(.float32) * MLXArray(-Foundation.log(10_000.0) / Float(half))
        )
        let args = timestep.asType(.float32).reshaped(timestep.dim(0), 1, 1) * MLXArray(1_000.0) * freqs
        var embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
        embedding = embedding.squeezed(axis: 1)
        if dim % 2 == 1 {
            embedding = MLX.concatenated([embedding, MLX.zeros([embedding.dim(0), 1], dtype: .float32)], axis: -1)
        }
        return embedding
    }
}

final class Krea2SingleStreamBlock: Module {
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "norm1") var norm1: Krea2RMSNorm
    @ModuleInfo(key: "norm2") var norm2: Krea2RMSNorm
    @ModuleInfo(key: "attn") var attn: Krea2Attention
    @ModuleInfo(key: "ff") var ff: Krea2SwiGLU

    let hiddenSize: Int

    init(configuration: Krea2TransformerConfiguration) {
        self.hiddenSize = configuration.hiddenSize
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([6, configuration.hiddenSize])
        self._norm1.wrappedValue = Krea2RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.normEps)
        self._norm2.wrappedValue = Krea2RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.normEps)
        self._attn.wrappedValue = Krea2Attention(
            dim: configuration.hiddenSize,
            heads: configuration.numAttentionHeads,
            kvHeads: configuration.numKeyValueHeads,
            eps: configuration.normEps
        )
        self._ff.wrappedValue = Krea2SwiGLU(
            dim: configuration.hiddenSize,
            hiddenDim: configuration.intermediateSize
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        modulation: MLXArray,
        rotary: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let table = scaleShiftTable.reshaped(1, 6, hiddenSize)
        let values = (modulation.reshaped(batch, 6, hiddenSize) + table).expandedDimensions(axis: 1)
        let prescale = values[0..., 0..., 0, 0...]
        let preshift = values[0..., 0..., 1, 0...]
        let pregate = values[0..., 0..., 2, 0...]
        let postscale = values[0..., 0..., 3, 0...]
        let postshift = values[0..., 0..., 4, 0...]
        let postgate = values[0..., 0..., 5, 0...]

        var out = x + pregate * attn((1 + prescale) * norm1(x) + preshift, rotary: rotary, mask: mask)
        out = out + postgate * ff((1 + postscale) * norm2(out) + postshift)
        return out
    }

    private var checkpointedForward: (([MLXArray]) -> [MLXArray])?

    /// Gradient-checkpointed block forward: activations inside the block are
    /// recomputed during backward instead of retained, trading ~one extra
    /// block forward for a per-block activation footprint.
    func checkpointed(
        _ x: MLXArray,
        modulation: MLXArray,
        rotary: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray
    ) -> MLXArray {
        if checkpointedForward == nil {
            checkpointedForward = checkpoint(model: self) { block, inputs in
                [block(
                    inputs[0],
                    modulation: inputs[1],
                    rotary: (cos: stopGradient(inputs[2]), sin: stopGradient(inputs[3])),
                    mask: stopGradient(inputs[4])
                )]
            }
        }
        guard let checkpointedForward else {
            preconditionFailure("Checkpointed Krea 2 block was not initialized.")
        }
        return checkpointedForward([x, modulation, rotary.cos, rotary.sin, mask])[0]
    }
}

final class Krea2FinalLayer: Module {
    @ModuleInfo(key: "norm") var norm: Krea2RMSNorm
    @ModuleInfo(key: "linear") var linear: Linear
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    let hiddenSize: Int

    init(hiddenSize: Int, outChannels: Int, eps: Float) {
        self.hiddenSize = hiddenSize
        self._norm.wrappedValue = Krea2RMSNorm(dimensions: hiddenSize, eps: eps)
        self._linear.wrappedValue = Linear(hiddenSize, outChannels, bias: true)
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([2, hiddenSize])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, timestepEmbedding: MLXArray) -> MLXArray {
        let batch = timestepEmbedding.dim(0)
        let values = timestepEmbedding.reshaped(batch, 1, hiddenSize)
            + scaleShiftTable.reshaped(1, 2, hiddenSize)
        let scale = values[0..., 0..<1, 0...]
        let shift = values[0..., 1..<2, 0...]
        return linear((1 + scale) * norm(x) + shift)
    }
}

final class Krea2PositionalEncoding {
    let axesDims: [Int]
    let theta: Float

    init(axesDims: [Int], theta: Float) {
        self.axesDims = axesDims
        self.theta = theta
    }

    func callAsFunction(positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let pos = positionIds.asType(.float32)
        var cosParts: [MLXArray] = []
        var sinParts: [MLXArray] = []
        for axis in 0..<axesDims.count {
            let dim = axesDims[axis]
            let axisPos = pos[0..., 0..., axis]
            let scale = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32) / Float(dim)
            let omega = 1.0 / MLX.pow(MLXArray(theta), scale)
            let angles = axisPos.expandedDimensions(axis: -1) * omega
            cosParts.append(MLX.cos(angles))
            sinParts.append(MLX.sin(angles))
        }
        return (
            MLX.concatenated(cosParts, axis: -1),
            MLX.concatenated(sinParts, axis: -1)
        )
    }

    static func applyRotary(
        _ x: MLXArray,
        rotary: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let outDtype = x.dtype
        let xFloat = x.asType(.float32)
        let shape = xFloat.shape
        let half = shape[3] / 2
        let paired = xFloat.reshaped(shape[0], shape[1], shape[2], half, 2)
        let real = paired[0..., 0..., 0..., 0..., 0]
        let imaginary = paired[0..., 0..., 0..., 0..., 1]
        let cosValues = rotary.cos.expandedDimensions(axis: 1)
        let sinValues = rotary.sin.expandedDimensions(axis: 1)
        let outReal = real * cosValues - imaginary * sinValues
        let outImaginary = real * sinValues + imaginary * cosValues
        return MLX.stacked([outReal, outImaginary], axis: -1).reshaped(shape).asType(outDtype)
    }
}

public final class Krea2Transformer: Module {
    public let configuration: Krea2TransformerConfiguration

    @ModuleInfo(key: "img_in") var imgIn: Linear
    @ModuleInfo(key: "txt_in") var txtIn: Krea2TextProjection
    @ModuleInfo(key: "text_fusion") var textFusion: Krea2TextFusionTransformer
    @ModuleInfo(key: "time_embed") var timeEmbed: Krea2TimeEmbed
    @ModuleInfo(key: "time_mod_proj") var timeModProj: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [Krea2SingleStreamBlock]
    @ModuleInfo(key: "final_layer") var finalLayer: Krea2FinalLayer

    /// Set by trainers: routes each transformer block through the
    /// gradient-checkpointed forward so backward recomputes activations
    /// instead of retaining them.
    public var gradientCheckpointing = false

    private let positionalEncoding: Krea2PositionalEncoding

    public init(configuration: Krea2TransformerConfiguration) {
        self.configuration = configuration
        self._imgIn.wrappedValue = Linear(configuration.inChannels, configuration.hiddenSize, bias: true)
        self._txtIn.wrappedValue = Krea2TextProjection(
            textDim: configuration.textHiddenDim,
            hiddenSize: configuration.hiddenSize,
            eps: configuration.normEps
        )
        self._textFusion.wrappedValue = Krea2TextFusionTransformer(configuration: configuration)
        self._timeEmbed.wrappedValue = Krea2TimeEmbed(
            embeddingDim: configuration.timestepEmbedDim,
            hiddenSize: configuration.hiddenSize
        )
        self._timeModProj.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize * 6, bias: true)
        self._transformerBlocks.wrappedValue = (0..<configuration.numLayers).map { _ in
            Krea2SingleStreamBlock(configuration: configuration)
        }
        self._finalLayer.wrappedValue = Krea2FinalLayer(
            hiddenSize: configuration.hiddenSize,
            outChannels: configuration.inChannels,
            eps: configuration.normEps
        )
        self.positionalEncoding = Krea2PositionalEncoding(
            axesDims: configuration.axesDimsRope,
            theta: configuration.ropeTheta
        )
        super.init()
    }

    public func callAsFunction(
        imageTokens: MLXArray,
        textContext: MLXArray,
        timestep: MLXArray,
        positionIds: MLXArray,
        validMask: MLXArray
    ) -> MLXArray {
        let textLength = textContext.dim(1)
        let imageLength = imageTokens.dim(1)
        let batch = imageTokens.dim(0)
        let imageHidden = imgIn(imageTokens)
        let timeEmbedding = timeEmbed(timestep, dtype: imageHidden.dtype)
        let timeModulation = timeModProj(MLXNN.geluApproximate(timeEmbedding))

        let textMask = validMask[0..., 0..<textLength]
        let textAttentionMask = Krea2SampleBuilder.attentionMask(
            validMask: textMask,
            dtype: imageHidden.dtype
        )
        let textHidden = txtIn(textFusion(
            textContext,
            mask: textAttentionMask,
            gradientCheckpointing: gradientCheckpointing
        ))
        var combined = MLX.concatenated([textHidden, imageHidden], axis: 1)
        var combinedMask = validMask
        var combinedPositions = positionIds
        let fullLength = combined.dim(1)
        let padLength = (256 - (fullLength % 256)) % 256
        if padLength > 0 {
            combined = MLX.concatenated([
                combined,
                MLX.zeros([batch, padLength, configuration.hiddenSize], dtype: combined.dtype),
            ], axis: 1)
            combinedMask = MLX.concatenated([
                combinedMask,
                MLX.zeros([batch, padLength], dtype: .int32),
            ], axis: 1)
            combinedPositions = MLX.concatenated([
                combinedPositions,
                MLX.zeros([batch, padLength, 3], dtype: combinedPositions.dtype),
            ], axis: 1)
        }

        let attentionMask = Krea2SampleBuilder.attentionMask(validMask: combinedMask, dtype: combined.dtype)
        let tokenMask = (combinedMask .== MLXArray(Int32(1))).expandedDimensions(axis: -1)
        let rotary = positionalEncoding(positionIds: combinedPositions)
        for block in transformerBlocks {
            combined = MLX.where(tokenMask, combined, MLX.zeros(combined.shape, dtype: combined.dtype))
            combined = gradientCheckpointing
                ? block.checkpointed(combined, modulation: timeModulation, rotary: rotary, mask: attentionMask)
                : block(combined, modulation: timeModulation, rotary: rotary, mask: attentionMask)
        }
        combined = MLX.where(tokenMask, combined, MLX.zeros(combined.shape, dtype: combined.dtype))
        let output = finalLayer(combined, timestepEmbedding: timeEmbedding)
        return output[0..., textLength..<(textLength + imageLength), 0...].asType(.float32)
    }
}
