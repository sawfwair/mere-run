import Foundation
import MLX
import MLXFast
import MLXNN

package final class LTXDistilledTransformer: Module {
    package let hiddenSize = 4096
    package let heads = 32
    package let headDim = 128
    package let outChannels = 128
    package let timestepScaleMultiplier: Float = 1000.0

    @ModuleInfo(key: "patchify_proj") package var patchifyProj: Linear
    @ModuleInfo(key: "adaln_single") package var adalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "caption_projection") package var captionProjection: LTXPixArtTextProjection
    @ModuleInfo(key: "scale_shift_table") package var scaleShiftTable: MLXArray
    @ModuleInfo(key: "norm_out") package var normOut: LayerNorm
    @ModuleInfo(key: "proj_out") package var projOut: Linear
    @ModuleInfo(key: "transformer_blocks") package var transformerBlocks: [LTXDistilledTransformerBlock]

    package override init() {
        self._patchifyProj.wrappedValue = Linear(128, 4096, bias: true)
        self._adalnSingle.wrappedValue = LTXAdaLayerNormSingle(embeddingDim: 4096, embeddingCoefficient: 6)
        self._captionProjection.wrappedValue = LTXPixArtTextProjection(inFeatures: 3840, hiddenSize: 4096, outFeatures: 4096, bias: true)
        self._scaleShiftTable.wrappedValue = MLX.zeros([2, 4096], dtype: .float32)
        self._normOut.wrappedValue = LayerNorm(dimensions: 4096, eps: 1e-6, affine: false)
        self._projOut.wrappedValue = Linear(4096, 128, bias: true)
        self._transformerBlocks.wrappedValue = (0..<48).map { _ in
            LTXDistilledTransformerBlock(dim: 4096, heads: 32, headDim: 128)
        }
        super.init()
    }

    package func forward(
        latent: MLXArray,
        timesteps: MLXArray,
        context: MLXArray,
        rope: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let batch = latent.dim(0)
        let tokenCount = latent.dim(1)

        var x = patchifyProj(latent)

        let scaledTimesteps = timesteps.asType(x.dtype) * MLXArray(timestepScaleMultiplier).asType(x.dtype)
        let (timeEmb, embeddedTime) = adalnSingle(timestep: scaledTimesteps.reshaped(-1), hiddenDType: x.dtype)
        let reshapedTimeEmb = timeEmb.reshaped(batch, tokenCount, -1)
        let reshapedEmbedded = embeddedTime.reshaped(batch, tokenCount, -1)

        let projectedContext = captionProjection(context).reshaped(batch, context.dim(1), hiddenSize)

        for block in transformerBlocks {
            x = block(
                x,
                context: projectedContext,
                timestepEmb: reshapedTimeEmb,
                rope: rope,
                contextMask: nil
            )
        }

        let timePairs = scaleShiftTable.reshaped(1, 1, 2, hiddenSize)
            + reshapedEmbedded.reshaped(batch, tokenCount, 1, hiddenSize)
        let shift = timePairs[0..., 0..., 0, 0...]
        let scale = timePairs[0..., 0..., 1, 0...]

        x = normOut(x)
        x = x * (MLXArray(1.0).asType(x.dtype) + scale) + shift
        return projOut(x)
    }
}

package final class LTXDistilledTransformerBlock: Module {
    package let dim: Int

    @ModuleInfo(key: "attn1") package var attn1: LTXDistilledAttention
    @ModuleInfo(key: "attn2") package var attn2: LTXDistilledAttention
    @ModuleInfo(key: "ff") package var ff: LTXDistilledFeedForward
    @ModuleInfo(key: "scale_shift_table") package var scaleShiftTable: MLXArray

    package init(dim: Int, heads: Int, headDim: Int) {
        self.dim = dim
        self._attn1.wrappedValue = LTXDistilledAttention(queryDim: dim, contextDim: nil, heads: heads, headDim: headDim, normEps: 1e-6)
        self._attn2.wrappedValue = LTXDistilledAttention(queryDim: dim, contextDim: dim, heads: heads, headDim: headDim, normEps: 1e-6)
        self._ff.wrappedValue = LTXDistilledFeedForward(dim: dim, dimOut: dim, mult: 4)
        self._scaleShiftTable.wrappedValue = MLX.zeros([6, dim], dtype: .float32)
    }

    package func callAsFunction(
        _ x: MLXArray,
        context: MLXArray,
        timestepEmb: MLXArray,
        rope: (cos: MLXArray, sin: MLXArray),
        contextMask: MLXArray?
    ) -> MLXArray {
        let batch = x.dim(0)
        let tokens = x.dim(1)

        let ada = scaleShiftTable.reshaped(1, 1, 6, dim) + timestepEmb.reshaped(batch, tokens, 6, dim)

        let shiftMSA = ada[0..., 0..., 0, 0...]
        let scaleMSA = ada[0..., 0..., 1, 0...]
        let gateMSA = ada[0..., 0..., 2, 0...]

        var h = rmsNormNoWeight(x)
        h = h * (MLXArray(1.0).asType(h.dtype) + scaleMSA) + shiftMSA
        h = attn1(h, context: nil, mask: nil, rope: rope)

        var out = x + h * gateMSA
        out = out + attn2(rmsNormNoWeight(out), context: context, mask: contextMask, rope: nil)

        let shiftMLP = ada[0..., 0..., 3, 0...]
        let scaleMLP = ada[0..., 0..., 4, 0...]
        let gateMLP = ada[0..., 0..., 5, 0...]

        var mlpInput = rmsNormNoWeight(out)
        mlpInput = mlpInput * (MLXArray(1.0).asType(mlpInput.dtype) + scaleMLP) + shiftMLP
        let mlpOut = ff(mlpInput)
        out = out + mlpOut * gateMLP

        return out
    }
}

package struct LTXAttentionProjectedContext {
    package let keys: MLXArray
    package let values: MLXArray
}

package final class LTXDistilledAttention: Module {
    package let heads: Int
    package let headDim: Int
    package let innerDim: Int

    @ModuleInfo(key: "to_q") package var toQ: Linear
    @ModuleInfo(key: "to_k") package var toK: Linear
    @ModuleInfo(key: "to_v") package var toV: Linear
    @ModuleInfo(key: "q_norm") package var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") package var kNorm: RMSNorm
    @ModuleInfo(key: "to_out") package var toOut: Linear
    @ModuleInfo(key: "to_gate_logits") package var toGateLogits: Linear?

    package init(
        queryDim: Int,
        contextDim: Int?,
        heads: Int,
        headDim: Int,
        normEps: Float,
        applyGatedAttention: Bool = false
    ) {
        self.heads = heads
        self.headDim = headDim
        self.innerDim = heads * headDim

        let effectiveContextDim = contextDim ?? queryDim

        self._toQ.wrappedValue = Linear(queryDim, innerDim, bias: true)
        self._toK.wrappedValue = Linear(effectiveContextDim, innerDim, bias: true)
        self._toV.wrappedValue = Linear(effectiveContextDim, innerDim, bias: true)
        self._qNorm.wrappedValue = RMSNorm(dimensions: innerDim, eps: normEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: innerDim, eps: normEps)
        self._toOut.wrappedValue = Linear(innerDim, queryDim, bias: true)
        self._toGateLogits.wrappedValue = applyGatedAttention ? Linear(queryDim, heads, bias: true) : nil
    }

    package func callAsFunction(
        _ x: MLXArray,
        context: MLXArray?,
        mask: MLXArray?,
        rope: (cos: MLXArray, sin: MLXArray)?,
        keyRope: (cos: MLXArray, sin: MLXArray)? = nil,
        projectedContext: LTXAttentionProjectedContext? = nil
    ) -> MLXArray {
        let q = qNorm(toQ(x))
        var qHeads = q.reshaped(q.dim(0), q.dim(1), heads, headDim).transposed(0, 2, 1, 3)
        var kHeads: MLXArray
        let vHeads: MLXArray
        if let projectedContext {
            kHeads = projectedContext.keys
            vHeads = projectedContext.values
        } else {
            let ctx = context ?? x
            let projection = projectContext(ctx, keyRope: keyRope ?? rope)
            kHeads = projection.keys
            vHeads = projection.values
        }

        if let rope {
            qHeads = applySplitRoPEHeads(qHeads, cosFreq: rope.cos, sinFreq: rope.sin)
        }

        var out = MLXFast.scaledDotProductAttention(
            queries: qHeads,
            keys: kHeads,
            values: vHeads,
            scale: 1.0 / Float(headDim).squareRoot(),
            mask: mask.map { .array($0) } ?? .none
        )

        if let toGateLogits {
            let gate = MLXArray(2.0).asType(out.dtype) * MLX.sigmoid(toGateLogits(x).asType(out.dtype))
            out = out * gate.transposed(0, 2, 1).expandedDimensions(axis: 3)
        }

        let merged = out.transposed(0, 2, 1, 3).reshaped(x.dim(0), x.dim(1), innerDim)
        return toOut(merged)
    }

    package func projectContext(
        _ context: MLXArray,
        keyRope: (cos: MLXArray, sin: MLXArray)? = nil
    ) -> LTXAttentionProjectedContext {
        let k = kNorm(toK(context))
        let v = toV(context)
        var keys = k.reshaped(k.dim(0), k.dim(1), heads, headDim).transposed(0, 2, 1, 3)
        if let keyRope {
            keys = applySplitRoPEHeads(keys, cosFreq: keyRope.cos, sinFreq: keyRope.sin)
        }
        let values = v.reshaped(v.dim(0), v.dim(1), heads, headDim).transposed(0, 2, 1, 3)
        return LTXAttentionProjectedContext(keys: keys, values: values)
    }
}

package final class LTXDistilledFeedForward: Module {
    @ModuleInfo(key: "proj_in") package var projIn: Linear
    @ModuleInfo(key: "proj_out") package var projOut: Linear

    package init(dim: Int, dimOut: Int, mult: Int = 4, bias: Bool = true) {
        let inner = dim * mult
        self._projIn.wrappedValue = Linear(dim, inner, bias: bias)
        self._projOut.wrappedValue = Linear(inner, dimOut, bias: bias)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        projOut(geluApproximate(projIn(x)))
    }
}

package final class LTXPixArtTextProjection: Module {
    @ModuleInfo(key: "linear1") package var linear1: Linear
    @ModuleInfo(key: "linear2") package var linear2: Linear

    package init(inFeatures: Int, hiddenSize: Int, outFeatures: Int?, bias: Bool = true) {
        let out = outFeatures ?? hiddenSize
        self._linear1.wrappedValue = Linear(inFeatures, hiddenSize, bias: bias)
        self._linear2.wrappedValue = Linear(hiddenSize, out, bias: bias)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(geluApproximate(linear1(x)))
    }
}

package final class LTXAdaLayerNormSingle: Module {
    @ModuleInfo(key: "emb") package var emb: LTXPixArtTimestepSizeEmbeddings
    @ModuleInfo(key: "linear") package var linear: Linear

    package init(embeddingDim: Int, embeddingCoefficient: Int) {
        self._emb.wrappedValue = LTXPixArtTimestepSizeEmbeddings(embeddingDim: embeddingDim)
        self._linear.wrappedValue = Linear(embeddingDim, embeddingCoefficient * embeddingDim, bias: true)
    }

    package func callAsFunction(timestep: MLXArray, hiddenDType: DType? = nil) -> (MLXArray, MLXArray) {
        let embedded = emb(timestep: timestep, hiddenDType: hiddenDType)
        let params = linear(silu(embedded))
        return (params, embedded)
    }
}

package final class LTXPixArtTimestepSizeEmbeddings: Module {
    @ModuleInfo(key: "timestep_embedder") package var timestepEmbedder: LTXTimestepEmbedding

    package init(embeddingDim: Int) {
        self._timestepEmbedder.wrappedValue = LTXTimestepEmbedding(inChannels: 256, timeEmbedDim: embeddingDim, outDim: embeddingDim)
    }

    package func callAsFunction(timestep: MLXArray, hiddenDType: DType?) -> MLXArray {
        var projected = getTimestepEmbedding(
            timesteps: timestep,
            embeddingDim: 256,
            flipSinToCos: true,
            downscaleFreqShift: 0,
            scale: 1,
            maxPeriod: 10_000
        )
        if let hiddenDType {
            projected = projected.asType(hiddenDType)
        }
        return timestepEmbedder(projected)
    }
}

package final class LTXTimestepEmbedding: Module {
    @ModuleInfo(key: "linear1") package var linear1: Linear
    @ModuleInfo(key: "linear2") package var linear2: Linear

    package init(inChannels: Int, timeEmbedDim: Int, outDim: Int) {
        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim, bias: true)
        self._linear2.wrappedValue = Linear(timeEmbedDim, outDim, bias: true)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}
