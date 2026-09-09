import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunKVCache

final class DecoderRotaryEmbedding {
    let dim: Int
    let maxPositionEmbeddings: Int
    let base: Float
    let invFreq: MLXArray

    init(dim: Int, maxPositionEmbeddings: Int, base: Float) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base
        let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
        self.invFreq = Float(1.0) / MLX.pow(MLXArray(base), indices / Float(dim))
    }

    func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let invFreqExpanded = invFreq[.newAxis, 0..., .newAxis].asType(.float32)
        let posExpanded = positionIds[0..., .newAxis, 0...].asType(.float32)
        let freqs = (invFreqExpanded * posExpanded).transposed(0, 2, 1)
        let emb = MLX.concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb).asType(x.dtype), MLX.sin(emb).asType(x.dtype))
    }
}

private func rotateHalfDecoder(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0..<half]
    let x2 = x[.ellipsis, half...]
    return MLX.concatenated([-x2, x1], axis: -1)
}

func applyDecoderRotary(_ q: MLXArray, _ k: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
    let cosExp = cos[0..., .newAxis, 0..., 0...]
    let sinExp = sin[0..., .newAxis, 0..., 0...]
    let qEmbed = (q * cosExp) + (rotateHalfDecoder(q) * sinExp)
    let kEmbed = (k * cosExp) + (rotateHalfDecoder(k) * sinExp)
    return (qEmbed, kEmbed)
}

final class DecoderAttention: Module {
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        self.headDim = config.headDim
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.scale = 1.0 / Float(headDim).squareRoot()

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: config.attentionBias)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, seqLen, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(batch, seqLen, numKVHeads, headDim).transposed(0, 2, 1, 3)

        (q, k) = applyDecoderRotary(q, k, cos: positionEmbeddings.cos, sin: positionEmbeddings.sin)

        if let cache {
            let updated = cache.update(keys: k, values: v)
            k = updated.0
            v = updated.1
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        output = output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)
        return oProj(output)
    }
}

final class DecoderMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Qwen3TTSTokenizerDecoderConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(MLXNN.silu(gateProj(x)) * upProj(x))
    }
}

final class DecoderTransformerLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DecoderAttention
    @ModuleInfo(key: "mlp") var mlp: DecoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: DecoderRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: DecoderRMSNorm
    @ModuleInfo(key: "self_attn_layer_scale") var selfAttnLayerScale: LayerScale
    @ModuleInfo(key: "mlp_layer_scale") var mlpLayerScale: LayerScale

    init(config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = DecoderAttention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = DecoderMLP(config: config)
        self._inputLayerNorm.wrappedValue = DecoderRMSNorm(hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = DecoderRMSNorm(hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
        self._selfAttnLayerScale.wrappedValue = LayerScale(channels: config.hiddenSize, initialScale: config.layerScaleInitialScale)
        self._mlpLayerScale.wrappedValue = LayerScale(channels: config.hiddenSize, initialScale: config.layerScaleInitialScale)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var hidden = x
        let residual1 = hidden
        hidden = inputLayerNorm(hidden)
        hidden = selfAttn(hidden, positionEmbeddings: positionEmbeddings, mask: mask, cache: cache)
        hidden = residual1 + selfAttnLayerScale(hidden)

        let residual2 = hidden
        hidden = postAttentionLayerNorm(hidden)
        hidden = mlp(hidden)
        hidden = residual2 + mlpLayerScale(hidden)
        return hidden
    }
}

final class DecoderTransformer: Module {
    let config: Qwen3TTSTokenizerDecoderConfig

    @ModuleInfo(key: "layers") var layers: [DecoderTransformerLayer]
    @ModuleInfo(key: "norm") var norm: DecoderRMSNorm
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "output_proj") var outputProj: Linear

    let rotaryEmb: DecoderRotaryEmbedding

    init(config: Qwen3TTSTokenizerDecoderConfig) {
        self.config = config
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { DecoderTransformerLayer(config: config, layerIdx: $0) }
        self._norm.wrappedValue = DecoderRMSNorm(hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
        self._inputProj.wrappedValue = Linear(config.latentDim, config.hiddenSize)
        self._outputProj.wrappedValue = Linear(config.hiddenSize, config.latentDim)
        self.rotaryEmb = DecoderRotaryEmbedding(dim: config.headDim, maxPositionEmbeddings: config.maxPositionEmbeddings, base: config.ropeTheta)
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let batch = inputsEmbeds.dim(0)
        let seqLen = inputsEmbeds.dim(1)

        var hidden = inputProj(inputsEmbeds)

        let offset = cache?.first?.offset ?? 0
        let base = MLXArray(Int32(offset)..<Int32(offset + seqLen)).reshaped(1, seqLen)
        let pos = broadcast(base, to: [batch, seqLen])
        let positionEmbeddings = rotaryEmb(hidden, positionIds: pos)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = {
            if let mask { return mask }
            if let cache, let first = cache.first { return first.makeMask(n: seqLen) }
            return seqLen == 1 ? .none : .causal
        }()

        for (idx, layer) in layers.enumerated() {
            let layerCache = cache?[idx]
            hidden = layer(hidden, positionEmbeddings: positionEmbeddings, mask: maskMode, cache: layerCache)
        }

        hidden = norm(hidden)
        hidden = outputProj(hidden)
        return hidden
    }

    func makeCache() -> [KVCache] {
        (0..<config.numHiddenLayers).map { _ in KVCacheSimple(step: 256) }
    }
}

// MARK: - Quantizers

final class Qwen3TTSSpeechTokenizerDecoder: Module {
    let config: Qwen3TTSTokenizerDecoderConfig
    let totalUpsample: Int

    @ModuleInfo(key: "pre_transformer") var preTransformer: DecoderTransformer
    @ModuleInfo(key: "quantizer") var quantizer: SplitResidualVectorQuantizer
    @ModuleInfo(key: "pre_conv") var preConv: CausalConv1d
    @ModuleInfo(key: "upsample") var upsample: [[AudioDecoderLayer]]
    @ModuleInfo(key: "decoder") var decoder: [AudioDecoderLayer]

    init(config: Qwen3TTSTokenizerDecoderConfig) {
        self.config = config
        self.totalUpsample = (config.upsampleRates + config.upsamplingRatios).reduce(1, *)

        self._preTransformer.wrappedValue = DecoderTransformer(config: config)
        self._quantizer.wrappedValue = SplitResidualVectorQuantizer(
            nQ: config.numQuantizers,
            nQSemantic: config.numSemanticQuantizers,
            dimension: config.codebookDim / 2,
            inputDimension: config.codebookDim,
            outputDimension: config.codebookDim,
            bins: config.codebookSize
        )
        self._preConv.wrappedValue = CausalConv1d(inChannels: config.codebookDim, outChannels: config.latentDim, kernelSize: 3)

        self._upsample.wrappedValue = config.upsamplingRatios.map { factor in
            [
                CausalTransposeConv1d(inChannels: config.latentDim, outChannels: config.latentDim, kernelSize: factor, stride: factor),
                ConvNeXtBlock(dim: config.latentDim)
            ]
        }

        let outputDim = config.decoderDim / (1 << config.upsampleRates.count)
        var decoderLayers: [AudioDecoderLayer] = []
        decoderLayers.append(DecoderInitialConv(latentDim: config.latentDim, decoderDim: config.decoderDim, kernelSize: 7))
        for idx in 0..<config.upsampleRates.count {
            decoderLayers.append(Qwen3TTSSpeechDecoderBlock(config: config, layerIdx: idx))
        }
        decoderLayers.append(DecoderOutputSnake(channels: outputDim))
        decoderLayers.append(DecoderOutputConv(channels: outputDim, kernelSize: 7))
        self._decoder.wrappedValue = decoderLayers
    }

    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        if codes.dim(1) != config.numQuantizers {
            return MLXArray.zeros([codes.dim(0), 1, 0])
        }

        var hidden = quantizer.decode(codes)
        hidden = preConv(hidden)
        hidden = hidden.transposed(0, 2, 1)
        hidden = preTransformer(hidden)
        hidden = hidden.transposed(0, 2, 1)

        for upsampleLayers in upsample {
            for layer in upsampleLayers {
                hidden = layer(hidden)
            }
        }

        var wav = hidden
        for layer in decoder {
            wav = layer(wav)
        }

        return MLX.clip(wav, min: -1.0, max: 1.0)
    }

    func chunkedDecode(codes: MLXArray, chunkSize: Int = 300, leftContextSize: Int = 25) -> MLXArray {
        var wavs: [MLXArray] = []
        var startIndex = 0
        let totalTokens = codes.dim(2)

        while startIndex < totalTokens {
            let endIndex = min(startIndex + chunkSize, totalTokens)
            let context = startIndex - leftContextSize > 0 ? leftContextSize : startIndex
            let chunk = codes[0..., 0..., (startIndex - context)..<endIndex]
            var wavChunk = callAsFunction(chunk)
            wavChunk = wavChunk[0..., 0..., (context * totalUpsample)..<wavChunk.dim(2)]
            MLX.eval(wavChunk)
            wavs.append(wavChunk)
            Memory.clearCache()
            startIndex = endIndex
        }

        if wavs.isEmpty {
            return MLXArray.zeros([codes.dim(0), 1, 0])
        }

        return MLX.concatenated(wavs, axis: 2)
    }
}
