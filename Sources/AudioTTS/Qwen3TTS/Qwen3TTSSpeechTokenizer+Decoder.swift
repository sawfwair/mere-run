import Foundation
import MLX
import MLXFast
import MLXNN
import AudioCodecs
import MereRunCore

// Owns the Qwen3 TTS speech tokenizer decoder architecture.
// This file keeps the decoder stack, quantizers, and audio synthesis blocks
// together so the root tokenizer file can stay focused on the public API.

// MARK: - Decoder Layer Base

class AudioDecoderLayer: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("Subclasses must implement")
    }
}

// MARK: - Causal Convs

final class CausalConv1d: AudioDecoderLayer {
    @ModuleInfo(key: "conv") var conv: Conv1d

    private let stride: Int
    private let kernelSize: Int
    private let dilation: Int
    private let padding: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1
    ) {
        self.stride = stride
        self.dilation = dilation
        self.kernelSize = (kernelSize - 1) * dilation + 1
        self.padding = self.kernelSize - stride
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: true
        )
    }

    private func extraPadding(length: Int) -> Int {
        let nFrames = (Double(length - kernelSize + padding) / Double(stride)) + 1
        let ideal = (ceil(nFrames) - 1) * Double(stride) + Double(kernelSize - padding)
        return max(0, Int(ideal - Double(length)))
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let length = x.dim(2)
        let extra = extraPadding(length: length)
        var paddedInput = padded(x, widths: [[0, 0], [0, 0], [padding, extra]])
        paddedInput = paddedInput.transposed(0, 2, 1)
        var out = conv(paddedInput)
        out = out.transposed(0, 2, 1)
        return out
    }
}

final class CausalTransposeConv1d: AudioDecoderLayer {
    @ModuleInfo(key: "conv") var conv: ConvTransposed1d
    private let trimRight: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int
    ) {
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true
        )
        self.trimRight = kernelSize - stride
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.transposed(0, 2, 1)
        out = conv(out)
        out = out.transposed(0, 2, 1)
        if trimRight > 0 {
            out = out[.ellipsis, 0..<out.dim(2) - trimRight]
        }
        return out
    }
}

// MARK: - Activations / Norms

final class SnakeBeta: AudioDecoderLayer {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray
    private let eps: Float = 1e-9

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.zeros([channels])
        self._beta.wrappedValue = MLXArray.zeros([channels])
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let alphaExp = MLX.exp(alpha)[.newAxis, 0..., .newAxis]
        let betaExp = MLX.exp(beta)[.newAxis, 0..., .newAxis]
        return x + (1.0 / (betaExp + eps)) * MLX.pow(MLX.sin(x * alphaExp), 2)
    }
}

final class DecoderRMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    private let eps: Float

    init(hiddenSize: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.ones([hiddenSize])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

final class LayerScale: Module, UnaryLayer {
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(channels: Int, initialScale: Float) {
        self._scale.wrappedValue = MLXArray.ones([channels]) * initialScale
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        scale * x
    }
}

// MARK: - ConvNeXt Block

final class ConvNeXtBlock: AudioDecoderLayer {
    @ModuleInfo(key: "dwconv") var dwconv: CausalConv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dim: Int) {
        self._dwconv.wrappedValue = CausalConv1d(inChannels: dim, outChannels: dim, kernelSize: 7, stride: 1, dilation: 1, groups: dim)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._pwconv1.wrappedValue = Linear(dim, 4 * dim)
        self._pwconv2.wrappedValue = Linear(4 * dim, dim)
        self._gamma.wrappedValue = MLXArray.ones([dim]) * 1e-6
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var out = dwconv(x)
        out = out.transposed(0, 2, 1)
        out = norm(out)
        out = pwconv1(out)
        out = MLXNN.gelu(out)
        out = pwconv2(out)
        out = gamma * out
        out = out.transposed(0, 2, 1)
        return residual + out
    }
}

// MARK: - Decoder Transformer

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

final class EuclideanCodebook: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    let dim: Int
    let codebookSize: Int

    init(dim: Int, codebookSize: Int) {
        self.dim = dim
        self.codebookSize = codebookSize
        self._embed.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: dim)
    }

    func encode(_ vectors: MLXArray) -> MLXArray {
        var targetShape: [Int] = []
        if vectors.ndim > 1 {
            targetShape.reserveCapacity(vectors.ndim - 1)
            for axis in 0..<(vectors.ndim - 1) {
                targetShape.append(vectors.dim(axis))
            }
        }

        let flat = vectors.reshaped(-1, vectors.dim(-1)).asType(.float32)
        let indices = MLXArray((0..<codebookSize).map { Int32($0) }).asType(.int32)
        let table = embed(indices).asType(.float32) // [K, D]
        let c2 = (table * table).sum(axis: -1) / 2
        let dot = MLX.matmul(flat, table.transposed(1, 0))
        let nearest = (c2[.newAxis, 0...] - dot).argMin(axis: -1).asType(.int32)
        return nearest.reshaped(targetShape)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        embed(codes)
    }
}

final class VectorQuantization: Module {
    @ModuleInfo(key: "project_out") var projectOut: Linear?
    @ModuleInfo(key: "codebook") var codebook: EuclideanCodebook
    let codebookSize: Int

    init(dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        let cbDim = codebookDim ?? dim
        if cbDim != dim {
            self._projectIn.wrappedValue = Linear(dim, cbDim)
            self._projectOut.wrappedValue = Linear(cbDim, dim)
        } else {
            self._projectIn.wrappedValue = nil
            self._projectOut.wrappedValue = nil
        }
        self._codebook.wrappedValue = EuclideanCodebook(dim: cbDim, codebookSize: codebookSize)
        self.codebookSize = codebookSize
    }

    @ModuleInfo(key: "project_in") var projectIn: Linear?

    func encode(_ xs: MLXArray) -> MLXArray {
        var inputs = xs.transposed(0, 2, 1) // [B, T, C]
        if let proj = projectIn {
            inputs = proj(inputs)
        }
        return codebook.encode(inputs)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        var quantized = codebook.decode(codes) // [B, T, codebook_dim]
        if let proj = projectOut {
            quantized = proj(quantized)
        }
        return quantized.transposed(0, 2, 1)
    }
}

final class ResidualVectorQuantization: Module {
    @ModuleInfo(key: "layers") var layers: [VectorQuantization]

    init(numQuantizers: Int, dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        self._layers.wrappedValue = (0..<numQuantizers).map { _ in
            VectorQuantization(dim: dim, codebookSize: codebookSize, codebookDim: codebookDim)
        }
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var residual = xs
        var codes: [MLXArray] = []
        codes.reserveCapacity(layers.count)

        for layer in layers {
            let indices = layer.encode(residual) // [B, T]
            let quantized = layer.decode(indices) // [B, C, T]
            residual = (residual.asType(.float32) - quantized.asType(.float32)).asType(residual.dtype)
            codes.append(indices)
        }

        return MLX.stacked(codes, axis: 0) // [Q, B, T]
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        var quantized = MLXArray.zeros([codes.dim(1), layers[0].codebook.dim, codes.dim(2)])
        for idx in 0..<layers.count {
            let layerCodes = codes[idx]
            quantized = quantized + layers[idx].decode(layerCodes)
        }
        return quantized
    }
}

final class ResidualVectorQuantizer: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Conv1d?
    @ModuleInfo(key: "output_proj") var outputProj: Conv1d?
    @ModuleInfo(key: "vq") var vq: ResidualVectorQuantization

    let nQ: Int
    let dimension: Int
    let inputDimension: Int
    let outputDimension: Int
    let bins: Int

    init(
        dimension: Int,
        inputDimension: Int?,
        outputDimension: Int?,
        nQ: Int,
        bins: Int,
        forceProjection: Bool
    ) {
        self.nQ = nQ
        self.dimension = dimension
        self.inputDimension = inputDimension ?? dimension
        self.outputDimension = outputDimension ?? dimension
        self.bins = bins

        if self.inputDimension == dimension && !forceProjection {
            self._inputProj.wrappedValue = nil
        } else {
            self._inputProj.wrappedValue = Conv1d(inputChannels: self.inputDimension, outputChannels: dimension, kernelSize: 1, stride: 1, padding: 0, dilation: 1, groups: 1, bias: false)
        }

        if self.outputDimension == dimension && !forceProjection {
            self._outputProj.wrappedValue = nil
        } else {
            self._outputProj.wrappedValue = Conv1d(inputChannels: dimension, outputChannels: self.outputDimension, kernelSize: 1, stride: 1, padding: 0, dilation: 1, groups: 1, bias: false)
        }

        self._vq.wrappedValue = ResidualVectorQuantization(numQuantizers: nQ, dim: dimension, codebookSize: bins)
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var hidden = xs
        if let inputProj = inputProj {
            hidden = hidden.transposed(0, 2, 1)
            hidden = inputProj(hidden)
            hidden = hidden.transposed(0, 2, 1)
        }
        return vq.encode(hidden).transposed(1, 0, 2) // [B, Q, T]
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        let codesTransposed = codes.transposed(1, 0, 2)
        var quantized = vq.decode(codesTransposed)
        if let outProj = outputProj {
            quantized = quantized.transposed(0, 2, 1)
            quantized = outProj(quantized)
            quantized = quantized.transposed(0, 2, 1)
        }
        return quantized
    }
}

final class SplitResidualVectorQuantizer: Module {
    @ModuleInfo(key: "rvq_first") var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo(key: "rvq_rest") var rvqRest: ResidualVectorQuantizer

    let nQSemantic: Int
    let nQAcoustic: Int

    init(
        nQ: Int,
        nQSemantic: Int,
        dimension: Int,
        inputDimension: Int?,
        outputDimension: Int?,
        bins: Int
    ) {
        self.nQSemantic = nQSemantic
        self.nQAcoustic = nQ - nQSemantic
        self._rvqFirst.wrappedValue = ResidualVectorQuantizer(
            dimension: dimension,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            nQ: nQSemantic,
            bins: bins,
            forceProjection: true
        )
        self._rvqRest.wrappedValue = ResidualVectorQuantizer(
            dimension: dimension,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            nQ: nQ - nQSemantic,
            bins: bins,
            forceProjection: true
        )
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var codes = rvqFirst.encode(xs)
        if nQAcoustic > 0 {
            let restCodes = rvqRest.encode(xs)
            codes = MLX.concatenated([codes, restCodes], axis: 1)
        }
        return codes
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        var quantized = rvqFirst.decode(codes[0..., 0..<nQSemantic, 0...])
        if codes.dim(1) > nQSemantic {
            let rest = rvqRest.decode(codes[0..., nQSemantic..., 0...])
            quantized = quantized + rest
        }
        return quantized
    }
}

// MARK: - Decoder Blocks

final class DecoderResidualUnit: AudioDecoderLayer {
    @ModuleInfo(key: "act1") var act1: SnakeBeta
    @ModuleInfo(key: "conv1") var conv1: CausalConv1d
    @ModuleInfo(key: "act2") var act2: SnakeBeta
    @ModuleInfo(key: "conv2") var conv2: CausalConv1d

    init(dim: Int, dilation: Int) {
        self._act1.wrappedValue = SnakeBeta(channels: dim)
        self._conv1.wrappedValue = CausalConv1d(inChannels: dim, outChannels: dim, kernelSize: 7, stride: 1, dilation: dilation, groups: 1)
        self._act2.wrappedValue = SnakeBeta(channels: dim)
        self._conv2.wrappedValue = CausalConv1d(inChannels: dim, outChannels: dim, kernelSize: 1, stride: 1, dilation: 1, groups: 1)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var out = act1(x)
        out = conv1(out)
        out = act2(out)
        out = conv2(out)
        return out + residual
    }
}

final class DecoderBlockUpsample: AudioDecoderLayer {
    @ModuleInfo(key: "conv") var conv: ConvTransposed1d
    private let trimRight: Int

    init(inDim: Int, outDim: Int, upsampleRate: Int) {
        let kernel = 2 * upsampleRate
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inDim,
            outputChannels: outDim,
            kernelSize: kernel,
            stride: upsampleRate,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true
        )
        self.trimRight = kernel - upsampleRate
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.transposed(0, 2, 1)
        out = conv(out)
        out = out.transposed(0, 2, 1)
        if trimRight > 0 {
            out = out[.ellipsis, 0..<out.dim(2) - trimRight]
        }
        return out
    }
}

final class Qwen3TTSSpeechDecoderBlock: AudioDecoderLayer {
    @ModuleInfo(key: "block") var block: [AudioDecoderLayer]

    init(config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        let inDim = config.decoderDim / (1 << layerIdx)
        let outDim = config.decoderDim / (1 << (layerIdx + 1))
        let upsampleRate = config.upsampleRates[layerIdx]

        self._block.wrappedValue = [
            SnakeBeta(channels: inDim),
            DecoderBlockUpsample(inDim: inDim, outDim: outDim, upsampleRate: upsampleRate),
            DecoderResidualUnit(dim: outDim, dilation: 1),
            DecoderResidualUnit(dim: outDim, dilation: 3),
            DecoderResidualUnit(dim: outDim, dilation: 9)
        ]
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in block {
            out = layer(out)
        }
        return out
    }
}

final class DecoderInitialConv: AudioDecoderLayer {
    @ModuleInfo(key: "conv") var conv: Conv1d
    private let kernelSize: Int

    init(latentDim: Int, decoderDim: Int, kernelSize: Int = 7) {
        self.kernelSize = kernelSize
        self._conv.wrappedValue = Conv1d(inputChannels: latentDim, outputChannels: decoderDim, kernelSize: kernelSize, stride: 1, padding: 0, dilation: 1, groups: 1, bias: true)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = padded(x, widths: [[0, 0], [0, 0], [kernelSize - 1, 0]])
        out = out.transposed(0, 2, 1)
        out = conv(out)
        out = out.transposed(0, 2, 1)
        return out
    }
}

final class DecoderOutputSnake: AudioDecoderLayer {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray
    private let eps: Float = 1e-9

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.zeros([channels])
        self._beta.wrappedValue = MLXArray.zeros([channels])
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let alphaExp = MLX.exp(alpha).reshaped(1, -1, 1)
        let betaExp = MLX.exp(beta).reshaped(1, -1, 1)
        return x + (1.0 / (betaExp + eps)) * MLX.pow(MLX.sin(x * alphaExp), 2)
    }
}

final class DecoderOutputConv: AudioDecoderLayer {
    @ModuleInfo(key: "conv") var conv: Conv1d
    private let kernelSize: Int

    init(channels: Int, kernelSize: Int = 7) {
        self.kernelSize = kernelSize
        self._conv.wrappedValue = Conv1d(inputChannels: channels, outputChannels: 1, kernelSize: kernelSize, stride: 1, padding: 0, dilation: 1, groups: 1, bias: true)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = padded(x, widths: [[0, 0], [0, 0], [kernelSize - 1, 0]])
        out = out.transposed(0, 2, 1)
        out = conv(out)
        out = out.transposed(0, 2, 1)
        return out
    }
}

// MARK: - Speech Tokenizer Decoder

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
