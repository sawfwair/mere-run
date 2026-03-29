import Foundation
import MLX
import MLXFast
import MLXNN
import AudioCodecs
import MereRunCore

// Owns the Mimi-style speech tokenizer encoder stack.
// This file intentionally keeps the encoder architecture together and
// leaves the public tokenizer entrypoints in the root tokenizer file.

enum MimiPadMode {
    case constant
    case edge
}

private func mimiElu(_ x: MLXArray) -> MLXArray {
    MLX.where(x .> 0, x, MLX.exp(x) - 1.0)
}

private func mimiPad1d(_ x: MLXArray, left: Int, right: Int, mode: MimiPadMode) -> MLXArray {
    guard left > 0 || right > 0 else { return x }

    switch mode {
    case .constant:
        return padded(x, widths: [[0, 0], [0, 0], [left, right]])
    case .edge:
        let batch = x.dim(0)
        let channels = x.dim(1)
        let length = x.dim(2)

        if length == 0 {
            return MLXArray.zeros([batch, channels, left + right], dtype: x.dtype)
        }

        var pieces: [MLXArray] = []
        if left > 0 {
            let first = x[0..., 0..., 0..<1]
            pieces.append(broadcast(first, to: [batch, channels, left]))
        }

        pieces.append(x)

        if right > 0 {
            let last = x[0..., 0..., (length - 1)..<length]
            pieces.append(broadcast(last, to: [batch, channels, right]))
        }

        return MLX.concatenated(pieces, axis: 2)
    }
}

final class MimiNormConv1d: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int,
        dilation: Int,
        groups: Int,
        bias: Bool
    ) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.transposed(0, 2, 1)
        out = conv(out)
        out = out.transposed(0, 2, 1)
        return out
    }
}

final class MimiStreamableConv1d: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: MimiNormConv1d

    private let causal: Bool
    private let padMode: MimiPadMode
    private let kernelSize: Int
    private let stride: Int
    private let dilation: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true,
        causal: Bool,
        padMode: MimiPadMode
    ) {
        self.causal = causal
        self.padMode = padMode
        self.kernelSize = kernelSize
        self.stride = stride
        self.dilation = dilation
        self._conv.wrappedValue = MimiNormConv1d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
    }

    private func extraPadding(length: Int) -> Int {
        let effectiveKernel = (kernelSize - 1) * dilation + 1
        let paddingTotal = effectiveKernel - stride
        let nFrames = (Double(length + paddingTotal - effectiveKernel) / Double(stride)) + 1.0
        let ideal = (ceil(nFrames) - 1.0) * Double(stride) + Double(effectiveKernel - paddingTotal)
        return max(0, Int(ideal - Double(length)))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let effectiveKernel = (kernelSize - 1) * dilation + 1
        let paddingTotal = effectiveKernel - stride
        let extra = extraPadding(length: x.dim(2))

        let padLeft: Int
        let padRight: Int
        if causal {
            padLeft = paddingTotal
            padRight = 0
        } else {
            padRight = paddingTotal / 2
            padLeft = paddingTotal - padRight
        }

        let paddedInput = mimiPad1d(x, left: padLeft, right: padRight + extra, mode: padMode)
        return conv(paddedInput)
    }
}

final class MimiLayerScale: Module, UnaryLayer {
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(channels: Int) {
        self._scale.wrappedValue = MLXArray.ones([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        scale * x
    }
}

final class MimiMLPNoGating: Module, UnaryLayer {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._linear1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._linear2.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(MLXNN.gelu(linear1(x)))
    }
}

final class MimiAttention: Module {
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let rotaryEmbedding: DecoderRotaryEmbedding

    init(config: Qwen3TTSTokenizerEncoderConfig) {
        self.headDim = max(1, config.headDim)
        self.numHeads = max(1, config.numAttentionHeads)
        self.numKVHeads = max(1, config.numKeyValueHeads)
        self.scale = 1.0 / Float(headDim).squareRoot()

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim
        self._inProj.wrappedValue = Linear(config.hiddenSize, qDim + (2 * kvDim), bias: false)
        self._outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        self.rotaryEmbedding = DecoderRotaryEmbedding(
            dim: headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: KVCache,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)
        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim

        let projected = inProj(x)

        var q = projected[0..., 0..., 0..<qDim]
        var k = projected[0..., 0..., qDim..<(qDim + kvDim)]
        var v = projected[0..., 0..., (qDim + kvDim)..<(qDim + (2 * kvDim))]

        q = q.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, seqLen, numKVHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, seqLen, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache.offset
        let base = MLXArray(Int32(offset)..<Int32(offset + seqLen)).reshaped(1, seqLen)
        let positionIds = broadcast(base, to: [batch, seqLen])
        let positionEmbeddings = rotaryEmbedding(x, positionIds: positionIds)
        (q, k) = applyDecoderRotary(q, k, cos: positionEmbeddings.cos, sin: positionEmbeddings.sin)

        let kvGroups = numHeads / max(1, numKVHeads)
        if kvGroups > 1 {
            k = repeatDecoderKVHeads(k, groups: kvGroups)
            v = repeatDecoderKVHeads(v, groups: kvGroups)
        }

        let updated = cache.update(keys: k, values: v)
        k = updated.0
        v = updated.1

        var out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        out = out.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)
        return outProj(out)
    }
}

final class MimiTransformerLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MimiAttention
    @ModuleInfo(key: "gating") var gating: MimiMLPNoGating
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "layer_scale_1") var layerScale1: MimiLayerScale
    @ModuleInfo(key: "layer_scale_2") var layerScale2: MimiLayerScale

    init(config: Qwen3TTSTokenizerEncoderConfig) {
        self._selfAttn.wrappedValue = MimiAttention(config: config)
        self._gating.wrappedValue = MimiMLPNoGating(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        self._norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
        self._norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
        self._layerScale1.wrappedValue = MimiLayerScale(channels: config.hiddenSize)
        self._layerScale2.wrappedValue = MimiLayerScale(channels: config.hiddenSize)
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: KVCache,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var hidden = x
        let attn = selfAttn(norm1(hidden), cache: cache, mask: mask)
        hidden = hidden + layerScale1(attn)
        hidden = hidden + layerScale2(gating(norm2(hidden)))
        return hidden
    }
}

final class MimiTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [MimiTransformerLayer]

    init(config: Qwen3TTSTokenizerEncoderConfig) {
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            MimiTransformerLayer(config: config)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: [KVCache],
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var hidden = x
        for (idx, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache[idx], mask: mask)
        }
        return hidden
    }

    func makeCache() -> [KVCache] {
        (0..<layers.count).map { _ in KVCacheSimple(step: 256) }
    }
}

final class MimiProjectedTransformer: Module {
    @ModuleInfo(key: "transformer") var transformer: MimiTransformer
    @ModuleInfo(key: "input_proj") var inputProj: Linear?

    let convLayout: Bool

    init(config: Qwen3TTSTokenizerEncoderConfig, inputDim: Int) {
        self.convLayout = true
        self._transformer.wrappedValue = MimiTransformer(config: config)
        if inputDim == config.hiddenSize {
            self._inputProj.wrappedValue = nil
        } else {
            self._inputProj.wrappedValue = Linear(inputDim, config.hiddenSize, bias: false)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: [KVCache],
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal
    ) -> [MLXArray] {
        var hidden = convLayout ? x.transposed(0, 2, 1) : x
        if let inputProj {
            hidden = inputProj(hidden)
        }
        hidden = transformer(hidden, cache: cache, mask: mask)
        if convLayout {
            hidden = hidden.transposed(0, 2, 1)
        }
        return [hidden]
    }

    func makeCache() -> [KVCache] {
        transformer.makeCache()
    }
}

final class MimiSeanetResnetBlock: Module {
    @ModuleInfo(key: "block") var block: [MimiStreamableConv1d]
    @ModuleInfo(key: "shortcut") var shortcut: MimiStreamableConv1d?

    init(config: Qwen3TTSTokenizerEncoderConfig, dim: Int, dilation: Int) {
        let hidden = max(1, dim / max(1, config.compress))
        self._block.wrappedValue = [
            MimiStreamableConv1d(
                inChannels: dim,
                outChannels: hidden,
                kernelSize: config.residualKernelSize,
                stride: 1,
                dilation: dilation,
                groups: 1,
                bias: true,
                causal: config.useCausalConv,
                padMode: .constant
            ),
            MimiStreamableConv1d(
                inChannels: hidden,
                outChannels: dim,
                kernelSize: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                bias: true,
                causal: config.useCausalConv,
                padMode: .constant
            )
        ]

        if config.useConvShortcut {
            self._shortcut.wrappedValue = MimiStreamableConv1d(
                inChannels: dim,
                outChannels: dim,
                kernelSize: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                bias: true,
                causal: config.useCausalConv,
                padMode: .constant
            )
        } else {
            self._shortcut.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        let residual = x
        for conv in block {
            hidden = conv(mimiElu(hidden))
        }
        if let shortcut {
            return hidden + shortcut(residual)
        }
        return hidden + residual
    }
}

final class MimiSeanetEncoderLayer: Module {
    @ModuleInfo(key: "residuals") var residuals: [MimiSeanetResnetBlock]
    @ModuleInfo(key: "downsample") var downsample: MimiStreamableConv1d

    init(config: Qwen3TTSTokenizerEncoderConfig, ratio: Int, mult: Int) {
        let dim = max(1, mult * config.numFilters)
        var dilation = 1
        self._residuals.wrappedValue = (0..<max(1, config.numResidualLayers)).map { _ in
            defer { dilation *= max(1, config.dilationGrowthRate) }
            return MimiSeanetResnetBlock(config: config, dim: dim, dilation: dilation)
        }

        self._downsample.wrappedValue = MimiStreamableConv1d(
            inChannels: dim,
            outChannels: dim * 2,
            kernelSize: ratio * 2,
            stride: ratio,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: true,
            padMode: .constant
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for residual in residuals {
            hidden = residual(hidden)
        }
        return downsample(mimiElu(hidden))
    }
}

final class MimiSeanetEncoder: Module {
    @ModuleInfo(key: "init_conv1d") var initConv1d: MimiStreamableConv1d
    @ModuleInfo(key: "layers") var layers: [MimiSeanetEncoderLayer]
    @ModuleInfo(key: "final_conv1d") var finalConv1d: MimiStreamableConv1d

    init(config: Qwen3TTSTokenizerEncoderConfig) {
        var mult = 1
        self._initConv1d.wrappedValue = MimiStreamableConv1d(
            inChannels: max(1, config.audioChannels),
            outChannels: mult * max(1, config.numFilters),
            kernelSize: max(1, config.kernelSize),
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: config.useCausalConv,
            padMode: .constant
        )

        var encoderLayers: [MimiSeanetEncoderLayer] = []
        for ratio in config.upsamplingRatios.reversed() {
            encoderLayers.append(MimiSeanetEncoderLayer(config: config, ratio: max(1, ratio), mult: mult))
            mult *= 2
        }
        self._layers.wrappedValue = encoderLayers

        self._finalConv1d.wrappedValue = MimiStreamableConv1d(
            inChannels: mult * max(1, config.numFilters),
            outChannels: max(1, config.hiddenSize),
            kernelSize: max(1, config.lastKernelSize),
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: config.useCausalConv,
            padMode: .constant
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = initConv1d(x)
        for layer in layers {
            hidden = layer(hidden)
        }
        hidden = mimiElu(hidden)
        return finalConv1d(hidden)
    }
}

final class MimiConvDownsample1d: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: MimiStreamableConv1d

    init(stride: Int, dim: Int, causal: Bool) {
        self._conv.wrappedValue = MimiStreamableConv1d(
            inChannels: dim,
            outChannels: dim,
            kernelSize: 2 * max(1, stride),
            stride: max(1, stride),
            dilation: 1,
            groups: 1,
            bias: false,
            causal: causal,
            padMode: .edge
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x)
    }
}

final class Qwen3TTSSpeechTokenizerEncoder: Module {
    let config: Qwen3TTSTokenizerEncoderConfig
    let validNumQuantizers: Int

    @ModuleInfo(key: "encoder") var encoder: MimiSeanetEncoder
    @ModuleInfo(key: "encoder_transformer") var encoderTransformer: MimiProjectedTransformer
    @ModuleInfo(key: "downsample") var downsample: MimiConvDownsample1d
    @ModuleInfo(key: "quantizer") var quantizer: SplitResidualVectorQuantizer

    init(config: Qwen3TTSTokenizerEncoderConfig, validNumQuantizers: Int) {
        self.config = config
        self.validNumQuantizers = max(1, validNumQuantizers)

        self._encoder.wrappedValue = MimiSeanetEncoder(config: config)
        self._encoderTransformer.wrappedValue = MimiProjectedTransformer(config: config, inputDim: config.hiddenSize)

        let encoderFrameRate = Float(config.samplingRate) / Float(max(1, config.upsamplingRatios.reduce(1, *)))
        let stride = max(1, Int((encoderFrameRate / max(config.frameRate, 1e-6)).rounded(.towardZero)))
        self._downsample.wrappedValue = MimiConvDownsample1d(stride: stride, dim: config.hiddenSize, causal: config.useCausalConv)

        self._quantizer.wrappedValue = SplitResidualVectorQuantizer(
            nQ: max(1, config.numQuantizers),
            nQSemantic: max(1, config.numSemanticQuantizers),
            dimension: max(1, config.codebookDim),
            inputDimension: max(1, config.hiddenSize),
            outputDimension: max(1, config.hiddenSize),
            bins: max(2, config.codebookSize)
        )
    }

    func encode(_ audio: MLXArray) -> MLXArray {
        var hidden = encoder(audio)
        let cache = encoderTransformer.makeCache()
        hidden = encoderTransformer(hidden, cache: cache, mask: .causal)[0]
        hidden = downsample(hidden)

        var codes = quantizer.encode(hidden) // [B, Q, T]
        if codes.dim(1) > validNumQuantizers {
            codes = codes[0..., 0..<validNumQuantizers, 0...]
        }
        return codes.asType(.int32)
    }
}
