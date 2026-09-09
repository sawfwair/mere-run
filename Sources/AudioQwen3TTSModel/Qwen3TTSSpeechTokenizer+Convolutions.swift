import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunKVCache

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
