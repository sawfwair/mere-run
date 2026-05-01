import Foundation
import MLX
import MLXNN
import MLXRandom

/// Oobleck uses PyTorch `weight_norm` on `ConvTranspose1d` with `dim=0`, which
/// scales each *input* channel (weight shape `[in, out, k]`).
///
/// After loading, `weight_v` is stored in MLX `convTransposed1d` format
/// `[out, k, in]`, but `weight_g` remains `[in, 1, 1]`.
final class OobleckWNConvTranspose1d: Module, @unchecked Sendable {
    @ModuleInfo(key: "weight_g") var weightG: MLXArray
    @ModuleInfo(key: "weight_v") var weightV: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let stride: Int
    let padding: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding

        self._weightG.wrappedValue = MLXArray.ones([inputChannels, 1, 1])
        self._weightV.wrappedValue = MLXRandom.normal([outputChannels, kernelSize, inputChannels]) * 0.02

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([outputChannels])
        } else {
            self._bias.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // weightV: [out, k, in]
        // weightG: [in, 1, 1] (PyTorch weight_norm dim=0)
        let vNorm = MLX.sqrt(MLX.sum(weightV * weightV, axes: [0, 1], keepDims: true) + 1e-12) // [1, 1, in]
        let g = weightG.reshaped(1, 1, -1) // [1, 1, in]
        let weight = weightV * (g / vNorm)

        var result = MLX.convTransposed1d(x, weight, stride: stride, padding: padding)
        if let bias {
            result = result + bias
        }
        return result
    }
}

final class OobleckDecoderBlock: Module {
    @ModuleInfo(key: "snake1") var snake1: OobleckSnakeBeta
    @ModuleInfo(key: "conv_t1") var convT1: OobleckWNConvTranspose1d
    @ModuleInfo(key: "res_unit1") var resUnit1: OobleckResUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: OobleckResUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: OobleckResUnit

    init(inputChannels: Int, outputChannels: Int, stride: Int) {
        self._snake1.wrappedValue = OobleckSnakeBeta(channels: inputChannels)

        let kernelSize = stride * 2
        let padding = stride / 2
        self._convT1.wrappedValue = OobleckWNConvTranspose1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            bias: true
        )

        // Match common Oobleck/HiFi-style dilations.
        self._resUnit1.wrappedValue = OobleckResUnit(channels: outputChannels, dilation: 1)
        self._resUnit2.wrappedValue = OobleckResUnit(channels: outputChannels, dilation: 3)
        self._resUnit3.wrappedValue = OobleckResUnit(channels: outputChannels, dilation: 9)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake1(x)
        h = convT1(h)
        h = resUnit1(h)
        h = resUnit2(h)
        h = resUnit3(h)
        return h
    }
}

final class OobleckEncoderBlock: Module {
    @ModuleInfo(key: "snake1") var snake1: OobleckSnakeBeta
    @ModuleInfo(key: "res_unit1") var resUnit1: OobleckResUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: OobleckResUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: OobleckResUnit
    @ModuleInfo(key: "conv1") var conv1: WNConv1d

    init(inputChannels: Int, outputChannels: Int, stride: Int) {
        self._snake1.wrappedValue = OobleckSnakeBeta(channels: inputChannels)

        self._resUnit1.wrappedValue = OobleckResUnit(channels: inputChannels, dilation: 1)
        self._resUnit2.wrappedValue = OobleckResUnit(channels: inputChannels, dilation: 3)
        self._resUnit3.wrappedValue = OobleckResUnit(channels: inputChannels, dilation: 9)

        let kernelSize = stride * 2
        let padding = stride / 2
        self._conv1.wrappedValue = WNConv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: 1,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake1(x)
        h = resUnit1(h)
        h = resUnit2(h)
        h = resUnit3(h)
        return conv1(h)
    }
}

final class OobleckEncoder: Module {
    let config: OobleckVAEConfig

    @ModuleInfo(key: "conv1") var conv1: WNConv1d
    @ModuleInfo(key: "block") var blocks: [OobleckEncoderBlock]
    @ModuleInfo(key: "snake1") var snake1: OobleckSnakeBeta
    @ModuleInfo(key: "conv2") var conv2: WNConv1d

    init(config: OobleckVAEConfig) {
        self.config = config

        let base = config.encoderHiddenSize
        let multiples = config.channelMultiples
        let startMult = multiples.first ?? 1
        let startChannels = base * startMult

        self._conv1.wrappedValue = WNConv1d(
            inputChannels: config.audioChannels,
            outputChannels: startChannels,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            bias: true
        )

        var currentChannels = startChannels
        var built: [OobleckEncoderBlock] = []
        built.reserveCapacity(config.downsamplingRatios.count)

        for (i, stride) in config.downsamplingRatios.enumerated() {
            let outMult = multiples.indices.contains(i) ? multiples[i] : (multiples.last ?? 1)
            let outChannels = base * outMult
            built.append(OobleckEncoderBlock(inputChannels: currentChannels, outputChannels: outChannels, stride: stride))
            currentChannels = outChannels
        }

        self._blocks.wrappedValue = built

        self._snake1.wrappedValue = OobleckSnakeBeta(channels: currentChannels)
        self._conv2.wrappedValue = WNConv1d(
            inputChannels: currentChannels,
            outputChannels: config.decoderInputChannels * 2,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            bias: true
        )
    }

    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        var h = conv1(audio)
        for block in blocks {
            h = block(h)
        }
        h = snake1(h)
        return conv2(h)
    }
}

final class OobleckDecoder: Module {
    let config: OobleckVAEConfig

    @ModuleInfo(key: "conv1") var conv1: WNConv1d
    @ModuleInfo(key: "block") var blocks: [OobleckDecoderBlock]
    @ModuleInfo(key: "snake1") var snake1: OobleckSnakeBeta
    @ModuleInfo(key: "conv2") var conv2: WNConv1d

    init(config: OobleckVAEConfig) {
        self.config = config

        let base = config.decoderChannels
        let multiples = config.channelMultiples
        let startMult = multiples.last ?? 1
        let startChannels = base * startMult

        self._conv1.wrappedValue = WNConv1d(
            inputChannels: config.decoderInputChannels,
            outputChannels: startChannels,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            bias: true
        )

        let ratios = config.downsamplingRatios.reversed()
        var currentChannels = startChannels
        var built: [OobleckDecoderBlock] = []
        built.reserveCapacity(config.downsamplingRatios.count)

        for (i, stride) in ratios.enumerated() {
            let multIndex = max(0, multiples.count - 2 - i)
            let outMult = multiples.indices.contains(multIndex) ? multiples[multIndex] : (multiples.first ?? 1)
            let outChannels = base * outMult
            built.append(OobleckDecoderBlock(inputChannels: currentChannels, outputChannels: outChannels, stride: stride))
            currentChannels = outChannels
        }

        self._blocks.wrappedValue = built

        self._snake1.wrappedValue = OobleckSnakeBeta(channels: currentChannels)
        self._conv2.wrappedValue = WNConv1d(
            inputChannels: currentChannels,
            outputChannels: config.audioChannels,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            dilation: 1,
            bias: false
        )
    }

    func callAsFunction(_ latents: MLXArray) -> MLXArray {
        var h = conv1(latents)
        for block in blocks {
            h = block(h)
        }
        h = snake1(h)
        return conv2(h)
    }
}
