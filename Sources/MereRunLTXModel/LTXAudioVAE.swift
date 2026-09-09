import Foundation
import MLX
import MLXFast
import MLXNN

package enum LTXAudioCausalityAxis {
    case none
    case height
}

package final class LTXAudioPixelNorm: Module {
    package let eps: Float

    package init(eps: Float = 1e-6) {
        self.eps = eps
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let meanSq = MLX.mean(x * x, axis: -1, keepDims: true)
        return x / MLX.sqrt(meanSq + MLXArray(eps).asType(x.dtype))
    }
}

package final class LTXAudioCausalConv2d: Module {
    @ModuleInfo(key: "conv") package var conv: Conv2d

    package let causalityAxis: LTXAudioCausalityAxis
    package let padTop: Int
    package let padBottom: Int
    package let padLeft: Int
    package let padRight: Int

    package init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        causalityAxis: LTXAudioCausalityAxis = .height
    ) {
        self.causalityAxis = causalityAxis
        let pad = kernelSize - 1
        switch causalityAxis {
        case .none:
            self.padTop = pad / 2
            self.padBottom = pad - (pad / 2)
            self.padLeft = pad / 2
            self.padRight = pad - (pad / 2)
        case .height:
            self.padTop = pad
            self.padBottom = 0
            self.padLeft = pad / 2
            self.padRight = pad - (pad / 2)
        }
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: .init([kernelSize, kernelSize]),
            stride: .init([stride, stride]),
            padding: .init(0),
            bias: true
        )
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let paddedInput = padded(
            x,
            widths: [
                [0, 0],
                [padTop, padBottom],
                [padLeft, padRight],
                [0, 0],
            ]
        )
        return conv(paddedInput)
    }
}

package final class LTXAudioResnetBlock2D: Module {
    package let inChannels: Int
    package let outChannels: Int

    @ModuleInfo(key: "norm1") package var norm1: LTXAudioPixelNorm
    @ModuleInfo(key: "conv1") package var conv1: LTXAudioCausalConv2d
    @ModuleInfo(key: "norm2") package var norm2: LTXAudioPixelNorm
    @ModuleInfo(key: "conv2") package var conv2: LTXAudioCausalConv2d
    @ModuleInfo(key: "nin_shortcut") package var ninShortcut: LTXAudioCausalConv2d?

    package init(inChannels: Int, outChannels: Int) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self._norm1.wrappedValue = LTXAudioPixelNorm()
        self._conv1.wrappedValue = LTXAudioCausalConv2d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: 3,
            stride: 1,
            causalityAxis: .height
        )
        self._norm2.wrappedValue = LTXAudioPixelNorm()
        self._conv2.wrappedValue = LTXAudioCausalConv2d(
            inChannels: outChannels,
            outChannels: outChannels,
            kernelSize: 3,
            stride: 1,
            causalityAxis: .height
        )
        if inChannels != outChannels {
            self._ninShortcut.wrappedValue = LTXAudioCausalConv2d(
                inChannels: inChannels,
                outChannels: outChannels,
                kernelSize: 1,
                stride: 1,
                causalityAxis: .height
            )
        } else {
            self._ninShortcut.wrappedValue = nil
        }
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm1(x)
        h = silu(h)
        h = conv1(h)
        h = norm2(h)
        h = silu(h)
        h = conv2(h)
        let residual = ninShortcut?(x) ?? x
        return residual + h
    }
}

package final class LTXAudioUpsample2d: Module {
    @ModuleInfo(key: "conv") package var conv: LTXAudioCausalConv2d

    package init(channels: Int) {
        self._conv.wrappedValue = LTXAudioCausalConv2d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: 3,
            stride: 1,
            causalityAxis: .height
        )
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = MLX.repeated(x, count: 2, axis: 1)
        y = MLX.repeated(y, count: 2, axis: 2)
        y = conv(y)
        y = y[0..., 1..., 0..., 0...]
        return y
    }
}

package final class LTXAudioDownsample2d: Module {
    @ModuleInfo(key: "conv") package var conv: Conv2d

    package init(channels: Int) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: .init([3, 3]),
            stride: .init([2, 2]),
            padding: .init(0),
            bias: true
        )
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Upstream HEIGHT causality pads (left: 0, right: 1, top: 2, bottom: 0)
        // before its stride-2 convolution.
        conv(padded(x, widths: [[0, 0], [2, 0], [0, 1], [0, 0]]))
    }
}

package final class LTXAudioEncoderStage: Module {
    @ModuleInfo(key: "block") package var blocks: [LTXAudioResnetBlock2D]
    @ModuleInfo(key: "downsample") package var downsample: LTXAudioDownsample2d?

    package init(blocks: [LTXAudioResnetBlock2D], downsampleChannels: Int?) {
        self._blocks.wrappedValue = blocks
        if let downsampleChannels {
            self._downsample.wrappedValue = LTXAudioDownsample2d(channels: downsampleChannels)
        } else {
            self._downsample.wrappedValue = nil
        }
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in blocks {
            h = block(h)
        }
        return downsample?(h) ?? h
    }
}

package final class LTXAudioDecoderStage: Module {
    @ModuleInfo(key: "block") package var blocks: [LTXAudioResnetBlock2D]
    @ModuleInfo(key: "upsample") package var upsample: LTXAudioUpsample2d?

    package init(blocks: [LTXAudioResnetBlock2D], upsampleChannels: Int?) {
        self._blocks.wrappedValue = blocks
        if let upsampleChannels {
            self._upsample.wrappedValue = LTXAudioUpsample2d(channels: upsampleChannels)
        } else {
            self._upsample.wrappedValue = nil
        }
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in blocks {
            h = block(h)
        }
        if let upsample {
            h = upsample(h)
        }
        return h
    }
}

package final class LTXAudioMidBlock: Module {
    @ModuleInfo(key: "block_1") package var block1: LTXAudioResnetBlock2D
    @ModuleInfo(key: "block_2") package var block2: LTXAudioResnetBlock2D

    package init(channels: Int) {
        self._block1.wrappedValue = LTXAudioResnetBlock2D(inChannels: channels, outChannels: channels)
        self._block2.wrappedValue = LTXAudioResnetBlock2D(inChannels: channels, outChannels: channels)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        block2(block1(x))
    }
}

package final class LTXAudioPerChannelStatistics: Module {
    @ParameterInfo(key: "_std_of_means") package var stdOfMeans: MLXArray
    @ParameterInfo(key: "_mean_of_means") package var meanOfMeans: MLXArray

    package init(latentChannels: Int = 128) {
        self._stdOfMeans.wrappedValue = MLX.ones([latentChannels], dtype: .float32)
        self._meanOfMeans.wrappedValue = MLX.zeros([latentChannels], dtype: .float32)
    }

    package func unNormalize(_ x: MLXArray) -> MLXArray {
        let std = stdOfMeans.asType(x.dtype)
        let mean = meanOfMeans.asType(x.dtype)
        return x * std + mean
    }

    package func normalize(_ x: MLXArray) -> MLXArray {
        let std = stdOfMeans.asType(x.dtype)
        let mean = meanOfMeans.asType(x.dtype)
        return (x - mean) / std
    }
}

package struct LTXAudioPatchifier {
    package func patchify(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let f = x.dim(2)
        let c = x.dim(3)
        return x.transposed(0, 1, 3, 2).reshaped(b, t, c * f)
    }

    package func unpatchify(_ x: MLXArray, channels: Int, melBins: Int) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        return x.reshaped(b, t, channels, melBins).transposed(0, 1, 3, 2)
    }
}

package final class LTXAudioEncoder: Module {
    @ModuleInfo(key: "per_channel_statistics") package var perChannelStatistics: LTXAudioPerChannelStatistics
    @ModuleInfo(key: "conv_in") package var convIn: LTXAudioCausalConv2d
    @ModuleInfo(key: "down") package var down: [LTXAudioEncoderStage]
    @ModuleInfo(key: "mid") package var mid: LTXAudioMidBlock
    @ModuleInfo(key: "norm_out") package var normOut: LTXAudioPixelNorm
    @ModuleInfo(key: "conv_out") package var convOut: LTXAudioCausalConv2d

    package let patchifier = LTXAudioPatchifier()

    package override init() {
        self._perChannelStatistics.wrappedValue = LTXAudioPerChannelStatistics(latentChannels: 128)
        self._convIn.wrappedValue = LTXAudioCausalConv2d(inChannels: 2, outChannels: 128)
        self._down.wrappedValue = [
            LTXAudioEncoderStage(
                blocks: [
                    LTXAudioResnetBlock2D(inChannels: 128, outChannels: 128),
                    LTXAudioResnetBlock2D(inChannels: 128, outChannels: 128),
                ],
                downsampleChannels: 128
            ),
            LTXAudioEncoderStage(
                blocks: [
                    LTXAudioResnetBlock2D(inChannels: 128, outChannels: 256),
                    LTXAudioResnetBlock2D(inChannels: 256, outChannels: 256),
                ],
                downsampleChannels: 256
            ),
            LTXAudioEncoderStage(
                blocks: [
                    LTXAudioResnetBlock2D(inChannels: 256, outChannels: 512),
                    LTXAudioResnetBlock2D(inChannels: 512, outChannels: 512),
                ],
                downsampleChannels: nil
            ),
        ]
        self._mid.wrappedValue = LTXAudioMidBlock(channels: 512)
        self._normOut.wrappedValue = LTXAudioPixelNorm()
        self._convOut.wrappedValue = LTXAudioCausalConv2d(inChannels: 512, outChannels: 16)
        super.init()
    }

    /// Encodes `[batch, channels, time, mel]` and returns transformer latents
    /// in `[batch, 8, latent time, 16]` layout.
    package func encode(spectrogram: MLXArray) -> MLXArray {
        precondition(
            spectrogram.ndim == 4 && spectrogram.dim(1) == 2 && spectrogram.dim(3) == 64,
            "LTX audio VAE expects [batch, 2, time, 64] log-mels."
        )
        var h = spectrogram.transposed(0, 2, 3, 1)
        h = convIn(h)
        for stage in down {
            h = stage(h)
        }
        h = mid(h)
        h = normOut(h)
        h = silu(h)
        h = convOut(h)

        let means = h[0..., 0..., 0..., 0..<LTXAudioLatentChannels]
        var patched = patchifier.patchify(means)
        patched = perChannelStatistics.normalize(patched)
        let normalized = patchifier.unpatchify(
            patched,
            channels: LTXAudioLatentChannels,
            melBins: means.dim(2)
        )
        return normalized.transposed(0, 3, 1, 2)
    }
}

package final class LTXAudioDecoder: Module {
    @ModuleInfo(key: "per_channel_statistics") package var perChannelStatistics: LTXAudioPerChannelStatistics
    @ModuleInfo(key: "conv_in") package var convIn: LTXAudioCausalConv2d
    @ModuleInfo(key: "mid") package var mid: LTXAudioMidBlock
    @ModuleInfo(key: "up") package var up: [LTXAudioDecoderStage]
    @ModuleInfo(key: "norm_out") package var normOut: LTXAudioPixelNorm
    @ModuleInfo(key: "conv_out") package var convOut: LTXAudioCausalConv2d

    package let latentChannels = LTXAudioLatentChannels
    package let outputChannels = 2
    package let outputMelBins = 64
    package let patchifier = LTXAudioPatchifier()

    package override init() {
        self._perChannelStatistics.wrappedValue = LTXAudioPerChannelStatistics(latentChannels: 128)
        self._convIn.wrappedValue = LTXAudioCausalConv2d(
            inChannels: LTXAudioLatentChannels,
            outChannels: 512,
            kernelSize: 3,
            stride: 1,
            causalityAxis: .height
        )
        self._mid.wrappedValue = LTXAudioMidBlock(channels: 512)
        self._up.wrappedValue = [
            LTXAudioDecoderStage(
                blocks: [
                    LTXAudioResnetBlock2D(inChannels: 256, outChannels: 128),
                    LTXAudioResnetBlock2D(inChannels: 128, outChannels: 128),
                    LTXAudioResnetBlock2D(inChannels: 128, outChannels: 128),
                ],
                upsampleChannels: nil
            ),
            LTXAudioDecoderStage(
                blocks: [
                    LTXAudioResnetBlock2D(inChannels: 512, outChannels: 256),
                    LTXAudioResnetBlock2D(inChannels: 256, outChannels: 256),
                    LTXAudioResnetBlock2D(inChannels: 256, outChannels: 256),
                ],
                upsampleChannels: 256
            ),
            LTXAudioDecoderStage(
                blocks: [
                    LTXAudioResnetBlock2D(inChannels: 512, outChannels: 512),
                    LTXAudioResnetBlock2D(inChannels: 512, outChannels: 512),
                    LTXAudioResnetBlock2D(inChannels: 512, outChannels: 512),
                ],
                upsampleChannels: 512
            ),
        ]
        self._normOut.wrappedValue = LTXAudioPixelNorm()
        self._convOut.wrappedValue = LTXAudioCausalConv2d(
            inChannels: 128,
            outChannels: 2,
            kernelSize: 3,
            stride: 1,
            causalityAxis: .height
        )
        super.init()
    }

    package func decode(latents: MLXArray) -> MLXArray {
        var sample = latents
        if sample.ndim == 4, sample.dim(1) == latentChannels {
            sample = sample.transposed(0, 2, 3, 1)
        }

        let originalFrames = sample.dim(1)
        let originalMelBins = sample.dim(2)
        let originalChannels = sample.dim(3)

        var patched = patchifier.patchify(sample)
        patched = perChannelStatistics.unNormalize(patched)
        sample = patchifier.unpatchify(patched, channels: originalChannels, melBins: originalMelBins)
        saveLTXAVDebugArray(sample, suffix: "audio_decoder_denormalized")

        var targetFrames = originalFrames * LTXAudioLatentDownsampleFactor
        targetFrames = max(1, targetFrames - (LTXAudioLatentDownsampleFactor - 1))
        let targetMelBins = outputMelBins

        var h = convIn(sample)
        saveLTXAVDebugArray(h, suffix: "audio_decoder_conv_in")
        h = mid(h)
        saveLTXAVDebugArray(h, suffix: "audio_decoder_mid")

        for level in stride(from: up.count - 1, through: 0, by: -1) {
            h = up[level](h)
            saveLTXAVDebugArray(h, suffix: "audio_decoder_up_\(level)")
        }

        h = normOut(h)
        h = silu(h)
        h = convOut(h)
        saveLTXAVDebugArray(h, suffix: "audio_decoder_conv_out")

        let croppedFrames = min(h.dim(1), targetFrames)
        let croppedMels = min(h.dim(2), targetMelBins)
        var output = h[0..., 0..<croppedFrames, 0..<croppedMels, 0..<outputChannels]

        let framePad = max(0, targetFrames - output.dim(1))
        let melPad = max(0, targetMelBins - output.dim(2))
        if framePad > 0 || melPad > 0 {
            output = padded(
                output,
                widths: [
                    [0, 0],
                    [0, framePad],
                    [0, melPad],
                    [0, 0],
                ]
            )
        }

        output = output[0..., 0..<targetFrames, 0..<targetMelBins, 0..<outputChannels]
        return output.transposed(0, 3, 1, 2)
    }
}
