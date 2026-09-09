import Foundation
import MLX
import MLXFast
import MLXNN

package final class LTXUpsamplerGroupNorm3d: Module {
    package let numGroups: Int
    package let numChannels: Int
    package let eps: Float

    @ModuleInfo(key: "weight") package var weight: MLXArray
    @ModuleInfo(key: "bias") package var bias: MLXArray

    package init(numGroups: Int, numChannels: Int, eps: Float = 1e-5) {
        self.numGroups = numGroups
        self.numChannels = numChannels
        self.eps = eps
        self._weight.wrappedValue = MLX.ones([numChannels], dtype: .float32)
        self._bias.wrappedValue = MLX.zeros([numChannels], dtype: .float32)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim == 5, "Expected NDHWC tensor for GroupNorm3d")

        let n = x.dim(0)
        let d = x.dim(1)
        let h = x.dim(2)
        let w = x.dim(3)
        let c = x.dim(4)
        precondition(c == numChannels, "Channel mismatch for GroupNorm3d")
        precondition(c % numGroups == 0, "GroupNorm3d requires channels divisible by groups")

        let inputDType = x.dtype
        var y = x.asType(.float32).reshaped(n, d * h * w, numGroups, c / numGroups)
        let mean = MLX.mean(y, axes: [1, 3], keepDims: true)
        let variance = MLX.mean((y - mean) * (y - mean), axes: [1, 3], keepDims: true)
        y = (y - mean) / MLX.sqrt(variance + MLXArray(eps))
        y = y.reshaped(n, d, h, w, c)

        let weight = self.weight.asType(.float32).reshaped(1, 1, 1, 1, c)
        let bias = self.bias.asType(.float32).reshaped(1, 1, 1, 1, c)
        y = y * weight + bias
        return y.asType(inputDType)
    }
}

package final class LTXUpsamplerResBlock3D: Module {
    @ModuleInfo(key: "conv1") package var conv1: Conv3d
    @ModuleInfo(key: "norm1") package var norm1: LTXUpsamplerGroupNorm3d
    @ModuleInfo(key: "conv2") package var conv2: Conv3d
    @ModuleInfo(key: "norm2") package var norm2: LTXUpsamplerGroupNorm3d

    package init(channels: Int) {
        self._conv1.wrappedValue = Conv3d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: .init([3, 3, 3]),
            stride: .init([1, 1, 1]),
            padding: .init([1, 1, 1]),
            bias: true
        )
        self._norm1.wrappedValue = LTXUpsamplerGroupNorm3d(numGroups: 32, numChannels: channels, eps: 1e-5)
        self._conv2.wrappedValue = Conv3d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: .init([3, 3, 3]),
            stride: .init([1, 1, 1]),
            padding: .init([1, 1, 1]),
            bias: true
        )
        self._norm2.wrappedValue = LTXUpsamplerGroupNorm3d(numGroups: 32, numChannels: channels, eps: 1e-5)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = conv1(x)
        h = norm1(h)
        h = silu(h)
        h = conv2(h)
        h = norm2(h)
        return silu(h + residual)
    }
}

package func pixelShuffle2D(_ x: MLXArray, upscaleFactor: Int = 2) -> MLXArray {
    let n = x.dim(0)
    let h = x.dim(1)
    let w = x.dim(2)
    let c = x.dim(3)
    let r = upscaleFactor
    precondition(c % (r * r) == 0, "PixelShuffle2D channel count must be divisible by scale^2")

    let outC = c / (r * r)
    var y = x.reshaped(n, h, w, outC, r, r)
    y = y.transposed(0, 1, 4, 2, 5, 3)
    return y.reshaped(n, h * r, w * r, outC)
}

package func ltxTemporalPixelShuffle(_ x: MLXArray, upscaleFactor: Int = 2) -> MLXArray {
    precondition(x.ndim == 5, "Expected NDHWC tensor")
    let n = x.dim(0)
    let d = x.dim(1)
    let h = x.dim(2)
    let w = x.dim(3)
    let c = x.dim(4)
    precondition(c % upscaleFactor == 0, "Temporal pixel-shuffle channels must divide by the scale")

    let outC = c / upscaleFactor
    return x.reshaped(n, d, h, w, outC, upscaleFactor)
        .transposed(0, 1, 5, 2, 3, 4)
        .reshaped(n, d * upscaleFactor, h, w, outC)
}

package final class LTXSpatialRationalResampler: Module {
    @ModuleInfo(key: "conv") package var conv: Conv2d

    package init(midChannels: Int, scale _: Float = 2.0) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: midChannels,
            outputChannels: 4 * midChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            bias: true
        )
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim == 5, "Expected NDHWC tensor")

        let n = x.dim(0)
        let d = x.dim(1)
        let h = x.dim(2)
        let w = x.dim(3)
        let c = x.dim(4)

        var y = x.reshaped(n * d, h, w, c)
        y = conv(y)
        y = pixelShuffle2D(y, upscaleFactor: 2)
        return y.reshaped(n, d, h * 2, w * 2, c)
    }
}

package final class LTXTemporalPixelShuffleUpsampler: Module {
    @ModuleInfo(key: "conv") package var conv: Conv3d

    package init(midChannels: Int) {
        self._conv.wrappedValue = Conv3d(
            inputChannels: midChannels,
            outputChannels: 2 * midChannels,
            kernelSize: .init([3, 3, 3]),
            stride: .init([1, 1, 1]),
            padding: .init([1, 1, 1]),
            bias: true
        )
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        ltxTemporalPixelShuffle(conv(x), upscaleFactor: 2)
    }
}

package final class LTXLatentUpsampler: Module {
    package let inChannels: Int
    package let midChannels: Int
    package let numBlocksPerStage: Int

    @ModuleInfo(key: "initial_conv") package var initialConv: Conv3d
    @ModuleInfo(key: "initial_norm") package var initialNorm: LTXUpsamplerGroupNorm3d
    @ModuleInfo(key: "res_blocks") package var resBlocks: [LTXUpsamplerResBlock3D]
    @ModuleInfo(key: "upsampler") package var upsampler: LTXSpatialRationalResampler
    @ModuleInfo(key: "post_upsample_res_blocks") package var postUpsampleResBlocks: [LTXUpsamplerResBlock3D]
    @ModuleInfo(key: "final_conv") package var finalConv: Conv3d

    package init(inChannels: Int = 128, midChannels: Int = 1024, numBlocksPerStage: Int = 4) {
        self.inChannels = inChannels
        self.midChannels = midChannels
        self.numBlocksPerStage = numBlocksPerStage

        self._initialConv.wrappedValue = Conv3d(
            inputChannels: inChannels,
            outputChannels: midChannels,
            kernelSize: .init([3, 3, 3]),
            stride: .init([1, 1, 1]),
            padding: .init([1, 1, 1]),
            bias: true
        )
        self._initialNorm.wrappedValue = LTXUpsamplerGroupNorm3d(numGroups: 32, numChannels: midChannels, eps: 1e-5)
        self._resBlocks.wrappedValue = (0..<numBlocksPerStage).map { _ in
            LTXUpsamplerResBlock3D(channels: midChannels)
        }
        self._upsampler.wrappedValue = LTXSpatialRationalResampler(midChannels: midChannels, scale: 2.0)
        self._postUpsampleResBlocks.wrappedValue = (0..<numBlocksPerStage).map { _ in
            LTXUpsamplerResBlock3D(channels: midChannels)
        }
        self._finalConv.wrappedValue = Conv3d(
            inputChannels: midChannels,
            outputChannels: inChannels,
            kernelSize: .init([3, 3, 3]),
            stride: .init([1, 1, 1]),
            padding: .init([1, 1, 1]),
            bias: true
        )
    }

    package func callAsFunction(_ latent: MLXArray) -> MLXArray {
        precondition(latent.ndim == 5, "Expected NCDHW latent tensor")

        var x = latent.transposed(0, 2, 3, 4, 1)
        x = initialConv(x)
        x = initialNorm(x)
        x = silu(x)

        for block in resBlocks {
            x = block(x)
        }

        x = upsampler(x)

        for block in postUpsampleResBlocks {
            x = block(x)
        }

        x = finalConv(x)
        return x.transposed(0, 4, 1, 2, 3)
    }
}

package final class LTXTemporalLatentUpsampler: Module {
    package let inChannels: Int
    package let midChannels: Int
    package let numBlocksPerStage: Int

    @ModuleInfo(key: "initial_conv") package var initialConv: Conv3d
    @ModuleInfo(key: "initial_norm") package var initialNorm: LTXUpsamplerGroupNorm3d
    @ModuleInfo(key: "res_blocks") package var resBlocks: [LTXUpsamplerResBlock3D]
    @ModuleInfo(key: "upsampler") package var upsampler: LTXTemporalPixelShuffleUpsampler
    @ModuleInfo(key: "post_upsample_res_blocks") package var postUpsampleResBlocks: [LTXUpsamplerResBlock3D]
    @ModuleInfo(key: "final_conv") package var finalConv: Conv3d

    package init(inChannels: Int = 128, midChannels: Int = 512, numBlocksPerStage: Int = 4) {
        self.inChannels = inChannels
        self.midChannels = midChannels
        self.numBlocksPerStage = numBlocksPerStage

        self._initialConv.wrappedValue = Conv3d(
            inputChannels: inChannels,
            outputChannels: midChannels,
            kernelSize: .init([3, 3, 3]),
            stride: .init([1, 1, 1]),
            padding: .init([1, 1, 1]),
            bias: true
        )
        self._initialNorm.wrappedValue = LTXUpsamplerGroupNorm3d(
            numGroups: 32,
            numChannels: midChannels,
            eps: 1e-5
        )
        self._resBlocks.wrappedValue = (0..<numBlocksPerStage).map { _ in
            LTXUpsamplerResBlock3D(channels: midChannels)
        }
        self._upsampler.wrappedValue = LTXTemporalPixelShuffleUpsampler(midChannels: midChannels)
        self._postUpsampleResBlocks.wrappedValue = (0..<numBlocksPerStage).map { _ in
            LTXUpsamplerResBlock3D(channels: midChannels)
        }
        self._finalConv.wrappedValue = Conv3d(
            inputChannels: midChannels,
            outputChannels: inChannels,
            kernelSize: .init([3, 3, 3]),
            stride: .init([1, 1, 1]),
            padding: .init([1, 1, 1]),
            bias: true
        )
    }

    package func callAsFunction(_ latent: MLXArray) -> MLXArray {
        precondition(latent.ndim == 5, "Expected NCDHW latent tensor")

        var x = latent.transposed(0, 2, 3, 4, 1)
        x = silu(initialNorm(initialConv(x)))
        for block in resBlocks {
            x = block(x)
        }
        x = upsampler(x)
        x = x[0..., 1..., 0..., 0..., 0...]
        for block in postUpsampleResBlocks {
            x = block(x)
        }
        x = finalConv(x)
        return x.transposed(0, 4, 1, 2, 3)
    }
}
