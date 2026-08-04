import Foundation
import MLX
import MLXNN
import MLXRandom

final class MiniMaxH3CausalConv3D: Module {
    let kernel: (Int, Int, Int)
    let stride: (Int, Int, Int)
    let spatialPadding: Int
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernel: (Int, Int, Int),
        stride: (Int, Int, Int) = (1, 1, 1),
        spatialPadding: Int = 0
    ) {
        self.kernel = kernel
        self.stride = stride
        self.spatialPadding = spatialPadding
        _weight.wrappedValue = MLXArray.zeros([outputChannels, kernel.0, kernel.1, kernel.2, inputChannels])
        _bias.wrappedValue = MLXArray.zeros([outputChannels])
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        var hidden = value
        if kernel.0 > 1 {
            hidden = MLX.concatenated([
                MLXArray.zeros(
                    [hidden.dim(0), kernel.0 - 1, hidden.dim(2), hidden.dim(3), hidden.dim(4)],
                    dtype: hidden.dtype
                ),
                hidden,
            ], axis: 1)
        }
        if spatialPadding > 0 {
            hidden = MiniMaxH3CausalConv3D.reflectPad(hidden, amount: spatialPadding)
        }
        return MLX.conv3d(
            hidden,
            weight,
            stride: .init([stride.0, stride.1, stride.2]),
            padding: .init(0)
        ) + bias
    }

    static func reflectPad(_ value: MLXArray, amount: Int) -> MLXArray {
        var hidden = value
        let reverse = MLXArray(Array(Swift.stride(from: amount - 1, through: 0, by: -1)).map(Int32.init))
        let top = MLX.take(hidden[0..., 0..., 1..<(amount + 1), 0..., 0...], reverse, axis: 2)
        let bottom = MLX.take(
            hidden[0..., 0..., (hidden.dim(2) - amount - 1)..<(hidden.dim(2) - 1), 0..., 0...],
            reverse,
            axis: 2
        )
        hidden = MLX.concatenated([top, hidden, bottom], axis: 2)
        let left = MLX.take(hidden[0..., 0..., 0..., 1..<(amount + 1), 0...], reverse, axis: 3)
        let right = MLX.take(
            hidden[0..., 0..., 0..., (hidden.dim(3) - amount - 1)..<(hidden.dim(3) - 1), 0...],
            reverse,
            axis: 3
        )
        return MLX.concatenated([left, hidden, right], axis: 3)
    }
}

final class MiniMaxH3FrameGroupNorm: Module {
    let groups: Int
    let channels: Int
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(groups: Int = 32, channels: Int) {
        self.groups = groups
        self.channels = channels
        _weight.wrappedValue = MLXArray.ones([channels])
        _bias.wrappedValue = MLXArray.zeros([channels])
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        let batch = value.dim(0)
        let frames = value.dim(1)
        let height = value.dim(2)
        let width = value.dim(3)
        var hidden = value.asType(.float32).reshaped(batch * frames, height * width, groups, channels / groups)
        let mean = MLX.mean(hidden, axes: [1, 3], keepDims: true)
        let variance = MLX.mean((hidden - mean) * (hidden - mean), axes: [1, 3], keepDims: true)
        hidden = (hidden - mean) / MLX.sqrt(variance + 1e-6)
        hidden = hidden.reshaped(batch, frames, height, width, channels)
        return (hidden * weight.reshaped(1, 1, 1, 1, channels) + bias.reshaped(1, 1, 1, 1, channels))
            .asType(value.dtype)
    }
}

final class MiniMaxH3VideoEncoderResnet: Module {
    @ModuleInfo(key: "norm1") var firstNorm: MiniMaxH3FrameGroupNorm
    @ModuleInfo(key: "conv1") var firstConvolution: MiniMaxH3CausalConv3D
    @ModuleInfo(key: "norm2") var secondNorm: MiniMaxH3FrameGroupNorm
    @ModuleInfo(key: "conv2") var secondConvolution: MiniMaxH3CausalConv3D
    @ModuleInfo(key: "conv_shortcut") var shortcut: MiniMaxH3CausalConv3D?

    init(inputChannels: Int, outputChannels: Int) {
        _firstNorm.wrappedValue = MiniMaxH3FrameGroupNorm(channels: inputChannels)
        _firstConvolution.wrappedValue = MiniMaxH3CausalConv3D(
            inputChannels: inputChannels, outputChannels: outputChannels, kernel: (3, 3, 3), spatialPadding: 1
        )
        _secondNorm.wrappedValue = MiniMaxH3FrameGroupNorm(channels: outputChannels)
        _secondConvolution.wrappedValue = MiniMaxH3CausalConv3D(
            inputChannels: outputChannels, outputChannels: outputChannels, kernel: (3, 3, 3), spatialPadding: 1
        )
        _shortcut.wrappedValue = inputChannels == outputChannels ? nil : MiniMaxH3CausalConv3D(
            inputChannels: inputChannels, outputChannels: outputChannels, kernel: (1, 1, 1)
        )
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        var hidden = firstConvolution(MLXNN.silu(firstNorm(value)))
        hidden = secondConvolution(MLXNN.silu(secondNorm(hidden)))
        return hidden + (shortcut?(value) ?? value)
    }
}

final class MiniMaxH3VideoDownsampler: Module {
    let spatialStride: Int
    @ModuleInfo(key: "conv") var convolution: MiniMaxH3CausalConv3D

    init(channels: Int, temporalStride: Int, spatialStride: Int) {
        self.spatialStride = spatialStride
        _convolution.wrappedValue = MiniMaxH3CausalConv3D(
            inputChannels: channels,
            outputChannels: channels,
            kernel: (3, 3, 3),
            stride: (temporalStride, spatialStride, spatialStride)
        )
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        guard spatialStride == 2 else { return convolution(value) }
        let bottom = value[0..., 0..., (value.dim(2) - 2)..<(value.dim(2) - 1), 0..., 0...]
        let spatial = MLX.concatenated([value, bottom], axis: 2)
        let right = spatial[0..., 0..., 0..., (spatial.dim(3) - 2)..<(spatial.dim(3) - 1), 0...]
        return convolution(MLX.concatenated([spatial, right], axis: 3))
    }
}

final class MiniMaxH3VideoDownBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [MiniMaxH3VideoEncoderResnet]
    @ModuleInfo(key: "downsamplers") var downsamplers: [MiniMaxH3VideoDownsampler]

    init(inputChannels: Int, outputChannels: Int, temporalFactor: Int, spatialFactor: Int) {
        _resnets.wrappedValue = [
            MiniMaxH3VideoEncoderResnet(inputChannels: inputChannels, outputChannels: outputChannels),
            MiniMaxH3VideoEncoderResnet(inputChannels: outputChannels, outputChannels: outputChannels),
        ]
        _downsamplers.wrappedValue = temporalFactor * spatialFactor > 1
            ? [MiniMaxH3VideoDownsampler(
                channels: outputChannels, temporalStride: temporalFactor, spatialStride: spatialFactor
            )]
            : []
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        var hidden = resnets.reduce(value) { current, block in block(current) }
        for downsampler in downsamplers { hidden = downsampler(hidden) }
        return hidden
    }
}

public final class MiniMaxH3VideoEncoder: Module {
    @ModuleInfo(key: "conv_in") var input: MiniMaxH3CausalConv3D
    @ModuleInfo(key: "down_blocks") var blocks: [MiniMaxH3VideoDownBlock]
    @ModuleInfo(key: "norm_out") var outputNorm: MiniMaxH3FrameGroupNorm
    @ModuleInfo(key: "conv_out") var output: MiniMaxH3CausalConv3D

    public override init() {
        let channels = [128, 256, 256, 512, 512, 1_024]
        let inputChannels = [128, 128, 256, 256, 512, 512]
        let spatial = [2, 2, 2, 2, 1, 1]
        let temporal = [1, 2, 2, 1, 1, 1]
        _input.wrappedValue = MiniMaxH3CausalConv3D(
            inputChannels: 3, outputChannels: 128, kernel: (3, 3, 3), spatialPadding: 1
        )
        _blocks.wrappedValue = channels.indices.map { index in
            MiniMaxH3VideoDownBlock(
                inputChannels: inputChannels[index],
                outputChannels: channels[index],
                temporalFactor: temporal[index],
                spatialFactor: spatial[index]
            )
        }
        _outputNorm.wrappedValue = MiniMaxH3FrameGroupNorm(channels: 1_024)
        _output.wrappedValue = MiniMaxH3CausalConv3D(
            inputChannels: 1_024, outputChannels: 48, kernel: (3, 3, 3), spatialPadding: 1
        )
        super.init()
    }

    public func callAsFunction(_ video: MLXArray) -> MLXArray {
        var hidden = input(video)
        for block in blocks { hidden = block(hidden) }
        return output(MLXNN.silu(outputNorm(hidden)))
    }
}
