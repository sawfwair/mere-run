import Foundation
import MLX
import MLXFast
import MLXNN

package final class LTXCausalConv3d: Module {
    package enum SpatialPaddingMode {
        case zeros
        case reflect
    }

    @ModuleInfo(key: "conv") package var conv: Conv3d

    package let kernelSize: (Int, Int, Int)
    package let stride: (Int, Int, Int)
    package let temporalPadding: Int
    package let spatialPadding: (Int, Int)
    package let spatialPaddingMode: SpatialPaddingMode

    package init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: (Int, Int, Int),
        stride: (Int, Int, Int) = (1, 1, 1),
        spatialPadding: (Int, Int) = (1, 1),
        spatialPaddingMode: SpatialPaddingMode = .zeros
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.temporalPadding = kernelSize.0 - 1
        self.spatialPadding = spatialPadding
        self.spatialPaddingMode = spatialPaddingMode

        self._conv.wrappedValue = Conv3d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: .init([kernelSize.0, kernelSize.1, kernelSize.2]),
            stride: .init([stride.0, stride.1, stride.2]),
            padding: .init(0),
            bias: true
        )
    }

    package func callAsFunction(_ x: MLXArray, causal: Bool) -> MLXArray {
        var hidden = x

        if kernelSize.0 > 1 {
            if causal {
                if temporalPadding > 0 {
                    let firstFrame = hidden[0..., 0..., 0..<1, 0..., 0...]
                    let repeated = tiled(firstFrame, repetitions: [1, 1, temporalPadding, 1, 1])
                    hidden = MLX.concatenated([repeated, hidden], axis: 2)
                }
            } else {
                let padSize = (kernelSize.0 - 1) / 2
                if padSize > 0 {
                    let firstFrame = hidden[0..., 0..., 0..<1, 0..., 0...]
                    let lastFrame = hidden[0..., 0..., (hidden.dim(2) - 1)..., 0..., 0...]
                    let front = tiled(firstFrame, repetitions: [1, 1, padSize, 1, 1])
                    let back = tiled(lastFrame, repetitions: [1, 1, padSize, 1, 1])
                    hidden = MLX.concatenated([front, hidden, back], axis: 2)
                }
            }
        }

        hidden = hidden.transposed(0, 2, 3, 4, 1)
        if spatialPadding.0 > 0 || spatialPadding.1 > 0 {
            switch spatialPaddingMode {
            case .zeros:
                hidden = padded(hidden, widths: [
                    [0, 0],
                    [0, 0],
                    [spatialPadding.0, spatialPadding.0],
                    [spatialPadding.1, spatialPadding.1],
                    [0, 0],
                ])
            case .reflect:
                hidden = reflectPad2DNDHWC(hidden, padH: spatialPadding.0, padW: spatialPadding.1)
            }
        }

        hidden = conv(hidden)
        return hidden.transposed(0, 4, 1, 2, 3)
    }
}

package func reflectPad2DNDHWC(_ x: MLXArray, padH: Int, padW: Int) -> MLXArray {
    var y = x

    if padH > 0 {
        let top = y[0..., 0..., 1..<(padH + 1), 0..., 0...]
        let bottomStart = max(0, y.dim(2) - padH - 1)
        let bottomEnd = max(bottomStart, y.dim(2) - 1)
        let bottom = y[0..., 0..., bottomStart..<bottomEnd, 0..., 0...]
        let reverseH = MLXArray(Array(stride(from: padH - 1, through: 0, by: -1)).map(Int32.init))
        let topReflected = top.take(reverseH, axis: 2)
        let bottomReflected = bottom.take(reverseH, axis: 2)
        y = MLX.concatenated([topReflected, y, bottomReflected], axis: 2)
    }

    if padW > 0 {
        let left = y[0..., 0..., 0..., 1..<(padW + 1), 0...]
        let rightStart = max(0, y.dim(3) - padW - 1)
        let rightEnd = max(rightStart, y.dim(3) - 1)
        let right = y[0..., 0..., 0..., rightStart..<rightEnd, 0...]
        let reverseW = MLXArray(Array(stride(from: padW - 1, through: 0, by: -1)).map(Int32.init))
        let leftReflected = left.take(reverseW, axis: 3)
        let rightReflected = right.take(reverseW, axis: 3)
        y = MLX.concatenated([leftReflected, y, rightReflected], axis: 3)
    }

    return y
}

package final class LTXResnetBlock3DSimple: Module {
    @ModuleInfo(key: "conv1") package var conv1: LTXCausalConv3d
    @ModuleInfo(key: "conv2") package var conv2: LTXCausalConv3d
    @ModuleInfo(key: "scale_shift_table") package var scaleShiftTable: MLXArray

    package let channels: Int
    package let timestepConditioning: Bool

    package init(
        channels: Int,
        timestepConditioning: Bool,
        spatialPaddingMode: LTXCausalConv3d.SpatialPaddingMode = .zeros
    ) {
        self.channels = channels
        self.timestepConditioning = timestepConditioning
        self._conv1.wrappedValue = LTXCausalConv3d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1),
            spatialPaddingMode: spatialPaddingMode
        )
        self._conv2.wrappedValue = LTXCausalConv3d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1),
            spatialPaddingMode: spatialPaddingMode
        )
        self._scaleShiftTable.wrappedValue = MLX.zeros([4, channels], dtype: .float32)
    }

    package func callAsFunction(
        _ x: MLXArray,
        causal: Bool,
        timestepEmbedding: MLXArray?
    ) -> MLXArray {
        let residual = x
        var h = pixelNormChannels(x)

        if timestepConditioning, let timestepEmbedding {
            let batch = x.dim(0)
            let ada = scaleShiftTable.reshaped(1, 4, channels, 1, 1)
                + timestepEmbedding.reshaped(batch, 4, channels, 1, 1)
            let shift1 = ada[0..., 0, 0..., 0..., 0...]
            let scale1 = ada[0..., 1, 0..., 0..., 0...]
            h = h * (MLXArray(1.0).asType(h.dtype) + scale1) + shift1
        }

        h = silu(h)
        h = conv1(h, causal: causal)
        h = pixelNormChannels(h)

        if timestepConditioning, let timestepEmbedding {
            let batch = x.dim(0)
            let ada = scaleShiftTable.reshaped(1, 4, channels, 1, 1)
                + timestepEmbedding.reshaped(batch, 4, channels, 1, 1)
            let shift2 = ada[0..., 2, 0..., 0..., 0...]
            let scale2 = ada[0..., 3, 0..., 0..., 0...]
            h = h * (MLXArray(1.0).asType(h.dtype) + scale2) + shift2
        }

        h = silu(h)
        h = conv2(h, causal: causal)
        return residual + h
    }
}

package enum LTXVideoVAEArchitecture: Equatable {
    case legacy
    case ltx23Split
}

package final class LTXVideoEncoder: Module {
    package let patchSize: Int = 4
    package let latentChannels: Int = 128

    @ModuleInfo(key: "conv_in") package var convIn: LTXCausalConv3d
    @ModuleInfo(key: "down_blocks") package var downBlocks: [LTXEncoderBlock]
    @ModuleInfo(key: "conv_out") package var convOut: LTXCausalConv3d

    package var latentsMean: MLXArray = MLX.zeros([128], dtype: .float32)
    package var latentsStd: MLXArray = MLX.ones([128], dtype: .float32)

    package init(architecture: LTXVideoVAEArchitecture = .legacy) {
        self._convIn.wrappedValue = LTXCausalConv3d(
            inChannels: 3 * 4 * 4,
            outChannels: 128,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1)
        )

        switch architecture {
        case .legacy:
            self._downBlocks.wrappedValue = [
                LTXEncoderBlock(channels: 128, numLayers: 4),
                LTXEncoderBlock(inChannels: 128, outChannels: 256, stride: (1, 2, 2)),
                LTXEncoderBlock(channels: 256, numLayers: 6),
                LTXEncoderBlock(inChannels: 256, outChannels: 512, stride: (2, 1, 1)),
                LTXEncoderBlock(channels: 512, numLayers: 6),
                LTXEncoderBlock(inChannels: 512, outChannels: 1024, stride: (2, 2, 2)),
                LTXEncoderBlock(channels: 1024, numLayers: 2),
                LTXEncoderBlock(inChannels: 1024, outChannels: 2048, stride: (2, 2, 2)),
                LTXEncoderBlock(channels: 2048, numLayers: 2),
            ]

            self._convOut.wrappedValue = LTXCausalConv3d(
                inChannels: 2048,
                outChannels: 129,
                kernelSize: (3, 3, 3),
                stride: (1, 1, 1),
                spatialPadding: (1, 1)
            )

        case .ltx23Split:
            self._downBlocks.wrappedValue = [
                LTXEncoderBlock(channels: 128, numLayers: 4),
                LTXEncoderBlock(inChannels: 128, outChannels: 256, stride: (1, 2, 2)),
                LTXEncoderBlock(channels: 256, numLayers: 6),
                LTXEncoderBlock(inChannels: 256, outChannels: 512, stride: (2, 1, 1)),
                LTXEncoderBlock(channels: 512, numLayers: 4),
                LTXEncoderBlock(inChannels: 512, outChannels: 1024, stride: (2, 2, 2)),
                LTXEncoderBlock(channels: 1024, numLayers: 2),
                LTXEncoderBlock(inChannels: 1024, outChannels: 1024, stride: (2, 2, 2)),
                LTXEncoderBlock(channels: 1024, numLayers: 2),
            ]

            self._convOut.wrappedValue = LTXCausalConv3d(
                inChannels: 1024,
                outChannels: 129,
                kernelSize: (3, 3, 3),
                stride: (1, 1, 1),
                spatialPadding: (1, 1)
            )
        }

        super.init()
    }

    package func encode(image: MLXArray) -> MLXArray {
        var sample = patchify3D(image, patchSizeHW: patchSize, patchSizeT: 1)
        sample = convIn(sample, causal: true)

        for block in downBlocks {
            sample = block(sample, causal: true)
        }

        sample = pixelNormChannels(sample)
        sample = silu(sample)
        sample = convOut(sample, causal: true)

        let means = sample[0..., 0..<latentChannels, 0..., 0..., 0...]
        return normalizeLatents(means)
    }

    private func normalizeLatents(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let mean = latentsMean.asType(.float32).reshaped(1, -1, 1, 1, 1)
        let std = latentsStd.asType(.float32).reshaped(1, -1, 1, 1, 1)
        return ((x.asType(.float32) - mean) / std).asType(dtype)
    }
}

package final class LTXEncoderBlock: Module {
    package enum Kind {
        case resnetGroup
        case downsample
    }

    package let kind: Kind
    package let stride: (Int, Int, Int)
    package let outChannels: Int
    package let groupSize: Int

    @ModuleInfo(key: "res_blocks") package var resBlocks: [LTXResnetBlock3DSimple]
    @ModuleInfo(key: "conv") package var conv: LTXCausalConv3d?

    package init(channels: Int, numLayers: Int) {
        self.kind = .resnetGroup
        self.stride = (1, 1, 1)
        self.outChannels = channels
        self.groupSize = 1
        self._resBlocks.wrappedValue = (0..<numLayers).map { _ in
            LTXResnetBlock3DSimple(channels: channels, timestepConditioning: false)
        }
        self._conv.wrappedValue = nil
    }

    package init(inChannels: Int, outChannels: Int, stride: (Int, Int, Int)) {
        self.kind = .downsample
        self.stride = stride
        self.outChannels = outChannels
        let multiplier = stride.0 * stride.1 * stride.2
        self.groupSize = max(1, inChannels * multiplier / outChannels)
        let convOutChannels = max(1, outChannels / multiplier)
        self._conv.wrappedValue = LTXCausalConv3d(
            inChannels: inChannels,
            outChannels: convOutChannels,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1)
        )
        self._resBlocks.wrappedValue = []
    }

    package func callAsFunction(_ x: MLXArray, causal: Bool) -> MLXArray {
        switch kind {
        case .resnetGroup:
            var h = x
            for block in resBlocks {
                h = block(h, causal: causal, timestepEmbedding: nil)
            }
            return h

        case .downsample:
            guard let conv else { return x }
            let st = stride.0
            let sh = stride.1
            let sw = stride.2

            var h = x
            if st == 2 {
                h = MLX.concatenated([h[0..., 0..., 0..<1, 0..., 0...], h], axis: 2)
            }

            let padD = (st - (h.dim(2) % st)) % st
            let padH = (sh - (h.dim(3) % sh)) % sh
            let padW = (sw - (h.dim(4) % sw)) % sw
            if padD > 0 || padH > 0 || padW > 0 {
                h = padded(h, widths: [
                    [0, 0],
                    [0, 0],
                    [0, padD],
                    [0, padH],
                    [0, padW],
                ])
            }

            let depthInput = spaceToDepth3D(h, stride: stride)
            let b = depthInput.dim(0)
            let d = depthInput.dim(2)
            let hh = depthInput.dim(3)
            let ww = depthInput.dim(4)

            let reduced = MLX.mean(
                depthInput.reshaped(b, outChannels, groupSize, d, hh, ww),
                axis: 2
            )
            let convOut = spaceToDepth3D(conv(h, causal: causal), stride: stride)
            return convOut + reduced
        }
    }
}
