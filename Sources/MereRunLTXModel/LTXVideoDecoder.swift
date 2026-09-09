import Foundation
import MLX
import MLXFast
import MLXNN

package final class LTXPixArtAlphaTimestepEmbedder: Module {
    @ModuleInfo(key: "timestep_embedder") package var timestepEmbedder: LTXTimestepEmbedding

    package init(embeddingDim: Int) {
        self._timestepEmbedder.wrappedValue = LTXTimestepEmbedding(
            inChannels: 256,
            timeEmbedDim: embeddingDim,
            outDim: embeddingDim
        )
    }

    package func callAsFunction(_ timestep: MLXArray, hiddenDType: DType) -> MLXArray {
        let projected = getTimestepEmbedding(
            timesteps: timestep,
            embeddingDim: 256,
            flipSinToCos: true,
            downscaleFreqShift: 0,
            scale: 1,
            maxPeriod: 10_000
        ).asType(hiddenDType)
        return timestepEmbedder(projected)
    }
}

package final class LTXDecoderBlock: Module {
    package enum Kind {
        case resnetGroup
        case upsample
    }

    package let kind: Kind
    package let channels: Int
    package let timestepConditioning: Bool
    package let stride: (Int, Int, Int)
    package let residualEnabled: Bool
    package let outChannelsReductionFactor: Int

    @ModuleInfo(key: "res_blocks") package var resBlocks: [LTXResnetBlock3DSimple]
    @ModuleInfo(key: "time_embedder") package var timeEmbedder: LTXPixArtAlphaTimestepEmbedder?
    @ModuleInfo(key: "conv") package var conv: LTXCausalConv3d?

    package init(
        channels: Int,
        numLayers: Int,
        timestepConditioning: Bool,
        spatialPaddingMode: LTXCausalConv3d.SpatialPaddingMode = .reflect
    ) {
        self.kind = .resnetGroup
        self.channels = channels
        self.timestepConditioning = timestepConditioning
        self.stride = (1, 1, 1)
        self.residualEnabled = false
        self.outChannelsReductionFactor = 1

        self._resBlocks.wrappedValue = (0..<numLayers).map { _ in
            LTXResnetBlock3DSimple(
                channels: channels,
                timestepConditioning: timestepConditioning,
                spatialPaddingMode: spatialPaddingMode
            )
        }
        self._timeEmbedder.wrappedValue = timestepConditioning
            ? LTXPixArtAlphaTimestepEmbedder(embeddingDim: channels * 4)
            : nil
        self._conv.wrappedValue = nil
    }

    package init(
        inChannels: Int,
        stride: (Int, Int, Int),
        residual: Bool,
        outChannelsReductionFactor: Int,
        spatialPaddingMode: LTXCausalConv3d.SpatialPaddingMode = .reflect
    ) {
        self.kind = .upsample
        self.channels = inChannels
        self.timestepConditioning = false
        self.stride = stride
        self.residualEnabled = residual
        self.outChannelsReductionFactor = outChannelsReductionFactor

        let multiplier = stride.0 * stride.1 * stride.2
        let outChannels = inChannels / outChannelsReductionFactor
        self._conv.wrappedValue = LTXCausalConv3d(
            inChannels: inChannels,
            outChannels: outChannels * multiplier,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1),
            spatialPaddingMode: spatialPaddingMode
        )
        self._resBlocks.wrappedValue = []
        self._timeEmbedder.wrappedValue = nil
    }

    package func callAsFunction(
        _ x: MLXArray,
        causal: Bool,
        timestep: MLXArray?
    ) -> MLXArray {
        switch kind {
        case .resnetGroup:
            var h = x
            let timestepEmbedding: MLXArray?
            if timestepConditioning, let timestep, let timeEmbedder {
                timestepEmbedding = timeEmbedder(timestep, hiddenDType: x.dtype)
            } else {
                timestepEmbedding = nil
            }
            for block in resBlocks {
                h = block(h, causal: causal, timestepEmbedding: timestepEmbedding)
            }
            return h

        case .upsample:
            guard let conv else { return x }
            let st = stride.0
            let sh = stride.1
            let sw = stride.2

            var residual: MLXArray?
            if residualEnabled {
                var up = depthToSpace3D(x, stride: stride)
                let repeats = max(1, (st * sh * sw) / outChannelsReductionFactor)
                up = tiled(up, repetitions: [1, repeats, 1, 1, 1])
                if st > 1 {
                    up = up[0..., 0..., 1..., 0..., 0...]
                }
                residual = up
            }

            var h = conv(x, causal: causal)
            h = depthToSpace3D(h, stride: stride)
            if st > 1 {
                h = h[0..., 0..., 1..., 0..., 0...]
            }
            if let residual {
                h = h + residual
            }
            return h
        }
    }
}

package final class LTXVideoDecoder: Module {
    package let patchSize: Int = 4
    package let timestepConditioning: Bool
    package let decodeNoiseScale: Float = 0.025
    package let decodeTimestep: Float = 0.05

    @ModuleInfo(key: "conv_in") package var convIn: LTXCausalConv3d
    @ModuleInfo(key: "up_blocks") package var upBlocks: [LTXDecoderBlock]
    @ModuleInfo(key: "conv_out") package var convOut: LTXCausalConv3d
    @ModuleInfo(key: "last_time_embedder") package var lastTimeEmbedder: LTXPixArtAlphaTimestepEmbedder
    @ModuleInfo(key: "last_scale_shift_table") package var lastScaleShiftTable: MLXArray

    package var latentsMean: MLXArray = MLX.zeros([128], dtype: .float32)
    package var latentsStd: MLXArray = MLX.ones([128], dtype: .float32)

    package init(timestepConditioning: Bool, architecture: LTXVideoVAEArchitecture = .legacy) {
        self.timestepConditioning = timestepConditioning
        let spatialPaddingMode: LTXCausalConv3d.SpatialPaddingMode = architecture == .ltx23Split ? .zeros : .reflect
        self._convIn.wrappedValue = LTXCausalConv3d(
            inChannels: 128,
            outChannels: 1024,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1),
            spatialPaddingMode: spatialPaddingMode
        )
        switch architecture {
        case .legacy:
            self._upBlocks.wrappedValue = [
                LTXDecoderBlock(channels: 1024, numLayers: 5, timestepConditioning: timestepConditioning),
                LTXDecoderBlock(inChannels: 1024, stride: (2, 2, 2), residual: true, outChannelsReductionFactor: 2),
                LTXDecoderBlock(channels: 512, numLayers: 5, timestepConditioning: timestepConditioning),
                LTXDecoderBlock(inChannels: 512, stride: (2, 2, 2), residual: true, outChannelsReductionFactor: 2),
                LTXDecoderBlock(channels: 256, numLayers: 5, timestepConditioning: timestepConditioning),
                LTXDecoderBlock(inChannels: 256, stride: (2, 2, 2), residual: true, outChannelsReductionFactor: 2),
                LTXDecoderBlock(channels: 128, numLayers: 5, timestepConditioning: timestepConditioning),
            ]

        case .ltx23Split:
            self._upBlocks.wrappedValue = [
                LTXDecoderBlock(
                    channels: 1024,
                    numLayers: 2,
                    timestepConditioning: timestepConditioning,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    inChannels: 1024,
                    stride: (2, 2, 2),
                    residual: false,
                    outChannelsReductionFactor: 2,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    channels: 512,
                    numLayers: 2,
                    timestepConditioning: timestepConditioning,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    inChannels: 512,
                    stride: (2, 2, 2),
                    residual: false,
                    outChannelsReductionFactor: 1,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    channels: 512,
                    numLayers: 4,
                    timestepConditioning: timestepConditioning,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    inChannels: 512,
                    stride: (2, 1, 1),
                    residual: false,
                    outChannelsReductionFactor: 2,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    channels: 256,
                    numLayers: 6,
                    timestepConditioning: timestepConditioning,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    inChannels: 256,
                    stride: (1, 2, 2),
                    residual: false,
                    outChannelsReductionFactor: 2,
                    spatialPaddingMode: spatialPaddingMode
                ),
                LTXDecoderBlock(
                    channels: 128,
                    numLayers: 4,
                    timestepConditioning: timestepConditioning,
                    spatialPaddingMode: spatialPaddingMode
                ),
            ]
        }
        self._convOut.wrappedValue = LTXCausalConv3d(
            inChannels: 128,
            outChannels: 3 * 4 * 4,
            kernelSize: (3, 3, 3),
            stride: (1, 1, 1),
            spatialPadding: (1, 1),
            spatialPaddingMode: spatialPaddingMode
        )
        self._lastTimeEmbedder.wrappedValue = LTXPixArtAlphaTimestepEmbedder(embeddingDim: 128 * 2)
        self._lastScaleShiftTable.wrappedValue = MLX.zeros([2, 128], dtype: .float32)
    }

    package func decode(sample: MLXArray, timestep: MLXArray?) -> MLXArray {
        let batch = sample.dim(0)
        var h = sample
        let debugDecoderPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_DECODER_PREFIX"]
        let debugDecoder = debugDecoderPrefix?.isEmpty == false
        var debugRoot: URL?
        var debugStem = ""

        if debugDecoder, let debugDecoderPrefix {
            let base = URL(fileURLWithPath: debugDecoderPrefix).standardizedFileURL
            let parent = base.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            debugRoot = parent
            debugStem = base.lastPathComponent
        }

        func dumpDecoder(_ name: String, _ array: MLXArray) {
            guard debugDecoder, let debugRoot else { return }
            let path = debugRoot.appendingPathComponent("\(debugStem)_\(name).npy")
            try? MLX.save(array: array, url: path)
        }

        dumpDecoder("input", h)

        if timestepConditioning {
            let noise = MLXRandom.normal(h.shape).asType(h.dtype) * MLXArray(decodeNoiseScale).asType(h.dtype)
            h = noise + (MLXArray(1.0 - decodeNoiseScale).asType(h.dtype) * h)
        }
        dumpDecoder("after_noise", h)

        h = denormalize(h)
        dumpDecoder("after_denormalize", h)

        var currentTimestep = timestep
        if currentTimestep == nil, timestepConditioning {
            currentTimestep = MLX.full([batch], values: MLXArray(decodeTimestep).asType(h.dtype))
        }

        var scaledTimestep: MLXArray?
        if timestepConditioning, let currentTimestep {
            scaledTimestep = currentTimestep * MLXArray(1000.0).asType(currentTimestep.dtype)
        }
        if let scaledTimestep {
            dumpDecoder("scaled_timestep", scaledTimestep)
        }

        h = convIn(h, causal: false)
        MLX.eval(h)
        dumpDecoder("after_conv_in", h)
        for block in upBlocks {
            h = block(h, causal: false, timestep: scaledTimestep)
            MLX.eval(h)
            Memory.clearCache()
            dumpDecoder("after_up_block_\(blockIndex(of: block, in: upBlocks) ?? -1)", h)
        }

        h = pixelNormChannels(h)
        MLX.eval(h)
        dumpDecoder("after_pixel_norm", h)
        if timestepConditioning, let scaledTimestep {
            let embedded = lastTimeEmbedder(scaledTimestep.reshaped(-1), hiddenDType: h.dtype)
            let ada = lastScaleShiftTable.reshaped(1, 2, 128, 1, 1)
                + embedded.reshaped(batch, 2, 128, 1, 1)
            let shift = ada[0..., 0, 0..., 0..., 0...]
            let scale = ada[0..., 1, 0..., 0..., 0...]
            h = h * (MLXArray(1.0).asType(h.dtype) + scale) + shift
            MLX.eval(h)
            dumpDecoder("after_last_ada", h)
        }

        h = silu(h)
        MLX.eval(h)
        dumpDecoder("after_silu", h)
        h = convOut(h, causal: false)
        MLX.eval(h)
        dumpDecoder("after_conv_out", h)
        let out = unpatchify3D(h, patchSizeHW: patchSize, patchSizeT: 1)
        MLX.eval(out)
        dumpDecoder("after_unpatchify", out)
        return out
    }

    private func denormalize(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let mean = latentsMean.asType(.float32).reshaped(1, -1, 1, 1, 1)
        let std = latentsStd.asType(.float32).reshaped(1, -1, 1, 1, 1)
        return (x.asType(.float32) * std + mean).asType(dtype)
    }
}
