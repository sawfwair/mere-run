import Foundation
import MLX
import MLXFast
import MLXNN

public struct MiniMaxH3VideoDecoderConfiguration: Hashable, Sendable {
    public let latentChannels: Int
    public let outputChannels: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let layerCount: Int
    public let headCount: Int
    public let headDimension: Int
    public let registerTokenCount: Int
    public let feedForwardMultiplier: Int
    public let rotaryDimensionRatio: Float
    public let rotaryTheta: Float

    public init(
        latentChannels: Int = 24,
        outputChannels: Int = 3,
        patchSize: Int = 16,
        temporalPatchSize: Int = 4,
        layerCount: Int = 36,
        headCount: Int = 32,
        headDimension: Int = 64,
        registerTokenCount: Int = 4,
        feedForwardMultiplier: Int = 4,
        rotaryDimensionRatio: Float = 0.75,
        rotaryTheta: Float = 100
    ) {
        self.latentChannels = latentChannels
        self.outputChannels = outputChannels
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.headDimension = headDimension
        self.registerTokenCount = registerTokenCount
        self.feedForwardMultiplier = feedForwardMultiplier
        self.rotaryDimensionRatio = rotaryDimensionRatio
        self.rotaryTheta = rotaryTheta
    }

    var hiddenSize: Int { headCount * headDimension }
    var rotaryDimension: Int { Int(Float(headDimension) * rotaryDimensionRatio) }
}

final class MiniMaxH3VideoDecoderAttention: Module {
    let headCount: Int
    let headDimension: Int
    let rotaryDimension: Int

    @ModuleInfo(key: "to_qkv") var queryKeyValue: Linear
    @ModuleInfo(key: "to_out") var output: Linear

    init(configuration: MiniMaxH3VideoDecoderConfiguration) {
        headCount = configuration.headCount
        headDimension = configuration.headDimension
        rotaryDimension = configuration.rotaryDimension
        let hidden = configuration.hiddenSize
        _queryKeyValue.wrappedValue = Linear(hidden, 3 * hidden, bias: true)
        _output.wrappedValue = Linear(hidden, hidden, bias: true)
    }

    func callAsFunction(_ input: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let projected = MLX.split(queryKeyValue(input), parts: 3, axis: -1)
        var q = projected[0].reshaped(batch, sequence, headCount, headDimension)
        var k = projected[1].reshaped(batch, sequence, headCount, headDimension)
        let v = projected[2].reshaped(batch, sequence, headCount, headDimension)

        q = MiniMaxH3VideoDecoderAttention.rmsNormalizeHeads(q)
        k = MiniMaxH3VideoDecoderAttention.rmsNormalizeHeads(k)
        q = applyRotary(q, cosine: cosine, sine: sine)
        k = applyRotary(k, cosine: cosine, sine: sine)

        let attended = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3),
            keys: k.transposed(0, 2, 1, 3),
            values: v.transposed(0, 2, 1, 3),
            scale: 1 / sqrt(Float(headDimension)),
            mask: .none
        )
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, -1))
    }

    private static func rmsNormalizeHeads(_ value: MLXArray) -> MLXArray {
        let value32 = value.asType(.float32)
        let denominator = MLX.sqrt(MLX.mean(value32 * value32, axis: -1, keepDims: true) + 1e-5)
        return (value32 / denominator).asType(value.dtype)
    }

    private func applyRotary(_ value: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let rotary = value[0..., 0..., 0..., 0..<rotaryDimension]
        let pass = value[0..., 0..., 0..., rotaryDimension...]
        let halves = MLX.split(rotary, parts: 2, axis: -1)
        let rotated = MLX.concatenated([-halves[1], halves[0]], axis: -1)
        return MLX.concatenated([rotary * cosine + rotated * sine, pass], axis: -1)
    }
}

final class MiniMaxH3VideoDecoderFeedForward: Module {
    @ModuleInfo(key: "linear_in") var input: Linear
    @ModuleInfo(key: "linear_out") var output: Linear

    init(configuration: MiniMaxH3VideoDecoderConfiguration) {
        let hidden = configuration.hiddenSize
        let intermediate = hidden * configuration.feedForwardMultiplier
        _input.wrappedValue = Linear(hidden, intermediate * 2, bias: true)
        _output.wrappedValue = Linear(intermediate, hidden, bias: true)
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        let parts = MLX.split(input(value), parts: 2, axis: -1)
        return output(MLXNN.silu(parts[0]) * parts[1])
    }
}

final class MiniMaxH3VideoDecoderBlock: Module {
    @ModuleInfo(key: "norm1") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attn") var attention: MiniMaxH3VideoDecoderAttention
    @ParameterInfo(key: "scale1") var attentionScale: MLXArray
    @ModuleInfo(key: "norm2") var feedForwardNorm: RMSNorm
    @ModuleInfo(key: "ff") var feedForward: MiniMaxH3VideoDecoderFeedForward
    @ParameterInfo(key: "scale2") var feedForwardScale: MLXArray

    init(configuration: MiniMaxH3VideoDecoderConfiguration) {
        _attentionNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: 1e-5)
        _attention.wrappedValue = MiniMaxH3VideoDecoderAttention(configuration: configuration)
        _attentionScale.wrappedValue = MLXArray.zeros([configuration.hiddenSize])
        _feedForwardNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: 1e-5)
        _feedForward.wrappedValue = MiniMaxH3VideoDecoderFeedForward(configuration: configuration)
        _feedForwardScale.wrappedValue = MLXArray.zeros([configuration.hiddenSize])
    }

    func callAsFunction(_ value: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        var hidden = value
        hidden = hidden + attention(attentionNorm(hidden), cosine: cosine, sine: sine) * attentionScale
        return hidden + feedForward(feedForwardNorm(hidden)) * feedForwardScale
    }
}

public final class MiniMaxH3VideoDecoder: Module {
    public let configuration: MiniMaxH3VideoDecoderConfiguration

    @ModuleInfo(key: "proj_in") var input: Linear
    @ParameterInfo(key: "register_tokens") var registerTokens: MLXArray
    @ModuleInfo(key: "transformer_blocks") var blocks: [MiniMaxH3VideoDecoderBlock]
    @ModuleInfo(key: "norm_out") var outputNorm: LayerNorm
    @ModuleInfo(key: "proj_out") var output: Linear

    public init(configuration: MiniMaxH3VideoDecoderConfiguration = .init()) {
        self.configuration = configuration
        _input.wrappedValue = Linear(configuration.latentChannels, configuration.hiddenSize, bias: true)
        _registerTokens.wrappedValue = MLXArray.zeros([
            1, configuration.registerTokenCount, configuration.hiddenSize,
        ])
        _blocks.wrappedValue = (0..<configuration.layerCount).map { _ in
            MiniMaxH3VideoDecoderBlock(configuration: configuration)
        }
        _outputNorm.wrappedValue = LayerNorm(dimensions: configuration.hiddenSize, eps: 1e-5)
        _output.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.outputChannels * configuration.temporalPatchSize
                * configuration.patchSize * configuration.patchSize,
            bias: true
        )
    }

    /// Decodes denormalized `[B, C, T, H, W]` H3 latents to ImageNet-normalized RGB.
    public func callAsFunction(_ latents: MLXArray) -> MLXArray {
        precondition(latents.ndim == 5 && latents.dim(1) == configuration.latentChannels)
        let batch = latents.dim(0)
        let frames = latents.dim(2)
        let height = latents.dim(3)
        let width = latents.dim(4)
        var hidden = input(
            latents.transposed(0, 2, 3, 4, 1)
                .reshaped(batch, frames * height * width, configuration.latentChannels)
        )
        let patchCount = hidden.dim(1)
        let registers = MLX.tiled(registerTokens, repetitions: [batch, 1, 1])
        let zeroToken = MLXArray.zeros([batch, 1, configuration.hiddenSize], dtype: hidden.dtype)
        hidden = MLX.concatenated([hidden, registers.asType(hidden.dtype), zeroToken], axis: 1)
        let (cosine, sine) = rotaryEmbedding(
            batch: batch,
            frames: frames,
            height: height,
            width: width,
            dtype: hidden.dtype
        )
        for block in blocks {
            hidden = block(hidden, cosine: cosine, sine: sine)
        }
        hidden = output(outputNorm(hidden))[0..., 0..<patchCount, 0...]
        let patch = configuration.patchSize
        let temporalPatch = configuration.temporalPatchSize
        return hidden
            .reshaped(batch, frames, height, width, configuration.outputChannels, temporalPatch, patch, patch)
            .transposed(0, 4, 1, 5, 2, 6, 3, 7)
            .reshaped(batch, configuration.outputChannels, frames * temporalPatch, height * patch, width * patch)
    }

    private func rotaryEmbedding(
        batch: Int,
        frames: Int,
        height: Int,
        width: Int,
        dtype: DType
    ) -> (MLXArray, MLXArray) {
        let axisWidth = configuration.rotaryDimension / 6
        let inverse = MLX.pow(
            MLXArray(configuration.rotaryTheta),
            -MLXArray(stride(from: 0, to: axisWidth * 2, by: 2).map(Float.init))
                / Float(axisWidth * 2)
        )
        func normalized(_ count: Int) -> MLXArray {
            (2 * ((MLXArray(0..<count).asType(.float32) + 0.5) / Float(count)) - 1) * (2 * Float.pi)
        }
        let t = normalized(frames).reshaped(frames, 1, 1, 1) * inverse
        let h = normalized(height).reshaped(1, height, 1, 1) * inverse
        let w = normalized(width).reshaped(1, 1, width, 1) * inverse
        var angles = MLX.concatenated([
            MLX.broadcast(t, to: [frames, height, width, axisWidth]),
            MLX.broadcast(h, to: [frames, height, width, axisWidth]),
            MLX.broadcast(w, to: [frames, height, width, axisWidth]),
        ], axis: -1).reshaped(1, frames * height * width, 1, configuration.rotaryDimension / 2)
        angles = MLX.concatenated([angles, angles], axis: -1)
        let suffix = MLXArray.zeros([
            1, configuration.registerTokenCount + 1, 1, configuration.rotaryDimension,
        ])
        angles = MLX.tiled(MLX.concatenated([angles, suffix], axis: 1), repetitions: [batch, 1, configuration.headCount, 1])
        return (MLX.cos(angles).asType(dtype), MLX.sin(angles).asType(dtype))
    }
}

public final class MiniMaxH3VideoVAE: Module {
    static let defaultSpatialTileSize = 256
    static let minimumSpatialTileOverlap = 64

    var spatialTileSize = defaultSpatialTileSize

    @ModuleInfo(key: "encoder") public var encoder: MiniMaxH3VideoEncoder
    @ModuleInfo(key: "quant_conv") var quantConvolution: Conv3d
    @ModuleInfo(key: "post_quant_conv") var postQuantConvolution: Conv3d
    @ModuleInfo(key: "decoder") public var decoder: MiniMaxH3VideoDecoder
    @ParameterInfo(key: "latents_mean") var latentMean: MLXArray
    @ParameterInfo(key: "latents_std") var latentStandardDeviation: MLXArray

    public override init() {
        _encoder.wrappedValue = MiniMaxH3VideoEncoder()
        _quantConvolution.wrappedValue = Conv3d(
            inputChannels: 48,
            outputChannels: 48,
            kernelSize: .init(1),
            bias: true
        )
        _postQuantConvolution.wrappedValue = Conv3d(
            inputChannels: 24,
            outputChannels: 24,
            kernelSize: .init(1),
            bias: true
        )
        _decoder.wrappedValue = MiniMaxH3VideoDecoder()
        _latentMean.wrappedValue = MLXArray.zeros([24])
        _latentStandardDeviation.wrappedValue = MLXArray.ones([24])
        super.init()
    }

    static func mapCheckpointWeight(key rawKey: String, value: MLXArray) -> [(String, MLXArray)] {
        if rawKey == "decoder.mask_token" { return [] }
        var key = rawKey
        key = key.replacingOccurrences(of: "decoder.x_embedder.", with: "decoder.proj_in.")
        key = key.replacingOccurrences(of: ".ff.w1.", with: ".ff.linear_in.")
        key = key.replacingOccurrences(of: ".ff.w2.", with: ".ff.linear_out.")
        key = key.replacingOccurrences(of: "encoder.down.", with: "encoder.down_blocks.")
        key = key.replacingOccurrences(of: ".block.", with: ".resnets.")
        key = key.replacingOccurrences(of: ".nin_shortcut.", with: ".conv_shortcut.")
        if key.contains(".downsample.conv.") {
            key = key.replacingOccurrences(of: ".downsample.conv.", with: ".downsamplers.0.conv.")
        }

        if key.contains(".attn.to_qkv.") {
            // The released VAE projects to [head, qkv, headDimension], not
            // [qkv, head, headDimension]. Deinterleave the per-head groups
            // before loading the fused global-QKV MLX Linear module.
            let headCount = 32
            let headDimension = 64
            precondition(value.dim(0) == headCount * 3 * headDimension)
            let trailingShape = Array(value.shape.dropFirst())
            let grouped = value.reshaped([headCount, 3, headDimension] + trailingShape)
            let pieces = MLX.split(grouped, parts: 3, axis: 1).map {
                $0.squeezed(axis: 1).reshaped([headCount * headDimension] + trailingShape)
            }
            return [(key, MLX.concatenated(pieces, axis: 0))]
        }
        if value.ndim == 5 {
            return [(key, value.transposed(0, 2, 3, 4, 1))]
        }
        return [(key, value)]
    }

    /// Encodes one prepared RGB keyframe `[1, H, W, 3]` with H3's fixed
    /// posterior seed and returns normalized `[1, 24, 1, H/16, W/16]` latents.
    public func encodeKeyframe(_ rgb: MLXArray) -> MLXArray {
        precondition(rgb.ndim == 4 && rgb.dim(0) == 1 && rgb.dim(3) == 3)
        let meanRGB = MLXArray([Float(0.485), 0.456, 0.406]).reshaped(1, 1, 1, 3)
        let stdRGB = MLXArray([Float(0.229), 0.224, 0.225]).reshaped(1, 1, 1, 3)
        let normalized = ((rgb - meanRGB) / stdRGB).expandedDimensions(axis: 1)
        let moments = encodeClip(normalized)
        return sampleAndNormalize(moments)
    }

    /// Encodes prepared `[1, T, H, W, 3]` RGB reference video. Frames are
    /// repeated to 17-frame chunks and H3's three trailing latent tokens are
    /// dropped after concatenation, matching the released Ref2VA pipeline.
    public func encodeReferenceVideo(_ rgb: MLXArray) -> MLXArray {
        precondition(rgb.ndim == 5 && rgb.dim(0) == 1 && rgb.dim(4) == 3 && rgb.dim(1) > 0)
        let meanRGB = MLXArray([Float(0.485), 0.456, 0.406]).reshaped(1, 1, 1, 1, 3)
        let stdRGB = MLXArray([Float(0.229), 0.224, 0.225]).reshaped(1, 1, 1, 1, 3)
        var normalized = (rgb - meanRGB) / stdRGB
        let remainder = normalized.dim(1) % MiniMaxH3Geometry.videoFramesPerChunk
        if remainder != 0 {
            let count = MiniMaxH3Geometry.videoFramesPerChunk - remainder
            let last = normalized[0..., (normalized.dim(1) - 1)..., 0..., 0..., 0...]
            normalized = MLX.concatenated(
                [normalized, MLX.tiled(last, repetitions: [1, count, 1, 1, 1])],
                axis: 1
            )
        }
        var chunks: [MLXArray] = []
        for start in stride(from: 0, to: normalized.dim(1), by: MiniMaxH3Geometry.videoFramesPerChunk) {
            chunks.append(encodeClip(
                normalized[0..., start..<(start + MiniMaxH3Geometry.videoFramesPerChunk), 0..., 0..., 0...]
            ))
        }
        let moments = MLX.concatenated(chunks, axis: 1)
        precondition(moments.dim(1) > 3)
        return sampleAndNormalize(moments[0..., 0..<(moments.dim(1) - 3), 0..., 0..., 0...])
    }

    private func sampleAndNormalize(_ moments: MLXArray) -> MLXArray {
        let parts = MLX.split(moments, parts: 2, axis: -1)
        MLXRandom.seed(42)
        let logVariance = MLX.clip(parts[1], min: -30, max: 20)
        let sample = parts[0] + MLX.exp(0.5 * logVariance) * MLXRandom.normal(parts[0].shape)
        let channelFirst = sample.asType(.float16).asType(.float32).transposed(0, 4, 1, 2, 3)
        return (channelFirst - latentMean.reshaped(1, 24, 1, 1, 1))
            / latentStandardDeviation.reshaped(1, 24, 1, 1, 1)
    }

    private func encodeClip(_ video: MLXArray) -> MLXArray {
        let pixelHeight = video.dim(2)
        let pixelWidth = video.dim(3)
        let y = Self.tilePlan(length: pixelHeight, tileSize: spatialTileSize)
        let x = Self.tilePlan(length: pixelWidth, tileSize: spatialTileSize)
        var tiles: [MLXArray] = []
        for (yIndex, yStart) in y.starts.enumerated() {
            for (xIndex, xStart) in x.starts.enumerated() {
                tiles.append(video[
                    0..., 0...,
                    yStart..<(yStart + y.lengths[yIndex]),
                    xStart..<(xStart + x.lengths[xIndex]),
                    0...
                ])
            }
        }
        let encodedTiles = MLX.split(
            quantConvolution(encoder(MLX.concatenated(tiles, axis: 0))),
            parts: tiles.count,
            axis: 0
        )
        var rows: [[MLXArray]] = []
        rows.reserveCapacity(y.starts.count)
        for rowIndex in y.starts.indices {
            let start = rowIndex * x.starts.count
            rows.append(Array(encodedTiles[start..<(start + x.starts.count)]))
        }

        let yOverlaps = y.overlaps.map { $0 / 16 }
        let xOverlaps = x.overlaps.map { $0 / 16 }
        var stitchedRows: [MLXArray] = []
        for rowIndex in rows.indices {
            var pieces: [MLXArray] = []
            for columnIndex in rows[rowIndex].indices {
                var tile = rows[rowIndex][columnIndex]
                if rowIndex > 0 {
                    tile = Self.blend(rows[rowIndex - 1][columnIndex], tile, extent: yOverlaps[rowIndex - 1], axis: 2)
                }
                if columnIndex > 0 {
                    tile = Self.blend(rows[rowIndex][columnIndex - 1], tile, extent: xOverlaps[columnIndex - 1], axis: 3)
                }
                if rowIndex + 1 < rows.count {
                    tile = tile[0..., 0..., 0..<(tile.dim(2) - yOverlaps[rowIndex]), 0..., 0...]
                }
                if columnIndex + 1 < rows[rowIndex].count {
                    tile = tile[0..., 0..., 0..., 0..<(tile.dim(3) - xOverlaps[columnIndex]), 0...]
                }
                pieces.append(tile)
            }
            stitchedRows.append(MLX.concatenated(pieces, axis: 3))
        }
        return MLX.concatenated(stitchedRows, axis: 2)
    }

    /// Converts normalized H3 latents to `[B, T, H, W, 3]` RGB in `[0, 1]`.
    public func decode(_ normalizedLatents: MLXArray) -> MLXArray {
        precondition(normalizedLatents.dim(2) >= 7, "H3 video decode requires at least seven latent frames")
        let mean = latentMean.reshaped(1, 24, 1, 1, 1)
        let standardDeviation = latentStandardDeviation.reshaped(1, 24, 1, 1, 1)
        var denormalized = normalizedLatents * standardDeviation + mean
        let tokensPerChunk = 5
        let tokenOverlap = 2
        let tokenDrop = 3
        let pixelFramesPerChunk = 20
        let framePrePadding = 3
        let frameOverlap = 5
        let padTokens = (-(denormalized.dim(2) + tokenDrop)).quotientAndRemainder(dividingBy: tokensPerChunk).remainder
        let resolvedPadTokens = padTokens == 0 ? 0 : padTokens + tokensPerChunk
        let originalTokenCount = denormalized.dim(2)
        if resolvedPadTokens > 0 {
            let last = denormalized[0..., 0..., (originalTokenCount - 1)..., 0..., 0...]
            denormalized = MLX.concatenated(
                [denormalized, MLX.tiled(last, repetitions: [1, 1, resolvedPadTokens, 1, 1])],
                axis: 2
            )
        }
        let chunkCount = (originalTokenCount + tokenDrop + resolvedPadTokens) / tokensPerChunk - 1
        var decodedChunks: [MLXArray] = []
        var overlap: MLXArray?
        let tileDecoder = MLX.compile { (channelLast: MLXArray) -> MLXArray in
            let projected = self.postQuantConvolution(channelLast).transposed(0, 4, 1, 2, 3)
            return self.decoder(projected)
        }
        for index in 0..<chunkCount {
            let start = index * tokensPerChunk
            let clip = decodeClip(
                denormalized[0..., 0..., start..<(start + tokensPerChunk + tokenOverlap), 0..., 0...],
                tileDecoder: tileDecoder
            )
            MLX.eval(clip)
            for part in 0..<2 {
                let frameStart = part * pixelFramesPerChunk
                let frameEnd = min(frameStart + pixelFramesPerChunk, clip.dim(2))
                guard frameEnd > frameStart + framePrePadding else { continue }
                var chunk = clip[0..., 0..., (frameStart + framePrePadding)..<frameEnd, 0..., 0...]
                if part == 0 {
                    if let overlap { chunk = Self.blend(overlap, chunk, extent: frameOverlap, axis: 2) }
                    decodedChunks.append(chunk)
                } else {
                    overlap = chunk
                }
            }
        }
        if let overlap { decodedChunks.append(overlap) }
        var imageNet = MLX.concatenated(decodedChunks, axis: 2)
        if resolvedPadTokens > 0 {
            let intraTail = 1
            var padFrames = 0
            for index in 0..<resolvedPadTokens {
                padFrames += (originalTokenCount + index).isMultiple(of: tokensPerChunk) ? intraTail : 4
            }
            imageNet = imageNet[0..., 0..., 0..<(imageNet.dim(2) - padFrames), 0..., 0...]
        }
        imageNet = imageNet.transposed(0, 2, 3, 4, 1)
        let meanRGB = MLXArray([Float(0.485), 0.456, 0.406]).reshaped(1, 1, 1, 1, 3)
        let stdRGB = MLXArray([Float(0.229), 0.224, 0.225]).reshaped(1, 1, 1, 1, 3)
        return MLX.clip(imageNet * stdRGB + meanRGB, min: 0, max: 1)
    }

    private func decodeClip(
        _ latent: MLXArray,
        tileDecoder: @Sendable (MLXArray) -> MLXArray
    ) -> MLXArray {
        let pixelHeight = latent.dim(3) * 16
        let pixelWidth = latent.dim(4) * 16
        let y = Self.tilePlan(length: pixelHeight, tileSize: spatialTileSize)
        let x = Self.tilePlan(length: pixelWidth, tileSize: spatialTileSize)
        var tiles: [MLXArray] = []
        for (yIndex, yStart) in y.starts.enumerated() {
            for (xIndex, xStart) in x.starts.enumerated() {
                let tile = latent[
                    0..., 0..., 0...,
                    (yStart / 16)..<((yStart + y.lengths[yIndex]) / 16),
                    (xStart / 16)..<((xStart + x.lengths[xIndex]) / 16)
                ]
                tiles.append(tile.transposed(0, 2, 3, 4, 1))
            }
        }
        let decodedTiles = MLX.split(
            tileDecoder(MLX.concatenated(tiles, axis: 0)),
            parts: tiles.count,
            axis: 0
        )
        var rows: [[MLXArray]] = []
        rows.reserveCapacity(y.starts.count)
        for rowIndex in y.starts.indices {
            let start = rowIndex * x.starts.count
            rows.append(Array(decodedTiles[start..<(start + x.starts.count)]))
        }

        var stitchedRows: [MLXArray] = []
        for rowIndex in rows.indices {
            var pieces: [MLXArray] = []
            for columnIndex in rows[rowIndex].indices {
                var tile = rows[rowIndex][columnIndex]
                if rowIndex > 0 {
                    tile = Self.blend(rows[rowIndex - 1][columnIndex], tile, extent: y.overlaps[rowIndex - 1], axis: 3)
                }
                if columnIndex > 0 {
                    tile = Self.blend(rows[rowIndex][columnIndex - 1], tile, extent: x.overlaps[columnIndex - 1], axis: 4)
                }
                if rowIndex + 1 < rows.count {
                    tile = tile[0..., 0..., 0..., 0..<(tile.dim(3) - y.overlaps[rowIndex]), 0...]
                }
                if columnIndex + 1 < rows[rowIndex].count {
                    tile = tile[0..., 0..., 0..., 0..., 0..<(tile.dim(4) - x.overlaps[columnIndex])]
                }
                pieces.append(tile)
            }
            stitchedRows.append(MLX.concatenated(pieces, axis: 4))
        }
        return MLX.concatenated(stitchedRows, axis: 3)
    }

    static func tilePlan(
        length: Int,
        tileSize: Int
    ) -> (starts: [Int], lengths: [Int], overlaps: [Int]) {
        precondition(tileSize >= minimumSpatialTileOverlap && tileSize.isMultiple(of: 16))
        let minimumOverlap = minimumSpatialTileOverlap
        guard length > tileSize else { return ([0], [length], []) }
        var count = Int(ceil(Double(length) / Double(tileSize)))
        while tileSize * count - minimumOverlap * (count - 1) < length { count += 1 }
        var overlaps = Array(repeating: minimumOverlap, count: count - 1)
        let remaining = tileSize * count - overlaps.reduce(0, +) - length
        for index in 0..<(remaining / 16) { overlaps[index % overlaps.count] += 16 }
        var starts = [0]
        for index in overlaps.indices { starts.append(starts.last! + tileSize - overlaps[index]) }
        return (starts, Array(repeating: tileSize, count: count), overlaps)
    }

    private static func blend(_ previous: MLXArray, _ current: MLXArray, extent: Int, axis: Int) -> MLXArray {
        let length = min(previous.dim(axis), current.dim(axis), extent)
        guard length > 0 else { return current }
        var shape = Array(repeating: 1, count: current.ndim)
        shape[axis] = length
        let position = MLXArray(0..<length).asType(.float32) / Float(length)
        let weightCurrent = position.reshaped(shape).asType(current.dtype)
        let weightPrevious = 1 - weightCurrent
        let previousIndices = MLXArray(((previous.dim(axis) - length)..<previous.dim(axis)).map(Int32.init))
        let currentIndices = MLXArray((0..<length).map(Int32.init))
        let blended = MLX.take(previous, previousIndices, axis: axis) * weightPrevious
            + MLX.take(current, currentIndices, axis: axis) * weightCurrent
        guard length < current.dim(axis) else { return blended }
        let restIndices = MLXArray((length..<current.dim(axis)).map(Int32.init))
        return MLX.concatenated([blended, MLX.take(current, restIndices, axis: axis)], axis: axis)
    }
}
