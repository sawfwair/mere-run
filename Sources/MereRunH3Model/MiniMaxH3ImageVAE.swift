import Foundation
import MLX
import MLXNN

public struct MiniMaxH3ImageVAEConfiguration: Decodable, Sendable {
    public let latentChannels: Int
    public let latentMean: [Float]
    public let latentStandardDeviation: [Float]
    public let uses2DDecoder: Bool
    private let decoder: DecoderConfiguration
    public var isReleasedArchitecture: Bool {
        latentChannels == 24 && uses2DDecoder && decoder.heads == 16 && decoder.headDimension == 64
            && decoder.layers == 24 && decoder.rotaryRatio == 0.5 && decoder.rotaryTheta == 100
    }
    private struct DecoderConfiguration: Decodable, Sendable {
        let heads: Int
        let headDimension: Int
        let layers: Int
        let rotaryRatio: Float
        let rotaryTheta: Float
        enum CodingKeys: String, CodingKey {
            case heads, headDimension = "dim_head", layers = "num_layers"
            case rotaryRatio = "rope_dim_ratio", rotaryTheta = "rope_theta"
        }
    }
    enum CodingKeys: String, CodingKey {
        case latentChannels = "latent_channels", latentMean = "latents_mean", latentStandardDeviation = "latents_std"
        case uses2DDecoder = "vit_decoder_2d", decoder = "vit_decoder_kwargs"
    }
}

public final class MiniMaxH3ImageVAE: Module {
    @ModuleInfo(key: "encoder") var encoder: MiniMaxH3VideoEncoder
    @ModuleInfo(key: "quant_conv") var quant: Conv3d
    @ModuleInfo(key: "post_quant_conv") var postQuant: Conv2d
    @ModuleInfo(key: "decoder") var decoder: MiniMaxH3ImageDecoder
    package var maximumDecodeTileBatchSize = 32
    // Statistics are configuration buffers, not checkpoint parameters discovered by Module reflection.
    private struct Normalization {
        let mean: MLXArray
        let standardDeviation: MLXArray
    }
    private let normalization: Normalization

    public init(configuration: MiniMaxH3ImageVAEConfiguration) {
        _encoder.wrappedValue = MiniMaxH3VideoEncoder()
        _quant.wrappedValue = Conv3d(inputChannels: 48, outputChannels: 48, kernelSize: .init(1))
        _postQuant.wrappedValue = Conv2d(inputChannels: 24, outputChannels: 24, kernelSize: .init(1))
        _decoder.wrappedValue = MiniMaxH3ImageDecoder()
        normalization = Normalization(mean: MLXArray(configuration.latentMean).reshaped(1, 24, 1, 1),
            standardDeviation: MLXArray(configuration.latentStandardDeviation).reshaped(1, 24, 1, 1))
    }

    /// RGB `[1,H,W,3]` -> posterior moments `[1,48,H/16,W/16]` before sampling.
    public func encodeMoments(_ rgb: MLXArray) -> MLXArray {
        let pixelMean = MLXArray([Float(0.485), 0.456, 0.406])
        let pixelStd = MLXArray([Float(0.229), 0.224, 0.225])
        let normalized = (rgb - pixelMean) / pixelStd
        return Self.tiled(normalized, heightAxis: 1, widthAxis: 2, outputRatio: 16) { tile in
            self.quant(self.encoder(tile.expandedDimensions(axis: 1))).squeezed(axis: 1).transposed(0, 3, 1, 2)
        }
    }

    public func sample(_ moments: MLXArray, noise: MLXArray) -> MLXArray {
        let parts = MLX.split(moments, parts: 2, axis: 1)
        let sampled = parts[0] + MLX.exp(0.5 * MLX.clip(parts[1], min: -30, max: 20)) * noise
        let height = sampled.dim(2) / 2 * 2, width = sampled.dim(3) / 2 * 2
        let rounded = sampled.asType(.float16).asType(.float32)[0..., 0..., 0..<height, 0..<width]
        return (rounded - normalization.mean) / normalization.standardDeviation
    }

    /// Normalized `[1,24,H,W]` -> RGB `[1,H*16,W*16,3]` in [0,1].
    public func decode(_ latents: MLXArray) -> MLXArray {
        let denormalized = latents * normalization.standardDeviation + normalization.mean
        let decoded = Self.tiled(denormalized, heightAxis: 2, widthAxis: 3, outputRatio: 1,
                                 latentInput: true, maximumBatchSize: maximumDecodeTileBatchSize) { tile in
            self.decoder(self.postQuant(tile.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2))
        }
        let pixelMean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped(1, 3, 1, 1)
        let pixelStd = MLXArray([Float(0.229), 0.224, 0.225]).reshaped(1, 3, 1, 1)
        return MLX.clip(decoded * pixelStd + pixelMean, min: 0, max: 1).transposed(0, 2, 3, 1)
    }

    package static func tiled(_ input: MLXArray, heightAxis: Int, widthAxis: Int, outputRatio: Int,
                              latentInput: Bool = false, maximumBatchSize: Int = 1,
                              transform: (MLXArray) -> MLXArray) -> MLXArray {
        precondition(maximumBatchSize > 0)
        let inputScale = latentInput ? 16 : 1
        let y = MiniMaxH3VideoVAE.tilePlan(length: input.dim(heightAxis) * inputScale, tileSize: 256)
        let x = MiniMaxH3VideoVAE.tilePlan(length: input.dim(widthAxis) * inputScale, tileSize: 256)
        var flatTiles: [MLXArray] = []
        var pending: [MLXArray] = []
        func flush() {
            guard !pending.isEmpty else { return }
            let result = transform(MLX.concatenated(pending, axis: 0))
            MLX.eval(result)
            let batch = input.dim(0)
            for index in pending.indices {
                flatTiles.append(result[(index * batch)..<((index + 1) * batch)])
            }
            pending.removeAll(keepingCapacity: true)
        }
        for (row, startY) in y.starts.enumerated() {
            for (column, startX) in x.starts.enumerated() {
                let ys = MLXArray((startY / inputScale..<(startY + y.lengths[row]) / inputScale).map(Int32.init))
                let xs = MLXArray((startX / inputScale..<(startX + x.lengths[column]) / inputScale).map(Int32.init))
                pending.append(MLX.take(MLX.take(input, ys, axis: heightAxis), xs, axis: widthAxis))
                if pending.count == maximumBatchSize { flush() }
            }
        }
        flush()
        let tiles = y.starts.indices.map { row in
            Array(flatTiles[(row * x.starts.count)..<((row + 1) * x.starts.count)])
        }
        var rows: [MLXArray] = []
        for row in tiles.indices {
            var pieces: [MLXArray] = []
            for column in tiles[row].indices {
                var tile = tiles[row][column]
                if row > 0 { tile = MiniMaxH3VideoVAE.blend(tiles[row - 1][column], tile, extent: y.overlaps[row - 1] / outputRatio, axis: 2) }
                if column > 0 { tile = MiniMaxH3VideoVAE.blend(tiles[row][column - 1], tile, extent: x.overlaps[column - 1] / outputRatio, axis: 3) }
                let h = tile.dim(2) - (row + 1 < tiles.count ? y.overlaps[row] / outputRatio : 0)
                let w = tile.dim(3) - (column + 1 < tiles[row].count ? x.overlaps[column] / outputRatio : 0)
                pieces.append(tile[0..., 0..., 0..<h, 0..<w])
            }
            rows.append(MLX.concatenated(pieces, axis: 3))
        }
        return MLX.concatenated(rows, axis: 2)
    }
}
