import Foundation
import MLX

public enum MiniMaxH3Geometry {
    public static let framesPerSecond = 24
    public static let audioLatentsPerSecond = 40
    public static let videoFramesPerChunk = 17
    public static let videoLatentsPerChunk = 5
    public static let frameSpanPattern: [Double] = [1, 4, 4, 4, 4]
    public static let frameSpanScale = 5.0 / 3.0

    public static func alignFrameCount(_ frameCount: Int) throws -> Int {
        guard frameCount > 0 else {
            throw MiniMaxH3LayoutError.invalidGeometry("frame count must be positive")
        }
        var aligned = frameCount
        while aligned % videoFramesPerChunk != videoLatentsPerChunk {
            aligned += 1
        }
        return aligned
    }

    public static func videoLatentFrameCount(for frameCount: Int) throws -> Int {
        guard frameCount % videoFramesPerChunk == videoLatentsPerChunk else {
            throw MiniMaxH3LayoutError.invalidGeometry("frame count must have form 17*n+5")
        }
        return ((frameCount - videoLatentsPerChunk) / videoFramesPerChunk) * videoLatentsPerChunk + 2
    }

    public static func audioLatentFrameCount(for frameCount: Int) -> Int {
        Int((Double(frameCount) / Double(framesPerSecond) * Double(audioLatentsPerSecond)).rounded())
    }

    public static func shiftedSigma(_ sigma: Float, from sourceShift: Float, to targetShift: Float) -> Float {
        let base = sigma / (sourceShift + sigma * (1 - sourceShift))
        return targetShift * base / (1 + (targetShift - 1) * base)
    }

    public static func shiftedSigmaSlope(_ sigma: Float, from sourceShift: Float, to targetShift: Float) -> Float {
        let base = sigma / (sourceShift + sigma * (1 - sourceShift))
        let numerator = targetShift * pow(1 + (sourceShift - 1) * base, 2)
        let denominator = sourceShift * pow(1 + (targetShift - 1) * base, 2)
        return numerator / denominator
    }

    public static func patchifyVideo(_ latent: MLXArray, patchSize: [Int] = [1, 2, 2]) -> MLXArray {
        precondition(latent.ndim == 5)
        let batch = latent.dim(0)
        let channels = latent.dim(1)
        let frames = latent.dim(2)
        let height = latent.dim(3)
        let width = latent.dim(4)
        let temporalPatch = patchSize[0]
        let heightPatch = patchSize[1]
        let widthPatch = patchSize[2]
        return latent
            .reshaped(
                batch, channels, frames / temporalPatch, temporalPatch,
                height / heightPatch, heightPatch, width / widthPatch, widthPatch
            )
            .transposed(0, 2, 4, 6, 1, 3, 5, 7)
            .reshaped(
                batch,
                (frames / temporalPatch) * (height / heightPatch) * (width / widthPatch),
                channels * temporalPatch * heightPatch * widthPatch
            )
    }

    public static func unpatchifyVideo(
        _ rows: MLXArray,
        frames: Int,
        height: Int,
        width: Int,
        channels: Int = 24,
        patchSize: [Int] = [1, 2, 2]
    ) -> MLXArray {
        let temporalPatch = patchSize[0]
        let heightPatch = patchSize[1]
        let widthPatch = patchSize[2]
        return rows
            .reshaped(
                rows.dim(0), frames / temporalPatch, height / heightPatch, width / widthPatch,
                channels, temporalPatch, heightPatch, widthPatch
            )
            .transposed(0, 4, 1, 5, 2, 6, 3, 7)
            .reshaped(rows.dim(0), channels, frames, height, width)
    }

    public static func packAudio(_ latent: MLXArray) -> MLXArray {
        precondition(latent.ndim == 4 && latent.dim(0) == 1 && latent.dim(2) == 2)
        return latent[0].transposed(1, 2, 0).reshaped(2 * latent.dim(3), latent.dim(1))
    }

    public static func unpackAudio(_ rows: MLXArray) -> MLXArray {
        let frames = rows.dim(0) / 2
        return rows.reshaped(2, frames, rows.dim(1)).transposed(2, 0, 1).expandedDimensions(axis: 0)
    }

}
