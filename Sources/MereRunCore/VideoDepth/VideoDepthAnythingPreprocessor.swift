import Foundation
import MediaIO
@preconcurrency import MLX
import MLXNN

public enum VideoDepthAnythingPreprocessingError: Error, Equatable, LocalizedError, Sendable {
    case emptyFrames
    case inconsistentFrameDimensions(
        expectedWidth: Int,
        expectedHeight: Int,
        actualWidth: Int,
        actualHeight: Int
    )
    case allocationSizeOverflow
    case invalidDepthShape([Int])
    case unexpectedResizeDimensions(expectedWidth: Int, expectedHeight: Int, actualWidth: Int, actualHeight: Int)

    public var errorDescription: String? {
        switch self {
        case .emptyFrames:
            "Video Depth Anything requires at least one decoded frame."
        case let .inconsistentFrameDimensions(expectedWidth, expectedHeight, actualWidth, actualHeight):
            "Video frame dimensions changed: expected \(expectedWidth)x\(expectedHeight), found \(actualWidth)x\(actualHeight)."
        case .allocationSizeOverflow:
            "Video Depth Anything preprocessing dimensions overflow the host allocation limit."
        case .invalidDepthShape(let shape):
            "Video Depth Anything depth must have [batch, frames, height, width] shape; found \(shape)."
        case let .unexpectedResizeDimensions(expectedWidth, expectedHeight, actualWidth, actualHeight):
            "Video Depth Anything resize produced \(actualWidth)x\(actualHeight); expected \(expectedWidth)x\(expectedHeight)."
        }
    }
}

public enum VideoDepthAnythingPreprocessor {
    /// Reproduces the upstream OpenCV cubic resize contract in MLX, followed
    /// by ImageNet normalization. Output layout is `[1, T, H, W, 3]`.
    public static func normalizedVideo(
        frames: [MediaImage],
        inputSize: Int = VideoDepthAnythingLimits.defaultInputSize
    ) throws -> (video: MLXArray, plan: VideoDepthAnythingPreprocessingPlan) {
        guard let first = frames.first else {
            throw VideoDepthAnythingPreprocessingError.emptyFrames
        }
        for frame in frames where frame.width != first.width || frame.height != first.height {
            throw VideoDepthAnythingPreprocessingError.inconsistentFrameDimensions(
                expectedWidth: first.width,
                expectedHeight: first.height,
                actualWidth: frame.width,
                actualHeight: frame.height
            )
        }
        try VideoDepthAnythingLimits.validateDecodedSequence(
            width: first.width,
            height: first.height,
            frameCount: frames.count
        )
        let plan = try VideoDepthAnythingPreprocessingPlan(
            sourceWidth: first.width,
            sourceHeight: first.height,
            requestedInputSize: inputSize
        )
        let (pixelCount, pixelOverflow) = first.width.multipliedReportingOverflow(by: first.height)
        let (framePixels, frameOverflow) = pixelCount.multipliedReportingOverflow(by: frames.count)
        let (valueCount, valueOverflow) = framePixels.multipliedReportingOverflow(by: 3)
        guard !pixelOverflow, !frameOverflow, !valueOverflow else {
            throw VideoDepthAnythingPreprocessingError.allocationSizeOverflow
        }
        var values = [Float]()
        values.reserveCapacity(valueCount)
        for frame in frames {
            for pixel in 0..<(frame.width * frame.height) {
                let offset = pixel * 4
                values.append(Float(frame.rgba8[offset]) / 255)
                values.append(Float(frame.rgba8[offset + 1]) / 255)
                values.append(Float(frame.rgba8[offset + 2]) / 255)
            }
        }
        var video = MLXArray(values).reshaped(frames.count, first.height, first.width, 3)
        if first.width != plan.networkWidth || first.height != plan.networkHeight {
            video = Upsample(
                scaleFactor: .array([
                    Float(plan.networkHeight) / Float(first.height) + 1e-6,
                    Float(plan.networkWidth) / Float(first.width) + 1e-6,
                ]),
                mode: .cubic(alignCorners: false)
            )(video)
        }
        guard video.dim(1) == plan.networkHeight, video.dim(2) == plan.networkWidth else {
            throw VideoDepthAnythingPreprocessingError.unexpectedResizeDimensions(
                expectedWidth: plan.networkWidth,
                expectedHeight: plan.networkHeight,
                actualWidth: video.dim(2),
                actualHeight: video.dim(1)
            )
        }
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped(1, 1, 1, 3)
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225]).reshaped(1, 1, 1, 3)
        video = ((video - mean) / standardDeviation)
            .reshaped(1, frames.count, plan.networkHeight, plan.networkWidth, 3)
        return (video, plan)
    }

    /// Resizes raw `[B,T,H,W]` depth back to the decoded source dimensions
    /// using the reference model's bilinear align-corners contract.
    public static func resizeDepth(
        _ depth: MLXArray,
        sourceWidth: Int,
        sourceHeight: Int
    ) throws -> MLXArray {
        guard depth.ndim == 4 else {
            throw VideoDepthAnythingPreprocessingError.invalidDepthShape(depth.shape)
        }
        try VideoDepthAnythingLimits.validateDecodedSequence(
            width: sourceWidth,
            height: sourceHeight,
            frameCount: depth.dim(1)
        )
        if depth.dim(2) == sourceHeight && depth.dim(3) == sourceWidth { return depth }
        let batch = depth.dim(0)
        let frames = depth.dim(1)
        let resized = Upsample(
            scaleFactor: .array([
                Float(sourceHeight) / Float(depth.dim(2)) + 1e-6,
                Float(sourceWidth) / Float(depth.dim(3)) + 1e-6,
            ]),
            mode: .linear(alignCorners: true)
        )(depth.reshaped(batch * frames, depth.dim(2), depth.dim(3), 1))
        guard resized.dim(1) == sourceHeight, resized.dim(2) == sourceWidth else {
            throw VideoDepthAnythingPreprocessingError.unexpectedResizeDimensions(
                expectedWidth: sourceWidth,
                expectedHeight: sourceHeight,
                actualWidth: resized.dim(2),
                actualHeight: resized.dim(1)
            )
        }
        return resized.squeezed(axis: -1).reshaped(batch, frames, sourceHeight, sourceWidth)
    }
}
