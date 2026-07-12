import Foundation

/// Production resource ceilings for Video Depth Anything. These bounds cover
/// controls known before opening a video as well as decoded-frame and network
/// tensor sizes discovered during extraction.
public enum VideoDepthAnythingLimits {
    public static let minimumInputSize = 14
    public static let maximumInputSize = 1_008
    public static let defaultInputSize = 518

    public static let defaultMaximumFrameCount = 240
    public static let maximumFrameCount = 2_400
    public static let maximumEncodedVideoBytes: Int64 = 512 * 1_024 * 1_024

    public static let maximumDecodedFrameDimension = 8_192
    public static let maximumDecodedPixelCountPerFrame = 16 * 1_024 * 1_024
    public static let maximumAggregateDecodedPixelCount = 512 * 1_024 * 1_024

    /// The largest patch-aligned tensor accepted after aspect-ratio planning.
    /// A 1008-pixel 16:9 request is about 1.8 million pixels.
    public static let maximumNetworkDimension = 2_048
    public static let maximumNetworkPixelCount = 2_100_000

    /// Validates controls without touching the input video, output directory,
    /// model cache, or checkpoint. A missing frame limit receives the bounded
    /// production default.
    @discardableResult
    public static func validateRequest(
        inputSize: Int,
        maximumFrameCount requestedMaximumFrameCount: Int?
    ) throws -> Int {
        guard (minimumInputSize...maximumInputSize).contains(inputSize) else {
            throw VideoDepthAnythingLimitError.inputSizeOutOfRange(
                actual: inputSize,
                minimum: minimumInputSize,
                maximum: maximumInputSize
            )
        }
        let maximumFrameCount = requestedMaximumFrameCount ?? defaultMaximumFrameCount
        guard (1...Self.maximumFrameCount).contains(maximumFrameCount) else {
            throw VideoDepthAnythingLimitError.maximumFrameCountOutOfRange(
                actual: maximumFrameCount,
                minimum: 1,
                maximum: Self.maximumFrameCount
            )
        }
        return maximumFrameCount
    }

    /// Validates the decoded dimensions and the worst-case extraction volume
    /// with overflow-reporting arithmetic.
    public static func validateDecodedSequence(
        width: Int,
        height: Int,
        frameCount: Int
    ) throws {
        guard width > 1, height > 1, frameCount > 0 else {
            throw VideoDepthAnythingLimitError.invalidDecodedSequence(
                width: width,
                height: height,
                frameCount: frameCount
            )
        }
        guard width <= maximumDecodedFrameDimension,
              height <= maximumDecodedFrameDimension else {
            throw VideoDepthAnythingLimitError.decodedFrameDimensionLimitExceeded(
                width: width,
                height: height,
                maximum: maximumDecodedFrameDimension
            )
        }
        let (pixelsPerFrame, pixelOverflow) = width.multipliedReportingOverflow(by: height)
        guard !pixelOverflow, pixelsPerFrame <= maximumDecodedPixelCountPerFrame else {
            throw VideoDepthAnythingLimitError.decodedFramePixelLimitExceeded(
                actual: pixelOverflow ? Int.max : pixelsPerFrame,
                maximum: maximumDecodedPixelCountPerFrame
            )
        }
        let (aggregatePixels, aggregateOverflow) = pixelsPerFrame.multipliedReportingOverflow(
            by: frameCount
        )
        guard !aggregateOverflow,
              aggregatePixels <= maximumAggregateDecodedPixelCount else {
            throw VideoDepthAnythingLimitError.aggregateDecodedPixelLimitExceeded(
                actual: aggregateOverflow ? Int.max : aggregatePixels,
                maximum: maximumAggregateDecodedPixelCount
            )
        }
    }

    public static func validateNetworkDimensions(width: Int, height: Int) throws {
        guard width > 0, height > 0 else {
            throw VideoDepthAnythingLimitError.invalidNetworkDimensions(
                width: width,
                height: height
            )
        }
        guard width <= maximumNetworkDimension, height <= maximumNetworkDimension else {
            throw VideoDepthAnythingLimitError.networkDimensionLimitExceeded(
                width: width,
                height: height,
                maximum: maximumNetworkDimension
            )
        }
        let (pixels, overflow) = width.multipliedReportingOverflow(by: height)
        guard !overflow, pixels <= maximumNetworkPixelCount else {
            throw VideoDepthAnythingLimitError.networkPixelLimitExceeded(
                actual: overflow ? Int.max : pixels,
                maximum: maximumNetworkPixelCount
            )
        }
    }
}

public enum VideoDepthAnythingLimitError: Error, Equatable, LocalizedError, Sendable {
    case inputSizeOutOfRange(actual: Int, minimum: Int, maximum: Int)
    case maximumFrameCountOutOfRange(actual: Int, minimum: Int, maximum: Int)
    case invalidDecodedSequence(width: Int, height: Int, frameCount: Int)
    case decodedFrameDimensionLimitExceeded(width: Int, height: Int, maximum: Int)
    case decodedFramePixelLimitExceeded(actual: Int, maximum: Int)
    case aggregateDecodedPixelLimitExceeded(actual: Int, maximum: Int)
    case invalidNetworkDimensions(width: Int, height: Int)
    case networkDimensionLimitExceeded(width: Int, height: Int, maximum: Int)
    case networkPixelLimitExceeded(actual: Int, maximum: Int)

    public var errorDescription: String? {
        switch self {
        case let .inputSizeOutOfRange(actual, minimum, maximum):
            "VDA input size must be between \(minimum) and \(maximum); received \(actual)."
        case let .maximumFrameCountOutOfRange(actual, minimum, maximum):
            "VDA maximum frame count must be between \(minimum) and \(maximum); received \(actual)."
        case let .invalidDecodedSequence(width, height, frameCount):
            "VDA decoded an invalid \(width)x\(height) sequence with \(frameCount) frames."
        case let .decodedFrameDimensionLimitExceeded(width, height, maximum):
            "VDA decoded frame dimensions \(width)x\(height), exceeding the \(maximum)-pixel side limit."
        case let .decodedFramePixelLimitExceeded(actual, maximum):
            "VDA decoded \(actual) pixels per frame, exceeding the \(maximum)-pixel frame budget."
        case let .aggregateDecodedPixelLimitExceeded(actual, maximum):
            "VDA would extract \(actual) decoded pixels, exceeding the \(maximum)-pixel sequence budget."
        case let .invalidNetworkDimensions(width, height):
            "VDA planned invalid network dimensions \(width)x\(height)."
        case let .networkDimensionLimitExceeded(width, height, maximum):
            "VDA planned network dimensions \(width)x\(height), exceeding the \(maximum)-pixel side limit."
        case let .networkPixelLimitExceeded(actual, maximum):
            "VDA planned \(actual) network pixels, exceeding the \(maximum)-pixel activation budget."
        }
    }
}
