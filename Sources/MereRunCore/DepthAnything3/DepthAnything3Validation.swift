import Foundation
import MediaIO

/// Resource ceilings for the native DA3-Small graph. These values bound both
/// decoded CPU image memory and the view-dependent MLX activation volume.
public enum DepthAnything3Limits {
    public static let minimumViewCount = 1
    public static let maximumViewCount = 16
    public static let minimumProcessResolution = 14
    public static let maximumProcessResolution = 1_008
    public static let maximumProcessedPixelCount = 8 * 504 * 504
    public static let maximumSourcePixelCountPerView = 8_192 * 8_192
    public static let maximumTotalSourcePixelCount = 2 * 8_192 * 8_192
    public static let maximumEncodedBytesPerView: Int64 = 512 * 1_024 * 1_024
    public static let maximumTotalEncodedBytes: Int64 = 1_024 * 1_024 * 1_024

    /// Admission limits used while copying caller-controlled view bytes into
    /// private immutable snapshots. Header inspection, decode, and durable
    /// provenance all operate on those same snapshots.
    public static let imageInputLimits = VFXImageInputLimits(
        maximumDimension: 8_192,
        maximumPixelsPerImage: maximumSourcePixelCountPerView,
        maximumAggregatePixels: maximumTotalSourcePixelCount,
        maximumEncodedBytesPerImage: maximumEncodedBytesPerView,
        maximumAggregateEncodedBytes: maximumTotalEncodedBytes
    )

    /// Validates limits knowable without opening an image. The processed-pixel
    /// estimate deliberately uses the next patch-aligned square: every actual
    /// DA3 batch is no larger than this conservative bound.
    public static func validateRequest(
        viewCount: Int,
        processResolution: Int
    ) throws {
        guard (minimumViewCount...maximumViewCount).contains(viewCount) else {
            throw DepthAnything3LimitError.viewCountOutOfRange(
                actual: viewCount,
                minimum: minimumViewCount,
                maximum: maximumViewCount
            )
        }
        guard (minimumProcessResolution...maximumProcessResolution).contains(processResolution) else {
            throw DepthAnything3LimitError.processResolutionOutOfRange(
                actual: processResolution,
                minimum: minimumProcessResolution,
                maximum: maximumProcessResolution
            )
        }
        let alignedResolution = ((processResolution + 13) / 14) * 14
        let processedPixels = viewCount * alignedResolution * alignedResolution
        guard processedPixels <= maximumProcessedPixelCount else {
            throw DepthAnything3LimitError.processedPixelBudgetExceeded(
                actual: processedPixels,
                maximum: maximumProcessedPixelCount
            )
        }
    }

    /// Reads only image headers and rejects unsafe decoded dimensions before
    /// allocating RGBA buffers.
    @discardableResult
    public static func validateImageURLs(
        _ urls: [URL]
    ) throws -> [(width: Int, height: Int)] {
        try validateEncodedByteCounts(
            urls.map { try ModelArtifactPin.fileByteCount($0.standardizedFileURL) }
        )
        let dimensions = try urls.map(MediaImageIO.size)
        try validateSourceDimensions(dimensions)
        return dimensions
    }

    /// Validates encoded inputs before header parsing. Execution enforces the
    /// same ceilings while streaming bytes into immutable snapshots; this
    /// count-only form keeps CLI dry-run preflight truthful without retaining
    /// a duplicate copy.
    public static func validateEncodedByteCounts(_ byteCounts: [Int64]) throws {
        var total: Int64 = 0
        for (index, byteCount) in byteCounts.enumerated() {
            guard byteCount >= 0, byteCount <= maximumEncodedBytesPerView else {
                throw DepthAnything3LimitError.encodedByteBudgetExceeded(
                    index: index,
                    actual: max(0, byteCount),
                    maximum: maximumEncodedBytesPerView
                )
            }
            let sum = total.addingReportingOverflow(byteCount)
            guard !sum.overflow, sum.partialValue <= maximumTotalEncodedBytes else {
                throw DepthAnything3LimitError.totalEncodedByteBudgetExceeded(
                    actual: sum.overflow ? Int64.max : sum.partialValue,
                    maximum: maximumTotalEncodedBytes
                )
            }
            total = sum.partialValue
        }
    }

    /// Validates dimensions from image headers or already-decoded images using
    /// overflow-reporting arithmetic so adversarial metadata cannot trap.
    public static func validateSourceDimensions(
        _ dimensions: [(width: Int, height: Int)]
    ) throws {
        var total = 0
        for (index, size) in dimensions.enumerated() {
            guard size.width > 0, size.height > 0 else {
                throw DepthAnything3LimitError.invalidSourceDimensions(
                    index: index,
                    width: size.width,
                    height: size.height
                )
            }
            let (pixels, overflowed) = size.width.multipliedReportingOverflow(by: size.height)
            guard !overflowed, pixels <= maximumSourcePixelCountPerView else {
                throw DepthAnything3LimitError.sourcePixelBudgetExceeded(
                    index: index,
                    width: size.width,
                    height: size.height,
                    actual: overflowed ? Int.max : pixels,
                    maximum: maximumSourcePixelCountPerView
                )
            }
            let (newTotal, totalOverflowed) = total.addingReportingOverflow(pixels)
            guard !totalOverflowed, newTotal <= maximumTotalSourcePixelCount else {
                throw DepthAnything3LimitError.totalSourcePixelBudgetExceeded(
                    actual: totalOverflowed ? Int.max : newTotal,
                    maximum: maximumTotalSourcePixelCount
                )
            }
            total = newTotal
        }
    }
}

public enum DepthAnything3LimitError: Error, Equatable, LocalizedError, Sendable {
    case viewCountOutOfRange(actual: Int, minimum: Int, maximum: Int)
    case processResolutionOutOfRange(actual: Int, minimum: Int, maximum: Int)
    case processedPixelBudgetExceeded(actual: Int, maximum: Int)
    case encodedByteBudgetExceeded(index: Int, actual: Int64, maximum: Int64)
    case totalEncodedByteBudgetExceeded(actual: Int64, maximum: Int64)
    case invalidSourceDimensions(index: Int, width: Int, height: Int)
    case sourcePixelBudgetExceeded(
        index: Int,
        width: Int,
        height: Int,
        actual: Int,
        maximum: Int
    )
    case totalSourcePixelBudgetExceeded(actual: Int, maximum: Int)

    public var errorDescription: String? {
        switch self {
        case let .viewCountOutOfRange(actual, minimum, maximum):
            "DA3 view count must be between \(minimum) and \(maximum); received \(actual)."
        case let .processResolutionOutOfRange(actual, minimum, maximum):
            "DA3 process resolution must be between \(minimum) and \(maximum); received \(actual)."
        case let .processedPixelBudgetExceeded(actual, maximum):
            "DA3 request needs up to \(actual) processed pixels, exceeding the \(maximum)-pixel activation budget."
        case let .encodedByteBudgetExceeded(index, actual, maximum):
            "DA3 source image \(index) is \(actual) encoded bytes, exceeding the \(maximum)-byte per-view limit."
        case let .totalEncodedByteBudgetExceeded(actual, maximum):
            "DA3 source images total \(actual) encoded bytes, exceeding the \(maximum)-byte aggregate limit."
        case let .invalidSourceDimensions(index, width, height):
            "DA3 source image \(index) has invalid dimensions \(width)x\(height)."
        case let .sourcePixelBudgetExceeded(index, width, height, actual, maximum):
            "DA3 source image \(index) is \(width)x\(height) (\(actual) pixels), exceeding the \(maximum)-pixel per-view decode budget."
        case let .totalSourcePixelBudgetExceeded(actual, maximum):
            "DA3 source images contain \(actual) pixels in total, exceeding the \(maximum)-pixel decode budget."
        }
    }
}

public struct DepthAnything3CameraValidationError: Error, Equatable, LocalizedError, Sendable {
    public let index: Int
    public let reason: String

    public init(index: Int, reason: String) {
        self.index = index
        self.reason = reason
    }

    public var errorDescription: String? {
        "DA3 camera \(index) is invalid: \(reason)"
    }
}

/// Strict validation for user-supplied DA3 camera conditioning. The native
/// graph consumes Float tensors and treats rotation transpose as inverse, so
/// merely checking JSON shape and Double finiteness is insufficient.
public enum DepthAnything3CameraValidation {
    public static let rotationTolerance = 1e-3

    public static func validate(
        _ camera: DepthAnything3KnownCamera,
        index: Int
    ) throws {
        if let issue = issue(for: camera) {
            throw DepthAnything3CameraValidationError(index: index, reason: issue)
        }
    }

    public static func issue(for camera: DepthAnything3KnownCamera) -> String? {
        let intrinsics = camera.intrinsics
        guard intrinsics.imageWidth > 0, intrinsics.imageHeight > 0 else {
            return "image dimensions must be positive"
        }
        let normalized: [(String, Double)] = [
            ("normalizedFX", intrinsics.normalizedFX),
            ("normalizedFY", intrinsics.normalizedFY),
            ("normalizedCX", intrinsics.normalizedCX),
            ("normalizedCY", intrinsics.normalizedCY),
        ]
        for (name, value) in normalized where !isFloatRepresentable(value) {
            return "\(name) must be finite and Float-representable"
        }
        guard intrinsics.normalizedFX > 0, intrinsics.normalizedFY > 0 else {
            return "focal lengths must be positive"
        }
        let pixels: [(String, Double)] = [
            ("pixelFX", intrinsics.pixelFX),
            ("pixelFY", intrinsics.pixelFY),
            ("pixelCX", intrinsics.pixelCX),
            ("pixelCY", intrinsics.pixelCY),
        ]
        for (name, value) in pixels where !isFloatRepresentable(value) {
            return "derived \(name) must be finite and Float-representable"
        }

        let rotation = camera.extrinsics.rotation
        let translation = camera.extrinsics.translation
        guard rotation.count == 9, translation.count == 3 else {
            return "extrinsics must contain a 3x3 rotation and 3-vector translation"
        }
        for (index, value) in rotation.enumerated() where !isFloatRepresentable(value) {
            return "rotation[\(index)] must be finite and Float-representable"
        }
        for (index, value) in translation.enumerated() where !isFloatRepresentable(value) {
            return "translation[\(index)] must be finite and Float-representable"
        }

        let rows = [
            Array(rotation[0..<3]),
            Array(rotation[3..<6]),
            Array(rotation[6..<9]),
        ]
        let columns = [
            [rotation[0], rotation[3], rotation[6]],
            [rotation[1], rotation[4], rotation[7]],
            [rotation[2], rotation[5], rotation[8]],
        ]
        for (label, axes) in [("row", rows), ("column", columns)] {
            for axis in 0..<3 where abs(dot(axes[axis], axes[axis]) - 1) > rotationTolerance {
                return "rotation \(label) \(axis) is not unit length"
            }
            for first in 0..<3 {
                for second in (first + 1)..<3
                where abs(dot(axes[first], axes[second])) > rotationTolerance {
                    return "rotation \(label)s \(first) and \(second) are not orthogonal"
                }
            }
        }
        let determinant = rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7])
            - rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6])
            + rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6])
        guard determinant.isFinite, abs(determinant - 1) <= rotationTolerance else {
            return "rotation must be proper with determinant +1"
        }
        return nil
    }

    private static func isFloatRepresentable(_ value: Double) -> Bool {
        value.isFinite && abs(value) <= Double(Float.greatestFiniteMagnitude)
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        zip(lhs, rhs).reduce(0) { $0 + $1.0 * $1.1 }
    }
}
