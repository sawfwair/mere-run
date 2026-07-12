import Foundation

public struct MoGe2TokenGrid: Equatable, Sendable {
    public static let minimumTokenCount = 1
    public static let maximumTokenCount = 3_600

    public let rows: Int
    public let columns: Int

    public var count: Int { rows * columns }

    private init(rows: Int, columns: Int) {
        self.rows = rows
        self.columns = columns
    }

    /// Converts MoGe's target token count into the aspect-ratio-preserving
    /// DINO patch grid while enforcing the production workload ceiling.
    public static func resolve(
        imageWidth: Int,
        imageHeight: Int,
        requestedTokenCount: Int
    ) throws -> MoGe2TokenGrid {
        guard imageWidth > 0, imageHeight > 0 else {
            throw MoGe2TokenGridError.invalidImageDimensions(
                width: imageWidth,
                height: imageHeight
            )
        }
        guard requestedTokenCount >= minimumTokenCount,
              requestedTokenCount <= maximumTokenCount else {
            throw MoGe2TokenGridError.tokenCountOutOfRange(
                actual: requestedTokenCount,
                minimum: minimumTokenCount,
                maximum: maximumTokenCount
            )
        }

        let aspectRatio = Double(imageWidth) / Double(imageHeight)
        let rows = max(1, Int((sqrt(Double(requestedTokenCount) / aspectRatio)).rounded()))
        let columns = max(1, Int((sqrt(Double(requestedTokenCount) * aspectRatio)).rounded()))
        let product = rows.multipliedReportingOverflow(by: columns)
        guard !product.overflow, product.partialValue <= maximumTokenCount else {
            throw MoGe2TokenGridError.tokenGridExceedsLimit(
                rows: rows,
                columns: columns,
                actual: product.overflow ? Int.max : product.partialValue,
                maximum: maximumTokenCount
            )
        }
        return MoGe2TokenGrid(rows: rows, columns: columns)
    }
}

public enum MoGe2TokenGridError: Error, Equatable, LocalizedError, Sendable {
    case invalidImageDimensions(width: Int, height: Int)
    case tokenCountOutOfRange(actual: Int, minimum: Int, maximum: Int)
    case tokenGridExceedsLimit(rows: Int, columns: Int, actual: Int, maximum: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidImageDimensions(let width, let height):
            "MoGe-2 image dimensions must be positive; received \(width)x\(height)."
        case let .tokenCountOutOfRange(actual, minimum, maximum):
            "MoGe-2 token count must be between \(minimum) and \(maximum); received \(actual)."
        case let .tokenGridExceedsLimit(rows, columns, actual, maximum):
            "MoGe-2 derived a \(rows)x\(columns) patch grid (\(actual) tokens); the native workload limit is \(maximum)."
        }
    }
}
