import Foundation

public struct QwenImageEditSize: Sendable, Hashable {
    public let width: Int
    public let height: Int

    public init(width: Int, height: Int) {
        precondition(width > 0 && height > 0)
        self.width = width
        self.height = height
    }

    public static func preservingAspectRatio(
        sourceWidth: Int,
        sourceHeight: Int,
        targetArea: Int
    ) -> QwenImageEditSize {
        precondition(sourceWidth > 0 && sourceHeight > 0 && targetArea > 0)
        let ratio = Double(sourceWidth) / Double(sourceHeight)
        let idealWidth = sqrt(Double(targetArea) * ratio)
        let idealHeight = idealWidth / ratio
        return QwenImageEditSize(
            width: max(32, Int((idealWidth / 32).rounded(.toNearestOrEven)) * 32),
            height: max(32, Int((idealHeight / 32).rounded(.toNearestOrEven)) * 32)
        )
    }
}

public struct QwenImageEditReferencePlan: Sendable, Hashable {
    public let source: URL
    public let sourceSize: QwenImageEditSize
    public let semanticInputSize: QwenImageEditSize
    public let semanticSize: QwenImageEditSize
    public let vaeSize: QwenImageEditSize

    public init(source: URL, sourceWidth: Int, sourceHeight: Int) {
        self.source = source
        self.sourceSize = QwenImageEditSize(width: sourceWidth, height: sourceHeight)
        let semanticInputSize = QwenImageEditSize.preservingAspectRatio(
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            targetArea: 384 * 384
        )
        self.semanticInputSize = semanticInputSize
        self.semanticSize = QwenImageEditSize(
            width: max(28, Int((Double(semanticInputSize.width) / 28).rounded(.toNearestOrEven)) * 28),
            height: max(28, Int((Double(semanticInputSize.height) / 28).rounded(.toNearestOrEven)) * 28)
        )
        self.vaeSize = QwenImageEditSize.preservingAspectRatio(
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            targetArea: 1_024 * 1_024
        )
    }
}

public struct QwenImageEditConditioningPlan: Sendable, Hashable {
    public static let maximumReferenceCount = 3

    public let outputSize: QwenImageEditSize
    public let references: [QwenImageEditReferencePlan]

    public init(
        outputWidth: Int,
        outputHeight: Int,
        references: [QwenImageEditReferencePlan]
    ) {
        precondition(!references.isEmpty && references.count <= Self.maximumReferenceCount)
        self.outputSize = QwenImageEditSize(width: outputWidth, height: outputHeight)
        self.references = references
    }

    public var transformerImageShapes: [(temporal: Int, height: Int, width: Int)] {
        let output = (
            temporal: 1,
            height: outputSize.height / 16,
            width: outputSize.width / 16
        )
        let inputs = references.map { reference in
            (
                temporal: 1,
                height: reference.vaeSize.height / 16,
                width: reference.vaeSize.width / 16
            )
        }
        return [output] + inputs
    }
}
