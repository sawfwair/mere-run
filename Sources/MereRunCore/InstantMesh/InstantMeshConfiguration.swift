import Foundation

public enum InstantMeshConfigurationError: Error, Equatable, LocalizedError, Sendable {
    case invalidParameter(String)

    public var errorDescription: String? {
        switch self {
        case .invalidParameter(let name):
            "InstantMesh configuration parameter '\(name)' is invalid."
        }
    }
}

/// Exact dimensions of TencentARC's Apache-2.0 `instant_mesh_base.ckpt`
/// reconstruction network. View synthesis is intentionally outside this type.
public struct InstantMeshConfiguration: Equatable, Sendable {
    public let conditioningImageSize: Int
    public let cameraDimension: Int

    public let imageHiddenSize: Int
    public let imageLayerCount: Int
    public let imageHeadCount: Int
    public let imageIntermediateSize: Int
    public let imagePatchSize: Int
    public let imagePositionGridSize: Int
    public let imageLayerNormEpsilon: Float
    public let imagePositionInterpolationOffset: Float

    public let triplaneLowResolution: Int
    public let triplaneHighResolution: Int
    public let triplaneChannels: Int
    public let transformerDimension: Int
    public let transformerLayerCount: Int
    public let transformerHeadCount: Int
    public let transformerBlockLayerNormEpsilon: Float
    public let transformerFinalLayerNormEpsilon: Float
    public let transformerMLPMultiplier: Int

    public let decoderHiddenSize: Int
    public let decoderHiddenLayerCount: Int
    public let gridResolution: Int
    public let gridScale: Float
    public let deformationDivisor: Float

    public init(
        conditioningImageSize: Int = 320,
        cameraDimension: Int = 16,
        imageHiddenSize: Int = 768,
        imageLayerCount: Int = 12,
        imageHeadCount: Int = 12,
        imageIntermediateSize: Int = 3_072,
        imagePatchSize: Int = 16,
        imagePositionGridSize: Int = 14,
        imageLayerNormEpsilon: Float = 1e-12,
        imagePositionInterpolationOffset: Float = 0.1,
        triplaneLowResolution: Int = 32,
        triplaneHighResolution: Int = 64,
        triplaneChannels: Int = 40,
        transformerDimension: Int = 1_024,
        transformerLayerCount: Int = 12,
        transformerHeadCount: Int = 16,
        transformerBlockLayerNormEpsilon: Float = 1e-5,
        transformerFinalLayerNormEpsilon: Float = 1e-6,
        transformerMLPMultiplier: Int = 4,
        decoderHiddenSize: Int = 64,
        decoderHiddenLayerCount: Int = 3,
        gridResolution: Int = 128,
        gridScale: Float = 2.1,
        deformationDivisor: Float = 4
    ) throws {
        guard imagePatchSize > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("imagePatchSize")
        }
        guard conditioningImageSize > 0,
              conditioningImageSize.isMultiple(of: imagePatchSize) else {
            throw InstantMeshConfigurationError.invalidParameter("conditioningImageSize")
        }
        guard cameraDimension > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("cameraDimension")
        }
        guard imageHeadCount > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("imageHeadCount")
        }
        guard imageHiddenSize > 0, imageHiddenSize.isMultiple(of: imageHeadCount) else {
            throw InstantMeshConfigurationError.invalidParameter("imageHiddenSize")
        }
        guard imageLayerCount > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("imageLayerCount")
        }
        guard imageIntermediateSize > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("imageIntermediateSize")
        }
        guard imagePositionGridSize > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("imagePositionGridSize")
        }
        guard imageLayerNormEpsilon.isFinite, imageLayerNormEpsilon > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("imageLayerNormEpsilon")
        }
        guard imagePositionInterpolationOffset.isFinite else {
            throw InstantMeshConfigurationError.invalidParameter("imagePositionInterpolationOffset")
        }
        guard triplaneLowResolution > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("triplaneLowResolution")
        }
        let doubledLowResolution = triplaneLowResolution.multipliedReportingOverflow(by: 2)
        guard !doubledLowResolution.overflow,
              triplaneHighResolution == doubledLowResolution.partialValue else {
            throw InstantMeshConfigurationError.invalidParameter("triplaneHighResolution")
        }
        guard triplaneChannels > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("triplaneChannels")
        }
        guard transformerHeadCount > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("transformerHeadCount")
        }
        guard transformerDimension > 0,
              transformerDimension.isMultiple(of: transformerHeadCount) else {
            throw InstantMeshConfigurationError.invalidParameter("transformerDimension")
        }
        guard transformerLayerCount > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("transformerLayerCount")
        }
        guard transformerBlockLayerNormEpsilon.isFinite,
              transformerBlockLayerNormEpsilon > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("transformerBlockLayerNormEpsilon")
        }
        guard transformerFinalLayerNormEpsilon.isFinite,
              transformerFinalLayerNormEpsilon > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("transformerFinalLayerNormEpsilon")
        }
        guard transformerMLPMultiplier > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("transformerMLPMultiplier")
        }
        guard decoderHiddenSize > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("decoderHiddenSize")
        }
        guard decoderHiddenLayerCount >= 2 else {
            throw InstantMeshConfigurationError.invalidParameter("decoderHiddenLayerCount")
        }
        guard gridResolution >= 2 else {
            throw InstantMeshConfigurationError.invalidParameter("gridResolution")
        }
        guard gridScale.isFinite, gridScale > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("gridScale")
        }
        guard deformationDivisor.isFinite, deformationDivisor > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("deformationDivisor")
        }
        let imageGrid = conditioningImageSize / imagePatchSize
        let imageGridSquare = imageGrid.multipliedReportingOverflow(by: imageGrid)
        let imageTokens = imageGridSquare.partialValue.addingReportingOverflow(1)
        let triplaneSquare = triplaneLowResolution.multipliedReportingOverflow(
            by: triplaneLowResolution
        )
        let triplaneTokens = triplaneSquare.partialValue.multipliedReportingOverflow(by: 3)
        let decoderInputSize = triplaneChannels.multipliedReportingOverflow(by: 3)
        guard !imageGridSquare.overflow, !imageTokens.overflow else {
            throw InstantMeshConfigurationError.invalidParameter("conditioningImageSize")
        }
        guard !triplaneSquare.overflow, !triplaneTokens.overflow else {
            throw InstantMeshConfigurationError.invalidParameter("triplaneLowResolution")
        }
        guard !decoderInputSize.overflow else {
            throw InstantMeshConfigurationError.invalidParameter("triplaneChannels")
        }

        self.conditioningImageSize = conditioningImageSize
        self.cameraDimension = cameraDimension
        self.imageHiddenSize = imageHiddenSize
        self.imageLayerCount = imageLayerCount
        self.imageHeadCount = imageHeadCount
        self.imageIntermediateSize = imageIntermediateSize
        self.imagePatchSize = imagePatchSize
        self.imagePositionGridSize = imagePositionGridSize
        self.imageLayerNormEpsilon = imageLayerNormEpsilon
        self.imagePositionInterpolationOffset = imagePositionInterpolationOffset
        self.triplaneLowResolution = triplaneLowResolution
        self.triplaneHighResolution = triplaneHighResolution
        self.triplaneChannels = triplaneChannels
        self.transformerDimension = transformerDimension
        self.transformerLayerCount = transformerLayerCount
        self.transformerHeadCount = transformerHeadCount
        self.transformerBlockLayerNormEpsilon = transformerBlockLayerNormEpsilon
        self.transformerFinalLayerNormEpsilon = transformerFinalLayerNormEpsilon
        self.transformerMLPMultiplier = transformerMLPMultiplier
        self.decoderHiddenSize = decoderHiddenSize
        self.decoderHiddenLayerCount = decoderHiddenLayerCount
        self.gridResolution = gridResolution
        self.gridScale = gridScale
        self.deformationDivisor = deformationDivisor
    }

    public static let production = InstantMeshConfiguration(uncheckedProductionDefaults: ())

    private init(uncheckedProductionDefaults: Void) {
        conditioningImageSize = 320
        cameraDimension = 16
        imageHiddenSize = 768
        imageLayerCount = 12
        imageHeadCount = 12
        imageIntermediateSize = 3_072
        imagePatchSize = 16
        imagePositionGridSize = 14
        imageLayerNormEpsilon = 1e-12
        imagePositionInterpolationOffset = 0.1
        triplaneLowResolution = 32
        triplaneHighResolution = 64
        triplaneChannels = 40
        transformerDimension = 1_024
        transformerLayerCount = 12
        transformerHeadCount = 16
        transformerBlockLayerNormEpsilon = 1e-5
        transformerFinalLayerNormEpsilon = 1e-6
        transformerMLPMultiplier = 4
        decoderHiddenSize = 64
        decoderHiddenLayerCount = 3
        gridResolution = 128
        gridScale = 2.1
        deformationDivisor = 4
    }

    public var imageTokenCount: Int {
        let grid = conditioningImageSize / imagePatchSize
        return 1 + grid * grid
    }

    public var triplaneTokenCount: Int {
        3 * triplaneLowResolution * triplaneLowResolution
    }

    public var decoderInputSize: Int { 3 * triplaneChannels }
}

/// Runtime chunk sizes only. These controls do not change checkpoint math.
public struct InstantMeshMemoryConfiguration: Equatable, Sendable {
    public let imageViewBatchSize: Int
    public let attentionQueryChunkSize: Int
    public let feedForwardTokenChunkSize: Int
    public let fieldQueryChunkSize: Int
    public let isosurfaceQueryChunkSize: Int

    public init(
        imageViewBatchSize: Int = 1,
        attentionQueryChunkSize: Int = 256,
        feedForwardTokenChunkSize: Int = 512,
        fieldQueryChunkSize: Int = 65_536,
        isosurfaceQueryChunkSize: Int = 65_536
    ) throws {
        guard imageViewBatchSize > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("imageViewBatchSize")
        }
        guard attentionQueryChunkSize > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("attentionQueryChunkSize")
        }
        guard feedForwardTokenChunkSize > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("feedForwardTokenChunkSize")
        }
        guard fieldQueryChunkSize > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("fieldQueryChunkSize")
        }
        guard isosurfaceQueryChunkSize > 0 else {
            throw InstantMeshConfigurationError.invalidParameter("isosurfaceQueryChunkSize")
        }
        self.imageViewBatchSize = imageViewBatchSize
        self.attentionQueryChunkSize = attentionQueryChunkSize
        self.feedForwardTokenChunkSize = feedForwardTokenChunkSize
        self.fieldQueryChunkSize = fieldQueryChunkSize
        self.isosurfaceQueryChunkSize = isosurfaceQueryChunkSize
    }

    public static let appleSilicon = InstantMeshMemoryConfiguration(uncheckedAppleSiliconDefaults: ())

    private init(uncheckedAppleSiliconDefaults: Void) {
        imageViewBatchSize = 1
        attentionQueryChunkSize = 256
        feedForwardTokenChunkSize = 512
        fieldQueryChunkSize = 65_536
        isosurfaceQueryChunkSize = 65_536
    }
}
