import Foundation

public enum TripoSRConfigurationError: Error, Equatable, LocalizedError, Sendable {
    case invalidParameter(String)

    public var errorDescription: String? {
        switch self {
        case .invalidParameter(let name):
            "TripoSR configuration parameter '\(name)' is invalid."
        }
    }
}

/// Exact inference graph dimensions for the MIT-licensed TripoSR checkpoint.
///
/// The initializer is intentionally configurable so the same implementation can
/// be exercised with tiny deterministic graphs in unit and parity tests. The
/// `.production` value is the only configuration accepted for published
/// `stabilityai/TripoSR` weights.
public struct TripoSRConfiguration: Equatable, Sendable {
    public let conditioningImageSize: Int

    public let imageHiddenSize: Int
    public let imageLayerCount: Int
    public let imageHeadCount: Int
    public let imageIntermediateSize: Int
    public let imagePatchSize: Int
    public let imagePositionGridSize: Int
    public let imageLayerNormEpsilon: Float
    public let imagePositionInterpolationOffset: Float

    public let planeSize: Int
    public let tokenChannels: Int
    public let transformerLayerCount: Int
    public let transformerHeadCount: Int
    public let transformerHeadDimension: Int
    public let transformerGroupCount: Int
    public let transformerLayerNormEpsilon: Float
    public let transformerGroupNormEpsilon: Float
    public let transformerFeedForwardMultiplier: Int

    public let scenePlaneChannels: Int
    public let scenePlaneScale: Int
    public let decoderHiddenSize: Int
    public let decoderHiddenLayerCount: Int
    public let rendererRadius: Float
    public let densityBias: Float
    public let densityThreshold: Float

    public init(
        conditioningImageSize: Int = 512,
        imageHiddenSize: Int = 768,
        imageLayerCount: Int = 12,
        imageHeadCount: Int = 12,
        imageIntermediateSize: Int = 3_072,
        imagePatchSize: Int = 16,
        imagePositionGridSize: Int = 14,
        imageLayerNormEpsilon: Float = 1e-12,
        imagePositionInterpolationOffset: Float = 0.1,
        planeSize: Int = 32,
        tokenChannels: Int = 1_024,
        transformerLayerCount: Int = 16,
        transformerHeadCount: Int = 16,
        transformerHeadDimension: Int = 64,
        transformerGroupCount: Int = 32,
        transformerLayerNormEpsilon: Float = 1e-5,
        transformerGroupNormEpsilon: Float = 1e-6,
        transformerFeedForwardMultiplier: Int = 4,
        scenePlaneChannels: Int = 40,
        scenePlaneScale: Int = 2,
        decoderHiddenSize: Int = 64,
        decoderHiddenLayerCount: Int = 9,
        rendererRadius: Float = 0.87,
        densityBias: Float = -1,
        densityThreshold: Float = 25
    ) throws {
        guard imagePatchSize > 0 else {
            throw TripoSRConfigurationError.invalidParameter("imagePatchSize")
        }
        guard conditioningImageSize > 0,
              conditioningImageSize.isMultiple(of: imagePatchSize) else {
            throw TripoSRConfigurationError.invalidParameter("conditioningImageSize")
        }
        guard imageHeadCount > 0 else {
            throw TripoSRConfigurationError.invalidParameter("imageHeadCount")
        }
        guard imageHiddenSize > 0, imageHiddenSize.isMultiple(of: imageHeadCount) else {
            throw TripoSRConfigurationError.invalidParameter("imageHiddenSize")
        }
        guard imageLayerCount > 0 else {
            throw TripoSRConfigurationError.invalidParameter("imageLayerCount")
        }
        guard imageIntermediateSize > 0 else {
            throw TripoSRConfigurationError.invalidParameter("imageIntermediateSize")
        }
        guard imagePositionGridSize > 0 else {
            throw TripoSRConfigurationError.invalidParameter("imagePositionGridSize")
        }
        guard imageLayerNormEpsilon.isFinite, imageLayerNormEpsilon > 0 else {
            throw TripoSRConfigurationError.invalidParameter("imageLayerNormEpsilon")
        }
        guard imagePositionInterpolationOffset.isFinite else {
            throw TripoSRConfigurationError.invalidParameter("imagePositionInterpolationOffset")
        }
        guard planeSize > 0 else {
            throw TripoSRConfigurationError.invalidParameter("planeSize")
        }
        guard tokenChannels > 0 else {
            throw TripoSRConfigurationError.invalidParameter("tokenChannels")
        }
        guard transformerLayerCount > 0 else {
            throw TripoSRConfigurationError.invalidParameter("transformerLayerCount")
        }
        guard transformerHeadCount > 0 else {
            throw TripoSRConfigurationError.invalidParameter("transformerHeadCount")
        }
        guard transformerHeadDimension > 0 else {
            throw TripoSRConfigurationError.invalidParameter("transformerHeadDimension")
        }
        let headProduct = transformerHeadCount.multipliedReportingOverflow(
            by: transformerHeadDimension
        )
        guard !headProduct.overflow, headProduct.partialValue == tokenChannels else {
            throw TripoSRConfigurationError.invalidParameter("transformerHeadDimension")
        }
        guard transformerGroupCount > 0,
              tokenChannels.isMultiple(of: transformerGroupCount) else {
            throw TripoSRConfigurationError.invalidParameter("transformerGroupCount")
        }
        guard transformerLayerNormEpsilon.isFinite, transformerLayerNormEpsilon > 0 else {
            throw TripoSRConfigurationError.invalidParameter("transformerLayerNormEpsilon")
        }
        guard transformerGroupNormEpsilon.isFinite, transformerGroupNormEpsilon > 0 else {
            throw TripoSRConfigurationError.invalidParameter("transformerGroupNormEpsilon")
        }
        guard transformerFeedForwardMultiplier > 0 else {
            throw TripoSRConfigurationError.invalidParameter("transformerFeedForwardMultiplier")
        }
        guard scenePlaneChannels > 0 else {
            throw TripoSRConfigurationError.invalidParameter("scenePlaneChannels")
        }
        guard scenePlaneScale == 2 else {
            throw TripoSRConfigurationError.invalidParameter("scenePlaneScale")
        }
        guard decoderHiddenSize > 0 else {
            throw TripoSRConfigurationError.invalidParameter("decoderHiddenSize")
        }
        guard decoderHiddenLayerCount > 0 else {
            throw TripoSRConfigurationError.invalidParameter("decoderHiddenLayerCount")
        }
        guard rendererRadius.isFinite, rendererRadius > 0 else {
            throw TripoSRConfigurationError.invalidParameter("rendererRadius")
        }
        guard densityBias.isFinite else {
            throw TripoSRConfigurationError.invalidParameter("densityBias")
        }
        guard densityThreshold.isFinite, densityThreshold > 0 else {
            throw TripoSRConfigurationError.invalidParameter("densityThreshold")
        }
        let imageGrid = conditioningImageSize / imagePatchSize
        let imageGridSquare = imageGrid.multipliedReportingOverflow(by: imageGrid)
        let imageTokens = imageGridSquare.partialValue.addingReportingOverflow(1)
        let planeSquare = planeSize.multipliedReportingOverflow(by: planeSize)
        let triplaneTokens = planeSquare.partialValue.multipliedReportingOverflow(by: 3)
        let scenePlaneSize = planeSize.multipliedReportingOverflow(by: scenePlaneScale)
        let decoderInputSize = scenePlaneChannels.multipliedReportingOverflow(by: 3)
        guard !imageGridSquare.overflow, !imageTokens.overflow else {
            throw TripoSRConfigurationError.invalidParameter("conditioningImageSize")
        }
        guard !planeSquare.overflow, !triplaneTokens.overflow else {
            throw TripoSRConfigurationError.invalidParameter("planeSize")
        }
        guard !scenePlaneSize.overflow else {
            throw TripoSRConfigurationError.invalidParameter("scenePlaneScale")
        }
        guard !decoderInputSize.overflow else {
            throw TripoSRConfigurationError.invalidParameter("scenePlaneChannels")
        }

        self.conditioningImageSize = conditioningImageSize
        self.imageHiddenSize = imageHiddenSize
        self.imageLayerCount = imageLayerCount
        self.imageHeadCount = imageHeadCount
        self.imageIntermediateSize = imageIntermediateSize
        self.imagePatchSize = imagePatchSize
        self.imagePositionGridSize = imagePositionGridSize
        self.imageLayerNormEpsilon = imageLayerNormEpsilon
        self.imagePositionInterpolationOffset = imagePositionInterpolationOffset
        self.planeSize = planeSize
        self.tokenChannels = tokenChannels
        self.transformerLayerCount = transformerLayerCount
        self.transformerHeadCount = transformerHeadCount
        self.transformerHeadDimension = transformerHeadDimension
        self.transformerGroupCount = transformerGroupCount
        self.transformerLayerNormEpsilon = transformerLayerNormEpsilon
        self.transformerGroupNormEpsilon = transformerGroupNormEpsilon
        self.transformerFeedForwardMultiplier = transformerFeedForwardMultiplier
        self.scenePlaneChannels = scenePlaneChannels
        self.scenePlaneScale = scenePlaneScale
        self.decoderHiddenSize = decoderHiddenSize
        self.decoderHiddenLayerCount = decoderHiddenLayerCount
        self.rendererRadius = rendererRadius
        self.densityBias = densityBias
        self.densityThreshold = densityThreshold
    }

    public static let production = TripoSRConfiguration(uncheckedProductionDefaults: ())

    private init(uncheckedProductionDefaults: Void) {
        conditioningImageSize = 512
        imageHiddenSize = 768
        imageLayerCount = 12
        imageHeadCount = 12
        imageIntermediateSize = 3_072
        imagePatchSize = 16
        imagePositionGridSize = 14
        imageLayerNormEpsilon = 1e-12
        imagePositionInterpolationOffset = 0.1
        planeSize = 32
        tokenChannels = 1_024
        transformerLayerCount = 16
        transformerHeadCount = 16
        transformerHeadDimension = 64
        transformerGroupCount = 32
        transformerLayerNormEpsilon = 1e-5
        transformerGroupNormEpsilon = 1e-6
        transformerFeedForwardMultiplier = 4
        scenePlaneChannels = 40
        scenePlaneScale = 2
        decoderHiddenSize = 64
        decoderHiddenLayerCount = 9
        rendererRadius = 0.87
        densityBias = -1
        densityThreshold = 25
    }

    public var imageTokenCount: Int {
        let grid = conditioningImageSize / imagePatchSize
        return 1 + grid * grid
    }

    public var triplaneTokenCount: Int { 3 * planeSize * planeSize }
    public var scenePlaneSize: Int { planeSize * scenePlaneScale }
    public var decoderInputSize: Int { 3 * scenePlaneChannels }
}

/// Runtime-only memory controls. They do not alter the checkpoint graph.
public struct TripoSRMemoryConfiguration: Equatable, Sendable {
    /// Maximum triplane queries processed by one attention kernel.
    public let attentionQueryChunkSize: Int
    /// Maximum triplane tokens processed by one GEGLU kernel.
    public let feedForwardTokenChunkSize: Int
    /// Maximum number of spatial positions evaluated by the NeRF MLP at once.
    public let queryChunkSize: Int
    /// Maximum number of grid points materialized when sampling density.
    public let isosurfaceChunkSize: Int

    public init(
        attentionQueryChunkSize: Int = 1_024,
        feedForwardTokenChunkSize: Int = 1_024,
        queryChunkSize: Int = 8_192,
        isosurfaceChunkSize: Int = 65_536
    ) throws {
        guard attentionQueryChunkSize > 0 else {
            throw TripoSRConfigurationError.invalidParameter("attentionQueryChunkSize")
        }
        guard feedForwardTokenChunkSize > 0 else {
            throw TripoSRConfigurationError.invalidParameter("feedForwardTokenChunkSize")
        }
        guard queryChunkSize > 0 else {
            throw TripoSRConfigurationError.invalidParameter("queryChunkSize")
        }
        guard isosurfaceChunkSize > 0 else {
            throw TripoSRConfigurationError.invalidParameter("isosurfaceChunkSize")
        }
        self.attentionQueryChunkSize = attentionQueryChunkSize
        self.feedForwardTokenChunkSize = feedForwardTokenChunkSize
        self.queryChunkSize = queryChunkSize
        self.isosurfaceChunkSize = isosurfaceChunkSize
    }

    public static let appleSilicon = TripoSRMemoryConfiguration(uncheckedAppleSiliconDefaults: ())

    private init(uncheckedAppleSiliconDefaults: Void) {
        attentionQueryChunkSize = 1_024
        feedForwardTokenChunkSize = 1_024
        queryChunkSize = 8_192
        isosurfaceChunkSize = 65_536
    }
}
