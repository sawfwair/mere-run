import Foundation
@preconcurrency import MLX

public enum DepthAnything3ConfigurationError: Error, Equatable, LocalizedError, Sendable {
    case invalidParameter(String)

    public var errorDescription: String? {
        switch self {
        case .invalidParameter(let name):
            "Depth Anything 3 configuration parameter '\(name)' is invalid."
        }
    }
}

/// Exact graph parameters for the Apache-2.0 `depth-anything/DA3-SMALL`
/// checkpoint. The defaults mirror the pinned upstream `da3-small` config;
/// the initializer remains configurable so the graph can be exercised in
/// small deterministic unit tests without allocating the production model.
public struct DepthAnything3Configuration: Equatable, Sendable {
    public let hiddenSize: Int
    public let layerCount: Int
    public let headCount: Int
    public let intermediateSize: Int
    public let patchSize: Int
    public let positionGridSize: Int
    public let outputLayers: [Int]
    public let alternateAttentionStart: Int
    public let queryKeyNormStart: Int
    public let rotaryEmbeddingStart: Int
    public let rotaryBaseFrequency: Float
    public let backboneLayerNormEpsilon: Float
    public let headLayerNormEpsilon: Float
    public let featureChannels: Int
    public let projectedChannels: [Int]
    public let cameraEncoderDepth: Int
    public let cameraEncoderHeadCount: Int
    public let cameraEncoderLayerScale: Float
    public let headMicroBatchSize: Int?

    public init(
        hiddenSize: Int = 384,
        layerCount: Int = 12,
        headCount: Int = 6,
        intermediateSize: Int = 1_536,
        patchSize: Int = 14,
        positionGridSize: Int = 37,
        outputLayers: [Int] = [5, 7, 9, 11],
        alternateAttentionStart: Int = 4,
        queryKeyNormStart: Int = 4,
        rotaryEmbeddingStart: Int = 4,
        rotaryBaseFrequency: Float = 100,
        backboneLayerNormEpsilon: Float = 1e-6,
        headLayerNormEpsilon: Float = 1e-5,
        featureChannels: Int = 64,
        projectedChannels: [Int] = [48, 96, 192, 384],
        cameraEncoderDepth: Int = 4,
        cameraEncoderHeadCount: Int = 16,
        cameraEncoderLayerScale: Float = 0.01,
        headMicroBatchSize: Int? = 8
    ) throws {
        guard hiddenSize > 0 else {
            throw DepthAnything3ConfigurationError.invalidParameter("hiddenSize")
        }
        guard layerCount > 0 else {
            throw DepthAnything3ConfigurationError.invalidParameter("layerCount")
        }
        guard headCount > 0, hiddenSize.isMultiple(of: headCount) else {
            throw DepthAnything3ConfigurationError.invalidParameter("headCount")
        }
        guard intermediateSize > 0 else {
            throw DepthAnything3ConfigurationError.invalidParameter("intermediateSize")
        }
        guard patchSize > 0 else {
            throw DepthAnything3ConfigurationError.invalidParameter("patchSize")
        }
        guard positionGridSize > 0 else {
            throw DepthAnything3ConfigurationError.invalidParameter("positionGridSize")
        }
        guard outputLayers.count == 4,
              outputLayers.allSatisfy({ $0 >= 0 && $0 < layerCount }),
              zip(outputLayers, outputLayers.dropFirst()).allSatisfy(<) else {
            throw DepthAnything3ConfigurationError.invalidParameter("outputLayers")
        }
        guard alternateAttentionStart >= 0, alternateAttentionStart < layerCount else {
            throw DepthAnything3ConfigurationError.invalidParameter("alternateAttentionStart")
        }
        guard queryKeyNormStart >= 0, queryKeyNormStart < layerCount else {
            throw DepthAnything3ConfigurationError.invalidParameter("queryKeyNormStart")
        }
        guard rotaryEmbeddingStart >= 0, rotaryEmbeddingStart < layerCount else {
            throw DepthAnything3ConfigurationError.invalidParameter("rotaryEmbeddingStart")
        }
        guard rotaryBaseFrequency.isFinite, rotaryBaseFrequency > 0 else {
            throw DepthAnything3ConfigurationError.invalidParameter("rotaryBaseFrequency")
        }
        guard backboneLayerNormEpsilon.isFinite, backboneLayerNormEpsilon > 0 else {
            throw DepthAnything3ConfigurationError.invalidParameter("backboneLayerNormEpsilon")
        }
        guard headLayerNormEpsilon.isFinite, headLayerNormEpsilon > 0 else {
            throw DepthAnything3ConfigurationError.invalidParameter("headLayerNormEpsilon")
        }
        guard featureChannels > 0, featureChannels.isMultiple(of: 4) else {
            throw DepthAnything3ConfigurationError.invalidParameter("featureChannels")
        }
        guard projectedChannels.count == 4, projectedChannels.allSatisfy({ $0 > 0 }) else {
            throw DepthAnything3ConfigurationError.invalidParameter("projectedChannels")
        }
        guard cameraEncoderDepth > 0 else {
            throw DepthAnything3ConfigurationError.invalidParameter("cameraEncoderDepth")
        }
        guard cameraEncoderHeadCount > 0,
              hiddenSize.isMultiple(of: cameraEncoderHeadCount) else {
            throw DepthAnything3ConfigurationError.invalidParameter("cameraEncoderHeadCount")
        }
        guard cameraEncoderLayerScale.isFinite, cameraEncoderLayerScale > 0 else {
            throw DepthAnything3ConfigurationError.invalidParameter("cameraEncoderLayerScale")
        }
        guard headMicroBatchSize.map({ $0 > 0 }) ?? true else {
            throw DepthAnything3ConfigurationError.invalidParameter("headMicroBatchSize")
        }
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.intermediateSize = intermediateSize
        self.patchSize = patchSize
        self.positionGridSize = positionGridSize
        self.outputLayers = outputLayers
        self.alternateAttentionStart = alternateAttentionStart
        self.queryKeyNormStart = queryKeyNormStart
        self.rotaryEmbeddingStart = rotaryEmbeddingStart
        self.rotaryBaseFrequency = rotaryBaseFrequency
        self.backboneLayerNormEpsilon = backboneLayerNormEpsilon
        self.headLayerNormEpsilon = headLayerNormEpsilon
        self.featureChannels = featureChannels
        self.projectedChannels = projectedChannels
        self.cameraEncoderDepth = cameraEncoderDepth
        self.cameraEncoderHeadCount = cameraEncoderHeadCount
        self.cameraEncoderLayerScale = cameraEncoderLayerScale
        self.headMicroBatchSize = headMicroBatchSize
    }

    public static let small = DepthAnything3Configuration(uncheckedSmallDefaults: ())

    private init(uncheckedSmallDefaults: Void) {
        hiddenSize = 384
        layerCount = 12
        headCount = 6
        intermediateSize = 1_536
        patchSize = 14
        positionGridSize = 37
        outputLayers = [5, 7, 9, 11]
        alternateAttentionStart = 4
        queryKeyNormStart = 4
        rotaryEmbeddingStart = 4
        rotaryBaseFrequency = 100
        backboneLayerNormEpsilon = 1e-6
        headLayerNormEpsilon = 1e-5
        featureChannels = 64
        projectedChannels = [48, 96, 192, 384]
        cameraEncoderDepth = 4
        cameraEncoderHeadCount = 16
        cameraEncoderLayerScale = 0.01
        headMicroBatchSize = 8
    }
}

public enum DepthAnything3ReferenceViewStrategy: String, Codable, CaseIterable, Sendable {
    case first
    case middle
    case saddleBalanced = "saddle-balanced"
    case saddleSimilarityRange = "saddle-similarity-range"
}

/// Optional known-camera conditioning for DA3's camera-token encoder.
///
/// `extrinsics` are world-to-camera matrices in `[batch, views, 4, 4]` and
/// `intrinsics` are pixel-space matrices in `[batch, views, 3, 3]`, matching
/// the authoritative implementation.
public enum DepthAnything3CameraConditioningError: Error, Equatable, LocalizedError, Sendable {
    case invalidExtrinsicsShape([Int])
    case invalidIntrinsicsShape([Int])
    case batchViewShapeMismatch(extrinsics: [Int], intrinsics: [Int])

    public var errorDescription: String? {
        switch self {
        case .invalidExtrinsicsShape(let shape):
            "DA3 camera extrinsics must have shape [batch, views, 4, 4]; received \(shape)."
        case .invalidIntrinsicsShape(let shape):
            "DA3 camera intrinsics must have shape [batch, views, 3, 3]; received \(shape)."
        case .batchViewShapeMismatch(let extrinsics, let intrinsics):
            "DA3 camera extrinsics and intrinsics disagree on batch/views: \(extrinsics) vs \(intrinsics)."
        }
    }
}

public struct DepthAnything3CameraConditioning {
    public let extrinsics: MLXArray
    public let intrinsics: MLXArray

    public init(extrinsics: MLXArray, intrinsics: MLXArray) throws {
        guard extrinsics.ndim == 4,
              extrinsics.dim(0) > 0,
              extrinsics.dim(1) > 0,
              extrinsics.dim(2) == 4,
              extrinsics.dim(3) == 4 else {
            throw DepthAnything3CameraConditioningError.invalidExtrinsicsShape(extrinsics.shape)
        }
        guard intrinsics.ndim == 4,
              intrinsics.dim(0) > 0,
              intrinsics.dim(1) > 0,
              intrinsics.dim(2) == 3,
              intrinsics.dim(3) == 3 else {
            throw DepthAnything3CameraConditioningError.invalidIntrinsicsShape(intrinsics.shape)
        }
        guard extrinsics.dim(0) == intrinsics.dim(0),
              extrinsics.dim(1) == intrinsics.dim(1) else {
            throw DepthAnything3CameraConditioningError.batchViewShapeMismatch(
                extrinsics: Array(extrinsics.shape.prefix(2)),
                intrinsics: Array(intrinsics.shape.prefix(2))
            )
        }
        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
    }
}

/// Native raw DA3-S output. Depth is relative (never meters for this
/// checkpoint), confidence follows upstream's `exp(logit) + 1` convention,
/// extrinsics are world-to-camera, and intrinsics are in input-image pixels.
public struct DepthAnything3RawOutput {
    public let depth: MLXArray
    public let confidence: MLXArray
    public let extrinsics: MLXArray
    public let intrinsics: MLXArray
    /// Six-channel ray representation retained for native 3D handoffs.
    public let ray: MLXArray
    public let rayConfidence: MLXArray

    public init(
        depth: MLXArray,
        confidence: MLXArray,
        extrinsics: MLXArray,
        intrinsics: MLXArray,
        ray: MLXArray,
        rayConfidence: MLXArray
    ) {
        self.depth = depth
        self.confidence = confidence
        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
        self.ray = ray
        self.rayConfidence = rayConfidence
    }
}

struct DepthAnything3BackboneOutput {
    /// Four tensors in configured output-layer order, each `[B,V,N,2C]`.
    let patchFeatures: [MLXArray]
    /// Concatenated local/global camera token from the final output layer.
    let cameraToken: MLXArray
    let patchHeight: Int
    let patchWidth: Int
}
