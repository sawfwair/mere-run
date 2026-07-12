import Foundation

public enum DepthSemantics: String, Codable, CaseIterable, Sendable {
    /// Values are only defined up to one affine scale and shift across the sequence.
    case affineRelative = "affine-relative"
    /// Positive camera-space Z depth measured in meters.
    case metricMeters = "metric-meters"
}

public struct DepthSequenceFrame: Sendable {
    public let index: Int
    public let timeSeconds: Double
    public let width: Int
    public let height: Int
    public let depth: [Float]
    public let confidence: [Float]?
    public let intrinsics: GeometryCameraIntrinsics?

    public init(
        index: Int,
        timeSeconds: Double,
        width: Int,
        height: Int,
        depth: [Float],
        confidence: [Float]? = nil,
        intrinsics: GeometryCameraIntrinsics? = nil
    ) throws {
        guard width > 0, height > 0 else {
            throw GeometryError.invalidDimensions(width: width, height: height)
        }
        let count = width * height
        guard depth.count == count else {
            throw GeometryError.invalidElementCount(field: "sequence depth", expected: count, actual: depth.count)
        }
        if let confidence, confidence.count != count {
            throw GeometryError.invalidElementCount(field: "sequence confidence", expected: count, actual: confidence.count)
        }
        if let intrinsics,
           intrinsics.imageWidth != width || intrinsics.imageHeight != height {
            throw GeometryError.invalidIntrinsics("sequence frame and camera dimensions differ")
        }
        self.index = index
        self.timeSeconds = timeSeconds
        self.width = width
        self.height = height
        self.depth = depth
        self.confidence = confidence
        self.intrinsics = intrinsics
    }
}

public struct DepthSequenceFrameManifest: Codable, Equatable, Sendable {
    public let index: Int
    public let timeSeconds: Double
    public let depthPath: String
    public let previewPath: String?
    public let confidencePath: String?
    public let intrinsics: GeometryCameraIntrinsics?
    public let artifacts: [GeometryArtifact]

    public init(
        index: Int,
        timeSeconds: Double,
        depthPath: String,
        previewPath: String? = nil,
        confidencePath: String? = nil,
        intrinsics: GeometryCameraIntrinsics? = nil,
        artifacts: [GeometryArtifact] = []
    ) {
        self.index = index
        self.timeSeconds = timeSeconds
        self.depthPath = depthPath
        self.previewPath = previewPath
        self.confidencePath = confidencePath
        self.intrinsics = intrinsics
        self.artifacts = artifacts
    }
}

public struct DepthSequenceManifest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let createdAt: Date
    public let inputPath: String
    public let outputDirectory: String
    public let width: Int
    public let height: Int
    public let fps: Double
    public let frameCount: Int
    public let semantics: DepthSemantics
    public let model: GeometryModelProvenance
    public let temporalWindowLength: Int
    public let temporalOverlap: Int
    public let frames: [DepthSequenceFrameManifest]

    public init(
        schemaVersion: Int = 1,
        createdAt: Date = Date(),
        inputPath: String,
        outputDirectory: String,
        width: Int,
        height: Int,
        fps: Double,
        semantics: DepthSemantics,
        model: GeometryModelProvenance,
        temporalWindowLength: Int,
        temporalOverlap: Int,
        frames: [DepthSequenceFrameManifest]
    ) {
        self.schemaVersion = schemaVersion
        self.createdAt = createdAt
        self.inputPath = inputPath
        self.outputDirectory = outputDirectory
        self.width = width
        self.height = height
        self.fps = fps
        self.frameCount = frames.count
        self.semantics = semantics
        self.model = model
        self.temporalWindowLength = temporalWindowLength
        self.temporalOverlap = temporalOverlap
        self.frames = frames
    }

    /// A depth-only model may be projected to points only when real camera
    /// intrinsics were supplied. This prevents silent assumed-camera geometry.
    public var canProjectEveryFrameToPoints: Bool {
        !frames.isEmpty && frames.allSatisfy { $0.intrinsics != nil }
    }
}
