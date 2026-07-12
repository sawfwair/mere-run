import Foundation

/// Camera-space convention used by all native geometry models.
///
/// X points right in the source image, Y points down, and positive Z points
/// away from the camera. Pixel centers are sampled at `(x + 0.5, y + 0.5)`.
public enum GeometryCoordinateSystem: String, Codable, CaseIterable, Sendable {
    case cameraXRightYDownZForward = "camera-x-right-y-down-z-forward"
    /// World frame recovered from a set of predicted or supplied cameras.
    /// Its orientation is anchored by the first/reference camera and therefore
    /// has no external geodetic meaning.
    case worldFromCameras = "world-from-cameras-x-right-y-down-z-forward"
}

public enum GeometryValueUnits: String, Codable, CaseIterable, Sendable {
    case meters
    case relative
}

public struct GeometryCameraIntrinsics: Codable, Equatable, Sendable {
    public let imageWidth: Int
    public let imageHeight: Int
    public let normalizedFX: Double
    public let normalizedFY: Double
    public let normalizedCX: Double
    public let normalizedCY: Double

    public init(
        imageWidth: Int,
        imageHeight: Int,
        normalizedFX: Double,
        normalizedFY: Double,
        normalizedCX: Double = 0.5,
        normalizedCY: Double = 0.5
    ) {
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.normalizedFX = normalizedFX
        self.normalizedFY = normalizedFY
        self.normalizedCX = normalizedCX
        self.normalizedCY = normalizedCY
    }

    public var pixelFX: Double { normalizedFX * Double(imageWidth) }
    public var pixelFY: Double { normalizedFY * Double(imageHeight) }
    public var pixelCX: Double { normalizedCX * Double(imageWidth) }
    public var pixelCY: Double { normalizedCY * Double(imageHeight) }

    public var normalizedMatrixRowMajor: [Double] {
        [
            normalizedFX, 0, normalizedCX,
            0, normalizedFY, normalizedCY,
            0, 0, 1,
        ]
    }

    public var pixelMatrixRowMajor: [Double] {
        [
            pixelFX, 0, pixelCX,
            0, pixelFY, pixelCY,
            0, 0, 1,
        ]
    }

    public var horizontalFieldOfViewDegrees: Double {
        2 * atan(0.5 / normalizedFX) * 180 / .pi
    }

    public var verticalFieldOfViewDegrees: Double {
        2 * atan(0.5 / normalizedFY) * 180 / .pi
    }
}

public struct GeometryCameraExtrinsics: Codable, Equatable, Sendable {
    /// Row-major world-to-camera rotation matrix.
    public let rotation: [Double]
    /// World-to-camera translation vector.
    public let translation: [Double]

    public init(rotation: [Double], translation: [Double]) throws {
        guard rotation.count == 9 else {
            throw GeometryError.invalidElementCount(field: "rotation", expected: 9, actual: rotation.count)
        }
        guard translation.count == 3 else {
            throw GeometryError.invalidElementCount(field: "translation", expected: 3, actual: translation.count)
        }
        self.rotation = rotation
        self.translation = translation
    }

    public static var identity: GeometryCameraExtrinsics {
        do {
            return try GeometryCameraExtrinsics(
                rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                translation: [0, 0, 0]
            )
        } catch {
            preconditionFailure("Static identity camera dimensions are invalid: \(error)")
        }
    }
}

public enum GeometryError: Error, Equatable, LocalizedError, Sendable {
    case invalidDimensions(width: Int, height: Int)
    case invalidElementCount(field: String, expected: Int, actual: Int)
    case invalidIntrinsics(String)
    case noValidGeometry
    case imageDimensionMismatch(expectedWidth: Int, expectedHeight: Int, actualWidth: Int, actualHeight: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidDimensions(let width, let height):
            "Geometry dimensions must be positive; received \(width)x\(height)."
        case .invalidElementCount(let field, let expected, let actual):
            "Geometry field '\(field)' expected \(expected) values but received \(actual)."
        case .invalidIntrinsics(let detail):
            "Invalid camera intrinsics: \(detail)"
        case .noValidGeometry:
            "The geometry output contains no valid finite points."
        case .imageDimensionMismatch(let expectedWidth, let expectedHeight, let actualWidth, let actualHeight):
            "Source image dimensions must be \(expectedWidth)x\(expectedHeight); received \(actualWidth)x\(actualHeight)."
        }
    }
}

/// Dense geometry in row-major image order.
///
/// `points` and `normals` are interleaved XYZ vectors. `validity` uses zero for
/// invalid pixels and non-zero for valid pixels. All arrays intentionally stay
/// outside the JSON manifest; exporters write them to production interchange
/// files instead of constructing enormous JSON payloads.
public struct DenseGeometryFrame: Sendable {
    public let width: Int
    public let height: Int
    public let units: GeometryValueUnits
    public let coordinateSystem: GeometryCoordinateSystem
    public let intrinsics: GeometryCameraIntrinsics
    public let extrinsics: GeometryCameraExtrinsics?
    public let depth: [Float]
    public let points: [Float]
    public let normals: [Float]?
    public let validity: [UInt8]
    public let confidence: [Float]?

    public init(
        width: Int,
        height: Int,
        units: GeometryValueUnits,
        coordinateSystem: GeometryCoordinateSystem = .cameraXRightYDownZForward,
        intrinsics: GeometryCameraIntrinsics,
        extrinsics: GeometryCameraExtrinsics? = nil,
        depth: [Float],
        points: [Float],
        normals: [Float]? = nil,
        validity: [UInt8],
        confidence: [Float]? = nil
    ) throws {
        guard width > 0, height > 0 else {
            throw GeometryError.invalidDimensions(width: width, height: height)
        }
        let pixels = width * height
        guard depth.count == pixels else {
            throw GeometryError.invalidElementCount(field: "depth", expected: pixels, actual: depth.count)
        }
        guard points.count == pixels * 3 else {
            throw GeometryError.invalidElementCount(field: "points", expected: pixels * 3, actual: points.count)
        }
        if let normals, normals.count != pixels * 3 {
            throw GeometryError.invalidElementCount(field: "normals", expected: pixels * 3, actual: normals.count)
        }
        guard validity.count == pixels else {
            throw GeometryError.invalidElementCount(field: "validity", expected: pixels, actual: validity.count)
        }
        if let confidence, confidence.count != pixels {
            throw GeometryError.invalidElementCount(field: "confidence", expected: pixels, actual: confidence.count)
        }
        guard intrinsics.imageWidth == width, intrinsics.imageHeight == height else {
            throw GeometryError.invalidIntrinsics(
                "matrix describes \(intrinsics.imageWidth)x\(intrinsics.imageHeight), not \(width)x\(height)"
            )
        }
        guard intrinsics.normalizedFX.isFinite, intrinsics.normalizedFX > 0,
              intrinsics.normalizedFY.isFinite, intrinsics.normalizedFY > 0 else {
            throw GeometryError.invalidIntrinsics("focal lengths must be positive and finite")
        }
        self.width = width
        self.height = height
        self.units = units
        self.coordinateSystem = coordinateSystem
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.depth = depth
        self.points = points
        self.normals = normals
        self.validity = validity
        self.confidence = confidence
    }

    public var pixelCount: Int { width * height }

    public func isValid(pixel: Int) -> Bool {
        guard pixel >= 0, pixel < pixelCount, validity[pixel] != 0 else { return false }
        let point = pixel * 3
        return depth[pixel].isFinite && depth[pixel] > 0
            && points[point].isFinite && points[point + 1].isFinite && points[point + 2].isFinite
    }
}

public enum GeometryArtifactKind: String, Codable, CaseIterable, Sendable {
    case depthEXR = "depth-exr"
    case depthPreview = "depth-preview"
    case normalEXR = "normal-exr"
    case normalPreview = "normal-preview"
    case validityMask = "validity-mask"
    case confidenceEXR = "confidence-exr"
    case pointCloud = "point-cloud"
    case camera = "camera"
    case manifest = "manifest"
    case mesh
    case texture
}

public struct GeometryArtifact: Codable, Equatable, Sendable {
    public let kind: GeometryArtifactKind
    public let relativePath: String
    public let mediaType: String
    public let byteCount: Int64
    public let sha256: String

    public init(kind: GeometryArtifactKind, relativePath: String, mediaType: String, byteCount: Int64, sha256: String) {
        self.kind = kind
        self.relativePath = relativePath
        self.mediaType = mediaType
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

public struct GeometryModelProvenance: Codable, Equatable, Sendable {
    public let modelID: String
    public let upstreamRepository: String
    public let upstreamRevision: String
    public let license: String
    public let weightsSHA256: String?
    public let inferenceBackend: String

    public init(
        modelID: String,
        upstreamRepository: String,
        upstreamRevision: String,
        license: String,
        weightsSHA256: String? = nil,
        inferenceBackend: String = "mere.run-native-mlx"
    ) {
        self.modelID = modelID
        self.upstreamRepository = upstreamRepository
        self.upstreamRevision = upstreamRevision
        self.license = license
        self.weightsSHA256 = weightsSHA256
        self.inferenceBackend = inferenceBackend
    }
}

public struct GeometryDepthStatistics: Codable, Equatable, Sendable {
    public let validPixelCount: Int
    public let invalidPixelCount: Int
    public let minimum: Double
    public let maximum: Double
    public let mean: Double

    public init(validPixelCount: Int, invalidPixelCount: Int, minimum: Double, maximum: Double, mean: Double) {
        self.validPixelCount = validPixelCount
        self.invalidPixelCount = invalidPixelCount
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
    }
}

public struct GeometryCameraManifest: Codable, Equatable, Sendable {
    public let intrinsics: GeometryCameraIntrinsics
    public let extrinsics: GeometryCameraExtrinsics?
    public let normalizedMatrixRowMajor: [Double]
    public let pixelMatrixRowMajor: [Double]
    public let horizontalFieldOfViewDegrees: Double
    public let verticalFieldOfViewDegrees: Double

    public init(intrinsics: GeometryCameraIntrinsics, extrinsics: GeometryCameraExtrinsics?) {
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.normalizedMatrixRowMajor = intrinsics.normalizedMatrixRowMajor
        self.pixelMatrixRowMajor = intrinsics.pixelMatrixRowMajor
        self.horizontalFieldOfViewDegrees = intrinsics.horizontalFieldOfViewDegrees
        self.verticalFieldOfViewDegrees = intrinsics.verticalFieldOfViewDegrees
    }
}

public struct GeometryOutputManifest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let createdAt: Date
    public let inputPath: String
    public let inputByteCount: Int64
    public let inputSHA256: String
    public let outputDirectory: String
    public let frameIndex: Int?
    public let frameTimeSeconds: Double?
    public let width: Int
    public let height: Int
    public let units: GeometryValueUnits
    public let coordinateSystem: GeometryCoordinateSystem
    public let model: GeometryModelProvenance
    public let camera: GeometryCameraManifest
    public let depthStatistics: GeometryDepthStatistics
    public let artifacts: [GeometryArtifact]

    public init(
        schemaVersion: Int = 2,
        createdAt: Date = Date(),
        inputPath: String,
        inputByteCount: Int64,
        inputSHA256: String,
        outputDirectory: String,
        frameIndex: Int? = nil,
        frameTimeSeconds: Double? = nil,
        width: Int,
        height: Int,
        units: GeometryValueUnits,
        coordinateSystem: GeometryCoordinateSystem,
        model: GeometryModelProvenance,
        camera: GeometryCameraManifest,
        depthStatistics: GeometryDepthStatistics,
        artifacts: [GeometryArtifact]
    ) {
        self.schemaVersion = schemaVersion
        self.createdAt = createdAt
        self.inputPath = inputPath
        self.inputByteCount = inputByteCount
        self.inputSHA256 = inputSHA256.lowercased()
        self.outputDirectory = outputDirectory
        self.frameIndex = frameIndex
        self.frameTimeSeconds = frameTimeSeconds
        self.width = width
        self.height = height
        self.units = units
        self.coordinateSystem = coordinateSystem
        self.model = model
        self.camera = camera
        self.depthStatistics = depthStatistics
        self.artifacts = artifacts
    }
}
