import Foundation
import MediaIO

public struct MultiViewGeometryView: Sendable {
    public let index: Int
    public let sourceURL: URL
    public let inputIdentity: DepthAnything3InputIdentity
    public let image: MediaImage
    public let preprocessingPlan: DepthAnything3PreprocessingPlan
    public let depth: [Float]
    public let confidence: [Float]
    public let intrinsics: GeometryCameraIntrinsics
    public let extrinsics: GeometryCameraExtrinsics

    public init(
        index: Int,
        sourceURL: URL,
        inputIdentity: DepthAnything3InputIdentity,
        image: MediaImage,
        preprocessingPlan: DepthAnything3PreprocessingPlan,
        depth: [Float],
        confidence: [Float],
        intrinsics: GeometryCameraIntrinsics,
        extrinsics: GeometryCameraExtrinsics
    ) throws {
        let pixels = image.width * image.height
        guard depth.count == pixels else {
            throw GeometryError.invalidElementCount(
                field: "multi-view depth",
                expected: pixels,
                actual: depth.count
            )
        }
        guard confidence.count == pixels else {
            throw GeometryError.invalidElementCount(
                field: "multi-view confidence",
                expected: pixels,
                actual: confidence.count
            )
        }
        guard intrinsics.imageWidth == image.width, intrinsics.imageHeight == image.height else {
            throw GeometryError.invalidIntrinsics("multi-view camera and processed image dimensions differ")
        }
        guard intrinsics.normalizedFX.isFinite, intrinsics.normalizedFX > 0,
              intrinsics.normalizedFY.isFinite, intrinsics.normalizedFY > 0 else {
            throw GeometryError.invalidIntrinsics("multi-view focal lengths must be positive and finite")
        }
        guard intrinsics.normalizedCX.isFinite, intrinsics.normalizedCY.isFinite,
              intrinsics.pixelFX.isFinite, intrinsics.pixelFY.isFinite,
              intrinsics.pixelCX.isFinite, intrinsics.pixelCY.isFinite else {
            throw GeometryError.invalidIntrinsics("multi-view camera parameters must be finite")
        }
        guard extrinsics.rotation.allSatisfy(\.isFinite),
              extrinsics.translation.allSatisfy(\.isFinite) else {
            throw GeometryError.invalidIntrinsics("multi-view camera extrinsics must be finite")
        }
        self.index = index
        self.sourceURL = sourceURL.standardizedFileURL
        self.inputIdentity = inputIdentity
        self.image = image
        self.preprocessingPlan = preprocessingPlan
        self.depth = depth
        self.confidence = confidence
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
    }

    public init(result: DepthAnything3ViewResult) throws {
        try self.init(
            index: result.index,
            sourceURL: result.sourceURL,
            inputIdentity: result.inputIdentity,
            image: result.processedImage,
            preprocessingPlan: result.preprocessingPlan,
            depth: result.depth,
            confidence: result.confidence,
            intrinsics: result.intrinsics,
            extrinsics: result.extrinsics
        )
    }
}

public enum MultiViewGeometryExportConfigurationError: Error, Equatable, LocalizedError, Sendable {
    case invalidConfidencePercentile(Double)
    case invalidMaximumPointCount(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidConfidencePercentile(let value):
            "Multi-view confidence percentile must be finite and between 0 and 100; received \(value)."
        case .invalidMaximumPointCount(let value):
            "Multi-view maximum point count must be positive; received \(value)."
        }
    }
}

public struct MultiViewGeometryExportConfiguration: Equatable, Sendable {
    public let confidencePercentile: Double
    public let maximumPointCount: Int

    public init(
        confidencePercentile: Double = 40,
        maximumPointCount: Int = 1_000_000
    ) throws {
        guard confidencePercentile.isFinite, (0...100).contains(confidencePercentile) else {
            throw MultiViewGeometryExportConfigurationError.invalidConfidencePercentile(
                confidencePercentile
            )
        }
        guard maximumPointCount > 0 else {
            throw MultiViewGeometryExportConfigurationError.invalidMaximumPointCount(
                maximumPointCount
            )
        }
        self.confidencePercentile = confidencePercentile
        self.maximumPointCount = maximumPointCount
    }
}

public enum MultiViewGeometryArtifactKind: String, Codable, CaseIterable, Sendable {
    case depthEXR = "depth-exr"
    case depthPreview = "depth-preview"
    case confidenceEXR = "confidence-exr"
    case processedImage = "processed-image"
    case pointCloudPLY = "point-cloud-ply"
    case pointCloudGLB = "point-cloud-glb"
    case camerasJSON = "cameras-json"
    case transformsJSON = "3dgs-transforms-json"
}

public struct MultiViewGeometryArtifact: Codable, Equatable, Sendable {
    public let kind: MultiViewGeometryArtifactKind
    public let viewIndex: Int?
    public let relativePath: String
    public let mediaType: String
    public let byteCount: Int64
    public let sha256: String
}

public struct MultiViewGeometryViewManifest: Codable, Equatable, Sendable {
    public let index: Int
    public let sourcePath: String
    public let sourceByteCount: Int64
    public let sourceSHA256: String
    public let width: Int
    public let height: Int
    public let preprocessing: DepthAnything3PreprocessingPlan
    public let camera: GeometryCameraManifest
    public let depthPath: String
    public let confidencePath: String
    public let previewPath: String
    public let processedImagePath: String
    public let selectedPointCount: Int
}

public struct MultiViewGeometryCheckpointManifest: Codable, Equatable, Sendable {
    public let modelID: String
    public let repository: String
    public let revision: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let license: String
    public let weightsByteCount: Int64
    public let weightsSHA256: String
    public let configurationByteCount: Int64
    public let configurationSHA256: String
    public let inferenceBackend: String

    public init(checkpoint: DepthAnything3Checkpoint) {
        self.modelID = checkpoint.modelID
        self.repository = checkpoint.repository
        self.revision = checkpoint.revision
        self.sourceRepository = checkpoint.sourceRepository
        self.sourceRevision = checkpoint.sourceRevision
        self.license = checkpoint.license
        self.weightsByteCount = checkpoint.weightsByteCount
        self.weightsSHA256 = checkpoint.weightsSHA256
        self.configurationByteCount = checkpoint.configurationByteCount
        self.configurationSHA256 = checkpoint.configurationSHA256
        self.inferenceBackend = "mere.run-native-mlx"
    }
}

public struct Geometry3DGSHandoffManifest: Codable, Equatable, Sendable {
    public let kind: String
    public let transformsPath: String
    public let pointCloudPath: String
    public let containsGaussianParameters: Bool
    public let note: String

    public init(transformsPath: String, pointCloudPath: String) {
        self.kind = "nerfstudio-transforms-plus-colored-point-cloud"
        self.transformsPath = transformsPath
        self.pointCloudPath = pointCloudPath
        self.containsGaussianParameters = false
        self.note = "Camera and colored-point initialization handoff; DA3-Small does not predict 3D Gaussian parameters."
    }
}

public struct MultiViewGeometryManifest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let createdAt: Date
    public let outputDirectory: String
    public let model: GeometryModelProvenance
    public let checkpoint: MultiViewGeometryCheckpointManifest
    public let units: GeometryValueUnits
    public let coordinateSystem: GeometryCoordinateSystem
    public let poseConditioned: Bool
    public let cameraSemantics: String
    public let cameraScaleAlignment: String
    public let depthScaleDivisor: Float
    public let processResolution: Int
    public let referenceViewStrategy: DepthAnything3ReferenceViewStrategy
    public let confidenceThreshold: Float
    public let confidencePercentile: Double
    public let maximumPointCount: Int
    public let pointSamplingPolicy: String
    public let pointCount: Int
    public let pointBounds: MeshBounds
    public let pointCloudRepresentation: String
    public let threeDGaussianHandoff: Geometry3DGSHandoffManifest
    public let views: [MultiViewGeometryViewManifest]
    public let artifacts: [MultiViewGeometryArtifact]

    public init(
        schemaVersion: Int = 2,
        createdAt: Date,
        outputDirectory: String,
        model: GeometryModelProvenance,
        checkpoint: MultiViewGeometryCheckpointManifest,
        poseConditioned: Bool,
        cameraSemantics: String,
        cameraScaleAlignment: String,
        depthScaleDivisor: Float,
        processResolution: Int,
        referenceViewStrategy: DepthAnything3ReferenceViewStrategy,
        confidenceThreshold: Float,
        confidencePercentile: Double,
        maximumPointCount: Int,
        pointCount: Int,
        pointBounds: MeshBounds,
        threeDGaussianHandoff: Geometry3DGSHandoffManifest,
        views: [MultiViewGeometryViewManifest],
        artifacts: [MultiViewGeometryArtifact]
    ) {
        self.schemaVersion = schemaVersion
        self.createdAt = createdAt
        self.outputDirectory = outputDirectory
        self.model = model
        self.checkpoint = checkpoint
        self.units = .relative
        self.coordinateSystem = .worldFromCameras
        self.poseConditioned = poseConditioned
        self.cameraSemantics = cameraSemantics
        self.cameraScaleAlignment = cameraScaleAlignment
        self.depthScaleDivisor = depthScaleDivisor
        self.processResolution = processResolution
        self.referenceViewStrategy = referenceViewStrategy
        self.confidenceThreshold = confidenceThreshold
        self.confidencePercentile = confidencePercentile
        self.maximumPointCount = maximumPointCount
        self.pointSamplingPolicy = "global-valid-row-major-stride-capped"
        self.pointCount = pointCount
        self.pointBounds = pointBounds
        self.pointCloudRepresentation = "colored-points-not-mesh"
        self.threeDGaussianHandoff = threeDGaussianHandoff
        self.views = views
        self.artifacts = artifacts
    }
}

public struct MultiViewGeometryExportResult: Equatable, Sendable {
    public let manifest: MultiViewGeometryManifest
    public let manifestURL: URL
}

public enum MultiViewGeometryExporter {
    @discardableResult
    public static func export(
        run: DepthAnything3RunResult,
        outputDirectory: URL,
        configuration: MultiViewGeometryExportConfiguration,
        createdAt: Date = Date()
    ) throws -> MultiViewGeometryExportResult {
        let inputViews = try run.views.map { try MultiViewGeometryView(result: $0) }
        guard !inputViews.isEmpty else { throw GeometryError.noValidGeometry }
        let views = inputViews.sorted { $0.index < $1.index }
        guard views.map(\.index) == Array(0..<views.count) else {
            throw GeometryError.invalidIntrinsics("multi-view indices must be contiguous from zero")
        }
        let destination = outputDirectory.standardizedFileURL
        let staging = destination.deletingLastPathComponent().appendingPathComponent(
            ".\(destination.lastPathComponent).mere-run-staging-\(UUID().uuidString)",
            isDirectory: true
        )
        var published = false
        defer {
            if !published { try? FileManager.default.removeItem(at: staging) }
        }
        let root = staging
        let viewsDirectory = root.appendingPathComponent("views", isDirectory: true)
        try FileManager.default.createDirectory(at: viewsDirectory, withIntermediateDirectories: true)

        let finiteConfidence = views.flatMap(\.confidence).filter(\.isFinite).sorted()
        guard !finiteConfidence.isEmpty else { throw GeometryError.noValidGeometry }
        let threshold = percentile(
            finiteConfidence,
            fraction: configuration.confidencePercentile / 100
        )
        let totalValid = views.reduce(0) { count, view in
            count + (0..<view.depth.count).reduce(0) { partial, pixel in
                partial + (isValid(view: view, pixel: pixel, confidenceThreshold: threshold) ? 1 : 0)
            }
        }
        guard totalValid > 0 else { throw GeometryError.noValidGeometry }
        let pointStride = max(1, Int(ceil(Double(totalValid) / Double(configuration.maximumPointCount))))

        var sampledDepth: [Float] = []
        let sampleStride = max(1, Int(ceil(Double(totalValid) / 2_000_000)))
        var validOrdinal = 0
        for view in views {
            for pixel in 0..<view.depth.count where isValid(
                view: view,
                pixel: pixel,
                confidenceThreshold: threshold
            ) {
                if validOrdinal.isMultiple(of: sampleStride) { sampledDepth.append(view.depth[pixel]) }
                validOrdinal += 1
            }
        }
        sampledDepth.sort()
        let depthNear = percentile(sampledDepth, fraction: 0.02)
        let depthFar = percentile(sampledDepth, fraction: 0.98)

        var positions: [Float] = []
        var colors: [UInt8] = []
        var confidences: [Float] = []
        var viewIndices: [UInt32] = []
        positions.reserveCapacity(min(totalValid, configuration.maximumPointCount) * 3)
        colors.reserveCapacity(min(totalValid, configuration.maximumPointCount) * 4)
        confidences.reserveCapacity(min(totalValid, configuration.maximumPointCount))
        viewIndices.reserveCapacity(min(totalValid, configuration.maximumPointCount))
        var artifacts: [MultiViewGeometryArtifact] = []
        var viewManifests: [MultiViewGeometryViewManifest] = []
        var globalValidOrdinal = 0

        for view in views {
            let prefix = String(format: "%06d", view.index)
            let depthURL = viewsDirectory.appendingPathComponent("\(prefix)-depth.exr")
            try OpenEXRWriter.writeFloatChannels(
                [(name: "Z", values: view.depth)],
                width: view.image.width,
                height: view.image.height,
                to: depthURL
            )
            artifacts.append(try artifact(.depthEXR, viewIndex: view.index, url: depthURL, root: root, mediaType: "image/x-exr"))

            let confidenceURL = viewsDirectory.appendingPathComponent("\(prefix)-confidence.exr")
            try OpenEXRWriter.writeFloatChannels(
                [(name: "Y", values: view.confidence)],
                width: view.image.width,
                height: view.image.height,
                to: confidenceURL
            )
            artifacts.append(try artifact(.confidenceEXR, viewIndex: view.index, url: confidenceURL, root: root, mediaType: "image/x-exr"))

            let previewURL = viewsDirectory.appendingPathComponent("\(prefix)-depth.png")
            let validity = (0..<view.depth.count).map {
                isValid(view: view, pixel: $0, confidenceThreshold: threshold) ? UInt8(1) : UInt8(0)
            }
            try GeometryPreviewWriter.writeDepth(
                view.depth,
                width: view.image.width,
                height: view.image.height,
                validity: validity,
                near: depthNear,
                far: depthFar,
                to: previewURL
            )
            artifacts.append(try artifact(.depthPreview, viewIndex: view.index, url: previewURL, root: root, mediaType: "image/png"))

            let imageURL = viewsDirectory.appendingPathComponent("\(prefix)-rgb.png")
            try MediaImageIO.writePNG(view.image, to: imageURL)
            artifacts.append(try artifact(.processedImage, viewIndex: view.index, url: imageURL, root: root, mediaType: "image/png"))

            var selectedInView = 0
            for pixel in 0..<view.depth.count where validity[pixel] != 0 {
                let select = globalValidOrdinal.isMultiple(of: pointStride)
                    && confidences.count < configuration.maximumPointCount
                globalValidOrdinal += 1
                guard select else { continue }
                positions.append(contentsOf: worldPoint(view: view, pixel: pixel))
                let color = pixel * 4
                colors.append(contentsOf: view.image.rgba8[color..<(color + 4)])
                confidences.append(view.confidence[pixel])
                viewIndices.append(UInt32(view.index))
                selectedInView += 1
            }
            viewManifests.append(
                MultiViewGeometryViewManifest(
                    index: view.index,
                    sourcePath: view.inputIdentity.path,
                    sourceByteCount: view.inputIdentity.byteCount,
                    sourceSHA256: view.inputIdentity.sha256,
                    width: view.image.width,
                    height: view.image.height,
                    preprocessing: view.preprocessingPlan,
                    camera: GeometryCameraManifest(
                        intrinsics: view.intrinsics,
                        extrinsics: view.extrinsics
                    ),
                    depthPath: relativePath(depthURL, root: root),
                    confidencePath: relativePath(confidenceURL, root: root),
                    previewPath: relativePath(previewURL, root: root),
                    processedImagePath: relativePath(imageURL, root: root),
                    selectedPointCount: selectedInView
                )
            )
        }

        let cloud = try PointCloudAsset(
            positions: positions,
            colorsRGBA8: colors,
            confidence: confidences,
            viewIndices: viewIndices,
            coordinateSystem: .worldFromCameras,
            units: .relative
        )
        let plyURL = root.appendingPathComponent("scene.ply")
        try PointCloudPLYWriter.write(cloud, to: plyURL)
        artifacts.append(try artifact(.pointCloudPLY, viewIndex: nil, url: plyURL, root: root, mediaType: "application/ply"))
        let glbURL = root.appendingPathComponent("scene.glb")
        try PointCloudGLBWriter.write(cloud, to: glbURL)
        artifacts.append(try artifact(.pointCloudGLB, viewIndex: nil, url: glbURL, root: root, mediaType: "model/gltf-binary"))

        let camerasURL = root.appendingPathComponent("cameras.json")
        try writeCameras(viewManifests, to: camerasURL)
        artifacts.append(try artifact(.camerasJSON, viewIndex: nil, url: camerasURL, root: root, mediaType: "application/json"))
        let transformsURL = root.appendingPathComponent("transforms.json")
        try writeNerfstudioTransforms(views: views, plyPath: plyURL.lastPathComponent, to: transformsURL)
        artifacts.append(try artifact(.transformsJSON, viewIndex: nil, url: transformsURL, root: root, mediaType: "application/json"))
        artifacts.sort {
            if $0.viewIndex != $1.viewIndex { return ($0.viewIndex ?? Int.max) < ($1.viewIndex ?? Int.max) }
            if $0.kind.rawValue != $1.kind.rawValue { return $0.kind.rawValue < $1.kind.rawValue }
            return $0.relativePath < $1.relativePath
        }

        let handoff = Geometry3DGSHandoffManifest(
            transformsPath: relativePath(transformsURL, root: root),
            pointCloudPath: relativePath(plyURL, root: root)
        )
        let manifest = MultiViewGeometryManifest(
            createdAt: createdAt,
            outputDirectory: destination.path,
            model: GeometryModelProvenance(
                modelID: run.checkpoint.modelID,
                upstreamRepository: run.checkpoint.repository,
                upstreamRevision: run.checkpoint.revision,
                license: run.checkpoint.license,
                weightsSHA256: run.checkpoint.weightsSHA256
            ),
            checkpoint: MultiViewGeometryCheckpointManifest(checkpoint: run.checkpoint),
            poseConditioned: run.cameraSemantics == .suppliedScaleAlignedRelative,
            cameraSemantics: run.cameraSemantics.rawValue,
            cameraScaleAlignment: run.cameraScaleAlignment,
            depthScaleDivisor: run.depthScaleDivisor,
            processResolution: run.processResolution,
            referenceViewStrategy: run.referenceViewStrategy,
            confidenceThreshold: threshold,
            confidencePercentile: configuration.confidencePercentile,
            maximumPointCount: configuration.maximumPointCount,
            pointCount: cloud.pointCount,
            pointBounds: cloud.bounds,
            threeDGaussianHandoff: handoff,
            views: viewManifests,
            artifacts: artifacts
        )
        let manifestURL = root.appendingPathComponent("scene-manifest.json")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
        try publish(staging: staging, to: destination)
        published = true
        return MultiViewGeometryExportResult(
            manifest: manifest,
            manifestURL: destination.appendingPathComponent("scene-manifest.json")
        )
    }

    private static func publish(staging: URL, to destination: URL) throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if fileManager.fileExists(atPath: destination.path) {
            _ = try fileManager.replaceItemAt(destination, withItemAt: staging)
        } else {
            try fileManager.moveItem(at: staging, to: destination)
        }
    }

    private static func isValid(
        view: MultiViewGeometryView,
        pixel: Int,
        confidenceThreshold: Float
    ) -> Bool {
        let depth = view.depth[pixel]
        let confidence = view.confidence[pixel]
        return depth.isFinite && depth > 0 && confidence.isFinite && confidence >= confidenceThreshold
    }

    static func worldPoint(view: MultiViewGeometryView, pixel: Int) -> [Float] {
        let x = pixel % view.image.width
        let y = pixel / view.image.width
        let z = view.depth[pixel]
        let camera = [
            (Float(x) + 0.5 - Float(view.intrinsics.pixelCX)) / Float(view.intrinsics.pixelFX) * z,
            (Float(y) + 0.5 - Float(view.intrinsics.pixelCY)) / Float(view.intrinsics.pixelFY) * z,
            z,
        ]
        let r = view.extrinsics.rotation.map(Float.init)
        let t = view.extrinsics.translation.map(Float.init)
        let centered = [camera[0] - t[0], camera[1] - t[1], camera[2] - t[2]]
        return [
            r[0] * centered[0] + r[3] * centered[1] + r[6] * centered[2],
            r[1] * centered[0] + r[4] * centered[1] + r[7] * centered[2],
            r[2] * centered[0] + r[5] * centered[1] + r[8] * centered[2],
        ]
    }

    private static func writeCameras(
        _ views: [MultiViewGeometryViewManifest],
        to url: URL
    ) throws {
        struct Document: Codable {
            let schemaVersion: Int
            let convention: String
            let units: GeometryValueUnits
            let views: [MultiViewGeometryViewManifest]
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(Document(
            schemaVersion: 1,
            convention: "OpenCV world-to-camera; X right, Y down, Z forward",
            units: .relative,
            views: views
        )).write(to: url, options: .atomic)
    }

    private static func writeNerfstudioTransforms(
        views: [MultiViewGeometryView],
        plyPath: String,
        to url: URL
    ) throws {
        let frames: [[String: Any]] = views.map { view in
            let imagePath = "views/\(String(format: "%06d", view.index))-rgb.png"
            return [
                "file_path": imagePath,
                "w": view.image.width,
                "h": view.image.height,
                "fl_x": view.intrinsics.pixelFX,
                "fl_y": view.intrinsics.pixelFY,
                "cx": view.intrinsics.pixelCX,
                "cy": view.intrinsics.pixelCY,
                "k1": 0,
                "k2": 0,
                "p1": 0,
                "p2": 0,
                "transform_matrix": openGLCameraToWorld(view.extrinsics),
            ]
        }
        let document: [String: Any] = [
            "camera_model": "OPENCV",
            "orientation_override": "none",
            "ply_file_path": plyPath,
            "frames": frames,
        ]
        let data = try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private static func openGLCameraToWorld(_ extrinsics: GeometryCameraExtrinsics) -> [[Double]] {
        let r = extrinsics.rotation
        let t = extrinsics.translation
        let c2wRotation = [
            r[0], r[3], r[6],
            r[1], r[4], r[7],
            r[2], r[5], r[8],
        ]
        let translation = [
            -(c2wRotation[0] * t[0] + c2wRotation[1] * t[1] + c2wRotation[2] * t[2]),
            -(c2wRotation[3] * t[0] + c2wRotation[4] * t[1] + c2wRotation[5] * t[2]),
            -(c2wRotation[6] * t[0] + c2wRotation[7] * t[1] + c2wRotation[8] * t[2]),
        ]
        // OpenCV camera axes (right, down, forward) to the OpenGL convention
        // expected by Nerfstudio transforms (right, up, backward).
        return [
            [c2wRotation[0], -c2wRotation[1], -c2wRotation[2], translation[0]],
            [c2wRotation[3], -c2wRotation[4], -c2wRotation[5], translation[1]],
            [c2wRotation[6], -c2wRotation[7], -c2wRotation[8], translation[2]],
            [0, 0, 0, 1],
        ]
    }

    private static func artifact(
        _ kind: MultiViewGeometryArtifactKind,
        viewIndex: Int?,
        url: URL,
        root: URL,
        mediaType: String
    ) throws -> MultiViewGeometryArtifact {
        MultiViewGeometryArtifact(
            kind: kind,
            viewIndex: viewIndex,
            relativePath: relativePath(url, root: root),
            mediaType: mediaType,
            byteCount: try ModelArtifactPin.fileByteCount(url),
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }

    private static func relativePath(_ url: URL, root: URL) -> String {
        let rootPath = root.path.hasSuffix("/") ? root.path : root.path + "/"
        return url.path.hasPrefix(rootPath)
            ? String(url.path.dropFirst(rootPath.count))
            : url.lastPathComponent
    }

    private static func percentile(_ sorted: [Float], fraction: Double) -> Float {
        let index = min(sorted.count - 1, max(0, Int((Double(sorted.count - 1) * fraction).rounded())))
        return sorted[index]
    }
}
