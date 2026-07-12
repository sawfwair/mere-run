import Foundation
import MediaIO
@preconcurrency import MLX

public enum InstantMeshProgress: Equatable, Sendable {
    case verifyingCheckpoint
    case decodingViews
    case preprocessingViews
    case loadingModel
    case encodingScene
    case extractingMesh
    case exportingAssets

    public var message: String {
        switch self {
        case .verifyingCheckpoint: "Verifying converted InstantMesh safetensors package"
        case .decodingViews: "Decoding user-supplied object views"
        case .preprocessingViews: "Preparing ordered InstantMesh views and camera conditioning"
        case .loadingModel: "Loading native MLX InstantMesh reconstruction model"
        case .encodingScene: "Encoding supplied views into the native triplane scene field"
        case .extractingMesh: "Extracting and coloring the normalized object mesh"
        case .exportingAssets: "Writing OBJ, PLY, GLB, and manifest artifacts"
        }
    }
}

public struct InstantMeshSourceDimensions: Codable, Equatable, Sendable {
    public let width: Int
    public let height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }
}

public struct InstantMeshRunResult: Sendable {
    public let export: MeshExportResult
    public let runManifest: InstantMeshRunManifestExport
    public let checkpoint: InstantMeshCheckpoint
    public let sourceDimensions: [InstantMeshSourceDimensions]
    public let viewCount: Int
    public let usedOfficialCameraRig: Bool
    public let extractionResolution: Int
    public let includesVertexColors: Bool
    public let upstreamEmptyFieldRepairApplied: Bool
    public let checkpointVerificationSeconds: Double
    public let decodingSeconds: Double
    public let preprocessingSeconds: Double
    public let modelLoadSeconds: Double
    public let sceneEncodingSeconds: Double
    public let meshExtractionSeconds: Double
    public let exportSeconds: Double

    public init(
        export: MeshExportResult,
        runManifest: InstantMeshRunManifestExport,
        checkpoint: InstantMeshCheckpoint,
        sourceDimensions: [InstantMeshSourceDimensions],
        viewCount: Int,
        usedOfficialCameraRig: Bool,
        extractionResolution: Int,
        includesVertexColors: Bool,
        upstreamEmptyFieldRepairApplied: Bool,
        checkpointVerificationSeconds: Double,
        decodingSeconds: Double,
        preprocessingSeconds: Double,
        modelLoadSeconds: Double,
        sceneEncodingSeconds: Double,
        meshExtractionSeconds: Double,
        exportSeconds: Double
    ) {
        self.export = export
        self.runManifest = runManifest
        self.checkpoint = checkpoint
        self.sourceDimensions = sourceDimensions
        self.viewCount = viewCount
        self.usedOfficialCameraRig = usedOfficialCameraRig
        self.extractionResolution = extractionResolution
        self.includesVertexColors = includesVertexColors
        self.upstreamEmptyFieldRepairApplied = upstreamEmptyFieldRepairApplied
        self.checkpointVerificationSeconds = checkpointVerificationSeconds
        self.decodingSeconds = decodingSeconds
        self.preprocessingSeconds = preprocessingSeconds
        self.modelLoadSeconds = modelLoadSeconds
        self.sceneEncodingSeconds = sceneEncodingSeconds
        self.meshExtractionSeconds = meshExtractionSeconds
        self.exportSeconds = exportSeconds
    }
}

public enum InstantMeshGeneratorError: Error, Equatable, LocalizedError, Sendable {
    case invalidViewCount(Int)
    case inputViewNotFound(String)
    case invalidExtractionResolution(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidViewCount(let count):
            "InstantMesh reconstruction requires exactly 4 or 6 user-supplied views; received \(count)."
        case .inputViewNotFound(let path):
            "InstantMesh input view was not found: \(path)"
        case .invalidExtractionResolution(let value):
            "InstantMesh extraction resolution must be between 2 and 256; received \(value)."
        }
    }
}

public actor InstantMeshGenerator {
    private var loadedModel: InstantMeshModel?
    private var loadedCheckpoint: InstantMeshCheckpoint?

    public init() {}

    public func generate(
        viewURLs: [URL],
        outputDirectory: URL,
        model requestedModel: String? = nil,
        cameras: [[Float]]? = nil,
        extractionResolution: Int = InstantMeshConfiguration.production.gridResolution,
        includeVertexColors: Bool = true,
        progress: (@Sendable (InstantMeshProgress) -> Void)? = nil
    ) async throws -> InstantMeshRunResult {
        guard viewURLs.count == 4 || viewURLs.count == 6 else {
            throw InstantMeshGeneratorError.invalidViewCount(viewURLs.count)
        }
        guard (2...256).contains(extractionResolution) else {
            throw InstantMeshGeneratorError.invalidExtractionResolution(extractionResolution)
        }
        let inputs = viewURLs.map(\.standardizedFileURL)
        for input in inputs where !FileManager.default.fileExists(atPath: input.path) {
            throw InstantMeshGeneratorError.inputViewNotFound(input.path)
        }
        // Freeze the exact admitted bytes before any potentially long model
        // verification. Decode and provenance then refer to the same inputs
        // even if an original path is replaced concurrently.
        let admittedInputs = try VFXImageInputSnapshotBatch.capture(inputs)
        defer { admittedInputs.cleanup() }

        progress?(.verifyingCheckpoint)
        let checkpointStart = Date()
        let checkpoint = try await InstantMeshResources.resolve(requestedModel: requestedModel)
        let checkpointVerificationSeconds = Date().timeIntervalSince(checkpointStart)

        progress?(.decodingViews)
        let decodingStart = Date()
        let images = try admittedInputs.snapshotURLs.map(MediaImageIO.decode)
        for (index, image) in images.enumerated() {
            _ = try VFXImageInputValidator.validate(
                width: image.width,
                height: image.height,
                path: inputs[index].path
            )
        }
        let dimensions = images.map { InstantMeshSourceDimensions(width: $0.width, height: $0.height) }
        let decodingSeconds = Date().timeIntervalSince(decodingStart)

        progress?(.preprocessingViews)
        let preprocessingStart = Date()
        let prepared = try InstantMeshPreprocessor.prepare(sourceImages: images, cameras: cameras)
        let preprocessingSeconds = Date().timeIntervalSince(preprocessingStart)

        progress?(.loadingModel)
        let loadStart = Date()
        let nativeModel = try loadModelIfNeeded(checkpoint)
        let modelLoadSeconds = Date().timeIntervalSince(loadStart)

        progress?(.encodingScene)
        let encodingStart = Date()
        let sceneCode = nativeModel(images: prepared.images, cameras: prepared.cameras)
        MLX.eval(sceneCode.planes)
        let sceneEncodingSeconds = Date().timeIntervalSince(encodingStart)

        progress?(.extractingMesh)
        let extractionStart = Date()
        let extraction = try InstantMeshIsosurfaceExtractor.extractMeshWithMetadata(
            model: nativeModel,
            sceneCode: sceneCode,
            configuration: InstantMeshMeshExtractionConfiguration(
                gridResolution: extractionResolution,
                includeVertexColors: includeVertexColors
            )
        )
        let mesh = extraction.mesh
        let meshExtractionSeconds = Date().timeIntervalSince(extractionStart)

        progress?(.exportingAssets)
        let exportStart = Date()
        let export = try InstantMeshAssetExporter.export(
            mesh: mesh,
            inputURLs: inputs,
            checkpoint: checkpoint,
            outputDirectory: outputDirectory.standardizedFileURL,
            stem: "instantmesh-\(inputs.count)-view",
            inputRecords: admittedInputs.inputRecords
        )
        let preparedDimensions = prepared.processedImages.map {
            InstantMeshSourceDimensions(width: $0.width, height: $0.height)
        }
        let runManifest = try InstantMeshRunManifestExporter.export(
            meshExport: export,
            checkpoint: checkpoint,
            inputURLs: inputs,
            sourceDimensions: dimensions,
            preparedDimensions: preparedDimensions,
            cameraValues: prepared.cameraValues,
            usedOfficialCameraRig: prepared.usedOfficialCameraRig,
            extractionResolution: extractionResolution,
            includesVertexColors: includeVertexColors,
            upstreamEmptyFieldRepairApplied: extraction.upstreamEmptyFieldRepairApplied
        )
        let exportSeconds = Date().timeIntervalSince(exportStart)
        MLX.Memory.clearCache()

        return InstantMeshRunResult(
            export: export,
            runManifest: runManifest,
            checkpoint: checkpoint,
            sourceDimensions: dimensions,
            viewCount: inputs.count,
            usedOfficialCameraRig: prepared.usedOfficialCameraRig,
            extractionResolution: extractionResolution,
            includesVertexColors: includeVertexColors,
            upstreamEmptyFieldRepairApplied: extraction.upstreamEmptyFieldRepairApplied,
            checkpointVerificationSeconds: checkpointVerificationSeconds,
            decodingSeconds: decodingSeconds,
            preprocessingSeconds: preprocessingSeconds,
            modelLoadSeconds: modelLoadSeconds,
            sceneEncodingSeconds: sceneEncodingSeconds,
            meshExtractionSeconds: meshExtractionSeconds,
            exportSeconds: exportSeconds
        )
    }

    public func unload() {
        loadedModel = nil
        loadedCheckpoint = nil
        MLX.Memory.clearCache()
    }

    private func loadModelIfNeeded(_ checkpoint: InstantMeshCheckpoint) throws -> InstantMeshModel {
        if let loadedModel, loadedCheckpoint == checkpoint { return loadedModel }
        let model = try InstantMeshResources.loadModel(from: checkpoint)
        loadedModel = model
        loadedCheckpoint = checkpoint
        return model
    }
}
