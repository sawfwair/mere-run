import Foundation
import MediaIO
@preconcurrency import MLX

public enum TripoSRProgress: Equatable, Sendable {
    case verifyingCheckpoint
    case decodingImage
    case preprocessingImage
    case loadingModel
    case encodingScene
    case extractingMesh
    case exportingAssets

    public var message: String {
        switch self {
        case .verifyingCheckpoint: "Verifying pinned TripoSR checkpoint"
        case .decodingImage: "Decoding source image"
        case .preprocessingImage: "Preparing TripoSR conditioning image"
        case .loadingModel: "Loading native MLX TripoSR model"
        case .encodingScene: "Encoding image into the native triplane scene field"
        case .extractingMesh: "Extracting and coloring the normalized object mesh"
        case .exportingAssets: "Writing OBJ, PLY, GLB, and manifest artifacts"
        }
    }
}

public struct TripoSRRunResult: Sendable {
    public let export: MeshExportResult
    public let runManifest: TripoSRRunManifestExport
    public let checkpoint: TripoSRCheckpoint
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let preparedWidth: Int
    public let preparedHeight: Int
    public let foregroundPolicy: String
    public let foregroundRatio: Float?
    public let croppedTransparentForeground: Bool
    public let extractionResolution: Int
    public let densityThreshold: Float
    public let includesVertexColors: Bool
    public let checkpointVerificationSeconds: Double
    public let decodingSeconds: Double
    public let preprocessingSeconds: Double
    public let modelLoadSeconds: Double
    public let sceneEncodingSeconds: Double
    public let meshExtractionSeconds: Double
    public let exportSeconds: Double

    public init(
        export: MeshExportResult,
        runManifest: TripoSRRunManifestExport,
        checkpoint: TripoSRCheckpoint,
        sourceWidth: Int,
        sourceHeight: Int,
        preparedWidth: Int,
        preparedHeight: Int,
        foregroundPolicy: String,
        foregroundRatio: Float?,
        croppedTransparentForeground: Bool,
        extractionResolution: Int,
        densityThreshold: Float,
        includesVertexColors: Bool,
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
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.preparedWidth = preparedWidth
        self.preparedHeight = preparedHeight
        self.foregroundPolicy = foregroundPolicy
        self.foregroundRatio = foregroundRatio
        self.croppedTransparentForeground = croppedTransparentForeground
        self.extractionResolution = extractionResolution
        self.densityThreshold = densityThreshold
        self.includesVertexColors = includesVertexColors
        self.checkpointVerificationSeconds = checkpointVerificationSeconds
        self.decodingSeconds = decodingSeconds
        self.preprocessingSeconds = preprocessingSeconds
        self.modelLoadSeconds = modelLoadSeconds
        self.sceneEncodingSeconds = sceneEncodingSeconds
        self.meshExtractionSeconds = meshExtractionSeconds
        self.exportSeconds = exportSeconds
    }
}

public enum TripoSRGeneratorError: Error, Equatable, LocalizedError, Sendable {
    case inputImageNotFound(String)
    case invalidExtractionResolution(Int)
    case invalidDensityThreshold(Float)

    public var errorDescription: String? {
        switch self {
        case .inputImageNotFound(let path):
            "TripoSR input image was not found: \(path)"
        case .invalidExtractionResolution(let value):
            "TripoSR extraction resolution must be between 2 and 512; received \(value)."
        case .invalidDensityThreshold(let value):
            "TripoSR density threshold must be finite; received \(value)."
        }
    }
}

public actor TripoSRGenerator {
    private var loadedModel: TripoSRModel?
    private var loadedCheckpoint: TripoSRCheckpoint?

    public init() {}

    public func generate(
        imageURL: URL,
        outputDirectory: URL,
        model requestedModel: String? = nil,
        foregroundPolicy: TripoSRForegroundPolicy = .automaticTransparentAlpha(),
        extractionResolution: Int = 256,
        densityThreshold: Float = TripoSRConfiguration.production.densityThreshold,
        includeVertexColors: Bool = true,
        progress: (@Sendable (TripoSRProgress) -> Void)? = nil
    ) async throws -> TripoSRRunResult {
        let inputURL = imageURL.standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw TripoSRGeneratorError.inputImageNotFound(inputURL.path)
        }
        // Copy caller-controlled bytes once before model resolution. Header
        // validation, decode, and durable provenance all use this immutable
        // private snapshot, closing the path-replacement gap.
        let admittedInput = try VFXImageInputSnapshotBatch.capture([inputURL])
        defer { admittedInput.cleanup() }
        let snapshotURL = admittedInput.snapshotURLs[0]
        guard (2...512).contains(extractionResolution) else {
            throw TripoSRGeneratorError.invalidExtractionResolution(extractionResolution)
        }
        guard densityThreshold.isFinite else {
            throw TripoSRGeneratorError.invalidDensityThreshold(densityThreshold)
        }
        let foregroundPolicyName: String
        let foregroundRatio: Float?
        switch foregroundPolicy {
        case .automaticTransparentAlpha(let ratio):
            foregroundPolicyName = "automatic-transparent-alpha"
            foregroundRatio = ratio
        case .alreadyFramed:
            foregroundPolicyName = "already-framed"
            foregroundRatio = nil
        }

        progress?(.verifyingCheckpoint)
        let checkpointStart = Date()
        let checkpoint = try await TripoSRResources.resolve(requestedModel: requestedModel)
        let checkpointVerificationSeconds = Date().timeIntervalSince(checkpointStart)

        progress?(.decodingImage)
        let decodingStart = Date()
        let image = try MediaImageIO.decode(snapshotURL)
        _ = try VFXImageInputValidator.validate(
            width: image.width,
            height: image.height,
            path: inputURL.path
        )
        let decodingSeconds = Date().timeIntervalSince(decodingStart)

        progress?(.preprocessingImage)
        let preprocessingStart = Date()
        let prepared = try TripoSRPreprocessor.prepare(
            image: image,
            foregroundPolicy: foregroundPolicy
        )
        let preprocessingSeconds = Date().timeIntervalSince(preprocessingStart)

        progress?(.loadingModel)
        let loadStart = Date()
        let nativeModel = try loadModelIfNeeded(checkpoint)
        let modelLoadSeconds = Date().timeIntervalSince(loadStart)

        progress?(.encodingScene)
        let encodingStart = Date()
        let sceneCode = nativeModel(prepared.image)
        MLX.eval(sceneCode.planes)
        let sceneEncodingSeconds = Date().timeIntervalSince(encodingStart)

        progress?(.extractingMesh)
        let extractionStart = Date()
        let mesh = try TripoSRIsosurfaceExtractor.extractMesh(
            model: nativeModel,
            sceneCode: sceneCode,
            configuration: TripoSRMeshExtractionConfiguration(
                resolution: extractionResolution,
                densityThreshold: densityThreshold,
                includeVertexColors: includeVertexColors
            )
        )
        let meshExtractionSeconds = Date().timeIntervalSince(extractionStart)

        progress?(.exportingAssets)
        let exportStart = Date()
        let export = try TripoSRAssetExporter.export(
            mesh: mesh,
            inputURL: inputURL,
            checkpoint: checkpoint,
            outputDirectory: outputDirectory.standardizedFileURL,
            stem: inputURL.deletingPathExtension().lastPathComponent,
            inputRecord: admittedInput.inputRecords[0]
        )
        let runManifest = try TripoSRRunManifestExporter.export(
            meshExport: export,
            checkpoint: checkpoint,
            inputURL: inputURL,
            sourceWidth: prepared.sourceWidth,
            sourceHeight: prepared.sourceHeight,
            preparedWidth: prepared.preparedWidth,
            preparedHeight: prepared.preparedHeight,
            foregroundPolicy: foregroundPolicyName,
            foregroundRatio: foregroundRatio,
            croppedTransparentForeground: prepared.croppedTransparentForeground,
            extractionResolution: extractionResolution,
            densityThreshold: densityThreshold,
            includesVertexColors: includeVertexColors
        )
        let exportSeconds = Date().timeIntervalSince(exportStart)
        MLX.Memory.clearCache()

        return TripoSRRunResult(
            export: export,
            runManifest: runManifest,
            checkpoint: checkpoint,
            sourceWidth: prepared.sourceWidth,
            sourceHeight: prepared.sourceHeight,
            preparedWidth: prepared.preparedWidth,
            preparedHeight: prepared.preparedHeight,
            foregroundPolicy: foregroundPolicyName,
            foregroundRatio: foregroundRatio,
            croppedTransparentForeground: prepared.croppedTransparentForeground,
            extractionResolution: extractionResolution,
            densityThreshold: densityThreshold,
            includesVertexColors: includeVertexColors,
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

    private func loadModelIfNeeded(_ checkpoint: TripoSRCheckpoint) throws -> TripoSRModel {
        if let loadedModel, loadedCheckpoint == checkpoint { return loadedModel }
        let model = try TripoSRResources.loadModel(from: checkpoint)
        loadedModel = model
        loadedCheckpoint = checkpoint
        return model
    }
}
