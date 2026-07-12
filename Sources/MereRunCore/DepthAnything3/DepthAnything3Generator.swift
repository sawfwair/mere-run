import Foundation
import MediaIO
@preconcurrency import MLX

public enum DepthAnything3Progress: Equatable, Sendable {
    case verifyingCheckpoint
    case decodingImages
    case preprocessingImages
    case loadingModel
    case runningInference
    case postprocessingGeometry

    public var message: String {
        switch self {
        case .verifyingCheckpoint: "Verifying pinned DA3-Small checkpoint"
        case .decodingImages: "Decoding multi-view source images"
        case .preprocessingImages: "Applying authoritative DA3 image and camera preprocessing"
        case .loadingModel: "Loading native MLX DA3-Small model"
        case .runningInference: "Solving multi-view depth, confidence, and cameras"
        case .postprocessingGeometry: "Aligning relative geometry and camera outputs"
        }
    }
}

/// Immutable identity of the exact source bytes consumed for one DA3 view.
/// The path can later disappear (for example, an API upload temporary file),
/// while the byte count and digest remain durable provenance.
public struct DepthAnything3InputIdentity: Codable, Equatable, Sendable {
    public let path: String
    public let byteCount: Int64
    public let sha256: String

    public init(path: String, byteCount: Int64, sha256: String) {
        self.path = URL(fileURLWithPath: path).standardizedFileURL.path
        self.byteCount = byteCount
        self.sha256 = sha256.lowercased()
    }

    init(record: MeshInputRecord) {
        self.init(
            path: record.path,
            byteCount: record.byteCount,
            sha256: record.sha256
        )
    }

    public static func capture(_ url: URL) throws -> DepthAnything3InputIdentity {
        let standardized = url.standardizedFileURL
        return DepthAnything3InputIdentity(
            path: standardized.path,
            byteCount: try ModelArtifactPin.fileByteCount(standardized),
            sha256: try ModelArtifactPin.fileSHA256(standardized)
        )
    }
}

public struct DepthAnything3RunResult: Sendable {
    public let views: [DepthAnything3ViewResult]
    public let checkpoint: DepthAnything3Checkpoint
    public let referenceViewStrategy: DepthAnything3ReferenceViewStrategy
    public let cameraSemantics: DepthAnything3CameraSemantics
    public let cameraScaleAlignment: String
    public let depthScaleDivisor: Float
    public let depthUnits: GeometryValueUnits
    public let processResolution: Int
    public let checkpointVerificationSeconds: Double
    public let decodingSeconds: Double
    public let preprocessingSeconds: Double
    public let modelLoadSeconds: Double
    public let inferenceSeconds: Double
    public let postprocessingSeconds: Double

    public init(
        views: [DepthAnything3ViewResult],
        checkpoint: DepthAnything3Checkpoint,
        referenceViewStrategy: DepthAnything3ReferenceViewStrategy,
        cameraSemantics: DepthAnything3CameraSemantics,
        cameraScaleAlignment: String,
        depthScaleDivisor: Float,
        processResolution: Int,
        checkpointVerificationSeconds: Double,
        decodingSeconds: Double,
        preprocessingSeconds: Double,
        modelLoadSeconds: Double,
        inferenceSeconds: Double,
        postprocessingSeconds: Double
    ) {
        self.views = views
        self.checkpoint = checkpoint
        self.referenceViewStrategy = referenceViewStrategy
        self.cameraSemantics = cameraSemantics
        self.cameraScaleAlignment = cameraScaleAlignment
        self.depthScaleDivisor = depthScaleDivisor
        self.depthUnits = .relative
        self.processResolution = processResolution
        self.checkpointVerificationSeconds = checkpointVerificationSeconds
        self.decodingSeconds = decodingSeconds
        self.preprocessingSeconds = preprocessingSeconds
        self.modelLoadSeconds = modelLoadSeconds
        self.inferenceSeconds = inferenceSeconds
        self.postprocessingSeconds = postprocessingSeconds
    }
}

public enum DepthAnything3GeneratorError: Error, Equatable, LocalizedError, Sendable {
    case noImages
    case imageNotFound(String)
    case invalidProcessResolution(Int)
    case cameraCountMismatch(images: Int, cameras: Int)
    case inputIdentityChanged(String)

    public var errorDescription: String? {
        switch self {
        case .noImages: "Depth Anything 3 requires at least one image."
        case .imageNotFound(let path): "Depth Anything 3 image was not found: \(path)"
        case .invalidProcessResolution(let value):
            "DA3 process resolution must be at least 14 pixels; received \(value)."
        case .cameraCountMismatch(let images, let cameras):
            "DA3 received \(images) images but \(cameras) supplied cameras."
        case .inputIdentityChanged(let path):
            "DA3 source bytes changed while the image was being decoded: \(path)"
        }
    }
}

public actor DepthAnything3Generator {
    private var loadedModel: DepthAnything3Model?
    private var loadedCheckpoint: DepthAnything3Checkpoint?

    public init() {}

    public func generate(
        imageURLs: [URL],
        model requestedModel: String? = nil,
        knownCameras: [DepthAnything3KnownCamera]? = nil,
        referenceViewStrategy: DepthAnything3ReferenceViewStrategy = .saddleBalanced,
        processResolution: Int = 504,
        progress: (@Sendable (DepthAnything3Progress) -> Void)? = nil
    ) async throws -> DepthAnything3RunResult {
        guard !imageURLs.isEmpty else { throw DepthAnything3GeneratorError.noImages }
        try DepthAnything3Limits.validateRequest(
            viewCount: imageURLs.count,
            processResolution: processResolution
        )
        if let knownCameras, knownCameras.count != imageURLs.count {
            throw DepthAnything3GeneratorError.cameraCountMismatch(
                images: imageURLs.count,
                cameras: knownCameras.count
            )
        }
        let urls = imageURLs.map(\.standardizedFileURL)
        for url in urls where !FileManager.default.fileExists(atPath: url.path) {
            throw DepthAnything3GeneratorError.imageNotFound(url.path)
        }
        // Copy every caller-controlled view exactly once before checkpoint
        // resolution. Header validation, decode, and persisted provenance all
        // use the resulting private snapshots, so path replacement cannot make
        // inference consume bytes different from the recorded identities.
        let admittedInputs = try VFXImageInputSnapshotBatch.capture(
            urls,
            limits: DepthAnything3Limits.imageInputLimits
        )
        defer { admittedInputs.cleanup() }
        let sourceDimensions = admittedInputs.dimensions.map {
            (width: $0.width, height: $0.height)
        }
        try DepthAnything3Limits.validateSourceDimensions(sourceDimensions)
        if let knownCameras {
            for index in knownCameras.indices {
                try DepthAnything3CameraValidation.validate(knownCameras[index], index: index)
                let size = sourceDimensions[index]
                let intrinsics = knownCameras[index].intrinsics
                guard intrinsics.imageWidth == size.width,
                      intrinsics.imageHeight == size.height else {
                    throw DepthAnything3PreprocessingError.cameraImageDimensionMismatch(
                        index: index,
                        expectedWidth: size.width,
                        expectedHeight: size.height,
                        actualWidth: intrinsics.imageWidth,
                        actualHeight: intrinsics.imageHeight
                    )
                }
            }
        }

        progress?(.verifyingCheckpoint)
        let verificationStart = Date()
        let checkpoint = try await DepthAnything3Resources.resolve(requestedModel: requestedModel)
        let checkpointVerificationSeconds = Date().timeIntervalSince(verificationStart)

        progress?(.decodingImages)
        let decodingStart = Date()
        let inputIdentities = admittedInputs.inputRecords.map(DepthAnything3InputIdentity.init)
        let sourceImages = try admittedInputs.snapshotURLs.map(MediaImageIO.decode)
        // Recheck decoded dimensions to close the metadata/decode gap. The
        // immutable snapshots prevent concurrent replacement between checks.
        try DepthAnything3Limits.validateSourceDimensions(
            sourceImages.map { (width: $0.width, height: $0.height) }
        )
        let decodingSeconds = Date().timeIntervalSince(decodingStart)

        progress?(.preprocessingImages)
        let preprocessingStart = Date()
        let batch = try DepthAnything3Preprocessor.prepare(
            sourceImages: sourceImages,
            knownCameras: knownCameras,
            processResolution: processResolution
        )
        let preprocessingSeconds = Date().timeIntervalSince(preprocessingStart)

        progress?(.loadingModel)
        let loadStart = Date()
        let nativeModel = try loadModelIfNeeded(checkpoint)
        let modelLoadSeconds = Date().timeIntervalSince(loadStart)

        progress?(.runningInference)
        let inferenceStart = Date()
        let raw = nativeModel(
            batch.normalizedImages,
            cameraConditioning: batch.conditioning,
            referenceViewStrategy: referenceViewStrategy
        )
        MLX.eval(
            raw.depth,
            raw.confidence,
            raw.extrinsics,
            raw.intrinsics,
            raw.ray,
            raw.rayConfidence
        )
        let inferenceSeconds = Date().timeIntervalSince(inferenceStart)

        progress?(.postprocessingGeometry)
        let postprocessingStart = Date()
        let processed = try DepthAnything3Postprocessor.process(
            raw: raw,
            sourceURLs: urls,
            inputIdentities: inputIdentities,
            sourceImages: sourceImages,
            processedImages: batch.processedImages,
            preprocessingPlans: batch.plans,
            suppliedCameras: batch.processedKnownCameras
        )
        let postprocessingSeconds = Date().timeIntervalSince(postprocessingStart)
        MLX.Memory.clearCache()

        return DepthAnything3RunResult(
            views: processed.views,
            checkpoint: checkpoint,
            referenceViewStrategy: referenceViewStrategy,
            cameraSemantics: processed.cameraSemantics,
            cameraScaleAlignment: processed.cameraScaleAlignment,
            depthScaleDivisor: processed.depthScaleDivisor,
            processResolution: processResolution,
            checkpointVerificationSeconds: checkpointVerificationSeconds,
            decodingSeconds: decodingSeconds,
            preprocessingSeconds: preprocessingSeconds,
            modelLoadSeconds: modelLoadSeconds,
            inferenceSeconds: inferenceSeconds,
            postprocessingSeconds: postprocessingSeconds
        )
    }

    public func unload() {
        loadedModel = nil
        loadedCheckpoint = nil
        MLX.Memory.clearCache()
    }

    private func loadModelIfNeeded(_ checkpoint: DepthAnything3Checkpoint) throws -> DepthAnything3Model {
        if let loadedModel, loadedCheckpoint == checkpoint { return loadedModel }
        let model = try DepthAnything3Resources.loadModel(from: checkpoint)
        loadedModel = model
        loadedCheckpoint = checkpoint
        return model
    }
}
