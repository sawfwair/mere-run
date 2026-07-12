import ArgumentParser
import Foundation
import MereRunCore

struct DepthAnything3CameraDocument: Codable, Equatable {
    let schemaVersion: Int
    let cameras: [DepthAnything3KnownCamera]

    init(schemaVersion: Int = 1, cameras: [DepthAnything3KnownCamera]) {
        self.schemaVersion = schemaVersion
        self.cameras = cameras
    }
}

struct VisionGeometryMultiView: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "geometry-multiview",
        abstract: "Solve native DA3-Small multi-view relative geometry, confidence, and cameras."
    )

    @Argument(help: "One or more input image paths in view order.")
    var images: [String] = []

    @Option(name: [.customShort("o"), .long], help: "Output scene directory.")
    var output: String?

    @Option(name: [.long], help: "Managed model id or exact pinned DA3-Small directory/safetensors path.")
    var model: String?

    @Option(name: [.long], help: "Optional JSON document containing one calibrated W2C camera per image.")
    var cameras: String?

    @Option(name: [.long], help: "Upper bound for the longest processed image side. (default: 504)")
    var processResolution: Int = 504

    @Option(
        name: [.long],
        help: "Reference-view strategy: first, middle, saddle-balanced, or saddle-similarity-range."
    )
    var referenceView: String = DepthAnything3ReferenceViewStrategy.saddleBalanced.rawValue

    @Option(name: [.long], help: "Discard points below this confidence percentile. (default: 40)")
    var confidencePercentile: Double = 40

    @Option(name: [.long], help: "Maximum deterministic colored points in scene exports. (default: 1000000)")
    var maxPoints: Int = 1_000_000

    @Flag(name: [.long], help: "Verify model and inputs, then print the execution plan without inference.")
    var dryRun = false

    @Flag(name: [.long], help: "Print structured JSON on stdout.")
    var json = false

    mutating func run() async throws {
        guard !images.isEmpty else { throw ValidationError("Provide at least one image path.") }
        try Self.validateRequestLimits(
            viewCount: images.count,
            processResolution: processResolution
        )
        guard confidencePercentile.isFinite, (0...100).contains(confidencePercentile) else {
            throw ValidationError("--confidence-percentile must be between 0 and 100")
        }
        guard maxPoints > 0 else { throw ValidationError("--max-points must be positive") }
        guard let strategy = DepthAnything3ReferenceViewStrategy(rawValue: referenceView.lowercased()) else {
            throw ValidationError("Unsupported --reference-view value: \(referenceView)")
        }
        let inputURLs = images.map { URL(fileURLWithPath: $0).standardizedFileURL }
        for url in inputURLs where !FileManager.default.fileExists(atPath: url.path) {
            throw ValidationError("Input image not found: \(url.path)")
        }
        let sourceDimensions = try Self.validateResourceLimits(
            imageURLs: inputURLs,
            processResolution: processResolution
        )
        let knownCameras = try Self.loadCameras(cameras, expectedCount: inputURLs.count)
        if let knownCameras {
            for index in knownCameras.indices {
                let dimensions = sourceDimensions[index]
                let intrinsics = knownCameras[index].intrinsics
                guard intrinsics.imageWidth == dimensions.width,
                      intrinsics.imageHeight == dimensions.height else {
                    throw ValidationError(
                        "Camera \(index) describes \(intrinsics.imageWidth)x\(intrinsics.imageHeight), "
                            + "not its \(dimensions.width)x\(dimensions.height) image"
                    )
                }
            }
        }
        let outputURL = Self.resolveOutputURL(output, firstInput: inputURLs[0])

        if dryRun {
            let checkpoint = try await DepthAnything3Resources.resolve(requestedModel: model)
            print(try Self.jsonString(VisionGeometryMultiViewPlanPayload(
                inputPaths: inputURLs.map(\.path),
                outputDirectory: outputURL.path,
                checkpoint: checkpoint,
                processResolution: processResolution,
                referenceViewStrategy: strategy,
                poseConditioned: knownCameras != nil,
                confidencePercentile: confidencePercentile,
                maximumPointCount: maxPoints
            )))
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: true)
        let generator = DepthAnything3Generator()
        do {
            let run = try await generator.generate(
                imageURLs: inputURLs,
                model: model,
                knownCameras: knownCameras,
                referenceViewStrategy: strategy,
                processResolution: processResolution,
                progress: { event in CLIStderr.write("[geometry-multiview] \(event.message)\n") }
            )
            let export = try MultiViewGeometryExporter.export(
                run: run,
                outputDirectory: outputURL,
                configuration: try MultiViewGeometryExportConfiguration(
                    confidencePercentile: confidencePercentile,
                    maximumPointCount: maxPoints
                )
            )
            await generator.unload()
            if json {
                print(try Self.jsonString(try VisionGeometryMultiViewRunPayload(run: run, export: export)))
            } else {
                print(export.manifestURL.path)
            }
        } catch {
            await generator.unload()
            throw error
        }
    }

    static func resolveOutputURL(_ raw: String?, firstInput: URL) -> URL {
        if let raw, !raw.isEmpty { return URL(fileURLWithPath: raw).standardizedFileURL }
        return firstInput.deletingLastPathComponent().appendingPathComponent(
            "\(firstInput.deletingPathExtension().lastPathComponent)-da3-scene",
            isDirectory: true
        )
    }

    static func validateResourceLimits(
        imageURLs: [URL],
        processResolution: Int
    ) throws -> [(width: Int, height: Int)] {
        try validateRequestLimits(
            viewCount: imageURLs.count,
            processResolution: processResolution
        )
        do {
            return try DepthAnything3Limits.validateImageURLs(imageURLs)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
    }

    static func validateRequestLimits(
        viewCount: Int,
        processResolution: Int
    ) throws {
        do {
            try DepthAnything3Limits.validateRequest(
                viewCount: viewCount,
                processResolution: processResolution
            )
        } catch {
            throw ValidationError(error.localizedDescription)
        }
    }

    static func loadCameras(_ path: String?, expectedCount: Int) throws -> [DepthAnything3KnownCamera]? {
        guard let path, !path.isEmpty else { return nil }
        let url = URL(fileURLWithPath: path).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("Camera document not found: \(url.path)")
        }
        let document: DepthAnything3CameraDocument
        do {
            document = try JSONDecoder().decode(
                DepthAnything3CameraDocument.self,
                from: Data(contentsOf: url)
            )
        } catch {
            throw ValidationError("Invalid camera document: \(error.localizedDescription)")
        }
        guard document.schemaVersion == 1 else {
            throw ValidationError("Unsupported camera document schemaVersion \(document.schemaVersion)")
        }
        guard document.cameras.count == expectedCount else {
            throw ValidationError(
                "Camera document has \(document.cameras.count) cameras for \(expectedCount) images"
            )
        }
        for index in document.cameras.indices {
            do {
                try DepthAnything3CameraValidation.validate(document.cameras[index], index: index)
            } catch {
                throw ValidationError(error.localizedDescription)
            }
        }
        return document.cameras
    }

    static func jsonString<Value: Encodable>(_ value: Value) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

struct VisionGeometryMultiViewPlanPayload: Codable, Equatable {
    let schemaVersion: Int
    let status: String
    let inputPaths: [String]
    let outputDirectory: String
    let modelID: String
    let checkpointPath: String
    let checkpointSHA256: String
    let checkpointVerified: Bool
    let repository: String
    let revision: String
    let sourceRepository: String
    let sourceRevision: String
    let license: String
    let processResolution: Int
    let referenceViewStrategy: DepthAnything3ReferenceViewStrategy
    let poseConditioned: Bool
    let depthUnits: GeometryValueUnits
    let confidencePercentile: Double
    let maximumPointCount: Int
    let outputKinds: [String]

    init(
        schemaVersion: Int = 1,
        status: String = "planned",
        inputPaths: [String],
        outputDirectory: String,
        checkpoint: DepthAnything3Checkpoint,
        processResolution: Int,
        referenceViewStrategy: DepthAnything3ReferenceViewStrategy,
        poseConditioned: Bool,
        confidencePercentile: Double,
        maximumPointCount: Int
    ) {
        self.schemaVersion = schemaVersion
        self.status = status
        self.inputPaths = inputPaths
        self.outputDirectory = outputDirectory
        self.modelID = checkpoint.modelID
        self.checkpointPath = checkpoint.weightsURL.path
        self.checkpointSHA256 = checkpoint.weightsSHA256
        self.checkpointVerified = true
        self.repository = checkpoint.repository
        self.revision = checkpoint.revision
        self.sourceRepository = checkpoint.sourceRepository
        self.sourceRevision = checkpoint.sourceRevision
        self.license = checkpoint.license
        self.processResolution = processResolution
        self.referenceViewStrategy = referenceViewStrategy
        self.poseConditioned = poseConditioned
        self.depthUnits = .relative
        self.confidencePercentile = confidencePercentile
        self.maximumPointCount = maximumPointCount
        self.outputKinds = [
            "per-view-depth-exr",
            "per-view-confidence-exr",
            "per-view-depth-preview-png",
            "per-view-processed-rgb-png",
            "camera-json",
            "colored-point-cloud-ply",
            "colored-point-cloud-glb",
            "nerfstudio-3dgs-initialization-handoff",
            "scene-manifest-json",
        ]
    }
}

struct VisionGeometryMultiViewRunPayload: Codable {
    let schemaVersion: Int
    let status: String
    let manifestPath: String
    let manifestSHA256: String
    let modelID: String
    let checkpointSHA256: String
    let viewCount: Int
    let width: Int
    let height: Int
    let depthUnits: GeometryValueUnits
    let cameraSemantics: DepthAnything3CameraSemantics
    let cameraScaleAlignment: String
    let depthScaleDivisor: Float
    let poseConditioned: Bool
    let referenceViewStrategy: DepthAnything3ReferenceViewStrategy
    let pointCount: Int
    let pointCloudRepresentation: String
    let containsGaussianParameters: Bool
    let artifacts: [MultiViewGeometryArtifact]
    let checkpointVerificationSeconds: Double
    let decodingSeconds: Double
    let preprocessingSeconds: Double
    let modelLoadSeconds: Double
    let inferenceSeconds: Double
    let postprocessingSeconds: Double

    init(
        run: DepthAnything3RunResult,
        export: MultiViewGeometryExportResult,
        schemaVersion: Int = 1,
        status: String = "completed"
    ) throws {
        self.schemaVersion = schemaVersion
        self.status = status
        self.manifestPath = export.manifestURL.path
        self.manifestSHA256 = try ModelArtifactPin.fileSHA256(export.manifestURL)
        self.modelID = run.checkpoint.modelID
        self.checkpointSHA256 = run.checkpoint.weightsSHA256
        self.viewCount = run.views.count
        self.width = run.views.first?.processedImage.width ?? 0
        self.height = run.views.first?.processedImage.height ?? 0
        self.depthUnits = run.depthUnits
        self.cameraSemantics = run.cameraSemantics
        self.cameraScaleAlignment = run.cameraScaleAlignment
        self.depthScaleDivisor = run.depthScaleDivisor
        self.poseConditioned = run.views.first?.suppliedCamera != nil
        self.referenceViewStrategy = run.referenceViewStrategy
        self.pointCount = export.manifest.pointCount
        self.pointCloudRepresentation = export.manifest.pointCloudRepresentation
        self.containsGaussianParameters = export.manifest.threeDGaussianHandoff.containsGaussianParameters
        self.artifacts = export.manifest.artifacts
        self.checkpointVerificationSeconds = run.checkpointVerificationSeconds
        self.decodingSeconds = run.decodingSeconds
        self.preprocessingSeconds = run.preprocessingSeconds
        self.modelLoadSeconds = run.modelLoadSeconds
        self.inferenceSeconds = run.inferenceSeconds
        self.postprocessingSeconds = run.postprocessingSeconds
    }
}
