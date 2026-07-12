import ArgumentParser
import Foundation
import MediaIO
import MereRunCore

struct VisionImageTo3D: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "image-to-3d",
        abstract: "Reconstruct a colored object mesh from one image with native TripoSR."
    )

    @Argument(help: "Input object image path. Transparent PNG foregrounds are cropped automatically.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output mesh directory.")
    var output: String?

    @Option(name: [.long], help: "Managed model id, pinned model.ckpt, or verified converted package.")
    var model: String?

    @Option(name: [.long], help: "Native density-grid resolution from 2 through 512.")
    var resolution: Int = 256

    @Option(name: [.long], help: "Activated-density isosurface threshold.")
    var densityThreshold: Float = TripoSRConfiguration.production.densityThreshold

    @Option(name: [.long], help: "Transparent foreground occupancy ratio in (0, 1].")
    var foregroundRatio: Float = 0.85

    @Flag(name: [.long], help: "Skip transparent-foreground crop/pad and treat the image as already framed.")
    var alreadyFramed = false

    @Flag(name: [.long], help: "Skip neural vertex-color queries and export geometry only.")
    var noVertexColors = false

    @Flag(name: [.long], help: "Verify the checkpoint and print the execution plan without loading weights.")
    var dryRun = false

    @Flag(name: [.long], help: "Print structured JSON on stdout.")
    var json = false

    mutating func run() async throws {
        try await Self.execute(
            input: input,
            output: output,
            model: model,
            resolution: resolution,
            densityThreshold: densityThreshold,
            foregroundRatio: foregroundRatio,
            alreadyFramed: alreadyFramed,
            noVertexColors: noVertexColors,
            dryRun: dryRun,
            json: json
        )
    }

    static func execute(
        input: String,
        output: String?,
        model: String?,
        resolution: Int,
        densityThreshold: Float,
        foregroundRatio: Float,
        alreadyFramed: Bool,
        noVertexColors: Bool,
        dryRun: Bool,
        json: Bool
    ) async throws {
        guard (2...512).contains(resolution) else {
            throw ValidationError("--resolution must be between 2 and 512")
        }
        guard densityThreshold.isFinite else {
            throw ValidationError("--density-threshold must be finite")
        }
        guard foregroundRatio.isFinite, foregroundRatio > 0, foregroundRatio <= 1 else {
            throw ValidationError("--foreground-ratio must be greater than 0 and at most 1")
        }

        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input image not found: \(inputURL.path)")
        }
        do {
            _ = try VFXImageInputValidator.inspectAndValidate([inputURL])
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        let outputURL = resolveOutputURL(output, inputURL: inputURL)
        let foregroundPolicy: TripoSRForegroundPolicy = alreadyFramed
            ? .alreadyFramed
            : .automaticTransparentAlpha(foregroundRatio: foregroundRatio)

        if dryRun {
            let checkpoint = try await TripoSRResources.resolve(requestedModel: model)
            let size = try MediaImageIO.size(of: inputURL)
            print(try jsonString(VisionImageTo3DPlanPayload(
                inputPath: inputURL.path,
                outputDirectory: outputURL.path,
                inputWidth: size.width,
                inputHeight: size.height,
                checkpoint: checkpoint,
                extractionResolution: resolution,
                densityThreshold: densityThreshold,
                foregroundPolicy: alreadyFramed ? "already-framed" : "automatic-transparent-alpha",
                foregroundRatio: alreadyFramed ? nil : foregroundRatio,
                includesVertexColors: !noVertexColors
            )))
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: true)
        let generator = TripoSRGenerator()
        do {
            let result = try await generator.generate(
                imageURL: inputURL,
                outputDirectory: outputURL,
                model: model,
                foregroundPolicy: foregroundPolicy,
                extractionResolution: resolution,
                densityThreshold: densityThreshold,
                includeVertexColors: !noVertexColors,
                progress: { event in CLIStderr.write("[image-to-3d] \(event.message)\n") }
            )
            await generator.unload()
            if json {
                print(try jsonString(try VisionImageTo3DRunPayload(result: result)))
            } else {
                print(result.runManifest.manifestURL.path)
            }
        } catch {
            await generator.unload()
            throw error
        }
    }

    static func resolveOutputURL(_ raw: String?, inputURL: URL) -> URL {
        if let raw, !raw.isEmpty { return URL(fileURLWithPath: raw).standardizedFileURL }
        return inputURL.deletingLastPathComponent().appendingPathComponent(
            "\(inputURL.deletingPathExtension().lastPathComponent)-3d",
            isDirectory: true
        )
    }

    static func jsonString<Value: Encodable>(_ value: Value) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

struct VisionImageTo3DPlanPayload: Codable, Equatable {
    let schemaVersion: Int
    let status: String
    let inputPath: String
    let outputDirectory: String
    let inputWidth: Int
    let inputHeight: Int
    let modelID: String
    let checkpointFormat: TripoSRCheckpointFormat
    let checkpointPath: String
    let checkpointSHA256: String
    let sourceCheckpointSHA256: String
    let checkpointVerified: Bool
    let repository: String
    let revision: String
    let sourceRepository: String
    let sourceRevision: String
    let license: String
    let extractionResolution: Int
    let densityThreshold: Float
    let foregroundPolicy: String
    let foregroundRatio: Float?
    let includesVertexColors: Bool
    let coordinateSystem: MeshCoordinateSystem
    let units: MeshUnits
    let inferredUnseenGeometry: Bool
    let meshExtractionAlgorithm: String
    let outputKinds: [String]

    init(
        schemaVersion: Int = 1,
        status: String = "planned",
        inputPath: String,
        outputDirectory: String,
        inputWidth: Int,
        inputHeight: Int,
        checkpoint: TripoSRCheckpoint,
        extractionResolution: Int,
        densityThreshold: Float,
        foregroundPolicy: String,
        foregroundRatio: Float?,
        includesVertexColors: Bool
    ) {
        self.schemaVersion = schemaVersion
        self.status = status
        self.inputPath = inputPath
        self.outputDirectory = outputDirectory
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.modelID = checkpoint.modelID
        self.checkpointFormat = checkpoint.format
        self.checkpointPath = checkpoint.weightsURL.path
        self.checkpointSHA256 = checkpoint.weightsSHA256
        self.sourceCheckpointSHA256 = checkpoint.sourceSHA256
        self.checkpointVerified = true
        self.repository = checkpoint.repository
        self.revision = checkpoint.revision
        self.sourceRepository = checkpoint.sourceRepository
        self.sourceRevision = checkpoint.sourceRevision
        self.license = checkpoint.license
        self.extractionResolution = extractionResolution
        self.densityThreshold = densityThreshold
        self.foregroundPolicy = foregroundPolicy
        self.foregroundRatio = foregroundRatio
        self.includesVertexColors = includesVertexColors
        self.coordinateSystem = .modelXRightYUpZForward
        self.units = .normalizedObjectSpace
        self.inferredUnseenGeometry = true
        self.meshExtractionAlgorithm = TripoSRRunManifestExporter.extractionAlgorithm
        self.outputKinds = ["mesh-obj", "mesh-ply", "mesh-glb", "mesh-manifest-json"]
    }
}

struct VisionImageTo3DRunPayload: Codable {
    let schemaVersion: Int
    let status: String
    let manifestPath: String
    let manifestSHA256: String
    let meshManifestPath: String
    let meshManifestSHA256: String
    let modelID: String
    let checkpointFormat: TripoSRCheckpointFormat
    let checkpointSHA256: String
    let sourceCheckpointSHA256: String
    let sourceWidth: Int
    let sourceHeight: Int
    let preparedWidth: Int
    let preparedHeight: Int
    let foregroundPolicy: String
    let foregroundRatio: Float?
    let croppedTransparentForeground: Bool
    let extractionResolution: Int
    let densityThreshold: Float
    let includesVertexColors: Bool
    let meshExtractionAlgorithm: String
    let coordinateSystem: MeshCoordinateSystem
    let units: MeshUnits
    let inferredUnseenGeometry: Bool
    let vertexCount: Int
    let triangleCount: Int
    let bounds: MeshBounds
    let artifacts: [TripoSRRunArtifact]
    let checkpointVerificationSeconds: Double
    let decodingSeconds: Double
    let preprocessingSeconds: Double
    let modelLoadSeconds: Double
    let sceneEncodingSeconds: Double
    let meshExtractionSeconds: Double
    let exportSeconds: Double

    init(
        result: TripoSRRunResult,
        schemaVersion: Int = 1,
        status: String = "completed"
    ) throws {
        let manifest = result.export.manifest
        self.schemaVersion = schemaVersion
        self.status = status
        self.manifestPath = result.runManifest.manifestURL.path
        self.manifestSHA256 = try ModelArtifactPin.fileSHA256(result.runManifest.manifestURL)
        self.meshManifestPath = result.export.manifestURL.path
        self.meshManifestSHA256 = try ModelArtifactPin.fileSHA256(result.export.manifestURL)
        self.modelID = result.checkpoint.modelID
        self.checkpointFormat = result.checkpoint.format
        self.checkpointSHA256 = result.checkpoint.weightsSHA256
        self.sourceCheckpointSHA256 = result.checkpoint.sourceSHA256
        self.sourceWidth = result.sourceWidth
        self.sourceHeight = result.sourceHeight
        self.preparedWidth = result.preparedWidth
        self.preparedHeight = result.preparedHeight
        self.foregroundPolicy = result.foregroundPolicy
        self.foregroundRatio = result.foregroundRatio
        self.croppedTransparentForeground = result.croppedTransparentForeground
        self.extractionResolution = result.extractionResolution
        self.densityThreshold = result.densityThreshold
        self.includesVertexColors = result.includesVertexColors
        self.meshExtractionAlgorithm = TripoSRRunManifestExporter.extractionAlgorithm
        self.coordinateSystem = manifest.coordinateSystem
        self.units = manifest.units
        self.inferredUnseenGeometry = manifest.inferredUnseenGeometry
        self.vertexCount = manifest.vertexCount
        self.triangleCount = manifest.triangleCount
        self.bounds = manifest.bounds
        self.artifacts = result.runManifest.manifest.artifacts
        self.checkpointVerificationSeconds = result.checkpointVerificationSeconds
        self.decodingSeconds = result.decodingSeconds
        self.preprocessingSeconds = result.preprocessingSeconds
        self.modelLoadSeconds = result.modelLoadSeconds
        self.sceneEncodingSeconds = result.sceneEncodingSeconds
        self.meshExtractionSeconds = result.meshExtractionSeconds
        self.exportSeconds = result.exportSeconds
    }
}
