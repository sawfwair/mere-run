import ArgumentParser
import Foundation
import MediaIO
import MereRunCore

struct VisionImageTo3DTrellis2: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "image-to-3d-trellis2",
        abstract: "Reconstruct a 512-resolution PBR O-Voxel mesh with native MLX TRELLIS.2."
    )

    @Argument(help: "Input object image path. Transparent alpha is required unless --already-framed is used.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output mesh and PBR artifact directory.")
    var output: String?

    @Option(name: [.long], help: "Managed TRELLIS.2 model id or verified checkpoint directory.")
    var model: String?

    @Option(name: [.long], help: "Deterministic MLX random seed.")
    var seed: UInt64 = 42

    @Option(name: [.long], help: "Safety limit for decoded 512-resolution O-Voxels.")
    var maxTokens: Int = Trellis2Generator.defaultMaximumSparseTokens

    @Flag(name: [.long], help: "Preserve framing and composite alpha over black without foreground cropping.")
    var alreadyFramed = false

    @Flag(name: [.long], help: "Export the raw porous dual-grid crust instead of the watertight narrow-band remesh.")
    var noRemesh = false

    @Option(name: [.long], help: "Narrow-band half-width in voxels for the watertight remesh; crust tears narrower than roughly twice this seal shut.")
    var remeshBand: Float = 1

    @Flag(name: [.long], help: "Verify inputs and checkpoints, then print the execution plan without loading weights.")
    var dryRun = false

    @Flag(name: [.long], help: "Print structured JSON on stdout.")
    var json = false

    mutating func run() async throws {
        try await Self.execute(
            input: input,
            output: output,
            model: model,
            seed: seed,
            maxTokens: maxTokens,
            alreadyFramed: alreadyFramed,
            noRemesh: noRemesh,
            remeshBand: remeshBand,
            dryRun: dryRun,
            json: json
        )
    }

    static func execute(
        input: String,
        output: String?,
        model: String?,
        seed: UInt64,
        maxTokens: Int,
        alreadyFramed: Bool,
        noRemesh: Bool = false,
        remeshBand: Float = 1,
        dryRun: Bool,
        json: Bool
    ) async throws {
        guard remeshBand > 0, remeshBand <= 8 else {
            throw ValidationError("--remesh-band must be in (0, 8]")
        }
        guard maxTokens >= 4_096 else {
            throw ValidationError("--max-tokens must be at least 4096")
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
        let policy: Trellis2ForegroundPolicy = alreadyFramed ? .alreadyFramed : .transparentAlpha

        if dryRun {
            let checkpoint = try await Trellis2Resources.resolve(requestedModel: model)
            let size = try MediaImageIO.size(of: inputURL)
            print(try jsonString(Trellis2PlanPayload(
                inputPath: inputURL.path,
                outputDirectory: outputURL.path,
                inputWidth: size.width,
                inputHeight: size.height,
                checkpoint: checkpoint,
                seed: seed,
                maximumSparseTokens: maxTokens,
                foregroundPolicy: alreadyFramed ? "already-framed" : "transparent-alpha"
            )))
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: true)
        let generator = Trellis2Generator()
        do {
            let result = try await generator.generate(
                imageURL: inputURL,
                outputDirectory: outputURL,
                model: model,
                foregroundPolicy: policy,
                seed: seed,
                maximumSparseTokens: maxTokens,
                remesh: noRemesh ? nil : Trellis2RemeshConfiguration(band: remeshBand),
                progress: { event in
                    CLIStderr.write("[image-to-3d-trellis2] \(event.message)\n")
                }
            )
            await generator.unload()
            if json {
                print(try jsonString(try Trellis2RunPayload(result: result)))
            } else {
                print(result.runManifest.manifestURL.path)
            }
        } catch {
            await generator.unload()
            throw error
        }
    }

    static func resolveOutputURL(_ raw: String?, inputURL: URL) -> URL {
        if let raw, !raw.isEmpty {
            return URL(fileURLWithPath: raw).standardizedFileURL
        }
        return inputURL.deletingLastPathComponent().appendingPathComponent(
            "\(inputURL.deletingPathExtension().lastPathComponent)-trellis2-3d",
            isDirectory: true
        )
    }

    static func jsonString<Value: Encodable>(_ value: Value) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

struct Trellis2PlanPayload: Codable, Equatable {
    let schemaVersion: Int
    let status: String
    let inputPath: String
    let outputDirectory: String
    let inputWidth: Int
    let inputHeight: Int
    let modelID: String
    let checkpointRoot: String
    let repository: String
    let revision: String
    let checkpointVerified: Bool
    let conditioningRepository: String
    let conditioningRevision: String
    let conditioningLicense: String
    let pipelineResolution: Int
    let seed: UInt64
    let maximumSparseTokens: Int
    let foregroundPolicy: String
    let inferenceBackend: String
    let outputKinds: [String]

    init(
        schemaVersion: Int = 1,
        status: String = "planned",
        inputPath: String,
        outputDirectory: String,
        inputWidth: Int,
        inputHeight: Int,
        checkpoint: Trellis2Checkpoint,
        seed: UInt64,
        maximumSparseTokens: Int,
        foregroundPolicy: String
    ) {
        self.schemaVersion = schemaVersion
        self.status = status
        self.inputPath = inputPath
        self.outputDirectory = outputDirectory
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.modelID = checkpoint.modelID
        self.checkpointRoot = checkpoint.rootURL.path
        self.repository = Trellis2Resources.repository
        self.revision = Trellis2Resources.revision
        self.checkpointVerified = true
        self.conditioningRepository = Trellis2Resources.dinoV3Repository
        self.conditioningRevision = Trellis2Resources.dinoV3Revision
        self.conditioningLicense = "DINOv3 License"
        self.pipelineResolution = 512
        self.seed = seed
        self.maximumSparseTokens = maximumSparseTokens
        self.foregroundPolicy = foregroundPolicy
        self.inferenceBackend = "mere.run-native-mlx"
        self.outputKinds = [
            "mesh-obj", "mesh-ply", "mesh-glb", "pbr-voxels",
            "mesh-manifest-json", "trellis2-run-manifest-json",
        ]
    }
}

struct Trellis2RunPayload: Codable {
    let schemaVersion: Int
    let status: String
    let manifestPath: String
    let manifestSHA256: String
    let meshManifestPath: String
    let meshManifestSHA256: String
    let pbrVoxelPath: String
    let pbrVoxelSHA256: String
    let modelID: String
    let repository: String
    let revision: String
    let seed: UInt64
    let maximumSparseTokens: Int
    let sourceWidth: Int
    let sourceHeight: Int
    let foregroundPolicy: String
    let croppedTransparentForeground: Bool
    let vertexCount: Int
    let triangleCount: Int
    let pbrVoxelCount: Int
    let artifacts: [Trellis2RunArtifact]

    init(result: Trellis2RunResult, schemaVersion: Int = 1, status: String = "completed") throws {
        let manifest = result.runManifest.manifest
        self.schemaVersion = schemaVersion
        self.status = status
        self.manifestPath = result.runManifest.manifestURL.path
        self.manifestSHA256 = try ModelArtifactPin.fileSHA256(result.runManifest.manifestURL)
        self.meshManifestPath = result.meshExport.manifestURL.path
        self.meshManifestSHA256 = try ModelArtifactPin.fileSHA256(result.meshExport.manifestURL)
        self.pbrVoxelPath = URL(
            fileURLWithPath: manifest.outputDirectory,
            isDirectory: true
        ).appendingPathComponent(result.pbrExport.relativePath).path
        self.pbrVoxelSHA256 = result.pbrExport.sha256
        self.modelID = result.checkpoint.modelID
        self.repository = Trellis2Resources.repository
        self.revision = Trellis2Resources.revision
        self.seed = result.seed
        self.maximumSparseTokens = result.maximumSparseTokens
        self.sourceWidth = result.sourceWidth
        self.sourceHeight = result.sourceHeight
        self.foregroundPolicy = result.foregroundPolicy
        self.croppedTransparentForeground = result.croppedTransparentForeground
        self.vertexCount = manifest.mesh.vertexCount
        self.triangleCount = manifest.mesh.triangleCount
        self.pbrVoxelCount = manifest.mesh.pbrVoxelCount
        self.artifacts = manifest.artifacts
    }
}
