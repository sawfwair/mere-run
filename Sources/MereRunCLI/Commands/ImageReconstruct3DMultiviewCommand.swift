import ArgumentParser
import Foundation
import MediaIO
import MereRunCore

struct InstantMeshCameraDocument: Codable, Equatable {
    let schemaVersion: Int
    let cameras: [[Float]]
}

struct ImageReconstruct3DMultiview: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "reconstruct-3d-multiview",
        abstract: "Reconstruct a colored mesh from 4 or 6 user-supplied views with native InstantMesh."
    )

    @Option(
        name: [.customLong("view")],
        parsing: .singleValue,
        help: "Ordered object view path. Repeat exactly 4 or 6 times. No views are generated."
    )
    var views: [String] = []

    @Option(name: [.customShort("o"), .long], help: "Output mesh directory.")
    var output: String?

    @Option(
        name: [.long],
        help: "Verified converted safetensors package (or managed id, which reports the required offline conversion)."
    )
    var model: String?

    @Option(name: [.long], help: "Optional JSON file with one 16-value C2W/intrinsics camera per view.")
    var cameras: String?

    @Option(name: [.long], help: "Native SDF-grid resolution from 2 through 256.")
    var resolution: Int = InstantMeshConfiguration.production.gridResolution

    @Flag(name: [.long], help: "Skip neural vertex-color queries and export geometry only.")
    var noVertexColors = false

    @Flag(name: [.long], help: "Verify inputs/checkpoint and print the execution plan without loading weights.")
    var dryRun = false

    @Flag(name: [.long], help: "Print structured JSON on stdout.")
    var json = false

    mutating func run() async throws {
        try await Self.execute(
            views: views,
            output: output,
            model: model,
            cameras: cameras,
            resolution: resolution,
            noVertexColors: noVertexColors,
            dryRun: dryRun,
            json: json
        )
    }

    static func execute(
        views: [String],
        output: String?,
        model: String?,
        cameras: String?,
        resolution: Int,
        noVertexColors: Bool,
        dryRun: Bool,
        json: Bool
    ) async throws {
        guard views.count == 4 || views.count == 6 else {
            throw ValidationError("Repeat --view exactly 4 or 6 times")
        }
        guard (2...256).contains(resolution) else {
            throw ValidationError("--resolution must be between 2 and 256")
        }
        let inputURLs = views.map { URL(fileURLWithPath: $0).standardizedFileURL }
        for inputURL in inputURLs where !FileManager.default.fileExists(atPath: inputURL.path) {
            throw ValidationError("Input view not found: \(inputURL.path)")
        }
        do {
            _ = try VFXImageInputValidator.inspectAndValidate(inputURLs)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        let cameraValues = try loadCameras(cameras, expectedCount: inputURLs.count)
        let outputURL = resolveOutputURL(output, firstViewURL: inputURLs[0])

        if dryRun {
            let checkpoint = try await InstantMeshResources.resolve(requestedModel: model)
            let dimensions = try inputURLs.map { url in
                let size = try MediaImageIO.size(of: url)
                return InstantMeshSourceDimensions(width: size.width, height: size.height)
            }
            print(try jsonString(InstantMeshPlanPayload(
                inputPaths: inputURLs.map(\.path),
                sourceDimensions: dimensions,
                outputDirectory: outputURL.path,
                checkpoint: checkpoint,
                usesSuppliedCameras: cameraValues != nil,
                extractionResolution: resolution,
                includesVertexColors: !noVertexColors
            )))
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: true)
        let generator = InstantMeshGenerator()
        do {
            let result = try await generator.generate(
                viewURLs: inputURLs,
                outputDirectory: outputURL,
                model: model,
                cameras: cameraValues,
                extractionResolution: resolution,
                includeVertexColors: !noVertexColors,
                progress: { event in CLIStderr.write("[image-to-3d-multiview] \(event.message)\n") }
            )
            await generator.unload()
            if json {
                print(try jsonString(try InstantMeshRunPayload(result: result)))
            } else {
                print(result.runManifest.manifestURL.path)
            }
        } catch {
            await generator.unload()
            throw error
        }
    }

    static func resolveOutputURL(_ raw: String?, firstViewURL: URL) -> URL {
        if let raw, !raw.isEmpty { return URL(fileURLWithPath: raw).standardizedFileURL }
        return firstViewURL.deletingLastPathComponent().appendingPathComponent(
            "\(firstViewURL.deletingPathExtension().lastPathComponent)-instantmesh-3d",
            isDirectory: true
        )
    }

    static func loadCameras(_ raw: String?, expectedCount: Int) throws -> [[Float]]? {
        guard let raw, !raw.isEmpty else { return nil }
        let url = URL(fileURLWithPath: raw).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("Camera JSON not found: \(url.path)")
        }
        let document: InstantMeshCameraDocument
        do {
            document = try JSONDecoder().decode(InstantMeshCameraDocument.self, from: Data(contentsOf: url))
        } catch {
            throw ValidationError("Invalid InstantMesh camera JSON: \(error.localizedDescription)")
        }
        guard document.schemaVersion == 1 else {
            throw ValidationError("InstantMesh camera JSON schemaVersion must be 1")
        }
        guard document.cameras.count == expectedCount else {
            throw ValidationError("Camera JSON must contain exactly \(expectedCount) cameras")
        }
        for (index, camera) in document.cameras.enumerated()
        where camera.count != 16 || !camera.allSatisfy(\.isFinite) {
            throw ValidationError("Camera \(index) must contain 16 finite values")
        }
        return document.cameras
    }

    static func jsonString<Value: Encodable>(_ value: Value) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

struct VisionImageTo3DMultiview: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "image-to-3d-multiview",
        abstract: "VFX alias for native 4/6-view InstantMesh reconstruction."
    )

    @Option(name: [.customLong("view")], parsing: .singleValue, help: "Ordered view path; repeat 4 or 6 times.")
    var views: [String] = []

    @Option(name: [.customShort("o"), .long], help: "Output mesh directory.")
    var output: String?

    @Option(name: [.long], help: "Verified converted safetensors package.")
    var model: String?

    @Option(name: [.long], help: "Optional camera JSON file.")
    var cameras: String?

    @Option(name: [.long], help: "Native SDF-grid resolution from 2 through 256.")
    var resolution: Int = InstantMeshConfiguration.production.gridResolution

    @Flag(name: [.long], help: "Skip neural vertex colors.")
    var noVertexColors = false

    @Flag(name: [.long], help: "Verify and print the execution plan only.")
    var dryRun = false

    @Flag(name: [.long], help: "Print structured JSON on stdout.")
    var json = false

    mutating func run() async throws {
        try await ImageReconstruct3DMultiview.execute(
            views: views,
            output: output,
            model: model,
            cameras: cameras,
            resolution: resolution,
            noVertexColors: noVertexColors,
            dryRun: dryRun,
            json: json
        )
    }
}

struct InstantMeshPlanPayload: Codable, Equatable {
    let schemaVersion: Int
    let status: String
    let inputPaths: [String]
    let sourceDimensions: [InstantMeshSourceDimensions]
    let outputDirectory: String
    let modelID: String
    let checkpointFormat: InstantMeshCheckpointFormat
    let checkpointPath: String
    let checkpointSHA256: String
    let sourceCheckpointSHA256: String
    let configurationSHA256: String
    let sourceManifestSHA256: String
    let checkpointVerified: Bool
    let repository: String
    let revision: String
    let sourceRepository: String
    let sourceRevision: String
    let license: String
    let viewCount: Int
    let userSuppliedViews: Bool
    let viewGenerationIncluded: Bool
    let zero123PlusPlusIncluded: Bool
    let runtimePython: Bool
    let proprietaryFlexiCubesIncluded: Bool
    let cameraRig: String
    let extractionResolution: Int
    let includesVertexColors: Bool
    let meshExtractionAlgorithm: String
    let topologyMatchesUpstreamFlexiCubes: Bool
    let coordinateSystem: MeshCoordinateSystem
    let units: MeshUnits
    let inferredUnseenGeometry: Bool
    let outputKinds: [String]

    init(
        schemaVersion: Int = 1,
        status: String = "planned",
        inputPaths: [String],
        sourceDimensions: [InstantMeshSourceDimensions],
        outputDirectory: String,
        checkpoint: InstantMeshCheckpoint,
        usesSuppliedCameras: Bool,
        extractionResolution: Int,
        includesVertexColors: Bool
    ) {
        self.schemaVersion = schemaVersion
        self.status = status
        self.inputPaths = inputPaths
        self.sourceDimensions = sourceDimensions
        self.outputDirectory = outputDirectory
        self.modelID = checkpoint.modelID
        self.checkpointFormat = checkpoint.format
        self.checkpointPath = checkpoint.weightsURL.path
        self.checkpointSHA256 = checkpoint.weightsSHA256
        self.sourceCheckpointSHA256 = checkpoint.sourceSHA256
        self.configurationSHA256 = checkpoint.configurationSHA256
        self.sourceManifestSHA256 = checkpoint.sourceManifestSHA256
        self.checkpointVerified = true
        self.repository = checkpoint.repository
        self.revision = checkpoint.revision
        self.sourceRepository = checkpoint.sourceRepository
        self.sourceRevision = checkpoint.sourceRevision
        self.license = checkpoint.license
        self.viewCount = inputPaths.count
        self.userSuppliedViews = true
        self.viewGenerationIncluded = false
        self.zero123PlusPlusIncluded = false
        self.runtimePython = false
        self.proprietaryFlexiCubesIncluded = false
        self.cameraRig = usesSuppliedCameras ? "supplied-c2w-intrinsics" : "official-deterministic-conditioning-rig"
        self.extractionResolution = extractionResolution
        self.includesVertexColors = includesVertexColors
        self.meshExtractionAlgorithm = InstantMeshRunManifestExporter.extractionAlgorithm
        self.topologyMatchesUpstreamFlexiCubes = false
        self.coordinateSystem = .modelXRightYUpZForward
        self.units = .normalizedObjectSpace
        self.inferredUnseenGeometry = true
        self.outputKinds = [
            "mesh-obj",
            "mesh-ply",
            "mesh-glb",
            "mesh-manifest-json",
            "instantmesh-run-manifest-json",
        ]
    }
}

struct InstantMeshRunPayload: Codable {
    let schemaVersion: Int
    let status: String
    let manifestPath: String
    let manifestSHA256: String
    let meshManifestPath: String
    let meshManifestSHA256: String
    let modelID: String
    let checkpointFormat: InstantMeshCheckpointFormat
    let checkpointSHA256: String
    let sourceCheckpointSHA256: String
    let sourceManifestSHA256: String
    let sourceDimensions: [InstantMeshSourceDimensions]
    let viewCount: Int
    let userSuppliedViews: Bool
    let usedOfficialCameraRig: Bool
    let viewGenerationIncluded: Bool
    let zero123PlusPlusIncluded: Bool
    let runtimePython: Bool
    let proprietaryFlexiCubesIncluded: Bool
    let extractionResolution: Int
    let includesVertexColors: Bool
    let meshExtractionAlgorithm: String
    let upstreamEmptyFieldRepairApplied: Bool
    let topologyMatchesUpstreamFlexiCubes: Bool
    let coordinateSystem: MeshCoordinateSystem
    let units: MeshUnits
    let inferredUnseenGeometry: Bool
    let vertexCount: Int
    let triangleCount: Int
    let bounds: MeshBounds
    let artifacts: [InstantMeshRunArtifact]
    let checkpointVerificationSeconds: Double
    let decodingSeconds: Double
    let preprocessingSeconds: Double
    let modelLoadSeconds: Double
    let sceneEncodingSeconds: Double
    let meshExtractionSeconds: Double
    let exportSeconds: Double

    init(result: InstantMeshRunResult, schemaVersion: Int = 1, status: String = "completed") throws {
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
        self.sourceManifestSHA256 = result.checkpoint.sourceManifestSHA256
        self.sourceDimensions = result.sourceDimensions
        self.viewCount = result.viewCount
        self.userSuppliedViews = true
        self.usedOfficialCameraRig = result.usedOfficialCameraRig
        self.viewGenerationIncluded = false
        self.zero123PlusPlusIncluded = false
        self.runtimePython = false
        self.proprietaryFlexiCubesIncluded = false
        self.extractionResolution = result.extractionResolution
        self.includesVertexColors = result.includesVertexColors
        self.meshExtractionAlgorithm = InstantMeshRunManifestExporter.extractionAlgorithm
        self.upstreamEmptyFieldRepairApplied = result.upstreamEmptyFieldRepairApplied
        self.topologyMatchesUpstreamFlexiCubes = false
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
