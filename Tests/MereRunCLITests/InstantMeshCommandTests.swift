import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class InstantMeshCommandTests: XCTestCase {
    func testBothPublicMultiviewCommandPathsRegister() {
        let imageCommands = Set(Image.configuration.subcommands.map { $0.configuration.commandName })
        let visionCommands = Set(Vision.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(imageCommands.contains("reconstruct-3d-multiview"))
        XCTAssertTrue(visionCommands.contains("image-to-3d-multiview"))
    }

    func testCanonicalCommandParsesRepeatedViewsAndProductionControls() throws {
        let command = try ImageReconstruct3DMultiview.parse([
            "--view", "/tmp/0.png",
            "--view", "/tmp/1.png",
            "--view", "/tmp/2.png",
            "--view", "/tmp/3.png",
            "--output", "/tmp/object-3d",
            "--model", "/tmp/instantmesh-native",
            "--cameras", "/tmp/cameras.json",
            "--resolution", "96",
            "--no-vertex-colors",
            "--dry-run",
            "--json",
        ])

        XCTAssertEqual(command.views, ["/tmp/0.png", "/tmp/1.png", "/tmp/2.png", "/tmp/3.png"])
        XCTAssertEqual(command.output, "/tmp/object-3d")
        XCTAssertEqual(command.model, "/tmp/instantmesh-native")
        XCTAssertEqual(command.cameras, "/tmp/cameras.json")
        XCTAssertEqual(command.resolution, 96)
        XCTAssertTrue(command.noVertexColors)
        XCTAssertTrue(command.dryRun)
        XCTAssertTrue(command.json)
    }

    func testVFXAliasParsesSixViews() throws {
        var arguments: [String] = []
        for index in 0..<6 {
            arguments.append(contentsOf: ["--view", "/tmp/\(index).png"])
        }
        let command = try VisionImageTo3DMultiview.parse(arguments)
        XCTAssertEqual(command.views.count, 6)
        XCTAssertEqual(command.resolution, 128)
    }

    func testDefaultOutputUsesFirstViewStem() {
        XCTAssertEqual(
            ImageReconstruct3DMultiview.resolveOutputURL(
                nil,
                firstViewURL: URL(fileURLWithPath: "/tmp/chair.front.png")
            ).path,
            "/tmp/chair.front-instantmesh-3d"
        )
    }

    func testCameraDocumentRequiresCountAndFiniteSixteenValueRows() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("instantmesh-camera-cli-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("cameras.json")
        let cameras = try InstantMeshCameraRig.official(viewCount: 4)
        try JSONEncoder().encode(
            InstantMeshCameraDocument(schemaVersion: 1, cameras: cameras)
        ).write(to: url)

        XCTAssertEqual(
            try ImageReconstruct3DMultiview.loadCameras(url.path, expectedCount: 4),
            cameras
        )
        XCTAssertThrowsError(
            try ImageReconstruct3DMultiview.loadCameras(url.path, expectedCount: 6)
        )
    }

    func testPlanDeclaresStrictLicenseRuntimeAndTopologyBoundaries() {
        let plan = InstantMeshPlanPayload(
            inputPaths: (0..<4).map { "/tmp/view-\($0).png" },
            sourceDimensions: (0..<4).map { _ in InstantMeshSourceDimensions(width: 512, height: 512) },
            outputDirectory: "/tmp/object-3d",
            checkpoint: checkpoint(root: URL(fileURLWithPath: "/tmp/instantmesh")),
            usesSuppliedCameras: false,
            extractionResolution: 128,
            includesVertexColors: true
        )

        XCTAssertEqual(plan.viewCount, 4)
        XCTAssertTrue(plan.userSuppliedViews)
        XCTAssertFalse(plan.viewGenerationIncluded)
        XCTAssertFalse(plan.zero123PlusPlusIncluded)
        XCTAssertFalse(plan.runtimePython)
        XCTAssertFalse(plan.proprietaryFlexiCubesIncluded)
        XCTAssertFalse(plan.topologyMatchesUpstreamFlexiCubes)
        XCTAssertEqual(plan.meshExtractionAlgorithm, "native-marching-tetrahedra")
        XCTAssertEqual(plan.checkpointFormat, .convertedSafetensors)
        XCTAssertTrue(plan.checkpointVerified)
        XCTAssertTrue(plan.license.contains("Apache-2.0"))
        XCTAssertEqual(
            Set(plan.outputKinds),
            [
                "mesh-obj", "mesh-ply", "mesh-glb", "mesh-manifest-json",
                "instantmesh-run-manifest-json",
            ]
        )
    }

    func testRunPayloadReturnsHashedManifestAndEveryMeshArtifact() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("instantmesh-cli-payload-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let checkpoint = checkpoint(root: root)
        let mesh = try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            normals: [0, 0, 1, 0, 0, 1, 0, 0, 1],
            colorsRGBA8: [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255],
            inferredUnseenGeometry: true
        )
        let inputs = try writeInputViews(in: root, count: 4)
        let export = try InstantMeshAssetExporter.export(
            mesh: mesh,
            inputURLs: inputs,
            checkpoint: checkpoint,
            outputDirectory: root,
            stem: "object",
            createdAt: Date(timeIntervalSince1970: 0)
        )
        let dimensions = (0..<4).map { _ in InstantMeshSourceDimensions(width: 640, height: 480) }
        let prepared = (0..<4).map { _ in InstantMeshSourceDimensions(width: 320, height: 320) }
        let cameras = try InstantMeshCameraRig.official(viewCount: 4)
        let runManifest = try InstantMeshRunManifestExporter.export(
            meshExport: export,
            checkpoint: checkpoint,
            inputURLs: inputs,
            sourceDimensions: dimensions,
            preparedDimensions: prepared,
            cameraValues: cameras,
            usedOfficialCameraRig: true,
            extractionResolution: 128,
            includesVertexColors: true,
            upstreamEmptyFieldRepairApplied: false
        )
        let result = InstantMeshRunResult(
            export: export,
            runManifest: runManifest,
            checkpoint: checkpoint,
            sourceDimensions: dimensions,
            viewCount: 4,
            usedOfficialCameraRig: true,
            extractionResolution: 128,
            includesVertexColors: true,
            upstreamEmptyFieldRepairApplied: false,
            checkpointVerificationSeconds: 0.1,
            decodingSeconds: 0.2,
            preprocessingSeconds: 0.3,
            modelLoadSeconds: 0.4,
            sceneEncodingSeconds: 0.5,
            meshExtractionSeconds: 0.6,
            exportSeconds: 0.7
        )

        let payload = try InstantMeshRunPayload(result: result)
        XCTAssertEqual(payload.manifestSHA256.count, 64)
        XCTAssertEqual(payload.meshManifestSHA256.count, 64)
        XCTAssertEqual(payload.vertexCount, 3)
        XCTAssertEqual(payload.triangleCount, 1)
        XCTAssertEqual(Set(payload.artifacts.map(\.kind)), ["obj", "ply", "glb", "mesh-manifest"])
        XCTAssertTrue(payload.artifacts.allSatisfy { $0.byteCount > 0 && $0.sha256.count == 64 })
        XCTAssertFalse(payload.zero123PlusPlusIncluded)
        XCTAssertFalse(payload.proprietaryFlexiCubesIncluded)
        XCTAssertFalse(payload.upstreamEmptyFieldRepairApplied)
    }

    func testGuideResolvesBothMultiviewSpellingsAndExplainsConversion() throws {
        let vision = try GuideCommand.resolveEntry(
            commandPath: ["vision", "image-to-3d-multiview"],
            model: ModelResolver.ModelID.image3DInstantMeshBase.rawValue
        )
        let image = try GuideCommand.resolveEntry(
            commandPath: ["image", "reconstruct-3d-multiview"],
            model: ModelResolver.ModelID.image3DInstantMeshBase.rawValue
        )
        XCTAssertEqual(vision, image)
        let rendered = try GuideCommand.render(entry: vision, model: nil, json: false)
        XCTAssertTrue(rendered.contains("does not create missing views"))
        XCTAssertTrue(rendered.contains("never deserialize its Pickle payload at runtime"))
        XCTAssertTrue(rendered.contains("does not claim triangle-topology parity"))
    }

    private func writeInputViews(in root: URL, count: Int) throws -> [URL] {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return try (0..<count).map { index in
            let url = root.appendingPathComponent("view-\(index).png")
            try Data("CLI view \(index)".utf8).write(to: url)
            return url
        }
    }

    private func checkpoint(root: URL) -> InstantMeshCheckpoint {
        InstantMeshCheckpoint(
            modelID: ModelResolver.ModelID.image3DInstantMeshBase.rawValue,
            repository: "TencentARC/InstantMesh",
            revision: "b785b4ecfb6636ef34a08c748f96f6a5686244d0",
            sourceRepository: "TencentARC/InstantMesh",
            sourceRevision: "08822c52fdc399b93ea00e4fa9e596344ed52ccc",
            license: "Apache-2.0 reconstruction weights; view generation excluded",
            format: .convertedSafetensors,
            rootURL: root,
            weightsURL: root.appendingPathComponent("model.safetensors"),
            configurationURL: root.appendingPathComponent("config.json"),
            sourceManifestURL: root.appendingPathComponent("SOURCE.json"),
            weightsByteCount: 1_253_463_832,
            weightsSHA256: "2380601d17f6a817de0bf5328188ccea397af9d75c07b4b3cc476322dcca76af",
            sourceSHA256: "22701cd25201d624ebb1568b93cf91b43a2c32006835c08fe73e1f3c9f6c44b5",
            configurationSHA256: "33f89581172ab2d46759a1632b6e57ca9f9f1c6c23567468157cb4b48a3bc781",
            sourceManifestSHA256: "9fbda0d3875744353a4ca6ee9ee836182cb46f72aa0d241c30ee62b746d60061",
            viewGenerationIncluded: false
        )
    }
}
