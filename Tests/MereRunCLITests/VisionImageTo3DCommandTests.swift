import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class VisionImageTo3DCommandTests: XCTestCase {
    func testBothPublicCommandPathsRegister() {
        let imageCommands = Set(Image.configuration.subcommands.map { $0.configuration.commandName })
        let visionCommands = Set(Vision.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(imageCommands.contains("reconstruct-3d"))
        XCTAssertTrue(visionCommands.contains("image-to-3d"))
    }

    func testVisionCommandParsesProductionControls() throws {
        let command = try VisionImageTo3D.parse([
            "/tmp/chair.png",
            "--output", "/tmp/chair-3d",
            "--model", "image-3d-triposr",
            "--resolution", "192",
            "--density-threshold", "21.5",
            "--foreground-ratio", "0.9",
            "--already-framed",
            "--no-vertex-colors",
            "--json",
        ])
        XCTAssertEqual(command.input, "/tmp/chair.png")
        XCTAssertEqual(command.output, "/tmp/chair-3d")
        XCTAssertEqual(command.model, "image-3d-triposr")
        XCTAssertEqual(command.resolution, 192)
        XCTAssertEqual(command.densityThreshold, 21.5)
        XCTAssertEqual(command.foregroundRatio, 0.9)
        XCTAssertTrue(command.alreadyFramed)
        XCTAssertTrue(command.noVertexColors)
        XCTAssertTrue(command.json)
    }

    func testCanonicalImageCommandParsesSameControls() throws {
        let command = try ImageReconstruct3D.parse([
            "/tmp/chair.png",
            "--resolution", "128",
            "--dry-run",
        ])
        XCTAssertEqual(command.input, "/tmp/chair.png")
        XCTAssertEqual(command.resolution, 128)
        XCTAssertTrue(command.dryRun)
    }

    func testDefaultOutputUsesInputStem() {
        let input = URL(fileURLWithPath: "/tmp/chair.asset.png")
        XCTAssertEqual(
            VisionImageTo3D.resolveOutputURL(nil, inputURL: input).path,
            "/tmp/chair.asset-3d"
        )
    }

    func testPlanDeclaresVerifiedNormalizedInferredMeshBoundary() {
        let plan = VisionImageTo3DPlanPayload(
            inputPath: "/tmp/chair.png",
            outputDirectory: "/tmp/chair-3d",
            inputWidth: 512,
            inputHeight: 512,
            checkpoint: checkpoint(),
            extractionResolution: 256,
            densityThreshold: 25,
            foregroundPolicy: "automatic-transparent-alpha",
            foregroundRatio: 0.85,
            includesVertexColors: true
        )
        XCTAssertTrue(plan.checkpointVerified)
        XCTAssertEqual(plan.checkpointFormat, .pinnedPyTorch)
        XCTAssertEqual(plan.coordinateSystem, .modelXRightYUpZForward)
        XCTAssertEqual(plan.units, .normalizedObjectSpace)
        XCTAssertTrue(plan.inferredUnseenGeometry)
        XCTAssertEqual(plan.meshExtractionAlgorithm, "native-marching-tetrahedra")
        XCTAssertEqual(
            Set(plan.outputKinds),
            ["mesh-obj", "mesh-ply", "mesh-glb", "mesh-manifest-json"]
        )
    }

    func testRunPayloadReturnsHashedManifestAndEveryMeshArtifact() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("triposr-cli-payload-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let mesh = try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            normals: [0, 0, 1, 0, 0, 1, 0, 0, 1],
            colorsRGBA8: [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255],
            inferredUnseenGeometry: true
        )
        let inputURL = root.appendingPathComponent("chair.png")
        try Data("CLI upload".utf8).write(to: inputURL)
        let checkpoint = checkpoint()
        let export = try TripoSRAssetExporter.export(
            mesh: mesh,
            inputURL: inputURL,
            checkpoint: checkpoint,
            outputDirectory: root,
            stem: "chair",
            createdAt: Date(timeIntervalSince1970: 0)
        )
        let runManifest = try TripoSRRunManifestExporter.export(
            meshExport: export,
            checkpoint: checkpoint,
            inputURL: inputURL,
            sourceWidth: 640,
            sourceHeight: 480,
            preparedWidth: 512,
            preparedHeight: 512,
            foregroundPolicy: "automatic-transparent-alpha",
            foregroundRatio: 0.85,
            croppedTransparentForeground: true,
            extractionResolution: 256,
            densityThreshold: 25,
            includesVertexColors: true
        )
        let result = TripoSRRunResult(
            export: export,
            runManifest: runManifest,
            checkpoint: checkpoint,
            sourceWidth: 640,
            sourceHeight: 480,
            preparedWidth: 512,
            preparedHeight: 512,
            foregroundPolicy: "automatic-transparent-alpha",
            foregroundRatio: 0.85,
            croppedTransparentForeground: true,
            extractionResolution: 256,
            densityThreshold: 25,
            includesVertexColors: true,
            checkpointVerificationSeconds: 0.1,
            decodingSeconds: 0.2,
            preprocessingSeconds: 0.3,
            modelLoadSeconds: 0.4,
            sceneEncodingSeconds: 0.5,
            meshExtractionSeconds: 0.6,
            exportSeconds: 0.7
        )

        let payload = try VisionImageTo3DRunPayload(result: result)
        XCTAssertEqual(payload.manifestSHA256.count, 64)
        XCTAssertEqual(payload.meshManifestSHA256.count, 64)
        XCTAssertEqual(payload.vertexCount, 3)
        XCTAssertEqual(payload.triangleCount, 1)
        XCTAssertEqual(Set(payload.artifacts.map(\.kind)), ["obj", "ply", "glb", "mesh-manifest"])
        XCTAssertTrue(payload.artifacts.allSatisfy { $0.byteCount > 0 && $0.sha256.count == 64 })
        XCTAssertEqual(payload.foregroundPolicy, "automatic-transparent-alpha")
        XCTAssertEqual(payload.foregroundRatio, 0.85)
        XCTAssertTrue(payload.inferredUnseenGeometry)
    }

    private func checkpoint() -> TripoSRCheckpoint {
        TripoSRCheckpoint(
            modelID: ModelResolver.ModelID.image3DTripoSR.rawValue,
            repository: "stabilityai/TripoSR",
            revision: "5b521936b01fbe1890f6f9baed0254ab6351c04a",
            sourceRepository: "VAST-AI-Research/TripoSR",
            sourceRevision: "107cefdc244c39106fa830359024f6a2f1c78871",
            license: "MIT",
            format: .pinnedPyTorch,
            rootURL: URL(fileURLWithPath: "/tmp/triposr"),
            weightsURL: URL(fileURLWithPath: "/tmp/triposr/model.ckpt"),
            configurationURL: URL(fileURLWithPath: "/tmp/triposr/config.yaml"),
            weightsByteCount: 1_677_246_742,
            weightsSHA256: "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee",
            sourceSHA256: "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee",
            configurationSHA256: "74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62"
        )
    }
}
