import Foundation
import MediaIO
import MereRunCore
import XCTest
@testable import MereRunCLI

final class VisionGeometryMultiViewCommandTests: XCTestCase {
    func testParsesProductionControls() throws {
        let command = try VisionGeometryMultiView.parse([
            "/tmp/a.png", "/tmp/b.png",
            "--output", "/tmp/scene",
            "--model", "vision-geometry-da3-small",
            "--cameras", "/tmp/cameras.json",
            "--process-resolution", "392",
            "--reference-view", "first",
            "--confidence-percentile", "55",
            "--max-points", "250000",
            "--json",
        ])
        XCTAssertEqual(command.images, ["/tmp/a.png", "/tmp/b.png"])
        XCTAssertEqual(command.output, "/tmp/scene")
        XCTAssertEqual(command.model, "vision-geometry-da3-small")
        XCTAssertEqual(command.cameras, "/tmp/cameras.json")
        XCTAssertEqual(command.processResolution, 392)
        XCTAssertEqual(command.referenceView, "first")
        XCTAssertEqual(command.confidencePercentile, 55)
        XCTAssertEqual(command.maxPoints, 250_000)
        XCTAssertTrue(command.json)
    }

    func testVisionRegistersGeometryMultiView() {
        let names = Set(Vision.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(names.contains("geometry-multiview"))
    }

    func testPreflightPayloadKeepsRelativePointCloudAnd3DGSBoundariesExplicit() {
        let payload = VisionGeometryMultiViewPlanPayload(
            inputPaths: ["/tmp/a.png", "/tmp/b.png"],
            outputDirectory: "/tmp/scene",
            checkpoint: checkpoint(),
            processResolution: 504,
            referenceViewStrategy: .saddleBalanced,
            poseConditioned: true,
            confidencePercentile: 40,
            maximumPointCount: 1_000_000
        )
        XCTAssertTrue(payload.checkpointVerified)
        XCTAssertEqual(payload.depthUnits, .relative)
        XCTAssertTrue(payload.poseConditioned)
        XCTAssertTrue(payload.outputKinds.contains("colored-point-cloud-glb"))
        XCTAssertTrue(payload.outputKinds.contains("nerfstudio-3dgs-initialization-handoff"))
        XCTAssertFalse(payload.outputKinds.contains("mesh"))
    }

    func testCameraDocumentRoundTripsAndEnforcesViewCount() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "da3-camera-document-tests-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let camera = DepthAnything3KnownCamera(
            intrinsics: GeometryCameraIntrinsics(
                imageWidth: 64,
                imageHeight: 48,
                normalizedFX: 1,
                normalizedFY: 1
            ),
            extrinsics: .identity
        )
        let url = root.appendingPathComponent("cameras.json")
        try JSONEncoder().encode(
            DepthAnything3CameraDocument(cameras: [camera, camera])
        ).write(to: url)

        let decoded = try XCTUnwrap(
            VisionGeometryMultiView.loadCameras(url.path, expectedCount: 2)
        )
        XCTAssertEqual(decoded, [camera, camera])
        XCTAssertThrowsError(
            try VisionGeometryMultiView.loadCameras(url.path, expectedCount: 1)
        )
    }

    func testCameraDocumentRejectsFloatOverflowAndNonRigidRotation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "da3-invalid-camera-document-tests-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let hugeTranslation = DepthAnything3KnownCamera(
            intrinsics: GeometryCameraIntrinsics(
                imageWidth: 64,
                imageHeight: 48,
                normalizedFX: 1,
                normalizedFY: 1
            ),
            extrinsics: try GeometryCameraExtrinsics(
                rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                translation: [Double(Float.greatestFiniteMagnitude) * 2, 0, 0]
            )
        )
        let overflowURL = root.appendingPathComponent("overflow.json")
        try JSONEncoder().encode(
            DepthAnything3CameraDocument(cameras: [hugeTranslation])
        ).write(to: overflowURL)
        XCTAssertThrowsError(
            try VisionGeometryMultiView.loadCameras(overflowURL.path, expectedCount: 1)
        ) { error in
            XCTAssertTrue(String(describing: error).contains("Float-representable"))
        }

        let nonRigid = DepthAnything3KnownCamera(
            intrinsics: hugeTranslation.intrinsics,
            extrinsics: try GeometryCameraExtrinsics(
                rotation: [1, 0.1, 0, 0, 1, 0, 0, 0, 1],
                translation: [0, 0, 0]
            )
        )
        let nonRigidURL = root.appendingPathComponent("non-rigid.json")
        try JSONEncoder().encode(
            DepthAnything3CameraDocument(cameras: [nonRigid])
        ).write(to: nonRigidURL)
        XCTAssertThrowsError(
            try VisionGeometryMultiView.loadCameras(nonRigidURL.path, expectedCount: 1)
        ) { error in
            XCTAssertTrue(String(describing: error).contains("rotation"))
        }
    }

    func testDefaultOutputUsesFirstInputStem() {
        XCTAssertEqual(
            VisionGeometryMultiView.resolveOutputURL(
                nil,
                firstInput: URL(fileURLWithPath: "/tmp/shot.a.png")
            ).path,
            "/tmp/shot.a-da3-scene"
        )
    }

    func testCLIResourcePreflightRejectsActivationBudgetBeforeReadingImages() {
        XCTAssertThrowsError(try VisionGeometryMultiView.validateRequestLimits(
            viewCount: 9,
            processResolution: 504
        )) { error in
            XCTAssertTrue(String(describing: error).contains("activation budget"))
        }
    }

    private func checkpoint() -> DepthAnything3Checkpoint {
        DepthAnything3Checkpoint(
            modelID: "vision-geometry-da3-small",
            repository: "depth-anything/DA3-SMALL",
            revision: String(repeating: "a", count: 40),
            sourceRepository: "ByteDance-Seed/Depth-Anything-3",
            sourceRevision: String(repeating: "b", count: 40),
            license: "Apache-2.0",
            rootURL: URL(fileURLWithPath: "/tmp/da3"),
            weightsURL: URL(fileURLWithPath: "/tmp/da3/model.safetensors"),
            configurationURL: URL(fileURLWithPath: "/tmp/da3/config.json"),
            weightsByteCount: 137_248_940,
            weightsSHA256: String(repeating: "c", count: 64),
            configurationByteCount: 1_202,
            configurationSHA256: String(repeating: "d", count: 64)
        )
    }
}
