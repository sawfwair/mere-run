@testable import StudioKit
import XCTest

/// The mesh counts a 3D result card shows are read by name from the manifests the CLI writes:
/// the engine's run manifest (`mesh.vertexCount`, and TRELLIS.2's `mesh.pbrVoxelCount`) or the
/// shared mesh manifest (`MeshOutputManifest`, counts at the top level).
final class StudioMeshSummaryTests: XCTestCase {
    /// `TripoSRRunManifest`, as `image reconstruct-3d` writes `<stem>-run-manifest.json`.
    private static let tripoSRRunManifest = """
    {"schemaVersion": 1, "createdAt": "2026-09-24T09:41:00Z", "outputDirectory": "/tmp/chair",
     "checkpoint": {"modelID": "image-3d-triposr", "format": "converted-mlx", "weightsSHA256": "ab", "sourceSHA256": "cd"},
     "input": {"path": "/tmp/chair.png", "sourceWidth": 512, "sourceHeight": 512, "foregroundPolicy": "opaque-already-framed"},
     "extraction": {"resolution": 256, "densityThreshold": 25, "includesVertexColors": true,
       "algorithm": "native-marching-tetrahedra", "topologyCompatibility": "same-sampled-isosurface"},
     "mesh": {"coordinateSystem": "x-right-y-up-z-forward", "units": "normalized-object-space",
       "inferredUnseenGeometry": true, "vertexCount": 12480, "triangleCount": 24956,
       "bounds": {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]}},
     "artifacts": [{"kind": "glb", "relativePath": "chair.glb", "mediaType": "model/gltf-binary", "byteCount": 1024, "sha256": "ef"}]}
    """

    /// `Trellis2RunManifest`, as `image reconstruct-3d-trellis2` writes `<stem>-trellis2-run-manifest.json`.
    private static let trellisRunManifest = """
    {"schemaVersion": 1, "createdAt": "2026-09-24T09:41:00Z", "outputDirectory": "/tmp/chair",
     "modelID": "image-3d-trellis2-4b", "repository": "microsoft/TRELLIS.2-4B", "revision": "main", "license": "MIT",
     "inferenceBackend": "native-mlx", "checkpointComponents": [],
     "input": {"path": "/tmp/chair.png", "sourceWidth": 1024, "sourceHeight": 1024},
     "generation": {"seed": 42, "textureSeed": 42, "maximumSparseTokens": 2097152},
     "mesh": {"coordinateSystem": "x-right-y-up-z-forward", "units": "normalized-object-space",
       "inferredUnseenGeometry": true, "vertexCount": 88312, "triangleCount": 176620,
       "bounds": {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]}, "pbrVoxelCount": 1203776,
       "includesVertexColors": true, "includesMetallicRoughnessSidecar": true},
     "artifacts": []}
    """

    /// `MeshOutputManifest`, the shared `<stem>-manifest.json` every engine writes.
    private static let meshManifest = """
    {"schemaVersion": 1, "createdAt": "2026-09-24T09:41:00Z", "inputPaths": ["/tmp/chair.png"],
     "outputDirectory": "/tmp/chair", "model": {"id": "image-3d-triposr", "family": "triposr"},
     "coordinateSystem": "x-right-y-up-z-forward", "units": "normalized-object-space",
     "inferredUnseenGeometry": true, "vertexCount": 12480, "triangleCount": 24956,
     "bounds": {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]}, "artifacts": []}
    """

    /// `IdolRunManifest`, as `vision image-to-3d-idol` writes `manifest.json` beside `object.ply`.
    private static let idolManifest = """
    {"schemaVersion": 1, "status": "completed", "backend": "mere.run-native-mlx", "repository": "speridlabs/idol",
     "revision": "main", "referenceRevision": "main", "inputPath": "/tmp/chair.png", "inputSHA256": "ab",
     "mode": "synthetic", "alreadyFramed": false,
     "settings": {"steps": 50, "guidance": 3, "timeShift": 3, "gaussianCount": 262144, "seed": 42},
     "gaussianCount": 262144, "camera": {}, "checkpoints": [], "artifacts": []}
    """

    func testRunManifestsDecodeTheirMeshCounts() throws {
        let tripoSR = try XCTUnwrap(StudioMeshSummary.decode(Data(Self.tripoSRRunManifest.utf8)))
        XCTAssertEqual(tripoSR, StudioMeshSummary(vertexCount: 12_480, triangleCount: 24_956))
        XCTAssertEqual(tripoSR.text, "12,480 vertices · 24,956 triangles")

        let trellis = try XCTUnwrap(StudioMeshSummary.decode(Data(Self.trellisRunManifest.utf8)))
        XCTAssertEqual(trellis, StudioMeshSummary(vertexCount: 88_312, triangleCount: 176_620, pbrVoxelCount: 1_203_776))
        XCTAssertEqual(trellis.text, "88,312 vertices · 176,620 triangles · 1,203,776 PBR voxels")
    }

    func testTheSharedMeshManifestDecodesAtTheTopLevel() throws {
        let summary = try XCTUnwrap(StudioMeshSummary.decode(Data(Self.meshManifest.utf8)))
        XCTAssertEqual(summary, StudioMeshSummary(vertexCount: 12_480, triangleCount: 24_956))
        XCTAssertNil(StudioMeshSummary.decode(Data(#"{"schemaVersion": 1, "status": "completed"}"#.utf8)), "a receipt is not a manifest")
        XCTAssertNil(StudioMeshSummary.decode(Data("not json".utf8)))
    }

    /// IDOL writes a Gaussian splat, not a mesh: its manifest counts the Gaussians, and a row
    /// reads it from the plain `manifest.json` IDOL names it.
    func testAnIdolManifestCountsItsGaussians() throws {
        let summary = try XCTUnwrap(StudioMeshSummary.decode(Data(Self.idolManifest.utf8)))
        XCTAssertEqual(summary, StudioMeshSummary(gaussianCount: 262_144))
        XCTAssertTrue(summary.isSplat)
        XCTAssertEqual(summary.text, "262,144 Gaussians")
        XCTAssertFalse(try XCTUnwrap(StudioMeshSummary.decode(Data(Self.meshManifest.utf8))).isSplat)

        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("splat-summary-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let ply = folder.appendingPathComponent("object.ply")
        let camera = folder.appendingPathComponent("camera.json")
        let manifest = folder.appendingPathComponent("manifest.json")
        try Data().write(to: ply)
        try Data(#"{"fov": 0.6}"#.utf8).write(to: camera)
        try Data(Self.idolManifest.utf8).write(to: manifest)
        let item = Self.row(outputURL: ply, artifacts: [ply, camera, manifest])
        XCTAssertEqual(StudioMeshSummary.load(item: item), StudioMeshSummary(gaussianCount: 262_144))
    }

    /// A row's summary comes from its run manifest when it has one (the only file that counts
    /// PBR voxels), else from the shared mesh manifest; other JSON beside the mesh is never read.
    func testARowsSummaryPrefersTheRunManifest() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("mesh-summary-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let glb = folder.appendingPathComponent("chair.glb")
        let mesh = folder.appendingPathComponent("chair-manifest.json")
        let run = folder.appendingPathComponent("chair-trellis2-run-manifest.json")
        let timings = folder.appendingPathComponent("chair-timings.json")
        try Data().write(to: glb)
        try Data(Self.meshManifest.utf8).write(to: mesh)
        try Data(Self.trellisRunManifest.utf8).write(to: run)
        try Data(#"{"total_seconds": 12.5}"#.utf8).write(to: timings)

        var item = Self.row(outputURL: glb, artifacts: [glb, timings, mesh, run])
        XCTAssertEqual(StudioMeshSummary.load(item: item)?.pbrVoxelCount, 1_203_776, "the run manifest wins")
        item = Self.row(outputURL: glb, artifacts: [glb, mesh, timings])
        XCTAssertEqual(StudioMeshSummary.load(item: item), StudioMeshSummary(vertexCount: 12_480, triangleCount: 24_956))
        item = Self.row(outputURL: glb, artifacts: [glb, timings])
        XCTAssertNil(StudioMeshSummary.load(item: item), "no manifest, no summary")
    }

    private static func row(outputURL: URL, artifacts: [URL]) -> StudioLibraryItem {
        StudioLibraryItem(
            id: UUID(), mode: .createImage, prompt: "", inputURL: nil, outputURL: outputURL, createdAt: Date(),
            updatedAt: Date(), status: .completed, exitCode: 0, commandPreview: "mere.run image reconstruct-3d chair.png",
            outputText: nil, templateID: .imageReconstruct3D, artifactURLs: artifacts
        )
    }
}
