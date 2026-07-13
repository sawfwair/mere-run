import Foundation
@testable import MereRunCore
import XCTest

final class Trellis2ResourcesAndManifestTests: XCTestCase {
    func testCatalogPinsOfficialRepositoriesAndEveryRequiredCheckpoint() {
        XCTAssertEqual(Trellis2Resources.defaultModelID, "image-3d-trellis2-4b")
        XCTAssertEqual(Trellis2Resources.repository, "microsoft/TRELLIS.2-4B")
        XCTAssertEqual(Trellis2Resources.revision.count, 40)
        XCTAssertEqual(Trellis2Resources.dinoV3Repository, "facebook/dinov3-vitl16-pretrain-lvd1689m")
        XCTAssertEqual(Trellis2Resources.checkpointComponentManifests.count, 7)
        XCTAssertEqual(
            Trellis2Resources.checkpointComponentManifests.reduce(Int64(0)) { $0 + $1.byteCount },
            11_010_775_158
        )
        XCTAssertTrue(Trellis2Resources.checkpointComponentManifests.allSatisfy {
            $0.revision.count == 40 && $0.sha256.count == 64 && $0.byteCount > 0
        })
        XCTAssertEqual(
            Trellis2Resources.checkpointComponentManifests.last?.license,
            "DINOv3 License"
        )

        let spec = ManagedModelCatalog.spec(for: Trellis2Resources.defaultModelID)
        XCTAssertEqual(spec?.category, .image3D)
        XCTAssertEqual(spec?.installShape, .structuredRoot)
        XCTAssertEqual(spec?.estimatedDownloadBytes, 11_010_775_158)
        XCTAssertEqual(spec?.mountedHubFallbacks.count, 2)
        XCTAssertEqual(Set(spec?.defaultCLICommands ?? []), [
            "image reconstruct-3d-trellis2",
            "vision image-to-3d-trellis2",
        ])
    }

    func testManifestTemplateDeclaresNativeTrellis2Capabilities() {
        let manifest = MereRunModelManifest.template(for: .image3DTrellis2)
        XCTAssertEqual(manifest.engine, .trellis2)
        XCTAssertEqual(manifest.family, .threeD)
        XCTAssertEqual(manifest.tier, .max)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(Set(manifest.supports ?? []), [.imageTo3D, .meshGeneration])
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "microsoft/TRELLIS.2-4B@af44b45f2e35a493886929c6d786e563ec68364d"
        )
    }

    func testPBRWriterAndRunManifestPreserveAllMaterialChannelsAndHashes() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "trellis2-run-manifest-\(UUID().uuidString)",
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let inputURL = root.appendingPathComponent("chair.png")
        let inputData = Data("stable input bytes".utf8)
        try inputData.write(to: inputURL)
        let mesh = try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            colorsRGBA8: [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255],
            inferredUnseenGeometry: true
        )
        let coordinates = [Trellis2VoxelCoordinate(x: 1, y: 2, z: 3)]
        let grid = try Trellis2PBRVoxelGrid(
            resolution: 512,
            coordinates: coordinates,
            attributes: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        )
        let asset = Trellis2DecodedAsset(
            mesh: mesh,
            pbrVoxels: grid,
            metallic: [0.4, 0.4, 0.4],
            roughness: [0.5, 0.5, 0.5]
        )
        let checkpoint = Trellis2Checkpoint(verifiedRootURL: root)
        let inputRecord = MeshInputRecord(
            path: inputURL.path,
            byteCount: Int64(inputData.count),
            sha256: try ModelArtifactPin.fileSHA256(inputURL)
        )
        let exported = try Trellis2ArtifactExporter.export(
            asset: asset,
            checkpoint: checkpoint,
            inputURL: inputURL,
            inputRecord: inputRecord,
            outputDirectory: root,
            stem: "chair",
            sourceWidth: 640,
            sourceHeight: 480,
            foregroundPolicy: "transparent-alpha",
            croppedTransparentForeground: true,
            seed: 42,
            maximumSparseTokens: 49_152,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(exported.pbr.byteCount, 60)
        XCTAssertEqual(exported.pbr.sha256.count, 64)
        XCTAssertEqual(exported.run.manifest.input.sha256, inputRecord.sha256)
        XCTAssertEqual(exported.run.manifest.generation.pipelineResolution, 512)
        XCTAssertEqual(exported.run.manifest.generation.seed, 42)
        XCTAssertEqual(exported.run.manifest.mesh.pbrVoxelCount, 1)
        XCTAssertTrue(exported.run.manifest.mesh.includesMetallicRoughnessSidecar)
        XCTAssertEqual(
            Set(exported.run.manifest.artifacts.map(\.kind)),
            ["obj", "ply", "glb", "pbr-voxels", "mesh-manifest"]
        )
        XCTAssertTrue(exported.run.manifest.artifacts.allSatisfy {
            $0.byteCount > 0 && $0.sha256.count == 64
        })

        let pbrURL = root.appendingPathComponent(exported.pbr.relativePath)
        XCTAssertEqual(String(decoding: try Data(contentsOf: pbrURL).prefix(8), as: UTF8.self), "MRPBRV01")
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(
            Trellis2RunManifest.self,
            from: Data(contentsOf: exported.run.manifestURL)
        )
        XCTAssertEqual(decoded, exported.run.manifest)
    }
}
