import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class InstantMeshAPIContractTests: XCTestCase {
    func testInstalledManagedModelIsAdvertisedAsCompanionPrimitive() {
        XCTAssertEqual(
            APIServerContract.companionModelIDs(installedModelIDs: [
                ModelResolver.ModelID.image3DInstantMeshBase.rawValue,
            ]),
            [ModelResolver.ModelID.image3DInstantMeshBase.rawValue]
        )
    }

    func testRouteAndUploadLimitAreExplicit() {
        XCTAssertEqual(APIServerContract.instantMeshRoutePath, "/v1/vision/image-to-3d-multiview")
        XCTAssertEqual(
            APIServerContract.instantMeshRouterPath.description,
            APIServerContract.instantMeshRoutePath
        )
        XCTAssertEqual(APIServerContract.maximumInstantMeshUploadByteCount, 512 * 1024 * 1024)
    }

    func testContractAcceptsOnlyManagedModelAndFourOrSixUploadedViews() throws {
        let cameras = try InstantMeshCameraRig.official(viewCount: 4)
        let cameraData = try JSONEncoder().encode(
            InstantMeshCameraDocument(schemaVersion: 1, cameras: cameras)
        )
        let form = MultipartFormData(parts: [
            .init(
                name: "model",
                filename: nil,
                contentType: nil,
                body: Data(ModelResolver.ModelID.image3DInstantMeshBase.rawValue.utf8)
            ),
            .init(name: "resolution", filename: nil, contentType: nil, body: Data("96".utf8)),
            .init(name: "vertex_colors", filename: nil, contentType: nil, body: Data("false".utf8)),
            .init(name: "cameras", filename: nil, contentType: nil, body: cameraData),
        ] + uploads(count: 4))

        let plan = try APIServerContract.instantMeshPlan(from: form)
        XCTAssertEqual(plan.modelID, ModelResolver.ModelID.image3DInstantMeshBase.rawValue)
        XCTAssertEqual(plan.extractionResolution, 96)
        XCTAssertFalse(plan.includesVertexColors)
        XCTAssertEqual(plan.cameras, cameras)

        let defaults = try APIServerContract.instantMeshPlan(
            from: MultipartFormData(parts: uploads(count: 6))
        )
        XCTAssertEqual(defaults.extractionResolution, 128)
        XCTAssertTrue(defaults.includesVertexColors)
        XCTAssertNil(defaults.cameras)
    }

    func testContractRejectsClientPathsCheckpointUploadsAndNonImageParts() {
        for pathField in [
            "input", "input_path", "view_path", "output", "output_path",
            "output_directory", "model_path", "checkpoint", "checkpoint_path",
        ] {
            XCTAssertThrowsError(try APIServerContract.instantMeshPlan(
                from: MultipartFormData(parts: [
                    .init(name: pathField, filename: nil, contentType: nil, body: Data("/tmp/client-path".utf8)),
                ] + uploads(count: 4))
            )) { error in
                XCTAssertTrue(error.localizedDescription.contains(pathField))
            }
        }

        XCTAssertThrowsError(try APIServerContract.instantMeshPlan(
            from: MultipartFormData(parts: uploads(count: 4) + [
                .init(
                    name: "checkpoint",
                    filename: "model.safetensors",
                    contentType: "application/octet-stream",
                    body: Data([1])
                ),
            ])
        ))
        XCTAssertThrowsError(try APIServerContract.instantMeshPlan(
            from: MultipartFormData(parts: [
                .init(name: "image[]", filename: "bad.txt", contentType: "text/plain", body: Data([1])),
            ] + uploads(count: 3))
        ))
    }

    func testContractRejectsWrongViewCountsModelsControlsAndCameras() throws {
        for count in [0, 1, 3, 5, 7] {
            XCTAssertThrowsError(try APIServerContract.instantMeshPlan(
                from: MultipartFormData(parts: uploads(count: count))
            ), "view count \(count)")
        }
        for (field, value) in [
            ("model", "/tmp/model"),
            ("resolution", "1"),
            ("resolution", "257"),
            ("resolution", "nan"),
            ("vertex_colors", "yes"),
        ] {
            XCTAssertThrowsError(try APIServerContract.instantMeshPlan(
                from: MultipartFormData(parts: [
                    .init(name: field, filename: nil, contentType: nil, body: Data(value.utf8)),
                ] + uploads(count: 4))
            ), "field \(field) should reject \(value)")
        }

        let wrongCount = try JSONEncoder().encode(
            InstantMeshCameraDocument(
                schemaVersion: 1,
                cameras: try InstantMeshCameraRig.official(viewCount: 6)
            )
        )
        XCTAssertThrowsError(try APIServerContract.instantMeshPlan(
            from: MultipartFormData(parts: [
                .init(name: "cameras", filename: nil, contentType: nil, body: wrongCount),
            ] + uploads(count: 4))
        ))
    }

    func testResponseReturnsStrictBoundaryAndEveryHashedMeshArtifact() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("api-instantmesh-response-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let checkpoint = checkpoint(root: root)
        let mesh = try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            normals: [0, 0, 1, 0, 0, 1, 0, 0, 1],
            colorsRGBA8: [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255],
            inferredUnseenGeometry: true
        )
        let inputURLs = try writeInputViews(in: root, count: 4)
        let export = try InstantMeshAssetExporter.export(
            mesh: mesh,
            inputURLs: inputURLs,
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
            inputURLs: inputURLs,
            sourceDimensions: dimensions,
            preparedDimensions: prepared,
            cameraValues: cameras,
            usedOfficialCameraRig: true,
            extractionResolution: 128,
            includesVertexColors: true,
            upstreamEmptyFieldRepairApplied: true
        )
        let run = InstantMeshRunResult(
            export: export,
            runManifest: runManifest,
            checkpoint: checkpoint,
            sourceDimensions: dimensions,
            viewCount: 4,
            usedOfficialCameraRig: true,
            extractionResolution: 128,
            includesVertexColors: true,
            upstreamEmptyFieldRepairApplied: true,
            checkpointVerificationSeconds: 0.1,
            decodingSeconds: 0.2,
            preprocessingSeconds: 0.3,
            modelLoadSeconds: 0.4,
            sceneEncodingSeconds: 0.5,
            meshExtractionSeconds: 0.6,
            exportSeconds: 0.7
        )

        let response = try APIServerContract.instantMeshResponse(
            from: run,
            createdAt: Date(timeIntervalSince1970: 123)
        )
        XCTAssertEqual(response.created, 123)
        XCTAssertEqual(response.object, "vision.image-to-3d-multiview")
        XCTAssertEqual(response.model, ModelResolver.ModelID.image3DInstantMeshBase.rawValue)
        XCTAssertEqual(response.viewCount, 4)
        XCTAssertTrue(response.userSuppliedViews)
        XCTAssertFalse(response.viewGenerationIncluded)
        XCTAssertFalse(response.checkpoint.viewGenerationIncluded)
        XCTAssertFalse(response.zero123PlusPlusIncluded)
        XCTAssertFalse(response.runtimePython)
        XCTAssertFalse(response.proprietaryFlexiCubesIncluded)
        XCTAssertFalse(response.topologyMatchesUpstreamFlexiCubes)
        XCTAssertTrue(response.upstreamEmptyFieldRepairApplied)
        XCTAssertEqual(response.meshExtractionAlgorithm, "native-marching-tetrahedra")
        XCTAssertEqual(response.artifacts.count, 4)
        XCTAssertEqual(Set(response.artifacts.map(\.kind)), ["obj", "ply", "glb", "mesh-manifest"])
        XCTAssertTrue(response.artifacts.allSatisfy {
            $0.url.hasPrefix("file://") && $0.sha256.count == 64 && $0.byteCount > 0
        })
        XCTAssertEqual(response.manifest.sha256.count, 64)
        XCTAssertEqual(response.meshManifest.sha256.count, 64)
        XCTAssertEqual(response.timing.totalSeconds, 2.8, accuracy: 0.000_001)

        let encoded = try JSONEncoder().encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(object["view_generation_included"] as? Bool, false)
        XCTAssertEqual(object["zero123_plus_plus_included"] as? Bool, false)
        XCTAssertEqual(object["proprietary_flexicubes_included"] as? Bool, false)
        XCTAssertEqual(object["topology_matches_upstream_flexicubes"] as? Bool, false)
    }

    private func uploads(count: Int) -> [MultipartFormData.Part] {
        (0..<count).map { index in
            MultipartFormData.Part(
                name: "image[]",
                filename: "view-\(index).png",
                contentType: "image/png",
                body: Data([UInt8(index + 1)])
            )
        }
    }

    private func writeInputViews(in root: URL, count: Int) throws -> [URL] {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return try (0..<count).map { index in
            let url = root.appendingPathComponent("view-\(index).png")
            try Data("API view \(index)".utf8).write(to: url)
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
