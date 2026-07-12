import Foundation
@testable import MereRunCore
import XCTest

final class InstantMeshResourcesTests: MereRunCoreTestCase {
    func testCanonicalModelAndConvertedPackagePinsAreExact() {
        XCTAssertEqual(InstantMeshResources.defaultModelID, "image-3d-instantmesh-base")
        XCTAssertEqual(InstantMeshResources.convertedWeightsPin.byteCount, 1_253_463_832)
        XCTAssertEqual(
            InstantMeshResources.convertedWeightsPin.sha256,
            "2380601d17f6a817de0bf5328188ccea397af9d75c07b4b3cc476322dcca76af"
        )
        XCTAssertEqual(InstantMeshResources.convertedConfigurationPin.byteCount, 486)
        XCTAssertEqual(
            InstantMeshResources.convertedSourceManifestPin.sha256,
            "9fbda0d3875744353a4ca6ee9ee836182cb46f72aa0d241c30ee62b746d60061"
        )
        XCTAssertEqual(InstantMeshResources.convertedLicensePin.byteCount, 11_357)
        XCTAssertTrue(GeometryModelPins.instantMeshBase.license.contains("Apache-2.0"))
        XCTAssertTrue(GeometryModelPins.instantMeshBase.license.contains("view generation excluded"))
    }

    func testRuntimeRejectsAnyUnverifiedLightningCheckpoint() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("instant_mesh_base.ckpt")
        try Data("pickle-must-never-run".utf8).write(to: source)

        XCTAssertThrowsError(try InstantMeshResources.inspectExplicit(source)) { error in
            XCTAssertEqual(
                error as? InstantMeshResourceError,
                .unrecognizedPinnedSource(source.path)
            )
        }
    }

    func testRuntimeRejectsUnknownCheckpointFileType() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("instantmesh.bin")
        try Data([0]).write(to: file)

        XCTAssertThrowsError(try InstantMeshResources.inspectExplicit(file)) { error in
            XCTAssertEqual(
                error as? InstantMeshResourceError,
                .unsupportedCheckpointPath(file.path)
            )
        }
    }

    func testInspectsExactConvertedPackageWhenFixtureIsAvailable() throws {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_INSTANTMESH_CONVERTED"] ?? ""
        try XCTSkipIf(path.isEmpty || !FileManager.default.fileExists(atPath: path))

        let checkpoint = try InstantMeshResources.inspectExplicit(URL(fileURLWithPath: path))
        XCTAssertEqual(checkpoint.modelID, ModelResolver.ModelID.image3DInstantMeshBase.rawValue)
        XCTAssertEqual(checkpoint.format, .convertedSafetensors)
        XCTAssertEqual(checkpoint.weightsSHA256, InstantMeshWeights.convertedSafetensorsSHA256)
        XCTAssertFalse(checkpoint.viewGenerationIncluded)
        XCTAssertEqual(
            checkpoint.sourceSHA256,
            GeometryModelPins.instantMeshBase.artifacts.first?.sha256
        )
    }

    func testGeneratorRejectsInvalidViewCountBeforeResolvingModel() async {
        let generator = InstantMeshGenerator()
        do {
            _ = try await generator.generate(
                viewURLs: [URL(fileURLWithPath: "/tmp/one.png")],
                outputDirectory: URL(fileURLWithPath: "/tmp/output")
            )
            XCTFail("Expected invalid view count")
        } catch {
            XCTAssertEqual(error as? InstantMeshGeneratorError, .invalidViewCount(1))
        }
    }

    func testGeneratorRejectsMissingInputBeforeResolvingModel() async {
        let generator = InstantMeshGenerator()
        let missing = (0..<4).map {
            URL(fileURLWithPath: "/tmp/mere-run-missing-\($0)-\(UUID().uuidString).png")
        }
        do {
            _ = try await generator.generate(
                viewURLs: missing,
                outputDirectory: URL(fileURLWithPath: "/tmp/output")
            )
            XCTFail("Expected missing input")
        } catch {
            XCTAssertEqual(error as? InstantMeshGeneratorError, .inputViewNotFound(missing[0].path))
        }
    }

    func testAssetExportIsDeterministicAndHashed() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let mesh = try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            normals: [0, 0, 1, 0, 0, 1, 0, 0, 1],
            colorsRGBA8: [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255],
            inferredUnseenGeometry: true
        )
        let checkpoint = fixtureCheckpoint(root: root)
        let inputURLs = try writeInputViews(in: root, count: 4, prefix: "deterministic")
        let date = Date(timeIntervalSince1970: 123)
        let first = try InstantMeshAssetExporter.export(
            mesh: mesh,
            inputURLs: inputURLs,
            checkpoint: checkpoint,
            outputDirectory: root,
            stem: "object",
            createdAt: date
        )
        let firstManifestHash = try ModelArtifactPin.fileSHA256(first.manifestURL)
        let firstArtifacts = first.manifest.artifacts
        let second = try InstantMeshAssetExporter.export(
            mesh: mesh,
            inputURLs: inputURLs,
            checkpoint: checkpoint,
            outputDirectory: root,
            stem: "object",
            createdAt: date
        )

        XCTAssertEqual(second.manifest.artifacts, firstArtifacts)
        XCTAssertEqual(try ModelArtifactPin.fileSHA256(second.manifestURL), firstManifestHash)
        XCTAssertEqual(Set(second.manifest.artifacts.map(\.kind)), [.obj, .ply, .glb])
        XCTAssertTrue(second.manifest.artifacts.allSatisfy { $0.byteCount > 0 && $0.sha256.count == 64 })
        XCTAssertEqual(second.manifest.model.weightsSHA256, checkpoint.weightsSHA256)
        XCTAssertEqual(second.manifest.inputs?.map(\.path), inputURLs.map(\.path))
        XCTAssertTrue(second.manifest.inputs?.allSatisfy {
            $0.byteCount > 0 && $0.sha256.count == 64
        } == true)
    }

    func testProgressAndCheckpointDescribeStrictNativeBoundary() {
        XCTAssertTrue(InstantMeshProgress.loadingModel.message.contains("native MLX"))
        XCTAssertTrue(InstantMeshProgress.verifyingCheckpoint.message.contains("safetensors"))
        XCTAssertTrue(InstantMeshProgress.decodingViews.message.contains("user-supplied"))
    }

    func testAuthoritativeRunManifestPersistsInputsPinsControlsAndExclusions() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let checkpoint = fixtureCheckpoint(root: root)
        let mesh = try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            inferredUnseenGeometry: true
        )
        let inputs = try writeInputViews(in: root, count: 4, prefix: "licensed")
        let meshExport = try InstantMeshAssetExporter.export(
            mesh: mesh,
            inputURLs: inputs,
            checkpoint: checkpoint,
            outputDirectory: root,
            stem: "object",
            createdAt: Date(timeIntervalSince1970: 0)
        )
        let cameras = try InstantMeshCameraRig.official(viewCount: 4)
        let run = try InstantMeshRunManifestExporter.export(
            meshExport: meshExport,
            checkpoint: checkpoint,
            inputURLs: inputs,
            sourceDimensions: (0..<4).map { _ in InstantMeshSourceDimensions(width: 640, height: 480) },
            preparedDimensions: (0..<4).map { _ in InstantMeshSourceDimensions(width: 320, height: 320) },
            cameraValues: cameras,
            usedOfficialCameraRig: false,
            extractionResolution: 96,
            includesVertexColors: false,
            upstreamEmptyFieldRepairApplied: true
        )

        XCTAssertEqual(run.manifest.input.viewCount, 4)
        XCTAssertEqual(run.manifest.input.orderedViews.map(\.path), inputs.map(\.path))
        XCTAssertTrue(run.manifest.input.orderedViews.allSatisfy {
            $0.byteCount > 0 && $0.sha256.count == 64
        })
        XCTAssertEqual(
            run.manifest.input.orderedViews.map(\.sha256),
            try XCTUnwrap(meshExport.manifest.inputs).map(\.sha256)
        )
        XCTAssertEqual(
            run.manifest.input.cameraConditioning,
            InstantMeshRunManifestExporter.suppliedCameraConditioning
        )
        XCTAssertEqual(run.manifest.input.cameras, cameras)
        XCTAssertEqual(run.manifest.extraction.resolution, 96)
        XCTAssertFalse(run.manifest.extraction.includesVertexColors)
        XCTAssertTrue(run.manifest.extraction.upstreamEmptyFieldRepairApplied)
        XCTAssertEqual(
            run.manifest.extraction.algorithm,
            InstantMeshRunManifestExporter.extractionAlgorithm
        )
        XCTAssertFalse(run.manifest.boundary.viewGenerationIncluded)
        XCTAssertFalse(run.manifest.boundary.zero123PlusPlusIncluded)
        XCTAssertFalse(run.manifest.boundary.runtimePython)
        XCTAssertFalse(run.manifest.boundary.proprietaryFlexiCubesIncluded)
        XCTAssertEqual(run.manifest.checkpoint.weightsSHA256, checkpoint.weightsSHA256)
        XCTAssertEqual(run.manifest.checkpoint.sourceSHA256, checkpoint.sourceSHA256)
        XCTAssertEqual(Set(run.manifest.artifacts.map(\.kind)), ["obj", "ply", "glb", "mesh-manifest"])
        XCTAssertTrue(run.manifest.artifacts.allSatisfy { $0.byteCount > 0 && $0.sha256.count == 64 })
        XCTAssertEqual(try ModelArtifactPin.fileSHA256(run.manifestURL).count, 64)

        for input in inputs { try FileManager.default.removeItem(at: input) }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decodedRun = try decoder.decode(
            InstantMeshRunManifest.self,
            from: Data(contentsOf: run.manifestURL)
        )
        let decodedMesh = try decoder.decode(
            MeshOutputManifest.self,
            from: Data(contentsOf: meshExport.manifestURL)
        )
        XCTAssertEqual(decodedRun.input.orderedViews.map(\.sha256), run.manifest.input.orderedViews.map(\.sha256))
        XCTAssertEqual(decodedRun.input.cameras, cameras)
        XCTAssertTrue(decodedRun.extraction.upstreamEmptyFieldRepairApplied)
        XCTAssertEqual(decodedMesh.inputs, meshExport.manifest.inputs)
    }

    private func fixtureCheckpoint(root: URL) -> InstantMeshCheckpoint {
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

    private func writeInputViews(in root: URL, count: Int, prefix: String) throws -> [URL] {
        try (0..<count).map { index in
            let url = root.appendingPathComponent("\(prefix)-view-\(index).png")
            try Data("view-\(index)-\(prefix)".utf8).write(to: url)
            return url
        }
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("instantmesh-resource-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
