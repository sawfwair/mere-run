import Foundation
import MediaIO
import XCTest
@testable import MereRunCore
@testable import MereRunCLI

final class InstantMeshGenerationOperationTests: XCTestCase {
    private enum Expected: Error { case failure }

    func testCLIAndAPIShareDefaultsAndOrderedSuppliedCameras() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for explicit in [false, true] {
            let cameras = try InstantMeshCameraRig.official(viewCount: 4)
            let document = try JSONEncoder().encode(InstantMeshCameraDocument(schemaVersion: 1, cameras: cameras))
            let cameraURL = root.appendingPathComponent("cameras.json")
            try document.write(to: cameraURL)
            let fields = explicit ? ["resolution": "16", "vertex_colors": "false",
                                     "cameras": String(decoding: document, as: UTF8.self)] : [:]
            let api = try APIServerContract.instantMeshPlan(from: Self.form(fields))
            let views = (0..<4).map { "/input-\($0).png" }
            let cli = try ImageReconstruct3DMultiview.makeGenerationRequest(
                views: views, output: "/output", model: api.modelID,
                cameras: explicit ? cameraURL.path : nil,
                resolution: explicit ? 16 : InstantMeshConfiguration.production.gridResolution,
                noVertexColors: explicit
            )
            XCTAssertEqual(cli, api.request(viewURLs: views.map { URL(fileURLWithPath: $0) },
                                           outputDirectory: URL(fileURLWithPath: "/output")))
            if explicit { XCTAssertEqual(cli.settings.cameras, cameras) }
        }
    }

    func testInvalidResolutionCameraCountAndRowsFailBeforeInputAccess() throws {
        for resolution in [1, 257, Int.max] {
            XCTAssertThrowsError(try InstantMeshGenerationSettings(extractionResolution: resolution))
            XCTAssertThrowsError(try APIServerContract.instantMeshPlan(from: Self.form(["resolution": String(resolution)])))
        }
        let cameras = try InstantMeshCameraRig.official(viewCount: 4)
        XCTAssertThrowsError(try InstantMeshGenerationSettings(cameras: cameras).validate(viewCount: 6))
        XCTAssertThrowsError(try InstantMeshGenerationSettings(cameras: [[Float](repeating: 0, count: 15)]))
        XCTAssertThrowsError(try InstantMeshGenerationSettings(cameras: [[Float](repeating: .nan, count: 16)]))
        XCTAssertThrowsError(try InstantMeshGenerationSettings().validate(viewCount: 5))
        XCTAssertThrowsError(try APIServerContract.instantMeshPlan(from: Self.form(["model": "/custom/model"])))
    }

    func testPreparationPreservesViewOrderAndCreatesNoOutput() throws {
        let request = try Self.request()
        defer { try? FileManager.default.removeItem(at: request.viewURLs[0].deletingLastPathComponent()) }
        XCTAssertEqual(try InstantMeshGenerationOperation.prepare(request).map(\.width), [2, 3, 4, 5])
        XCTAssertFalse(FileManager.default.fileExists(atPath: request.outputDirectory.path))
    }

    func testExecutionRechecksInputBeforeRuntimePreparation() async throws {
        let request = try Self.request()
        defer { try? FileManager.default.removeItem(at: request.viewURLs[0].deletingLastPathComponent()) }
        _ = try InstantMeshGenerationOperation.prepare(request)
        try Data("invalid replacement".utf8).write(to: request.viewURLs[0])
        do {
            _ = try await InstantMeshGenerationOperation.execute(request, prepareRuntime: {
                XCTFail("Changed input must fail before runtime preparation")
            }, makeRuntime: {
                XCTFail("Changed input must fail before construction")
                return .init(generate: { _, _ in throw Expected.failure }, unload: {})
            })
            XCTFail("Expected image validation failure")
        } catch { XCTAssertFalse(error is Expected) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: request.outputDirectory.path))
    }

    func testSuccessAndFailureAwaitOneUnloadWithinAdmission() async throws {
        for fail in [false, true] {
            let request = try Self.request()
            defer { try? FileManager.default.removeItem(at: request.viewURLs[0].deletingLastPathComponent()) }
            let trace = InstantMeshOperationTrace()
            let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
            let runtime = InstantMeshGenerationOperation.Runtime(generate: { actual, _ in
                XCTAssertEqual(actual, request)
                await trace.record("generate")
                if fail { throw Expected.failure }
                return try InstantMeshGenerationOperationTests.fixtureResult(actual)
            }, unload: {
                let state = await admission.snapshot()
                XCTAssertEqual(state.activeRequests, 1)
                await Task.yield()
                await trace.record("unload")
            })
            do {
                let result = try await withVFXRequestAdmission(using: admission) {
                    try await InstantMeshGenerationOperation.execute(request, makeRuntime: { runtime })
                }
                XCTAssertFalse(fail)
                for artifact in result.runManifest.manifest.artifacts {
                    let url = request.outputDirectory.appendingPathComponent(artifact.relativePath)
                    XCTAssertEqual(try ModelArtifactPin.fileSHA256(url), artifact.sha256)
                }
            } catch Expected.failure { XCTAssertTrue(fail) }
            let events = await trace.events
            XCTAssertEqual(events, ["generate", "unload"])
            let state = await admission.snapshot()
            XCTAssertEqual(state.activeRequests, 0)
            XCTAssertEqual(state.totalAdmittedRequests, 1)
        }
    }

    func testCancellationDuringGenerationOrUnloadCannotReturnSuccess() async throws {
        for cancelInUnload in [false, true] {
            let request = try Self.request()
            defer { try? FileManager.default.removeItem(at: request.viewURLs[0].deletingLastPathComponent()) }
            let trace = InstantMeshOperationTrace()
            let runtime = InstantMeshGenerationOperation.Runtime(generate: { actual, _ in
                if !cancelInUnload { withUnsafeCurrentTask { $0?.cancel() } }
                return try InstantMeshGenerationOperationTests.fixtureResult(actual)
            }, unload: {
                if cancelInUnload { withUnsafeCurrentTask { $0?.cancel() } }
                await trace.record("unload")
            })
            let task = Task { @Sendable in
                try await InstantMeshGenerationOperation.execute(request, makeRuntime: { runtime })
            }
            do { _ = try await task.value; XCTFail("Cancellation must propagate") }
            catch is CancellationError {}
            let events = await trace.events
            XCTAssertEqual(events, ["unload"])
        }
    }

    private static func form(_ fields: [String: String]) -> MultipartFormData {
        MultipartFormData(parts: fields.map {
            .init(name: $0.key, filename: nil, contentType: nil, body: Data($0.value.utf8))
        } + (0..<4).map { index in
            .init(name: "image[]", filename: "input-\(index).png", contentType: "image/png", body: Data([1]))
        })
    }

    private static func request() throws -> InstantMeshGenerationRequest {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("instantmesh-operation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let views = try (0..<4).map { index in
            let image = root.appendingPathComponent("input-\(index).png")
            let width = index + 2
            try MediaImageIO.writePNG(
                try MediaImage(width: width, height: 2, rgba8: Array(repeating: 255, count: width * 8)), to: image
            )
            return image
        }
        return InstantMeshGenerationRequest(viewURLs: views, outputDirectory: root.appendingPathComponent("output"),
                                           settings: try InstantMeshGenerationSettings())
    }

    private static func fixtureResult(_ request: InstantMeshGenerationRequest) throws -> InstantMeshRunResult {
        let checkpoint = checkpoint(root: request.outputDirectory)
        let mesh = try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            normals: [0, 0, 1, 0, 0, 1, 0, 0, 1],
            colorsRGBA8: [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255],
            inferredUnseenGeometry: true
        )
        let inputs = request.viewURLs
        let export = try InstantMeshAssetExporter.export(
            mesh: mesh,
            inputURLs: inputs,
            checkpoint: checkpoint,
            outputDirectory: request.outputDirectory,
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
        return InstantMeshRunResult(
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

    }

    private static func checkpoint(root: URL) -> InstantMeshCheckpoint {
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

private actor InstantMeshOperationTrace {
    var events: [String] = []
    func record(_ event: String) { events.append(event) }
}
