import Foundation
import MediaIO
import XCTest
@testable import MereRunCore
@testable import MereRunCLI

final class TripoSRGenerationOperationTests: XCTestCase {
    private enum Expected: Error { case failure }

    func testCLIAndAPIShareDefaultAndExplicitSettings() throws {
        for explicit in [false, true] {
            let fields = explicit ? ["resolution": "32", "density_threshold": "18", "foreground_ratio": "0.9",
                                     "already_framed": "true", "vertex_colors": "false"] : [:]
            let api = try APIServerContract.imageTo3DPlan(from: Self.form(fields))
            let cli = try VisionImageTo3D.makeGenerationRequest(
                input: "/input.png", output: "/output", model: api.modelID,
                resolution: explicit ? 32 : TripoSRGenerationSettings.defaultResolution,
                densityThreshold: explicit ? 18 : TripoSRConfiguration.production.densityThreshold,
                foregroundRatio: explicit ? 0.9 : TripoSRGenerationSettings.defaultForegroundRatio,
                alreadyFramed: explicit, noVertexColors: explicit
            )
            XCTAssertEqual(cli, api.request(imageURL: URL(fileURLWithPath: "/input.png"),
                                            outputDirectory: URL(fileURLWithPath: "/output")))
        }
    }

    func testInvalidSettingsFailBeforeInputAccessIncludingUnusedRatio() throws {
        for fields in [["resolution": "1"], ["resolution": "513"], ["density_threshold": "nan"],
                       ["foreground_ratio": "0"], ["foreground_ratio": "inf", "already_framed": "true"]] {
            XCTAssertThrowsError(try APIServerContract.imageTo3DPlan(from: Self.form(fields)))
        }
        XCTAssertThrowsError(try TripoSRGenerationSettings(extractionResolution: Int.max))
        XCTAssertThrowsError(try TripoSRGenerationSettings(densityThreshold: .infinity))
        XCTAssertThrowsError(try TripoSRGenerationSettings(foregroundRatio: .nan, alreadyFramed: true))
        XCTAssertThrowsError(try APIServerContract.imageTo3DPlan(from: Self.form(["model": "/custom/model"])))
    }

    func testExecutionRechecksInputBeforeRuntimePreparation() async throws {
        let request = try Self.request()
        defer { try? FileManager.default.removeItem(at: request.imageURL.deletingLastPathComponent()) }
        _ = try TripoSRGenerationOperation.prepare(request)
        try Data("invalid replacement".utf8).write(to: request.imageURL)
        do {
            _ = try await TripoSRGenerationOperation.execute(request, prepareRuntime: {
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
            defer { try? FileManager.default.removeItem(at: request.imageURL.deletingLastPathComponent()) }
            let trace = TripoSROperationTrace()
            let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
            let runtime = TripoSRGenerationOperation.Runtime(generate: { actual, _ in
                XCTAssertEqual(actual, request)
                await trace.record("generate")
                if fail { throw Expected.failure }
                return try TripoSRGenerationOperationTests.fixtureResult(actual)
            }, unload: {
                let state = await admission.snapshot()
                XCTAssertEqual(state.activeRequests, 1)
                await Task.yield()
                await trace.record("unload")
            })
            do {
                let result = try await withVFXRequestAdmission(using: admission) {
                    try await TripoSRGenerationOperation.execute(request, makeRuntime: { runtime })
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
            defer { try? FileManager.default.removeItem(at: request.imageURL.deletingLastPathComponent()) }
            let trace = TripoSROperationTrace()
            let runtime = TripoSRGenerationOperation.Runtime(generate: { actual, _ in
                if !cancelInUnload { withUnsafeCurrentTask { $0?.cancel() } }
                return try TripoSRGenerationOperationTests.fixtureResult(actual)
            }, unload: {
                if cancelInUnload { withUnsafeCurrentTask { $0?.cancel() } }
                await trace.record("unload")
            })
            let task = Task { @Sendable in
                try await TripoSRGenerationOperation.execute(request, makeRuntime: { runtime })
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
        } + [.init(name: "image", filename: "input.png", contentType: "image/png", body: Data([1]))])
    }

    private static func request() throws -> TripoSRGenerationRequest {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("triposr-operation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let image = root.appendingPathComponent("input.png")
        try MediaImageIO.writePNG(try MediaImage(width: 2, height: 2, rgba8: Array(repeating: 255, count: 16)), to: image)
        return TripoSRGenerationRequest(imageURL: image, outputDirectory: root.appendingPathComponent("output"),
                                       settings: try TripoSRGenerationSettings())
    }

    private static func fixtureResult(_ request: TripoSRGenerationRequest) throws -> TripoSRRunResult {
        let mesh = try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            normals: [0, 0, 1, 0, 0, 1, 0, 0, 1],
            colorsRGBA8: [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255],
            inferredUnseenGeometry: true
        )
        let inputURL = request.imageURL
        let checkpoint = checkpoint()
        let export = try TripoSRAssetExporter.export(
            mesh: mesh,
            inputURL: inputURL,
            checkpoint: checkpoint,
            outputDirectory: request.outputDirectory,
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
        return TripoSRRunResult(
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

    }

    private static func checkpoint() -> TripoSRCheckpoint {
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

private actor TripoSROperationTrace {
    var events: [String] = []
    func record(_ event: String) { events.append(event) }
}
