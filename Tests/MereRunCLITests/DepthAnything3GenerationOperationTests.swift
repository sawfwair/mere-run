import Foundation
import MediaIO
import XCTest
@testable import MereRunCore
@testable import MereRunCLI

final class DepthAnything3GenerationOperationTests: XCTestCase {
    private enum Expected: Error { case failure }

    func testCLIAndAPIShareCameraConditioningAndExportSettings() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for explicit in [false, true] {
            let cameras = try (0..<2).map { try Self.camera(index: $0) }
            let document = try JSONEncoder().encode(DepthAnything3CameraDocument(cameras: cameras))
            let cameraURL = root.appendingPathComponent("cameras.json")
            try document.write(to: cameraURL)
            let fields = explicit ? ["process_resolution": "112", "reference_view": "first",
                                     "confidence_percentile": "65", "max_points": "3",
                                     "cameras": String(decoding: document, as: UTF8.self)] : [:]
            let api = try APIServerContract.multiViewGeometryPlan(from: Self.form(fields))
            let images = ["/input-0.png", "/input-1.png"]
            let args = images + ["--output", "/output", "--model", api.modelID]
                + (explicit ? ["--process-resolution", "112", "--reference-view", "first",
                               "--confidence-percentile", "65", "--max-points", "3", "--cameras", cameraURL.path] : [])
            let cli = try VisionGeometryMultiView.parse(args).makeGenerationRequest()
            XCTAssertEqual(cli, api.request(imageURLs: images.map { URL(fileURLWithPath: $0) },
                                           outputDirectory: URL(fileURLWithPath: "/output")))
        }
    }

    func testInvalidSettingsAndCombinedViewBudgetFailBeforeInputAccess() throws {
        for fields in [["process_resolution": "13"], ["process_resolution": "1009"],
                       ["confidence_percentile": "nan"], ["confidence_percentile": "101"], ["max_points": "0"]] {
            XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(from: Self.form(fields)))
        }
        XCTAssertThrowsError(try DepthAnything3GenerationSettings(processResolution: 1008).validate(viewCount: 16))
        XCTAssertThrowsError(try DepthAnything3GenerationSettings(knownCameras: [Self.camera(index: 0)]).validate(viewCount: 2))
        XCTAssertThrowsError(try APIServerContract.multiViewGeometryPlan(from: Self.form(["model": "/custom/model"])))
    }

    func testCameraDimensionsAreRecheckedAgainstCurrentImages() async throws {
        let original = try Self.request()
        defer { try? FileManager.default.removeItem(at: original.imageURLs[0].deletingLastPathComponent()) }
        let request = DepthAnything3GenerationRequest(
            imageURLs: original.imageURLs, outputDirectory: original.outputDirectory,
            settings: try DepthAnything3GenerationSettings(knownCameras: (0..<2).map { try Self.camera(index: $0) })
        )
        try DepthAnything3GenerationOperation.prepare(request)
        try MediaImageIO.writePNG(try MediaImage(width: 3, height: 2, rgba8: Array(repeating: 255, count: 24)),
                                 to: request.imageURLs[1])
        do {
            _ = try await DepthAnything3GenerationOperation.execute(request, makeRuntime: {
                XCTFail("Camera dimensions must fail before runtime construction")
                return .init(generate: { _, _ in throw Expected.failure }, unload: {})
            })
            XCTFail("Expected camera/image mismatch")
        } catch DepthAnything3PreprocessingError.cameraImageDimensionMismatch(let index, _, _, _, _) {
            XCTAssertEqual(index, 1)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: request.outputDirectory.path))
    }

    func testExecutionRechecksInputBeforeRuntimePreparation() async throws {
        let request = try Self.request()
        defer { try? FileManager.default.removeItem(at: request.imageURLs[0].deletingLastPathComponent()) }
        try DepthAnything3GenerationOperation.prepare(request)
        try Data("invalid replacement".utf8).write(to: request.imageURLs[0])
        do {
            _ = try await DepthAnything3GenerationOperation.execute(request, prepareRuntime: {
                XCTFail("Changed input must fail before runtime preparation")
            }, makeRuntime: {
                XCTFail("Changed input must fail before construction")
                return .init(generate: { _, _ in throw Expected.failure }, unload: {})
            })
            XCTFail("Expected image validation failure")
        } catch { XCTAssertFalse(error is Expected) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: request.outputDirectory.path))
    }

    func testGenerationAndExportFailuresAwaitOneUnloadWithinAdmission() async throws {
        for failureStage in ["none", "generation", "export"] {
            let request = try Self.request()
            defer { try? FileManager.default.removeItem(at: request.imageURLs[0].deletingLastPathComponent()) }
            let trace = DepthAnything3OperationTrace()
            let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
            let runtime = DepthAnything3GenerationOperation.Runtime(generate: { actual, _ in
                XCTAssertEqual(actual, request)
                await trace.record("generate")
                if failureStage == "generation" { throw Expected.failure }
                return try DepthAnything3GenerationOperationTests.fixtureResult(actual, invalidFocal: failureStage == "export")
            }, unload: {
                let state = await admission.snapshot()
                XCTAssertEqual(state.activeRequests, 1)
                await Task.yield()
                await trace.record("unload")
            })
            do {
                let result = try await withVFXRequestAdmission(using: admission) {
                    try await DepthAnything3GenerationOperation.execute(request, makeRuntime: { runtime })
                }
                XCTAssertEqual(failureStage, "none")
                XCTAssertLessThanOrEqual(result.export.manifest.pointCount, request.settings.export.maximumPointCount)
                XCTAssertEqual(result.run.views.count, 2)
                for artifact in result.export.manifest.artifacts {
                    let url = request.outputDirectory.appendingPathComponent(artifact.relativePath)
                    XCTAssertEqual(try ModelArtifactPin.fileSHA256(url), artifact.sha256)
                }
            } catch Expected.failure { XCTAssertEqual(failureStage, "generation") }
            catch is GeometryError { XCTAssertEqual(failureStage, "export") }
            let events = await trace.events
            XCTAssertEqual(events, ["generate", "unload"])
            let state = await admission.snapshot()
            XCTAssertEqual(state.activeRequests, 0)
            XCTAssertEqual(state.totalAdmittedRequests, 1)
            if failureStage != "none" {
                XCTAssertFalse(FileManager.default.fileExists(atPath: request.outputDirectory.path))
            }
        }
    }

    func testCancellationDuringGenerationOrUnloadCannotReturnSuccess() async throws {
        for cancelInUnload in [false, true] {
            let request = try Self.request()
            defer { try? FileManager.default.removeItem(at: request.imageURLs[0].deletingLastPathComponent()) }
            let trace = DepthAnything3OperationTrace()
            let runtime = DepthAnything3GenerationOperation.Runtime(generate: { actual, _ in
                if !cancelInUnload { withUnsafeCurrentTask { $0?.cancel() } }
                return try DepthAnything3GenerationOperationTests.fixtureResult(actual)
            }, unload: {
                if cancelInUnload { withUnsafeCurrentTask { $0?.cancel() } }
                await trace.record("unload")
            })
            let task = Task { @Sendable in
                try await DepthAnything3GenerationOperation.execute(request, makeRuntime: { runtime })
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
        } + (0..<2).map { index in
            .init(name: "image[]", filename: "input-\(index).png", contentType: "image/png", body: Data([1]))
        })
    }

    private static func request() throws -> DepthAnything3GenerationRequest {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("da3-operation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let images = try (0..<2).map { index in
            let url = root.appendingPathComponent("input-\(index).png")
            try MediaImageIO.writePNG(try MediaImage(width: 2, height: 2, rgba8: Array(repeating: 255, count: 16)), to: url)
            return url
        }
        return DepthAnything3GenerationRequest(
            imageURLs: images, outputDirectory: root.appendingPathComponent("output"),
            settings: try DepthAnything3GenerationSettings(maximumPointCount: 3)
        )
    }

    private static func camera(index: Int) throws -> DepthAnything3KnownCamera {
        DepthAnything3KnownCamera(
            intrinsics: GeometryCameraIntrinsics(imageWidth: 2, imageHeight: 2, normalizedFX: 1, normalizedFY: 1),
            extrinsics: try GeometryCameraExtrinsics(rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [Double(index), 0, 0])
        )
    }

    private static func fixtureResult(
        _ request: DepthAnything3GenerationRequest, invalidFocal: Bool = false
    ) throws -> DepthAnything3RunResult {
        let image = try MediaImage(width: 2, height: 2, rgba8: Array(repeating: 255, count: 16))
        let intrinsics = GeometryCameraIntrinsics(
            imageWidth: 2, imageHeight: 2, normalizedFX: invalidFocal ? 0 : 1, normalizedFY: invalidFocal ? 0 : 1
        )
        let views = try request.imageURLs.enumerated().map { index, source in
            let extrinsics = try Self.camera(index: index).extrinsics
            return DepthAnything3ViewResult(
                index: index, sourceURL: source, inputIdentity: try DepthAnything3InputIdentity.capture(source),
                sourceImage: image, processedImage: image,
                preprocessingPlan: DepthAnything3PreprocessingPlan(
                    sourceWidth: 2, sourceHeight: 2, processResolution: 504,
                    boundaryWidth: 504, boundaryHeight: 504, divisibleWidth: 504, divisibleHeight: 504,
                    batchCropLeft: 0, batchCropTop: 0, processedWidth: 2, processedHeight: 2
                ),
                depth: [1, 2, 3, 4], confidence: [1, 1, 1, 1], intrinsics: intrinsics, extrinsics: extrinsics,
                predictedIntrinsics: intrinsics, predictedExtrinsics: extrinsics, suppliedCamera: nil
            )
        }
        return DepthAnything3RunResult(
            views: views, checkpoint: checkpoint(root: request.outputDirectory), referenceViewStrategy: .saddleBalanced,
            cameraSemantics: .predictedRelative, cameraScaleAlignment: "predicted-relative", depthScaleDivisor: 1,
            processResolution: 504, checkpointVerificationSeconds: 0, decodingSeconds: 0,
            preprocessingSeconds: 0, modelLoadSeconds: 0, inferenceSeconds: 0, postprocessingSeconds: 0
        )
    }

    private static func checkpoint(root: URL) -> DepthAnything3Checkpoint {
        DepthAnything3Checkpoint(
            modelID: "vision-geometry-da3-small",
            repository: "depth-anything/DA3-SMALL",
            revision: String(repeating: "a", count: 40),
            sourceRepository: "ByteDance-Seed/Depth-Anything-3",
            sourceRevision: String(repeating: "b", count: 40),
            license: "Apache-2.0",
            rootURL: root,
            weightsURL: root.appendingPathComponent("model.safetensors"),
            configurationURL: root.appendingPathComponent("config.json"),
            weightsByteCount: 137_248_940,
            weightsSHA256: String(repeating: "c", count: 64),
            configurationByteCount: 1_202,
            configurationSHA256: String(repeating: "d", count: 64)
        )
    }

}

private actor DepthAnything3OperationTrace {
    var events: [String] = []
    func record(_ event: String) { events.append(event) }
}
