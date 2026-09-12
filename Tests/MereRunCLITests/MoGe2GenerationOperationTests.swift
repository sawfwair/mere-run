import Foundation
import MediaIO
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

private actor GeometryOperationTrace {
    var events: [String] = []
    func record(_ event: String) { events.append(event) }
}

final class MoGe2GenerationOperationTests: XCTestCase {
    func testCLIAndAPIUseTheSameDefaultsAndExplicitSettings() throws {
        let input = URL(fileURLWithPath: "/tmp/geometry-input.png")
        let output = URL(fileURLWithPath: "/tmp/geometry-output")
        for fields in [[:], ["resolution_level": "3", "token_count": "256", "max_points": "1000"]] {
            var arguments = [input.path, "--output", output.path]
            for (field, flag) in [("resolution_level", "--resolution-level"), ("token_count", "--token-count"), ("max_points", "--max-points")] {
                if let value = fields[field] { arguments += [flag, value] }
            }
            let cli = try VisionGeometry.parse(arguments).makeGenerationRequest()
            let api = try APIServerContract.geometryPlan(from: Self.form(fields))
                .request(imageURL: input, outputDirectory: output)
            XCTAssertEqual(cli, api)
            XCTAssertNil(api.model)
        }
        let defaults = try MoGe2GenerationSettings()
        XCTAssertEqual(defaults.configuration.effectiveTokenCount, 3_600)
        XCTAssertNil(defaults.configuration.maximumPointCount)
    }

    func testBothAdaptersRejectInvalidSettingsWithoutCompatibilityClamping() throws {
        for (field, flag, values) in [
            ("resolution_level", "--resolution-level", [-1, 10, Int.max]),
            ("token_count", "--token-count", [0, -1, 3_601, Int.max]),
            ("max_points", "--max-points", [0, -1]),
        ] {
            for value in values {
                let cli = try VisionGeometry.parse(["/missing/input.png", "\(flag)=\(value)"])
                XCTAssertThrowsError(try cli.makeGenerationRequest()) { XCTAssertTrue($0 is MoGe2GenerationError) }
                XCTAssertThrowsError(try APIServerContract.geometryPlan(from: Self.form([field: String(value)]))) { error in
                    guard case APIRequestValidationError.invalidField(let actual, _) = error else {
                        return XCTFail("Expected a field validation error, got \(error)")
                    }
                    XCTAssertEqual(actual, field)
                }
            }
        }
        XCTAssertEqual(APIVFXClientErrorPolicy.status(for: MoGe2GenerationError.inputNotFound("missing")), .badRequest)
    }

    func testModelSelectionKeepsTransportPolicy() throws {
        let cli = try VisionGeometry.parse(["input.png", "--model", "/custom/model.onnx"]).makeGenerationRequest()
        XCTAssertEqual(cli.model, "/custom/model.onnx")
        XCTAssertThrowsError(try APIServerContract.geometryPlan(from: Self.form(["model": "/custom/model.onnx"])))
        XCTAssertThrowsError(try APIServerContract.geometryPlan(from: Self.form(["model": "VISION-GEOMETRY-MOGE2-SMALL"])))
        let api = try APIServerContract.geometryPlan(from: Self.form(["model": " vision-geometry-moge2-small "]))
        XCTAssertEqual(api.modelID, MoGe2GenerationRequest.defaultModelID)
    }

    func testExecutionRechecksChangedDimensionsBeforeRuntimeSetup() async throws {
        let root = try Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let request = try Self.request(root)
        let plan = try MoGe2GenerationOperation.prepare(request)
        XCTAssertEqual(plan.tokenGrid.count, 3_600)
        XCTAssertFalse(FileManager.default.fileExists(atPath: request.outputDirectory.path))
        try Self.writeImage(request.imageURL, width: 2_000, height: 1_080)
        do {
            _ = try await MoGe2GenerationOperation.execute(request, prepareRuntime: {
                XCTFail("The changed input must be rejected before runtime setup")
            }, makeRuntime: Self.unexpectedRuntime)
            XCTFail("Expected the rounded patch grid to exceed the workload limit")
        } catch MoGe2TokenGridError.tokenGridExceedsLimit(let rows, let columns, let actual, let maximum) {
            XCTAssertEqual(rows * columns, 3_608)
            XCTAssertEqual(actual, 3_608)
            XCTAssertEqual(maximum, 3_600)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: request.outputDirectory.path))
    }

    func testDeletedInputIsRejectedAfterPlanning() async throws {
        let root = try Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let request = try Self.request(root)
        _ = try MoGe2GenerationOperation.prepare(request)
        try FileManager.default.removeItem(at: request.imageURL)
        do {
            _ = try await MoGe2GenerationOperation.execute(request, makeRuntime: Self.unexpectedRuntime)
            XCTFail("Expected a missing input error")
        } catch MoGe2GenerationError.inputNotFound(let path) {
            XCTAssertEqual(path, request.imageURL.path)
        }
    }

    func testSuccessAwaitsUnloadInsideOneAdmissionAndPreservesArtifactProof() async throws {
        let root = try Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let request = try Self.request(root)
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let trace = GeometryOperationTrace()
        let result = try await withVFXRequestAdmission(using: admission) {
            try await MoGe2GenerationOperation.execute(request, makeRuntime: {
                MoGe2GenerationOperation.Runtime(generate: { actual, progress in
                    XCTAssertEqual(actual, request)
                    XCTAssertNil(progress)
                    let state = await admission.snapshot()
                    XCTAssertEqual(state.activeRequests, 1)
                    XCTAssertEqual(state.totalAdmittedRequests, 1)
                    await trace.record("generated")
                    return try Self.fixtureResult(actual)
                }, unload: {
                    let state = await admission.snapshot()
                    XCTAssertEqual(state.activeRequests, 1)
                    await trace.record("unloaded")
                })
            })
        }
        let events = await trace.events
        XCTAssertEqual(events, ["generated", "unloaded"])
        let response = try APIServerContract.geometryResponse(from: result)
        for artifact in response.artifacts {
            let url = try XCTUnwrap(URL(string: artifact.url))
            XCTAssertEqual(artifact.sha256, try ModelArtifactPin.fileSHA256(url))
            XCTAssertEqual(artifact.byteCount, try ModelArtifactPin.fileByteCount(url))
        }
        let state = await admission.snapshot()
        XCTAssertEqual(state.activeRequests, 0)
        XCTAssertEqual(state.totalAdmittedRequests, 1)
    }

    func testFailureAwaitsOneUnloadAndReleasesAdmission() async throws {
        let root = try Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let request = try Self.request(root)
        let trace = GeometryOperationTrace()
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        do {
            _ = try await withVFXRequestAdmission(using: admission) {
                try await MoGe2GenerationOperation.execute(request, makeRuntime: {
                    MoGe2GenerationOperation.Runtime(generate: { _, _ in
                        await trace.record("failed")
                        throw FixtureError.expected
                    }, unload: { await trace.record("unloaded") })
                })
            }
            XCTFail("Expected the runtime failure")
        } catch FixtureError.expected {}
        let events = await trace.events
        XCTAssertEqual(events, ["failed", "unloaded"])
        let state = await admission.snapshot()
        XCTAssertEqual(state.activeRequests, 0)
    }

    func testCancellationDuringGenerationCannotReturnSuccess() async throws {
        try await assertCancellation(cancelDuringUnload: false)
    }

    func testCancellationDuringUnloadCannotReturnSuccess() async throws {
        try await assertCancellation(cancelDuringUnload: true)
    }

    private func assertCancellation(cancelDuringUnload: Bool) async throws {
        let root = try Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let request = try Self.request(root)
        let trace = GeometryOperationTrace()
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let runtime = MoGe2GenerationOperation.Runtime(generate: { actual, _ in
            let result = try MoGe2GenerationOperationTests.fixtureResult(actual)
            if !cancelDuringUnload { withUnsafeCurrentTask { $0?.cancel() } }
            return result
        }, unload: {
            if cancelDuringUnload { withUnsafeCurrentTask { $0?.cancel() } }
            await trace.record("unloaded")
        })
        let task = Task { @Sendable in
            try await withVFXRequestAdmission(using: admission) {
                try await MoGe2GenerationOperation.execute(request, makeRuntime: { runtime })
            }
        }
        do {
            _ = try await task.value
            XCTFail("Cancellation must win over the runtime result")
        } catch is CancellationError {}
        let events = await trace.events
        XCTAssertEqual(events, ["unloaded"])
        let state = await admission.snapshot()
        XCTAssertEqual(state.activeRequests, 0)
        XCTAssertEqual(state.totalCancelledRequests, 1)
    }

    private enum FixtureError: Error { case expected }

    private static func unexpectedRuntime() -> MoGe2GenerationOperation.Runtime {
        XCTFail("Invalid input must prevent runtime construction")
        return .init(generate: { _, _ in throw FixtureError.expected }, unload: {})
    }

    private static func form(_ fields: [String: String]) -> MultipartFormData {
        MultipartFormData(parts: fields.map { .init(name: $0.key, filename: nil, contentType: nil, body: Data($0.value.utf8)) })
    }

    private static func makeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("geometry-operation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private static func writeImage(_ url: URL, width: Int = 2, height: Int = 2) throws {
        try MediaImageIO.writePNG(try MediaImage(width: width, height: height, rgba8: [UInt8](repeating: 255, count: width * height * 4)), to: url)
    }

    private static func request(_ root: URL) throws -> MoGe2GenerationRequest {
        let image = root.appendingPathComponent("input.png")
        try writeImage(image)
        return MoGe2GenerationRequest(
            imageURL: image, outputDirectory: root.appendingPathComponent("output"), settings: try MoGe2GenerationSettings()
        )
    }

    private static func fixtureResult(_ request: MoGe2GenerationRequest) throws -> MoGe2RunResult {
        let intrinsics = GeometryCameraIntrinsics(imageWidth: 2, imageHeight: 2, normalizedFX: 1, normalizedFY: 1)
        let frame = try DenseGeometryFrame(
            width: 2, height: 2, units: .meters, intrinsics: intrinsics,
            depth: [1, 2, 3, 4], points: [-0.25, -0.25, 1, 0.5, -0.5, 2, -0.75, 0.75, 3, 1, 1, 4],
            normals: [Float](repeating: 0, count: 12), validity: [1, 1, 1, 1], confidence: [1, 1, 1, 1]
        )
        let export = try GeometryArtifactExporter.export(
            frame: frame, inputURL: request.imageURL, outputDirectory: request.outputDirectory,
            provenance: GeometryModelProvenance(
                modelID: MoGe2GenerationRequest.defaultModelID, upstreamRepository: "fixture", upstreamRevision: "fixture", license: "MIT"
            ), createdAt: Date(timeIntervalSince1970: 0)
        )
        return MoGe2RunResult(
            export: export, focalShift: MoGe2FocalShiftSolution(focal: 1.5, shift: 0.1, iterationCount: 3, residualMeanSquare: 0),
            metricScale: 2, tokenCount: request.settings.configuration.effectiveTokenCount,
            modelLoadSeconds: 0, inferenceSeconds: 0, postprocessSeconds: 0
        )
    }
}
