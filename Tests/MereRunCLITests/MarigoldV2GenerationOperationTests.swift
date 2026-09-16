import Foundation
import MediaIO
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

private actor DepthOperationTrace {
    var events: [String] = []
    func record(_ event: String) { events.append(event) }
}

final class MarigoldV2GenerationOperationTests: XCTestCase {
    func testSettingsRejectInvalidValuesBeforeConfiguration() throws {
        XCTAssertThrowsError(try MarigoldV2GenerationSettings(maximumEdge: 512, nativeResolution: true)) {
            XCTAssertEqual($0 as? MarigoldV2GenerationError, .nativeResolutionConflictsWithMaximumEdge)
        }
        for edge in [0, 8, 15, -1] {
            XCTAssertThrowsError(try MarigoldV2GenerationSettings(maximumEdge: edge)) {
                XCTAssertEqual($0 as? MarigoldV2GenerationError, .maximumEdgeBelowAlignment(edge))
            }
        }
        XCTAssertEqual(try MarigoldV2GenerationSettings(maximumEdge: 16).configuration.maximumEdge, 16)
        XCTAssertEqual(
            try MarigoldV2GenerationSettings().configuration,
            MarigoldV2InferenceConfiguration()
        )
    }

    func testPlanRejectsOversizedInputsBeforeExecution() throws {
        let request = MarigoldV2GenerationRequest(
            imageURL: URL(fileURLWithPath: "/tmp/huge.png"),
            outputDirectory: URL(fileURLWithPath: "/tmp/huge-depth"),
            settings: try MarigoldV2GenerationSettings()
        )
        XCTAssertThrowsError(try MarigoldV2GenerationPlan(
            request: request, imageWidth: 16_385, imageHeight: 16, managedModelInstalled: false
        ))
    }

    func testDeletedInputIsRejectedAfterPlanning() async throws {
        let root = try Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let request = try Self.request(root)
        let plan = try MarigoldV2GenerationOperation.prepare(request)
        XCTAssertEqual(plan.inferenceWidth, 16)
        XCTAssertEqual(plan.inferenceHeight, 16)
        XCTAssertFalse(FileManager.default.fileExists(atPath: request.outputDirectory.path))
        try FileManager.default.removeItem(at: request.imageURL)
        do {
            _ = try await MarigoldV2GenerationOperation.execute(request, prepareRuntime: {
                XCTFail("The missing input must be rejected before runtime setup")
            }, makeRuntime: Self.unexpectedRuntime)
            XCTFail("Expected a missing input error")
        } catch MarigoldV2GenerationError.inputNotFound(let path) {
            XCTAssertEqual(path, request.imageURL.path)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: request.outputDirectory.path))
    }

    func testSuccessAwaitsUnloadInsideOneAdmission() async throws {
        let root = try Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let request = try Self.request(root)
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let trace = DepthOperationTrace()
        let result = try await withVFXRequestAdmission(using: admission) {
            try await MarigoldV2GenerationOperation.execute(request, makeRuntime: {
                MarigoldV2GenerationOperation.Runtime(generate: { actual, progress in
                    XCTAssertEqual(actual, request)
                    XCTAssertNil(progress)
                    let state = await admission.snapshot()
                    XCTAssertEqual(state.activeRequests, 1)
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
        for url in [result.export.depthURL, result.export.previewURL, result.export.manifestURL] {
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), url.path)
        }
        XCTAssertEqual(result.export.manifest.artifacts.map(\.kind), [.depthEXR, .depthPreview])
        let state = await admission.snapshot()
        XCTAssertEqual(state.activeRequests, 0)
        XCTAssertEqual(state.totalAdmittedRequests, 1)
    }

    func testFailureAwaitsOneUnloadAndReleasesAdmission() async throws {
        let root = try Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let request = try Self.request(root)
        let trace = DepthOperationTrace()
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        do {
            _ = try await withVFXRequestAdmission(using: admission) {
                try await MarigoldV2GenerationOperation.execute(request, makeRuntime: {
                    MarigoldV2GenerationOperation.Runtime(generate: { _, _ in
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
        let trace = DepthOperationTrace()
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let runtime = MarigoldV2GenerationOperation.Runtime(generate: { actual, _ in
            let result = try MarigoldV2GenerationOperationTests.fixtureResult(actual)
            if !cancelDuringUnload { withUnsafeCurrentTask { $0?.cancel() } }
            return result
        }, unload: {
            if cancelDuringUnload { withUnsafeCurrentTask { $0?.cancel() } }
            await trace.record("unloaded")
        })
        let task = Task { @Sendable in
            try await withVFXRequestAdmission(using: admission) {
                try await MarigoldV2GenerationOperation.execute(request, makeRuntime: { runtime })
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

    private static func unexpectedRuntime() -> MarigoldV2GenerationOperation.Runtime {
        XCTFail("Invalid input must prevent runtime construction")
        return .init(generate: { _, _ in throw FixtureError.expected }, unload: {})
    }

    private static func makeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("depth-operation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private static func writeImage(_ url: URL, width: Int = 2, height: Int = 2) throws {
        try MediaImageIO.writePNG(
            try MediaImage(width: width, height: height, rgba8: [UInt8](repeating: 255, count: width * height * 4)),
            to: url
        )
    }

    private static func request(_ root: URL) throws -> MarigoldV2GenerationRequest {
        let image = root.appendingPathComponent("input.png")
        try writeImage(image)
        return MarigoldV2GenerationRequest(
            imageURL: image,
            outputDirectory: root.appendingPathComponent("output"),
            settings: try MarigoldV2GenerationSettings()
        )
    }

    private static func fixtureResult(_ request: MarigoldV2GenerationRequest) throws -> MarigoldV2RunResult {
        let checkpoint = request.settings.configuration.checkpoint
        let export = try MarigoldV2DepthArtifactExporter.export(
            depth: [0.25, 0.5, 0.75, 1],
            width: 2,
            height: 2,
            inferenceWidth: 16,
            inferenceHeight: 16,
            statistics: MarigoldV2DepthStatistics(
                rawMinimum: 0, rawMaximum: 1, normalizationNear: 0, normalizationFar: 1, normalizedFloor: 0.25
            ),
            checkpoint: checkpoint,
            inputURL: request.imageURL,
            outputDirectory: request.outputDirectory,
            provenance: GeometryModelProvenance(
                modelID: MarigoldV2GenerationRequest.defaultModelID,
                upstreamRepository: "fixture", upstreamRevision: "fixture", license: "Apache-2.0"
            ),
            createdAt: Date(timeIntervalSince1970: 0)
        )
        return MarigoldV2RunResult(
            export: export, checkpoint: checkpoint, inferenceWidth: 16, inferenceHeight: 16,
            promptTokenCount: 1, adapterPairCount: 1, vaeDecoderTensorCount: 0,
            modelLoadSeconds: 0, inferenceSeconds: 0, postprocessSeconds: 0
        )
    }
}
