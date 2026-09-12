import Foundation
import XCTest
@testable import MereRunCore
@testable import MereRunCLI

final class VideoDepthAnythingGenerationOperationTests: XCTestCase {
    private enum Expected: Error { case failure }

    func testCLIAndAPIShareDefaultsAndMetricModelPolicy() throws {
        for explicit in [false, true] {
            let model = explicit ? "vision-depth-vda-small-metric" : "vision-depth-vda-small"
            let fields = explicit ? ["model": model, "input_size": "112", "max_frames": "2"] : [:]
            let api = try APIServerContract.depthVideoPlan(from: Self.form(fields))
            let args = ["/input.mp4", "--output", "/output", "--model", model]
                + (explicit ? ["--input-size", "112", "--max-frames", "2"] : [])
            let cli = try VisionDepthVideo.parse(args).makeGenerationRequest()
            XCTAssertEqual(cli, api.request(videoURL: URL(fileURLWithPath: "/input.mp4"),
                                           outputDirectory: URL(fileURLWithPath: "/output")))
        }
        XCTAssertEqual(try VideoDepthAnythingGenerationSettings().maximumFrameCount, 240)
        XCTAssertThrowsError(try APIServerContract.depthVideoPlan(from: Self.form(["model": "/custom/model"])))
        let custom = try VisionDepthVideo.parse(["input.mp4", "--model", "/custom/model"]).makeGenerationRequest()
        XCTAssertEqual(custom.model, "/custom/model")
    }

    func testInvalidLimitsFailBeforeInputAccess() throws {
        for fields in [["input_size": "13"], ["input_size": "1009"], ["max_frames": "0"], ["max_frames": "2401"]] {
            XCTAssertThrowsError(try APIServerContract.depthVideoPlan(from: Self.form(fields)))
        }
        XCTAssertThrowsError(try VideoDepthAnythingGenerationSettings(inputSize: Int.max))
        XCTAssertThrowsError(try VideoDepthAnythingGenerationSettings(maximumFrameCount: Int.max))
    }

    func testExecutionRechecksDeletedOrOversizedFileBeforeRuntimePreparation() async throws {
        for oversized in [false, true] {
            let request = try Self.request()
            defer { try? FileManager.default.removeItem(at: request.videoURL.deletingLastPathComponent()) }
            try VideoDepthAnythingGenerationOperation.prepare(request)
            if oversized {
                let handle = try FileHandle(forWritingTo: request.videoURL)
                try handle.truncate(atOffset: UInt64(VideoDepthAnythingLimits.maximumEncodedVideoBytes) + 1)
                try handle.close()
            } else {
                try FileManager.default.removeItem(at: request.videoURL)
            }
            do {
                _ = try await VideoDepthAnythingGenerationOperation.execute(request, prepareRuntime: {
                    XCTFail("Changed file must fail before runtime preparation")
                }, makeRuntime: {
                    XCTFail("Changed file must fail before construction")
                    return .init(generate: { _, _ in throw Expected.failure }, unload: {})
                })
                XCTFail("Expected file validation failure")
            } catch {
                if oversized { XCTAssertTrue(error is VFXImageInputValidationError) }
                else { XCTAssertTrue(error is VideoDepthAnythingGeneratorError) }
            }
            XCTAssertFalse(FileManager.default.fileExists(atPath: request.outputDirectory.path))
        }
    }

    func testSuccessAndFailureAwaitOneUnloadWithinAdmission() async throws {
        for fail in [false, true] {
            let request = try Self.request()
            defer { try? FileManager.default.removeItem(at: request.videoURL.deletingLastPathComponent()) }
            let trace = VideoDepthAnythingOperationTrace()
            let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
            let runtime = VideoDepthAnythingGenerationOperation.Runtime(generate: { actual, _ in
                XCTAssertEqual(actual, request)
                await trace.record("generate")
                if fail { throw Expected.failure }
                return try VideoDepthAnythingGenerationOperationTests.fixtureResult(actual)
            }, unload: {
                let state = await admission.snapshot()
                XCTAssertEqual(state.activeRequests, 1)
                await Task.yield()
                await trace.record("unload")
            })
            do {
                let result = try await withVFXRequestAdmission(using: admission) {
                    try await VideoDepthAnythingGenerationOperation.execute(request, makeRuntime: { runtime })
                }
                XCTAssertFalse(fail)
                for artifact in result.export.manifest.frames.flatMap(\.artifacts) {
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
            defer { try? FileManager.default.removeItem(at: request.videoURL.deletingLastPathComponent()) }
            let trace = VideoDepthAnythingOperationTrace()
            let runtime = VideoDepthAnythingGenerationOperation.Runtime(generate: { actual, _ in
                if !cancelInUnload { withUnsafeCurrentTask { $0?.cancel() } }
                return try VideoDepthAnythingGenerationOperationTests.fixtureResult(actual)
            }, unload: {
                if cancelInUnload { withUnsafeCurrentTask { $0?.cancel() } }
                await trace.record("unload")
            })
            let task = Task { @Sendable in
                try await VideoDepthAnythingGenerationOperation.execute(request, makeRuntime: { runtime })
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
        } + [.init(name: "video", filename: "input.mp4", contentType: "video/mp4", body: Data([1]))])
    }

    private static func request() throws -> VideoDepthAnythingGenerationRequest {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("video-depth-operation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let video = root.appendingPathComponent("input.mp4")
        // The injected runtime isolates lifecycle tests from native media decoding.
        try Data("video fixture identity".utf8).write(to: video)
        return VideoDepthAnythingGenerationRequest(
            videoURL: video, outputDirectory: root.appendingPathComponent("output"),
            settings: try VideoDepthAnythingGenerationSettings()
        )
    }

    private static func fixtureResult(_ request: VideoDepthAnythingGenerationRequest) throws -> VideoDepthAnythingRunResult {
        let checkpoint = VideoDepthAnythingCheckpoint(
            variant: .relative, format: .pinnedPyTorch, weightsURL: URL(fileURLWithPath: "/fixture/relative.pth"),
            weightsByteCount: 1, weightsSHA256: String(repeating: "a", count: 64),
            sourceSHA256: String(repeating: "a", count: 64)
        )
        let export = try DepthSequenceArtifactExporter.export(
            frames: [try DepthSequenceFrame(index: 0, timeSeconds: 0, width: 2, height: 2, depth: [1, 2, 3, 4])],
            inputURL: request.videoURL, outputDirectory: request.outputDirectory,
            fps: 24, semantics: .affineRelative,
            provenance: GeometryModelProvenance(
                modelID: checkpoint.variant.modelID, upstreamRepository: "fixture",
                upstreamRevision: "fixture", license: "Apache-2.0"
            )
        )
        let reviewURL = request.outputDirectory.appendingPathComponent("depth-review.mp4")
        try Data("review fixture".utf8).write(to: reviewURL)
        let review = VideoDepthReviewArtifact(
            relativePath: reviewURL.lastPathComponent, byteCount: try ModelArtifactPin.fileByteCount(reviewURL),
            sha256: try ModelArtifactPin.fileSHA256(reviewURL)
        )
        return VideoDepthAnythingRunResult(
            export: export, reviewVideo: review, checkpoint: checkpoint, sourceFPS: 24, windowCount: 1,
            checkpointVerificationSeconds: 0, frameExtractionSeconds: 0, modelLoadSeconds: 0,
            inferenceSeconds: 0, exportSeconds: 0
        )
    }
}

private actor VideoDepthAnythingOperationTrace {
    var events: [String] = []
    func record(_ event: String) { events.append(event) }
}
