import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class VisionServeCommandTests: XCTestCase {
    func testParsesLoopbackServerOptions() throws {
        let command = try VisionServe.parse([
            "--host", "127.0.0.1",
            "--port", "9091",
            "--model", "vision-ground-falcon-perception",
            "--max-frame-bytes", "4194304",
        ])

        XCTAssertEqual(command.host, "127.0.0.1")
        XCTAssertEqual(command.port, 9_091)
        XCTAssertEqual(command.model, "vision-ground-falcon-perception")
        XCTAssertEqual(command.maxFrameBytes, 4_194_304)
    }

    func testRejectsNonLoopbackServerWithoutAPIKey() {
        XCTAssertThrowsError(try VisionServe.parse(["--host", "0.0.0.0"]))
    }

    func testAcceptsNonLoopbackServerWithAPIKey() throws {
        let command = try VisionServe.parse([
            "--host", "0.0.0.0",
            "--api-key", "test-secret",
        ])
        XCTAssertEqual(command.apiKey, "test-secret")
    }

    func testBuildsRepeatedQueryRequestPlan() throws {
        let form = MultipartFormData(parts: [
            .init(name: "query", filename: nil, contentType: nil, body: Data("car".utf8)),
            .init(name: "query[]", filename: nil, contentType: nil, body: Data("person".utf8)),
            .init(name: "stream_id", filename: nil, contentType: nil, body: Data("camera-7".utf8)),
            .init(name: "frame_id", filename: nil, contentType: nil, body: Data("frame-42".utf8)),
            .init(name: "max_new_tokens", filename: nil, contentType: nil, body: Data("256".utf8)),
            .init(name: "segmentation_threshold", filename: nil, contentType: nil, body: Data("0.35".utf8)),
        ])

        let plan = try VisionGroundingServer.requestPlan(from: form)
        XCTAssertEqual(plan.queries, ["car", "person"])
        XCTAssertEqual(plan.streamID, "camera-7")
        XCTAssertEqual(plan.frameID, "frame-42")
        XCTAssertEqual(plan.maxNewTokens, 256)
        XCTAssertEqual(plan.segmentationThreshold, 0.35, accuracy: 0.0001)
    }

    func testRequiresAtLeastOneQuery() {
        let form = MultipartFormData(parts: [])
        XCTAssertThrowsError(try VisionGroundingServer.requestPlan(from: form))
    }

    func testResponseDoesNotExposeServerLocalPaths() throws {
        let detection = FalconPerceptionDetection(
            label: "vehicle",
            xy: .init(x: 0.5, y: 0.4),
            hw: .init(h: 0.2, w: 0.3),
            box: .init(x1: 0.35, y1: 0.3, x2: 0.65, y2: 0.5),
            score: 0.88,
            maskPath: "/private/tmp/secret-mask.png"
        )
        let metadata = FalconPerceptionGroundingMetadata(
            schemaVersion: 1,
            modelID: "vision-ground-falcon-perception",
            inputImagePath: "/private/tmp/frame.jpg",
            annotatedImagePath: "/private/tmp/annotated.png",
            jsonOutputPath: "/private/tmp/detections.json",
            queries: ["vehicle"],
            detections: [detection]
        )
        let run = FalconPerceptionGroundingRun(
            modelID: metadata.modelID,
            annotatedImageURL: URL(fileURLWithPath: metadata.annotatedImagePath),
            jsonOutputURL: URL(fileURLWithPath: metadata.jsonOutputPath),
            detections: [detection],
            metadata: metadata
        )
        let response = VisionGroundingServer.response(
            from: run,
            plan: .init(
                queries: ["vehicle"],
                streamID: "stream-1",
                frameID: "frame-1",
                capturedAt: "2026-08-10T00:00:00Z",
                maxNewTokens: 512,
                segmentationThreshold: 0.5
            ),
            imageSHA256: String(repeating: "a", count: 64),
            startedAt: Date(timeIntervalSince1970: 10),
            inferenceStartedAt: Date(timeIntervalSince1970: 11),
            finishedAt: Date(timeIntervalSince1970: 13)
        )

        XCTAssertEqual(response.detections.count, 1)
        XCTAssertEqual(response.detections[0].label, "vehicle")
        XCTAssertEqual(response.timing.inferenceSeconds, 2)
        XCTAssertEqual(response.timing.totalSeconds, 3)
        let encoded = String(decoding: try JSONEncoder().encode(response), as: UTF8.self)
        XCTAssertFalse(encoded.contains("/private/tmp"))
        XCTAssertFalse(encoded.contains("maskPath"))
    }
}
