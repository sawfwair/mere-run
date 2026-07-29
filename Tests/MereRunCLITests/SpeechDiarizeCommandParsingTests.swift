import Foundation
import MereRunCore
import XCTest

@testable import MereRunCLI

final class SpeechDiarizeCommandParsingTests: XCTestCase {
    func testParsesDiarizationOptions() throws {
        let command = try SpeechDiarize.parse([
            "/tmp/meeting.wav",
            "--model", ModelResolver.ModelID.sortformerDiarization.rawValue,
            "--format", "rttm",
            "--output", "/tmp/meeting.rttm",
            "--threshold", "0.42",
            "--min-duration", "0.3",
            "--merge-gap", "0.4",
            "--quiet",
        ])

        XCTAssertEqual(command.audio, "/tmp/meeting.wav")
        XCTAssertEqual(command.model, ModelResolver.ModelID.sortformerDiarization.rawValue)
        XCTAssertEqual(command.format, .rttm)
        XCTAssertEqual(command.output, "/tmp/meeting.rttm")
        XCTAssertEqual(command.threshold, 0.42)
        XCTAssertEqual(command.minDuration, 0.3)
        XCTAssertEqual(command.mergeGap, 0.4)
        XCTAssertTrue(command.quiet)
    }

    func testRejectsOutOfRangeOptions() {
        XCTAssertThrowsError(try SpeechDiarize.parse(["/tmp/meeting.wav", "--threshold", "1.1"]))
        XCTAssertThrowsError(try SpeechDiarize.parse(["/tmp/meeting.wav", "--min-duration", "-0.1"]))
        XCTAssertThrowsError(try SpeechDiarize.parse(["/tmp/meeting.wav", "--merge-gap", "-0.1"]))
    }

    func testResolvesLocalModelDirectory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertEqual(try SpeechDiarize.resolveModelRoot(root.path), root.standardizedFileURL)
    }

    func testDiarizationPayloadMakesRuntimeDeviceExplicit() throws {
        let payload = SpeechDiarizationPayload(
            schemaVersion: 1,
            model: ModelResolver.ModelID.sortformerDiarization.rawValue,
            source: "meeting.wav",
            runtime: "native MLX (default device: gpu)",
            device: "gpu",
            durationSeconds: 12,
            speakerCount: 2,
            processingSeconds: 0.5,
            segments: [
                SpeechDiarizationSegmentPayload(
                    speaker: "speaker_0",
                    speakerIndex: 0,
                    startSeconds: 0,
                    endSeconds: 4,
                    durationSeconds: 4
                )
            ]
        )

        let encoded = try JSONEncoder().encode(payload)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(object["runtime"] as? String, "native MLX (default device: gpu)")
        XCTAssertEqual(object["device"] as? String, "gpu")
    }
}
