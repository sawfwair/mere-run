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
        XCTAssertThrowsError(try SpeechDiarize.parse(["/tmp/meeting.wav", "--min-duration", "inf"]))
    }

    func testParsesNemotron3LatencyAndRecognizesLocalArchive() throws {
        let command = try SpeechDiarize.parse([
            "/tmp/meeting.wav", "--model", ModelResolver.ModelID.nemotron3Diarization.rawValue,
            "--latency", "0.64",
        ])
        XCTAssertEqual(command.latency, .low)
        XCTAssertEqual(command.latency.configuration.chunk, 6)

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertFalse(SpeechDiarize.isNemotron3(model: root.path, root: root))
        try Data().write(to: root.appendingPathComponent("Nemotron-3-Diarization.nemo"))
        XCTAssertTrue(SpeechDiarize.isNemotron3(model: root.path, root: root))
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

    func testLiveDiarizationParsesPCMAndRejectsOfflineBuffer() throws {
        let command = try SpeechDiarizeLive.parse([
            "--stdin", "--latency", "0.64", "--threshold", "0.42", "--quiet",
        ])
        XCTAssertTrue(command.stdin)
        XCTAssertEqual(command.latency, .low)
        XCTAssertEqual(command.threshold, 0.42)
        XCTAssertTrue(command.quiet)
        XCTAssertThrowsError(try SpeechDiarizeLive.parse(["--latency", "offline"]))
        XCTAssertThrowsError(try SpeechDiarizeLive.parse(["--stdin", "--device", "input-1"]))
        XCTAssertThrowsError(try SpeechDiarizeLive.parse(["--threshold", "nan"]))
    }
}
