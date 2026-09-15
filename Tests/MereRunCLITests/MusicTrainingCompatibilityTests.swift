import ArgumentParser
import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class MusicTrainingCompatibilityTests: XCTestCase {
    func testDefaults() throws {
        let command = try MusicTrainAdapter.parse(["--dataset", "data.json", "--output", "out.safetensors"])
        XCTAssertEqual(command.model, ModelResolver.ModelID.aceStep.rawValue)
        XCTAssertEqual(command.kind, .lora)
        XCTAssertEqual(command.rank, 8)
        XCTAssertEqual(command.alpha, 16)
        XCTAssertEqual(command.factor, -1)
        XCTAssertEqual(command.steps, 1_000)
        XCTAssertEqual(command.learningRate, 1e-4)
        XCTAssertEqual(command.weightDecay, 1e-4)
        XCTAssertEqual(command.seed, 42)
        XCTAssertEqual(command.maxDurationSeconds, 30)
        XCTAssertEqual(command.logEvery, 10)
        XCTAssertEqual(command.decoderSubdirectory, "acestep-v15-turbo")
        XCTAssertEqual(command.vaeSubdirectory, "vae")
        XCTAssertNil(command.textSubdirectory)
        XCTAssertNil(command.checkpointsRoot)
    }

    func testInvalidOptionsFailBeforeOpeningDataset() async throws {
        for (arguments, message) in [
            (["--kind", "auto"], "--kind must be lora or lokr."),
            (["--max-duration", "0"], "--max-duration must be in (0, 600]."),
            (["--max-duration", "601"], "--max-duration must be in (0, 600]."),
            (["--log-every", "0"], "--log-every must be greater than zero."),
        ] {
            let command = try MusicTrainAdapter.parse(["--dataset", "/missing.json", "--output", "out.safetensors"] + arguments)
            do {
                try await command.run()
                XCTFail("Expected validation failure")
            } catch let error as ValidationError {
                XCTAssertEqual(error.message, message)
            }
        }
    }

    func testArrayAndJSONLPreserveCaptionsAndLyrics() throws {
        let record = #"{"audio":"sub/one.wav","caption":" caption ","lyrics":" lyric "}"#
        for source in ["[\(record)]", "\n\(record)\n\n"] {
            try withManifest(source) { url in
                let records = try MusicTrainAdapter.loadManifest(from: url)
                XCTAssertEqual(records.count, 1)
                XCTAssertEqual(records[0].audio, "sub/one.wav")
                XCTAssertEqual(records[0].caption, " caption ")
                XCTAssertEqual(records[0].lyrics, " lyric ")
            }
        }
    }

    func testEmptyAndInvalidRecords() throws {
        for (source, message) in [
            ("[]", "Dataset manifest contains no examples."),
            (#"[{"audio":" ","caption":"caption"}]"#, "Dataset record 1 has an empty audio path."),
            (#"[{"audio":"one.wav","caption":" "}]"#, "Dataset record 1 has an empty caption."),
        ] {
            try withManifest(source) { url in
                XCTAssertThrowsError(try MusicTrainAdapter.loadManifest(from: url)) { error in
                    XCTAssertEqual((error as? ValidationError)?.message, message)
                }
            }
        }
    }

    private func withManifest(_ source: String, body: (URL) throws -> Void) throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("music-training-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try source.write(to: url, atomically: true, encoding: .utf8)
        try body(url)
    }
}
