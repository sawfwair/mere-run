import AudioCodecs
import AudioSTT
import Foundation
import XCTest

/// Opt-in tuning only. Release CLI measurements qualify the selected setting.
final class SOVPrefillProfileTests: MereRunCoreTestCase {
    private struct Row: Encodable {
        let fixture: String
        let chunkSize: Int
        let tokenParityWith256: Bool
        let measurements: [SOVMeasuredTranscription]
    }

    private struct Receipt: Encodable {
        let buildConfiguration = "debug-tuning-only"
        let rows: [Row]
    }

    func testProfileReleasedCheckpointPrefillChunks() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_PROFILE_SOV_PREFILL"] == "1",
              environment["MERERUN_TEST_MLX_DEVICE"] == "gpu",
              let target = environment["MERERUN_TEST_SOV_PATH"],
              let draft = environment["MERERUN_TEST_SOV_DFLASH_PATH"],
              let output = environment["MERERUN_TEST_SOV_PREFILL_OUTPUT"] else {
            throw XCTSkip("Set the SOV target/draft paths, GPU device, MERERUN_PROFILE_SOV_PREFILL=1, and MERERUN_TEST_SOV_PREFILL_OUTPUT for tuning.")
        }
        let generator = SOVTranscribeGenerator()
        var rows: [Row] = []
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            try await generator.prepare(modelPath: target, draftModel: draft)
            let resources = try XCTUnwrap(Bundle.module.resourceURL)
            for fixture in ["D1/released/speech.wav", "Nemotron3/returning-speaker.wav"] {
                let samples = try AudioReader.readAudio(from: resources.appendingPathComponent("Fixtures/\(fixture)"))
                var reference: [Int]?
                for chunkSize in [256, 512, 1024] {
                    var measurements: [SOVMeasuredTranscription] = []
                    for iteration in 0..<5 {
                        let measured = try await generator.transcribePreparedMeasured(samples: samples, prefillChunkSize: chunkSize)
                        if iteration >= 2 { measurements.append(measured) }
                    }
                    if reference == nil { reference = measurements[0].tokens }
                    rows.append(Row(fixture: fixture, chunkSize: chunkSize,
                        tokenParityWith256: measurements.allSatisfy { $0.tokens == reference }, measurements: measurements))
                    try encoder.encode(Receipt(rows: rows)).write(to: URL(fileURLWithPath: output))
                }
            }
            await generator.unload()
        } catch {
            await generator.unload()
            throw error
        }
    }
}
