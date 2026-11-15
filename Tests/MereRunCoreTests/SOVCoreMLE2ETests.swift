import AudioCore
import AudioCodecs
import AudioSTT
import Foundation
import XCTest

final class SOVCoreMLE2ETests: MereRunCoreTestCase {
    func testOriginalCheckpointDFlashTokensMatchAcrossAudioProviders() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let target = environment["MERERUN_TEST_SOV_PATH"],
              let draft = environment["MERERUN_TEST_SOV_DFLASH_PATH"],
              let artifact = environment["MERERUN_TEST_SOV_COREML_PATH"],
              environment["MERERUN_TEST_MLX_DEVICE"] == "gpu" else {
            throw XCTSkip("Set MERERUN_TEST_SOV_PATH, MERERUN_TEST_SOV_DFLASH_PATH, MERERUN_TEST_SOV_COREML_PATH, and MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        let resources = try XCTUnwrap(Bundle.module.resourceURL)
        let fixtures = ["D1/released/speech.wav", "Nemotron3/returning-speaker.wav"]
        var references: [String: [Int]] = [:]
        let providers: [AudioCore.ParakeetExecutionProvider] = [.mlx, .coreML(artifactURL: URL(fileURLWithPath: artifact))]
        for provider in providers {
            let generator = SOVTranscribeGenerator(executionProvider: provider)
            do {
                try await generator.prepare(modelPath: target, draftModel: draft)
                for fixture in fixtures {
                    let audio = resources.appendingPathComponent("Fixtures/\(fixture)")
                    let samples = try AudioReader.readAudio(from: audio)
                    let measured = try await generator.transcribePreparedMeasured(samples: samples, maxTokens: 448)
                    if fixture == fixtures[0] {
                        XCTAssertEqual(measured.text, "The parcel arrived on time, but the glass inside was broken.")
                    }
                    if let reference = references[fixture] {
                        XCTAssertEqual(measured.tokens, reference, "Provider token parity: \(fixture)")
                    } else {
                        references[fixture] = measured.tokens
                    }
                    XCTAssertGreaterThan(measured.verificationRounds, 0)
                    XCTAssertEqual(measured.reusedVerificationRounds ?? 0, measured.rejectedRounds)
                }
                await generator.unload()
            } catch {
                await generator.unload()
                throw error
            }
        }
    }
}
