import AudioCodecs
import Foundation
import XCTest
@testable import AudioSTT

final class ParakeetCoreMLE2ETests: MereRunCoreTestCase {
    func testPinnedArtifactMatchesExpectedTranscript() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let artifactPath = environment["MERERUN_TEST_PARAKEET_COREML_ARTIFACT"],
              let audioPath = environment["MERERUN_TEST_PARAKEET_COREML_AUDIO"],
              let expected = environment["MERERUN_TEST_PARAKEET_COREML_TRANSCRIPT"] else {
            throw XCTSkip(
                "Set the Parakeet Core ML artifact, audio, and expected transcript variables."
            )
        }

        let artifactURL = URL(fileURLWithPath: artifactPath).standardizedFileURL
        let audioURL = URL(fileURLWithPath: audioPath).standardizedFileURL
        let samples = try AudioReader.readAudio(from: audioURL)
        let generator = ParakeetGenerator(
            executionProvider: .coreML(artifactURL: artifactURL)
        )
        try await generator.prepare(modelPath: artifactURL.path)

        let measured = try await generator.transcribePreparedMeasured(samples: samples)

        XCTAssertEqual(measured.result.text, expected)
        XCTAssertGreaterThan(measured.timings.encoderSeconds, 0)
        XCTAssertGreaterThan(measured.timings.decoderSeconds, 0)
        XCTAssertGreaterThan(measured.timings.totalSeconds, 0)
    }
}
