import AudioCore
import AudioSTT
import Foundation
import MLX
import XCTest

final class NemotronTranscribeE2ETests: MereRunCoreTestCase {
    func testOriginalBF16CheckpointTranscribesKnownAudio() async throws {
        try await transcribeKnownAudio(environmentKey: "MERERUN_TEST_NEMOTRON_TRANSCRIBE_PATH")
    }

    func testRoutedExpertINT4CheckpointTranscribesKnownAudio() async throws {
        try await transcribeKnownAudio(environmentKey: "MERERUN_TEST_NEMOTRON_TRANSCRIBE_INT4_PATH")
    }

    private func transcribeKnownAudio(environmentKey: String) async throws {
        guard let path = ProcessInfo.processInfo.environment[environmentKey] else {
            throw XCTSkip("Set \(environmentKey) to the checkpoint and MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        guard ProcessInfo.processInfo.environment["MERERUN_TEST_MLX_DEVICE"] == "gpu" else {
            throw XCTSkip("The installed checkpoint test requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        let audio = try XCTUnwrap(Bundle.module.resourceURL).appendingPathComponent("Fixtures/D1/released/speech.wav")
        let plan = try SpeechTranscriptionResolver.resolve(
            request: ASRRequest(audioURL: audio, maxTokens: 96), preferredBackend: .nemotron, modelOverride: path
        )
        let outcome = try await SpeechTranscriptionOperation.execute(plan, executor: NativeSpeechTranscriptionExecutor())
        XCTAssertEqual(outcome.backend, .nemotron)
        XCTAssertEqual(outcome.result.text, "The parcel arrived on time, but the glass inside was broken.")
    }
}
