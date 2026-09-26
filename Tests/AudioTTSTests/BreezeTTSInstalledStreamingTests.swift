import AudioCore
import AudioTTS
import Foundation
import XCTest

/// Opt-in checkpoint proof: MERERUN_TEST_BREEZE_ROOT=/path/to/Breeze-TTS-2 swift test --filter BreezeTTSInstalledStreamingTests
final class BreezeTTSInstalledStreamingTests: XCTestCase {
    func testEnglishVocalEventStreamsAudioBeforeCompletion() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["MERERUN_TEST_BREEZE_ROOT"] else {
            throw XCTSkip("Set MERERUN_TEST_BREEZE_ROOT to run the real Breeze checkpoint.")
        }
        let generator = BreezeTTSGenerator()
        let request = TTSRequest(
            text: "(laugh) That was unexpected.",
            voiceDescription: "An amused voice with a light laugh",
            temperature: 0.6,
            seed: 42,
            cfgScale: 4,
            outputURL: URL(fileURLWithPath: "/tmp/breeze-stream-proof.wav")
        )
        let start = Date()
        var firstChunkSeconds: TimeInterval?
        var chunkCount = 0
        var sampleCount = 0
        var completedDuration: TimeInterval?
        for try await event in generator.generateStream(
            request, options: TTSStreamingOptions(chunkTokenInterval: 5), modelPath: modelPath
        ) {
            switch event {
            case .token: break
            case .audioChunk(let samples, let sampleRate):
                XCTAssertEqual(sampleRate, 24_000)
                XCTAssertFalse(samples.isEmpty)
                firstChunkSeconds = firstChunkSeconds ?? Date().timeIntervalSince(start)
                chunkCount += 1
                sampleCount += samples.count
            case .completed(let result):
                completedDuration = result.duration
                XCTAssertNotNil(firstChunkSeconds)
            }
        }
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertGreaterThan(chunkCount, 0)
        XCTAssertGreaterThan(sampleCount, 0)
        XCTAssertNotNil(completedDuration)
        print("Breeze real checkpoint: firstChunk=\(firstChunkSeconds ?? -1)s total=\(elapsed)s chunks=\(chunkCount) streamedSamples=\(sampleCount) duration=\(completedDuration ?? -1)s")
        await generator.unload()
    }
}
