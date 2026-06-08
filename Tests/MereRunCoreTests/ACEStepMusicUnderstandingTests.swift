import MLX
import XCTest
@testable import MereRunCore

final class ACEStepMusicUnderstandingTests: MereRunCoreTestCase {
    func testAudioCodeStringSerializesClampedIndices() {
        let indices = MLXArray([Int32(1), Int32(64_500), Int32(-4)], [1, 3, 1])

        XCTAssertEqual(
            ACEStepPipeline.audioCodeString(fromIndices: indices),
            "<|audio_code_1|><|audio_code_63999|><|audio_code_0|>"
        )
    }

    func testParseUnderstandingOutputExtractsReasoningMetadataAndLyrics() {
        let output = """
        <think>
        bpm: 95
        caption: Latin pop groove.
          Bright percussion and club bass.
        duration: 273
        genres: reggaeton
        keyscale: G major
        language: en
        timesignature: 4
        </think>
        [Verse]
        We move around the room
        """

        let metadata = ACEStepPipeline.parseUnderstandingOutput(output)

        XCTAssertEqual(metadata.bpm, 95)
        XCTAssertEqual(metadata.caption, "Latin pop groove.\nBright percussion and club bass.")
        XCTAssertEqual(metadata.durationSeconds, 273)
        XCTAssertEqual(metadata.keyscale, "G major")
        XCTAssertEqual(metadata.language, "en")
        XCTAssertEqual(metadata.timesignature, "4")
        XCTAssertEqual(metadata.lyrics, "[Verse]\nWe move around the room")
    }

    func testParseUnderstandingOutputTreatsNAAsMissing() {
        let output = """
        <think>
        bpm: N/A
        caption: N/A
        duration: N/A
        keyscale: N/A
        language: unknown
        timesignature: N/A
        </think>
        """

        let metadata = ACEStepPipeline.parseUnderstandingOutput(output)

        XCTAssertNil(metadata.bpm)
        XCTAssertNil(metadata.caption)
        XCTAssertNil(metadata.durationSeconds)
        XCTAssertNil(metadata.keyscale)
        XCTAssertNil(metadata.language)
        XCTAssertNil(metadata.timesignature)
    }
}
