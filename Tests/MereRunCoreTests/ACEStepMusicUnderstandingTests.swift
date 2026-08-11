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

    func testParseUnderstandingOutputAcceptsOnlyOneUpstreamLanguageCode() {
        let valid = ACEStepPipeline.parseUnderstandingOutput(
            "<think>\nlanguage: EN\n</think>"
        )
        let multiple = ACEStepPipeline.parseUnderstandingOutput(
            "<think>\nlanguage: en, ja, ko\n</think>"
        )
        let unsupported = ACEStepPipeline.parseUnderstandingOutput(
            "<think>\nlanguage: zxx\n</think>"
        )

        XCTAssertEqual(valid.language, "en")
        XCTAssertNil(multiple.language)
        XCTAssertNil(unsupported.language)
    }

    func testPlanningPolicyKeepsExplicitDurationAndLanguage() {
        let merged = ACEStepPlanningPolicy.merge(
            userMetadata: .init(
                bpm: nil,
                duration: "30",
                language: "en"
            ),
            plan: .init(
                bpm: 110,
                durationSeconds: 560,
                keyscale: "C minor",
                language: "de",
                timesignature: "4"
            ),
            caption: "English vocal synth pop",
            durationSeconds: 30
        )

        XCTAssertEqual(merged.bpm, "110")
        XCTAssertEqual(merged.duration, "30")
        XCTAssertEqual(merged.language, "en")
        XCTAssertEqual(merged.keyscale, "C minor")
        XCTAssertEqual(merged.timesignature, "4")
        XCTAssertEqual(
            ACEStepPlanningPolicy.summary(merged),
            "bpm=110, keyscale=C minor, timesignature=4, language=en, duration=30s"
        )
    }

    func testPlanningPolicyUsesOneLanguageForPlannerAndLyrics() {
        XCTAssertEqual(
            ACEStepPlanningPolicy.effectiveLanguage(
                vocalLanguage: "en",
                metadataLanguage: nil
            ),
            "en"
        )
        XCTAssertEqual(
            ACEStepPlanningPolicy.effectiveLanguage(
                vocalLanguage: "en",
                metadataLanguage: "ja"
            ),
            "ja"
        )
    }

    func testCodeGenerationContextReserializesEffectivePlannerScalars() {
        let context = ACEStepLMCodeGenerationContext(
            caption: "original prompt",
            lyrics: "[Instrumental]",
            reasoning: """
            <think>
            bpm: 200
            caption: A multiline cinematic description.
              With an indented continuation.
            duration: 55)
            keyscale: G minor

            language: zxx4388
            </think>
            """
        ).applying(
            userMetadata: .init(
                bpm: "200",
                duration: "85",
                keyscale: "G minor",
                language: "en",
                timesignature: "4"
            )
        )

        XCTAssertTrue(context.reasoning.contains("duration: 85\n"))
        XCTAssertTrue(context.reasoning.contains("language: en\n"))
        XCTAssertTrue(context.reasoning.contains("timesignature: 4\n</think>"))
        XCTAssertFalse(context.reasoning.contains("55)"))
        XCTAssertFalse(context.reasoning.contains("zxx4388"))
        XCTAssertTrue(context.reasoning.contains("  With an indented continuation."))
    }
}
