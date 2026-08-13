import MLX
@testable import MereRunCore
import XCTest

final class LTXTextToAudioTests: XCTestCase {
    func testAudioOnlyCheckpointFilterKeepsOnlyRequiredAudioTransformerWeights() {
        XCTAssertTrue(isLTXAudioOnlyTransformerWeight(
            "model.diffusion_model.audio_patchify_proj.weight"
        ))
        XCTAssertTrue(isLTXAudioOnlyTransformerWeight(
            "model.diffusion_model.transformer_blocks.0.audio_attn1.to_q.weight"
        ))
        XCTAssertTrue(isLTXAudioOnlyTransformerWeight(
            "model.diffusion_model.transformer_blocks.47.audio_prompt_scale_shift_table"
        ))
        XCTAssertFalse(isLTXAudioOnlyTransformerWeight(
            "model.diffusion_model.patchify_proj.weight"
        ))
        XCTAssertFalse(isLTXAudioOnlyTransformerWeight(
            "model.diffusion_model.transformer_blocks.0.video_to_audio_attn.to_q.weight"
        ))
    }

    func testCustomSigmaScheduleAcceptsDescendingTerminalZeroValues() throws {
        XCTAssertEqual(
            try validatedLTXSigmaSchedule([1, 0.75, 0.2, 0]),
            [1, 0.75, 0.2, 0]
        )
    }

    func testCustomSigmaScheduleRejectsAscendingOrNonterminalValues() {
        XCTAssertThrowsError(try validatedLTXSigmaSchedule([1, 0.2, 0.4, 0]))
        XCTAssertThrowsError(try validatedLTXSigmaSchedule([1, 0.5, 0.1]))
    }
}
