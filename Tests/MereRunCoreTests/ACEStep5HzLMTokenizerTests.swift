import Foundation
import XCTest
@testable import MereRunCore

final class ACEStep5HzLMTokenizerTests: MereRunCoreTestCase {

    func testGenerationDefaultsMatchUpstreamPlannerSampling() {
        let config = ACEStep5HzLMGenerationConfig()

        XCTAssertEqual(config.temperature, 0.85, accuracy: 0.0001)
        XCTAssertNil(config.repetitionPenalty)
        XCTAssertEqual(config.repetitionContextSize, 40_960)
    }

    func testLoadTokenizerAudioCodeMap() throws {
        let env = ProcessInfo.processInfo.environment
        guard let root = env["MERERUN_TEST_ACESTEP_5HZ_ROOT"], !root.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_5HZ_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-5Hz-lm-1.7B to run this test.")
        }

        let tok = try ACEStep5HzLMTokenizer.load(from: URL(fileURLWithPath: root))
        XCTAssertEqual(tok.audioCodeTokenIds.count, 64_000)
        XCTAssertEqual(tok.audioCodeTokenIdToValue.count, 64_000)

        let id0 = try XCTUnwrap(tok.convertTokenToId("<|audio_code_0|>"))
        XCTAssertEqual(tok.audioCodeTokenIdToValue[id0], 0)

        let idMax = try XCTUnwrap(tok.convertTokenToId("<|audio_code_63999|>"))
        XCTAssertEqual(tok.audioCodeTokenIdToValue[idMax], 63_999)
    }
}
