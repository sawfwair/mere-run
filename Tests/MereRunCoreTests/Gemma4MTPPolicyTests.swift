import XCTest
@testable import MereRunCore

final class Gemma4MTPPolicyTests: MereRunCoreTestCase {
    func testGemma4MTPPolicyGatesOnGreedyPromptAndEnvironment() {
        let greedy = GenerationConfig(maxTokens: 16, temperature: 0, topP: 1)
        let sampled = GenerationConfig(maxTokens: 16, temperature: 0.7, topP: 0.9)
        let env = [
            "MERERUN_GEMMA4_MTP_MIN_PROMPT_TOKENS": "8",
            "MERERUN_GEMMA4_MTP_BLOCK_SIZE": "6",
        ]

        XCTAssertNil(Gemma4MTPPolicy.activationReason(
            assistantAvailable: true,
            promptTokenCount: 8,
            generationConfig: greedy,
            prefixSeedWasUsed: false,
            environment: env
        ))
        // Sampled requests stay on the pipelined path by default (sampled-MTP
        // acceptance economics measured worse)…
        XCTAssertEqual(Gemma4MTPPolicy.activationReason(
            assistantAvailable: true,
            promptTokenCount: 8,
            generationConfig: sampled,
            prefixSeedWasUsed: false,
            environment: env
        ), "non-greedy sampling (MERERUN_GEMMA4_MTP_SAMPLED unset)")
        // …but can opt in to speculative sampled decode.
        XCTAssertNil(Gemma4MTPPolicy.activationReason(
            assistantAvailable: true,
            promptTokenCount: 8,
            generationConfig: sampled,
            prefixSeedWasUsed: false,
            environment: env.merging(["MERERUN_GEMMA4_MTP_SAMPLED": "1"]) { _, new in new }
        ))
        XCTAssertEqual(Gemma4MTPPolicy.activationReason(
            assistantAvailable: false,
            promptTokenCount: 8,
            generationConfig: greedy,
            prefixSeedWasUsed: false,
            environment: env
        ), "assistant not installed")
        XCTAssertEqual(Gemma4MTPPolicy.blockSize(configured: 4, environment: env), 6)
    }

}
