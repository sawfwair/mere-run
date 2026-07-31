import XCTest
@testable import MereRunCore

final class InklingTextSFTTokenizerTests: XCTestCase {
    func testAssistantTargetUsesInklingVisibleTextAndSamplingBoundary() {
        XCTAssertEqual(
            InklingTextSFTTokenizer.assistantTargetText("  A concise answer.\n"),
            "<|content_text|>A concise answer.<|end_message|><|content_model_end_sampling|>"
        )
    }

    func testOfficialTokenizerBuildsAssistantOnlyTargetsWhenAvailable() async throws {
        guard let path = ProcessInfo.processInfo.environment[
            "MERERUN_INKLING_TOKENIZER_PATH"
        ] else {
            throw XCTSkip(
                "Set MERERUN_INKLING_TOKENIZER_PATH to run the official Inkling SFT tokenizer test."
            )
        }
        let tokenizer = try await InklingTokenizerAndTemplate.load(
            from: URL(fileURLWithPath: path),
            maxLengthOverride: 256
        )
        let examples = try InklingTextSFTTokenizer.tokenize(
            [
                TextSFTExample(
                    id: "official-tokenizer",
                    sources: ["test"],
                    messages: [
                        ChatMessage(role: .system, content: "You answer concisely."),
                        ChatMessage(role: .user, content: "Say ready now."),
                        ChatMessage(role: .assistant, content: "  ready\n"),
                    ]
                ),
            ],
            tokenizerAndTemplate: tokenizer,
            maxSequenceLength: 256
        )

        let example = try XCTUnwrap(examples.first)
        XCTAssertEqual(example.inputTokenIds.count, example.labelTokenIds.count)
        XCTAssertEqual(example.inputTokenIds.count, example.lossMask.count)
        XCTAssertTrue(example.lossMask.contains(0))
        XCTAssertTrue(example.lossMask.contains(1))
        XCTAssertEqual(example.labelTokenIds.last, tokenizer.endSamplingTokenID)
    }
}
