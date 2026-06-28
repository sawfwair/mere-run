import XCTest
@testable import MereRunCore

final class TextGenerationStopSequencesTests: XCTestCase {
    func testTrimmingStopsAtNorthRenderedChannelMarker() {
        let response = """
        def has_close_elements(numbers, threshold):
            return False
        <|CHANNEL_END|><|END_OFTURN_TOKEN|>|

        def has_close_elements(numbers, threshold):
            return True
        """

        let trimmed = TextGenerationStopSequences.trimming(
            response,
            sequences: TextGenerationStopSequences.defaultRenderedChatStops
        )

        XCTAssertEqual(trimmed.matchedSequence, "<|CHANNEL_END|>")
        XCTAssertEqual(
            trimmed.text,
            """
            def has_close_elements(numbers, threshold):
                return False
            """
        )
    }

    func testFirstMatchPrefersEarliestStopSequence() {
        let response = "answer<|im_end|> trailing <|CHANNEL_END|>"

        let match = TextGenerationStopSequences.firstMatch(
            in: response,
            sequences: TextGenerationStopSequences.defaultRenderedChatStops
        )

        XCTAssertEqual(match?.sequence, "<|im_end|>")
    }

    func testMergedPreservesLeadingNewlineStopSequences() {
        let sequences = TextGenerationStopSequences.merged([
            "\nif __name__",
            "\nif __name__",
            "   ",
        ])

        XCTAssertEqual(sequences, ["\nif __name__"])
    }

    func testChatRequestAndResponseCarryStopMetadata() {
        let request = ChatRequest(
            messages: [ChatMessage(role: .user, content: "hi")],
            stopSequences: ["END"]
        )
        let response = ChatResponse(
            response: "ok",
            tokensGenerated: 1,
            finishReason: .stopSequence
        )

        XCTAssertEqual(request.stopSequences, ["END"])
        XCTAssertEqual(response.finishReason, .stopSequence)
    }

    func testReasoningMarkupSplitsThinkBlocks() {
        let split = ChatReasoningMarkup.splitThinkBlocks(
            in: "<think>first thought</think>Final <think>second thought</think>answer"
        )

        XCTAssertEqual(split.visibleContent, "Final answer")
        XCTAssertEqual(split.reasoningContent, "first thought\n\nsecond thought")
        XCTAssertFalse(split.hasIncompleteReasoning)
        XCTAssertEqual(split.reasoningBlockCount, 2)
        XCTAssertTrue(split.hasReopenedReasoning)
    }

    func testReasoningMarkupCapturesIncompleteThinkBlock() {
        let split = ChatReasoningMarkup.splitThinkBlocks(in: "before <think>still working")

        XCTAssertEqual(split.visibleContent, "before")
        XCTAssertEqual(split.reasoningContent, "still working")
        XCTAssertTrue(split.hasIncompleteReasoning)
        XCTAssertEqual(split.reasoningBlockCount, 1)
        XCTAssertFalse(split.hasReopenedReasoning)
    }

    func testReasoningMarkupHandlesLeadingOrphanClose() {
        let split = ChatReasoningMarkup.splitThinkBlocks(
            in: "hidden reasoning</think>The visible answer."
        )

        XCTAssertEqual(split.visibleContent, "The visible answer.")
        XCTAssertEqual(split.reasoningContent, "hidden reasoning")
        XCTAssertEqual(split.reasoningBlockCount, 1)
        XCTAssertFalse(split.hasReopenedReasoning)
    }

    func testReasoningMarkupKeepsVisibleTextBeforeLateOrphanClose() {
        let split = ChatReasoningMarkup.splitThinkBlocks(
            in: "<think>draft</think>Final answer.</think>"
        )

        XCTAssertEqual(split.visibleContent, "Final answer.")
        XCTAssertEqual(split.reasoningContent, "draft")
        XCTAssertEqual(split.reasoningBlockCount, 1)
        XCTAssertFalse(split.hasReopenedReasoning)
    }

    func testReasoningMarkupFlagsThinkBlockReopenedAfterFinalText() {
        let split = ChatReasoningMarkup.splitThinkBlocks(
            in: "<think>draft</think>Final answer.<think>looping again</think>Duplicate answer."
        )

        XCTAssertEqual(split.visibleContent, "Final answer.Duplicate answer.")
        XCTAssertEqual(split.reasoningContent, "draft\n\nlooping again")
        XCTAssertEqual(split.reasoningBlockCount, 2)
        XCTAssertTrue(split.hasReopenedReasoning)
    }

    func testChatResponseSeparatesReasoningWhenThinkingHidden() {
        let response = ChatResponse(
            generatedText: "<think>work</think>Answer",
            tokensGenerated: 4,
            showThinking: false
        )

        XCTAssertEqual(response.response, "Answer")
        XCTAssertEqual(response.reasoningContent, "work")
        XCTAssertEqual(response.reasoningBlockCount, 1)
        XCTAssertFalse(response.hasReopenedReasoning)
    }
}
