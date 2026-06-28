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
}
