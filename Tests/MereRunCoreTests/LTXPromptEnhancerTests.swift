import Foundation
@testable import MereRunCore
import XCTest

final class LTXPromptEnhancerTests: XCTestCase {
    func testTextToVideoMessagesMatchUpstreamRequestShape() {
        let messages = LTXPromptEnhancer.messages(prompt: "A fox runs")

        XCTAssertEqual(messages.count, 2)
        XCTAssertEqual(messages[0].role, .system)
        XCTAssertTrue(messages[0].content.hasPrefix("You are given a user's short text-to-video request."))
        XCTAssertEqual(messages[1], ChatMessage(role: .user, content: "user prompt: A fox runs"))
    }

    func testImageToVideoMessagesCarryReferenceImage() {
        let image = URL(fileURLWithPath: "/tmp/frame.png")
        let messages = LTXPromptEnhancer.messages(
            prompt: "The fox turns",
            referenceImage: image
        )

        XCTAssertEqual(messages.count, 2)
        XCTAssertTrue(messages[0].content.hasPrefix("You are given a REFERENCE IMAGE"))
        XCTAssertEqual(messages[1].content, "User Raw Input Prompt: The fox turns.")
        XCTAssertEqual(messages[1].imageUrl, image.absoluteString)
    }

    func testCleanResponseMatchesUpstreamNormalization() {
        XCTAssertEqual(
            LTXPromptEnhancer.cleanResponse("## \u{201C}A fox\u{2014}runs\u{201D}"),
            "A fox-runs\""
        )
    }

    func testNoRepeatNgramBansOnlyCompletingTokens() {
        XCTAssertEqual(
            Gemma4Generator.noRepeatNgramBannedTokens(
                history: [1, 2, 3, 4, 5, 8, 2, 3, 4, 5],
                size: 5
            ),
            Set([8])
        )
        XCTAssertEqual(
            Gemma4Generator.noRepeatNgramBannedTokens(history: [1, 2, 1], size: 1),
            Set([1, 2])
        )
    }
}
