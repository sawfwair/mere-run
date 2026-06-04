import XCTest
@testable import MereRunCore

final class Gemma4ResponseCleaningTests: MereRunCoreTestCase {
    func testKeepsFinalChannelWhenThinkingIsHidden() {
        let cleaned = Gemma4Generator.cleanedResponse(
            """
            <|channel>thought
            Work privately.
            <|channel>final
            gemma4 text ok
            """,
            showThinking: false
        )

        XCTAssertEqual(cleaned, "gemma4 text ok")
    }

    func testKeepsTextAfterThoughtCloseMarkerWhenThinkingIsHidden() {
        let cleaned = Gemma4Generator.cleanedResponse(
            """
            <|channel>thought
            Work privately.<channel|>gemma4 text ok
            """,
            showThinking: false
        )

        XCTAssertEqual(cleaned, "gemma4 text ok")
    }

    func testRemovesDanglingThoughtChannelWhenThinkingIsHidden() {
        let cleaned = Gemma4Generator.cleanedResponse(
            """
            <|channel>thought
            Work privately.
            """,
            showThinking: false
        )

        XCTAssertEqual(cleaned, "")
    }

    func testPreservesThinkingWhenRequested() {
        let response = "<|channel>thought\nWork privately."

        XCTAssertEqual(
            Gemma4Generator.cleanedResponse(response, showThinking: true),
            response
        )
    }
}
