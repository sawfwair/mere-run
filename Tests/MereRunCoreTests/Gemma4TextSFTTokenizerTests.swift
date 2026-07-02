import XCTest
@testable import MereRunCore

final class Gemma4TextSFTTokenizerTests: XCTestCase {
    func testAssistantTargetTextStripsThinkingChannels() {
        XCTAssertEqual(
            Gemma4TextSFTTokenizer.assistantTargetText("Before <|channel>thought\nhidden\n<channel|> after "),
            "Before  after"
        )
    }

    func testCommonPrefixLengthStopsAtFirstDifference() {
        XCTAssertEqual(
            Gemma4TextSFTTokenizer.commonPrefixLength([1, 2, 3, 9], [1, 2, 4, 5]),
            2
        )
    }

    func testCommonPrefixLengthHandlesIdenticalPrefix() {
        XCTAssertEqual(
            Gemma4TextSFTTokenizer.commonPrefixLength([1, 2], [1, 2, 3, 4]),
            2
        )
    }
}
