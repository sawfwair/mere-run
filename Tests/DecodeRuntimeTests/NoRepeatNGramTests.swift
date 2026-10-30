import XCTest
@testable import MereRunDecode

final class NoRepeatNGramTests: XCTestCase {
    func testBansOnlyPreviouslyCompletedContinuationOfTheCurrentPrefix() {
        XCTAssertEqual(NoRepeatNGram.bannedTokens(in: [1, 2, 3, 4, 1, 2], size: 3, window: 128), [3])
        XCTAssertEqual(NoRepeatNGram.bannedTokens(in: [1, 2, 3, 4, 1, 2], size: 3, window: 3), [])
        XCTAssertEqual(NoRepeatNGram.bannedTokens(in: [1, 2], size: 3, window: 128), [])
        XCTAssertEqual(NoRepeatNGram.bannedTokens(in: [1, 2, 1, 2], size: 0, window: 128), [])
    }

    func testUnigramsAndAllowedTokensUseTheGeneratedWindow() {
        XCTAssertEqual(NoRepeatNGram.bannedTokens(in: [1, 2, 1, 3], size: 1, window: 2), [1, 3])
        XCTAssertEqual(NoRepeatNGram.bannedTokens(in: [1, 2, 1, 3], size: 1, window: 2, allowedTokens: [3]), [1])
    }

    func testThirtyFiveTokenDottedLineLoopIsStoppedAtTheUpstreamBoundary() {
        XCTAssertEqual(NoRepeatNGram.bannedTokens(in: [Int](repeating: 5, count: 34), size: 35, window: 128), [])
        XCTAssertEqual(NoRepeatNGram.bannedTokens(in: [Int](repeating: 5, count: 35), size: 35, window: 128), [5])
        XCTAssertEqual(NoRepeatNGram.bannedTokens(in: [5, 5, 5], size: 3, window: 2), [])
    }
}
