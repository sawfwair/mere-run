import XCTest
@testable import AudioSTT

final class ParakeetWindowMergeTests: XCTestCase {
    func testDeduplicatesBoundaryTokenWithLaterTimestampInNextWindow() {
        let first = [token(1, 13.0)]
        let second = [token(1, 13.08), token(2, 14)]

        let merged = ParakeetAlignment.mergeLongestCommonSubsequence(
            first, second, overlapDuration: 2, windowOverlap: 13..<15
        )

        XCTAssertEqual(merged.map(\.id), [1, 2])
    }

    func testPreservesRepeatedWordOutsideAudioOverlap() {
        let merged = ParakeetAlignment.mergeLongestCommonSubsequence(
            [token(1, 12.8)], [token(1, 13.1), token(2, 14)],
            overlapDuration: 2, windowOverlap: 13..<15
        )
        XCTAssertEqual(merged.map(\.id), [1, 1, 2])
    }

    func testPreservesSpeechBeforeSilenceInNextWindow() {
        let merged = ParakeetAlignment.mergeLongestCommonSubsequence(
            [token(1, 14.2)], [token(2, 20)],
            overlapDuration: 2, windowOverlap: 13..<15
        )
        XCTAssertEqual(merged.map(\.id), [1, 2])
    }

    func testUnmatchedTokenSpanningMidpointIsNotDroppedFromBothWindows() {
        let merged = ParakeetAlignment.mergeLongestCommonSubsequence(
            [token(1, 13.8, duration: 0.4)], [token(2, 13.9), token(3, 14.4)],
            overlapDuration: 2, windowOverlap: 13..<15
        )
        XCTAssertEqual(merged.map(\.id), [1, 3])
    }

    func testDefaultMergePreservesNonoverlappingRepeatedTokens() {
        let merged = ParakeetAlignment.mergeLongestCommonSubsequence(
            [token(1, 13)], [token(1, 13.08)], overlapDuration: 2
        )
        XCTAssertEqual(merged.map(\.id), [1, 1])
    }

    private func token(_ id: Int, _ start: Double, duration: Double = 0) -> ParakeetAlignedToken {
        ParakeetAlignedToken(id: id, text: " \(id)", start: start, duration: duration)
    }
}
