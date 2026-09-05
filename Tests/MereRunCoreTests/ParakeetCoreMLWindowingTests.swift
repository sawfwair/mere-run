import XCTest
@testable import AudioSTT

final class ParakeetCoreMLWindowingTests: XCTestCase {
    func testBoundaryMovesIntoQuietIntervalWithinContextBudget() {
        var samples = [Float](repeating: 0.1, count: 3_200)
        samples.replaceSubrange(980..<1_020, with: repeatElement(Float.zero, count: 40))

        let ranges = ParakeetCoreMLWindowing.sampleRanges(samples: samples, sampleRate: 100)

        XCTAssertTrue((980..<1_020).contains(ranges[1].lowerBound))
        XCTAssertEqual(ranges.first?.lowerBound, 0)
        XCTAssertEqual(ranges.last?.upperBound, samples.count)
        for (previous, next) in zip(ranges, ranges.dropFirst()) {
            let overlap = previous.upperBound - next.lowerBound
            XCTAssertGreaterThanOrEqual(overlap, 200)
            XCTAssertLessThanOrEqual(overlap, 800)
            XCTAssertGreaterThan(next.lowerBound, previous.lowerBound)
            XCTAssertLessThanOrEqual(next.count, 1_500)
        }
    }

    func testBriefZeroCrossingDoesNotCountAsQuietSpeechBoundary() {
        var samples = [Float](repeating: 0.1, count: 3_200)
        samples[1_000] = 0

        let ranges = ParakeetCoreMLWindowing.sampleRanges(samples: samples, sampleRate: 100)

        XCTAssertEqual(ranges, [0..<1_500, 1_300..<2_800, 2_600..<3_200])
    }

    func testSilenceStillMakesProgressAndCoversFinalSamples() {
        let samples = [Float](repeating: 0, count: 4_001)

        let ranges = ParakeetCoreMLWindowing.sampleRanges(samples: samples, sampleRate: 100)

        XCTAssertEqual(ranges.first?.lowerBound, 0)
        XCTAssertEqual(ranges.last?.upperBound, samples.count)
        XCTAssertTrue(ranges.allSatisfy { !$0.isEmpty && $0.count <= 1_500 })
        XCTAssertTrue(zip(ranges, ranges.dropFirst()).allSatisfy { previous, next in
            next.lowerBound > previous.lowerBound && next.lowerBound < previous.upperBound
        })
    }

    func testShortInputNeedsNoBoundarySearch() {
        XCTAssertEqual(
            ParakeetCoreMLWindowing.sampleRanges(samples: [0, 0.1, 0], sampleRate: 100),
            [0..<3]
        )
        XCTAssertEqual(ParakeetCoreMLWindowing.sampleRanges(samples: [], sampleRate: 100), [])
        XCTAssertEqual(ParakeetCoreMLWindowing.sampleRanges(samples: [0], sampleRate: 0), [])
    }

    func testLowSampleRateRetainsValidWindowBounds() {
        let ranges = ParakeetCoreMLWindowing.sampleRanges(samples: Array(repeating: 0, count: 40), sampleRate: 1)

        XCTAssertEqual(ranges.first?.lowerBound, 0)
        XCTAssertEqual(ranges.last?.upperBound, 40)
        XCTAssertTrue(ranges.allSatisfy { !$0.isEmpty && $0.count <= 15 })
    }
}
