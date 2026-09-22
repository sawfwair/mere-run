import XCTest
@testable import MereRunCore

final class FalconPerceptionCoordinateDecoderTests: XCTestCase {
    private func logits(bins: Int, x: Int, y: Int, nextX: Int, nextY: Int) -> [Float] {
        var values = [Float](repeating: -10, count: bins * 2)
        values[nextX] = 1
        values[bins + nextY] = 1
        values[x] = 2
        values[bins + y] = 2
        return values
    }

    func testFirstCoordinateUsesIndependentAxisArgmax() {
        var decoder = FalconPerceptionCoordinateDecoder()
        let coordinate = decoder.decode(logits: logits(bins: 101, x: 20, y: 70, nextX: 40, nextY: 80))
        XCTAssertEqual(coordinate, .init(x: 0.2, y: 0.7))
        XCTAssertEqual(decoder.history, [coordinate])
    }

    func testRepeatSuppressesBothChosenBinsBeforeAutoregressiveFeedback() {
        var decoder = FalconPerceptionCoordinateDecoder()
        let values = logits(bins: 101, x: 20, y: 70, nextX: 40, nextY: 80)
        _ = decoder.decode(logits: values)
        XCTAssertEqual(decoder.decode(logits: values), .init(x: 0.4, y: 0.8))
        XCTAssertEqual(decoder.history.count, 2)
    }

    func testNearRepeatUsesAllEarlierCoordinatesIncludingUnfinishedDetections() {
        var decoder = FalconPerceptionCoordinateDecoder()
        _ = decoder.decode(logits: logits(bins: 1001, x: 200, y: 700, nextX: 400, nextY: 800))
        _ = decoder.decode(logits: logits(bins: 1001, x: 900, y: 900, nextX: 400, nextY: 800))
        XCTAssertEqual(
            decoder.decode(logits: logits(bins: 1001, x: 205, y: 705, nextX: 400, nextY: 800)),
            .init(x: 0.4, y: 0.8)
        )
    }

    func testOneNearbyAxisDoesNotSuppressTheOtherLocation() {
        var decoder = FalconPerceptionCoordinateDecoder()
        _ = decoder.decode(logits: logits(bins: 101, x: 20, y: 70, nextX: 40, nextY: 80))
        XCTAssertEqual(
            decoder.decode(logits: logits(bins: 101, x: 20, y: 90, nextX: 40, nextY: 80)),
            .init(x: 0.2, y: 0.9)
        )
    }

    func testExactlyOnePercentIsNotSuppressed() {
        var decoder = FalconPerceptionCoordinateDecoder()
        _ = decoder.decode(logits: logits(bins: 101, x: 0, y: 0, nextX: 50, nextY: 50))
        XCTAssertEqual(
            decoder.decode(logits: logits(bins: 101, x: 1, y: 0, nextX: 50, nextY: 50)),
            .init(x: 0.01, y: 0)
        )
    }

    func testHistoryUsesDoubleBinRatiosAtTheStrictThreshold() {
        var decoder = FalconPerceptionCoordinateDecoder()
        _ = decoder.decode(logits: logits(bins: 101, x: 8, y: 8, nextX: 50, nextY: 50))
        // Python's 0.09 - 0.08 is below 0.01; prematurely casting history to Float changes that decision.
        XCTAssertEqual(
            decoder.decode(logits: logits(bins: 101, x: 9, y: 9, nextX: 50, nextY: 50)),
            .init(x: 0.5, y: 0.5)
        )
    }

    func testBatchSlotsAndNewQueriesHaveIndependentHistories() {
        var decoders = Array(repeating: FalconPerceptionCoordinateDecoder(), count: 2)
        let values = logits(bins: 101, x: 20, y: 70, nextX: 40, nextY: 80)
        _ = decoders[0].decode(logits: values)
        XCTAssertEqual(decoders[0].decode(logits: values), .init(x: 0.4, y: 0.8))
        XCTAssertEqual(decoders[1].decode(logits: values), .init(x: 0.2, y: 0.7))
        var newQuery = FalconPerceptionCoordinateDecoder()
        XCTAssertEqual(newQuery.decode(logits: values), .init(x: 0.2, y: 0.7))
    }

    func testFirstMaximumTieAndNaNFollowReferenceArgmax() {
        var decoder = FalconPerceptionCoordinateDecoder()
        XCTAssertEqual(decoder.decode(logits: [1, 1, 0, 0, 1, 1]), .init(x: 0, y: 0.5))
        var nanDecoder = FalconPerceptionCoordinateDecoder()
        XCTAssertEqual(nanDecoder.decode(logits: [0, .nan, .nan, .nan, 1, .nan]), .init(x: 0.5, y: 0))
    }

    func testAttemptLimitReturnsLastCandidateEvenWhenItRepeats() {
        var decoder = FalconPerceptionCoordinateDecoder()
        let values = [Float](repeating: 0, count: 2 * 20001)
        _ = decoder.decode(logits: values)
        let coordinate = decoder.decode(logits: values)
        XCTAssertEqual(coordinate, .init(x: 99.0 / 20000, y: 99.0 / 20000))
        XCTAssertEqual(decoder.history.count, 2)
    }
}
