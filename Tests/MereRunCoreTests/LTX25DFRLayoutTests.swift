import MLX
@testable import MereRunCore
import XCTest

final class LTX25DFRLayoutTests: MereRunCoreTestCase {
    func testCanvasMatchesUpstreamSegmentSelectionAndPadding() throws {
        XCTAssertEqual(
            try LTX25DFRLayout.resolveCanvas(frameCount: 97),
            LTX25DFRCanvas(
                frameCount: 97,
                segmentLength: 32,
                keyframePositions: [32, 64, 96]
            )
        )
        XCTAssertEqual(
            try LTX25DFRLayout.resolveCanvas(frameCount: 81),
            LTX25DFRCanvas(
                frameCount: 97,
                segmentLength: 32,
                keyframePositions: [32, 64, 96]
            )
        )
    }

    func testTemporalRoundBuildsLeadInTilesAndLocalSlots() throws {
        let ranges = try LTX25DFRLayout.tileRanges(
            seamPositions: [64, 128, 192],
            frameCount: 193,
            tileCount: 2
        )

        XCTAssertEqual(ranges.count, 2)
        XCTAssertEqual(ranges[0].pixelStart, 0)
        XCTAssertEqual(ranges[0].pixelEnd, 128)
        XCTAssertEqual(ranges[0].anchorKeyframes, [64, 128])
        XCTAssertEqual(ranges[0].slotKeyframes, [32, 96])
        XCTAssertEqual(ranges[0].dropLatentPrefix, 0)
        XCTAssertEqual(ranges[1].pixelStart, 64)
        XCTAssertEqual(ranges[1].pixelEnd, 192)
        XCTAssertEqual(ranges[1].anchorKeyframes, [64, 128, 192])
        XCTAssertEqual(ranges[1].slotKeyframes, [96, 160])
        XCTAssertEqual(ranges[1].dropLatentPrefix, 9)
        XCTAssertEqual(
            LTX25DFRLayout.remapPositionsToLocal([96, 160], pixelStart: 64),
            [32, 96]
        )
    }

    func testStitchDropsLeadInAndSharedSeam() throws {
        let ranges = try LTX25DFRLayout.tileRanges(
            seamPositions: [32, 64],
            frameCount: 65,
            tileCount: 2
        )
        let first = MLXArray(Array(0..<5).map(Float.init)).reshaped(1, 1, 5, 1, 1)
        let second = MLXArray(Array(100..<109).map(Float.init)).reshaped(1, 1, 9, 1, 1)
        let stitched = try LTX25DFRLayout.stitchTileLatents([first, second], ranges: ranges)
        MLX.eval(stitched)

        XCTAssertEqual(stitched.shape, [1, 1, 9, 1, 1])
        XCTAssertEqual(
            stitched.asArray(Float.self),
            [0, 1, 2, 3, 4, 105, 106, 107, 108]
        )
    }
}
