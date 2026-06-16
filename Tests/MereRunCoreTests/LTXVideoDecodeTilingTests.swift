@testable import MereRunCore
import XCTest

final class LTXVideoDecodeTilingTests: XCTestCase {
    func testSmallRepresentativeClipDecodesWithoutTilingUnderDefaultReferenceBudget() {
        let tiling = selectDecodeTilingConfig(
            width: 640,
            height: 384,
            numFrames: 361,
            fps: 24,
            decodeBudgetGiB: 8.0
        )

        XCTAssertNil(tiling)
    }

    func testRepresentative720pClipUsesTemporalOnlyTilingUnderDefaultReferenceBudget() throws {
        let tiling = try XCTUnwrap(selectDecodeTilingConfig(
            width: 1280,
            height: 704,
            numFrames: 361,
            fps: 24,
            decodeBudgetGiB: 8.0
        ))

        XCTAssertNil(tiling.spatialTileSizeInPixels)
        XCTAssertEqual(tiling.spatialTileOverlapInPixels, 0)
        XCTAssertEqual(tiling.temporalTileSizeInFrames, 144)
        XCTAssertEqual(tiling.temporalTileOverlapInFrames, 24)
    }

    func testLong720pClipUsesReferenceTemporalOnlyTiling() throws {
        let tiling = try XCTUnwrap(selectDecodeTilingConfig(
            width: 1280,
            height: 704,
            numFrames: 2401,
            fps: 24,
            decodeBudgetGiB: 8.0
        ))

        XCTAssertNil(tiling.spatialTileSizeInPixels)
        XCTAssertEqual(tiling.spatialTileOverlapInPixels, 0)
        XCTAssertEqual(tiling.temporalTileSizeInFrames, 144)
        XCTAssertEqual(tiling.temporalTileOverlapInFrames, 24)
    }
}
