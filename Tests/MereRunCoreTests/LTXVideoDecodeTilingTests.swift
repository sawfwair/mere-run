import MLX
@testable import MereRunCore
import XCTest

final class LTXVideoDecodeTilingTests: MereRunCoreTestCase {
    func testGPUBlendAccumulatorMatchesWeightedOverlapReference() {
        let output = MLX.zeros([1, 3, 1, 1, 3], dtype: .float32)
        let weights = MLX.zeros([1, 1, 1, 1, 3], dtype: .float32)
        let first = MLXArray([
            Float(-1), 0,
            0, 0.5,
            1, -0.5,
        ], [1, 3, 1, 1, 2])
        let second = MLXArray([
            Float(1), 0.5,
            -1, 0,
            0.5, 1,
        ], [1, 3, 1, 1, 2])

        accumulateLTXDecodedTile(
            output: output,
            weights: weights,
            tileDecoded: first,
            outputFrameStart: 0,
            outputHeightStart: 0,
            outputWidthStart: 0,
            temporalMask: [1],
            heightMask: [1],
            widthMask: [1, 1]
        )
        accumulateLTXDecodedTile(
            output: output,
            weights: weights,
            tileDecoded: second,
            outputFrameStart: 0,
            outputHeightStart: 0,
            outputWidthStart: 1,
            temporalMask: [1],
            heightMask: [1],
            widthMask: [1, 1]
        )
        MLX.eval(output, weights)

        XCTAssertEqual(weights.asArray(Float.self), [1, 2, 1])
        XCTAssertEqual(output.asArray(Float.self), [-1, 1, 0.5, 0, -0.5, 0, 1, 0, 1])

        let frames = finalizeLTXDecodedTiles(output: output, weights: weights)
        MLX.eval(frames)
        XCTAssertEqual(frames.shape, [1, 1, 3, 3])
        XCTAssertEqual(frames.asArray(UInt8.self), [0, 127, 255, 191, 95, 127, 191, 127, 255])
    }

    func testShortLowResolutionClipDecodesWithoutTilingUnderDefaultReferenceBudget() {
        let tiling = selectDecodeTilingConfig(
            width: 512,
            height: 320,
            numFrames: 33,
            fps: 24,
            decodeBudgetGiB: 8.0
        )

        XCTAssertNil(tiling)
    }

    func testFiveSecond1024By576ClipUsesBoundedTemporalDecode() throws {
        let tiling = try XCTUnwrap(selectDecodeTilingConfig(
            width: 1_024,
            height: 576,
            numFrames: 121,
            fps: 24,
            decodeBudgetGiB: 8.0
        ))

        XCTAssertNil(tiling.spatialTileSizeInPixels)
        XCTAssertEqual(tiling.spatialTileOverlapInPixels, 0)
        XCTAssertEqual(tiling.temporalTileSizeInFrames, 32)
        XCTAssertEqual(tiling.temporalTileOverlapInFrames, 8)
    }

    func testLongLowerResolutionClipStillUsesBoundedTemporalDecode() throws {
        let tiling = try XCTUnwrap(selectDecodeTilingConfig(
            width: 640,
            height: 384,
            numFrames: 361,
            fps: 24,
            decodeBudgetGiB: 8.0
        ))

        XCTAssertEqual(tiling.temporalTileSizeInFrames, 32)
        XCTAssertEqual(tiling.temporalTileOverlapInFrames, 8)
    }

    func testExplicitSpatialTileCreatesSpatialOnlyDecodePlan() throws {
        let tiling = try XCTUnwrap(selectDecodeTilingConfig(
            width: 1_920,
            height: 1_080,
            numFrames: 33,
            fps: 24,
            decodeBudgetGiB: 64,
            spatialTileSizeInPixels: 768,
            spatialTileOverlapInPixels: 128
        ))

        XCTAssertEqual(tiling.spatialTileSizeInPixels, 768)
        XCTAssertEqual(tiling.spatialTileOverlapInPixels, 128)
        XCTAssertNil(tiling.temporalTileSizeInFrames)
        XCTAssertEqual(tiling.temporalTileOverlapInFrames, 0)
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
        XCTAssertEqual(tiling.temporalTileSizeInFrames, 32)
        XCTAssertEqual(tiling.temporalTileOverlapInFrames, 8)
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
        XCTAssertEqual(tiling.temporalTileSizeInFrames, 32)
        XCTAssertEqual(tiling.temporalTileOverlapInFrames, 8)
    }
}
