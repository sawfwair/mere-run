import MediaIO
import MLX
import XCTest
@testable import MereRunCore

final class SCAIL2GeneratorTests: MereRunCoreTestCase {
    func testSegmentBuilderMatchesPublishedLongVideoWindows() {
        XCTAssertEqual(
            SCAIL2SegmentBuilder.build(frameCount: 200, segmentLength: 81, segmentOverlap: 5),
            [0..<81, 76..<157]
        )
        XCTAssertEqual(
            SCAIL2SegmentBuilder.build(frameCount: 80, segmentLength: 81, segmentOverlap: 5),
            [0..<77]
        )
        XCTAssertEqual(
            SCAIL2SegmentBuilder.build(frameCount: 1, segmentLength: 81, segmentOverlap: 5),
            [0..<1]
        )
    }

    func testPadTrimFrameCountCompletesFinalWindow() {
        XCTAssertEqual(
            SCAIL2SegmentBuilder.paddedFrameCount(
                frameCount: 1,
                segmentLength: 81,
                segmentOverlap: 5
            ),
            81
        )
        XCTAssertEqual(
            SCAIL2SegmentBuilder.paddedFrameCount(
                frameCount: 82,
                segmentLength: 81,
                segmentOverlap: 5
            ),
            157
        )
        XCTAssertEqual(
            SCAIL2SegmentBuilder.paddedFrameCount(
                frameCount: 157,
                segmentLength: 81,
                segmentOverlap: 5
            ),
            157
        )
    }

    func testCleanHistoryReplacesOnlyLatentPrefix() {
        let latent = MLXArray(Array(0..<8).map(Float.init), [1, 4, 1, 2])
        let history = MLX.full([1, 2, 1, 2], values: MLXArray(Float(99)))
        let result = SCAIL2Generator.applyCleanHistory(latent, history: history)
        eval(result)

        XCTAssertEqual(result.asArray(Float.self), [99, 99, 99, 99, 4, 5, 6, 7])
    }

    func testHistoryPixelsDropsDecodedBatchBeforeNextVAEEncode() {
        let decoded = MLXArray(Array(0..<90).map(Float.init), [1, 6, 3, 5, 1])

        let history = SCAIL2Generator.historyPixels(from: decoded, overlap: 2)
        eval(history)

        XCTAssertEqual(history.shape, [2, 3, 5, 1])
        XCTAssertEqual(history.asArray(Float.self), Array(60..<90).map(Float.init))
    }

    func testReferenceOrientationFollowsUpstreamSwapRule() {
        XCTAssertEqual(
            SCAIL2Generator.resolvedDimensions(
                sourceWidth: 1_000,
                sourceHeight: 2_000,
                requestedWidth: 896,
                requestedHeight: 512
            ).width,
            512
        )
        XCTAssertEqual(
            SCAIL2Generator.resolvedDimensions(
                sourceWidth: 1_000,
                sourceHeight: 1_000,
                requestedWidth: 896,
                requestedHeight: 512
            ).width,
            896
        )
    }

    func testCenterCropKeepsPublishedHWCNormalization() throws {
        var rgba = [UInt8](repeating: 255, count: 4 * 2 * 4)
        let columns: [UInt8] = [0, 64, 128, 255]
        for y in 0..<2 {
            for x in 0..<4 {
                let offset = (y * 4 + x) * 4
                rgba[offset] = columns[x]
                rgba[offset + 1] = columns[x]
                rgba[offset + 2] = columns[x]
            }
        }
        let image = try MediaImage(width: 4, height: 2, rgba8: rgba)
        let result = SCAIL2InputPreprocessor.centerCroppedTensor(image: image, width: 2, height: 2)
        eval(result)

        XCTAssertEqual(result.shape, [1, 2, 2, 3])
        let values = result.asArray(Float.self)
        XCTAssertEqual(values[0], Float(64) / 127.5 - 1, accuracy: 1e-6)
        XCTAssertEqual(values[3], Float(128) / 127.5 - 1, accuracy: 1e-6)
    }

    func testHalfResolutionUsesAlignCornersFalseBilinearCenters() {
        let values = (0..<16).flatMap { value in [Float(value), Float(value), Float(value)] }
        let result = SCAIL2InputPreprocessor.halfResolutionBilinear(
            MLXArray(values, [1, 4, 4, 3])
        )
        eval(result)

        XCTAssertEqual(result.shape, [1, 2, 2, 3])
        let output = result.asArray(Float.self)
        XCTAssertEqual(output[0], 2.5, accuracy: 1e-6)
        XCTAssertEqual(output[3], 4.5, accuracy: 1e-6)
        XCTAssertEqual(output[6], 10.5, accuracy: 1e-6)
        XCTAssertEqual(output[9], 12.5, accuracy: 1e-6)
    }

    func testMaskBackgroundMatchesOfficialModeAndRoleSemantics() throws {
        let mask = try MediaImage(
            width: 3,
            height: 1,
            rgba8: [
                255, 255, 255, 255,
                0, 0, 0, 255,
                0, 0, 255, 255,
            ]
        )

        XCTAssertEqual(
            try SCAIL2Generator.normalizedMaskForMode(
                mask,
                mode: .animation,
                role: .mainReference
            ).rgba8,
            [
                255, 255, 255, 255,
                255, 255, 255, 255,
                0, 0, 255, 255,
            ]
        )
        XCTAssertEqual(
            try SCAIL2Generator.normalizedMaskForMode(
                mask,
                mode: .animation,
                role: .driving
            ).rgba8,
            [
                0, 0, 0, 255,
                0, 0, 0, 255,
                0, 0, 255, 255,
            ]
        )
        XCTAssertEqual(
            try SCAIL2Generator.normalizedMaskForMode(
                mask,
                mode: .replacement,
                role: .mainReference
            ).rgba8,
            [
                0, 0, 0, 255,
                0, 0, 0, 255,
                0, 0, 255, 255,
            ]
        )
        XCTAssertEqual(
            try SCAIL2Generator.normalizedMaskForMode(
                mask,
                mode: .replacement,
                role: .driving
            ).rgba8,
            [
                255, 255, 255, 255,
                255, 255, 255, 255,
                0, 0, 255, 255,
            ]
        )
        XCTAssertEqual(
            try SCAIL2Generator.normalizedMaskForMode(
                mask,
                mode: .animation,
                role: .additionalSubjectReference
            ).rgba8,
            [
                0, 0, 0, 255,
                0, 0, 0, 255,
                0, 0, 255, 255,
            ]
        )
    }
}
