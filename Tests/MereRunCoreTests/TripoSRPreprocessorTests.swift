import MediaIO
import MLX
@testable import MereRunCore
import XCTest

final class TripoSRPreprocessorTests: MereRunCoreTestCase {
    func testTransparentForegroundIsCroppedPaddedAndGrayComposited() throws {
        var rgba = [UInt8](repeating: 0, count: 6 * 4 * 4)
        for y in 1...2 {
            for x in 2...4 {
                let offset = (y * 6 + x) * 4
                rgba[offset] = 255
                rgba[offset + 1] = 64
                rgba[offset + 2] = 0
                rgba[offset + 3] = 255
            }
        }
        let image = try MediaImage(width: 6, height: 4, rgba8: rgba)
        let result = try TripoSRPreprocessor.prepare(
            image: image,
            size: 4,
            foregroundPolicy: .automaticTransparentAlpha(foregroundRatio: 0.5)
        )
        MLX.eval(result.image)
        XCTAssertTrue(result.croppedTransparentForeground)
        // Upstream's exclusive maximum crop yields 2x1, square 2, padded to 4.
        XCTAssertEqual(result.preparedWidth, 4)
        XCTAssertEqual(result.preparedHeight, 4)
        XCTAssertEqual(result.image.shape, [1, 4, 4, 3])
        let values = result.image.asArray(Float.self)
        XCTAssertEqual(values[0], 127.0 / 255.0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 127.0 / 255.0, accuracy: 1e-6)
        XCTAssertEqual(values[2], 127.0 / 255.0, accuracy: 1e-6)
    }

    func testOpaqueInputIsTreatedAsAlreadyFramed() throws {
        let image = try MediaImage(
            width: 2,
            height: 2,
            rgba8: [
                255, 0, 0, 255, 0, 255, 0, 255,
                0, 0, 255, 255, 255, 255, 255, 255,
            ]
        )
        let result = try TripoSRPreprocessor.prepare(image: image, size: 2)
        XCTAssertFalse(result.croppedTransparentForeground)
        XCTAssertEqual(result.preparedWidth, 2)
        XCTAssertEqual(result.image.asArray(Float.self).prefix(3), [1, 0, 0])
    }

    func testRejectsNonPositivePublicConditioningSize() throws {
        let image = try MediaImage(width: 1, height: 1, rgba8: [0, 0, 0, 255])
        XCTAssertThrowsError(try TripoSRPreprocessor.prepare(image: image, size: 0)) {
            XCTAssertEqual($0 as? TripoSRPreprocessingError, .invalidImageSize(0))
        }
    }

    func testAntialiasedBilinearDownsampleMatchesPinnedPyTorchFixture() {
        let source = (0..<(3 * 5 * 3)).map { Float($0) / 44 }
        let output = TripoSRPreprocessor.antialiasedBilinearResize(
            source,
            sourceWidth: 5,
            sourceHeight: 3,
            targetWidth: 3,
            targetHeight: 2
        )
        let expected: [Float] = [
            0.157_061_696, 0.179_788_977, 0.202_516_243,
            0.264_204_562, 0.286_931_813, 0.309_659_094,
            0.371_347_398, 0.394_074_649, 0.416_801_959,
            0.583_198_071, 0.605_925_322, 0.628_652_632,
            0.690_340_936, 0.713_068_187, 0.735_795_498,
            0.797_483_742, 0.820_211_053, 0.842_938_304,
        ]
        XCTAssertEqual(output.count, expected.count)
        for (actual, expected) in zip(output, expected) {
            XCTAssertEqual(actual, expected, accuracy: 2e-6)
        }
    }
}
