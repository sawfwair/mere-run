import MLX
import XCTest
@testable import MereRunCore

final class SCAIL2MaskEncoderTests: MereRunCoreTestCase {
    func testSevenColorsMapToPublishedChannelOrder() {
        let colors: [Float] = [
            1, 1, 1,
            1, -1, -1,
            -1, 1, -1,
            -1, -1, 1,
            1, 1, -1,
            1, -1, 1,
            -1, 1, 1,
            -1, -1, -1,
        ]
        let mask = MLXArray(colors, [1, 1, 8, 3])
        let encoded = SCAIL2MaskEncoder.encode(mask, spatialCompression: 1)
        eval(encoded)

        XCTAssertEqual(encoded.shape, [28, 1, 1, 8])
        let values = encoded.asArray(Float.self)
        for repeatedFrame in 0..<4 {
            for color in 0..<7 {
                for pixel in 0..<8 {
                    let channel = repeatedFrame * 7 + color
                    XCTAssertEqual(values[channel * 8 + pixel], pixel == color ? 1 : 0)
                }
            }
        }
    }

    func testTemporalPackingRepeatsFirstFrameThenGroupsFourFrames() {
        var values = [Float](repeating: -1, count: 5 * 8 * 8 * 3)
        for frame in 0..<5 {
            let channel = frame % 3
            for pixel in 0..<(8 * 8) {
                values[frame * 8 * 8 * 3 + pixel * 3 + channel] = 1
            }
        }
        let encoded = SCAIL2MaskEncoder.encode(MLXArray(values, [5, 8, 8, 3]))
        eval(encoded)

        XCTAssertEqual(encoded.shape, [28, 2, 1, 1])
        let packed = encoded.asArray(Float.self)
        XCTAssertEqual(packed[1 * 2], 1)
        XCTAssertEqual(packed[2 * 2 + 1], 1)
        XCTAssertEqual(packed[10 * 2 + 1], 1)
        XCTAssertEqual(packed[15 * 2 + 1], 1)
        XCTAssertEqual(packed[23 * 2 + 1], 1)
    }

    func testAreaDownsamplePreservesCoverage() {
        var values = [Float](repeating: -1, count: 8 * 8 * 3)
        for pixel in 0..<16 {
            values[pixel * 3] = 1
        }
        let encoded = SCAIL2MaskEncoder.encode(MLXArray(values, [1, 8, 8, 3]))
        eval(encoded)

        XCTAssertEqual(encoded.shape, [28, 1, 1, 1])
        XCTAssertEqual(encoded.asArray(Float.self)[1], 0.25, accuracy: 1e-6)
    }
}
