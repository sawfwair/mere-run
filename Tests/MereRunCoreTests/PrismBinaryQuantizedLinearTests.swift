import MLX
import XCTest
@testable import MereRunCore

final class PrismBinaryQuantizedLinearTests: XCTestCase {
    func testBinaryAffineLinearUnpacksOneBitWeights() {
        let layer = PrismBinaryQuantizedLinear(
            weight: MLXArray([UInt32.max, UInt32(0)], [2, 1]),
            bias: nil,
            scales: MLXArray([Float(2), Float(2)], [2, 1]),
            biases: MLXArray([Float(-1), Float(-1)], [2, 1]),
            groupSize: 32,
            bits: 1
        )
        let input = MLXArray(Array(repeating: Float(1), count: 32), [1, 32])

        let output = layer(input)
        eval(output)

        XCTAssertEqual(output[0, 0].item(Float.self), 32, accuracy: 0.001)
        XCTAssertEqual(output[0, 1].item(Float.self), -32, accuracy: 0.001)
    }

    func testBinaryAffineLinearHandlesRankThreeInputAndLinearBias() {
        let layer = PrismBinaryQuantizedLinear(
            weight: MLXArray([UInt32(0x0000_FFFF)], [1, 1]),
            bias: MLXArray([Float(3)], [1]),
            scales: MLXArray([Float(2), Float(4)], [1, 2]),
            biases: MLXArray([Float(-1), Float(0.5)], [1, 2]),
            groupSize: 16,
            bits: 1
        )
        let input = MLXArray(
            Array(repeating: Float(1), count: 32) + Array(repeating: Float(2), count: 32),
            [1, 2, 32]
        )

        let output = layer(input)
        eval(output)

        XCTAssertEqual(output.shape, [1, 2, 1])
        XCTAssertEqual(output[0, 0, 0].item(Float.self), 27, accuracy: 0.001)
        XCTAssertEqual(output[0, 1, 0].item(Float.self), 51, accuracy: 0.001)
    }
}
