import MLX
import XCTest
@testable import MereRunCore

final class PrismLowBitQuantizationTests: XCTestCase {
    func testOneBitPreQuantizedEmbeddingDequantizesOnlySelectedRows() {
        let embedding = PreQuantizedEmbedding(
            weight: MLXArray([UInt32.max, UInt32(0)], [2, 1]),
            scales: MLXArray([Float(2), Float(2)], [2, 1]),
            biases: MLXArray([Float(-1), Float(-1)], [2, 1]),
            groupSize: 32,
            bits: 1
        )

        let output = embedding(MLXArray([Int32(0), Int32(1)]))
        eval(output)

        XCTAssertEqual(output.shape, [2, 32])
        XCTAssertEqual(output[0, 0].item(Float.self), 1, accuracy: 0.001)
        XCTAssertEqual(output[1, 0].item(Float.self), -1, accuracy: 0.001)
    }

    func testNativeBinaryAffineLinearUnpacksOneBitWeights() {
        let layer = PortableQuantizedLinear(
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

    func testNativeBinaryAffineLinearHandlesRankThreeInputAndLinearBias() {
        let layer = PortableQuantizedLinear(
            weight: MLXArray([UInt32.max], [1, 1]),
            bias: MLXArray([Float(3)], [1]),
            scales: MLXArray([Float(2)], [1, 1]),
            biases: MLXArray([Float(-1)], [1, 1]),
            groupSize: 32,
            bits: 1
        )
        let input = MLXArray(
            Array(repeating: Float(1), count: 32) + Array(repeating: Float(2), count: 32),
            [1, 2, 32]
        )

        let output = layer(input)
        eval(output)

        XCTAssertEqual(output.shape, [1, 2, 1])
        XCTAssertEqual(output[0, 0, 0].item(Float.self), 35, accuracy: 0.001)
        XCTAssertEqual(output[0, 1, 0].item(Float.self), 67, accuracy: 0.001)
    }

    func testNativeTernaryAffineLinearUnpacksTwoBitWeights() {
        let layer = PortableQuantizedLinear(
            weight: MLXArray(
                [UInt32(0xAAAA_AAAA), UInt32(0xAAAA_AAAA), UInt32(0), UInt32(0)],
                [2, 2]
            ),
            bias: nil,
            scales: MLXArray([Float(1), Float(1)], [2, 1]),
            biases: MLXArray([Float(-1), Float(-1)], [2, 1]),
            groupSize: 32,
            bits: 2
        )
        let input = MLXArray(Array(repeating: Float(1), count: 32), [1, 32])

        let output = layer(input)
        eval(output)

        XCTAssertEqual(output[0, 0].item(Float.self), 32, accuracy: 0.001)
        XCTAssertEqual(output[0, 1].item(Float.self), -32, accuracy: 0.001)
    }
}
