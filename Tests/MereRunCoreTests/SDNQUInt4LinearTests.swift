import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class SDNQUInt4LinearTests: XCTestCase {
    private final class TinyLinearModel: Module {
        @ModuleInfo(key: "proj") var proj: Linear

        override init() {
            self._proj.wrappedValue = Linear(4, 2, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            proj(x)
        }
    }

    private final class TinyEmbeddingModel: Module {
        @ModuleInfo(key: "embed") var embed: Embedding

        override init() {
            self._embed.wrappedValue = Embedding(embeddingCount: 3, dimensions: 4)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            embed(x)
        }
    }

    private final class TinyConvModel: Module {
        @ModuleInfo(key: "conv") var conv: Conv2d

        override init() {
            self._conv.wrappedValue = Conv2d(
                inputChannels: 4,
                outputChannels: 1,
                kernelSize: 1,
                bias: true
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            conv(x)
        }
    }

    func testAsymmetricUInt4LinearUnpacksLowNibbleFirst() {
        let layer = SDNQUInt4Linear(
            weight: MLXArray([UInt8(0x10), UInt8(0x32), UInt8(0xFF), UInt8(0x00)], [4]),
            bias: nil,
            scales: MLXArray([Float(2), Float(3), Float(1), Float(2)], [2, 2, 1]),
            zeroPoints: MLXArray([Float(1), Float(-1), Float(0), Float(0)], [2, 2, 1]),
            inputDimensions: 4,
            outputDimensions: 2,
            groupSize: 2
        )
        let input = MLXArray(Array(repeating: Float(1), count: 4), [1, 4])

        let output = layer(input)
        eval(output)

        XCTAssertEqual(output.shape, [1, 2])
        XCTAssertEqual(output[0, 0].item(Float.self), 17, accuracy: 0.001)
        XCTAssertEqual(output[0, 1].item(Float.self), 30, accuracy: 0.001)
    }

    func testAsymmetricUInt4LinearHandlesRankThreeInputAndBias() {
        let layer = SDNQUInt4Linear(
            weight: MLXArray([UInt8(0x21), UInt8(0x43)], [2]),
            bias: MLXArray([Float(5)], [1]),
            scales: MLXArray([Float(1), Float(1)], [1, 2, 1]),
            zeroPoints: MLXArray([Float(0), Float(0)], [1, 2, 1]),
            inputDimensions: 4,
            outputDimensions: 1,
            groupSize: 2
        )
        let input = MLXArray(
            Array(repeating: Float(1), count: 4) + Array(repeating: Float(2), count: 4),
            [1, 2, 4]
        )

        let output = layer(input)
        eval(output)

        XCTAssertEqual(output.shape, [1, 2, 1])
        XCTAssertEqual(output[0, 0, 0].item(Float.self), 15, accuracy: 0.001)
        XCTAssertEqual(output[0, 1, 0].item(Float.self), 25, accuracy: 0.001)
    }

    func testSDNQWeightsLoaderReplacesLinearLeaves() throws {
        let model = TinyLinearModel()
        let arrays: [String: MLXArray] = [
            "proj.weight": MLXArray([UInt8(0x10), UInt8(0x32), UInt8(0xFF), UInt8(0x00)], [4]),
            "proj.scale": MLXArray([Float(2), Float(3), Float(1), Float(2)], [2, 2, 1]),
            "proj.zero_point": MLXArray([Float(1), Float(-1), Float(0), Float(0)], [2, 2, 1]),
            "proj.bias": MLXArray([Float(5), Float(-5)], [2]),
        ]

        try SDNQWeightsLoader.applyWeightsFromArrays(arrays, to: model, dtype: .float32)

        let input = MLXArray(Array(repeating: Float(1), count: 4), [1, 4])
        let output = model(input)
        eval(output)

        XCTAssertEqual(output[0, 0].item(Float.self), 22, accuracy: 0.001)
        XCTAssertEqual(output[0, 1].item(Float.self), 25, accuracy: 0.001)
    }

    func testAsymmetricUInt4EmbeddingGathersPackedRowsBeforeDequantizing() {
        let layer = SDNQUInt4Embedding(
            weight: MLXArray(
                [UInt8(0x21), UInt8(0x43), UInt8(0x65), UInt8(0x87), UInt8(0xA9), UInt8(0xCB)],
                [6]
            ),
            scales: MLXArray(Array(repeating: Float(1), count: 6), [3, 2, 1]),
            zeroPoints: MLXArray(Array(repeating: Float(0), count: 6), [3, 2, 1]),
            embeddingCount: 3,
            dimensions: 4,
            groupSize: 2,
            outputDType: .float32
        )
        let tokenIds = MLXArray([Int32(2), Int32(0)], [1, 2])

        let output = layer(tokenIds)
        eval(output)

        XCTAssertEqual(output.shape, [1, 2, 4])
        XCTAssertEqual(output[0, 0, 0].item(Float.self), 9, accuracy: 0.001)
        XCTAssertEqual(output[0, 0, 3].item(Float.self), 12, accuracy: 0.001)
        XCTAssertEqual(output[0, 1, 0].item(Float.self), 1, accuracy: 0.001)
        XCTAssertEqual(output[0, 1, 3].item(Float.self), 4, accuracy: 0.001)
    }

    func testSDNQWeightsLoaderReplacesEmbeddingLeaves() throws {
        let model = TinyEmbeddingModel()
        let arrays: [String: MLXArray] = [
            "embed.weight": MLXArray(
                [UInt8(0x21), UInt8(0x43), UInt8(0x65), UInt8(0x87), UInt8(0xA9), UInt8(0xCB)],
                [6]
            ),
            "embed.scale": MLXArray(Array(repeating: Float(1), count: 6), [3, 2, 1]),
            "embed.zero_point": MLXArray(Array(repeating: Float(0), count: 6), [3, 2, 1]),
        ]

        try SDNQWeightsLoader.applyWeightsFromArrays(arrays, to: model, dtype: .float32)

        let output = model(MLXArray([Int32(1)], [1, 1]))
        eval(output)

        XCTAssertEqual(output[0, 0, 0].item(Float.self), 5, accuracy: 0.001)
        XCTAssertEqual(output[0, 0, 3].item(Float.self), 8, accuracy: 0.001)
    }

    func testSDNQWeightsLoaderDequantizesConv2dWeights() throws {
        let model = TinyConvModel()
        let arrays: [String: MLXArray] = [
            "conv.weight": MLXArray([UInt8(0x21), UInt8(0x43)], [2]),
            "conv.scale": MLXArray([Float(1), Float(1)], [1, 2, 1, 1, 1]),
            "conv.zero_point": MLXArray([Float(0), Float(0)], [1, 2, 1, 1, 1]),
            "conv.bias": MLXArray([Float(5)], [1]),
        ]

        try SDNQWeightsLoader.applyWeightsFromArrays(arrays, to: model, dtype: .float32)

        let output = model(MLXArray([Float(1), Float(1), Float(1), Float(1)], [1, 1, 1, 4]))
        eval(output)

        XCTAssertEqual(output.shape, [1, 1, 1, 1])
        XCTAssertEqual(output[0, 0, 0, 0].item(Float.self), 15, accuracy: 0.001)
    }
}
