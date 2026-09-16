import XCTest
import MLX
import MLXRandom
@testable import MereRunGemmaModel
import MereRunMLXTestSupport

final class Gemma4QuantizedTensorRangeTests: MLXTestCase {
    func testPaddedAndInteriorRangesMatchCompactDequantization() throws {
        MLXRandom.seed(45)
        for dtype: DType in [.bfloat16, .float32] {
            for bits in [2, 4, 8] {
                for groupSize in [32, 64] {
                    let source = MLXRandom.uniform(-1.0 ..< 1.0, [2, 2, 520, 128]).asType(dtype)
                    var state = Gemma4QuantizedTensorState(
                        source: source[0..., 0..., 0..<7, 0...], groupSize: groupSize, bits: bits
                    )
                    for range in [7..<256, 256..<257, 257..<519, 519..<520] {
                        state = state.appending(source[0..., 0..., range, 0...])
                        MLX.eval(state.weight, state.scales, try XCTUnwrap(state.biases))
                        for readRange in [0..<range.upperBound, 3..<range.upperBound, range] {
                            let reference = Gemma4QuantizedTensorState(
                                source: MLX.contiguous(source[0..., 0..., readRange, 0...]),
                                groupSize: groupSize, bits: bits
                            ).dequantized()
                            let actual = state.dequantized(tokenRange: readRange)
                            XCTAssertEqual(actual.shape, [2, 2, readRange.count, 128])
                            XCTAssertTrue(
                                MLX.arrayEqual(actual, reference).item(Bool.self),
                                "dtype=\(dtype), bits=\(bits), group=\(groupSize), range=\(readRange)"
                            )
                        }
                    }
                    XCTAssertGreaterThan(state.tokenCapacity, state.tokenCount)
                }
            }
        }
    }
}
