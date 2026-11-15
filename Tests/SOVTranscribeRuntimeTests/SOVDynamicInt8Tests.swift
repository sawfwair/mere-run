import XCTest
import MLX
import MereRunMLXTestSupport
@testable import AudioSOVTranscribeModel

final class SOVDynamicInt8Tests: MLXTestCase {
    func testLiveQuantizationPreservesBiasAndPaddedRows() throws {
        guard SOVDynamicInt8.isSupported else { throw XCTSkip("Requires an M5 GPU.") }
        let weights = sin(MLXArray(0..<(128 * 512)).asType(.float32) * 0.031).reshaped(128, 512).asType(.bfloat16)
        let input = cos(MLXArray(0..<(67 * 512)).asType(.float32) * 0.073).reshaped(1, 67, 512).asType(.bfloat16)
        let bias = MLXArray(0..<128).asType(.bfloat16) / 128
        for (rotate, exponent) in [(false, 0), (true, 0), (false, 2), (true, 2)] {
            let kernel = try SOVDynamicInt8(weight: weights, rotate: rotate, exponent: exponent)
            let output = kernel(input, bias: bias)
            let expected = addMM(bias, input, weights.T)
            XCTAssertEqual(output.shape, [1, 67, 128])
            // Oscillatory inputs stress cancellation after rotation. Integer arithmetic
            // is checked separately against an independent CPU oracle.
            XCTAssertLessThan(((output - expected).asType(.float32).square().mean() / expected.asType(.float32).square().mean()).sqrt().item(Float.self), 0.03, "rotate=\(rotate), exponent=\(exponent)")
            let silent = kernel(MLXArray.zeros(input.shape, dtype: .bfloat16), bias: bias)
            XCTAssertEqual(abs(silent - bias).max().item(Float.self), 0)
            // A second input must repack live activations, not reuse a prior result.
            let changed = kernel(-input, bias: bias)
            XCTAssertGreaterThan(abs(changed - output).max().item(Float.self), 1)
        }
    }

    func testPackingRejectsUnsupportedDimensions() throws {
        guard SOVDynamicInt8.isSupported else { throw XCTSkip("Requires an M5 GPU.") }
        for shape in [[63, 256], [64, 257], [64, 8448]] {
            XCTAssertThrowsError(try SOVDynamicInt8(weight: MLXArray.zeros(shape, dtype: .bfloat16), rotate: false))
        }
        XCTAssertThrowsError(try SOVDynamicInt8(weight: MLXArray.zeros([64, 256]), rotate: false))
    }
}
