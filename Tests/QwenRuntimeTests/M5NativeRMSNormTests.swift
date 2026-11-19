import MLX
import MLXFast
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunTensor
#if os(macOS)
import Metal

final class M5NativeRMSNormTests: MLXTestCase {
    func testResidualNormAndSerialGroupSumsMatchMLX() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(933)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        let program = try M5NativeRMSNorm(device: device)
        let weight = MLXRandom.uniform(0.5..<1.5, [5_120]).asType(.bfloat16)
        let weightBuffer = try XCTUnwrap(weight.asMTLBuffer(device: device))
        for rows in 1...16 {
            for wide in [false, true] {
                let factor: Float = wide ? 65_536 : 1
                let input = (MLXRandom.uniform(-1..<1, [1, rows, 5_120]) * factor).asType(.bfloat16)
                let residual = (MLXRandom.uniform(-1..<1, [1, rows, 5_120]) * factor).asType(.bfloat16)
                for includeResidual in [false, true] {
                    let expectedCombined = includeResidual ? input + residual : input
                    let expected = MLXFast.rmsNorm(expectedCombined, weight: weight, eps: 1e-6)
                    MLX.eval(expected, expectedCombined)
                    let inputBuffer = try XCTUnwrap(input.asMTLBuffer(device: device))
                    let residualBuffer = try XCTUnwrap(residual.asMTLBuffer(device: device))
                    let normalized = try XCTUnwrap(device.makeBuffer(length: 16 * 5_120 * 2, options: .storageModeShared))
                    let sums = try XCTUnwrap(device.makeBuffer(length: 80 * 16 * 4, options: .storageModeShared))
                    let command = try XCTUnwrap(queue.makeCommandBuffer())
                    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                    // Combined storage deliberately aliases the input, as a
                    // resident layer program can do once its readers finish.
                    program.encode(into: encoder, input: inputBuffer, residual: includeResidual ? residualBuffer : nil,
                        weight: weightBuffer, combined: inputBuffer, normalized: normalized,
                        sums: sums, rows: rows, epsilon: 1e-6)
                    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                    XCTAssertNil(command.error)
                    let actual = read(normalized, shape: [1, rows, 5_120], dtype: .bfloat16)
                    let combined = read(inputBuffer, shape: [1, rows, 5_120], dtype: .bfloat16)
                    XCTAssertEqual(delta(combined, expectedCombined), 0)
                    XCTAssertEqual(delta(actual, expected), 0, "rows=\(rows), wide=\(wide), residual=\(includeResidual)")
                    let values = expected.asType(.float32).asArray(Float.self)
                    var expectedSums = Array(repeating: Float(0), count: 80 * 16)
                    for group in 0..<80 {
                        for row in 0..<rows {
                            var total = Float(0)
                            for column in 0..<64 { total += values[row * 5_120 + group * 64 + column] }
                            expectedSums[group * 16 + row] = total
                        }
                    }
                    XCTAssertEqual(read(sums, shape: [80, 16], dtype: .float32).asArray(Float.self), expectedSums)
                }
            }
        }
    }

    private func read(_ buffer: any MTLBuffer, shape: [Int], dtype: DType) -> MLXArray {
        MLXArray(rawPointer: buffer.contents(), shape, dtype: dtype) { _ = buffer }
    }

    private func delta(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        (lhs.asType(.float32) - rhs.asType(.float32)).abs().max().item(Float.self)
    }
}
#endif
