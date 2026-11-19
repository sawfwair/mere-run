import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor
#if os(macOS)
import Metal

final class Q35NativeFullOutputTests: MLXTestCase {
    func testBF16GateAndSerialGroupSumsMatchMLX() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(943)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        let program = try Q35NativeFullOutput(device: device)
        for rows in [1, 3, 7, 8, 15, 16] {
            let attention = MLXRandom.uniform(-4..<4, [1, rows, 6144]).asType(.bfloat16)
            let qkv = MLXRandom.uniform(-8..<8, [1, 16, 14336]).asType(.bfloat16)
            let gate = qkv[0..., 0..<rows, 0..<12288].reshaped(1, rows, 24, 512)[.ellipsis, 256...].reshaped(1, rows, 6144)
            let expected = attention * MLX.sigmoid(gate)
            MLX.eval(attention, qkv, expected)
            let input = try XCTUnwrap(attention.asMTLBuffer(device: device)), projected = try XCTUnwrap(qkv.asMTLBuffer(device: device))
            let output = try XCTUnwrap(device.makeBuffer(length: 16 * 6144 * 2, options: .storageModeShared))
            let sums = try XCTUnwrap(device.makeBuffer(length: 96 * 16 * 4, options: .storageModeShared))
            let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            program.encode(into: encoder, attention: input, qkv: projected, output: output, sums: sums, rows: rows)
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            XCTAssertNil(command.error)
            let actual = MLXArray(rawPointer: output.contents(), expected.shape, dtype: .bfloat16) { _ = output }
            XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0)
            let values = expected.asType(.float32).asArray(Float.self)
            var expectedSums = Array(repeating: Float(0), count: 96 * 16)
            for group in 0..<96 {
                for row in 0..<rows {
                    var total = Float(0)
                    for column in 0..<64 { total += values[row * 6144 + group * 64 + column] }
                    expectedSums[group * 16 + row] = total
                }
            }
            let actualSums = MLXArray(rawPointer: sums.contents(), [96, 16], dtype: .float32) { _ = sums }
            XCTAssertEqual(actualSums.asArray(Float.self), expectedSums)
        }
    }
}
#endif
