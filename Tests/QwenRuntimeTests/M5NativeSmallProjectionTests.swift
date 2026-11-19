import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunTensor
#if os(macOS)
import Metal

final class M5NativeSmallProjectionTests: MLXTestCase {
    func testGateProjectionsPreserveSerialQMVAtEveryWidth() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(936)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        for n in [48, 96, 16_384] {
            let weights = MLXRandom.randInt(0..<UInt32.max, [n, 640])
            let scales = MLXRandom.uniform(0.001..<0.01, [n, 80]).asType(.bfloat16)
            let biases = MLXRandom.uniform(-0.1..<0.1, [n, 80]).asType(.bfloat16)
            let prepared = try M5NativeSmallProjection.Weights(weight: weights, scales: scales, biases: biases, device: device)
            let program = try M5NativeSmallProjection(device: device, inputSize: 5_120, outputSize: n)
            for factor in [Float(1), 65_536] {
                let x = (MLXRandom.uniform(-1..<1, [1, 16, 5_120]) * factor).asType(.bfloat16)
                let expected = MLX.concatenated((0..<16).map { row in
                    MLX.quantizedMM(x[0..., row..<(row + 1), 0...], weights,
                        scales: scales, biases: biases, groupSize: 64, bits: 4)
                }, axis: 1)
                MLX.eval(expected)
                let input = try M5NativeArrayBuffer(x, device: device)
                let output = try XCTUnwrap(device.makeBuffer(length: 16 * n * 2, options: .storageModeShared))
                for rows in 1...16 {
                    let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                    program.encode(into: encoder, input: input.buffer, output: output,
                        weights: prepared, rows: rows, inputOffset: input.offset)
                    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                    XCTAssertNil(command.error)
                    let actual = MLXArray(rawPointer: output.contents(), [1, rows, n], dtype: .bfloat16) { _ = output }
                    XCTAssertEqual((actual.asType(.float32) - expected[0..., 0..<rows, 0...].asType(.float32))
                        .abs().max().item(Float.self), 0, "N=\(n), rows=\(rows), factor=\(factor)")
                }
            }
        }
    }
}
#endif
