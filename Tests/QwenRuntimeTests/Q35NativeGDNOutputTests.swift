import MLX
import MLXRandom
import MLXNN
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor
#if os(macOS)
import Metal

final class Q35NativeGDNOutputTests: MLXTestCase {
    func testNormGateAndProjectionSumsMatchMLX() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(940)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        let program = try Q35NativeGDNOutput(device: device)
        let reference = Q35RMSNormGated(hiddenSize: 128, eps: 1e-6)
        try reference.update(parameters: ModuleParameters.unflattened([
            "weight": MLXRandom.uniform(0.5..<1.5, [128]).asType(.bfloat16)
        ]), verify: [.all])
        let weight = try M5NativeArrayBuffer(reference.weight, device: device)
        for rows in [1, 3, 7, 8, 15, 16] {
            let updated = MLXRandom.uniform(-1..<1, [1, rows, 48, 128]).asType(.bfloat16)
            let qkvz = MLXRandom.uniform(-4..<4, [1, 16, 16_384]).asType(.bfloat16)
            let gate = qkvz[0..., 0..<rows, 10_240...].reshaped(1, rows, 48, 128)
            let expected = reference(updated, gate: gate).reshaped(1, rows, 6_144); MLX.eval(expected)
            let input = try XCTUnwrap(updated.asMTLBuffer(device: device)), projected = try XCTUnwrap(qkvz.asMTLBuffer(device: device))
            let output = try XCTUnwrap(device.makeBuffer(length: 16 * 6_144 * 2, options: .storageModeShared))
            let sums = try XCTUnwrap(device.makeBuffer(length: 96 * 16 * 4, options: .storageModeShared))
            let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            program.encode(into: encoder, updated: input, qkvz: projected, weight: weight,
                output: output, sums: sums, rows: rows, epsilon: 1e-6)
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            XCTAssertNil(command.error)
            let actual = MLXArray(rawPointer: output.contents(), expected.shape, dtype: .bfloat16) { _ = output }
            XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0)
            let values = expected.asType(.float32).asArray(Float.self)
            var expectedSums = Array(repeating: Float(0), count: 96 * 16)
            for group in 0..<96 {
                for row in 0..<rows {
                    var total = Float(0)
                    for i in 0..<64 { total += values[row * 6_144 + group * 64 + i] }
                    expectedSums[group * 16 + row] = total
                }
            }
            let actualSums = MLXArray(rawPointer: sums.contents(), [96, 16], dtype: .float32) { _ = sums }
            XCTAssertEqual(actualSums.asArray(Float.self), expectedSums)
        }
    }
}
#endif
