import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor
#if os(macOS)
import Metal

final class Q35NativeFullPreworkTests: MLXTestCase {
    func testNormalizationRotaryAndValueLayoutMatchMLX() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(941)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        let program = try Q35NativeFullPrework(device: device)
        let qw = MLXRandom.uniform(0.5..<1.5, [256]).asType(.bfloat16)
        let kw = MLXRandom.uniform(0.5..<1.5, [256]).asType(.bfloat16)
        let qWeight = try M5NativeArrayBuffer(qw, device: device), kWeight = try M5NativeArrayBuffer(kw, device: device)
        let rope = RoPE(dimensions: 64, traditional: false, base: 10_000_000)
        func buffer(_ count: Int) throws -> any MTLBuffer {
            try XCTUnwrap(device.makeBuffer(length: count * 2, options: .storageModeShared))
        }
        for rows in [1, 3, 7, 8, 15, 16] {
            let tree = try Q35VerificationTree(parents: [-1] + (1..<rows).map { ($0 - 1) / 2 })
            let plan = try Q35NativeTreePlan(tree: tree, device: device)
            let qkv = MLXRandom.uniform(-4..<4, [1, 16, 14_336]).asType(.bfloat16)
            MLX.eval(qkv)
            let source = try XCTUnwrap(qkv.asMTLBuffer(device: device))
            let q = MLXFast.rmsNorm(qkv[0..., 0..<rows, 0..<12_288].reshaped(1, rows, 24, 512)[.ellipsis, 0..<256], weight: qw, eps: 1e-6)
            let k = MLXFast.rmsNorm(qkv[0..., 0..<rows, 12_288..<13_312].reshaped(1, rows, 4, 256), weight: kw, eps: 1e-6)
            let v = qkv[0..., 0..<rows, 13_312...].reshaped(1, rows, 4, 256)
            for prefix in [0, 1023, 4095, 8192, 65536] {
                let positions = MLXArray(tree.depths.map { Int32(prefix + $0) })
                let expectedQ = rope(q.transposed(1, 2, 0, 3), offset: positions).transposed(2, 0, 1, 3)
                let expectedK = rope(k.transposed(1, 2, 0, 3), offset: positions).transposed(2, 1, 0, 3)
                let expectedV = v.transposed(0, 2, 1, 3)
                MLX.eval(expectedQ, expectedK, expectedV)
                let queries = try buffer(16 * 24 * 256), keys = try buffer(4 * 16 * 256), values = try buffer(4 * 16 * 256)
                let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                program.encode(into: encoder, qkv: source, queryWeight: qWeight, keyWeight: kWeight,
                    queries: queries, keys: keys, values: values, plan: plan, prefix: prefix, epsilon: 1e-6)
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                XCTAssertNil(command.error)
                let actualQ = MLXArray(rawPointer: queries.contents(), [1, rows, 24, 256], dtype: .bfloat16) { _ = queries }
                let actualK = MLXArray(rawPointer: keys.contents(), [1, 4, 16, 256], dtype: .bfloat16) { _ = keys }[0..., 0..., 0..<rows, 0...]
                let actualV = MLXArray(rawPointer: values.contents(), [1, 4, 16, 256], dtype: .bfloat16) { _ = values }[0..., 0..., 0..<rows, 0...]
                for (actual, expected) in [(actualQ, expectedQ), (actualK, expectedK), (actualV, expectedV)] {
                    XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0,
                        "rows=\(rows), prefix=\(prefix)")
                }
            }
        }
    }
}
#endif
