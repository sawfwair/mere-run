import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor
#if os(macOS)
import Metal

final class Q35NativeFullAttentionTests: MLXTestCase {
    func testSeparatePrefixAndTreeRowsMatchSerialSDPA() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(942)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        let program = Q35NativeFullAttention(device: device)
        var scratch: Q35NativeFullAttention.Scratch?
        for prefix in [0, 7, 1022, 1023, 1024, 4094, 4095, 8191, 8192, 16383] {
            for rows in [1, 3, 8, 16] {
                let tree = try Q35VerificationTree(parents: [-1] + (1..<rows).map { ($0 - 1) / 2 })
                let plan = try Q35NativeTreePlan(tree: tree, device: device)
                let prepared = try program.prepare(plan: plan, prefix: prefix, scratch: &scratch)
                let q = MLXRandom.uniform(-1..<1, [1, rows, 24, 256]).asType(.bfloat16)
                let baseK = MLXRandom.uniform(-1..<1, [1, 4, max(1, prefix + 2), 256]).asType(.bfloat16)
                let baseV = MLXRandom.uniform(-1..<1, [1, 4, max(1, prefix + 2), 256]).asType(.bfloat16)
                let k = MLXRandom.uniform(-1..<1, [1, 4, 16, 256]).asType(.bfloat16)
                let v = MLXRandom.uniform(-1..<1, [1, 4, 16, 256]).asType(.bfloat16)
                let prefixK = baseK[0..., 0..., 1..<(prefix + 1), 0...], prefixV = baseV[0..., 0..., 1..<(prefix + 1), 0...]
                let fullK = concatenated([prefixK, k[0..., 0..., 0..<rows, 0...]], axis: 2)
                let fullV = concatenated([prefixV, v[0..., 0..., 0..<rows, 0...]], axis: 2)
                let expected = Q35IndexedTreeAttention.$enabled.withValue(false) {
                    q35TreeAttention(q: q.transposed(0, 2, 1, 3), k: fullK, v: fullV, tree: tree, scale: 1.0 / 16)
                }.transposed(0, 2, 1, 3)
                MLX.eval(q, k, v, baseK, baseV, expected)
                let query = try XCTUnwrap(q.asMTLBuffer(device: device)), branchK = try XCTUnwrap(k.asMTLBuffer(device: device))
                let branchV = try XCTUnwrap(v.asMTLBuffer(device: device))
                let keyView = try prefix == 0 ? nil : M5NativeArrayBuffer(prefixK, device: device)
                let valueView = try prefix == 0 ? nil : M5NativeArrayBuffer(prefixV, device: device)
                let output = try XCTUnwrap(device.makeBuffer(length: 16 * 24 * 256 * 2, options: .storageModeShared))
                let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                program.encode(into: encoder, queries: query, prefixKeys: keyView, prefixValues: valueView,
                    branchKeys: branchK, branchValues: branchV, output: output, prepared: prepared)
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                XCTAssertNil(command.error)
                let actual = MLXArray(rawPointer: output.contents(), expected.shape, dtype: .bfloat16) { _ = output }
                XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0,
                    "rows=\(rows), prefix=\(prefix)")
            }
        }
    }
}
#endif
