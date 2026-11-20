import MLX
import MLXFast
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel

final class Q35TreeAttentionTests: MLXTestCase {
    #if canImport(Metal)
    func testIndexedPrefixAndEveryPathMatchNativeSerialExactly() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Indexed tree attention requires Metal.")
        MLXRandom.seed(778)
        for parents in [Array(-1..<15), [-1, 0, 0, 1, 2, 3, 4, 5, 2, 8, 7, 6, 9, 10, 11, 12]] {
            let tree = try Q35VerificationTree(parents: parents)
            for prefix in [0, 54, 1_016, 1_023, 1_024, 4_088, 8_185, 16_375, 32_760, 65_530] {
                let q = MLXRandom.normal([1, 24, tree.count, 256]).asType(.bfloat16)
                // Unused allocation rows between heads exercise strided shared KV.
                let full = MLXRandom.normal([1, 4, prefix + tree.count + 19, 256]).asType(.bfloat16)
                let values = MLXRandom.normal(full.shape).asType(.bfloat16)
                let k = full[0..., 0..., ..<(prefix + tree.count), 0...]
                let v = values[0..., 0..., ..<(prefix + tree.count), 0...]
                let actual = try XCTUnwrap(Q35IndexedTreeAttention.$enabled.withValue(true) {
                    Q35IndexedTreeAttention.apply(q: q, k: k, v: v, tree: tree, scale: 0.0625)
                })
                eval(actual)
                for row in parents.indices {
                    let indices = Array(0..<prefix) + tree.paths[row].map { prefix + $0 }
                    let expected = MLXFast.scaledDotProductAttention(
                        queries: q[0..., 0..., row..<(row + 1), 0...],
                        keys: Q35VerificationTree.take(k, rows: indices, axis: 2),
                        values: Q35VerificationTree.take(v, rows: indices, axis: 2),
                        scale: 0.0625, mask: .none)
                    let error = (actual[0..., 0..., row..<(row + 1), 0...].asType(.float32)
                                 - expected.asType(.float32)).abs().max().item(Float.self)
                    XCTAssertEqual(error, 0, "prefix=\(prefix), row=\(row), parents=\(parents)")
                }
            }
        }
    }
    #endif

    func testPaddedTreeRowsMatchIndependentSerialAttentionExactly() throws {
        MLXRandom.seed(729)
        let gpu = Device.defaultDevice().deviceType == .gpu
        let topologies = [Array(-1..<15), [-1, 0, 0, 1, 2, 3, 4, 5, 2, 8, 7, 6, 9, 10, 11, 12]]
        let prefixes = gpu ? [0, 54, 60, 1016, 1023, 1024, 4088, 4095, 8185, 8192, 16_375, 32_760, 65_530] : [0, 54]
        for parents in topologies {
            let tree = try Q35VerificationTree(parents: parents)
            for (heads, kvHeads, width) in [(24, 4, 256), (8, 1, 128)] {
                for prefix in prefixes {
                    let q = MLXRandom.normal([1, heads, tree.count, width]).asType(.bfloat16)
                    let k = MLXRandom.normal([1, kvHeads, prefix + tree.count, width]).asType(.bfloat16)
                    let v = MLXRandom.normal(k.shape).asType(.bfloat16)
                    let scale = 1 / sqrt(Float(width))
                    let actual = q35TreeAttention(q: q, k: k, v: v, tree: tree, scale: scale)
                    eval(actual)
                    for row in parents.indices {
                        let indices = Array(0..<prefix) + tree.paths[row].map { prefix + $0 }
                        let expected = MLXFast.scaledDotProductAttention(
                            queries: q[0..., 0..., row..<(row + 1), 0...],
                            keys: Q35VerificationTree.take(k, rows: indices, axis: 2),
                            values: Q35VerificationTree.take(v, rows: indices, axis: 2),
                            scale: scale, mask: .none
                        )
                        let error = (actual[0..., 0..., row..<(row + 1), 0...].asType(.float32)
                                     - expected.asType(.float32)).abs().max().item(Float.self)
                        XCTAssertEqual(error, 0, "heads=\(heads), prefix=\(prefix), row=\(row), parents=\(parents)")
                    }
                }
            }
        }
    }
}
