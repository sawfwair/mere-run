import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunGemmaModel

final class Gemma4HeadNormRoPETests: MLXTestCase {
    func testFusedHeadsPreserveStagedRoundingAndRotaryLayout() throws {
        guard Gemma4MetalExperts.isSupported else { throw XCTSkip("Requires Metal GPU.") }
        MLXRandom.seed(872)
        for (dim, kvHeads, proportional) in [(256, 8, false), (512, 2, true)] {
            let heads = 16, base: Float = proportional ? 1_000_000 : 10_000
            let rotated = proportional ? 128 : dim
            let fused = try Gemma4HeadNormRoPE(heads: heads, kvHeads: kvHeads, dim: dim,
                rotatedDims: rotated, proportional: proportional, base: base, eps: 1e-6)
            let rope: any OffsetLayer = proportional
                ? Gemma4ProportionalRoPE(dims: dim, base: base, partialRotaryFactor: 0.25)
                : RoPE(dimensions: dim, traditional: false, base: base)
            for (batch, count) in [(1, 1), (1, 7), (2, 16)] {
                // Gaps between batch rows exercise the kernel's contiguous-input adapter.
                let q = MLXRandom.normal([batch, count + 1, heads * dim]).asType(.bfloat16)[0..., ..<count, 0...]
                let k = MLXRandom.normal([batch, count + 1, kvHeads * dim]).asType(.bfloat16)[0..., ..<count, 0...]
                let v = MLXRandom.normal([batch, count + 1, kvHeads * dim]).asType(.bfloat16)[0..., ..<count, 0...]
                let qw = MLXRandom.uniform(low: 0.1, high: 2, [dim]).asType(.bfloat16)
                let kw = MLXRandom.uniform(low: 0.1, high: 2, [dim]).asType(.bfloat16)
                for offset in [0, 779, 1024, 65_537] {
                    let result = try XCTUnwrap(fused(queries: q, keys: k, values: v, queryWeight: qw, keyWeight: kw,
                                                   offset: MLXArray([Int32(offset)])))
                    let expected = [
                        rope(MLXFast.rmsNorm(q.reshaped(batch, count, heads, dim), weight: qw, eps: 1e-6).transposed(0, 2, 1, 3), offset: offset),
                        rope(MLXFast.rmsNorm(k.reshaped(batch, count, kvHeads, dim), weight: kw, eps: 1e-6).transposed(0, 2, 1, 3), offset: offset),
                        gemma4RMSNormNoScale(v.reshaped(batch, count, kvHeads, dim), eps: 1e-6).transposed(0, 2, 1, 3)
                    ]
                    for (actual, reference) in zip([result.0, result.1, result.2], expected) {
                        XCTAssertEqual(actual.shape, reference.shape)
                        XCTAssertEqual(abs(actual - reference).max().item(Float.self), 0,
                                       "dim=\(dim), count=\(count), offset=\(offset)")
                    }
                }
            }
        }
    }

    func testCurrentParametersAndPrefillFallback() throws {
        guard Gemma4MetalExperts.isSupported else { throw XCTSkip("Requires Metal GPU.") }
        let fused = try Gemma4HeadNormRoPE(heads: 2, kvHeads: 1, dim: 256,
                                         rotatedDims: 256, proportional: false, base: 10_000, eps: 1e-6)
        let q = MLXRandom.normal([1, 16, 512]).asType(.bfloat16)
        let k = MLXRandom.normal([1, 16, 256]).asType(.bfloat16)
        let weight = MLXArray.ones([256], dtype: .bfloat16), zero = MLXArray.zeros([256], dtype: .bfloat16)
        let position = MLXArray([Int32(23)])
        let initial = try XCTUnwrap(fused(queries: q, keys: k, values: k, queryWeight: weight, keyWeight: weight, offset: position))
        XCTAssertGreaterThan(abs(initial.0).max().item(Float.self), 0)
        let updated = try XCTUnwrap(fused(queries: q, keys: k, values: k, queryWeight: zero, keyWeight: zero, offset: position))
        XCTAssertEqual(abs(updated.0).max().item(Float.self), 0)
        XCTAssertEqual(abs(updated.1).max().item(Float.self), 0)
        XCTAssertEqual(abs(updated.2 - initial.2).max().item(Float.self), 0)
        XCTAssertNil(fused(queries: q.asType(.float32), keys: k, values: k, queryWeight: weight, keyWeight: weight, offset: position))
        XCTAssertNil(fused(queries: q, keys: k, values: k, queryWeight: weight.asType(.float32), keyWeight: weight, offset: position))
        XCTAssertNil(fused(queries: MLXArray.zeros([1, 17, 512], dtype: .bfloat16),
                           keys: MLXArray.zeros([1, 17, 256], dtype: .bfloat16), values: MLXArray.zeros([1, 17, 256], dtype: .bfloat16),
                           queryWeight: weight, keyWeight: weight, offset: position))
    }
}
