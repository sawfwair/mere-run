import Foundation
import MLX
import MLXFast
import MLXRandom
import XCTest

/// Regression tests for the fused single-token-decode attention kernels.
///
/// On 2026-07-03 a stale AOT metallib (built from pre-0.30 mlx sources but
/// loaded by the mlx 0.31.x host dispatch) made `MLXFast.scaledDotProductAttention`
/// return garbage, nondeterministically, whenever the key length crossed 1024 —
/// the point where the Metal backend switches from the 1-pass to the 2-pass
/// sdpa_vector kernel on 'd'/'s'-class Apple GPUs. Every MLX text runtime
/// (Q35, Gemma4, LFM2, Psi) produced incoherent output past ~1024 tokens of
/// context while short prompts stayed clean, so nothing short caught it.
///
/// These tests pin the exact failing shape (Q35 decode: q [1,16,1,256] bf16,
/// GQA 16:2, cache-style strided k/v views) on both sides of the 1024
/// boundary, comparing against an unfused fp32 reference and requiring
/// bit-identical results across repeated runs.
final class SDPAVectorKernelTests: MereRunCoreTestCase {
    private func skipUnlessGPU() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("SDPA vector kernels are Metal-only; set MERERUN_TEST_MLX_DEVICE=gpu to run them.")
        }
    }

    /// Unfused fp32 attention with GQA expansion.
    private func referenceAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float
    ) -> MLXArray {
        let repeats = queries.dim(1) / keys.dim(1)
        let expandedKeys = MLX.repeated(keys.asType(.float32), count: repeats, axis: 1)
        let expandedValues = MLX.repeated(values.asType(.float32), count: repeats, axis: 1)
        let scores = MLX.matmul(queries.asType(.float32), expandedKeys.transposed(0, 1, 3, 2)) * scale
        return MLX.matmul(MLX.softmax(scores, axis: -1), expandedValues)
    }

    func testSingleTokenDecodeAcross1024Boundary() throws {
        try skipUnlessGPU()

        let headDim = 1 << 8
        let scale = 1.0 / Float(Double(headDim).squareRoot())

        // 1023 stays on the 1-pass kernel (control); 1100 and 3300 engage the
        // 2-pass kernel that the stale metallib corrupted.
        for keyLength in [1023, 1100, 3300] {
            MLXRandom.seed(0)
            let queries = (MLXRandom.normal([1, 16, 1, headDim]) * 0.5).asType(.bfloat16)

            // Mirror KVCacheSimple: a step-256 preallocated buffer sliced back
            // to the live length, so k/v reach the kernel as strided views.
            let capacity = ((keyLength + 255) / 256) * 256
            let keyBuffer = (MLXRandom.normal([1, 2, capacity, headDim]) * 0.5).asType(.bfloat16)
            let valueBuffer = (MLXRandom.normal([1, 2, capacity, headDim]) * 0.5).asType(.bfloat16)
            let keys = keyBuffer[0..., 0..., ..<keyLength, 0...]
            let values = valueBuffer[0..., 0..., ..<keyLength, 0...]

            var outputs: [MLXArray] = []
            for _ in 0..<3 {
                let out = MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: .none
                )
                MLX.eval(out)
                outputs.append(out.asType(.float32))
            }

            let reference = referenceAttention(queries: queries, keys: keys, values: values, scale: scale)
            MLX.eval(reference)

            let referenceError = MLX.abs(outputs[0] - reference).max().item(Float.self)
            XCTAssertFalse(
                referenceError.isNaN,
                "SDPA produced NaN at keyLength=\(keyLength) — broken/stale metallib kernel?"
            )
            // bf16 rounding lands around 2e-4; kernel corruption is O(1) or NaN.
            XCTAssertLessThanOrEqual(
                referenceError, 0.02,
                "SDPA diverged from unfused reference at keyLength=\(keyLength)"
            )

            for run in 1..<outputs.count {
                let drift = MLX.abs(outputs[run] - outputs[0]).max().item(Float.self)
                XCTAssertEqual(
                    drift, 0,
                    "SDPA nondeterministic at keyLength=\(keyLength) (run \(run) differs) — broken/stale metallib kernel?"
                )
            }
        }
    }
}
