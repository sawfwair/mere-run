import Foundation
import XCTest
import MLX
import MLXFast
import MLXNN
import MLXRandom
@testable import MereRunCore

final class Gemma4DecodeFusedKernelsTests: MereRunCoreTestCase {
    private func skipUnlessGPUForFusedDecodeKernels() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Gemma4 fused decode kernels use MLXFast Metal kernels; set MERERUN_TEST_MLX_DEVICE=gpu to run them.")
        }
    }

    private func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// The kernels accumulate in float32 and round once, while the op-by-op
    /// reference rounds to bfloat16 at every step — so differences scale with
    /// magnitude (one bf16 ULP). Compare relative to |reference| + 1: rounding
    /// order lands around 1e-3 while any indexing corruption is O(1).
    private func maxRelativeDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        let lhs = a.asType(.float32)
        let rhs = b.asType(.float32)
        return (MLX.abs(lhs - rhs) / (MLX.abs(rhs) + 1)).max().item(Float.self)
    }

    func testQKVNormsMatchesReference() throws {
        try skipUnlessGPUForFusedDecodeKernels()

        let numHeads = 4
        let numKVHeads = 2
        let headDim = 64
        let eps: Float = 1e-6
        let batch = 2
        let total = (numHeads + 2 * numKVHeads) * headDim

        MLXRandom.seed(7)
        let qkv = MLXRandom.normal([batch, 1, total]).asType(.bfloat16)
        let qWeight = MLXRandom.normal([headDim]).asType(.bfloat16)
        let kWeight = MLXRandom.normal([headDim]).asType(.bfloat16)

        let (q, k, v) = Gemma4DecodeFusedKernels.qkvNorms(
            qkv: qkv,
            qNormWeight: qWeight,
            kNormWeight: kWeight,
            eps: Gemma4DecodeScalarCache.epsilon(eps),
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim
        )

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim
        let rawQ = qkv[0..., 0..., 0..<qDim].reshaped(batch, 1, numHeads, headDim)
        let rawK = qkv[0..., 0..., qDim..<(qDim + kvDim)].reshaped(batch, 1, numKVHeads, headDim)
        let rawV = qkv[0..., 0..., (qDim + kvDim)...].reshaped(batch, 1, numKVHeads, headDim)

        let expectedQ = MLXFast.rmsNorm(rawQ, weight: qWeight, eps: eps).transposed(0, 2, 1, 3)
        let expectedK = MLXFast.rmsNorm(rawK, weight: kWeight, eps: eps).transposed(0, 2, 1, 3)
        let expectedV = MLXFast.rmsNorm(rawV, weight: MLXArray.ones([headDim]).asType(.bfloat16), eps: eps)
            .transposed(0, 2, 1, 3)

        XCTAssertEqual(q.shape, expectedQ.shape)
        XCTAssertEqual(k.shape, expectedK.shape)
        XCTAssertEqual(v.shape, expectedV.shape)
        XCTAssertLessThan(maxRelativeDifference(q, expectedQ), 0.02)
        XCTAssertLessThan(maxRelativeDifference(k, expectedK), 0.02)
        XCTAssertLessThan(maxRelativeDifference(v, expectedV), 0.02)
    }

    func testResidualDoubleNormMatchesReference() throws {
        try skipUnlessGPUForFusedDecodeKernels()

        let hidden = 1_536
        let eps: Float = 1e-6
        let batch = 2

        MLXRandom.seed(11)
        let attentionOutput = MLXRandom.normal([batch, 1, hidden]).asType(.bfloat16)
        let residual = MLXRandom.normal([batch, 1, hidden]).asType(.bfloat16)
        let postWeight = MLXRandom.normal([hidden]).asType(.bfloat16)
        let preWeight = MLXRandom.normal([hidden]).asType(.bfloat16)

        let (mlpResidual, mlpInput) = Gemma4DecodeFusedKernels.residualDoubleNorm(
            attentionOutput: attentionOutput,
            residual: residual,
            postNormWeight: postWeight,
            preNormWeight: preWeight,
            eps: Gemma4DecodeScalarCache.epsilon(eps),
            hidden: hidden
        )

        let expectedResidual = residual + MLXFast.rmsNorm(attentionOutput, weight: postWeight, eps: eps)
        let expectedInput = MLXFast.rmsNorm(expectedResidual, weight: preWeight, eps: eps)

        XCTAssertEqual(mlpResidual.shape, expectedResidual.shape)
        XCTAssertEqual(mlpInput.shape, expectedInput.shape)
        XCTAssertLessThan(maxRelativeDifference(mlpResidual, expectedResidual), 0.02)
        XCTAssertLessThan(maxRelativeDifference(mlpInput, expectedInput), 0.02)
    }

    func testGeluMulMatchesReference() throws {
        try skipUnlessGPUForFusedDecodeKernels()

        let intermediate = 2_048
        let batch = 2

        MLXRandom.seed(13)
        let gateUp = MLXRandom.normal([batch, 1, 2 * intermediate]).asType(.bfloat16)

        let activated = Gemma4DecodeFusedKernels.geluMul(gateUp: gateUp, intermediate: intermediate)

        let gate = gateUp[0..., 0..., 0..<intermediate]
        let up = gateUp[0..., 0..., intermediate...]
        let expected = geluApproximate(gate) * up

        XCTAssertEqual(activated.shape, expected.shape)
        XCTAssertLessThan(maxRelativeDifference(activated, expected), 0.02)
    }

    func testFFNResidualScaleMatchesReference() throws {
        try skipUnlessGPUForFusedDecodeKernels()

        let hidden = 1_536
        let eps: Float = 1e-6
        let batch = 2

        MLXRandom.seed(17)
        let downOutput = MLXRandom.normal([batch, 1, hidden]).asType(.bfloat16)
        let mlpResidual = MLXRandom.normal([batch, 1, hidden]).asType(.bfloat16)
        let postWeight = MLXRandom.normal([hidden]).asType(.bfloat16)
        let layerScalar = MLXArray([Float(1.25)])

        let output = Gemma4DecodeFusedKernels.ffnResidualScale(
            downOutput: downOutput,
            mlpResidual: mlpResidual,
            postNormWeight: postWeight,
            layerScalar: layerScalar,
            eps: Gemma4DecodeScalarCache.epsilon(eps),
            hidden: hidden
        )

        let expected = (mlpResidual + MLXFast.rmsNorm(downOutput, weight: postWeight, eps: eps)) * layerScalar

        XCTAssertEqual(output.shape, expected.shape)
        XCTAssertLessThan(maxRelativeDifference(output, expected), 0.02)
    }

    func testFusedProjectionMatchesSeparateProjections() {
        let input = 256
        let outputs = [192, 96, 96]

        MLXRandom.seed(19)
        let projections = outputs.map { out -> QuantizedLinear in
            let weight = MLXRandom.normal([out, input]).asType(.float16)
            return QuantizedLinear(weight: weight, bias: nil, groupSize: 64, bits: 4)
        }

        guard let fused = Gemma4FusedQuantizedProjection.fuse(projections) else {
            XCTFail("expected fusion to succeed for uniform QuantizedLinear projections")
            return
        }
        XCTAssertTrue(fused.matches(projections))

        // Gemma and Q35 retain this concatenated projection outside the module
        // tree. Replacing even one source with its production LoRA wrapper must
        // invalidate the retained layout and must not silently drop the delta by
        // building a new fusion over the wrapper.
        let adapted: [Linear?] = [
            LoRAQuantizedLinear(base: projections[0], rank: 2),
            projections[1],
            projections[2],
        ]
        XCTAssertFalse(fused.matches(adapted))
        XCTAssertNil(Gemma4FusedQuantizedProjection.fuse(adapted))

        let x = MLXRandom.normal([2, 1, input]).asType(.float16)
        let fusedParts = fused.callSplit(x)
        XCTAssertEqual(fusedParts.count, projections.count)
        for (part, projection) in zip(fusedParts, projections) {
            let expected = projection(x)
            XCTAssertEqual(part.shape, expected.shape)
            XCTAssertLessThan(maxAbsDifference(part, expected), 1e-4)
        }
    }
}
