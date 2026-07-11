import MLX
import MLXFast
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore

final class GLM47AccelerationTests: MereRunCoreTestCase {
    func testCompressedMLAIsExplicitOrThresholdGuarded() {
        let dimensions = (rank: 512, nope: 128, rope: 64, value: 128)
        XCTAssertFalse(GLM47CompressedMLAPolicy.isEnabled(
            promptTokenCount: 8_192,
            dimensions: dimensions,
            environment: [:]
        ))
        XCTAssertTrue(GLM47CompressedMLAPolicy.isEnabled(
            promptTokenCount: 1,
            dimensions: dimensions,
            environment: ["MERERUN_PSI_COMPRESSED_MLA": "1"]
        ))
        XCTAssertFalse(GLM47CompressedMLAPolicy.isEnabled(
            promptTokenCount: 1_999,
            dimensions: dimensions,
            environment: [
                "MERERUN_PSI_COMPRESSED_MLA": "auto",
                "MERERUN_PSI_COMPRESSED_MLA_MIN_PROMPT_TOKENS": "2000",
            ]
        ))
        XCTAssertTrue(GLM47CompressedMLAPolicy.isEnabled(
            promptTokenCount: 2_000,
            dimensions: dimensions,
            environment: [
                "MERERUN_PSI_COMPRESSED_MLA": "auto",
                "MERERUN_PSI_COMPRESSED_MLA_MIN_PROMPT_TOKENS": "2000",
            ]
        ))
    }

    func testCompressedMLAStructuralCacheReduction() {
        let elements = GLM47CompressedMLAPolicy.elementsPerToken(
            heads: 32,
            rank: 512,
            nope: 128,
            rope: 64,
            value: 128
        )
        XCTAssertEqual(elements.expanded, 10_240)
        XCTAssertEqual(elements.compressed, 1_088)
        XCTAssertGreaterThan(Double(elements.expanded) / Double(elements.compressed), 9.4)
    }

    func testAbsorbedMLAMatchesExpandedAlgebraInFloat32() {
        MLXRandom.seed(7101)
        let heads = 2
        let queryCount = 3
        let tokenCount = 3
        let nope = 4
        let rope = 2
        let rank = 3
        let value = 4
        let scale = 1 / sqrt(Float(nope + rope))

        let qNope = MLXRandom.uniform(low: -0.5, high: 0.5, [1, heads, queryCount, nope])
        let qPe = MLXRandom.uniform(low: -0.5, high: 0.5, [1, heads, queryCount, rope])
        let latent = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 1, tokenCount, rank])
        let ropeKeys = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 1, tokenCount, rope])
        let keyWeight = MLXRandom.uniform(low: -0.5, high: 0.5, [heads, nope, rank])
        let valueWeight = MLXRandom.uniform(low: -0.5, high: 0.5, [heads, value, rank])

        let absorbed = GLM47AbsorbedMLAWeights(key: keyWeight, value: valueWeight).attend(
            qNope: qNope,
            qPe: qPe,
            latentKeys: latent,
            ropeKeys: ropeKeys,
            latentValues: latent,
            scale: scale,
            mask: .causal
        )

        let repeatedLatent = MLX.repeated(latent, count: heads, axis: 1)
        let expandedKeys = MLX.matmul(repeatedLatent, keyWeight.swappedAxes(-1, -2))
        let expandedValues = MLX.matmul(repeatedLatent, valueWeight.swappedAxes(-1, -2))
        let baseline = MLXFast.scaledDotProductAttention(
            queries: concatenated([qNope, qPe], axis: -1),
            keys: concatenated(
                [expandedKeys, MLX.repeated(ropeKeys, count: heads, axis: 1)],
                axis: -1
            ),
            values: expandedValues,
            scale: scale,
            mask: .causal
        )
        MLX.eval(absorbed, baseline)

        let maxError = MLX.max(MLX.abs(absorbed - baseline)).item(Float.self)
        XCTAssertLessThan(maxError, 0.000_01)
    }

    func testFusedMoEPolicyRequiresExplicitOptIn() {
        XCTAssertFalse(GLM47FusedMoEPolicy.isEnabled(environment: [:]))
        XCTAssertTrue(GLM47FusedMoEPolicy.isEnabled(environment: ["MERERUN_PSI_FUSED_MOE": "on"]))
    }

    func testSwitchLinearStartsWithPackedStorage() {
        let linear = GLM47SwitchLinear(
            inputDims: 64,
            outputDims: 32,
            numExperts: 4,
            groupSize: 32,
            bits: 4,
            bias: false
        )
        XCTAssertEqual(linear.weight.dtype, .uint32)
        XCTAssertEqual(linear.weight.shape, [4, 32, 8])
    }

    func testFusedGateUpMatchesSeparateQuantizedGathers() throws {
        MLXRandom.seed(7102)
        let glu = GLM47SwitchGLU(
            inputDims: 64,
            hiddenDims: 64,
            numExperts: 4,
            groupSize: 32,
            bits: 4
        )
        try installQuantized(
            MLXRandom.uniform(low: -0.25, high: 0.25, [4, 64, 64]),
            into: glu.gateProj
        )
        try installQuantized(
            MLXRandom.uniform(low: -0.25, high: 0.25, [4, 64, 64]),
            into: glu.upProj
        )
        try installQuantized(
            MLXRandom.uniform(low: -0.25, high: 0.25, [4, 64, 64]),
            into: glu.downProj
        )

        let input = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 4, 64])
        let indices = MLXArray([Int32](
            arrayLiteral: 0, 1, 2, 3, 1, 2, 3, 0
        )).reshaped(1, 4, 2)
        let separate = glu.unfusedForTesting(input, indices: indices)
        let fused = try XCTUnwrap(glu.fusedForTesting(input, indices: indices))
        MLX.eval(separate, fused)

        XCTAssertEqual(fused.shape, separate.shape)
        let maxError = MLX.max(MLX.abs(fused - separate)).item(Float.self)
        XCTAssertLessThan(maxError, 0.000_01)
    }

    private func installQuantized(_ dense: MLXArray, into linear: GLM47SwitchLinear) throws {
        let quantized = MLX.quantized(dense, groupSize: 32, bits: 4, mode: .affine)
        var updates: [(String, MLXArray)] = [
            ("weight", quantized.wq),
            ("scales", quantized.scales),
        ]
        if let biases = quantized.biases {
            updates.append(("biases", biases))
        }
        try linear.update(
            parameters: ModuleParameters.unflattened(updates),
            verify: .none
        )
    }
}
