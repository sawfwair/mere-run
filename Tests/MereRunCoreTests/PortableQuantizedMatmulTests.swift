import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class PortableQuantizedMatmulTests: MereRunCoreTestCase {
    func testQ35SwitchLinearInfersMixedPrecisionExpertBits() throws {
        let weightValues = (0..<384).map { Float($0 % 19) / 18.0 - 0.5 }
        let denseWeight = MLXArray(weightValues, [3, 4, 32])
        let (weight, scales, biases) = MLX.quantized(denseWeight, groupSize: 32, bits: 8)
        let x = MLXArray([
            0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70, -0.80,
            0.15, -0.25, 0.35, -0.45, 0.55, -0.65, 0.75, -0.85,
            0.12, -0.22, 0.32, -0.42, 0.52, -0.62, 0.72, -0.82,
            0.18, -0.28, 0.38, -0.48, 0.58, -0.68, 0.78, -0.88,
            -0.15, 0.25, -0.35, 0.45, -0.55, 0.65, -0.75, 0.85,
            -0.10, 0.20, -0.30, 0.40, -0.50, 0.60, -0.70, 0.80,
            -0.12, 0.22, -0.32, 0.42, -0.52, 0.62, -0.72, 0.82,
            -0.18, 0.28, -0.38, 0.48, -0.58, 0.68, -0.78, 0.88,
        ] as [Float], [1, 2, 32])
        let indices = MLXArray([0, 2] as [Int32], [1, 2, 1])

        let layer = Q35SwitchLinear(
            inputDims: 32,
            outputDims: 4,
            numExperts: 3,
            groupSize: 32,
            bits: 4,
            quantized: true,
            bias: false
        )
        var updates = [
            ("weight", weight),
            ("scales", scales),
        ]
        if let biases {
            updates.append(("biases", biases))
        }
        try layer.update(parameters: ModuleParameters.unflattened(updates), verify: .none)

        let expected = MLX.gatherQuantizedMM(
            x.reshaped([2, 1, 32]),
            weight,
            scales: scales,
            biases: biases,
            rhsIndices: MLXArray([0, 2] as [Int32], [2]),
            transpose: true,
            groupSize: 32,
            bits: 8,
            mode: .affine,
            sortedIndices: false
        ).reshaped([1, 2, 1, 4])

        let actual = layer(x, indices: indices)
        let maxDiff = MLX.max(MLX.abs(expected.asType(.float32) - actual.asType(.float32))).item(Float.self)
        XCTAssertLessThan(maxDiff, 0.001)
    }

    func testDequantizedGatherFallbackMatchesGatherQMM() {
        let weightValues = (0..<384).map { Float($0 % 17) / 16.0 - 0.5 }
        let denseWeight = MLXArray(weightValues, [3, 4, 32])
        let (weight, scales, biases) = MLX.quantized(denseWeight, groupSize: 32, bits: 4)
        let x = MLXArray([
            0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70, -0.80,
            0.15, -0.25, 0.35, -0.45, 0.55, -0.65, 0.75, -0.85,
            0.12, -0.22, 0.32, -0.42, 0.52, -0.62, 0.72, -0.82,
            0.18, -0.28, 0.38, -0.48, 0.58, -0.68, 0.78, -0.88,
            -0.15, 0.25, -0.35, 0.45, -0.55, 0.65, -0.75, 0.85,
            -0.10, 0.20, -0.30, 0.40, -0.50, 0.60, -0.70, 0.80,
            -0.12, 0.22, -0.32, 0.42, -0.52, 0.62, -0.72, 0.82,
            -0.18, 0.28, -0.38, 0.48, -0.58, 0.68, -0.78, 0.88,
        ] as [Float], [2, 1, 32])
        let indices = MLXArray([0, 2] as [Int32], [2])

        let expected = MLX.gatherQuantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: 32,
            bits: 4,
            mode: .affine,
            sortedIndices: false
        )
        let actual = portableGatherQuantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: 32,
            bits: 4,
            mode: .affine,
            sortedIndices: false,
            forceDequantizedFallback: true
        )

        let maxDiff = MLX.max(MLX.abs(expected.asType(.float32) - actual.asType(.float32))).item(Float.self)
        XCTAssertLessThan(maxDiff, 0.001)
    }
}
