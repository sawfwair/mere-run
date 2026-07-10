import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class PortableQuantizedMatmulTests: MereRunCoreTestCase {
    func testQ35SwitchLinearDenseExpertPathTransposesWeights() throws {
        let weight = MLXArray([
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            -1.0, 0.5, 2.0,
            0.0, -2.0, 1.0,
        ] as [Float], [2, 2, 3])
        let x = MLXArray([
            0.5, 1.0, -1.0,
            2.0, -0.5, 1.5,
        ] as [Float], [1, 2, 3])
        let indices = MLXArray([0, 1] as [Int32], [1, 2, 1])

        let layer = Q35SwitchLinear(
            inputDims: 3,
            outputDims: 2,
            numExperts: 2,
            groupSize: 64,
            bits: 4,
            quantized: false,
            bias: false
        )
        try layer.update(parameters: ModuleParameters.unflattened([("weight", weight)]), verify: .none)

        let expected = MLX.gatherMM(
            x.reshaped([2, 1, 3]),
            weight.swappedAxes(-1, -2),
            rhsIndices: MLXArray([0, 1] as [Int32], [2]),
            sortedIndices: false
        ).reshaped([1, 2, 1, 2])

        let actual = layer(x, indices: indices)
        let maxDiff = MLX.max(MLX.abs(expected.asType(.float32) - actual.asType(.float32))).item(Float.self)
        XCTAssertLessThan(maxDiff, 0.0001)
    }

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

    func testQ35SwitchLinearAcross1024TokenBoundary() throws {
        let numExperts = 16
        let inputDims = 32
        let outputDims = 16
        let topK = 8
        let denseWeight = MLXArray(
            (0..<(numExperts * outputDims * inputDims)).map {
                Float(($0 % 37) - 18) / 24.0
            },
            [numExperts, outputDims, inputDims]
        )
        let (weight, scales, biases) = MLX.quantized(
            denseWeight,
            groupSize: 32,
            bits: 4
        )
        let layer = Q35SwitchLinear(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
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

        for tokenCount in [1000, 1240] {
            let input = MLXArray(
                (0..<(tokenCount * inputDims)).map {
                    Float(($0 % 29) - 14) / 17.0
                },
                [1, tokenCount, inputDims]
            ).asType(.bfloat16)
            let indices = MLXArray(
                (0..<(tokenCount * topK)).map { Int32(($0 * 5 + 3) % numExperts) },
                [1, tokenCount, topK]
            )

            let actual = layer(input, indices: indices).asType(.float32)
            let expandedInput = MLX.expandedDimensions(input, axes: [-2, -3])
            let expected = portableGatherQuantizedMM(
                expandedInput,
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
            ).squeezed(axis: -2).asType(.float32)
            MLX.eval(actual, expected)

            let maxDiff = MLX.max(MLX.abs(actual - expected)).item(Float.self)
            XCTAssertLessThan(maxDiff, 0.02, "GatherQMM diverged at tokenCount=\(tokenCount)")
        }
    }

    func testQ35SwitchLinearSortedRouteMatchesUnsortedRoute() throws {
        let routeCount = 64
        let inputDims = 4
        let outputDims = 3
        let numExperts = 5
        let weightValues = (0..<(numExperts * outputDims * inputDims)).map {
            Float(($0 % 23) - 11) / 13.0
        }
        let inputValues = (0..<(routeCount * inputDims)).map {
            Float(($0 % 17) - 8) / 9.0
        }
        let expertIndices = (0..<routeCount).map { Int32(($0 * 3 + 2) % numExperts) }
        let flatX = MLXArray(inputValues, [routeCount, 1, inputDims])
        let indices = MLXArray(expertIndices, [routeCount])

        let layer = Q35SwitchLinear(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            groupSize: 64,
            bits: 4,
            quantized: false,
            bias: false
        )
        try layer.update(
            parameters: ModuleParameters.unflattened([
                ("weight", MLXArray(weightValues, [numExperts, outputDims, inputDims])),
            ]),
            verify: .none
        )

        let expected = layer.applyFlat(flatX, indices: indices, sortedIndices: false)
        let order = argSort(indices, axis: 0)
        let inverseOrder = argSort(order, axis: 0)
        let actual = layer.applyFlat(
            flatX.take(order, axis: 0),
            indices: indices.take(order, axis: 0),
            sortedIndices: true
        )
        .take(inverseOrder, axis: 0)

        let maxDiff = MLX.max(MLX.abs(expected.asType(.float32) - actual.asType(.float32))).item(Float.self)
        XCTAssertLessThan(maxDiff, 0.0001)
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
