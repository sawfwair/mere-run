import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class PortableQuantizedMatmulTests: MereRunCoreTestCase {
    #if os(macOS)
    func testQ38DraftTop32MatchesStableMLXSelection() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Qwen3.8 draft top-32 parity requires MERERUN_TEST_MLX_DEVICE=gpu."
            )
        }
        MLXRandom.seed(38)
        let logits = (MLXRandom.uniform(
            -32..<32,
            [Q35DraftRerank.realCount]
        ) * 4).round().asType(.bfloat16)
        let expected = argPartition(
            logits,
            kth: Q35DraftRerank.realCount - Q35DraftRerank.candidateCount
        )[(Q35DraftRerank.realCount - Q35DraftRerank.candidateCount)...]
            .asType(.uint32)
        let actual = Q35DraftRerank.topCandidates(logits)
        MLX.eval(expected, actual)

        XCTAssertEqual(expected.asArray(Int.self), actual.asArray(Int.self))
    }

    func testSmallBatchAffineQMVMatchesSerialRowsBitExactly() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Small-batch affine QMV parity requires MERERUN_TEST_MLX_DEVICE=gpu."
            )
        }
        MLXRandom.seed(71)
        let inputSize = 512
        for outputSize in [96, 4_096] {
            let denseWeight = MLXRandom.uniform(
                -0.2..<0.2,
                [outputSize, inputSize]
            ).asType(.bfloat16)
            let (weight, scales, optionalBiases) = MLX.quantized(
                denseWeight,
                groupSize: 64,
                bits: 4,
                mode: .affine
            )
            let biases = try XCTUnwrap(optionalBiases)

            for width in [2, 4, 7, 9] {
                let input = MLXRandom.uniform(
                    -0.5..<0.5,
                    [1, width, inputSize]
                ).asType(.bfloat16)
                let expectedRows = (0..<width).map { row in
                    MLX.quantizedMM(
                        input[0..., row..<(row + 1), 0...],
                        weight,
                        scales: scales,
                        biases: biases,
                        transpose: true,
                        groupSize: 64,
                        bits: 4,
                        mode: .affine
                    )
                }
                let expected = MLX.concatenated(expectedRows, axis: 1)
                let actual = try XCTUnwrap(SmallBatchAffineQMV.matmul(
                    input,
                    weight: weight,
                    scales: scales,
                    biases: biases,
                    groupSize: 64,
                    bits: 4,
                    mode: .affine
                ))
                MLX.eval(expected, actual)

                let maximumError = MLX.max(MLX.abs(
                    expected.asType(.float32) - actual.asType(.float32)
                )).item(Float.self)
                XCTAssertEqual(
                    maximumError,
                    0,
                    "width \(width), output size \(outputSize) changed serial QMV output"
                )
            }
        }
    }

    func testQ38WideAffineQMVShapesMatchSerialRowsBitExactly() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Wide Flash-Next affine QMV parity requires MERERUN_TEST_MLX_DEVICE=gpu."
            )
        }
        MLXRandom.seed(73)
        for (outputSize, inputSize) in [(320, 10_240), (10_240, 320)] {
            let denseWeight = MLXRandom.uniform(
                -0.2..<0.2,
                [outputSize, inputSize]
            ).asType(.bfloat16)
            let (weight, scales, optionalBiases) = MLX.quantized(
                denseWeight,
                groupSize: 64,
                bits: 4,
                mode: .affine
            )
            let biases = try XCTUnwrap(optionalBiases)

            for width in [16, 32] {
                let input = MLXRandom.uniform(
                    -0.5..<0.5,
                    [1, width, inputSize]
                ).asType(.bfloat16)
                let expected = MLX.concatenated((0..<width).map { row in
                    MLX.quantizedMM(
                        input[0..., row..<(row + 1), 0...],
                        weight,
                        scales: scales,
                        biases: biases,
                        transpose: true,
                        groupSize: 64,
                        bits: 4,
                        mode: .affine
                    )
                }, axis: 1)
                let actual = try XCTUnwrap(SmallBatchAffineQMV.matmul(
                    input,
                    weight: weight,
                    scales: scales,
                    biases: biases,
                    groupSize: 64,
                    bits: 4,
                    mode: .affine
                ))
                MLX.eval(expected, actual)

                let maximumError = MLX.max(MLX.abs(
                    expected.asType(.float32) - actual.asType(.float32)
                )).item(Float.self)
                XCTAssertEqual(maximumError, 0, "\(outputSize)x\(inputSize), width \(width)")
            }
        }
    }

    func testSmallBatchAffineGatherQMVMatchesSerialRoutesBitExactly() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Small-batch affine gather QMV parity requires MERERUN_TEST_MLX_DEVICE=gpu."
            )
        }
        MLXRandom.seed(72)
        let expertCount = 7
        let routeCount = 23
        for inputSize in [512, 640] {
            let outputSize = 96
            let denseWeight = MLXRandom.uniform(
                -0.2..<0.2,
                [expertCount, outputSize, inputSize]
            ).asType(.bfloat16)
            let (weight, scales, optionalBiases) = MLX.quantized(
                denseWeight,
                groupSize: 64,
                bits: 3,
                mode: .affine
            )
            let biases = try XCTUnwrap(optionalBiases)
            let input = MLXRandom.uniform(
                -0.5..<0.5,
                [routeCount, 1, inputSize]
            ).asType(.bfloat16)
            let indexValues = (0..<routeCount).map { UInt32(($0 * 3) % expertCount) }
            let indices = MLXArray(indexValues)
            let expected = MLX.concatenated((0..<routeCount).map { route in
                let expert = Int(indexValues[route])
                return MLX.quantizedMM(
                    input[route..<(route + 1), 0..., 0...],
                    weight[expert, 0..., 0...],
                    scales: scales[expert, 0..., 0...],
                    biases: biases[expert, 0..., 0...],
                    transpose: true,
                    groupSize: 64,
                    bits: 3,
                    mode: .affine
                )
            }, axis: 0)
            let actual = try XCTUnwrap(SmallBatchAffineGatherQMV.matmul(
                input,
                weight: weight,
                scales: scales,
                biases: biases,
                indices: indices,
                groupSize: 64,
                bits: 3
            ))
            MLX.eval(expected, actual)

            let maximumError = MLX.max(MLX.abs(
                expected.asType(.float32) - actual.asType(.float32)
            )).item(Float.self)
            XCTAssertEqual(
                maximumError,
                0,
                "K=\(inputSize) changed serial gather QMV output"
            )
        }
    }
    func testFlashNextUnalignedQuantizedProjectionsMatchSerialRows() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Flash-Next projection parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        MLXRandom.seed(90)
        for (inputSize, outputSize) in [(320, 10_240), (640, 2_560), (10_240, 4)] {
            let dense = MLXRandom.uniform(-0.2..<0.2, [outputSize, inputSize]).asType(.bfloat16)
            let (weight, scales, biases) = MLX.quantized(dense, groupSize: 64, bits: 4)
            let layer = PortableQuantizedLinear(
                weight: weight, bias: nil, scales: scales, biases: biases,
                groupSize: 64, bits: 4, mode: .affine
            )
            let counts = inputSize == 320 ? [2, 3, 4, 7, 9, 16, 32] : [2, 3, 4, 7, 9]
            for count in counts {
                let input = MLXRandom.uniform(-0.5..<0.5, [1, count, inputSize]).asType(.bfloat16)
                let serial = MLX.concatenated((0..<count).map {
                    layer(input[0..., $0..<($0 + 1), 0...])
                }, axis: 1)
                let batched = layer(input)
                let error = MLX.abs(serial.asType(.float32) - batched.asType(.float32)).max().item(Float.self)
                XCTAssertEqual(error, 0, "M=\(count), K=\(inputSize), N=\(outputSize) changed serial QMV")
            }
        }
    }

    func testFlashNextDenseProjectionsMatchSerialRows() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Flash-Next projection parity requires MERERUN_TEST_MLX_DEVICE=gpu.")
        }
        MLXRandom.seed(93)
        for (inputSize, outputSize) in [(2_560, 512), (2_560, 640), (10_240, 4), (320, 10_240)] {
            let dense = MLXRandom.uniform(-0.2..<0.2, [outputSize, inputSize]).asType(.bfloat16)
            let layer = Linear(weight: dense, bias: nil)
            for count in [2, 3, 4, 7, 9] {
                let input = MLXRandom.uniform(-0.5..<0.5, [1, count, inputSize]).asType(.bfloat16)
                let serial = MLX.concatenated((0..<count).map {
                    layer(input[0..., $0..<($0 + 1), 0...])
                }, axis: 1)
                let actual = q38SmallBatchProjection(layer, input)
                let error = MLX.abs(serial.asType(.float32) - actual.asType(.float32)).max().item(Float.self)
                XCTAssertEqual(error, 0, "M=\(count), K=\(inputSize), N=\(outputSize) changed serial GEMV")
            }
        }
    }

    func testFlashNextDenseProjectionKeepsPrefillAndBatchingNative() {
        let layer = Linear(weight: MLXArray.ones([16, 32], dtype: .bfloat16), bias: nil)
        for shape in [[1, 1, 32], [1, 33, 32], [2, 4, 32]] {
            let input = MLXRandom.uniform(-0.5..<0.5, shape).asType(.bfloat16)
            let expected = layer(input)
            let actual = q38SmallBatchProjection(layer, input)
            XCTAssertEqual(actual.shape, expected.shape)
            XCTAssertEqual((actual - expected).abs().max().item(Float.self), 0)
        }
    }
    #endif

    func testCUDAQuantModeParsingDefaultsToAutomatic() {
        XCTAssertEqual(MLXCUDAQuant.parseMode(nil), .automatic)
        XCTAssertEqual(MLXCUDAQuant.parseMode(""), .automatic)
        XCTAssertEqual(MLXCUDAQuant.parseMode("auto"), .automatic)
        XCTAssertEqual(MLXCUDAQuant.parseMode("unexpected"), .automatic)
        XCTAssertEqual(MLXCUDAQuant.parseMode("1"), .native)
        XCTAssertEqual(MLXCUDAQuant.parseMode("native"), .native)
        XCTAssertEqual(MLXCUDAQuant.parseMode("YES"), .native)
        XCTAssertEqual(MLXCUDAQuant.parseMode("0"), .dense)
        XCTAssertEqual(MLXCUDAQuant.parseMode("dense"), .dense)
        XCTAssertEqual(MLXCUDAQuant.parseMode("off"), .dense)
    }

    func testLegacyCUDAQuantPreferenceRemainsSourceCompatible() {
        XCTAssertEqual(
            MLXCUDAQuant.preferNativeQuant,
            MLXCUDAQuant.mode == .native
        )
    }

    func testCUDAQuantCapabilitiesAreScopedByPackedLayout() {
        let binary = MLXCUDAQuant.CapabilityKey(
            operation: .quantizedMM,
            bits: 1,
            groupSize: 32,
            quantizationMode: .affine
        )
        let ternary = MLXCUDAQuant.CapabilityKey(
            operation: .quantizedMM,
            bits: 2,
            groupSize: 32,
            quantizationMode: .affine
        )
        let fourBit = MLXCUDAQuant.CapabilityKey(
            operation: .quantizedMM,
            bits: 4,
            groupSize: 64,
            quantizationMode: .affine
        )

        XCTAssertNotEqual(binary, ternary)
        XCTAssertNotEqual(binary, fourBit)
        XCTAssertNotEqual(ternary, fourBit)
    }

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

    func testQ35SwitchLinearDenseBF16ExpertPathUsesSelectedBatchedMatmul() throws {
        let weight = MLXArray([
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            -1.0, 0.5, 2.0,
            0.0, -2.0, 1.0,
        ] as [Float], [2, 2, 3]).asType(.bfloat16)
        let x = MLXArray([
            0.5, 1.0, -1.0,
            2.0, -0.5, 1.5,
        ] as [Float], [1, 2, 3]).asType(.bfloat16)
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

        let actual = layer(x, indices: indices).asType(.float32)
        MLX.eval(actual)

        let expected: [Float] = [-0.5, 1.0, 0.75, 2.5]
        for (value, reference) in zip(actual.asArray(Float.self), expected) {
            XCTAssertEqual(value, reference, accuracy: 0.02)
        }
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
