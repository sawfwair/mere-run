import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import XCTest
@testable import MereRunCore

final class MiniMaxH3FusedKernelTests: MereRunCoreTestCase {
    #if os(macOS)
    func testGateAttentionAndPrepareFeedForwardMatchesDecomposedGraph() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Set MERERUN_TEST_MLX_DEVICE=gpu to run the H3 fused Metal canary."
            )
        }

        let rows = 7
        let hiddenSize = 5_376
        let modulationRows = 9
        let epsilon = Float(1e-5)
        MLXRandom.seed(2_026_081_001)
        let residual = MLXRandom.uniform(
            -0.5 ..< 0.5,
            [1, rows, hiddenSize]
        ).asType(.bfloat16)
        let attentionOutput = MLXRandom.uniform(
            -0.25 ..< 0.25,
            [1, rows, hiddenSize]
        ).asType(.bfloat16)
        let normWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [hiddenSize]
        ).asType(.bfloat16)
        let modulation = MLXRandom.uniform(
            -0.2 ..< 0.2,
            [modulationRows, 6 * hiddenSize]
        ).asType(.bfloat16)
        let rowIndices = MLXArray((0..<rows).map { Int32(($0 * 4) % modulationRows) })

        let parts = MLX.split(modulation, parts: 6, axis: -1)
        func selected(_ part: Int) -> MLXArray {
            MLX.take(parts[part], rowIndices, axis: 0).expandedDimensions(axis: 0)
        }
        let residualReference = residual + selected(2) * attentionOutput
        let normalizedReference = MLXFast.rmsNorm(
            residualReference,
            weight: normWeight,
            eps: epsilon
        )
        let feedForwardReference = normalizedReference * (1 + selected(4)) + selected(3)
        let gateReference = selected(5)

        let fused = try XCTUnwrap(
            MiniMaxH3FusedKernels.gateAttentionAndPrepareFeedForward(
                residual: residual,
                attentionOutput: attentionOutput,
                normWeight: normWeight,
                modulation: modulation,
                rowIndices: rowIndices,
                eps: epsilon
            )
        )
        MLX.eval(
            residualReference,
            feedForwardReference,
            gateReference,
            fused.residual,
            fused.feedForwardInput,
            fused.feedForwardGate
        )

        XCTAssertEqual(fused.residual.shape, residualReference.shape)
        XCTAssertEqual(fused.feedForwardInput.shape, feedForwardReference.shape)
        XCTAssertEqual(fused.feedForwardGate.shape, gateReference.shape)
        XCTAssertEqual(maximumDifference(fused.residual, residualReference), 0)
        XCTAssertLessThan(maximumDifference(fused.feedForwardInput, feedForwardReference), 0.016)
        XCTAssertEqual(maximumDifference(fused.feedForwardGate, gateReference), 0)
    }

    func testGateAttentionAndPrepareFeedForwardMatchesMixedPrecisionGraph() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Set MERERUN_TEST_MLX_DEVICE=gpu to run the mixed H3 fused Metal canary."
            )
        }

        let rows = 7
        let hiddenSize = 5_376
        let modulationRows = 9
        let epsilon = Float(1e-5)
        MLXRandom.seed(2_026_081_014)
        let baseResidual = MLXRandom.uniform(-0.5 ..< 0.5, [1, rows, hiddenSize])
        let attentionOutput = MLXRandom.uniform(
            -0.25 ..< 0.25,
            [1, rows, hiddenSize]
        )
        let normWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [hiddenSize]
        ).asType(.bfloat16)
        let modulation = MLXRandom.uniform(
            -0.2 ..< 0.2,
            [modulationRows, 6 * hiddenSize]
        ).asType(.bfloat16)
        let rowIndices = MLXArray((0..<rows).map { Int32(($0 * 4) % modulationRows) })
        let parts = MLX.split(modulation, parts: 6, axis: -1)
        func selected(_ part: Int) -> MLXArray {
            MLX.take(parts[part], rowIndices, axis: 0).expandedDimensions(axis: 0)
        }

        for residualDType in [DType.bfloat16, .float32] {
            let residual = baseResidual.asType(residualDType)
            let residualReference = residual + selected(2) * attentionOutput
            let feedForwardReference = MLXFast.rmsNorm(
                residualReference,
                weight: normWeight,
                eps: epsilon
            ) * (1 + selected(4)) + selected(3)
            let fused = try XCTUnwrap(
                MiniMaxH3FusedKernels.gateAttentionAndPrepareFeedForward(
                    residual: residual,
                    attentionOutput: attentionOutput,
                    normWeight: normWeight,
                    modulation: modulation,
                    rowIndices: rowIndices,
                    eps: epsilon
                )
            )
            MLX.eval(
                residualReference,
                feedForwardReference,
                fused.residual,
                fused.feedForwardInput,
                fused.feedForwardGate
            )

            let residualError = maximumDifference(fused.residual, residualReference)
            let feedForwardError = maximumDifference(
                fused.feedForwardInput,
                feedForwardReference
            )
            print(
                "[h3-transfer] k1-mixed dtype=\(residualDType) "
                    + "residual_max_abs=\(residualError) "
                    + "feed_forward_max_abs=\(feedForwardError)"
            )
            XCTAssertLessThan(residualError, 1e-5)
            XCTAssertLessThan(feedForwardError, 1e-4)
            XCTAssertEqual(maximumDifference(fused.feedForwardGate, selected(5)), 0)
        }
    }

    func testGateAttentionAndQuantizeFeedForwardMatchesStandaloneBoundary() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Set MERERUN_TEST_MLX_DEVICE=gpu to run the H3 fused INT8 canary."
            )
        }

        let rows = 7
        let hiddenSize = 5_376
        let modulationRows = 9
        let epsilon = Float(1e-5)
        MLXRandom.seed(2_026_081_003)
        let residual = MLXRandom.uniform(
            -0.5 ..< 0.5,
            [1, rows, hiddenSize]
        ).asType(.bfloat16)
        let attentionOutput = MLXRandom.uniform(
            -0.25 ..< 0.25,
            [1, rows, hiddenSize]
        ).asType(.bfloat16)
        let normWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [hiddenSize]
        ).asType(.bfloat16)
        let modulation = MLXRandom.uniform(
            -0.2 ..< 0.2,
            [modulationRows, 6 * hiddenSize]
        ).asType(.bfloat16)
        let rowIndices = MLXArray((0..<rows).map { Int32(($0 * 4) % modulationRows) })

        let floating = try XCTUnwrap(
            MiniMaxH3FusedKernels.gateAttentionAndPrepareFeedForward(
                residual: residual,
                attentionOutput: attentionOutput,
                normWeight: normWeight,
                modulation: modulation,
                rowIndices: rowIndices,
                eps: epsilon
            )
        )
        let standalone = try XCTUnwrap(
            MiniMaxH3FusedKernels.quantizeRowsSymmetricInt8(floating.feedForwardInput)
        )
        let fused = try XCTUnwrap(
            MiniMaxH3FusedKernels.gateAttentionAndQuantizeFeedForward(
                residual: residual,
                attentionOutput: attentionOutput,
                normWeight: normWeight,
                modulation: modulation,
                rowIndices: rowIndices,
                eps: epsilon
            )
        )
        MLX.eval(
            floating.residual,
            floating.feedForwardInput,
            standalone.values,
            standalone.scales,
            fused.residual,
            fused.feedForwardInput.values,
            fused.feedForwardInput.scales
        )

        XCTAssertEqual(fused.residual.shape, [1, rows, hiddenSize])
        XCTAssertEqual(fused.feedForwardInput.values.shape, [1, rows, hiddenSize])
        XCTAssertEqual(fused.feedForwardInput.scales.shape, [rows])
        XCTAssertEqual(fused.feedForwardInput.values.dtype, .int8)
        XCTAssertEqual(fused.feedForwardInput.scales.dtype, .float32)
        XCTAssertEqual(maximumDifference(fused.residual, floating.residual), 0)
        XCTAssertEqual(
            fused.feedForwardInput.values.asArray(Int8.self),
            standalone.values.asArray(Int8.self)
        )
        XCTAssertEqual(
            maximumDifference(fused.feedForwardInput.scales, standalone.scales),
            0
        )

        let floatingValues = floating.feedForwardInput.asType(.float32).asArray(Float.self)
        let quantizedValues = fused.feedForwardInput.values.asArray(Int8.self)
        let scales = fused.feedForwardInput.scales.asArray(Float.self)
        for row in 0..<rows {
            let range = (row * hiddenSize)..<((row + 1) * hiddenSize)
            let maximum = floatingValues[range].reduce(Float.zero) {
                max($0, abs($1))
            }
            let expectedScale = maximum > 0 ? maximum / 127 : 1 / 127
            XCTAssertEqual(scales[row], expectedScale, accuracy: 1e-7)
            let quantizeScale = maximum > 0 ? 127 / maximum : 127
            for index in range {
                let rounded = Int(
                    (floatingValues[index] * quantizeScale).rounded(.toNearestOrEven)
                )
                let expected = Int8(clamping: max(-127, min(127, rounded)))
                XCTAssertEqual(quantizedValues[index], expected)
            }
        }
    }

    func testGateAttentionAndQuantizeFeedForwardUsesFiniteZeroRowScale() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Set MERERUN_TEST_MLX_DEVICE=gpu to run the H3 fused INT8 canary."
            )
        }

        let hiddenSize = 5_376
        let residual = MLXArray.zeros([1, 1, hiddenSize], dtype: .bfloat16)
        let attentionOutput = MLXArray.zeros([1, 1, hiddenSize], dtype: .bfloat16)
        let normWeight = MLXArray.ones([hiddenSize], dtype: .bfloat16)
        let modulation = MLXArray.zeros([1, 6 * hiddenSize], dtype: .bfloat16)
        let rowIndices = MLXArray([Int32(0)])
        let fused = try XCTUnwrap(
            MiniMaxH3FusedKernels.gateAttentionAndQuantizeFeedForward(
                residual: residual,
                attentionOutput: attentionOutput,
                normWeight: normWeight,
                modulation: modulation,
                rowIndices: rowIndices,
                eps: 1e-5
            )
        )
        MLX.eval(fused.residual, fused.feedForwardInput.values, fused.feedForwardInput.scales)

        XCTAssertEqual(fused.residual.asType(.float32).asArray(Float.self), [Float](
            repeating: 0,
            count: hiddenSize
        ))
        XCTAssertEqual(fused.feedForwardInput.values.asArray(Int8.self), [Int8](
            repeating: 0,
            count: hiddenSize
        ))
        XCTAssertEqual(
            fused.feedForwardInput.scales.item(Float.self),
            1 / 127,
            accuracy: 1e-8
        )
    }

    func testFusedKernelContractsRejectUnsupportedInputs() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Set MERERUN_TEST_MLX_DEVICE=gpu to run the H3 fused Metal canary."
            )
        }

        let hiddenSize = 5_376
        let residual = MLXArray.zeros([1, 1, hiddenSize], dtype: .bfloat16)
        let attentionOutput = MLXArray.zeros([1, 1, hiddenSize], dtype: .bfloat16)
        let normWeight = MLXArray.ones([hiddenSize], dtype: .bfloat16)
        let modulation = MLXArray.zeros([1, 6 * hiddenSize], dtype: .bfloat16)
        let rowIndices = MLXArray([Int32(0)])

        XCTAssertNil(MiniMaxH3FusedKernels.gateAttentionAndPrepareFeedForward(
            residual: residual,
            attentionOutput: attentionOutput.asType(.float16),
            normWeight: normWeight,
            modulation: modulation,
            rowIndices: rowIndices,
            eps: 1e-5
        ))
        XCTAssertNil(MiniMaxH3FusedKernels.gateAttentionAndQuantizeFeedForward(
            residual: residual,
            attentionOutput: attentionOutput,
            normWeight: normWeight,
            modulation: modulation,
            rowIndices: rowIndices,
            eps: 0
        ))
        XCTAssertNil(MiniMaxH3FusedKernels.quantizeRowsSymmetricInt8(
            residual.asType(.float32)
        ))

        let projected = MLXArray.zeros([1, 1, 3 * 56 * 128], dtype: .bfloat16)
        let headWeight = MLXArray.ones([128], dtype: .bfloat16)
        let rope = MLXArray.ones([1, 1, 1, 96], dtype: .bfloat16)
        XCTAssertNil(MiniMaxH3FusedKernels.prepareHeadMajorQKV(
            projected: projected,
            queryNormWeight: headWeight,
            keyNormWeight: headWeight,
            ropeCosine: rope,
            ropeSine: rope,
            eps: -.infinity
        ))
        XCTAssertNil(MiniMaxH3FusedKernels.prepareHeadMajorQKV(
            projected: projected.asType(.float32),
            queryNormWeight: headWeight,
            keyNormWeight: headWeight,
            ropeCosine: rope,
            ropeSine: rope,
            eps: 1e-5
        ))
        let invalidQKVCodes = MLXArray.zeros([1], dtype: .uint32)
        let invalidQKVParameters = MLXArray.zeros([1], dtype: .bfloat16)
        XCTAssertNil(MiniMaxH3FusedKernels.projectHeadMajorQKVAffineInt8(
            input: residual,
            weightCodes: invalidQKVCodes,
            weightScales: invalidQKVParameters,
            weightBiases: invalidQKVParameters,
            queryNormWeight: headWeight,
            keyNormWeight: headWeight,
            ropeCosine: rope,
            ropeSine: rope,
            eps: 1e-5
        ))

        let attention = MLXArray.zeros([1, 56, 1, 128], dtype: .bfloat16)
        let codes = MLXArray.zeros([5_376, 7_168 / 4], dtype: .uint32)
        let scales = MLXArray.ones([5_376, 7_168 / 64], dtype: .bfloat16)
        let biases = MLXArray.zeros(scales.shape, dtype: .bfloat16)
        XCTAssertNil(MiniMaxH3FusedKernels.projectHeadMajorAttentionAffineInt8(
            attention: attention.asType(.float16),
            weightCodes: codes,
            weightScales: scales,
            weightBiases: biases
        ))
        XCTAssertNil(MiniMaxH3FusedKernels.projectHeadMajorAttentionAffineInt8(
            attention: attention,
            weightCodes: codes.asType(.int32),
            weightScales: scales,
            weightBiases: biases
        ))

        let invalidCodes = MLXArray.zeros([1], dtype: .uint32)
        let invalidScales = MLXArray.ones([1], dtype: .bfloat16)
        XCTAssertNil(MiniMaxH3FusedKernels.projectFeedForwardInputAffineInt8SwiGLU(
            input: residual,
            weightCodes: invalidCodes,
            weightScales: invalidScales,
            weightBiases: invalidScales
        ))
        let feedForward = MLXArray.zeros([1, 1, 14_336], dtype: .bfloat16)
        XCTAssertNil(MiniMaxH3FusedKernels.projectFeedForwardOutputAffineInt8(
            input: feedForward,
            weightCodes: invalidCodes,
            weightScales: invalidScales,
            weightBiases: invalidScales
        ))
    }

    func testPrepareHeadMajorQKVMatchesDecomposedGraph() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Set MERERUN_TEST_MLX_DEVICE=gpu to run the H3 fused QKV canary."
            )
        }

        let rows = 7
        let heads = 56
        let headDimension = 128
        let rotaryDimension = 96
        let epsilon = Float(1e-5)
        MLXRandom.seed(2_026_081_005)
        let projected = MLXRandom.normal(
            [1, rows, 3 * heads * headDimension]
        ).asType(.bfloat16)
        let queryNormWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [headDimension]
        ).asType(.bfloat16)
        let keyNormWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [headDimension]
        ).asType(.bfloat16)
        let angles = MLXRandom.uniform(
            -1 ..< 1,
            [1, rows, 1, rotaryDimension]
        )
        let ropeCosine = MLX.cos(angles).asType(.bfloat16)
        let ropeSine = MLX.sin(angles).asType(.bfloat16)

        let parts = miniMaxH3SplitProjectedQKV(
            projected,
            heads: heads,
            headDimension: headDimension
        )
        let rope = MiniMaxH3RotaryEmbedding(cosine: ropeCosine, sine: ropeSine)
        let queryReference = rope.apply(MLXFast.rmsNorm(
            parts[0],
            weight: queryNormWeight,
            eps: epsilon
        )).transposed(0, 2, 1, 3).contiguous()
        let keyReference = rope.apply(MLXFast.rmsNorm(
            parts[1],
            weight: keyNormWeight,
            eps: epsilon
        )).transposed(0, 2, 1, 3).contiguous()
        let valueReference = parts[2].transposed(0, 2, 1, 3).contiguous()
        let fused = try XCTUnwrap(MiniMaxH3FusedKernels.prepareHeadMajorQKV(
            projected: projected,
            queryNormWeight: queryNormWeight,
            keyNormWeight: keyNormWeight,
            ropeCosine: ropeCosine,
            ropeSine: ropeSine,
            eps: epsilon
        ))
        MLX.eval(
            queryReference,
            keyReference,
            valueReference,
            fused.query,
            fused.key,
            fused.value
        )

        let expectedShape = [1, heads, rows, headDimension]
        XCTAssertEqual(fused.query.shape, expectedShape)
        XCTAssertEqual(fused.key.shape, expectedShape)
        XCTAssertEqual(fused.value.shape, expectedShape)
        XCTAssertLessThan(maximumDifference(fused.query, queryReference), 0.032)
        XCTAssertLessThan(maximumDifference(fused.key, keyReference), 0.032)
        XCTAssertEqual(maximumDifference(fused.value, valueReference), 0)
    }

    func testProjectHeadMajorQKVAffineInt8MatchesManagedQ8Contract() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Set MERERUN_TEST_MLX_DEVICE=gpu to run the H3 projection-direct QKV canary."
            )
        }

        let rows = 1
        let hiddenSize = 5_376
        let heads = 56
        let headDimension = 128
        let innerDimension = heads * headDimension
        let projectionWidth = 3 * innerDimension
        let rotaryDimension = 96
        let groupSize = 64
        let epsilon = Float(1e-5)
        MLXRandom.seed(2_026_081_011)
        let input = MLXRandom.uniform(
            -0.25 ..< 0.25,
            [1, rows, hiddenSize]
        ).asType(.bfloat16)
        let codes = MLXArray.full(
            [projectionWidth, hiddenSize / 4],
            values: MLXArray(UInt32(0x0403_0201))
        )
        let scales = MLXRandom.uniform(
            0.0005 ..< 0.0025,
            [projectionWidth, hiddenSize / groupSize]
        ).asType(.bfloat16)
        let biases = MLXRandom.uniform(
            -0.01 ..< 0.01,
            scales.shape
        ).asType(.bfloat16)
        let queryNormWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [headDimension]
        ).asType(.bfloat16)
        let keyNormWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [headDimension]
        ).asType(.bfloat16)
        let angles = MLXRandom.uniform(
            -1 ..< 1,
            [1, rows, 1, rotaryDimension]
        )
        let ropeCosine = MLX.cos(angles).asType(.bfloat16)
        let ropeSine = MLX.sin(angles).asType(.bfloat16)

        let projected = MLX.quantizedMM(
            input,
            codes,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: 8,
            mode: .affine
        )
        let parts = miniMaxH3SplitProjectedQKV(
            projected,
            heads: heads,
            headDimension: headDimension
        )
        let rope = MiniMaxH3RotaryEmbedding(cosine: ropeCosine, sine: ropeSine)
        let queryReference = rope.apply(MLXFast.rmsNorm(
            parts[0],
            weight: queryNormWeight,
            eps: epsilon
        )).transposed(0, 2, 1, 3).contiguous()
        let keyReference = rope.apply(MLXFast.rmsNorm(
            parts[1],
            weight: keyNormWeight,
            eps: epsilon
        )).transposed(0, 2, 1, 3).contiguous()
        let valueReference = parts[2].transposed(0, 2, 1, 3).contiguous()
        let candidate = try XCTUnwrap(
            MiniMaxH3FusedKernels.projectHeadMajorQKVAffineInt8(
                input: input,
                weightCodes: codes,
                weightScales: scales,
                weightBiases: biases,
                queryNormWeight: queryNormWeight,
                keyNormWeight: keyNormWeight,
                ropeCosine: ropeCosine,
                ropeSine: ropeSine,
                eps: epsilon
            )
        )
        MLX.eval(
            queryReference,
            keyReference,
            valueReference,
            candidate.query,
            candidate.key,
            candidate.value
        )

        let expectedShape = [1, heads, rows, headDimension]
        XCTAssertEqual(candidate.query.shape, expectedShape)
        XCTAssertEqual(candidate.key.shape, expectedShape)
        XCTAssertEqual(candidate.value.shape, expectedShape)
        XCTAssertLessThan(maximumDifference(candidate.query, queryReference), 0.032)
        XCTAssertLessThan(maximumDifference(candidate.key, keyReference), 0.032)
        XCTAssertLessThan(maximumDifference(candidate.value, valueReference), 0.032)

        let floatRope = MiniMaxH3RotaryEmbedding(
            cosine: MLX.cos(angles),
            sine: MLX.sin(angles)
        )
        for inputDType in [DType.bfloat16, .float32] {
            let mixedInput = input.asType(inputDType)
            let mixedProjected = MLX.quantizedMM(
                mixedInput,
                codes,
                scales: scales,
                biases: biases,
                transpose: true,
                groupSize: groupSize,
                bits: 8,
                mode: .affine
            )
            let mixedParts = miniMaxH3SplitProjectedQKV(
                mixedProjected,
                heads: heads,
                headDimension: headDimension
            )
            let mixedQueryReference = floatRope.apply(MLXFast.rmsNorm(
                mixedParts[0],
                weight: queryNormWeight,
                eps: epsilon
            )).transposed(0, 2, 1, 3).contiguous()
            let mixedKeyReference = floatRope.apply(MLXFast.rmsNorm(
                mixedParts[1],
                weight: keyNormWeight,
                eps: epsilon
            )).transposed(0, 2, 1, 3).contiguous()
            let mixedValueReference = mixedParts[2]
                .transposed(0, 2, 1, 3)
                .contiguous()
            let mixedCandidate = try XCTUnwrap(
                MiniMaxH3FusedKernels.projectHeadMajorQKVAffineInt8(
                    input: mixedInput,
                    weightCodes: codes,
                    weightScales: scales,
                    weightBiases: biases,
                    queryNormWeight: queryNormWeight,
                    keyNormWeight: keyNormWeight,
                    ropeCosine: floatRope.cosine,
                    ropeSine: floatRope.sine,
                    eps: epsilon
                )
            )
            MLX.eval(
                mixedQueryReference,
                mixedKeyReference,
                mixedValueReference,
                mixedCandidate.query,
                mixedCandidate.key,
                mixedCandidate.value
            )
            let queryError = maximumDifference(
                mixedCandidate.query,
                mixedQueryReference
            )
            let keyError = maximumDifference(
                mixedCandidate.key,
                mixedKeyReference
            )
            let valueError = maximumDifference(
                mixedCandidate.value,
                mixedValueReference
            )
            print(
                "[h3-transfer] k2b-mixed dtype=\(inputDType) "
                    + "q_max_abs=\(queryError) k_max_abs=\(keyError) "
                    + "v_max_abs=\(valueError)"
            )
            XCTAssertLessThan(queryError, 0.032)
            XCTAssertLessThan(keyError, 0.032)
            XCTAssertLessThan(valueError, 0.032)
        }
    }

    func testProjectHeadMajorAttentionAffineInt8MatchesManagedQ8Contract() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Set MERERUN_TEST_MLX_DEVICE=gpu to run the H3 affine output canary."
            )
        }

        let rows = 2
        let heads = 56
        let headDimension = 128
        let inputWidth = heads * headDimension
        let outputWidth = 5_376
        let groupSize = 64
        MLXRandom.seed(2_026_081_007)
        let attention = MLXRandom.uniform(
            -0.25 ..< 0.25,
            [1, heads, rows, headDimension]
        ).asType(.bfloat16)
        let codes = MLXArray.full(
            [outputWidth, inputWidth / 4],
            values: MLXArray(UInt32(0x0403_0201))
        )
        let scales = MLXRandom.uniform(
            0.0005 ..< 0.0025,
            [outputWidth, inputWidth / groupSize]
        ).asType(.bfloat16)
        let biases = MLXRandom.uniform(
            -0.01 ..< 0.01,
            scales.shape
        ).asType(.bfloat16)
        let rowMajor = attention.transposed(0, 2, 1, 3).reshaped(
            1,
            rows,
            inputWidth
        )
        let reference = MLX.quantizedMM(
            rowMajor,
            codes,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: 8,
            mode: .affine
        )
        let candidate = try XCTUnwrap(
            MiniMaxH3FusedKernels.projectHeadMajorAttentionAffineInt8(
                attention: attention,
                weightCodes: codes,
                weightScales: scales,
                weightBiases: biases
            )
        )
        MLX.eval(reference, candidate)

        XCTAssertEqual(candidate.shape, [1, rows, outputWidth])
        XCTAssertEqual(candidate.dtype, .bfloat16)
        XCTAssertLessThan(maximumDifference(candidate, reference), 0.032)
    }

    func testProjectFeedForwardInputAffineInt8SwiGLUMatchesManagedQ8Contract() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Set MERERUN_TEST_MLX_DEVICE=gpu to run the H3 affine FC1 canary."
            )
        }

        let rows = 1
        let inputWidth = 5_376
        let outputWidth = 14_336
        let groupSize = 64
        MLXRandom.seed(2_026_081_009)
        let input = MLXRandom.uniform(
            -0.25 ..< 0.25,
            [1, rows, inputWidth]
        ).asType(.bfloat16)
        let codes = MLXArray.full(
            [2 * outputWidth, inputWidth / 4],
            values: MLXArray(UInt32(0x0403_0201))
        )
        let scales = MLXRandom.uniform(
            0.0005 ..< 0.0025,
            [2 * outputWidth, inputWidth / groupSize]
        ).asType(.bfloat16)
        let biases = MLXRandom.uniform(
            -0.01 ..< 0.01,
            scales.shape
        ).asType(.bfloat16)
        let projected = MLX.quantizedMM(
            input,
            codes,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: 8,
            mode: .affine
        )
        let parts = MLX.split(projected, parts: 2, axis: -1)
        let reference = MLXNN.silu(parts[0]) * parts[1]
        let candidate = try XCTUnwrap(
            MiniMaxH3FusedKernels.projectFeedForwardInputAffineInt8SwiGLU(
                input: input,
                weightCodes: codes,
                weightScales: scales,
                weightBiases: biases
            )
        )
        MLX.eval(reference, candidate)

        XCTAssertEqual(candidate.shape, [1, rows, outputWidth])
        XCTAssertEqual(candidate.dtype, .bfloat16)
        XCTAssertLessThan(maximumDifference(candidate, reference), 0.032)
    }

    func testProjectFeedForwardOutputAffineInt8MatchesManagedQ8Contract() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Set MERERUN_TEST_MLX_DEVICE=gpu to run the H3 affine FC2 canary."
            )
        }

        let rows = 1
        let inputWidth = 14_336
        let outputWidth = 5_376
        let groupSize = 64
        MLXRandom.seed(2_026_081_010)
        let input = MLXRandom.uniform(
            -0.25 ..< 0.25,
            [1, rows, inputWidth]
        ).asType(.bfloat16)
        let codes = MLXArray.full(
            [outputWidth, inputWidth / 4],
            values: MLXArray(UInt32(0x0403_0201))
        )
        let scales = MLXRandom.uniform(
            0.0005 ..< 0.0025,
            [outputWidth, inputWidth / groupSize]
        ).asType(.bfloat16)
        let biases = MLXRandom.uniform(
            -0.01 ..< 0.01,
            scales.shape
        ).asType(.bfloat16)
        let reference = MLX.quantizedMM(
            input,
            codes,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: 8,
            mode: .affine
        )
        let candidate = try XCTUnwrap(
            MiniMaxH3FusedKernels.projectFeedForwardOutputAffineInt8(
                input: input,
                weightCodes: codes,
                weightScales: scales,
                weightBiases: biases
            )
        )
        MLX.eval(reference, candidate)

        XCTAssertEqual(candidate.shape, [1, rows, outputWidth])
        XCTAssertEqual(candidate.dtype, .bfloat16)
        XCTAssertLessThan(maximumDifference(candidate, reference), 0.032)
    }

    func testCompiledResidualBoundaryDonatesUnretainedH3Buffer() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip(
                "Set MERERUN_TEST_MLX_DEVICE=gpu to run the H3 buffer-donation canary."
            )
        }

        let rows = 512
        let hiddenSize = 5_376
        let tensorBytes = rows * hiddenSize * 2
        let branch = MLXArray.full(
            [1, rows, hiddenSize],
            values: MLXArray(Float(0.25)).asType(.bfloat16)
        )
        let gate = MLXArray.full(
            [1, rows, hiddenSize],
            values: MLXArray(Float(0.5)).asType(.bfloat16)
        )
        let residualBoundary = MLX.compile { inputs in
            [inputs[0] + inputs[1] * inputs[2]]
        }
        MLX.eval(branch, gate)

        func makeUnretainedOutput() -> MLXArray {
            let input = MLXArray.ones([1, rows, hiddenSize], dtype: .bfloat16)
            MLX.eval(input)
            return residualBoundary([input, branch, gate])[0]
        }

        func donatedPeakIncrement() -> Int {
            Memory.clearCache()
            let output = makeUnretainedOutput()
            let baseline = Memory.activeMemory
            Memory.peakMemory = 0
            MLX.eval(output)
            return max(0, Memory.peakMemory - baseline)
        }

        func retainedPeakIncrement() -> Int {
            Memory.clearCache()
            let input = MLXArray.ones([1, rows, hiddenSize], dtype: .bfloat16)
            MLX.eval(input)
            let output = residualBoundary([input, branch, gate])[0]
            let baseline = Memory.activeMemory
            Memory.peakMemory = 0
            withExtendedLifetime(input) {
                MLX.eval(output)
            }
            return max(0, Memory.peakMemory - baseline)
        }

        let warmup = makeUnretainedOutput()
        MLX.eval(warmup)
        let donatedIncrement = donatedPeakIncrement()
        let retainedIncrement = retainedPeakIncrement()

        XCTAssertLessThan(donatedIncrement, tensorBytes / 4)
        XCTAssertGreaterThanOrEqual(retainedIncrement, tensorBytes)
        XCTAssertGreaterThan(retainedIncrement, donatedIncrement)
    }

    func testCompiledResidualBoundaryDonationReleaseBenchmark() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_H3_BUFFER_DONATION_BENCH"] == "1" else {
            throw XCTSkip(
                "Set MERERUN_H3_BUFFER_DONATION_BENCH=1 and MERERUN_TEST_MLX_DEVICE=gpu "
                    + "to run the H3 compiled-buffer donation benchmark."
            )
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The H3 compiled-buffer donation benchmark requires a Metal GPU.")
        }

        let rows = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 14_958
        )
        let hiddenSize = 5_376
        let tensorBytes = rows * hiddenSize * 2
        let branch = MLXArray.full(
            [1, rows, hiddenSize],
            values: MLXArray(Float(0.25)).asType(.bfloat16)
        )
        let gate = MLXArray.full(
            [1, rows, hiddenSize],
            values: MLXArray(Float(0.5)).asType(.bfloat16)
        )
        let residualBoundary = MLX.compile { inputs in
            [inputs[0] + inputs[1] * inputs[2]]
        }
        MLX.eval(branch, gate)

        func makeUnretainedOutput() -> MLXArray {
            let input = MLXArray.ones([1, rows, hiddenSize], dtype: .bfloat16)
            MLX.eval(input)
            return residualBoundary([input, branch, gate])[0]
        }

        func donatedPeakIncrement() -> (output: MLXArray, bytes: Int) {
            Memory.clearCache()
            let output = makeUnretainedOutput()
            let baseline = Memory.activeMemory
            Memory.peakMemory = 0
            MLX.eval(output)
            return (output, max(0, Memory.peakMemory - baseline))
        }

        func retainedPeakIncrement() -> Int {
            Memory.clearCache()
            let input = MLXArray.ones([1, rows, hiddenSize], dtype: .bfloat16)
            MLX.eval(input)
            let output = residualBoundary([input, branch, gate])[0]
            let baseline = Memory.activeMemory
            Memory.peakMemory = 0
            withExtendedLifetime(input) {
                MLX.eval(output)
            }
            return max(0, Memory.peakMemory - baseline)
        }

        let warmup = makeUnretainedOutput()
        MLX.eval(warmup)
        let donated = donatedPeakIncrement()
        let retainedIncrement = retainedPeakIncrement()
        let savedBytes = max(0, retainedIncrement - donated.bytes)
        print(String(
            format: "[h3-transfer] buffer-donation rows=%d tensor_bytes=%d "
                + "donated_peak_increment_bytes=%d retained_peak_increment_bytes=%d "
                + "saved_peak_bytes=%d",
            rows,
            tensorBytes,
            donated.bytes,
            retainedIncrement,
            savedBytes
        ))

        XCTAssertEqual(donated.output.shape, [1, rows, hiddenSize])
        XCTAssertEqual(donated.output.dtype, .bfloat16)
        XCTAssertLessThan(donated.bytes, tensorBytes / 4)
        XCTAssertGreaterThanOrEqual(retainedIncrement, tensorBytes)
        XCTAssertGreaterThan(retainedIncrement, donated.bytes)
    }

    func testGateAttentionAndPrepareFeedForwardReleaseBenchmark() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_H3_FUSED_KERNEL_BENCH"] == "1" else {
            throw XCTSkip(
                "Set MERERUN_H3_FUSED_KERNEL_BENCH=1 and MERERUN_TEST_MLX_DEVICE=gpu "
                    + "to run the H3 fused-boundary benchmark."
            )
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The H3 fused-boundary benchmark requires a Metal GPU.")
        }

        let rows = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 14_958
        )
        let rounds = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 4
        )
        let hiddenSize = 5_376
        let modulationRows = 9
        let epsilon = Float(1e-5)
        MLXRandom.seed(2_026_081_002)
        let residual = MLXRandom.normal([1, rows, hiddenSize]).asType(.bfloat16)
        let attentionOutput = MLXRandom.normal([1, rows, hiddenSize]).asType(.bfloat16)
        let normWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [hiddenSize]
        ).asType(.bfloat16)
        let modulation = (
            MLXRandom.normal([modulationRows, 6 * hiddenSize]) * Float(0.1)
        ).asType(.bfloat16)
        let rowIndices = MLXArray((0..<rows).map { Int32($0 % modulationRows) })
        MLX.eval(residual, attentionOutput, normWeight, modulation, rowIndices)

        let portable = MLX.compile { (inputs: [MLXArray]) -> [MLXArray] in
            let parts = MLX.split(inputs[3], parts: 6, axis: -1)
            func selected(_ part: Int) -> MLXArray {
                MLX.take(parts[part], inputs[4], axis: 0).expandedDimensions(axis: 0)
            }
            let attended = inputs[0] + selected(2) * inputs[1]
            let normalized = MLXFast.rmsNorm(attended, weight: inputs[2], eps: epsilon)
            return [attended, normalized * (1 + selected(4)) + selected(3), selected(5)]
        }
        let inputs = [residual, attentionOutput, normWeight, modulation, rowIndices]

        func portableRun() -> [MLXArray] {
            portable(inputs)
        }
        func fusedRun() -> [MLXArray] {
            guard let output = MiniMaxH3FusedKernels.gateAttentionAndPrepareFeedForward(
                residual: residual,
                attentionOutput: attentionOutput,
                normWeight: normWeight,
                modulation: modulation,
                rowIndices: rowIndices,
                eps: epsilon
            ) else {
                XCTFail("Expected the H3 fused-boundary Metal kernel.")
                return portableRun()
            }
            return [output.residual, output.feedForwardInput, output.feedForwardGate]
        }

        MLX.eval(portableRun(), fusedRun())
        var portableSeconds = 0.0
        var fusedSeconds = 0.0
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                portableSeconds += measure(portableRun)
                fusedSeconds += measure(fusedRun)
            } else {
                fusedSeconds += measure(fusedRun)
                portableSeconds += measure(portableRun)
            }
        }
        portableSeconds /= Double(rounds)
        fusedSeconds /= Double(rounds)

        let reference = portableRun()
        let candidate = fusedRun()
        MLX.eval(reference, candidate)
        let residualDifference = maximumDifference(reference[0], candidate[0])
        let inputDifference = maximumDifference(reference[1], candidate[1])
        let gateDifference = maximumDifference(reference[2], candidate[2])
        print(String(
            format: "[h3-transfer] gate-adaln rows=%d portable_ms=%.3f fused_ms=%.3f "
                + "speedup=%.3fx residual_max_abs=%.6g input_max_abs=%.6g gate_max_abs=%.6g",
            rows,
            portableSeconds * 1_000,
            fusedSeconds * 1_000,
            portableSeconds / fusedSeconds,
            residualDifference,
            inputDifference,
            gateDifference
        ))

        XCTAssertLessThanOrEqual(residualDifference, 0.00390625)
        XCTAssertLessThan(inputDifference, 0.016)
        XCTAssertEqual(gateDifference, 0)
    }

    func testGateAttentionAndQuantizeFeedForwardReleaseBenchmark() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_H3_FUSED_INT8_BENCH"] == "1" else {
            throw XCTSkip(
                "Set MERERUN_H3_FUSED_INT8_BENCH=1 and MERERUN_TEST_MLX_DEVICE=gpu "
                    + "to run the H3 fused INT8 boundary benchmark."
            )
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The H3 fused INT8 boundary benchmark requires a Metal GPU.")
        }

        let rows = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 14_958
        )
        let rounds = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 4
        )
        let hiddenSize = 5_376
        let modulationRows = 9
        let epsilon = Float(1e-5)
        MLXRandom.seed(2_026_081_004)
        let residual = MLXRandom.normal([1, rows, hiddenSize]).asType(.bfloat16)
        let attentionOutput = MLXRandom.normal([1, rows, hiddenSize]).asType(.bfloat16)
        let normWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [hiddenSize]
        ).asType(.bfloat16)
        let modulation = (
            MLXRandom.normal([modulationRows, 6 * hiddenSize]) * Float(0.1)
        ).asType(.bfloat16)
        let rowIndices = MLXArray((0..<rows).map { Int32($0 % modulationRows) })
        MLX.eval(residual, attentionOutput, normWeight, modulation, rowIndices)

        func unfusedRun() -> [MLXArray] {
            guard let floating = MiniMaxH3FusedKernels.gateAttentionAndPrepareFeedForward(
                residual: residual,
                attentionOutput: attentionOutput,
                normWeight: normWeight,
                modulation: modulation,
                rowIndices: rowIndices,
                eps: epsilon
            ), let quantized = MiniMaxH3FusedKernels.quantizeRowsSymmetricInt8(
                floating.feedForwardInput
            ) else {
                XCTFail("Expected the H3 two-kernel INT8 boundary.")
                return []
            }
            return [floating.residual, quantized.values, quantized.scales]
        }
        func fusedRun() -> [MLXArray] {
            guard let output = MiniMaxH3FusedKernels.gateAttentionAndQuantizeFeedForward(
                residual: residual,
                attentionOutput: attentionOutput,
                normWeight: normWeight,
                modulation: modulation,
                rowIndices: rowIndices,
                eps: epsilon
            ) else {
                XCTFail("Expected the H3 fused INT8 boundary.")
                return []
            }
            return [
                output.residual,
                output.feedForwardInput.values,
                output.feedForwardInput.scales,
            ]
        }

        MLX.eval(unfusedRun(), fusedRun())
        var unfusedSeconds = 0.0
        var fusedSeconds = 0.0
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                unfusedSeconds += measure(unfusedRun)
                fusedSeconds += measure(fusedRun)
            } else {
                fusedSeconds += measure(fusedRun)
                unfusedSeconds += measure(unfusedRun)
            }
        }
        unfusedSeconds /= Double(rounds)
        fusedSeconds /= Double(rounds)

        let reference = unfusedRun()
        let candidate = fusedRun()
        MLX.eval(reference, candidate)
        let residualDifference = maximumDifference(reference[0], candidate[0])
        let scaleDifference = maximumDifference(reference[2], candidate[2])
        let quantizedMatches = reference[1].asArray(Int8.self)
            == candidate[1].asArray(Int8.self)
        print(String(
            format: "[h3-transfer] gate-adaln-int8 rows=%d unfused_ms=%.3f "
                + "fused_ms=%.3f speedup=%.3fx residual_max_abs=%.6g "
                + "scale_max_abs=%.6g quantized_exact=%@",
            rows,
            unfusedSeconds * 1_000,
            fusedSeconds * 1_000,
            unfusedSeconds / fusedSeconds,
            residualDifference,
            scaleDifference,
            quantizedMatches ? "true" : "false"
        ))

        XCTAssertEqual(residualDifference, 0)
        XCTAssertEqual(scaleDifference, 0)
        XCTAssertTrue(quantizedMatches)
    }

    func testPrepareHeadMajorQKVReleaseBenchmark() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_H3_QKV_LAYOUT_BENCH"] == "1" else {
            throw XCTSkip(
                "Set MERERUN_H3_QKV_LAYOUT_BENCH=1 and MERERUN_TEST_MLX_DEVICE=gpu "
                    + "to run the H3 fused QKV-layout benchmark."
            )
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The H3 fused QKV-layout benchmark requires a Metal GPU.")
        }

        let rows = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 14_958
        )
        let rounds = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 4
        )
        let heads = 56
        let headDimension = 128
        let rotaryDimension = 96
        let epsilon = Float(1e-5)
        MLXRandom.seed(2_026_081_006)
        let projected = MLXRandom.normal(
            [1, rows, 3 * heads * headDimension]
        ).asType(.bfloat16)
        let queryNormWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [headDimension]
        ).asType(.bfloat16)
        let keyNormWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [headDimension]
        ).asType(.bfloat16)
        let angles = MLXRandom.uniform(
            -1 ..< 1,
            [1, rows, 1, rotaryDimension]
        )
        let ropeCosine = MLX.cos(angles).asType(.bfloat16)
        let ropeSine = MLX.sin(angles).asType(.bfloat16)
        MLX.eval(
            projected,
            queryNormWeight,
            keyNormWeight,
            ropeCosine,
            ropeSine
        )

        func portableRun() -> [MLXArray] {
            let parts = miniMaxH3SplitProjectedQKV(
                projected,
                heads: heads,
                headDimension: headDimension
            )
            let rope = MiniMaxH3RotaryEmbedding(cosine: ropeCosine, sine: ropeSine)
            let query = rope.apply(MLXFast.rmsNorm(
                parts[0],
                weight: queryNormWeight,
                eps: epsilon
            )).transposed(0, 2, 1, 3).contiguous()
            let key = rope.apply(MLXFast.rmsNorm(
                parts[1],
                weight: keyNormWeight,
                eps: epsilon
            )).transposed(0, 2, 1, 3).contiguous()
            let value = parts[2].transposed(0, 2, 1, 3).contiguous()
            return [query, key, value]
        }
        func fusedRun() -> [MLXArray] {
            guard let output = MiniMaxH3FusedKernels.prepareHeadMajorQKV(
                projected: projected,
                queryNormWeight: queryNormWeight,
                keyNormWeight: keyNormWeight,
                ropeCosine: ropeCosine,
                ropeSine: ropeSine,
                eps: epsilon
            ) else {
                XCTFail("Expected the H3 fused QKV-layout kernel.")
                return []
            }
            return [output.query, output.key, output.value]
        }

        MLX.eval(portableRun(), fusedRun())
        var portableSeconds = 0.0
        var fusedSeconds = 0.0
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                portableSeconds += measure(portableRun)
                fusedSeconds += measure(fusedRun)
            } else {
                fusedSeconds += measure(fusedRun)
                portableSeconds += measure(portableRun)
            }
        }
        portableSeconds /= Double(rounds)
        fusedSeconds /= Double(rounds)

        let reference = portableRun()
        let candidate = fusedRun()
        MLX.eval(reference, candidate)
        let queryDifference = maximumDifference(reference[0], candidate[0])
        let keyDifference = maximumDifference(reference[1], candidate[1])
        let valueDifference = maximumDifference(reference[2], candidate[2])
        print(String(
            format: "[h3-transfer] qkv-layout rows=%d portable_ms=%.3f "
                + "fused_ms=%.3f speedup=%.3fx query_max_abs=%.6g "
                + "key_max_abs=%.6g value_max_abs=%.6g",
            rows,
            portableSeconds * 1_000,
            fusedSeconds * 1_000,
            portableSeconds / fusedSeconds,
            queryDifference,
            keyDifference,
            valueDifference
        ))

        XCTAssertLessThan(queryDifference, 0.032)
        XCTAssertLessThan(keyDifference, 0.032)
        XCTAssertEqual(valueDifference, 0)
    }

    func testProjectHeadMajorQKVAffineInt8ReleaseBenchmark() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_H3_QKV_DIRECT_BENCH"] == "1" else {
            throw XCTSkip(
                "Set MERERUN_H3_QKV_DIRECT_BENCH=1 and MERERUN_TEST_MLX_DEVICE=gpu "
                    + "to run the H3 projection-direct QKV benchmark."
            )
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The H3 projection-direct QKV benchmark requires a Metal GPU.")
        }

        let rows = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 14_958
        )
        let rounds = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 4
        )
        let hiddenSize = 5_376
        let heads = 56
        let headDimension = 128
        let innerDimension = heads * headDimension
        let projectionWidth = 3 * innerDimension
        let rotaryDimension = 96
        let groupSize = 64
        let epsilon = Float(1e-5)
        MLXRandom.seed(2_026_081_012)
        let input = MLXRandom.normal([1, rows, hiddenSize]).asType(.bfloat16)
        let codes = MLXArray.full(
            [projectionWidth, hiddenSize / 4],
            values: MLXArray(UInt32(0x0403_0201))
        )
        let scales = MLXRandom.uniform(
            0.0005 ..< 0.0025,
            [projectionWidth, hiddenSize / groupSize]
        ).asType(.bfloat16)
        let biases = MLXRandom.uniform(
            -0.01 ..< 0.01,
            scales.shape
        ).asType(.bfloat16)
        let queryNormWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [headDimension]
        ).asType(.bfloat16)
        let keyNormWeight = MLXRandom.uniform(
            0.8 ..< 1.2,
            [headDimension]
        ).asType(.bfloat16)
        let angles = MLXRandom.uniform(
            -1 ..< 1,
            [1, rows, 1, rotaryDimension]
        )
        let ropeCosine = MLX.cos(angles).asType(.bfloat16)
        let ropeSine = MLX.sin(angles).asType(.bfloat16)
        MLX.eval(
            input,
            codes,
            scales,
            biases,
            queryNormWeight,
            keyNormWeight,
            ropeCosine,
            ropeSine
        )

        func portableRun() -> [MLXArray] {
            let projected = MLX.quantizedMM(
                input,
                codes,
                scales: scales,
                biases: biases,
                transpose: true,
                groupSize: groupSize,
                bits: 8,
                mode: .affine
            )
            let parts = miniMaxH3SplitProjectedQKV(
                projected,
                heads: heads,
                headDimension: headDimension
            )
            let rope = MiniMaxH3RotaryEmbedding(cosine: ropeCosine, sine: ropeSine)
            let query = rope.apply(MLXFast.rmsNorm(
                parts[0],
                weight: queryNormWeight,
                eps: epsilon
            )).transposed(0, 2, 1, 3).contiguous()
            let key = rope.apply(MLXFast.rmsNorm(
                parts[1],
                weight: keyNormWeight,
                eps: epsilon
            )).transposed(0, 2, 1, 3).contiguous()
            let value = parts[2].transposed(0, 2, 1, 3).contiguous()
            return [query, key, value]
        }
        func fusedRun() -> [MLXArray] {
            guard let output = MiniMaxH3FusedKernels.projectHeadMajorQKVAffineInt8(
                input: input,
                weightCodes: codes,
                weightScales: scales,
                weightBiases: biases,
                queryNormWeight: queryNormWeight,
                keyNormWeight: keyNormWeight,
                ropeCosine: ropeCosine,
                ropeSine: ropeSine,
                eps: epsilon
            ) else {
                XCTFail("Expected the H3 projection-direct QKV kernel path.")
                return []
            }
            return [output.query, output.key, output.value]
        }

        MLX.eval(portableRun(), fusedRun())
        var portableSeconds = 0.0
        var fusedSeconds = 0.0
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                portableSeconds += measure(portableRun)
                fusedSeconds += measure(fusedRun)
            } else {
                fusedSeconds += measure(fusedRun)
                portableSeconds += measure(portableRun)
            }
        }
        portableSeconds /= Double(rounds)
        fusedSeconds /= Double(rounds)

        let reference = portableRun()
        let candidate = fusedRun()
        MLX.eval(reference, candidate)
        let queryDifference = maximumDifference(reference[0], candidate[0])
        let keyDifference = maximumDifference(reference[1], candidate[1])
        let valueDifference = maximumDifference(reference[2], candidate[2])
        let globalProjectionBytes = UInt64(rows * projectionWidth * 2)
        print(String(
            format: "[h3-transfer] qkv-direct rows=%d portable_ms=%.3f "
                + "fused_ms=%.3f speedup=%.3fx query_max_abs=%.6g "
                + "key_max_abs=%.6g value_max_abs=%.6g "
                + "global_projection_avoided_bytes=%llu",
            rows,
            portableSeconds * 1_000,
            fusedSeconds * 1_000,
            portableSeconds / fusedSeconds,
            queryDifference,
            keyDifference,
            valueDifference,
            globalProjectionBytes
        ))

        XCTAssertLessThan(queryDifference, 0.032)
        XCTAssertLessThan(keyDifference, 0.032)
        XCTAssertLessThan(valueDifference, 0.032)
    }

    func testProjectHeadMajorAttentionAffineInt8ReleaseBenchmark() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_H3_AFFINE_OPROJ_BENCH"] == "1" else {
            throw XCTSkip(
                "Set MERERUN_H3_AFFINE_OPROJ_BENCH=1 and MERERUN_TEST_MLX_DEVICE=gpu "
                    + "to run the H3 affine output-projection benchmark."
            )
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The H3 affine output-projection benchmark requires a Metal GPU.")
        }

        let rows = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 14_958
        )
        let rounds = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 4
        )
        let heads = 56
        let headDimension = 128
        let inputWidth = heads * headDimension
        let outputWidth = 5_376
        let groupSize = 64
        MLXRandom.seed(2_026_081_008)
        let attention = MLXRandom.normal(
            [1, heads, rows, headDimension]
        ).asType(.bfloat16)
        let codes = MLXArray.full(
            [outputWidth, inputWidth / 4],
            values: MLXArray(UInt32(0x0403_0201))
        )
        let scales = MLXRandom.uniform(
            0.0005 ..< 0.0025,
            [outputWidth, inputWidth / groupSize]
        ).asType(.bfloat16)
        let biases = MLXRandom.uniform(
            -0.01 ..< 0.01,
            scales.shape
        ).asType(.bfloat16)
        MLX.eval(attention, codes, scales, biases)

        func portableRun() -> [MLXArray] {
            let rowMajor = attention.transposed(0, 2, 1, 3).reshaped(
                1,
                rows,
                inputWidth
            )
            return [MLX.quantizedMM(
                rowMajor,
                codes,
                scales: scales,
                biases: biases,
                transpose: true,
                groupSize: groupSize,
                bits: 8,
                mode: .affine
            )]
        }
        func fusedRun() -> [MLXArray] {
            guard let output = MiniMaxH3FusedKernels.projectHeadMajorAttentionAffineInt8(
                attention: attention,
                weightCodes: codes,
                weightScales: scales,
                weightBiases: biases
            ) else {
                XCTFail("Expected the H3 head-major affine output kernel.")
                return []
            }
            return [output]
        }

        MLX.eval(portableRun(), fusedRun())
        var portableSeconds = 0.0
        var fusedSeconds = 0.0
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                portableSeconds += measure(portableRun)
                fusedSeconds += measure(fusedRun)
            } else {
                fusedSeconds += measure(fusedRun)
                portableSeconds += measure(portableRun)
            }
        }
        portableSeconds /= Double(rounds)
        fusedSeconds /= Double(rounds)

        let reference = portableRun()[0]
        let candidate = fusedRun()[0]
        MLX.eval(reference, candidate)
        let difference = maximumDifference(reference, candidate)
        print(String(
            format: "[h3-transfer] affine-oproj rows=%d portable_ms=%.3f "
                + "fused_ms=%.3f speedup=%.3fx max_abs=%.6g",
            rows,
            portableSeconds * 1_000,
            fusedSeconds * 1_000,
            portableSeconds / fusedSeconds,
            difference
        ))

        XCTAssertLessThan(difference, 0.032)
    }

    func testFusedFeedForwardAffineInt8ReleaseBenchmark() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_H3_AFFINE_FFN_BENCH"] == "1" else {
            throw XCTSkip(
                "Set MERERUN_H3_AFFINE_FFN_BENCH=1 and MERERUN_TEST_MLX_DEVICE=gpu "
                    + "to run the H3 affine feed-forward benchmark."
            )
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("The H3 affine feed-forward benchmark requires a Metal GPU.")
        }

        let rows = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROWS"] ?? "") ?? 14_958
        )
        let rounds = max(
            1,
            Int(environment["MERERUN_H3_BENCH_ROUNDS"] ?? "") ?? 4
        )
        let hiddenSize = 5_376
        let feedForwardSize = 14_336
        let groupSize = 64
        MLXRandom.seed(2_026_081_011)
        let input = MLXRandom.normal([1, rows, hiddenSize]).asType(.bfloat16)
        let fc1Codes = MLXArray.full(
            [2 * feedForwardSize, hiddenSize / 4],
            values: MLXArray(UInt32(0x0403_0201))
        )
        let fc1Scales = MLXRandom.uniform(
            0.0005 ..< 0.0025,
            [2 * feedForwardSize, hiddenSize / groupSize]
        ).asType(.bfloat16)
        let fc1Biases = MLXRandom.uniform(
            -0.01 ..< 0.01,
            fc1Scales.shape
        ).asType(.bfloat16)
        let fc2Codes = MLXArray.full(
            [hiddenSize, feedForwardSize / 4],
            values: MLXArray(UInt32(0x0403_0201))
        )
        let fc2Scales = MLXRandom.uniform(
            0.0005 ..< 0.0025,
            [hiddenSize, feedForwardSize / groupSize]
        ).asType(.bfloat16)
        let fc2Biases = MLXRandom.uniform(
            -0.01 ..< 0.01,
            fc2Scales.shape
        ).asType(.bfloat16)
        MLX.eval(
            input,
            fc1Codes,
            fc1Scales,
            fc1Biases,
            fc2Codes,
            fc2Scales,
            fc2Biases
        )

        func portableRun() -> [MLXArray] {
            let projected = MLX.quantizedMM(
                input,
                fc1Codes,
                scales: fc1Scales,
                biases: fc1Biases,
                transpose: true,
                groupSize: groupSize,
                bits: 8,
                mode: .affine
            )
            let parts = MLX.split(projected, parts: 2, axis: -1)
            let activated = MLXNN.silu(parts[0]) * parts[1]
            return [MLX.quantizedMM(
                activated,
                fc2Codes,
                scales: fc2Scales,
                biases: fc2Biases,
                transpose: true,
                groupSize: groupSize,
                bits: 8,
                mode: .affine
            )]
        }
        func fusedRun() -> [MLXArray] {
            guard let activated = MiniMaxH3FusedKernels
                .projectFeedForwardInputAffineInt8SwiGLU(
                    input: input,
                    weightCodes: fc1Codes,
                    weightScales: fc1Scales,
                    weightBiases: fc1Biases
                ),
                let projected = MiniMaxH3FusedKernels.projectFeedForwardOutputAffineInt8(
                    input: activated,
                    weightCodes: fc2Codes,
                    weightScales: fc2Scales,
                    weightBiases: fc2Biases
                ) else {
                XCTFail("Expected both H3 affine feed-forward kernels.")
                return []
            }
            return [projected]
        }

        MLX.eval(portableRun(), fusedRun())
        var portableSeconds = 0.0
        var fusedSeconds = 0.0
        for round in 0..<rounds {
            if round.isMultiple(of: 2) {
                portableSeconds += measure(portableRun)
                fusedSeconds += measure(fusedRun)
            } else {
                fusedSeconds += measure(fusedRun)
                portableSeconds += measure(portableRun)
            }
        }
        portableSeconds /= Double(rounds)
        fusedSeconds /= Double(rounds)

        let reference = portableRun()[0]
        let candidate = fusedRun()[0]
        MLX.eval(reference, candidate)
        let difference = maximumDifference(reference, candidate)
        let avoidedBytes = UInt64(rows) * UInt64(2 * feedForwardSize) * 2
        print(String(
            format: "[h3-transfer] affine-ffn rows=%d portable_ms=%.3f "
                + "fused_ms=%.3f speedup=%.3fx max_abs=%.6g "
                + "fc1_materialization_avoided_bytes=%llu",
            rows,
            portableSeconds * 1_000,
            fusedSeconds * 1_000,
            portableSeconds / fusedSeconds,
            difference,
            avoidedBytes
        ))

        XCTAssertLessThan(difference, 0.125)
    }
    #endif

    private func maximumDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        MLX.max(MLX.abs(lhs.asType(.float32) - rhs.asType(.float32))).item(Float.self)
    }

    private func measure(_ body: () -> [MLXArray]) -> Double {
        let started = CFAbsoluteTimeGetCurrent()
        MLX.eval(body())
        return CFAbsoluteTimeGetCurrent() - started
    }
}
