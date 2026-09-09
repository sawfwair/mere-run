import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import XCTest
import MereRunMLXTestSupport
@testable import MereRunH3Model
@testable import MereRunTensor

final class H3ComputationTests: MLXTestCase {
    func testFastVSAGeometryKeepsPrefixSegmentsPureAndTilesVideoIn3D() throws {
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: Array(repeating: 1, count: 70),
            videoLatentFrames: 5,
            latentHeight: 10,
            latentWidth: 10,
            audioLatentFrames: 3,
            keyframeAnchors: []
        )
        let geometry = MiniMaxH3FastVSAGeometry(layout: layout)

        XCTAssertEqual(geometry.prefixTileCount, 3)
        XCTAssertEqual(geometry.videoTileCount, 8)
        XCTAssertEqual(Array(geometry.blockSizes.prefix(3)), [64, 6, 6])
        XCTAssertEqual(geometry.blockSizes.reduce(0, { $0 + Int($1) }), layout.sequenceLength)
        XCTAssertEqual(geometry.originalToPadded.count, layout.sequenceLength)
        XCTAssertEqual(
            Set(geometry.originalToPadded).count,
            layout.sequenceLength
        )
        for (original, padded) in geometry.originalToPadded.enumerated() {
            XCTAssertEqual(geometry.paddedToOriginal[Int(padded)], Int32(original))
        }
    }

    func testFastVSARoutingKeepsPrefixDenseAndTopKVideoKeys() {
        let scores = MLXArray((0..<25).map(Float.init)).reshaped(1, 1, 5, 5)
        let routes = MiniMaxH3FastVSA.routesForTesting(
            scores: scores,
            prefixTileCount: 2,
            videoTileCount: 3,
            sparsity: 0.9
        )
        MLX.eval(routes)
        let values = routes.asArray(UInt8.self)
        for query in 0..<2 {
            XCTAssertEqual(Array(values[(query * 5)..<(query * 5 + 5)]), [1, 1, 1, 1, 1])
        }
        for query in 2..<5 {
            let row = Array(values[(query * 5)..<(query * 5 + 5)])
            XCTAssertEqual(Array(row.prefix(2)), [1, 1])
            XCTAssertEqual(row.suffix(3).reduce(0, +), 1)
        }

        let compact = MiniMaxH3FastVSA.selectedVideoRoutesForTesting(
            scores: scores,
            prefixTileCount: 2,
            videoTileCount: 3,
            sparsity: 0.9
        )
        MLX.eval(compact)
        XCTAssertEqual(compact.shape, [1, 1, 5, 1])
        XCTAssertEqual(compact.asArray(Int32.self), [4, 4, 4, 4, 4])
    }

    func testFastVSAMetalDenseRouteMatchesFusedSDPA() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("FastH3 VSA parity requires a Metal GPU.")
        }
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: Array(repeating: 1, count: 7),
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: []
        )
        MLXRandom.seed(2_026_082_9)
        let shape = [1, 2, layout.sequenceLength, MiniMaxH3FastVSA.headDimension]
        let queries = (MLXRandom.normal(shape) * Float(0.2)).asType(.bfloat16)
        let keys = (MLXRandom.normal(shape) * Float(0.2)).asType(.bfloat16)
        let values = MLXRandom.normal(shape).asType(.bfloat16)
        let gate = MLXArray.zeros(shape, dtype: .bfloat16)
        let candidate = try XCTUnwrap(MiniMaxH3FastVSA.call(
            queries: queries,
            keys: keys,
            values: values,
            compressionGate: gate,
            layout: layout,
            sparsity: 0
        ))
        let reference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(Float(MiniMaxH3FastVSA.headDimension)),
            mask: .none
        )
        let delta = candidate.asType(.float32) - reference.asType(.float32)
        let relativeL2 = MLX.sqrt(
            MLX.sum(delta * delta)
                / MLX.maximum(
                    MLX.sum(reference.asType(.float32) * reference.asType(.float32)),
                    MLXArray(Float(1e-12))
                )
        )
        MLX.eval(candidate, reference, relativeL2)
        XCTAssertLessThanOrEqual(relativeL2.item(Float.self), 0.005)
    }

    func testFastVSAMetalAcceptsFP32RotaryQueriesAndKeys() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("FastH3 VSA rotary promotion coverage requires a Metal GPU.")
        }
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: Array(repeating: 1, count: 7),
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: []
        )
        MLXRandom.seed(2_026_083_1)
        let shape = [1, 2, layout.sequenceLength, MiniMaxH3FastVSA.headDimension]
        let queries = MLXRandom.normal(shape) * Float(0.2)
        let keys = MLXRandom.normal(shape) * Float(0.2)
        let values = MLXRandom.normal(shape).asType(.bfloat16)
        let gate = MLXArray.zeros(shape, dtype: .bfloat16)
        let candidate = try XCTUnwrap(MiniMaxH3FastVSA.call(
            queries: queries,
            keys: keys,
            values: values,
            compressionGate: gate,
            layout: layout,
            sparsity: 0
        ))
        let expected = try XCTUnwrap(MiniMaxH3FastVSA.call(
            queries: queries.asType(.bfloat16),
            keys: keys.asType(.bfloat16),
            values: values,
            compressionGate: gate,
            layout: layout,
            sparsity: 0
        ))
        MLX.eval(candidate, expected)
        XCTAssertEqual(candidate.dtype, .bfloat16)
        XCTAssertEqual(candidate.asArray(Float.self), expected.asArray(Float.self))
    }

    func testFastVSAFullTileKernelMatchesHalfTileKernel() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("FastH3 VSA tile schedule parity requires a Metal GPU.")
        }
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: Array(repeating: 1, count: 70),
            videoLatentFrames: 5,
            latentHeight: 10,
            latentWidth: 10,
            audioLatentFrames: 3,
            keyframeAnchors: []
        )
        MLXRandom.seed(2_026_082_9)
        let shape = [1, 2, layout.sequenceLength, MiniMaxH3FastVSA.headDimension]
        let queries = (MLXRandom.normal(shape) * Float(0.2)).asType(.bfloat16)
        let keys = (MLXRandom.normal(shape) * Float(0.2)).asType(.bfloat16)
        let values = MLXRandom.normal(shape).asType(.bfloat16)
        let gate = (MLXRandom.normal(shape) * Float(0.05)).asType(.bfloat16)
        let halfTile = try XCTUnwrap(MiniMaxH3FastVSA.call(
            queries: queries,
            keys: keys,
            values: values,
            compressionGate: gate,
            layout: layout,
            kernelMode: .halfTile
        ))
        let fullTile = try XCTUnwrap(MiniMaxH3FastVSA.call(
            queries: queries,
            keys: keys,
            values: values,
            compressionGate: gate,
            layout: layout,
            kernelMode: .fullTile
        ))
        let fullTileKV16 = try XCTUnwrap(MiniMaxH3FastVSA.call(
            queries: queries,
            keys: keys,
            values: values,
            compressionGate: gate,
            layout: layout,
            kernelMode: .fullTileKV16
        ))
        MLX.eval(halfTile, fullTile, fullTileKV16)
        XCTAssertEqual(halfTile.asArray(Float.self), fullTile.asArray(Float.self))
        let reference = halfTile.asType(.float32)
        let candidate = fullTileKV16.asType(.float32)
        let delta = candidate - reference
        let maximumAbsoluteError = MLX.max(MLX.abs(delta))
        let relativeL2Error = MLX.sqrt(
            MLX.sum(delta * delta)
                / MLX.maximum(MLX.sum(reference * reference), MLXArray(Float(1e-12)))
        )
        MLX.eval(maximumAbsoluteError, relativeL2Error)
        XCTAssertLessThanOrEqual(maximumAbsoluteError.item(Float.self), 0.02)
        XCTAssertLessThanOrEqual(relativeL2Error.item(Float.self), 0.01)
    }

    func testFastVSAMetalSparseRouteAndCompressionMatchTensorReference() throws {
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("FastH3 sparse VSA parity requires a Metal GPU.")
        }
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: Array(repeating: 1, count: 7),
            videoLatentFrames: 5,
            latentHeight: 10,
            latentWidth: 10,
            audioLatentFrames: 3,
            keyframeAnchors: []
        )
        let geometry = MiniMaxH3FastVSAGeometry(layout: layout)
        MLXRandom.seed(2_026_083_0)
        let shape = [1, 2, layout.sequenceLength, MiniMaxH3FastVSA.headDimension]
        let queries = (MLXRandom.normal(shape) * Float(0.2)).asType(.bfloat16)
        let keys = (MLXRandom.normal(shape) * Float(0.2)).asType(.bfloat16)
        let values = MLXRandom.normal(shape).asType(.bfloat16)
        let gate = (MLXRandom.normal(shape) * Float(0.05)).asType(.bfloat16)
        let candidate = try XCTUnwrap(MiniMaxH3FastVSA.call(
            queries: queries,
            keys: keys,
            values: values,
            compressionGate: gate,
            layout: layout
        ))

        let sentinel = MLXArray.zeros([1, 2, 1, MiniMaxH3FastVSA.headDimension], dtype: .bfloat16)
        let paddedIndices = MLXArray(geometry.paddedToOriginal)
        let inverseIndices = MLXArray(geometry.originalToPadded)
        func tiled(_ value: MLXArray) -> MLXArray {
            MLX.take(MLX.concatenated([value, sentinel], axis: 2), paddedIndices, axis: 2)
        }
        let tiledQueries = tiled(queries)
        let tiledKeys = tiled(keys)
        let tiledValues = tiled(values)
        let tiledGate = tiled(gate)
        let tileCount = geometry.tileCount
        let paddedCount = geometry.paddedTokenCount
        let sizes = MLXArray(geometry.blockSizes).asType(.float32).reshaped(1, 1, tileCount, 1)
        func pooled(_ value: MLXArray) -> MLXArray {
            value.asType(.float32)
                .reshaped(1, 2, tileCount, MiniMaxH3FastVSAGeometry.tileSize, MiniMaxH3FastVSA.headDimension)
                .sum(axis: 3) / sizes
        }
        let pooledQueries = pooled(tiledQueries)
        let pooledKeys = pooled(tiledKeys)
        let pooledValues = pooled(tiledValues)
        let scale = 1 / sqrt(Float(MiniMaxH3FastVSA.headDimension))
        let routeScores = MLX.matmul(pooledQueries, pooledKeys.transposed(0, 1, 3, 2)) * scale
        let routes = MiniMaxH3FastVSA.routesForTesting(
            scores: routeScores,
            prefixTileCount: geometry.prefixTileCount,
            videoTileCount: geometry.videoTileCount
        )
        let tokenRoutes = MLX.broadcast(
            routes.reshaped(1, 2, tileCount, 1, tileCount, 1),
            to: [1, 2, tileCount, 64, tileCount, 64]
        ).reshaped(1, 2, paddedCount, paddedCount)
        let validKeys = MLXArray(geometry.paddedToOriginal.map {
            $0 == Int32(layout.sequenceLength) ? UInt8(0) : UInt8(1)
        }).reshaped(1, 1, 1, paddedCount)
        let exactScores = MLX.matmul(
            tiledQueries.asType(.float32),
            tiledKeys.asType(.float32).transposed(0, 1, 3, 2)
        ) * scale
        let maskedScores = MLX.where(
            (tokenRoutes * validKeys) .== 1,
            exactScores,
            MLXArray(-Float.infinity)
        )
        let exact = MLX.matmul(
            MLX.softmax(maskedScores, axis: -1, precise: true),
            tiledValues.asType(.float32)
        )
        let compressed = MLX.matmul(
            MLX.softmax(routeScores, axis: -1, precise: true),
            pooledValues
        )
        let referenceTiled = exact.reshaped(
            1, 2, tileCount, 64, MiniMaxH3FastVSA.headDimension
        ) + compressed.expandedDimensions(axis: 3)
            * tiledGate.asType(.float32).reshaped(1, 2, tileCount, 64, MiniMaxH3FastVSA.headDimension)
        let reference = MLX.take(
            referenceTiled.reshaped(1, 2, paddedCount, MiniMaxH3FastVSA.headDimension),
            inverseIndices,
            axis: 2
        )
        let delta = candidate.asType(.float32) - reference
        let relativeL2 = MLX.sqrt(
            MLX.sum(delta * delta)
                / MLX.maximum(MLX.sum(reference * reference), MLXArray(Float(1e-12)))
        )
        MLX.eval(candidate, reference, relativeL2)
        XCTAssertLessThanOrEqual(relativeL2.item(Float.self), 0.01)
    }

    func testRuntimeLoRAAppliesDeltaInActivationSpace() {
        let base = Linear(
            weight: MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2),
            bias: nil
        )
        let layer = MiniMaxH3RuntimeLoRALinear(
            base: base,
            loraDown: MLXArray([Float(1), 0]).reshaped(1, 2),
            loraUp: MLXArray([Float(2), 3]).reshaped(2, 1),
            strength: 0.5
        )
        let output = layer(MLXArray([Float(4), 5]).reshaped(1, 2))
        MLX.eval(output)
        XCTAssertEqual(output.asArray(Float.self), [8, 11])
    }

    func testQuantizedRuntimeLoRAUsesStockQuantizedBaseMath() {
        let inputSize = 32
        let outputSize = 4
        let dense = Linear(
            weight: MLXArray((0..<(inputSize * outputSize)).map { Float($0) / 128 })
                .reshaped(outputSize, inputSize),
            bias: MLXArray.zeros([outputSize])
        )
        let base = QuantizedLinear(dense, groupSize: 32, bits: 8)
        var downValues = [Float](repeating: 0, count: inputSize)
        downValues[0] = 1
        downValues[1] = 1
        let down = MLXArray(downValues).reshaped(1, inputSize)
        let up = MLXArray([Float(1), 2, 3, 4]).reshaped(4, 1)
        let layer = MiniMaxH3RuntimeQuantizedLoRALinear(
            base: base,
            loraDown: down,
            loraUp: up,
            strength: 0.5
        )
        let input = MLXArray((0..<inputSize).map { Float($0 + 1) }).reshaped(1, inputSize)
        let candidate = layer(input)
        let reference = base(input)
            + MLX.matmul(MLX.matmul(input, down.T), up.T) * 0.5
        MLX.eval(candidate, reference)
        XCTAssertLessThanOrEqual(
            MLX.abs(candidate.asType(.float32) - reference.asType(.float32))
                .max().item(Float.self),
            1e-5
        )
    }

    func testQuantizedRuntimeQKVLoRAPreservesGlobalSlabOrdering() {
        let inputSize = 32
        let branchSize = 32
        let dense = Linear(
            weight: MLXArray.zeros([branchSize * 3, inputSize]),
            bias: MLXArray.zeros([branchSize * 3])
        )
        let base = QuantizedLinear(dense, groupSize: 32, bits: 8)
        let down = MLXArray.ones([1, inputSize])
        let queryUp = MLXArray.ones([branchSize, 1])
        let keyUp = MLXArray.ones([branchSize, 1]) * 2
        let valueUp = MLXArray.ones([branchSize, 1]) * 3
        let layer = MiniMaxH3RuntimeQuantizedQKVLoRALinear(
            base: base,
            queryDown: down,
            queryUp: queryUp,
            keyDown: down,
            keyUp: keyUp,
            valueDown: down,
            valueUp: valueUp,
            strength: 0.5
        )
        let input = MLXArray.ones([1, inputSize])
        let candidate = layer(input)
        let reference = base(input) + MLX.concatenated([
            MLX.matmul(MLX.matmul(input, down.T), queryUp.T) * 0.5,
            MLX.matmul(MLX.matmul(input, down.T), keyUp.T) * 0.5,
            MLX.matmul(MLX.matmul(input, down.T), valueUp.T) * 0.5,
        ], axis: -1)
        MLX.eval(candidate, reference)
        XCTAssertLessThanOrEqual(
            MLX.abs(candidate.asType(.float32) - reference.asType(.float32))
                .max().item(Float.self),
            1e-5
        )
        let values = candidate.asArray(Float.self)
        XCTAssertEqual(Array(values[0..<branchSize]), [Float](repeating: 16, count: branchSize))
        XCTAssertEqual(
            Array(values[branchSize..<(branchSize * 2)]),
            [Float](repeating: 32, count: branchSize)
        )
        XCTAssertEqual(
            Array(values[(branchSize * 2)..<(branchSize * 3)]),
            [Float](repeating: 48, count: branchSize)
        )
    }

    func testTokenReductionPreservesPairDetailThroughBypassedReconstruction() {
        let layout = MiniMaxH3PackedLayout(
            positions: MLXArray.zeros([7, 3]),
            tokenTags: Array(repeating: Int32(0), count: 7),
            textRows: 0..<1,
            conditionRows: 1..<1,
            conditionSegments: [],
            conditionVideoRowCount: 0,
            conditionAudioRowCount: 0,
            targetAudioRows: 1..<1,
            targetVideoRows: 1..<7,
            videoLatentFrames: 1,
            latentHeight: 4,
            latentWidth: 6,
            audioLatentFrames: 0
        )
        let map = MiniMaxH3TokenReductionMap(layout: layout)
        let hidden = MLXArray([Float(100), 0, 2, 10, 14, 20, 26])
            .reshaped(1, 7, 1)
        let state = map.pool(hidden)
        MLX.eval(state.reducedHidden)

        XCTAssertEqual(state.reducedHidden.shape, [1, 5, 1])
        XCTAssertEqual(state.reducedHidden.asArray(Float.self), [100, 1, 10, 17, 26])

        let processed = MLXArray([Float(101), 4, 13, 21, 30]).reshaped(1, 5, 1)
        let restored = map.restore(processed, state: state, updateScale: 1)
        MLX.eval(restored)
        XCTAssertEqual(restored.asArray(Float.self), [101, 3, 5, 13, 18, 24, 30])
    }

    func testTransformerQKVUsesConvertedGlobalSlabs() {
        let projected = MLXArray((0..<12).map(Float.init)).reshaped(1, 1, 12)
        let parts = miniMaxH3SplitProjectedQKV(projected, heads: 2, headDimension: 2)
        MLX.eval(parts)

        XCTAssertEqual(parts[0].asArray(Float.self), [0, 1, 2, 3])
        XCTAssertEqual(parts[1].asArray(Float.self), [4, 5, 6, 7])
        XCTAssertEqual(parts[2].asArray(Float.self), [8, 9, 10, 11])
    }

    func testPinnedMLXArtifactConfigurationDecodes() throws {
        let data = Data(#"""
        {
          "model_type": "minimax_h3",
          "partition": "fl2va",
          "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
          "quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
          "transformer": {
            "hidden_size": 5376,
            "num_layers": 50,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "ffn_hidden_size": 14336,
            "latents_dim": 24,
            "audio_latents_dim": 32,
            "text_dim": 5120,
            "time_embed_dim": 2688,
            "rope_inv_freq_len": 16
          }
        }
        """#.utf8)
        let configuration = try JSONDecoder().decode(MiniMaxH3Configuration.self, from: data)
        XCTAssertEqual(configuration.task, "fl2va")
        XCTAssertEqual(configuration.quantization?.bits, 8)
        XCTAssertEqual(configuration.textEncoderQuantization?.bits, 8)
        XCTAssertEqual(configuration.timeEmbeddingHiddenSize, 5_376)
        XCTAssertEqual(configuration.timeEmbeddingDimension, 2_688)
        XCTAssertTrue(configuration.validationIssues().isEmpty)
    }

    func testMixedTransformerAndTextEncoderQuantizationDecodes() throws {
        let data = Data(#"""
        {
          "model_type": "minimax_h3",
          "partition": "fl2va",
          "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
          "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
          "text_encoder_quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
          "transformer": {
            "hidden_size": 5376,
            "num_layers": 50,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "ffn_hidden_size": 14336,
            "latents_dim": 24,
            "audio_latents_dim": 32,
            "text_dim": 5120,
            "time_embed_dim": 2688,
            "rope_inv_freq_len": 16
          }
        }
        """#.utf8)
        let configuration = try JSONDecoder().decode(MiniMaxH3Configuration.self, from: data)
        XCTAssertEqual(configuration.quantization?.bits, 4)
        XCTAssertEqual(configuration.textEncoderQuantization?.bits, 8)
        XCTAssertTrue(configuration.validationIssues().isEmpty)
    }

    func testReleasedTemporalGeometry() throws {
        XCTAssertEqual(try MiniMaxH3Geometry.alignFrameCount(120), 124)
        XCTAssertEqual(try MiniMaxH3Geometry.videoLatentFrameCount(for: 124), 37)
        XCTAssertEqual(MiniMaxH3Geometry.audioLatentFrameCount(for: 124), 207)
        XCTAssertThrowsError(try MiniMaxH3Geometry.videoLatentFrameCount(for: 120))
    }

    func testFL2VAPackedLayoutRangesAndTags() throws {
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 0, 1],
            videoLatentFrames: 7,
            latentHeight: 4,
            latentWidth: 6,
            audioLatentFrames: 5,
            keyframeAnchors: [.first, .last]
        )
        XCTAssertEqual(layout.textRows, 0..<3)
        XCTAssertEqual(layout.conditionVideoRowCount, 12)
        XCTAssertEqual(layout.targetAudioRows.count, 10)
        XCTAssertEqual(layout.targetVideoRows.count, 42)
        XCTAssertEqual(layout.sequenceLength, 67)
        XCTAssertEqual(layout.positions.shape, [67, 3])
        XCTAssertEqual(layout.tokenTags[0..<3], [1, 0, 1])
        XCTAssertTrue(layout.tokenTags[3..<15].allSatisfy { $0 == MiniMaxH3Modality.video.rawValue })
    }

    func testKeyframeAnchorPreservesStringAndCodableCompatibility() throws {
        let anchors: [MiniMaxH3KeyframeAnchor] = [
            .first,
            .last,
            .history(latentFrameCount: 6),
            .frame(11),
        ]
        XCTAssertEqual(anchors.map(\.rawValue), ["first", "last", "history:6", "frame:11"])
        XCTAssertEqual(anchors.compactMap { MiniMaxH3KeyframeAnchor(rawValue: $0.rawValue) }, anchors)

        let encoded = try JSONEncoder().encode(anchors)
        XCTAssertEqual(try JSONDecoder().decode([MiniMaxH3KeyframeAnchor].self, from: encoded), anchors)
        XCTAssertEqual(
            try JSONDecoder().decode(MiniMaxH3KeyframeAnchor.self, from: Data("\"first\"".utf8)),
            .first
        )
    }

    func testFL2VAHistoryFrameAndAudioConditionsShareShiftedTargetTimeline() throws {
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: [
                .history(latentFrameCount: 2),
                .first,
                .frame(6),
                .last,
            ],
            audioConditionAnchors: [
                .history(latentFrameCount: 3),
                .first(latentFrameCount: 2),
            ]
        )
        XCTAssertEqual(layout.textRows, 0..<2)
        XCTAssertEqual(layout.conditionRows, 2..<32)
        XCTAssertEqual(layout.conditionVideoRowCount, 20)
        XCTAssertEqual(layout.conditionAudioRowCount, 10)
        XCTAssertEqual(layout.targetAudioRows, 32..<38)
        XCTAssertEqual(layout.targetVideoRows, 38..<46)
        XCTAssertEqual(layout.conditionSegments.map(\.modality), [.video, .audio])
        XCTAssertEqual(layout.conditionSegments.map(\.sourceRows), [0..<20, 0..<10])

        let positions = layout.positions.asArray(Float.self)
        func time(at row: Int) -> Float { positions[row * 3] }
        XCTAssertEqual(time(at: 2), 2, accuracy: 1e-5)
        XCTAssertEqual(time(at: 6), 2 + 5.0 / 3.0, accuracy: 1e-5)
        XCTAssertEqual(time(at: 10), 2 + 25.0 / 3.0, accuracy: 1e-5)
        XCTAssertEqual(time(at: 14), 2 + 55.0 / 3.0, accuracy: 1e-5)
        XCTAssertEqual(time(at: 18), 17, accuracy: 1e-5)
        XCTAssertEqual(time(at: 22), 2, accuracy: 1e-5)
        XCTAssertEqual(time(at: 28), 2 + 25.0 / 3.0, accuracy: 1e-5)
        XCTAssertEqual(time(at: layout.targetAudioRows.lowerBound), 2 + 25.0 / 3.0, accuracy: 1e-5)
        XCTAssertEqual(time(at: layout.targetVideoRows.lowerBound), 2 + 25.0 / 3.0, accuracy: 1e-5)
    }

    func testVideoAndAudioPackingRoundTrip() {
        let video = MLXArray(0..<384).asType(.float32).reshaped(1, 3, 2, 8, 8)
        let rows = MiniMaxH3Geometry.patchifyVideo(video)
        let roundTrip = MiniMaxH3Geometry.unpatchifyVideo(
            rows,
            frames: 2,
            height: 8,
            width: 8,
            channels: 3
        )
        MLX.eval(roundTrip)
        XCTAssertEqual(roundTrip.shape, video.shape)
        XCTAssertEqual(roundTrip.asArray(Float.self), video.asArray(Float.self))

        let audio = MLXArray(0..<40).asType(.float32).reshaped(1, 4, 2, 5)
        let audioRows = MiniMaxH3Geometry.packAudio(audio)
        let audioRoundTrip = MiniMaxH3Geometry.unpackAudio(audioRows)
        MLX.eval(audioRoundTrip)
        XCTAssertEqual(audioRoundTrip.shape, audio.shape)
        XCTAssertEqual(audioRoundTrip.asArray(Float.self), audio.asArray(Float.self))
    }

    func testVideoVAETilePlansPreserveCanvasAndMinimumOverlap() {
        XCTAssertEqual(MiniMaxH3VideoVAE.defaultSpatialTileSize, 256)
        for tileSize in [256, 304, 320] {
            for length in [480, 832, 1_344] {
                let plan = MiniMaxH3VideoVAE.tilePlan(length: length, tileSize: tileSize)
                XCTAssertEqual(plan.starts.first, 0)
                XCTAssertEqual(plan.starts.last! + plan.lengths.last!, length)
                XCTAssertTrue(plan.lengths.allSatisfy { $0 == tileSize || $0 == length })
                XCTAssertTrue(plan.overlaps.allSatisfy {
                    $0 >= MiniMaxH3VideoVAE.minimumSpatialTileOverlap
                })
            }
        }
    }

    func testRef2VAPackedLayoutPreservesOrderedReferenceBlocks() throws {
        let layout = try MiniMaxH3Geometry.buildRef2VA(
            textTokenTags: [1, 0],
            references: [
                .init(kind: .image, videoLatentFrames: 1, latentHeight: 4, latentWidth: 4),
                .init(
                    kind: .video,
                    videoLatentFrames: 2,
                    latentHeight: 4,
                    latentWidth: 6,
                    audioLatentFrames: 3
                ),
                .init(kind: .audio, audioLatentFrames: 2),
            ],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 4
        )
        XCTAssertEqual(layout.conditionVideoRowCount, 16)
        XCTAssertEqual(layout.conditionAudioRowCount, 10)
        XCTAssertEqual(layout.conditionRows, 2..<28)
        XCTAssertEqual(layout.targetAudioRows, 28..<36)
        XCTAssertEqual(layout.targetVideoRows, 36..<44)
        XCTAssertEqual(layout.conditionSegments.map(\.modality), [.video, .audio, .video, .audio])
        XCTAssertEqual(layout.conditionSegments.map(\.packedRows), [2..<6, 6..<12, 12..<24, 24..<28])
        XCTAssertEqual(layout.conditionSegments.map(\.sourceRows), [0..<4, 0..<6, 4..<16, 6..<10])
        XCTAssertTrue(layout.tokenTags[6..<12].allSatisfy { $0 == MiniMaxH3Modality.audio.rawValue })
        XCTAssertTrue(layout.tokenTags[12..<24].allSatisfy { $0 == MiniMaxH3Modality.video.rawValue })
    }

    func testRef2VAContinuationPrecedesReferencesAndShiftsTarget() throws {
        let layout = try MiniMaxH3Geometry.buildRef2VA(
            textTokenTags: [1, 0],
            references: [
                .init(kind: .image, videoLatentFrames: 1, latentHeight: 4, latentWidth: 4),
                .init(
                    kind: .video,
                    videoLatentFrames: 2,
                    latentHeight: 4,
                    latentWidth: 6,
                    audioLatentFrames: 3
                ),
                .init(kind: .audio, audioLatentFrames: 2),
            ],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 4,
            keyframeAnchors: [.history(latentFrameCount: 2), .first],
            audioConditionAnchors: [
                .history(latentFrameCount: 3),
                .first(latentFrameCount: 2),
            ]
        )

        XCTAssertEqual(layout.conditionVideoRowCount, 28)
        XCTAssertEqual(layout.conditionAudioRowCount, 20)
        XCTAssertEqual(layout.conditionRows, 2..<50)
        XCTAssertEqual(layout.targetAudioRows, 50..<58)
        XCTAssertEqual(layout.targetVideoRows, 58..<66)
        XCTAssertEqual(
            layout.conditionSegments.map(\.modality),
            [.video, .audio, .video, .audio, .video, .audio]
        )
        XCTAssertEqual(layout.conditionSegments.map(\.packedRows), [
            2..<14,
            14..<24,
            24..<28,
            28..<34,
            34..<46,
            46..<50,
        ])
        XCTAssertEqual(layout.conditionSegments.map(\.sourceRows), [
            0..<12,
            0..<10,
            12..<16,
            10..<16,
            16..<28,
            16..<20,
        ])

        let positions = layout.positions.asArray(Float.self)
        func time(at row: Int) -> Float { positions[row * 3] }
        let referenceEnd = Float(40.0 / 3.0)
        let targetOrigin = Float(65.0 / 3.0)
        XCTAssertEqual(time(at: 2), referenceEnd, accuracy: 1e-5)
        XCTAssertEqual(time(at: 10), targetOrigin, accuracy: 1e-5)
        XCTAssertEqual(time(at: 14), referenceEnd, accuracy: 1e-5)
        XCTAssertEqual(time(at: 20), targetOrigin, accuracy: 1e-5)
        XCTAssertEqual(time(at: layout.targetAudioRows.lowerBound), targetOrigin, accuracy: 1e-5)
        XCTAssertEqual(time(at: layout.targetVideoRows.lowerBound), targetOrigin, accuracy: 1e-5)
    }

    func testShiftedSchedulesTerminateAtCleanEndpoint() throws {
        let video = try MiniMaxH3Schedule(pointCount: 5, shift: 12)
        let audio = try MiniMaxH3Schedule(pointCount: 5, shift: 3)
        XCTAssertEqual(video.sigmas.first, 1)
        XCTAssertEqual(video.sigmas.last, 0)
        XCTAssertEqual(audio.sigmas.first, 1)
        XCTAssertEqual(audio.sigmas.last, 0)
        XCTAssertEqual(video.timesteps.count, 4)
        XCTAssertEqual(video.timesteps.first, 0)

        let sample = MLXArray([1, 2, 3, 4]).reshaped(1, 4)
        let velocity = MLXArray.ones([1, 4])
        let final = video.step(sample: sample, velocity: velocity, index: video.timesteps.count - 1)
        MLX.eval(final)
        XCTAssertTrue(final.asArray(Float.self).allSatisfy(\.isFinite))
    }


    func testFastH3QuantizedCompressionGateMatchesMLXAffineProjection() throws {
        let weight = (MLXArray(0..<320).reshaped(5, 64).asType(.bfloat16) - 160) / 64
        let input = (MLXArray(0..<128).reshaped(2, 64).asType(.bfloat16) - 64) / 32
        let (codes, scales, optionalBiases) = MLX.quantized(
            weight,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        let biases = try XCTUnwrap(optionalBiases)
        let gate = MiniMaxH3FastH3CompressionGate(
            codes: codes,
            scales: scales,
            biases: biases,
            groupSize: 64,
            bits: 8
        )
        let candidate = gate.project(input)
        let reference = MLX.quantizedMM(
            input,
            codes,
            scales: scales,
            biases: biases,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )
        MLX.eval(candidate, reference)
        XCTAssertEqual(candidate.shape, [2, 5])
        XCTAssertEqual(
            MLX.max(MLX.abs(candidate.asType(.float32) - reference.asType(.float32)))
                .item(Float.self),
            0
        )
    }

    func testExplicitDMDJumpScheduleRejectsInvalidBaseSigmas() {
        XCTAssertThrowsError(try MiniMaxH3Schedule(baseSigmas: [1, 0.5], shift: 12))
        XCTAssertThrowsError(try MiniMaxH3Schedule(baseSigmas: [0.5, 0.75, 0], shift: 12))
        XCTAssertThrowsError(try MiniMaxH3Schedule(baseSigmas: [0.999, 0], shift: 0))
    }

    func testFirstBlockChangeSeparatesGlobalAndTemporalAudioVideoDrift() throws {
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [MiniMaxH3Modality.text.rawValue, MiniMaxH3Modality.text.rawValue],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: []
        )
        let targetRowCount = layout.targetAudioRows.count + layout.targetVideoRows.count
        let hiddenSize = 2
        let previousValues = [Float](repeating: 1, count: targetRowCount * hiddenSize)
        var currentValues = previousValues

        for row in 0..<layout.targetAudioRows.count {
            for hidden in 0..<hiddenSize {
                currentValues[row * hiddenSize + hidden] = 1.02
            }
        }
        for row in layout.targetAudioRows.count..<targetRowCount {
            for hidden in 0..<hiddenSize {
                currentValues[row * hiddenSize + hidden] = 1.05
            }
        }
        let uniform = MiniMaxH3FirstBlockChange.measure(
            current: MLXArray(currentValues, [1, targetRowCount, hiddenSize]),
            previous: MLXArray(previousValues, [1, targetRowCount, hiddenSize]),
            layout: layout
        )
        XCTAssertEqual(uniform.videoGlobal, 0.05, accuracy: 1e-5)
        XCTAssertEqual(uniform.audioGlobal, 0.02, accuracy: 1e-5)
        XCTAssertEqual(uniform.videoTemporalMaximum, 0.05, accuracy: 1e-5)
        XCTAssertEqual(uniform.audioTemporalMaximum, 0.02, accuracy: 1e-5)

        currentValues = previousValues
        let audioFrames = layout.audioLatentFrames
        for row in [1, audioFrames + 1] {
            for hidden in 0..<hiddenSize {
                currentValues[row * hiddenSize + hidden] = 1.3
            }
        }
        let videoStart = layout.targetAudioRows.count
        let videoRowsPerFrame = layout.targetVideoRows.count / layout.videoLatentFrames
        for row in videoStart..<(videoStart + videoRowsPerFrame) {
            for hidden in 0..<hiddenSize {
                currentValues[row * hiddenSize + hidden] = 1.2
            }
        }
        let localized = MiniMaxH3FirstBlockChange.measure(
            current: MLXArray(currentValues, [1, targetRowCount, hiddenSize]),
            previous: MLXArray(previousValues, [1, targetRowCount, hiddenSize]),
            layout: layout
        )
        XCTAssertEqual(localized.videoGlobal, 0.10, accuracy: 1e-5)
        XCTAssertEqual(localized.videoTemporalMaximum, 0.20, accuracy: 1e-5)
        XCTAssertEqual(localized.audioGlobal, 0.10, accuracy: 1e-5)
        XCTAssertEqual(localized.audioTemporalMaximum, 0.30, accuracy: 1e-5)
    }

    func testExactScheduleCacheSurvivesDiscardingBF16AdaLNWeights() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 32,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 4,
            attentionHeadDimension: 8,
            feedForwardSize: 64,
            videoLatentChannels: 8,
            audioLatentChannels: 32,
            patchSize: [1, 2, 2],
            textDimension: 32,
            timeFrequencyDimension: 32,
            timeEmbeddingHiddenSize: 32,
            timeEmbeddingDimension: 32,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        let videoSchedule = try MiniMaxH3Schedule(pointCount: 4, shift: 12)
        let audioSchedule = try MiniMaxH3Schedule(pointCount: 4, shift: 3)
        let cache = model.precomputeAdaLN(
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: "test"
        )
        XCTAssertEqual(cache.stepCount, 3)
        let scheduleBytesBefore = model.parameters().flattened()
            .filter { $0.0.contains("adaln_proj") || $0.0.hasPrefix("time_embedder.") }
            .reduce(0) { $0 + $1.1.nbytes }
        XCTAssertGreaterThan(scheduleBytesBefore, 1_000)

        model.discardAdaLNWeights()

        let scheduleParametersAfter = model.parameters().flattened()
            .filter { $0.0.contains("adaln_proj") || $0.0.hasPrefix("time_embedder.") }
        XCTAssertTrue(scheduleParametersAfter.allSatisfy { $0.1.size == 1 })
        XCTAssertLessThan(scheduleParametersAfter.reduce(0) { $0 + $1.1.nbytes }, scheduleBytesBefore)
        XCTAssertEqual(cache.step(at: 0).blockModulations.count, 2)
    }

    func testTinyTransformerResidentBF16MatchesQuantizedExecution() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 32,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 4,
            attentionHeadDimension: 8,
            feedForwardSize: 64,
            videoLatentChannels: 8,
            audioLatentChannels: 32,
            patchSize: [1, 2, 2],
            textDimension: 32,
            timeFrequencyDimension: 32,
            timeEmbeddingHiddenSize: 32,
            timeEmbeddingDimension: 32,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        quantize(
            model: model,
            groupSize: 32,
            bits: 4,
            filter: { _, _ in true },
            apply: { module, groupSize, bits, mode in
                guard let quantized = quantizeSingle(
                    layer: module,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                ) as? QuantizedLinear else { return nil }
                return PortableQuantizedLinear(
                    weight: quantized.weight,
                    bias: quantized.bias,
                    scales: quantized.scales,
                    biases: quantized.biases,
                    groupSize: quantized.groupSize,
                    bits: quantized.bits,
                    mode: quantized.mode,
                    globalScale: quantized.globalScale
                )
            }
        )
        let quantizedCount = model.leafModules().flattened()
            .count(where: { $0.1 is QuantizedLinear })
        let estimatedBytes = model.estimatedResidentBF16ByteCount
        XCTAssertGreaterThan(quantizedCount, 0)
        XCTAssertGreaterThan(estimatedBytes, 0)

        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: [.first]
        )
        let video = MLXArray.zeros([1, 12, 32], dtype: .bfloat16)
        let audio = MLXArray.zeros([1, 6, 32], dtype: .bfloat16)
        let text = MLXArray.zeros([1, 2, 32], dtype: .bfloat16)
        let quantizedOutput = model(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(quantizedOutput.videoVelocityRows, quantizedOutput.audioVelocityRows)

        let materialized = model.materializeResidentBF16()
        XCTAssertTrue(model.usesResidentBF16)
        XCTAssertEqual(materialized.linearCount, quantizedCount)
        XCTAssertEqual(materialized.byteCount, estimatedBytes)
        XCTAssertFalse(model.leafModules().flattened().contains { $0.1 is QuantizedLinear })
        let denseOutput = model(
            videoRows: video,
            audioRows: audio,
            textStates: text,
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(denseOutput.videoVelocityRows, denseOutput.audioVelocityRows)
        XCTAssertLessThanOrEqual(
            MLX.abs(
                quantizedOutput.videoVelocityRows.asType(.float32)
                    - denseOutput.videoVelocityRows.asType(.float32)
            ).max().item(Float.self),
            0.05
        )
        XCTAssertLessThanOrEqual(
            MLX.abs(
                quantizedOutput.audioVelocityRows.asType(.float32)
                    - denseOutput.audioVelocityRows.asType(.float32)
            ).max().item(Float.self),
            0.05
        )
    }

    func testTinyDenseBF16TransformerIsActuallyMaterialized() {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 32,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 4,
            attentionHeadDimension: 8,
            feedForwardSize: 64,
            videoLatentChannels: 8,
            audioLatentChannels: 32,
            patchSize: [1, 2, 2],
            textDimension: 32,
            timeFrequencyDimension: 32,
            timeEmbeddingHiddenSize: 32,
            timeEmbeddingDimension: 32,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        let linearCount = model.leafModules().flattened().count { $0.1 is Linear }
        let estimatedBytes = model.estimatedResidentBF16ByteCount

        XCTAssertGreaterThan(linearCount, 0)
        XCTAssertGreaterThan(estimatedBytes, 0)
        XCTAssertFalse(model.usesResidentBF16)

        let materialized = model.materializeResidentBF16()

        XCTAssertTrue(model.usesResidentBF16)
        XCTAssertEqual(materialized.linearCount, linearCount)
        XCTAssertEqual(materialized.byteCount, estimatedBytes)
        XCTAssertFalse(model.leafModules().flattened().contains { $0.1 is QuantizedLinear })
    }

    func testTinyTransformerPreservesTargetShapes() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 12,
            layerCount: 2,
            refinerLayerCount: 1,
            attentionHeadCount: 2,
            attentionHeadDimension: 6,
            feedForwardSize: 16,
            videoLatentChannels: 3,
            audioLatentChannels: 4,
            patchSize: [1, 2, 2],
            textDimension: 10,
            timeFrequencyDimension: 4,
            timeEmbeddingHiddenSize: 12,
            timeEmbeddingDimension: 8,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3,
            keyframeAnchors: [.first]
        )
        let result = model(
            videoRows: MLXArray.zeros([1, 12, 12]),
            audioRows: MLXArray.zeros([1, 6, 4]),
            textStates: MLXArray.zeros([1, 2, 10]),
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(result.videoVelocityRows, result.audioVelocityRows)
        XCTAssertEqual(result.videoVelocityRows.shape, [1, 8, 12])
        XCTAssertEqual(result.audioVelocityRows.shape, [1, 6, 4])
    }

    func testTinyTransformerExecutesTokenReductionAcrossFullAndReducedBoundaries() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 12,
            layerCount: 4,
            refinerLayerCount: 1,
            attentionHeadCount: 2,
            attentionHeadDimension: 6,
            feedForwardSize: 16,
            videoLatentChannels: 3,
            audioLatentChannels: 4,
            patchSize: [1, 2, 2],
            textDimension: 10,
            timeFrequencyDimension: 4,
            timeEmbeddingHiddenSize: 12,
            timeEmbeddingDimension: 8,
            ropeFrequencyCount: 1
        )
        let model = MiniMaxH3Transformer(configuration: configuration)
        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: [1, 1],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 6,
            audioLatentFrames: 3,
            keyframeAnchors: []
        )
        let video = MLXArray.zeros([1, 12, 12])
        let audio = MLXArray.zeros([1, 6, 4])
        let text = MLXArray.zeros([1, 2, 10])
        let context = model.prepare(textStates: text, layout: layout)
        let reduction = model.prepareTokenReduction(context: context)
        let output = model.callWithTokenReduction(
            videoRows: video,
            audioRows: audio,
            context: context,
            reduction: reduction,
            timesteps: MLXArray([Float(0.2), 0.4, 0.999]),
            cachedAdaLN: nil,
            policy: MiniMaxH3TokenReductionPolicy(
                beginBlock: 1,
                endBlock: 2,
                earlyStepCount: 1,
                earlyEndBlock: 3
            ),
            stepIndex: 0
        )
        MLX.eval(output.videoVelocityRows, output.audioVelocityRows)

        XCTAssertEqual(reduction.reducedContext.layout.sequenceLength, 16)
        XCTAssertEqual(output.videoVelocityRows.shape, [1, 12, 12])
        XCTAssertEqual(output.audioVelocityRows.shape, [1, 6, 4])
        XCTAssertTrue(output.videoVelocityRows.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertTrue(output.audioVelocityRows.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testTinyTransformerAcceptsRef2VAConditionAudioAndVideo() throws {
        let configuration = MiniMaxH3TransformerConfiguration(
            hiddenSize: 12,
            layerCount: 1,
            refinerLayerCount: 1,
            attentionHeadCount: 2,
            attentionHeadDimension: 6,
            feedForwardSize: 16,
            videoLatentChannels: 3,
            audioLatentChannels: 4,
            patchSize: [1, 2, 2],
            textDimension: 10,
            timeFrequencyDimension: 4,
            timeEmbeddingHiddenSize: 12,
            timeEmbeddingDimension: 8,
            ropeFrequencyCount: 1
        )
        let layout = try MiniMaxH3Geometry.buildRef2VA(
            textTokenTags: [1, 1],
            references: [
                .init(
                    kind: .video,
                    videoLatentFrames: 1,
                    latentHeight: 4,
                    latentWidth: 4,
                    audioLatentFrames: 2
                ),
            ],
            videoLatentFrames: 2,
            latentHeight: 4,
            latentWidth: 4,
            audioLatentFrames: 3
        )
        let result = MiniMaxH3Transformer(configuration: configuration)(
            videoRows: MLXArray.zeros([1, 12, 12]),
            audioRows: MLXArray.zeros([1, 10, 4]),
            textStates: MLXArray.zeros([1, 2, 10]),
            layout: layout,
            videoTimestep: 0.2,
            audioTimestep: 0.4
        )
        MLX.eval(result.videoVelocityRows, result.audioVelocityRows)
        XCTAssertEqual(result.videoVelocityRows.shape, [1, 8, 12])
        XCTAssertEqual(result.audioVelocityRows.shape, [1, 6, 4])
    }


    func testTinyVideoDecoderShape() {
        let decoder = MiniMaxH3VideoDecoder(configuration: .init(
            latentChannels: 2,
            outputChannels: 3,
            patchSize: 2,
            temporalPatchSize: 2,
            layerCount: 1,
            headCount: 2,
            headDimension: 6,
            registerTokenCount: 1,
            feedForwardMultiplier: 2,
            rotaryDimensionRatio: 1
        ))
        let result = decoder(MLXArray.zeros([1, 2, 2, 2, 2]))
        MLX.eval(result)
        XCTAssertEqual(result.shape, [1, 3, 4, 4, 4])
    }
}
