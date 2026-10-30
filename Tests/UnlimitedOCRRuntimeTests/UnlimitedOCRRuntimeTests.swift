import Foundation
import MLX
import XCTest
import MereRunMLXTestSupport
@testable import MereRunUnlimitedOCRModel

final class UnlimitedOCRRuntimeTests: MLXTestCase {
    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    }

    func testDASHScalingPreservesPythonScalarPromotion() throws {
        struct Reference: Decodable {
            let dimension: Int
            let scale: Float
            let input: [Float]
            let summary: [Float]
            let routing: [Float]
        }
        let ref = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: fixtures.appendingPathComponent("router-scaling.json")))
        let scores = MLXArray(ref.input)
        XCTAssertEqual(UnlimitedOCRMath.scaleRouterScores(scores, dimension: ref.dimension).asArray(Float.self), ref.summary)
        XCTAssertEqual(UnlimitedOCRMath.scaleRouterScores(scores, dimension: ref.dimension, factor: ref.scale).asArray(Float.self), ref.routing)
    }

    func testReleasedConfigurationAndCheckpointShapes() throws {
        let data = try Data(contentsOf: fixtures.appendingPathComponent("config.json"))
        let config = try UnlimitedOCRConfiguration.load(data)
        XCTAssertEqual(config.numHiddenLayers, 12)
        XCTAssertEqual(config.nRoutedExperts, 64)
        XCTAssertEqual(UnlimitedOCRCheckpoint.shapes.count, 2819)
        XCTAssertEqual(UnlimitedOCRCheckpoint.shapes["model.layers.11.mlp.experts.63.down_proj.weight"], [1280, 896])
        XCTAssertEqual(UnlimitedOCRCheckpoint.shapes["model.sam_model.blocks.2.attn.rel_pos_h"], [127, 64])
        XCTAssertEqual(UnlimitedOCRCheckpoint.shapes["model.sam_model.blocks.3.attn.rel_pos_h"], [27, 64])
        XCTAssertThrowsError(try UnlimitedOCRCheckpoint.validate([:]))
        let wrong = String(decoding: data, as: UTF8.self).replacingOccurrences(of: "\"hidden_size\": 1280", with: "\"hidden_size\": 2560")
        XCTAssertThrowsError(try UnlimitedOCRConfiguration.load(Data(wrong.utf8)))
    }

    func testPageCoordinatesContinueAcrossPromptAndResponse() throws {
        var state = try UnlimitedOCRPageState(tokens: [0, 128827, 128815, 128815, 128827, 128815, 7, 8])
        XCTAssertEqual(state.pagePositions, [0, 1, 1, 1, 2, 2, 0, 0])
        XCTAssertEqual(state.tokenPositions, [0, 0, 1, 2, 0, 1, 1, 2])
        XCTAssertEqual(state.queryPages, [-1, 0, 0, 0, 1, 1, -1, -1])
        state.append(128827)
        state.append(9)
        state.append(128827)
        state.append(10)
        XCTAssertEqual(Array(state.pagePositions.suffix(4)), [1, 1, 2, 2])
        XCTAssertEqual(Array(state.tokenPositions.suffix(4)), [3, 4, 2, 3])
        XCTAssertEqual(Array(state.queryPages.suffix(4)), [0, 0, 1, 1])
    }

    func testEntmaxAgainstAnalyticProbabilities() {
        let result = UnlimitedOCRMath.entmax(MLXArray([Float(1), 0, -10, 0, 0, 0]).reshaped(2, 3))
        let expected: [Float] = [0.8307189, 0.1692811, 0, 1 / 3, 1 / 3, 1 / 3]
        XCTAssertLessThan(abs(result - MLXArray(expected).reshaped(2, 3)).max().item(Float.self), 1e-6)
    }

    func testVisionNormsPreserveTheirDistinctBF16Rounding() {
        let x = MLXArray([Float(1), 2, 4, 7]).asType(.bfloat16)
        let weight = MLXArray([Float(1.1), 0.9, 1.3, 0.8]).asType(.bfloat16)
        let bias = MLXArray([Float(0.01), 0.03, -0.02, 0.07]).asType(.bfloat16)
        // Analytic mean=3.5 and variance=5.25. Fused LayerNorm rounds after
        // FP32 affine math; SAM's custom neck rounds each intermediate to BF16.
        let fused = UnlimitedOCRMath.norm(x, weight: weight, bias: bias, eps: 1e-6)
        XCTAssertEqual(fused.asType(.float32).asArray(Float.self), [-1.1953125, -0.55859375, 0.263671875, 1.296875])
        let weights = UnlimitedOCRWeights(tensors: ["neck.weight": weight, "neck.bias": bias])
        let neck = weights.channelNorm(x, "neck")
        XCTAssertEqual(neck.asType(.float32).asArray(Float.self), [-1.1875, -0.5546875, 0.263671875, 1.2890625])
    }

    func testIntrinsicActivationsMatchPyTorchBF16Rounding() {
        // PyTorch 2.13 CPU and MPS agree on these BF16 intrinsic outputs.
        let x = MLXArray([Float(0.1), 0.5, 1, 1.5, 2, -0.1, -0.5, -1, -1.5, -2]).asType(.bfloat16)
        let gelu: [Float] = [0.053955078125, 0.345703125, 0.83984375, 1.3984375, 1.953125,
                             -0.046142578125, -0.154296875, -0.158203125, -0.10009765625, -0.04541015625]
        let silu: [Float] = [0.052490234375, 0.310546875, 0.73046875, 1.2265625, 1.7578125,
                             -0.047607421875, -0.1884765625, -0.26953125, -0.2734375, -0.23828125]
        XCTAssertEqual(UnlimitedOCRMath.gelu(x).asType(.float32).asArray(Float.self), gelu)
        XCTAssertEqual(UnlimitedOCRMath.silu(x).asType(.float32).asArray(Float.self), silu)
    }

    func testQuickGELUKeepsTheScalarInFP32AndRoundsAtUpstreamBoundaries() {
        let x = MLXArray([Float(-0.1), -0.5, -1, -1.5, -2, -2.4375, 0.1, 0.5, 1, 1.5, 2, 2.4375]).asType(.bfloat16)
        let expected: [Float] = [-0.045654296875, -0.1494140625, -0.154296875, -0.109375, -0.06396484375, -0.03759765625,
                                  0.054443359375, 0.349609375, 0.84765625, 1.390625, 1.9375, 2.40625]
        XCTAssertEqual(UnlimitedOCRMath.quickGelu(x).asType(.float32).asArray(Float.self), expected)
    }

    func testPositionalGridUsesUpstreamAntialiasedHorizontalThenVerticalPasses() throws {
        struct Reference: Decodable { let positionInput: [Float]; let positionOutput: [Float] }
        let ref = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: fixtures.appendingPathComponent("precision.json")))
        let input = MLXArray(ref.positionInput).reshaped(1, 7, 5, 3).asType(.bfloat16)
        let output = UnlimitedOCRMath.resizeGrid(input, height: 4, width: 3)
        XCTAssertEqual(output.asType(.float32).asArray(Float.self), ref.positionOutput)
    }

    func testBF16SparsePrefillAndDecodeMatchUpstreamPrecisionBoundaries() throws {
        struct Reference: Decodable {
            let count: Int
            let q: [Float]; let k: [Float]; let v: [Float]
            let indexQ: [Float]; let indexK: [Float]; let head: [Float]
            let prefill: [Float]; let decodePositions: [Int]; let decode: [[Float]]
        }
        let ref = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: fixtures.appendingPathComponent("precision.json")))
        let q = MLXArray(ref.q).reshaped(1, ref.count, 4).asType(.bfloat16)
        let k = MLXArray(ref.k).reshaped(1, ref.count, 4).asType(.bfloat16)
        let v = MLXArray(ref.v).reshaped(1, ref.count, 4).asType(.bfloat16)
        let iq = MLXArray(ref.indexQ).reshaped(ref.count, 4)
        let ik = MLXArray(ref.indexK).reshaped(ref.count, 4)
        // The image remains in the local/future region, so page filtering falls
        // back to the unrestricted remote set used by this upstream fixture.
        let state = try UnlimitedOCRPageState(tokens: [128827] + [Int](repeating: 1, count: ref.count - 2) + [128815])
        let output = UnlimitedOCRAttention.attend(q: q, k: k, v: v, indexQ: iq, indexK: ik,
            head: MLXArray(ref.head), state: state, offset: 0, scale: 4, floor: 0.0001, sigma: 1,
            blockSize: 2, localBlocks: 2, budget: 2, topM: 3)
        // The independent oracle uses FP32 accumulation after upstream's BF16
        // mask rounding. BF16 SDPA kernels vary across Metal and CPU; bound
        // prefill error by 1% of the input-value scale. Decode stays FP32 and
        // must match its BF16-rounded oracle exactly below.
        let expected = MLXArray(ref.prefill).reshaped(1, ref.count, 4)
        XCTAssertEqual(output.dtype, .bfloat16)
        XCTAssertLessThan(abs(output.asType(.float32) - expected).max().item(Float.self),
                          Float(0.01) * abs(v.asType(.float32)).max().item(Float.self))
        for (index, position) in ref.decodePositions.enumerated() {
            let end = position + 1
            let output = UnlimitedOCRAttention.attend(q: q[0..., position..<end, 0...],
                k: k[0..., ..<end, 0...], v: v[0..., ..<end, 0...], indexQ: iq[position..<end], indexK: ik[..<end],
                head: MLXArray(ref.head), state: state, offset: position, scale: 4, floor: 0.0001, sigma: 1,
                blockSize: 2, localBlocks: 2, budget: 2, topM: 3)
            XCTAssertEqual(output.asType(.float32).asArray(Float.self), ref.decode[index])
        }
    }

    func testSAMRelativePositionInterpolationAgainstScalarValues() {
        let table = MLXArray([Float(0), 1, 2]).reshaped(3, 1)
        let positions = UnlimitedOCRVision.relativePositions(table, size: 3)
        let expected: [Float] = [1, 0.4, 0, 1.6, 1, 0.4, 2, 1.6, 1]
        XCTAssertLessThan(abs(positions - MLXArray(expected).reshaped(3, 3, 1)).max().item(Float.self), 1e-6)
    }

    func testSAMAttentionKeepsSpatialIndicesIntegral() {
        // Zero Q/K and identity V give uniform attention over four spatial cells.
        let zeros = MLXArray.zeros([768, 768])
        let weights = UnlimitedOCRWeights(tensors: [
            "attention.qkv.weight": concatenated([zeros, zeros, MLXArray.eye(768)], axis: 0),
            "attention.qkv.bias": MLXArray.zeros([2304]),
            "attention.proj.weight": MLXArray.eye(768),
            "attention.proj.bias": MLXArray.zeros([768]),
            "attention.rel_pos_h": MLXArray.zeros([3, 64]),
            "attention.rel_pos_w": MLXArray.zeros([3, 64])
        ])
        let input = broadcast(MLXArray([Float(1), 2, 3, 4]).reshaped(1, 2, 2, 1), to: [1, 2, 2, 768])
        let output = UnlimitedOCRVision(weights: weights).samAttention(input, path: "attention")
        XCTAssertLessThan(abs(output - Float(2.5)).max().item(Float.self), 1e-6)
    }

    func testVisionLinearFusesBiasBeforeBF16OutputRounding() {
        let weights = UnlimitedOCRWeights(tensors: [
            "projection.weight": MLXArray([Float(1.1), -0.9, 0.2, 0.3]).reshaped(1, 4).asType(.bfloat16),
            "projection.bias": MLXArray([Float(0.01)]).asType(.bfloat16)
        ])
        let x = MLXArray([Float(1), 2, 4, 7]).reshaped(1, 4).asType(.bfloat16)
        XCTAssertEqual(weights.linear(x, "projection").asType(.float32).item(Float.self), 2.21875)
    }

    func testDASHRouterUsesGemmaNormOffsetAndFP32AffineMath() {
        let x = MLXArray([Float(1), 2, 4, 7])
        let result = UnlimitedOCRMath.routerRMS(x, weight: MLXArray([Float(0), 1, -0.5, 0.25]))
        let scale: Float = 1 / sqrt(17.5 + 1e-6)
        let expected = MLXArray([Float(1), 4, 2, 8.75]) * scale
        XCTAssertLessThan(abs(result - expected).max().item(Float.self), 1e-6)
        let saturated = UnlimitedOCRMath.routerRMS(x, weight: MLXArray.full([4], values: MLXArray(Float(1e8))))
        XCTAssertEqual(saturated.asArray(Float.self), [10000, 10000, 10000, 10000])
    }

    func testMoEMixtureAccumulatesInFP32WithoutRoundingRouterProbabilities() {
        let values = MLXArray([Float(1), 2, 4]).reshaped(1, 3, 1).asType(.bfloat16)
        let probabilities = MLXArray([Float(0.21), 0.32, 0.45]).reshaped(1, 3)
        let mixed = UnlimitedOCRMath.mixExperts(values, probabilities: probabilities)
        XCTAssertEqual(mixed.dtype, .bfloat16)
        XCTAssertEqual(mixed.asType(.float32).item(Float.self), 2.65625)
    }

    func testPageRotaryParityAndScalarAngles() {
        let input = MLXArray((0..<24).map { Float($0 + 1) / 10 }).reshaped(1, 3, 8)
        let result = UnlimitedOCRMath.rope(input, tokens: MLXArray([Int32(0), 2, 2]), pages: MLXArray([Int32(0), 1, 3]))
        var expected = [Float]()
        let values = input.asArray(Float.self)
        for row in 0..<3 {
            for dim in 0..<8 {
                let pair = dim % 4
                let angle: Float = row == 0 ? 0 : (pair == 3 ? 1 : 2 * pow(10000, -Float(pair) / 3))
                let rotated = dim < 4 ? -values[row * 8 + dim + 4] : values[row * 8 + dim - 4]
                expected.append(values[row * 8 + dim] * cos(angle) + rotated * sin(angle))
            }
        }
        XCTAssertLessThan(abs(result - MLXArray(expected).reshaped(1, 3, 8)).max().item(Float.self), 2e-6)
    }

    func testSparseAttentionAgainstIndependentScalarReference() throws {
        struct Reference: Decodable {
            let tokens: [Int]
            let q: [Float]
            let k: [Float]
            let v: [Float]
            let indexQ: [Float]
            let indexK: [Float]
            let head: [Float]
            let output: [Float]
        }
        let ref = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: fixtures.appendingPathComponent("attention.json")))
        let state = try UnlimitedOCRPageState(tokens: ref.tokens)
        let count = ref.tokens.count
        let q = MLXArray(ref.q).reshaped(1, count, 4)
        let k = MLXArray(ref.k).reshaped(1, count, 4)
        let v = MLXArray(ref.v).reshaped(1, count, 4)
        let iq = MLXArray(ref.indexQ).reshaped(count, 4)
        let ik = MLXArray(ref.indexK).reshaped(count, 4)
        let expected = MLXArray(ref.output).reshaped(1, count, 4)
        let output = UnlimitedOCRAttention.attend(q: q, k: k, v: v, indexQ: iq, indexK: ik,
            head: MLXArray(ref.head), state: state, offset: 0, scale: 4, floor: 0.0001, sigma: 1,
            blockSize: 2, localBlocks: 2, budget: 2, topM: 3)
        XCTAssertLessThan(abs(output - expected).max().item(Float.self), 2e-5)
        // Single-token decode must use the same read set as full prefill, including a partial final block.
        for position in [7, 17, count - 1] {
            let end = position + 1
            let output = UnlimitedOCRAttention.attend(q: q[0..., position..<end, 0...],
                k: k[0..., ..<end, 0...], v: v[0..., ..<end, 0...],
                indexQ: iq[position..<end], indexK: ik[..<end], head: MLXArray(ref.head),
                state: state, offset: position, scale: 4, floor: 0.0001, sigma: 1,
                blockSize: 2, localBlocks: 2, budget: 2, topM: 3)
            XCTAssertLessThan(abs(output - expected[0..., position..<end, 0...]).max().item(Float.self), 2e-5)
        }
    }

    func testBF16AttentionPreservesUpstreamOutputDType() throws {
        let zeros = MLXArray.zeros([1, 3, 4], dtype: .bfloat16)
        let values = broadcast(MLXArray([Float(1), 2, 3]).reshaped(1, 3, 1), to: [1, 3, 4]).asType(.bfloat16)
        let state = try UnlimitedOCRPageState(tokens: [0, 128827, 128815])
        let result = UnlimitedOCRAttention.attend(
            q: zeros, k: zeros, v: values, indexQ: MLXArray.zeros([3, 4]), indexK: MLXArray.zeros([3, 4]),
            head: MLXArray.zeros([4]), state: state, offset: 0, scale: 4, floor: 0.0001, sigma: 1,
            blockSize: 2, localBlocks: 1, budget: 1, topM: 1
        )
        XCTAssertEqual(result.dtype, .bfloat16)
        let expected = broadcast(MLXArray([Float(1), 1.5, 2]).reshaped(1, 3, 1), to: [1, 3, 4])
        XCTAssertLessThan(abs(result.asType(.float32) - expected).max().item(Float.self), 1e-6)
    }

    func testTinyMoEDecoderAgainstIndependentScalarReferenceAndCachedSteps() throws {
        struct Tensor: Decodable { let shape: [Int]; let values: [Float] }
        struct Reference: Decodable {
            let config: UnlimitedOCRConfiguration
            let tensors: [String: Tensor]
            let tokens: [Int]
            let inputs: [[Float]]
            let logits: [[Float]]
        }
        let parser = JSONDecoder()
        parser.keyDecodingStrategy = .convertFromSnakeCase
        let ref = try parser.decode(Reference.self, from: Data(contentsOf: fixtures.appendingPathComponent("decoder.json")))
        let tensors = ref.tensors.mapValues { MLXArray($0.values).reshaped($0.shape) }
        let model = UnlimitedOCRDecoder(config: ref.config, tensors: tensors)
        let input = MLXArray(ref.inputs.flatMap { $0 }).reshaped(-1, 8)
        let state = try UnlimitedOCRPageState(tokens: ref.tokens)
        let full = model.forward(input, state: state, cache: (0..<2).map { _ in UnlimitedOCRCache() })
        XCTAssertLessThan(abs(full - MLXArray(ref.logits.last ?? [])).max().item(Float.self), 2e-5)
        let cache = (0..<2).map { _ in UnlimitedOCRCache() }
        var incremental = try UnlimitedOCRPageState(tokens: Array(ref.tokens.prefix(5)))
        var logits = model.forward(input[..<5], state: incremental, cache: cache)
        XCTAssertLessThan(abs(logits - MLXArray(ref.logits[4])).max().item(Float.self), 2e-5)
        for row in 5..<ref.tokens.count {
            incremental.append(ref.tokens[row])
            logits = model.forward(input[row..<(row + 1)], state: incremental, cache: cache)
            XCTAssertLessThan(abs(logits - MLXArray(ref.logits[row])).max().item(Float.self), 2e-5)
            XCTAssertEqual(cache[0].keys?.dim(1), row + 1)
        }
    }
}
