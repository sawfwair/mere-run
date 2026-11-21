import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunInkling2Model

#if os(macOS)
final class Inkling2FusedEpilogueTests: MLXTestCase {
    private func values(_ shape: [Int], stride: Int) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray((0..<count).map { Float(($0 * stride) % 109 - 53) / 64 }).reshaped(shape)
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(error, tolerance, file: file, line: line)
    }

    func testFusedHeadNormalizationPreservesBFloat16Boundaries() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu)
        let q = values([2, 2, 6, 128], stride: 7).asType(.bfloat16)
        let k = values([2, 2, 2, 128], stride: 13).asType(.bfloat16)
        let v = values([2, 2, 2, 128], stride: 17).asType(.bfloat16)
        let r = values([2, 2, 6, 16], stride: 19).asType(.bfloat16)
        let qScale = (1 + values([6, 128], stride: 11)).asType(.bfloat16)
        let rScale = (1 + values([6, 16], stride: 3)).asType(.bfloat16)
        let thresholds = MLXArray([Float(3), 1])
        let tau = MLXArray([Float(1), 1.25])
        let (qOut, kOut, vOut, rOut) = Inkling2HeadNormalization.apply(
            query: q, key: k, value: v, relative: r, queryScale: qScale, relativeScale: rScale,
            thresholds: thresholds, epsilon: 1e-6, tau: tau
        )
        let factors = tau.reshaped(1, 2, 1, 1)
        let expectedQ = (Inkling2Math.clip(q, scale: qScale, threshold: 3).asType(.float32) * factors).asType(.bfloat16)
        let expectedR = (Inkling2Math.clip(r, scale: rScale, threshold: 1).asType(.float32) * factors).asType(.bfloat16)
        assertClose(qOut, expectedQ, tolerance: 0.008)
        assertClose(rOut, expectedR, tolerance: 0.008)
        assertClose(kOut, Inkling2Math.clip(k, threshold: 3), tolerance: 0.004)
        assertClose(vOut, Inkling2Math.rms(v).asType(.bfloat16), tolerance: 0.016)
    }

    func testVerificationAttentionMatchesSerialCausalAndSlidingWindows() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu)
        let tokens = 4, width = 64, heads = 6, extent = 8, keyCount = 11
        let q = values([2, tokens, heads, width], stride: 7).asType(.bfloat16)
        let k = values([2, 2, keyCount, width], stride: 13).asType(.bfloat16)
        let v = values([2, 2, keyCount, width], stride: 17).asType(.bfloat16)
        let r = values([2, tokens, heads, 8], stride: 19).asType(.bfloat16)
        let profiles = values([8, extent], stride: 11).asType(.bfloat16)
        let sinks = values([2, tokens, heads], stride: 5).asType(.bfloat16)
        let gate = values([2, tokens, heads * width], stride: 3).asType(.bfloat16)
        for local in [false, true] {
            let fused = Inkling2DecodeAttention.apply(query: q, keys: k, values: v, relative: r,
                                                     profiles: profiles, sinks: sinks, gate: gate, local: local)
            let serial = (0..<tokens).map { token in
                let row = token..<(token + 1)
                let end = keyCount - tokens + token + 1
                let window = (local ? max(0, end - extent) : 0)..<end
                return Inkling2DecodeAttention.apply(query: q[0..., row], keys: k[0..., 0..., window], values: v[0..., 0..., window],
                                                    relative: r[0..., row], profiles: profiles, sinks: sinks[0..., row], gate: gate[0..., row])
            }
            assertClose(fused, concatenated(serial, axis: 1), tolerance: 0.003)
        }
    }

    func testFusedRouterUsesCorrectedRankingAndUncorrectedWeights() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu)
        let logits = values([2, 128], stride: 19) * 3
        let bias = MLXArray((0..<128).map { Float($0) / 359 })
        let (indices, routing) = Inkling2Router.select(logits, bias: bias, scale: 8)
        let raw = sigmoid(logits)
        let expectedIndices = argSort(-(raw + bias), axis: -1)[0..., ..<8]
        let selected = takeAlong(raw, expectedIndices, axis: -1)
        XCTAssertEqual(indices.asArray(UInt32.self), expectedIndices.asType(.uint32).asArray(UInt32.self))
        assertClose(routing, selected / selected.sum(axis: -1, keepDims: true) * 8, tolerance: 2e-6)
        let (ties, _) = Inkling2Router.select(MLXArray.zeros([1, 128]), bias: MLXArray.zeros([128]), scale: 8)
        XCTAssertEqual(ties.asArray(UInt32.self), (0..<8).map(UInt32.init))
    }

    func testBatchedCrossPassMessagesPreserveLayerAndControllerOrder() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu)
        let blocks = [values([1, 4, 128], stride: 7), values([1, 4, 128], stride: 13)]
        let partial = values([1, 4, 128], stride: 17)
        let queries = values([6, 128], stride: 3)
        let kvQueries = values([6, 128], stride: 19)
        let reference = Inkling2LoopMessages(
            coreStart: 6, blocks: blocks, partial: partial, queries: queries, keyValueQueries: kvQueries,
            fusedDecode: true, batched: nil
        )
        let outputs = Inkling2Math.residual(blocks, partial: partial, query: concatenated([queries, kvQueries]),
                                            fusedDecode: true)
        let batched = Inkling2LoopMessages(
            coreStart: 6, blocks: blocks, partial: partial, queries: queries, keyValueQueries: kvQueries,
            fusedDecode: true, batched: outputs
        )
        for layer in 6..<9 {
            for (actual, expected) in zip(batched.read(layer: layer), reference.read(layer: layer)) {
                XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
            }
        }
    }

    func testBatchedLoopControllersMatchIndependentProjections() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu)
        let channels = 3, width = 128, rank = 16
        let inputs = [7, 11, 13].map { values([2, 4, width], stride: $0) }
        let messages = [17, 23, 29].map { values([2, 4, width], stride: $0) }
        let down = (values([channels, 2 * width, rank], stride: 3) / sqrt(Float(width))).asType(.bfloat16)
        let up = values([channels, rank, 2 * width], stride: 19) / sqrt(Float(rank))
        let weights = (1 + values([channels, width], stride: 5)).asType(.bfloat16)
        let controller = Inkling2LoopController(down: down, up: up, normWeights: weights)
        let normalized = controller.apply(inputs, messages: messages, epsilon: 1e-6, dtype: .bfloat16)
        for channel in 0..<channels {
            let x = inputs[channel], m = messages[channel]
            let pair = concatenated([Inkling2Math.rms(x), Inkling2Math.rms(m)], axis: -1).asType(.bfloat16)
            let gates = tanh(matmul(matmul(pair, down[channel]).asType(.float32), up[channel]))
            let blended = (1 - gates[.ellipsis, ..<width]) * x + gates[.ellipsis, width...] * m
            let expected = (Inkling2Math.rms(blended) * weights[channel].asType(.float32)).asType(.bfloat16)
            assertClose(normalized[channel], expected, tolerance: 0.016)
        }
    }

    func testFusedNVFP4ProjectionsPreservePackedGatherRounding() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu)
        let rows = 64, tokens = 4, choices = 8
        let indices = MLXArray((0..<(tokens * choices)).map { Int32(($0 * 5 + $0 / choices) % 3) }).reshaped(tokens, choices)
        let outer = MLXArray([Float(0.13), 0.21, 0.37])
        let routing = softmax(values([tokens, choices], stride: 13), axis: -1)
        for width in [512, 1152, 2304] {
            let packed = MLXArray((0..<(3 * rows * width / 2)).map { UInt8(truncatingIfNeeded: ($0 * 43) ^ ($0 / 19)) })
                .reshaped(3, rows, width / 2).view(dtype: .uint32)
            let scales = MLXArray((0..<(3 * rows * width / 16)).map { UInt8(0x31 + ($0 * 17) % 13) })
                .reshaped(3, rows, width / 16)
            let weights = Inkling2PackedExperts.Projection(packed: packed, scales: scales, expertScales: outer)
            let inputs = values([tokens, width], stride: 7).asType(.bfloat16)
            let gates = weights.rawMatmul(inputs.reshaped(tokens, 1, 1, width), indices: indices)
            let expected = Inkling2ExpertEpilogues.activate(gates, indices: indices, scales: outer,
                                                           dtype: .bfloat16, gateCap: 4, upCap: 25)
            let fused = Inkling2ExpertEpilogues.fusedGate(inputs, packed: packed, scales: scales, indices: indices,
                                                         outer: outer, gateCap: 4, upCap: 25)
            XCTAssertEqual(fused.asType(.float32).asArray(Float.self), expected.asType(.float32).asArray(Float.self))
            let hidden = values([tokens, choices, 1, width], stride: 19).asType(.bfloat16)
            let projected = weights.rawMatmul(hidden, indices: indices)
            let reference = Inkling2ExpertEpilogues.reduce(projected, indices: indices, scales: outer,
                                                           routing: routing, dtype: .bfloat16)
            let reduced = Inkling2ExpertEpilogues.fusedDown(hidden, packed: packed, scales: scales, indices: indices,
                                                           outer: outer, routing: routing)
            XCTAssertEqual(reduced.asType(.float32).asArray(Float.self), reference.asType(.float32).asArray(Float.self))
        }
    }

    func testExpertEpiloguesMatchScaledAndUnscaledReference() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu)
        let indices = MLXArray([UInt32(1), 0, 1, 0, 1, 0]).reshaped(2, 3)
        let routing = MLXArray([Float(0.1), 0.3, 0.6, 0.2, 0.5, 0.3]).reshaped(2, 3)
        for scaled in [false, true] {
            let scales = MLXArray([Float(0.125), 0.75])
            let factor = expandedDimensions(take(scales, indices, axis: 0), axes: [-1, -2])
            let rawGateUp = (values([2, 3, 1, 64], stride: 13) * 7).asType(scaled ? .float32 : .bfloat16)
            let rawDown = (values([2, 3, 1, 32], stride: 17) * 2).asType(scaled ? .float32 : .bfloat16)
            let gateUp = (scaled ? rawGateUp * factor : rawGateUp).asType(.bfloat16).reshaped(2, 3, 1, 32, 2)
            let expectedActivation = Inkling2Math.activation(gate: gateUp[.ellipsis, 0], up: gateUp[.ellipsis, 1],
                                                            gateCap: 4, upCap: 25)
            let activation = Inkling2ExpertEpilogues.activate(rawGateUp, indices: indices, scales: scaled ? scales : nil,
                                                              dtype: .bfloat16, gateCap: 4, upCap: 25)
            assertClose(activation, expectedActivation, tolerance: 0.032)
            let down = (scaled ? rawDown * factor : rawDown).asType(.bfloat16).squeezed(axis: -2)
            let expectedReduced = (down.asType(.float32) * routing.expandedDimensions(axis: -1))
                .sum(axis: 1).asType(.bfloat16)
            let reduced = Inkling2ExpertEpilogues.reduce(rawDown, indices: indices, scales: scaled ? scales : nil,
                                                        routing: routing, dtype: .bfloat16)
            assertClose(reduced, expectedReduced, tolerance: 0.008)
        }
    }
}
#endif
