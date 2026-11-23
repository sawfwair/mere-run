import XCTest
import MLX
import MereRunMLXTestSupport
@testable import MereRunInkling2Model

final class Inkling2SplitAttentionTests: MLXTestCase {
    func testSplitVerificationMatchesGroupedAttentionAcrossWindowBoundary() throws {
        #if os(macOS)
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Split attention uses Metal.")
        let rows = 4, heads = 24, kvHeads = 8, width = 128, length = 1_097, extent = 1_024
        func input(_ shape: [Int], seed: UInt64, scale: Float) -> MLXArray {
            (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale).asType(.bfloat16)
        }
        let query = input([1, rows, heads, width], seed: 1, scale: 0.5)
        let keys = input([1, length, kvHeads, width], seed: 2, scale: 0.5).transposed(0, 2, 1, 3)
        let past = input([1, length - rows, kvHeads, width], seed: 3, scale: 0.5)
        let tail = input([1, rows, kvHeads, width], seed: 4, scale: 5)
        let values = concatenated([past, tail], axis: 1).transposed(0, 2, 1, 3)
        let states = input([1, rows, heads, 16], seed: 5, scale: 0.5)
        let profiles = input([16, extent], seed: 6, scale: 0.25)
        let sinks = input([1, heads, rows], seed: 7, scale: 0.5).asType(.float32).transposed(0, 2, 1)
        let gate = input([1, rows, heads * width], seed: 8, scale: 0.5)
        let scores = Inkling2Math.groupedAttentionDots(query: query, keys: keys)
        let relative = matmul(states.asType(.float32), profiles.asType(.float32))
        for local in [false, true] {
            let weights = Inkling2MaskedAttention.probabilities(
                scores: scores, relative: relative, sinks: sinks, divisor: sqrt(Float(width)),
                dtype: .bfloat16, local: local
            )
            let attended = Inkling2Math.groupedAttentionValues(weights, values: values)
            let expected = (attended.asType(.float32) * sigmoid(gate.asType(.float32))).asType(.bfloat16)
            let actual = Inkling2SplitAttention.apply(
                query: query, keys: keys, values: values, states: states, profiles: profiles, sinks: sinks,
                gate: gate, window: local ? extent : 0, packQueryRows: true
            )
            XCTAssertEqual(actual.shape, expected.shape)
            let difference = abs(actual.asType(.float32) - expected.asType(.float32))
            XCTAssertLessThan(difference.max().item(Float.self), 0.008)
        }
        #else
        throw XCTSkip("Split attention uses Metal.")
        #endif
    }

    func testBankedVerificationUsesLogicalLengthWithOneGraph() throws {
        #if os(macOS)
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Split attention uses Metal.")
        let rows = 5, heads = 24, kvHeads = 8, width = 128, capacity = 1152, extent = 1024
        func input(_ shape: [Int], seed: UInt64) -> MLXArray {
            (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * 0.5).asType(.bfloat16)
        }
        let query = input([1, rows, heads, width], seed: 1)
        let keys = input([1, capacity, kvHeads, width], seed: 2).transposed(0, 2, 1, 3)
        let values = input([1, capacity, kvHeads, width], seed: 3).transposed(0, 2, 1, 3)
        let states = input([1, rows, heads, 16], seed: 4)
        let profiles = input([16, extent], seed: 5)
        let sinks = input([1, heads, rows], seed: 6).asType(.float32).transposed(0, 2, 1)
        let gate = input([1, rows, heads * width], seed: 7)
        let pendingK = input([1, rows, kvHeads, width], seed: 8).transposed(0, 2, 1, 3)
        let pendingV = input([1, rows, kvHeads, width], seed: 9).transposed(0, 2, 1, 3)
        eval(query, keys, values, states, profiles, sinks, gate, pendingK, pendingV)
        var traces = 0
        let compiled = MLX.compile { inputs in
            traces += 1
            return [Inkling2SplitAttention.apply(
                query: query, keys: keys, values: values, states: states, profiles: profiles,
                sinks: sinks, gate: gate, window: 0, packQueryRows: true,
                bank: .init(position: inputs[0], keys: pendingK, values: pendingV)
            )]
        }
        for position in [1023, 1024, 1091, capacity] {
            let completeK = concatenated([keys[0..., 0..., ..<position, 0...], pendingK], axis: 2)
            let completeV = concatenated([values[0..., 0..., ..<position, 0...], pendingV], axis: 2)
            let expected = Inkling2SplitAttention.apply(query: query, keys: completeK, values: completeV,
                                                        states: states, profiles: profiles, sinks: sinks, gate: gate, window: 0,
                                                        packQueryRows: true)
            let actual = compiled([MLXArray(Int32(position))])[0]
            XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self), "Committed position \(position)")
        }
        XCTAssertEqual(traces, 1)
        #else
        throw XCTSkip("Split attention uses Metal.")
        #endif
    }
}
