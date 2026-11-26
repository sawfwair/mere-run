import XCTest
import MLX
import MereRunMLXTestSupport
@testable import MereRunInkling2Model

final class Inkling2BandedPrefillTests: MLXTestCase {
    func testGlobalBandedPrefillMatchesDenseAttentionWithRaggedTiles() throws {
        #if os(macOS)
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Banded attention uses Metal.")
        try XCTSkipUnless(GPU.deviceInfo().architecture == "applegpu_g17s", "Banded attention requires NAX.")
        let rows = 33, heads = 6, kvHeads = 2, width = 128, extent = 1_024
        func input(_ shape: [Int], seed: UInt64, scale: Float = 0.5) -> MLXArray {
            (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale).asType(.bfloat16)
        }
        let query = input([1, rows, heads, width], seed: 1)
        let relative = input([1, rows, heads, extent], seed: 2, scale: 1).asType(.float32)
        let sinks = input([1, heads, rows], seed: 3).asType(.float32).transposed(0, 2, 1)
        for length in [rows, 257, 8_213] {
            let keys = input([1, length, kvHeads, width], seed: 4).transposed(0, 2, 1, 3)
            let values = input([1, length, kvHeads, width], seed: 5).transposed(0, 2, 1, 3)
            let mask = Inkling2AttentionMask(.init(offset: length - rows, queryCount: rows, keyStart: 0,
                                                   keyCount: length, extent: extent, local: false))
            let indices = broadcast(mask.relativeIndices, to: [1, heads, rows, length])
            let relativeInHeadOrder = relative.transposed(0, 2, 1, 3)
            let bias = MLX.where(mask.withinExtent, takeAlong(relativeInHeadOrder, indices, axis: -1), MLXArray(Float(0)))
            let expected = Inkling2Math.groupedAttention(query: query, keys: keys, values: values, bias: bias,
                                                         allowed: mask.allowed, sinks: sinks)
            let actual = Inkling2BandedPrefill.apply(query: query, keys: keys, values: values,
                                                     relative: relative, sinks: sinks)
            XCTAssertEqual(actual.shape, expected.shape)
            let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
            XCTAssertLessThan(error, 0.01, "\(length) keys")
        }
        #else
        throw XCTSkip("Banded attention uses Metal.")
        #endif
    }
}
