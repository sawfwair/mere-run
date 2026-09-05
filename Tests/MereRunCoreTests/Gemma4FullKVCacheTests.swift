import MLX
import XCTest
@testable import MereRunCore

final class Gemma4FullKVCacheTests: MereRunCoreTestCase {
    func testPrefillGrowthPreservesEveryKeyAndValue() throws {
        // Semantic prompt boundaries need not align with the 256-token capacity
        // step. Growing after 175 tokens must retain room for all 337 new tokens.
        for chunks in [[175, 337, 1, 511], [255, 258], [256, 256, 1], [257, 512]] {
            let cache = Gemma4FullKVCache()
            let count = chunks.reduce(0, +)
            let keys = MLXArray((0..<(count * 8)).map(Float.init)).reshaped(1, 1, count, 8)
            let values = MLXArray((0..<(count * 4)).map { -Float($0) }).reshaped(1, 1, count, 4)
            var offset = 0

            for chunk in chunks {
                let end = offset + chunk
                cache.append(
                    keys: keys[0..., 0..., offset..<end, 0...],
                    values: values[0..., 0..., offset..<end, 0...]
                )
                cache.evaluateStorage()
                let state = try XCTUnwrap(cache.currentState())
                XCTAssertEqual(cache.offset, end)
                XCTAssertEqual(state.0.shape, [1, 1, end, 8])
                XCTAssertEqual(state.1.shape, [1, 1, end, 4])
                XCTAssertEqual(MLX.abs(state.0 - keys[0..., 0..., 0..<end, 0...]).max().item(Float.self), 0)
                XCTAssertEqual(MLX.abs(state.1 - values[0..., 0..., 0..<end, 0...]).max().item(Float.self), 0)
                offset = end
            }
        }
    }
}
