import MLX
import MereRunMLXTestSupport
import XCTest
@testable import AudioSOVTranscribeModel
@testable import MereRunGemmaModel

final class SOVRollbackCacheTests: MLXTestCase {
    private func rows(_ start: Int, _ count: Int, width: Int) -> MLXArray {
        MLXArray((0..<(2 * count * width)).map { Float(start * 100 + $0) }).reshaped(1, 2, count, width)
    }

    func testRollbackPreservesEveryCommittedPrefixAcrossCompaction() throws {
        for window in [nil, 2, 4, 32] as [Int?] {
            let cache = SOVRollbackKVCache(window: window)
            let reference: Gemma4AttentionCache = window.map { Gemma4SlidingKVCache(maxSize: $0) } ?? Gemma4FullKVCache()
            for iteration in 0..<100 {
                let k = rows(iteration, 3, width: 4), v = rows(iteration, 3, width: 2)
                cache.append(keys: k, values: v); reference.append(keys: k, values: v)
                let previous = try XCTUnwrap(cache.currentState())
                cache.beginTrial(count: 16)
                let trialK = rows(1000 + iteration, 16, width: 4), trialV = rows(1000 + iteration, 16, width: 2)
                let attended = try XCTUnwrap(cache.attentionState(appending: trialK, values: trialV))
                XCTAssertEqual(attended.0.asArray(Float.self), concatenated([previous.0, trialK], axis: 2).asArray(Float.self))
                let accepted = iteration % 17
                cache.commit(count: accepted)
                if accepted > 0 {
                    reference.append(keys: trialK[0..., 0..., ..<accepted, 0...], values: trialV[0..., 0..., ..<accepted, 0...])
                }
                XCTAssertEqual(cache.offset, reference.offset)
                let actual = try XCTUnwrap(cache.currentState()), expected = try XCTUnwrap(reference.currentState())
                // An oversized trial retains its committed prefix, not rejected rows.
                XCTAssertEqual(actual.0.asArray(Float.self), expected.0.asArray(Float.self))
                XCTAssertEqual(actual.1.asArray(Float.self), expected.1.asArray(Float.self))
            }
        }
    }
}
