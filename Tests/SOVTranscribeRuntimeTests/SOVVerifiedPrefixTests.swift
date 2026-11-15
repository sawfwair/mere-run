import XCTest
import MLX
import MereRunMLXTestSupport
@testable import MereRunGemmaModel
@testable import AudioSOVTranscribeModel

final class SOVVerifiedPrefixTests: MLXTestCase {
    private func rows(_ values: [Float]) -> MLXArray { MLXArray(values).reshaped(1, 1, values.count, 1) }

    func testCommitRetainsHistoryAcrossWrappedSlidingAndFullCaches() throws {
        for window in [nil, 4, 8] as [Int?] {
            for count in 1...3 {
                let base: Gemma4AttentionCache = window.map { Gemma4SlidingKVCache(maxSize: $0) } ?? Gemma4FullKVCache()
                for index in 0..<12 {
                    base.append(keys: rows([Float(index)]), values: rows([Float(index)*10]))
                }
                let trial = base.fork()
                trial.append(keys: rows([100, 101, 102]), values: rows([1000, 1010, 1020]))
                XCTAssertTrue(SOVTranscribeDecoder.commitVerifiedPrefix([base], from: [trial], proposedCount: 3, committedCount: count))
                XCTAssertEqual(base.offset, 12+count)
                XCTAssertEqual(trial.offset, 15)
                let expected = (Array(0..<12).map(Float.init) + Array([Float(100), 101, 102].prefix(count))).suffix(window ?? (12+count))
                let state = try XCTUnwrap(base.currentState())
                XCTAssertEqual(state.0.asArray(Float.self), Array(expected))
                XCTAssertEqual(state.1.asArray(Float.self), expected.map { $0*10 })
                base.append(keys: rows([777]), values: rows([7770]))
                let advanced = try XCTUnwrap(base.currentState())
                XCTAssertEqual(advanced.0.asArray(Float.self), Array((Array(expected)+[777]).suffix(window ?? (13+count))))
                XCTAssertEqual(trial.offset, 15)
            }
        }
    }

    func testTooShortTrialWindowLeavesEveryBaseCacheUntouched() throws {
        let base: [Gemma4AttentionCache] = [Gemma4FullKVCache(), Gemma4SlidingKVCache(maxSize: 2)]
        for entry in base {
            for index in 0..<8 { entry.append(keys: rows([Float(index)]), values: rows([Float(index)])) }
        }
        let before = try base.map { try XCTUnwrap($0.currentState()).0.asArray(Float.self) }
        let trial = base.map { $0.fork() }
        for entry in trial {
            for value in [Float(100), 101, 102] { entry.append(keys: rows([value]), values: rows([value])) }
        }
        XCTAssertFalse(SOVTranscribeDecoder.commitVerifiedPrefix(base, from: trial, proposedCount: 3, committedCount: 1))
        XCTAssertEqual(base.map(\.offset), [8, 8])
        XCTAssertEqual(try base.map { try XCTUnwrap($0.currentState()).0.asArray(Float.self) }, before)
    }
}
