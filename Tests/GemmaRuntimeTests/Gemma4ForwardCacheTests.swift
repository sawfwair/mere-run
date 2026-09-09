import XCTest
import MLX
import MereRunMLXTestSupport
@testable import MereRunGemmaModel

final class Gemma4ForwardCacheTests: MLXTestCase {
    func testSpecializedSingleTokenDecodeDoesNotMaterializeDenseState() throws {
        let cache = SpecializedCache()
        let view = Gemma4ForwardAttentionCache(cache)
        let token = MLXArray.zeros([1, 1, 1, 4])
        view.append(keys: token, values: token)
        let output = try XCTUnwrap(view.specializedAttention(queries: token, repeats: 1, scale: 1))
        MLX.eval(output)

        XCTAssertEqual(view.offset, 3)
        XCTAssertEqual(cache.offset, 4)
        XCTAssertEqual(cache.denseReads, 0)
        XCTAssertEqual(cache.specializedCalls, 1)
    }

    private final class SpecializedCache: Gemma4AttentionCache {
        var offset = 3
        var denseReads = 0
        var specializedCalls = 0

        func currentState() -> (MLXArray, MLXArray)? {
            denseReads += 1
            return nil
        }

        func append(keys: MLXArray, values: MLXArray) {
            offset += keys.dim(2)
        }

        func specializedAttention(queries: MLXArray, repeats: Int, scale: Float) -> MLXArray? {
            specializedCalls += 1
            return MLXArray.zeros(queries.shape)
        }

        func fork() -> Gemma4AttentionCache {
            let copy = SpecializedCache()
            copy.offset = offset
            return copy
        }
    }
}
