import XCTest
import MereRunCoreMLModel

final class EmbeddingGemma2CoreMLBatchTests: XCTestCase {
    func testPackingPreservesOrderAndSeparatesMediaAndCapacityBoundaries() {
        let counts = [32, 32, 192, 1, 256, 257, 127, 129, 10, 10, 10]
        var sequences = counts.map { MultimodalSequence(ids: Array(repeating: 2, count: $0)) }
        sequences[9] = MultimodalSequence(ids: sequences[9].ids, softTokens: Array(repeating: 0, count: 512), slots: [2])
        XCTAssertEqual(CoreMLTextBatching.groups(for: sequences), [[0, 1, 2], [3], [4], [5], [6, 7], [8], [9], [10]])
        XCTAssertEqual(CoreMLTextBatching.groups(for: []), [])
        XCTAssertEqual(CoreMLTextBatching.groups(for: Array(repeating: MultimodalSequence(ids: [2, 1]), count: 17)),
                       [Array(0..<8), Array(8..<16), [16]])
    }
}
