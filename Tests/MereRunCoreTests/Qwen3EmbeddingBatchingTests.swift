import XCTest
@testable import MereRunCore

final class Qwen3EmbeddingBatchingTests: XCTestCase {
    func testAdaptiveSubbatchesStayWithinBudgetAndRestoreInputOrder() {
        let inputs = [
            tokenizedInput(index: 0, count: 4_000),
            tokenizedInput(index: 1, count: 100),
            tokenizedInput(index: 2, count: 100),
            tokenizedInput(index: 3, count: 4_000),
        ]

        let batches = Qwen3EmbeddingBatcher.subbatches(
            inputs: inputs,
            maxPaddedTokens: 8_000
        )
        XCTAssertEqual(batches.map { $0.map(\.originalIndex) }, [[0, 3], [1, 2]])
        XCTAssertTrue(batches.allSatisfy { batch in
            let longest = batch.map(\.tokenIDs.count).max() ?? 0
            return longest * batch.count <= 8_000
        })

        let outputs = Qwen3EmbeddingBatcher.mapInInputOrder(
            inputs: inputs,
            maxPaddedTokens: 8_000
        ) { batch in
            batch.map { "embedding-\($0.originalIndex)" }
        }
        XCTAssertEqual(outputs, ["embedding-0", "embedding-1", "embedding-2", "embedding-3"])
    }

    func testOversizeSingleInputRunsAloneWithoutReorderingOutputs() throws {
        let inputs = [
            tokenizedInput(index: 0, count: 32),
            tokenizedInput(index: 1, count: 9_000),
            tokenizedInput(index: 2, count: 16),
        ]

        let batches = Qwen3EmbeddingBatcher.subbatches(
            inputs: inputs,
            maxPaddedTokens: 8_192
        )
        let oversizeBatch = try XCTUnwrap(batches.first { batch in
            batch.contains { $0.originalIndex == 1 }
        })
        XCTAssertEqual(oversizeBatch.map(\.originalIndex), [1])
        XCTAssertEqual(oversizeBatch[0].tokenIDs.count, 9_000)

        let outputs = Qwen3EmbeddingBatcher.mapInInputOrder(
            inputs: inputs,
            maxPaddedTokens: 8_192
        ) { batch in
            batch.map(\.originalIndex)
        }
        XCTAssertEqual(outputs, [0, 1, 2])
    }

    private func tokenizedInput(
        index: Int,
        count: Int
    ) -> Qwen3EmbeddingTokenizedInput {
        Qwen3EmbeddingTokenizedInput(
            originalIndex: index,
            tokenIDs: Array(repeating: Int32(index), count: count)
        )
    }
}
