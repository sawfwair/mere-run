import MLX
import XCTest
@testable import MereRunCore

final class TextSFTTrainingBatchTests: MereRunCoreTestCase {
    func testShiftedTargetExampleMasksOnlyTargetTokens() throws {
        let example = try TextSFTTrainingBatchBuilder.shiftedTargetExample(
            prefixTokenIds: [10, 11, 12, 13],
            targetTokenIds: [20, 21, 22],
            maxSequenceLength: 16
        )

        XCTAssertEqual(example.inputTokenIds, [10, 11, 12, 13, 20, 21])
        XCTAssertEqual(example.labelTokenIds, [11, 12, 13, 20, 21, 22])
        XCTAssertEqual(example.lossMask, [0, 0, 0, 1, 1, 1])
    }

    func testShiftedTargetExamplePreservesTargetMaskAfterLeftTruncation() throws {
        let example = try TextSFTTrainingBatchBuilder.shiftedTargetExample(
            prefixTokenIds: [1, 2, 3, 4],
            targetTokenIds: [5, 6, 7],
            maxSequenceLength: 5
        )

        XCTAssertEqual(example.inputTokenIds, [3, 4, 5, 6])
        XCTAssertEqual(example.labelTokenIds, [4, 5, 6, 7])
        XCTAssertEqual(example.lossMask, [0, 1, 1, 1])
    }

    func testShiftedExampleMasksOnlyAssistantTargets() throws {
        let example = try TextSFTTrainingBatchBuilder.shiftedExample(
            messageTokenIds: [[10, 11], [20, 21], [30, 31, 32]],
            messageRoles: [.system, .user, .assistant],
            maxSequenceLength: 16
        )

        XCTAssertEqual(example.inputTokenIds, [10, 11, 20, 21, 30, 31])
        XCTAssertEqual(example.labelTokenIds, [11, 20, 21, 30, 31, 32])
        XCTAssertEqual(example.lossMask, [0, 0, 0, 1, 1, 1])
    }

    func testShiftedExampleKeepsAssistantTargetsAfterLeftTruncation() throws {
        let example = try TextSFTTrainingBatchBuilder.shiftedExample(
            messageTokenIds: [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
            messageRoles: [.system, .user, .assistant],
            maxSequenceLength: 5
        )

        XCTAssertEqual(example.inputTokenIds, [5, 6, 7, 8])
        XCTAssertEqual(example.labelTokenIds, [6, 7, 8, 9])
        XCTAssertEqual(example.lossMask, [1, 1, 1, 1])
    }

    func testMakeBatchPadsInputsLabelsAndMasks() throws {
        let first = TextSFTTokenizedExample(
            inputTokenIds: [1, 2],
            labelTokenIds: [2, 3],
            lossMask: [0, 1]
        )
        let second = TextSFTTokenizedExample(
            inputTokenIds: [4],
            labelTokenIds: [5],
            lossMask: [1]
        )

        let batch = try TextSFTTrainingBatchBuilder.makeBatch([first, second], padTokenId: 99)

        XCTAssertEqual(batch.inputIds.shape, [2, 2])
        XCTAssertEqual(batch.labels.shape, [2, 2])
        XCTAssertEqual(batch.lossMask.shape, [2, 2])
        XCTAssertEqual(batch.inputIds[1, 1].item(Int32.self), 99)
        XCTAssertEqual(batch.labels[1, 1].item(Int32.self), 99)
        XCTAssertEqual(batch.lossMask[1, 1].item(Float.self), 0)
    }

    func testMaskedCrossEntropyIgnoresMaskedTokens() {
        let logits = MLXArray([
            Float(10), 0, 0,
            0, 0, Float(10),
        ], [1, 2, 3])
        let labels = MLXArray([Int32(1), Int32(2)], [1, 2])
        let mask = MLXArray([Float(0), Float(1)], [1, 2])

        let loss = TextSFTTrainingLoss.maskedNextTokenCrossEntropy(
            logits: logits,
            labels: labels,
            lossMask: mask
        )

        XCTAssertLessThan(loss.item(Float.self), 0.001)
    }
}
