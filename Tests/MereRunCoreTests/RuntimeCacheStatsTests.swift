import XCTest
@testable import MereRunCore

final class RuntimeCacheStatsTests: XCTestCase {
    func testPrefillCheckpointPlannerStopsAtSemanticCheckpointInsideChunk() {
        let end = RuntimePrefillCheckpointPlanner.nextEnd(
            processed: 0,
            total: 1_000,
            chunkSize: 512,
            checkpoints: [128]
        )

        XCTAssertEqual(end, 128)
    }

    func testPrefillCheckpointPlannerFallsBackToChunkAfterCheckpoint() {
        let end = RuntimePrefillCheckpointPlanner.nextEnd(
            processed: 128,
            total: 1_000,
            chunkSize: 512,
            checkpoints: [128]
        )

        XCTAssertEqual(end, 640)
    }

    func testPrefillCheckpointPlannerNormalizesUsableInteriorCounts() {
        let checkpoints = RuntimePrefillCheckpointPlanner.normalizedCheckpoints(
            [-1, 0, 32, 32, 1_000],
            total: 1_000
        )

        XCTAssertEqual(checkpoints, [32])
    }

    func testPrefillCheckpointPlannerStoresOnlySemanticAndFinalCheckpoints() {
        let semantic = Set([768])

        XCTAssertNil(RuntimePrefillCheckpointPlanner.storagePriority(
            tokenCount: 512,
            total: 1_280,
            semanticCheckpoints: semantic
        ))
        XCTAssertEqual(RuntimePrefillCheckpointPlanner.storagePriority(
            tokenCount: 768,
            total: 1_280,
            semanticCheckpoints: semantic
        ), .semantic)
        XCTAssertEqual(RuntimePrefillCheckpointPlanner.storagePriority(
            tokenCount: 1_280,
            total: 1_280,
            semanticCheckpoints: semantic
        ), .chunk)
    }

    func testPrefixCacheRetentionPrunesChunkBeforeSemanticCheckpoint() {
        let base = Date(timeIntervalSince1970: 1_000)
        let key = RuntimePrefixCacheRetentionPlanner.keyToPrune(entries: [
            "semantic-old": RuntimePrefixCacheRetentionMetadata(
                priority: .semantic,
                lastAccess: base
            ),
            "chunk-new": RuntimePrefixCacheRetentionMetadata(
                priority: .chunk,
                lastAccess: base.addingTimeInterval(100)
            ),
            "chunk-old": RuntimePrefixCacheRetentionMetadata(
                priority: .chunk,
                lastAccess: base.addingTimeInterval(50)
            ),
        ])

        XCTAssertEqual(key, "chunk-old")
    }

    func testPrefixCacheRetentionFallsBackToOldestSemanticWhenAllAreSemantic() {
        let base = Date(timeIntervalSince1970: 2_000)
        let key = RuntimePrefixCacheRetentionPlanner.keyToPrune(entries: [
            "semantic-new": RuntimePrefixCacheRetentionMetadata(
                priority: .semantic,
                lastAccess: base.addingTimeInterval(10)
            ),
            "semantic-old": RuntimePrefixCacheRetentionMetadata(
                priority: .semantic,
                lastAccess: base
            ),
        ])

        XCTAssertEqual(key, "semantic-old")
    }

    func testDecodeBatchPlannerSelectsEarliestCompatibleSignatureGroup() {
        let rows = [
            RuntimeDecodeBatchRowMetadata(row: "a", signature: "offset:4", position: 4),
            RuntimeDecodeBatchRowMetadata(row: "b", signature: "offset:8", position: 8),
            RuntimeDecodeBatchRowMetadata(row: "c", signature: "offset:8", position: 8),
            RuntimeDecodeBatchRowMetadata(row: "d", signature: "offset:8", position: 8),
            RuntimeDecodeBatchRowMetadata(row: "e", signature: "offset:4", position: 4),
        ]

        XCTAssertEqual(RuntimeDecodeBatchPlanner.selectRows(rows), ["a", "e"])
    }

    func testDecodeBatchPlannerSelectsLargestGroupAtEarliestPosition() {
        let rows = [
            RuntimeDecodeBatchRowMetadata(row: "a", signature: "offset:4-full", position: 4),
            RuntimeDecodeBatchRowMetadata(row: "b", signature: "offset:4-linear", position: 4),
            RuntimeDecodeBatchRowMetadata(row: "c", signature: "offset:4-linear", position: 4),
            RuntimeDecodeBatchRowMetadata(row: "d", signature: "offset:4-linear", position: 4),
            RuntimeDecodeBatchRowMetadata(row: "e", signature: "offset:4-full", position: 4),
        ]

        XCTAssertEqual(RuntimeDecodeBatchPlanner.selectRows(rows), ["b", "c", "d"])
    }

    func testDecodeBatchPlannerCatchesUpEarliestRowBeforeLaterCompatibleBatch() {
        let rows = [
            RuntimeDecodeBatchRowMetadata(row: "short", signature: "offset:4", position: 4),
            RuntimeDecodeBatchRowMetadata(row: "long-a", signature: "offset:8", position: 8),
            RuntimeDecodeBatchRowMetadata(row: "long-b", signature: "offset:8", position: 8),
        ]

        XCTAssertEqual(RuntimeDecodeBatchPlanner.selectRows(rows), ["short"])
    }

    func testDecodeBatchPlannerAllowsVariablePositionRowsWithCompatibleSignature() {
        let rows = [
            RuntimeDecodeBatchRowMetadata(row: "short", signature: "linear-shape", position: 4),
            RuntimeDecodeBatchRowMetadata(row: "middle", signature: "linear-shape", position: 8),
            RuntimeDecodeBatchRowMetadata(row: "long", signature: "linear-shape", position: 12),
        ]

        XCTAssertEqual(RuntimeDecodeBatchPlanner.selectRows(rows), ["short", "middle", "long"])
    }

    func testDecodeBatchPositionKindCountsVariablePositionBatch() {
        XCTAssertEqual(RuntimeDecodeBatchPositionKind.variablePositionBatchCount([4, 4, 4]), 0)
        XCTAssertEqual(RuntimeDecodeBatchPositionKind.variablePositionBatchCount([4, 8]), 1)
    }

    func testDecodeBatchPlannerCatchesUpShortestRowWhenNoRowsAreCompatible() {
        let rows = [
            RuntimeDecodeBatchRowMetadata(row: "long", signature: "offset:12", position: 12),
            RuntimeDecodeBatchRowMetadata(row: "short", signature: "offset:4", position: 4),
            RuntimeDecodeBatchRowMetadata(row: "middle", signature: "offset:8", position: 8),
        ]

        XCTAssertEqual(RuntimeDecodeBatchPlanner.selectRows(rows), ["short"])
    }

    func testDecodeBatchPlannerPreservesSingleEligibleRow() {
        let rows = [
            RuntimeDecodeBatchRowMetadata(row: "only", signature: "offset:4", position: 4),
        ]

        XCTAssertEqual(RuntimeDecodeBatchPlanner.selectRows(rows), ["only"])
    }
}
