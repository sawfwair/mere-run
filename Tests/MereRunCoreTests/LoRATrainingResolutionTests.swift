import XCTest
@testable import MereRunCore

final class LoRATrainingResolutionTests: XCTestCase {
    func testResolveWithoutClampRoundsDownToMultiple() throws {
        let resolved = try LoRATrainingResolution.resolve(
            width: 1031,
            height: 769,
            maxResolution: nil,
            multiple: 32,
            errorContext: "unit-test"
        )

        XCTAssertEqual(resolved.width, 1024)
        XCTAssertEqual(resolved.height, 768)
    }

    func testResolveWithClampPreservesAspectAndRoundsDown() throws {
        let resolved = try LoRATrainingResolution.resolve(
            width: 4000,
            height: 2000,
            maxResolution: 2048,
            multiple: 32,
            errorContext: "unit-test"
        )

        XCTAssertEqual(resolved.width, 2048)
        XCTAssertEqual(resolved.height, 1024)
    }

    func testResolveRejectsTooSmallAfterRounding() {
        XCTAssertThrowsError(
            try LoRATrainingResolution.resolve(
                width: 15,
                height: 15,
                maxResolution: nil,
                multiple: 16,
                errorContext: "tiny"
            )
        )
    }

    func testAllocateStepsDistributesAndSums() {
        let allocated = LoRATrainingResolution.allocateSteps(
            totalSteps: 100,
            bucketSizes: [2, 8, 10]
        )

        XCTAssertEqual(allocated.reduce(0, +), 100)
        XCTAssertEqual(allocated.count, 3)
        XCTAssertTrue(allocated.allSatisfy { $0 >= 1 })
        XCTAssertGreaterThanOrEqual(allocated[2], allocated[1])
        XCTAssertGreaterThanOrEqual(allocated[1], allocated[0])
    }

    func testAllocateStepsWithFewerStepsThanBucketsPicksLargestBuckets() {
        let allocated = LoRATrainingResolution.allocateSteps(
            totalSteps: 2,
            bucketSizes: [10, 5, 1]
        )

        XCTAssertEqual(allocated.reduce(0, +), 2)
        XCTAssertEqual(allocated, [1, 1, 0])
    }
}
