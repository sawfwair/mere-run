import XCTest
@testable import MereRunCore

final class ACEStepContinuousSchedulerTests: MereRunCoreTestCase {
    func testSuppliedTimestepsAreUsedInFull() {
        // The continuous schedule is not quantized onto a reference table, so a caller
        // supplying more than 20 timesteps must get all of them back. Truncating here
        // silently stops a 50-step run a third of the way through its trajectory.
        let supplied: [Float] = (0..<50).map { 1 - Float($0) / 50 }

        let scheduler = ACEStepContinuousScheduler(
            inferenceSteps: 50,
            shift: 3,
            timesteps: supplied
        )

        XCTAssertEqual(scheduler.timesteps.count, 50)
        XCTAssertEqual(scheduler.timesteps, supplied)
    }

    func testTrailingZeroTimestepsAreTrimmed() {
        let scheduler = ACEStepContinuousScheduler(
            inferenceSteps: 4,
            timesteps: [1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
        )

        XCTAssertEqual(scheduler.timesteps, [1.0, 0.75, 0.5, 0.25])
    }

    func testEmptySuppliedTimestepsFallBackToGeneratedSchedule() {
        let scheduler = ACEStepContinuousScheduler(
            inferenceSteps: 8,
            shift: 3,
            timesteps: [0.0]
        )

        XCTAssertEqual(
            scheduler.timesteps,
            ACEStepContinuousScheduler.makeTimesteps(inferenceSteps: 8, shift: 3)
        )
    }

}
