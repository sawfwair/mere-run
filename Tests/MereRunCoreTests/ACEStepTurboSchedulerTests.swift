import XCTest
@testable import MereRunCore

final class ACEStepTurboSchedulerTests: XCTestCase {

    func testFixNFE8ShiftTimestepsMatchReference() {
        let expectedShift1: [Float] = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
        let expectedShift2: [Float] = [
            1.0,
            0.9333333333333333,
            0.8571428571428571,
            0.7692307692307693,
            0.6666666666666666,
            0.5454545454545454,
            0.4,
            0.2222222222222222,
        ]
        let expectedShift3: [Float] = [
            1.0,
            0.9545454545454546,
            0.9,
            0.8333333333333334,
            0.75,
            0.6428571428571429,
            0.5,
            0.3,
        ]

        XCTAssertEqual(ACEStepTurboScheduler(fixNFE: 8, shift: 1.0).timesteps.count, 8)
        assertClose(
            ACEStepTurboScheduler(fixNFE: 8, shift: 1.0).timesteps,
            expectedShift1
        )

        XCTAssertEqual(ACEStepTurboScheduler(fixNFE: 8, shift: 2.0).timesteps.count, 8)
        assertClose(
            ACEStepTurboScheduler(fixNFE: 8, shift: 2.0).timesteps,
            expectedShift2
        )

        XCTAssertEqual(ACEStepTurboScheduler(fixNFE: 8, shift: 3.0).timesteps.count, 8)
        assertClose(
            ACEStepTurboScheduler(fixNFE: 8, shift: 3.0).timesteps,
            expectedShift3
        )
    }

    func testCustomTimestepsTrimAndMapToNearestValid() {
        let sched = ACEStepTurboScheduler(fixNFE: 8, shift: 3.0, timesteps: [1.0, 0.95, 0.0])
        XCTAssertEqual(sched.timesteps.count, 2)
        XCTAssertEqual(sched.timesteps.first ?? 0, 1.0, accuracy: 1e-6)
        XCTAssertEqual(sched.timesteps[1], 0.9545454545454546, accuracy: 1e-6)
    }

    func testCustomTimestepsUseReferenceCandidateSetIndependentOfFixNFE() {
        let sched = ACEStepTurboScheduler(fixNFE: 12, shift: 1.0, timesteps: [0.95])
        XCTAssertEqual(sched.timesteps, [0.95454545])
    }

    private func assertClose(_ got: [Float], _ expected: [Float], accuracy: Float = 1e-6) {
        XCTAssertEqual(got.count, expected.count)
        for (g, e) in zip(got, expected) {
            XCTAssertEqual(g, e, accuracy: accuracy)
        }
    }
}
