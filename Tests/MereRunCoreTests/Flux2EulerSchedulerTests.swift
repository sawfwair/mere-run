import XCTest
@testable import MereRunCore

final class Flux2EulerSchedulerTests: MereRunCoreTestCase {

    func testSingleStepDistilled() {
        let scheduler = Flux2EulerScheduler(
            numInferenceSteps: 1,
            numTrainTimesteps: 1000,
            imageSeqLen: 64 * 64,
            isDistilled: true
        )

        let sigma = scheduler.sigma(at: 0).item(Float.self)
        XCTAssertFalse(sigma.isNaN)
        XCTAssertFalse(sigma.isInfinite)
        XCTAssertGreaterThan(sigma, 0)
    }

    func testTwoStepDistilled() {
        let scheduler = Flux2EulerScheduler(
            numInferenceSteps: 2,
            numTrainTimesteps: 1000,
            imageSeqLen: 64 * 64,
            isDistilled: true
        )

        let sigma0 = scheduler.sigma(at: 0).item(Float.self)
        let sigma1 = scheduler.sigma(at: 1).item(Float.self)

        XCTAssertFalse(sigma0.isNaN)
        XCTAssertFalse(sigma1.isNaN)
        XCTAssertGreaterThan(sigma0, sigma1)
    }

    func testFourStepDistilled() {
        let scheduler = Flux2EulerScheduler(
            numInferenceSteps: 4,
            numTrainTimesteps: 1000,
            imageSeqLen: 64 * 64,
            isDistilled: true
        )

        XCTAssertEqual(scheduler.numInferenceSteps, 4)

        var prevSigma: Float = .infinity
        for i in 0..<4 {
            let sigma = scheduler.sigma(at: i).item(Float.self)
            XCTAssertFalse(sigma.isNaN)
            XCTAssertFalse(sigma.isInfinite)
            XCTAssertLessThanOrEqual(sigma, prevSigma)
            prevSigma = sigma
        }

        let finalSigma = scheduler.sigma(at: 4).item(Float.self)
        XCTAssertEqual(finalSigma, 0)
    }

    func testSingleStepBase() {
        let scheduler = Flux2EulerScheduler(
            numInferenceSteps: 1,
            numTrainTimesteps: 1000,
            imageSeqLen: 64 * 64,
            isDistilled: false
        )

        let sigma = scheduler.sigma(at: 0).item(Float.self)
        XCTAssertFalse(sigma.isNaN)
        XCTAssertFalse(sigma.isInfinite)
    }
}
