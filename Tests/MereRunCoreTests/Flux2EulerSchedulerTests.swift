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

    func testScalarSigmaShiftRaisesIntermediateSigmas() {
        let unshifted = Flux2EulerScheduler(
            numInferenceSteps: 4,
            numTrainTimesteps: 1000,
            imageSeqLen: 64 * 64,
            isDistilled: true
        )
        let shifted = Flux2EulerScheduler(
            numInferenceSteps: 4,
            numTrainTimesteps: 1000,
            imageSeqLen: 64 * 64,
            isDistilled: true,
            sigmaShift: 3.0
        )

        XCTAssertGreaterThan(shifted.sigma(at: 1).item(Float.self), unshifted.sigma(at: 1).item(Float.self))
        XCTAssertEqual(shifted.sigma(at: 4).item(Float.self), 0)
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

    func testFlux2DevEmpiricalMuMatchesPublishedFit() {
        XCTAssertEqual(
            Flux2EulerScheduler.computeEmpiricalMu(imageSeqLen: 4_096, numSteps: 50),
            2.02335,
            accuracy: 0.00001
        )
        XCTAssertEqual(
            Flux2EulerScheduler.computeEmpiricalMu(imageSeqLen: 5_000, numSteps: 50),
            1.3030167,
            accuracy: 0.00001
        )
    }

    func testCustomSigmasArePreservedAndReceiveTerminalZero() throws {
        let custom: [Float] = [1, 0.6509, 0.4374, 0.00031]
        try Flux2EulerScheduler.validateCustomSigmas(custom, expectedSteps: 4)
        let scheduler = Flux2EulerScheduler(
            numInferenceSteps: 4,
            imageSeqLen: 4_096,
            isDistilled: false,
            sigmaShift: 8,
            customSigmas: custom
        )

        XCTAssertEqual(scheduler.sigmas.asArray(Float.self), custom + [0])
        XCTAssertEqual(
            scheduler.timesteps.asArray(Float.self),
            custom.map { $0 * 1_000 }
        )
    }

    func testCustomSigmaValidationRejectsWrongCountAndOrder() {
        XCTAssertThrowsError(
            try Flux2EulerScheduler.validateCustomSigmas([1, 0.5], expectedSteps: 3)
        )
        XCTAssertThrowsError(
            try Flux2EulerScheduler.validateCustomSigmas([1, 0.5, 0.5], expectedSteps: 3)
        )
        XCTAssertThrowsError(
            try Flux2EulerScheduler.validateCustomSigmas([1, 0, -0.5], expectedSteps: 3)
        )
    }
}
