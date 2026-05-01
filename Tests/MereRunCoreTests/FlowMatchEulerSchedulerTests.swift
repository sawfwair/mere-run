import MLX
import XCTest
@testable import MereRunCore

final class FlowMatchEulerSchedulerTests: MereRunCoreTestCase {

    func testSigmaShapeAndTerminalZero() {
        let scheduler = FlowMatchEulerScheduler(numInferenceSteps: 4, numTrainTimesteps: 1000, shift: 1.0)

        XCTAssertEqual(scheduler.numInferenceSteps, 4)
        XCTAssertEqual(scheduler.sigmas.shape[0], 5)
        XCTAssertEqual(scheduler.timesteps.shape[0], 4)

        let terminal = scheduler.sigmas[4].item(Float.self)
        XCTAssertEqual(terminal, 0)
    }

    func testSigmasAreFiniteAndDecreasing() {
        let scheduler = FlowMatchEulerScheduler(numInferenceSteps: 8, numTrainTimesteps: 1000, shift: 1.0)

        var prev: Float = .infinity
        for i in 0..<scheduler.numInferenceSteps {
            let sigma = scheduler.sigma(at: i).item(Float.self)
            XCTAssertFalse(sigma.isNaN)
            XCTAssertFalse(sigma.isInfinite)
            XCTAssertGreaterThanOrEqual(sigma, 0)
            XCTAssertLessThanOrEqual(sigma, prev)
            prev = sigma
        }

        let terminal = scheduler.sigmas[scheduler.numInferenceSteps].item(Float.self)
        XCTAssertEqual(terminal, 0)
    }

    func testScaleNoiseUsesSigma0() {
        let scheduler = FlowMatchEulerScheduler(numInferenceSteps: 4, numTrainTimesteps: 1000, shift: 1.0)
        let noise = ones([2, 2], dtype: .float32)
        let scaled = scheduler.scaleNoise(noise, timestepIndex: 0)

        let sigma0 = scheduler.sigmas[0].item(Float.self)
        let element = scaled[0, 0].item(Float.self)
        XCTAssertEqual(element, sigma0)
    }
}
