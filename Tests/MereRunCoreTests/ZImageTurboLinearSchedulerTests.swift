import MLX
import XCTest
@testable import MereRunCore

final class ZImageTurboLinearSchedulerTests: MereRunCoreTestCase {

    func testSigmaShapeAndTerminalZero() {
        let config = ZImageTurboInferenceConfig(width: 512, height: 512, numInferenceSteps: 4)
        let scheduler = ZImageTurboLinearScheduler(config: config, requiresSigmaShift: false)

        XCTAssertEqual(scheduler.sigmas.shape[0], 5)
        XCTAssertEqual(scheduler.timesteps.shape[0], 4)

        let terminal = scheduler.sigmas[4].item(Float.self)
        XCTAssertEqual(terminal, 0)
    }

    func testLinearSigmasWithoutShift() {
        let config = ZImageTurboInferenceConfig(width: 512, height: 512, numInferenceSteps: 2)
        let scheduler = ZImageTurboLinearScheduler(config: config, requiresSigmaShift: false)

        let sigma0 = scheduler.sigmas[0].item(Float.self)
        let sigma1 = scheduler.sigmas[1].item(Float.self)
        let sigma2 = scheduler.sigmas[2].item(Float.self)

        XCTAssertEqual(sigma0, 1)
        XCTAssertEqual(sigma1, 0.5)
        XCTAssertEqual(sigma2, 0)

        let t0 = scheduler.timesteps[0].item(Float.self)
        let t1 = scheduler.timesteps[1].item(Float.self)
        XCTAssertEqual(t0, 1000)
        XCTAssertEqual(t1, 500)
    }

    func testShiftedSigmasAreFiniteAndDecreasing() {
        let config = ZImageTurboInferenceConfig(width: 1024, height: 1024, numInferenceSteps: 4)
        let scheduler = ZImageTurboLinearScheduler(config: config, requiresSigmaShift: true, sigmaShift: nil)

        var prev: Float = .infinity
        for i in 0..<4 {
            let sigma = scheduler.sigmas[i].item(Float.self)
            XCTAssertFalse(sigma.isNaN)
            XCTAssertFalse(sigma.isInfinite)
            XCTAssertGreaterThanOrEqual(sigma, 0)
            XCTAssertLessThanOrEqual(sigma, prev)
            prev = sigma
        }

        let terminal = scheduler.sigmas[4].item(Float.self)
        XCTAssertEqual(terminal, 0)
    }
}
