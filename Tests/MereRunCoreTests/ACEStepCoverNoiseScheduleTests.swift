import MLX
import XCTest
@testable import MereRunCore

final class ACEStepCoverNoiseScheduleTests: MereRunCoreTestCase {
    func testDisabledCoverNoiseKeepsPureNoiseSchedule() {
        let noise = MLXArray([Float(10), 20, 30, 40], [1, 2, 2]).asType(.float32)
        let source = MLXArray([Float(2), 4, 6, 8], [1, 2, 2]).asType(.float32)
        let timesteps: [Float] = [1.0, 0.75, 0.5, 0.25]

        let prepared = ACEStepPipeline.prepareCoverNoiseSchedule(
            noise: noise,
            sourceLatents: source,
            timesteps: timesteps,
            coverNoiseStrength: 0.0
        )

        XCTAssertEqual(prepared.timesteps, timesteps)
        XCTAssertEqual(prepared.latents.asArray(Float.self), [10, 20, 30, 40])
    }

    func testCoverNoiseStartsFromNearestRenoisedSourceLatents() {
        let noise = MLXArray([Float(10), 20, 30, 40], [1, 2, 2]).asType(.float32)
        let source = MLXArray([Float(2), 4, 6, 8], [1, 2, 2]).asType(.float32)
        let timesteps: [Float] = [1.0, 0.75, 0.5, 0.25]

        let prepared = ACEStepPipeline.prepareCoverNoiseSchedule(
            noise: noise,
            sourceLatents: source,
            timesteps: timesteps,
            coverNoiseStrength: 0.5
        )

        XCTAssertEqual(prepared.timesteps, [0.5, 0.25])
        XCTAssertEqual(prepared.latents.asArray(Float.self), [6, 12, 18, 24])
    }
}
