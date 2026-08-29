import Foundation
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

    func testQwen2511FourStepScheduleMatchesPinnedDiffusersReference() throws {
        let scheduler = FlowMatchEulerScheduler(
            config: try Self.qwen2511SchedulerConfig(),
            numInferenceSteps: 4,
            imageSeqLen: 4_096
        )
        MLX.eval(scheduler.sigmas, scheduler.timesteps)

        let expectedTimesteps: [Float] = [1_000, 766.709_5, 455.613_8, 20]
        let actualTimesteps = scheduler.timesteps.asArray(Float.self)
        XCTAssertEqual(actualTimesteps.count, expectedTimesteps.count)
        for (actual, expected) in zip(actualTimesteps, expectedTimesteps) {
            XCTAssertEqual(actual, expected, accuracy: 0.001)
        }
        let expectedModelTimesteps: [Float] = [1, 0.766_709_5, 0.455_613_8, 0.02]
        let actualModelTimesteps = (0..<4).map {
            scheduler.modelTimestep(at: $0).item(Float.self)
        }
        for (actual, expected) in zip(actualModelTimesteps, expectedModelTimesteps) {
            XCTAssertEqual(actual, expected, accuracy: 0.000_001)
        }
        XCTAssertEqual(scheduler.sigmas[3].item(Float.self), 0.02, accuracy: 0.000_001)
        XCTAssertEqual(scheduler.sigmas[4].item(Float.self), 0, accuracy: 0.000_001)
    }

    private static func qwen2511SchedulerConfig() throws -> QwenImageEditSchedulerConfig {
        let json = #"{"base_image_seq_len":256,"base_shift":0.5,"invert_sigmas":false,"max_image_seq_len":8192,"max_shift":0.9,"num_train_timesteps":1000,"shift":1.0,"shift_terminal":0.02,"time_shift_type":"exponential","use_dynamic_shifting":true}"#
        return try JSONDecoder().decode(QwenImageEditSchedulerConfig.self, from: Data(json.utf8))
    }
}
