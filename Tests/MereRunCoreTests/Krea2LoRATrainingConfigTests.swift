import XCTest
@testable import MereRunCore

final class Krea2LoRATrainingConfigTests: XCTestCase {
    func testUseCompileDefaultsToTrue() {
        XCTAssertTrue(Krea2LoRATrainingConfig().useCompile)
    }

    func testLRSchedulerDefaultsToFlat() {
        let config = Krea2LoRATrainingConfig()

        XCTAssertEqual(config.lrWarmupSteps, 0)
        XCTAssertFalse(config.useCosineScheduler)
        XCTAssertEqual(config.lrMinFactor, 0)
    }

    func testUseCompileCanBeDisabled() {
        let config = Krea2LoRATrainingConfig(useCompile: false)

        XCTAssertFalse(config.useCompile)
    }

    func testScheduledLearningRateIsFlatWhenCosineDisabled() {
        let lr = Krea2LoRATrainer.scheduledLearningRate(
            baseLearningRate: 0.001,
            step: 50,
            totalSteps: 100,
            warmupSteps: 10,
            useCosineScheduler: false,
            minFactor: 0
        )

        XCTAssertEqual(lr, 0.001, accuracy: 0.000001)
    }

    func testScheduledLearningRateWarmsUpThenCosineDecays() {
        let warmupStart = Krea2LoRATrainer.scheduledLearningRate(
            baseLearningRate: 0.001,
            step: 0,
            totalSteps: 100,
            warmupSteps: 10,
            useCosineScheduler: true,
            minFactor: 0
        )
        let warmupEnd = Krea2LoRATrainer.scheduledLearningRate(
            baseLearningRate: 0.001,
            step: 9,
            totalSteps: 100,
            warmupSteps: 10,
            useCosineScheduler: true,
            minFactor: 0
        )
        let final = Krea2LoRATrainer.scheduledLearningRate(
            baseLearningRate: 0.001,
            step: 99,
            totalSteps: 100,
            warmupSteps: 10,
            useCosineScheduler: true,
            minFactor: 0
        )

        XCTAssertEqual(warmupStart, 0.0001, accuracy: 0.000001)
        XCTAssertEqual(warmupEnd, 0.001, accuracy: 0.000001)
        XCTAssertLessThan(final, 0.000001)
    }
}
