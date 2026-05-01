import XCTest
@testable import MereRunCore

final class ZImageTurboInferenceConfigTests: MereRunCoreTestCase {

    func testInitTimeStepZeroWhenImg2ImgDisabled() {
        XCTAssertEqual(
            ZImageTurboInferenceConfig(width: 512, height: 512, numInferenceSteps: 4, imageStrength: nil).initTimeStep,
            0
        )
        XCTAssertEqual(
            ZImageTurboInferenceConfig(width: 512, height: 512, numInferenceSteps: 4, imageStrength: 0.0).initTimeStep,
            0
        )
        XCTAssertEqual(
            ZImageTurboInferenceConfig(width: 512, height: 512, numInferenceSteps: 4, imageStrength: -0.2).initTimeStep,
            0
        )
    }

    func testInitTimeStepMatchesMfluxStrengthSemantics() {
        XCTAssertEqual(
            ZImageTurboInferenceConfig(width: 512, height: 512, numInferenceSteps: 4, imageStrength: 0.45).initTimeStep,
            1
        )
        XCTAssertEqual(
            ZImageTurboInferenceConfig(width: 512, height: 512, numInferenceSteps: 4, imageStrength: 0.75).initTimeStep,
            3
        )
        XCTAssertEqual(
            ZImageTurboInferenceConfig(width: 512, height: 512, numInferenceSteps: 4, imageStrength: 1.0).initTimeStep,
            4
        )
        XCTAssertEqual(
            ZImageTurboInferenceConfig(width: 512, height: 512, numInferenceSteps: 4, imageStrength: 2.0).initTimeStep,
            4
        )
    }

    func testTimeStepRangeUsesInitTimeStep() {
        let config = ZImageTurboInferenceConfig(width: 512, height: 512, numInferenceSteps: 4, imageStrength: 0.75)
        XCTAssertEqual(config.timeSteps, 3..<4)

        let preserveConfig = ZImageTurboInferenceConfig(width: 512, height: 512, numInferenceSteps: 4, imageStrength: 1.0)
        XCTAssertTrue(preserveConfig.timeSteps.isEmpty)
    }
}
