import XCTest
@testable import MereRunCore

final class Q35RuntimeTuningTests: XCTestCase {
    func testSchedulingDefaultsApplyOnlyToQualifiedManagedQ4Models() {
        let qualified = [Q35Resources.q38TwentySevenB4BitModelId, Q35Resources.ornith35BMLX4BitModelId]
        let other = ["vision-chat-q38-27b", "text-agent-ornith-35b-mlx", "vision-chat-q38-flash-next-3bit", "custom-qwen"]
        for feature in Q35RuntimeTuning.Feature.allCases {
            for model in qualified {
                XCTAssertTrue(Q35RuntimeTuning.isEnabled(feature, modelID: model, environment: [:]))
            }
            for model in other {
                XCTAssertFalse(Q35RuntimeTuning.isEnabled(feature, modelID: model, environment: [:]))
            }
        }
    }

    func testExplicitOverridesPreserveDiagnosticControlsForEveryFeature() {
        for feature in Q35RuntimeTuning.Feature.allCases {
            XCTAssertFalse(Q35RuntimeTuning.isEnabled(
                feature, modelID: Q35Resources.ornith35BMLX4BitModelId, environment: [feature.rawValue: "0"]
            ))
            XCTAssertTrue(Q35RuntimeTuning.isEnabled(
                feature, modelID: "custom-qwen", environment: [feature.rawValue: "1"]
            ))
        }
    }
}
