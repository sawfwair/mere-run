import MLX
import XCTest
@testable import MereRunCore
@testable import MereRunLTXModel

final class LTXSamplerSupportTests: XCTestCase {
    func testLTX25AncestralStageOneIsLimitedToTheDistilledPipeline() {
        XCTAssertTrue(
            ltx25UsesDistilledAncestralStage1(
                isLTX25: true,
                isFullTwoStage: false,
                usesDFR: false,
                usesHDRICLoRA: false,
                usesRetake: false,
                usesDubIt: false,
                hasReferenceVideos: false
            )
        )

        let specializedRecipes = [
            (true, false, false, false, false, false),
            (false, true, false, false, false, false),
            (false, false, true, false, false, false),
            (false, false, false, true, false, false),
            (false, false, false, false, true, false),
            (false, false, false, false, false, true),
        ]
        for recipe in specializedRecipes {
            XCTAssertFalse(
                ltx25UsesDistilledAncestralStage1(
                    isLTX25: true,
                    isFullTwoStage: recipe.0,
                    usesDFR: recipe.1,
                    usesHDRICLoRA: recipe.2,
                    usesRetake: recipe.3,
                    usesDubIt: recipe.4,
                    hasReferenceVideos: recipe.5
                )
            )
        }
        XCTAssertFalse(
            ltx25UsesDistilledAncestralStage1(
                isLTX25: false,
                isFullTwoStage: false,
                usesDFR: false,
                usesHDRICLoRA: false,
                usesRetake: false,
                usesDubIt: false,
                hasReferenceVideos: false
            )
        )
    }
}
