import MLX
import XCTest
@testable import MereRunCore

final class Gemma4UnifiedImageProcessorTests: MereRunCoreTestCase {
    func testExpandsBareImagePlaceholdersToUnifiedSoftTokens() throws {
        let expanded = try Gemma4UnifiedImageProcessor.expandedPromptTokens(
            [10, 258_880, 20, 258_880, 30],
            softTokenCounts: [2, 3],
            imageTokenId: 258_880,
            boiTokenId: 255_999,
            eoiTokenId: 258_882
        )

        XCTAssertEqual(expanded, [
            10,
            255_999,
            258_880,
            258_880,
            258_882,
            20,
            255_999,
            258_880,
            258_880,
            258_880,
            258_882,
            30,
        ])
    }

    func testAlreadyExpandedPromptIsLeftUnchanged() throws {
        let tokens = [255_999, 258_880, 258_880, 258_882, 42]
        let expanded = try Gemma4UnifiedImageProcessor.expandedPromptTokens(
            tokens,
            softTokenCounts: [2],
            imageTokenId: 258_880,
            boiTokenId: 255_999,
            eoiTokenId: 258_882
        )

        XCTAssertEqual(expanded, tokens)
    }

    func testMultimodalTokenTypesMarkImageTokensOnly() {
        let mmTokenTypeIds = Gemma4UnifiedImageProcessor.mmTokenTypeIds(
            tokens: [255_999, 258_880, 258_880, 258_882, 42],
            imageTokenId: 258_880
        )

        MLX.eval(mmTokenTypeIds)
        XCTAssertEqual(mmTokenTypeIds.shape, [1, 5])
        XCTAssertEqual(mmTokenTypeIds.asArray(Int32.self), [0, 1, 1, 0, 0])
    }
}
