import MLX
import XCTest
@testable import MereRunCore

final class QwenVLDevicePipelineTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        MLXTestSupport.ensureMetalLibraryAvailable()
    }

    func testContiguousImageTokenRangeUsesHostTokenizationResult() {
        XCTAssertEqual(
            QwenVLCaptioner.contiguousTokenRange(in: [1, 2, 99, 99, 99, 3], matching: 99),
            2..<5
        )
        XCTAssertNil(QwenVLCaptioner.contiguousTokenRange(in: [1, 99, 2, 99], matching: 99))
        XCTAssertNil(QwenVLCaptioner.contiguousTokenRange(in: [1, 2, 3], matching: 99))
    }

    func testDeepstackFeaturesAreAddedOnlyInsideVisualSpan() {
        let hidden = MLXArray(Array(0..<12).map(Float.init)).reshaped(1, 4, 3)
        let visual = MLXArray([
            Float(100), 101, 102,
            200, 201, 202
        ]).reshaped(2, 3)

        let output = QwenEncoder.applyingDeepstackFeatures(
            hiddenStates: hidden,
            visualTokenRange: 1..<3,
            visualEmbeds: visual
        )
        MLX.eval(output)

        XCTAssertEqual(
            output.asArray(Float.self),
            [0, 1, 2, 103, 105, 107, 206, 208, 210, 9, 10, 11]
        )
    }

    func testInvalidDeepstackSpanLeavesHiddenStatesUnchanged() {
        let hidden = MLXArray(Array(0..<12).map(Float.init)).reshaped(1, 4, 3)
        let visual = MLXArray.zeros([2, 3])

        let output = QwenEncoder.applyingDeepstackFeatures(
            hiddenStates: hidden,
            visualTokenRange: 3..<5,
            visualEmbeds: visual
        )
        MLX.eval(output)

        XCTAssertEqual(output.asArray(Float.self), hidden.asArray(Float.self))
    }
}
