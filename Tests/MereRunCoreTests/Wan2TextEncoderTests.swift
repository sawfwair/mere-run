import MLX
import XCTest
@testable import MereRunCore

final class Wan2TextEncoderTests: MereRunCoreTestCase {
    func testTinyT5EncoderPreservesTokenGeometry() {
        let configuration = Wan2TextEncoderConfiguration(
            vocabularySize: 32,
            hiddenSize: 16,
            attentionSize: 16,
            feedForwardSize: 32,
            headCount: 2,
            layerCount: 2,
            relativePositionBuckets: 8
        )
        let model = Wan2TextEncoderModel(configuration: configuration)
        let ids = MLXArray([1, 2, 3, 0], [1, 4])
        let mask = MLXArray([1, 1, 1, 0], [1, 4])
        let output = model(tokenIDs: ids, mask: mask)
        eval(output)

        XCTAssertEqual(output.shape, [1, 4, 16])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
    }
}
