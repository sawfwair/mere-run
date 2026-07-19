import MediaIO
import MLX
import XCTest
@testable import MereRunCore

final class SCAIL2CLIPTests: MereRunCoreTestCase {
    func testPreprocessorUsesPublishedOpenCLIPNormalization() throws {
        let image = try MediaImage(
            width: 1,
            height: 1,
            rgba8: [255, 0, 127, 255]
        )
        let configuration = SCAIL2CLIPVisionConfiguration(
            imageSize: 2,
            patchSize: 1,
            hiddenSize: 8,
            feedForwardSize: 16,
            headCount: 2,
            layerCount: 1
        )
        let normalized = SCAIL2CLIPPreprocessor.normalizedNHWC(
            image,
            configuration: configuration
        )
        eval(normalized)
        let values = normalized.asArray(Float.self)

        XCTAssertEqual(normalized.shape, [1, 2, 2, 3])
        XCTAssertEqual(values[0], (1 - SCAIL2CLIPPreprocessor.mean[0]) / SCAIL2CLIPPreprocessor.standardDeviation[0], accuracy: 1e-6)
        XCTAssertEqual(values[1], -SCAIL2CLIPPreprocessor.mean[1] / SCAIL2CLIPPreprocessor.standardDeviation[1], accuracy: 1e-6)
        XCTAssertEqual(values[2], (Float(127) / 255 - SCAIL2CLIPPreprocessor.mean[2]) / SCAIL2CLIPPreprocessor.standardDeviation[2], accuracy: 1e-6)
    }

    func testVisionTowerReturnsTokensAfterFirstThirtyOneBlockContract() {
        let configuration = SCAIL2CLIPVisionConfiguration(
            imageSize: 4,
            patchSize: 2,
            hiddenSize: 8,
            feedForwardSize: 16,
            headCount: 2,
            layerCount: 2
        )
        let model = SCAIL2CLIPVisionModel(configuration: configuration)
        let output = model(MLX.zeros([1, 4, 4, 3]))
        eval(output)

        XCTAssertEqual(output.shape, [1, 5, 8])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
    }
}
