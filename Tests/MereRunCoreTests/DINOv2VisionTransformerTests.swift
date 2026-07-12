import MLX
import MereRunCore
import XCTest

final class DINOv2VisionTransformerTests: MereRunCoreTestCase {
    func testReturnsRequestedNormalizedIntermediateGrids() throws {
        let configuration = DINOv2Configuration(
            hiddenSize: 12,
            layerCount: 3,
            headCount: 3,
            intermediateSize: 24,
            patchSize: 2,
            positionGridSize: 2
        )
        let model = DINOv2VisionTransformer(configuration: configuration)
        let input = MLX.zeros([1, 6, 4, 3], dtype: .float32)
        let output = model(input, intermediateLayers: [0, 2])
        MLX.eval(Array(output.featuresByLayer.values) + [output.classToken])
        XCTAssertEqual(output.gridHeight, 3)
        XCTAssertEqual(output.gridWidth, 2)
        XCTAssertEqual(output.featuresByLayer[0]?.shape, [1, 3, 2, 12])
        XCTAssertEqual(output.featuresByLayer[2]?.shape, [1, 3, 2, 12])
        XCTAssertNil(output.featuresByLayer[1])
        XCTAssertEqual(output.classToken.shape, [1, 12])
    }
}
