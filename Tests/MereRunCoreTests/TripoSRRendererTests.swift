import MLX
@testable import MereRunCore
import XCTest

final class TripoSRRendererTests: MereRunCoreTestCase {
    func testBilinearGridSampleMatchesAlignCornersFalseCoordinates() {
        let plane = MLXArray([Float(0), 1, 2, 3]).reshaped(2, 2, 1)
        let sampled = TripoSRRenderer.bilinearGridSample(
            plane: plane,
            x: MLXArray([Float(-0.5), 0, 0.5]),
            y: MLXArray([Float(-0.5), 0, 0.5])
        )
        MLX.eval(sampled)
        let values = sampled.reshaped(-1).asArray(Float.self)
        XCTAssertEqual(values[0], 0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 1.5, accuracy: 1e-6)
        XCTAssertEqual(values[2], 3, accuracy: 1e-6)
    }

    func testBilinearGridSampleUsesZeroPadding() {
        let plane = MLX.ones([2, 2, 1], dtype: .float32)
        let sampled = TripoSRRenderer.bilinearGridSample(
            plane: plane,
            x: MLXArray([Float(-1), 1, -1.5]),
            y: MLXArray([Float(-1), 1, -1.5])
        )
        MLX.eval(sampled)
        let values = sampled.reshaped(-1).asArray(Float.self)
        XCTAssertEqual(values[0], 0.25, accuracy: 1e-6)
        XCTAssertEqual(values[1], 0.25, accuracy: 1e-6)
        XCTAssertEqual(values[2], 0, accuracy: 1e-6)
    }
}
