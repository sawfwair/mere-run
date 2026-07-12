import Foundation
import MereRunCore
import XCTest

final class GeometryTypesTests: XCTestCase {
    func testIntrinsicsExposeNormalizedAndPixelMatrices() {
        let intrinsics = GeometryCameraIntrinsics(
            imageWidth: 640,
            imageHeight: 480,
            normalizedFX: 0.75,
            normalizedFY: 1.0
        )
        XCTAssertEqual(intrinsics.pixelFX, 480, accuracy: 1e-12)
        XCTAssertEqual(intrinsics.pixelFY, 480, accuracy: 1e-12)
        XCTAssertEqual(intrinsics.pixelCX, 320, accuracy: 1e-12)
        XCTAssertEqual(intrinsics.pixelCY, 240, accuracy: 1e-12)
        XCTAssertEqual(intrinsics.normalizedMatrixRowMajor, [0.75, 0, 0.5, 0, 1, 0.5, 0, 0, 1])
        XCTAssertEqual(intrinsics.pixelMatrixRowMajor, [480, 0, 320, 0, 480, 240, 0, 0, 1])
    }

    func testProjectionUsesPixelCentersAndNormalizedIntrinsics() throws {
        let intrinsics = GeometryCameraIntrinsics(
            imageWidth: 2,
            imageHeight: 2,
            normalizedFX: 0.5,
            normalizedFY: 0.5
        )
        let points = try GeometryProjection.pointMap(
            depth: [2, 2, 2, 2],
            validity: [1, 1, 1, 1],
            intrinsics: intrinsics
        )
        XCTAssertEqual(points, [
            -1, -1, 2,
             1, -1, 2,
            -1,  1, 2,
             1,  1, 2,
        ])
    }

    func testDenseFrameRejectsMismatchedFields() {
        let intrinsics = GeometryCameraIntrinsics(
            imageWidth: 2,
            imageHeight: 2,
            normalizedFX: 1,
            normalizedFY: 1
        )
        XCTAssertThrowsError(try DenseGeometryFrame(
            width: 2,
            height: 2,
            units: .meters,
            intrinsics: intrinsics,
            depth: [1, 2],
            points: [Float](repeating: 0, count: 12),
            validity: [1, 1, 1, 1]
        )) { error in
            XCTAssertEqual(
                error as? GeometryError,
                .invalidElementCount(field: "depth", expected: 4, actual: 2)
            )
        }
    }

    func testStatisticsIgnoreInvalidAndNonFiniteSamples() throws {
        let frame = try makeFrame(
            depth: [1, 2, .infinity, 4],
            validity: [1, 1, 1, 0]
        )
        let stats = try GeometryProjection.depthStatistics(for: frame)
        XCTAssertEqual(stats.validPixelCount, 2)
        XCTAssertEqual(stats.invalidPixelCount, 2)
        XCTAssertEqual(stats.minimum, 1)
        XCTAssertEqual(stats.maximum, 2)
        XCTAssertEqual(stats.mean, 1.5)
    }

    private func makeFrame(depth: [Float], validity: [UInt8]) throws -> DenseGeometryFrame {
        let intrinsics = GeometryCameraIntrinsics(
            imageWidth: 2,
            imageHeight: 2,
            normalizedFX: 1,
            normalizedFY: 1
        )
        let safeDepth = depth.map { $0.isFinite && $0 > 0 ? $0 : 1 }
        let points = try GeometryProjection.pointMap(depth: safeDepth, validity: validity, intrinsics: intrinsics)
        return try DenseGeometryFrame(
            width: 2,
            height: 2,
            units: .meters,
            intrinsics: intrinsics,
            depth: depth,
            points: points,
            validity: validity
        )
    }
}
