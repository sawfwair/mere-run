import Foundation
@testable import MereRunCore
import XCTest

final class InstantMeshIsosurfaceTests: MereRunCoreTestCase {
    func testPublicConfigurationRetainsInvalidRuntimeValueWithoutTrapping() {
        let configuration = InstantMeshMeshExtractionConfiguration(gridResolution: Int.max)
        XCTAssertEqual(configuration.gridResolution, Int.max)
    }

    func testNativePermissivePolygonizerUsesDeformedGridPositions() throws {
        let pointsPerAxis = 14
        let radius: Float = 0.58
        var sdf: [Float] = []
        var positions: [Float] = []
        for x in 0..<pointsPerAxis {
            for y in 0..<pointsPerAxis {
                for z in 0..<pointsPerAxis {
                    let px = Float(x) / Float(pointsPerAxis - 1) * 2 - 1
                    let py = Float(y) / Float(pointsPerAxis - 1) * 2 - 1
                    let pz = Float(z) / Float(pointsPerAxis - 1) * 2 - 1
                    sdf.append(radius - sqrt(px * px + py * py + pz * pz))
                    positions.append(px + 0.1)
                    positions.append(py)
                    positions.append(pz)
                }
            }
        }
        let mesh = try InstantMeshIsosurfaceExtractor.polygonize(
            signedDistance: sdf,
            positions: positions,
            pointsPerAxis: pointsPerAxis
        )
        XCTAssertGreaterThan(mesh.vertexCount, 100)
        XCTAssertGreaterThan(mesh.triangleCount, 100)
        XCTAssertEqual(mesh.bounds.minimum[0], -radius + 0.1, accuracy: 0.08)
        XCTAssertEqual(mesh.bounds.maximum[0], radius + 0.1, accuracy: 0.08)
        XCTAssertTrue(mesh.inferredUnseenGeometry)
        XCTAssertEqual(mesh.units, .normalizedObjectSpace)
    }

    func testUniformFieldProducesExplicitEmptySurfaceError() {
        let points = 4
        var positions: [Float] = []
        for x in 0..<points {
            for y in 0..<points {
                for z in 0..<points {
                    positions.append(contentsOf: [Float(x), Float(y), Float(z)])
                }
            }
        }
        XCTAssertThrowsError(try InstantMeshIsosurfaceExtractor.polygonize(
            signedDistance: [Float](repeating: 1, count: points * points * points),
            positions: positions,
            pointsPerAxis: points
        )) {
            XCTAssertEqual(
                $0 as? InstantMeshIsosurfaceError,
                .emptySurface(gridResolution: points - 1)
            )
        }
    }

    func testPinnedEmptyFieldRepairMatchesUpstreamSentinels() {
        let points = 7
        var sdf = [Float](repeating: 0.2, count: points * points * points)
        XCTAssertTrue(InstantMeshIsosurfaceExtractor.repairEmptyInteriorField(
            &sdf,
            pointsPerAxis: points
        ))
        let index = { (x: Int, y: Int, z: Int) in
            x * points * points + y * points + z
        }
        XCTAssertEqual(sdf[index(4, 4, 4)], 0.8, accuracy: 1e-6)
        XCTAssertEqual(sdf[index(0, 3, 3)], -1.2, accuracy: 1e-6)
        XCTAssertEqual(sdf[index(3, 3, 3)], 0.2, accuracy: 1e-6)
        XCTAssertFalse(InstantMeshIsosurfaceExtractor.repairEmptyInteriorField(
            &sdf,
            pointsPerAxis: points
        ))
    }

    func testPinnedEmptyFieldRepairPreservesAdditiveOverlapOnTinyGrid() {
        let points = 3
        var sdf = [Float](repeating: 0.2, count: points * points * points)
        XCTAssertTrue(InstantMeshIsosurfaceExtractor.repairEmptyInteriorField(
            &sdf,
            pointsPerAxis: points
        ))
        // grid_res / 2 + 1 is (2,2,2), which also belongs to the two-sample
        // boundary shell. Upstream adds both updates:
        // (1 - 0.2) + (-1 - 0.2).
        XCTAssertEqual(sdf[26], -0.4, accuracy: 1e-6)
    }
}
