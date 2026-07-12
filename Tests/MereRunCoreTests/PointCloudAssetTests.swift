import Foundation
import MereRunCore
import XCTest

final class PointCloudAssetTests: XCTestCase {
    func testWritesColoredConfidencePointCloudAsPLYAndGLB() throws {
        let cloud = try PointCloudAsset(
            positions: [0, 0, 1, 1, 2, 3],
            colorsRGBA8: [255, 0, 0, 255, 0, 255, 0, 255],
            confidence: [1.5, 2.5],
            viewIndices: [0, 1]
        )
        XCTAssertEqual(cloud.pointCount, 2)
        XCTAssertEqual(cloud.bounds.minimum, [0, 0, 1])
        XCTAssertEqual(cloud.bounds.maximum, [1, 2, 3])

        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "point-cloud-tests-\(UUID().uuidString)",
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        let ply = root.appendingPathComponent("scene.ply")
        let glb = root.appendingPathComponent("scene.glb")
        try PointCloudPLYWriter.write(cloud, to: ply)
        try PointCloudGLBWriter.write(cloud, to: glb)

        let plyData = try Data(contentsOf: ply)
        let header = String(decoding: plyData.prefix(512), as: UTF8.self)
        XCTAssertTrue(header.contains("element vertex 2"))
        XCTAssertTrue(header.contains("property float z\nproperty uchar red\n"))
        XCTAssertFalse(header.contains("zproperty"))
        XCTAssertTrue(header.contains("property float confidence"))
        XCTAssertTrue(header.contains("property uint view_index"))
        let glbData = try Data(contentsOf: glb)
        XCTAssertGreaterThan(glbData.count, 20)
        XCTAssertEqual(Array(glbData.prefix(4)), [0x67, 0x6C, 0x54, 0x46])
    }

    func testRejectsMismatchedPerPointFields() {
        XCTAssertThrowsError(try PointCloudAsset(
            positions: [0, 0, 0],
            confidence: [1, 2]
        )) { error in
            XCTAssertEqual(
                error as? PointCloudAssetError,
                .invalidElementCount(field: "confidence", expected: 1, actual: 2)
            )
        }
    }
}
