import Foundation
import MereRunCore
import XCTest

final class ModelArtifactPinTests: XCTestCase {
    func testVerifiesSizeAndSHA256() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let data = Data("mere.run geometry\n".utf8)
        try data.write(to: root.appendingPathComponent("model.bin"))
        let pin = ModelArtifactPin(
            filename: "model.bin",
            byteCount: Int64(data.count),
            sha256: "6d59bc8d0d6c8277af03929fff4e119085c2597ebedcdd38d7eb920e17606ad2"
        )
        XCTAssertNoThrow(try pin.verify(in: root))
    }

    func testRejectsWrongSizeBeforeHashing() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data([1, 2, 3]).write(to: root.appendingPathComponent("model.bin"))
        let pin = ModelArtifactPin(filename: "model.bin", byteCount: 4, sha256: String(repeating: "0", count: 64))
        XCTAssertThrowsError(try pin.verify(in: root)) { error in
            XCTAssertEqual(
                error as? ModelArtifactVerificationError,
                .sizeMismatch(path: root.appendingPathComponent("model.bin").path, expected: 4, actual: 3)
            )
        }
    }

    func testVerifiesManagedSymlinkTargetSizeAndHash() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let cache = root.appendingPathComponent("cache", isDirectory: true)
        let install = root.appendingPathComponent("install", isDirectory: true)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
        let data = Data("mere.run geometry\n".utf8)
        let target = cache.appendingPathComponent("checkpoint.bin")
        try data.write(to: target)
        try FileManager.default.createSymbolicLink(
            at: install.appendingPathComponent("model.bin"),
            withDestinationURL: target
        )
        let pin = ModelArtifactPin(
            filename: "model.bin",
            byteCount: Int64(data.count),
            sha256: "6d59bc8d0d6c8277af03929fff4e119085c2597ebedcdd38d7eb920e17606ad2"
        )

        XCTAssertEqual(try pin.verify(in: install), install.appendingPathComponent("model.bin"))
    }

    func testGeometryPinsCoverEveryRoadmapModelID() {
        XCTAssertEqual(
            Set(GeometryModelPins.all.map(\.modelID)),
            [
                ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue,
                ModelResolver.ModelID.visionDepthVDASmall.rawValue,
                ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue,
                ModelResolver.ModelID.visionGeometryDA3Small.rawValue,
                ModelResolver.ModelID.image3DTripoSR.rawValue,
                ModelResolver.ModelID.image3DInstantMeshBase.rawValue,
            ]
        )
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-artifact-pin-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
