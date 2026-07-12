import Foundation
import MereRunCore
import XCTest

final class GeometryExportersTests: XCTestCase {
    func testOpenEXRHasMagicHeaderAndScanlineOffsets() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("depth.exr")
        try OpenEXRWriter.writeFloatChannels(
            [(name: "Z", values: [1, 2, 3, 4])],
            width: 2,
            height: 2,
            to: url
        )
        let data = try Data(contentsOf: url)
        XCTAssertGreaterThan(data.count, 100)
        XCTAssertEqual(readUInt32(data, at: 0), 20_000_630)
        XCTAssertEqual(readUInt32(data, at: 4), 2)
        XCTAssertNotNil(String(data: data, encoding: .isoLatin1)?.range(of: "channels"))
        XCTAssertNotNil(String(data: data, encoding: .isoLatin1)?.range(of: "dataWindow"))
    }

    func testPLYWritesOnlyValidPointsAndProperties() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let frame = try makeFrame()
        let url = directory.appendingPathComponent("points.ply")
        try GeometryPLYWriter.write(
            frame: frame,
            rgba8: [
                255, 0, 0, 255,
                0, 255, 0, 255,
                0, 0, 255, 255,
                255, 255, 255, 255,
            ],
            to: url
        )
        let data = try Data(contentsOf: url)
        let text = String(decoding: data.prefix(800), as: UTF8.self)
        XCTAssertTrue(text.contains("format binary_little_endian 1.0"))
        XCTAssertTrue(text.contains("element vertex 3"))
        XCTAssertTrue(text.contains("property float nx"))
        XCTAssertTrue(text.contains("property uchar red"))
        XCTAssertTrue(text.contains("property float confidence"))
    }

    func testArtifactExporterWritesManifestAndChecksums() throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let frame = try makeFrame()
        let provenance = GeometryModelProvenance(
            modelID: "test-geometry",
            upstreamRepository: "example/test",
            upstreamRevision: "abc123",
            license: "MIT",
            weightsSHA256: String(repeating: "a", count: 64)
        )
        let result = try GeometryArtifactExporter.export(
            frame: frame,
            inputURL: URL(fileURLWithPath: "/tmp/input.png"),
            outputDirectory: directory,
            provenance: provenance,
            createdAt: Date(timeIntervalSince1970: 0),
            options: GeometryExportOptions(stem: "shot-001")
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.manifestURL.path))
        XCTAssertEqual(result.manifest.depthStatistics.validPixelCount, 3)
        XCTAssertEqual(result.manifest.artifacts.count, 8)
        XCTAssertEqual(Set(result.manifest.artifacts.map(\.sha256.count)), [64])
        for artifact in result.manifest.artifacts {
            XCTAssertFalse(artifact.relativePath.hasPrefix("/"))
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(artifact.relativePath).path
            ))
        }

        let decoded = try JSONDecoder.iso8601.decode(
            GeometryOutputManifest.self,
            from: Data(contentsOf: result.manifestURL)
        )
        XCTAssertEqual(decoded, result.manifest)
    }

    private func makeFrame() throws -> DenseGeometryFrame {
        let intrinsics = GeometryCameraIntrinsics(
            imageWidth: 2,
            imageHeight: 2,
            normalizedFX: 0.5,
            normalizedFY: 0.5
        )
        let depth: [Float] = [1, 2, 3, 4]
        let validity: [UInt8] = [1, 1, 0, 1]
        return try DenseGeometryFrame(
            width: 2,
            height: 2,
            units: .meters,
            intrinsics: intrinsics,
            depth: depth,
            points: GeometryProjection.pointMap(depth: depth, validity: validity, intrinsics: intrinsics),
            normals: [
                0, 0, 1,
                0, 0, 1,
                0, 0, 0,
                0, 0, 1,
            ],
            validity: validity,
            confidence: [0.9, 0.8, 0, 0.7]
        )
    }

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-geometry-tests-\(UUID().uuidString)", isDirectory: true)
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
        data[offset..<(offset + 4)].enumerated().reduce(UInt32(0)) { result, item in
            result | (UInt32(item.element) << UInt32(item.offset * 8))
        }
    }
}

private extension JSONDecoder {
    static var iso8601: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
