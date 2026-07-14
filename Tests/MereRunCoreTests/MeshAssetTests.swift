import Foundation
import MereRunCore
import XCTest

final class MeshAssetTests: XCTestCase {
    func testGeneratesNormalsAndBounds() throws {
        let mesh = try triangleMesh(normals: false)
        let generated = try mesh.withGeneratedNormals()
        XCTAssertEqual(generated.bounds.minimum, [0, 0, 0])
        XCTAssertEqual(generated.bounds.maximum, [1, 1, 0])
        XCTAssertEqual(generated.normals, [0, 0, 1, 0, 0, 1, 0, 0, 1])
    }

    func testRejectsOutOfBoundsIndex() {
        XCTAssertThrowsError(try MeshAsset(
            vertices: [0, 0, 0],
            indices: [0, 0, 1],
            inferredUnseenGeometry: true
        )) { error in
            XCTAssertEqual(error as? MeshAssetError, .indexOutOfBounds(index: 1, vertexCount: 1))
        }
    }

    func testExportsOBJPLYAndValidGLBHeader() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let mesh = try triangleMesh(normals: true)
        let inputURL = root.appendingPathComponent("object.png")
        let inputData = Data("durable mesh input".utf8)
        try inputData.write(to: inputURL)
        let result = try MeshArtifactExporter.export(
            mesh: mesh,
            inputURLs: [inputURL],
            outputDirectory: root,
            stem: "object asset",
            provenance: GeometryModelProvenance(
                modelID: "image-3d-test",
                upstreamRepository: "example/test",
                upstreamRevision: "pin",
                license: "MIT"
            ),
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(result.manifest.vertexCount, 3)
        XCTAssertEqual(result.manifest.triangleCount, 1)
        XCTAssertTrue(result.manifest.inferredUnseenGeometry)
        XCTAssertEqual(result.manifest.inputPaths, [inputURL.path])
        XCTAssertEqual(result.manifest.inputs, [MeshInputRecord(
            path: inputURL.path,
            byteCount: Int64(inputData.count),
            sha256: try ModelArtifactPin.fileSHA256(inputURL)
        )])
        XCTAssertEqual(Set(result.manifest.artifacts.map(\.kind)), [.obj, .ply, .glb])
        let ply = try Data(contentsOf: root.appendingPathComponent("object-asset.ply"))
        let headerEnd = try XCTUnwrap(ply.range(of: Data("end_header\n".utf8)))
        let plyHeader = String(decoding: ply[..<headerEnd.upperBound], as: UTF8.self)
        XCTAssertTrue(plyHeader.contains("property float nz\nproperty uchar red\n"))
        XCTAssertFalse(plyHeader.contains("nzproperty"))
        let glb = try Data(contentsOf: root.appendingPathComponent("object-asset.glb"))
        XCTAssertEqual(Array(glb.prefix(4)), [0x67, 0x6C, 0x54, 0x46])
        XCTAssertEqual(readUInt32(glb, offset: 4), 2)
        XCTAssertEqual(Int(readUInt32(glb, offset: 8)), glb.count)

        var legacyObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: result.manifestURL))
                as? [String: Any]
        )
        legacyObject.removeValue(forKey: "inputs")
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let legacy = try decoder.decode(
            MeshOutputManifest.self,
            from: JSONSerialization.data(withJSONObject: legacyObject)
        )
        XCTAssertEqual(legacy.inputPaths, [inputURL.path])
        XCTAssertNil(legacy.inputs)
    }

    func testGLBMaterialFactorsDefaultAndOverride() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let mesh = try triangleMesh(normals: true)

        let plain = root.appendingPathComponent("plain.glb")
        try MeshGLBWriter.write(mesh, to: plain)
        var material = try glbMaterial(at: plain)
        XCTAssertEqual(material["metallicFactor"] as? Double, 0)
        XCTAssertEqual(material["roughnessFactor"] as? Double, 1)

        let pbr = root.appendingPathComponent("pbr.glb")
        try MeshGLBWriter.write(
            mesh,
            to: pbr,
            material: MeshPBRMaterialFactors(metallicFactor: 0.25, roughnessFactor: 0.625)
        )
        material = try glbMaterial(at: pbr)
        XCTAssertEqual(try XCTUnwrap(material["metallicFactor"] as? Double), 0.25, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(material["roughnessFactor"] as? Double), 0.625, accuracy: 1e-6)
    }

    private func glbMaterial(at url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        let jsonLength = Int(readUInt32(data, offset: 12))
        let document = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data.subdata(in: 20..<(20 + jsonLength)))
                as? [String: Any]
        )
        let materials = try XCTUnwrap(document["materials"] as? [[String: Any]])
        return try XCTUnwrap(materials[0]["pbrMetallicRoughness"] as? [String: Any])
    }

    private func triangleMesh(normals: Bool) throws -> MeshAsset {
        try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            normals: normals ? [0, 0, 1, 0, 0, 1, 0, 0, 1] : nil,
            colorsRGBA8: [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255],
            textureCoordinates: [0, 0, 1, 0, 0, 1],
            inferredUnseenGeometry: true
        )
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-mesh-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func readUInt32(_ data: Data, offset: Int) -> UInt32 {
        data[offset..<(offset + 4)].enumerated().reduce(0) { result, item in
            result | (UInt32(item.element) << UInt32(item.offset * 8))
        }
    }
}
