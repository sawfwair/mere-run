import Foundation
import MereRunCore
import XCTest

/// Vertex-color round-trip parity across the OBJ, PLY, and GLB writers.
///
/// glTF 2.0 defines `COLOR_0` as linear, so a standards-compliant viewer
/// (three.js, Blender) applies the linear-to-sRGB transfer before display.
/// These tests emulate that viewer convention with an independent transfer
/// implementation and require the displayed bytes to equal the authored
/// sRGB bytes exactly, while OBJ and PLY keep the authored bytes verbatim.
final class VertexColorParityTests: XCTestCase {
    /// Authored sRGB vertex colors covering the empirical mustard-yellow
    /// regression, both sides of the piecewise sRGB knee (10 and 11), the
    /// darkest nonzero step, both extremes, and non-opaque alpha.
    private let authoredSRGBA8: [UInt8] = [
        196, 158, 44, 255,
        1, 10, 11, 128,
        0, 128, 255, 0,
    ]

    func testGLBColorsDisplayAsAuthoredSRGBUnderViewerConvention() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("mesh.glb")
        try MeshGLBWriter.write(try triangleMesh(), to: url)
        let color = try GLBColorProbe.readColor0(try Data(contentsOf: url))

        XCTAssertEqual(color.componentType, 5_123, "COLOR_0 should be 16-bit to avoid banding after linearization")
        XCTAssertTrue(color.normalized)
        XCTAssertEqual(color.type, "VEC4")
        XCTAssertEqual(color.values.count, authoredSRGBA8.count)

        for vertex in 0..<(authoredSRGBA8.count / 4) {
            for channel in 0..<3 {
                let index = vertex * 4 + channel
                let linear = Float(color.values[index]) / 65535
                XCTAssertEqual(
                    displayedSRGBByte(fromLinear: linear),
                    authoredSRGBA8[index],
                    "vertex \(vertex) channel \(channel) should display as authored"
                )
            }
            let alphaIndex = vertex * 4 + 3
            XCTAssertEqual(
                UInt8((Float(color.values[alphaIndex]) / 65535 * 255).rounded()),
                authoredSRGBA8[alphaIndex],
                "alpha is linear coverage and must rescale without a transfer"
            )
        }
    }

    func testOBJAndPLYKeepAuthoredDisplayReferredBytes() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let mesh = try triangleMesh()

        let objURL = root.appendingPathComponent("mesh.obj")
        try MeshOBJWriter.write(mesh, to: objURL)
        let vertexLines = String(decoding: try Data(contentsOf: objURL), as: UTF8.self)
            .split(separator: "\n")
            .filter { $0.hasPrefix("v ") }
        XCTAssertEqual(vertexLines.count, 3)
        for (vertex, line) in vertexLines.enumerated() {
            let fields = line.split(separator: " ").dropFirst(4).compactMap { Float($0) }
            XCTAssertEqual(fields.count, 3, "vertex line should carry r g b")
            for channel in 0..<3 {
                XCTAssertEqual(
                    fields[channel],
                    Float(authoredSRGBA8[vertex * 4 + channel]) / 255,
                    accuracy: 1e-6
                )
            }
        }

        let plyURL = root.appendingPathComponent("mesh.ply")
        try MeshPLYWriter.write(mesh, to: plyURL)
        let ply = try Data(contentsOf: plyURL)
        let headerEnd = try XCTUnwrap(ply.range(of: Data("end_header\n".utf8))).upperBound
        let vertexStride = 6 * 4 + 4
        for vertex in 0..<3 {
            let colorOffset = headerEnd + vertex * vertexStride + 6 * 4
            XCTAssertEqual(
                Array(ply[colorOffset..<(colorOffset + 4)]),
                Array(authoredSRGBA8[(vertex * 4)..<(vertex * 4 + 4)])
            )
        }
    }

    func testPointCloudGLBEncodesColorsIdenticallyToMeshGLB() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let meshURL = root.appendingPathComponent("mesh.glb")
        let cloudURL = root.appendingPathComponent("cloud.glb")
        try MeshGLBWriter.write(try triangleMesh(), to: meshURL)
        try PointCloudGLBWriter.write(
            try PointCloudAsset(
                positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
                colorsRGBA8: authoredSRGBA8
            ),
            to: cloudURL
        )
        let meshColor = try GLBColorProbe.readColor0(try Data(contentsOf: meshURL))
        let cloudColor = try GLBColorProbe.readColor0(try Data(contentsOf: cloudURL))
        XCTAssertEqual(cloudColor.componentType, meshColor.componentType)
        XCTAssertEqual(cloudColor.normalized, meshColor.normalized)
        XCTAssertEqual(cloudColor.type, meshColor.type)
        XCTAssertEqual(cloudColor.values, meshColor.values)
    }

    private func triangleMesh() throws -> MeshAsset {
        try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            colorsRGBA8: authoredSRGBA8,
            inferredUnseenGeometry: true
        )
    }

    /// Independent linear-to-sRGB OETF (IEC 61966-2-1) emulating what a
    /// compliant viewer applies to COLOR_0 before display. Deliberately not
    /// implemented via `VertexColorTransfer` so an encoder bug cannot cancel
    /// itself out in the round trip.
    private func displayedSRGBByte(fromLinear linear: Float) -> UInt8 {
        let encoded = linear <= 0.0031308
            ? linear * 12.92
            : 1.055 * pow(linear, 1 / 2.4) - 0.055
        return UInt8((min(1, max(0, encoded)) * 255).rounded())
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vertex-color-parity-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

/// Minimal GLB reader for the COLOR_0 accessor of the first primitive.
enum GLBColorProbe {
    struct Color0 {
        let componentType: Int
        let normalized: Bool
        let type: String
        let values: [UInt16]
    }

    static func readColor0(_ data: Data) throws -> Color0 {
        let jsonLength = Int(readUInt32(data, offset: 12))
        let document = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data.subdata(in: 20..<(20 + jsonLength)))
                as? [String: Any]
        )
        let meshes = try XCTUnwrap(document["meshes"] as? [[String: Any]])
        let primitive = try XCTUnwrap((meshes[0]["primitives"] as? [[String: Any]])?.first)
        let attributes = try XCTUnwrap(primitive["attributes"] as? [String: Any])
        let accessorIndex = try XCTUnwrap(attributes["COLOR_0"] as? Int)
        let accessor = try XCTUnwrap(document["accessors"] as? [[String: Any]])[accessorIndex]
        let viewIndex = try XCTUnwrap(accessor["bufferView"] as? Int)
        let view = try XCTUnwrap(document["bufferViews"] as? [[String: Any]])[viewIndex]
        let count = try XCTUnwrap(accessor["count"] as? Int)

        let binaryStart = 20 + jsonLength + 8
        let byteOffset = binaryStart + (view["byteOffset"] as? Int ?? 0)
        let values = (0..<(count * 4)).map { element in
            UInt16(data[byteOffset + element * 2]) | (UInt16(data[byteOffset + element * 2 + 1]) << 8)
        }
        return Color0(
            componentType: try XCTUnwrap(accessor["componentType"] as? Int),
            normalized: accessor["normalized"] as? Bool ?? false,
            type: try XCTUnwrap(accessor["type"] as? String),
            values: values
        )
    }

    private static func readUInt32(_ data: Data, offset: Int) -> UInt32 {
        data[offset..<(offset + 4)].enumerated().reduce(0) { result, item in
            result | (UInt32(item.element) << UInt32(item.offset * 8))
        }
    }
}
