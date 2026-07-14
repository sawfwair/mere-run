import Foundation
import MediaIO

/// Dependency-free glTF 2.0 binary writer for the baked-atlas TRELLIS.2 mesh:
/// split vertices with UVs, an sRGB baseColorTexture (with field alpha), and
/// a linear metallicRoughnessTexture, both embedded as PNG.
enum Trellis2TexturedGLBWriter {
    static func write(
        _ baked: Trellis2BakedTexturedMesh,
        coordinateSystem: MeshCoordinateSystem,
        units: MeshUnits,
        inferredUnseenGeometry: Bool,
        to url: URL
    ) throws {
        let basePNG = try MediaImageIO.pngData(from: MediaImage(
            width: baked.atlasWidth,
            height: baked.atlasHeight,
            rgba8: baked.baseColorRGBA8
        ))
        let metallicRoughnessPNG = try MediaImageIO.pngData(from: MediaImage(
            width: baked.atlasWidth,
            height: baked.atlasHeight,
            rgba8: baked.metallicRoughnessRGBA8
        ))

        var binary = Data()
        var bufferViews: [[String: Any]] = []

        func appendView(_ data: Data, target: Int?) -> Int {
            while !binary.count.isMultiple(of: 4) { binary.append(0) }
            let offset = binary.count
            binary.append(data)
            let index = bufferViews.count
            var view: [String: Any] = [
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": data.count,
            ]
            if let target { view["target"] = target }
            bufferViews.append(view)
            return index
        }

        var minimum = [Float](repeating: .infinity, count: 3)
        var maximum = [Float](repeating: -.infinity, count: 3)
        for vertex in 0..<baked.vertexCount {
            for axis in 0..<3 {
                let value = baked.positions[vertex * 3 + axis]
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
            }
        }

        let positionView = appendView(floatData(baked.positions), target: 34_962)
        let normalView = appendView(floatData(baked.normals), target: 34_962)
        let uvView = appendView(floatData(baked.uvs), target: 34_962)
        let indexView = appendView(uint32Data(baked.indices), target: 34_963)
        let baseImageView = appendView(basePNG, target: nil)
        let metallicRoughnessImageView = appendView(metallicRoughnessPNG, target: nil)

        let accessors: [[String: Any]] = [
            [
                "bufferView": positionView,
                "componentType": 5_126,
                "count": baked.vertexCount,
                "type": "VEC3",
                "min": minimum,
                "max": maximum,
            ],
            [
                "bufferView": normalView,
                "componentType": 5_126,
                "count": baked.vertexCount,
                "type": "VEC3",
            ],
            [
                "bufferView": uvView,
                "componentType": 5_126,
                "count": baked.vertexCount,
                "type": "VEC2",
            ],
            [
                "bufferView": indexView,
                "componentType": 5_125,
                "count": baked.indices.count,
                "type": "SCALAR",
            ],
        ]

        let json: [String: Any] = [
            "asset": ["version": "2.0", "generator": "mere.run native TRELLIS.2 baked atlas"],
            "scene": 0,
            "scenes": [["nodes": [0]]],
            "nodes": [["mesh": 0, "name": "Reconstructed Asset"]],
            "meshes": [[
                "name": "Baked PBR Mesh",
                "primitives": [[
                    "attributes": [
                        "POSITION": 0,
                        "NORMAL": 1,
                        "TEXCOORD_0": 2,
                    ],
                    "indices": 3,
                    "material": 0,
                    "mode": 4,
                ]],
            ]],
            "materials": [[
                "name": "Baked PBR",
                "doubleSided": true,
                "pbrMetallicRoughness": [
                    "baseColorTexture": ["index": 0],
                    "metallicRoughnessTexture": ["index": 1],
                ],
            ]],
            "textures": [
                ["sampler": 0, "source": 0],
                ["sampler": 0, "source": 1],
            ],
            // Mipmapped minification: consecutive cells are spatially
            // coherent (decode order follows the voxel scan), so mip blends
            // average nearby surface colors instead of aliasing across
            // unrelated blocks.
            "samplers": [[
                "magFilter": 9_729,
                "minFilter": 9_987,
                "wrapS": 33_071,
                "wrapT": 33_071,
            ]],
            "images": [
                ["bufferView": baseImageView, "mimeType": "image/png"],
                ["bufferView": metallicRoughnessImageView, "mimeType": "image/png"],
            ],
            "buffers": [["byteLength": binary.count]],
            "bufferViews": bufferViews,
            "accessors": accessors,
            "extras": [
                "coordinateSystem": coordinateSystem.rawValue,
                "units": units.rawValue,
                "inferredUnseenGeometry": inferredUnseenGeometry,
            ],
        ]

        var jsonData = try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
        while !jsonData.count.isMultiple(of: 4) { jsonData.append(0x20) }
        while !binary.count.isMultiple(of: 4) { binary.append(0) }

        let totalLength = 12 + 8 + jsonData.count + 8 + binary.count
        var glb = Data()
        appendUInt32(0x4654_6C67, to: &glb)
        appendUInt32(2, to: &glb)
        appendUInt32(UInt32(totalLength), to: &glb)
        appendUInt32(UInt32(jsonData.count), to: &glb)
        appendUInt32(0x4E4F_534A, to: &glb)
        glb.append(jsonData)
        appendUInt32(UInt32(binary.count), to: &glb)
        appendUInt32(0x004E_4942, to: &glb)
        glb.append(binary)

        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try glb.write(to: url, options: .atomic)
    }

    private static func floatData(_ values: [Float]) -> Data {
        var data = Data()
        data.reserveCapacity(values.count * 4)
        for value in values { appendUInt32(value.bitPattern, to: &data) }
        return data
    }

    private static func uint32Data(_ values: [UInt32]) -> Data {
        var data = Data()
        data.reserveCapacity(values.count * 4)
        for value in values { appendUInt32(value, to: &data) }
        return data
    }

    private static func appendUInt32(_ value: UInt32, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
}
