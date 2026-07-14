import Foundation

/// Dependency-free glTF 2.0 binary writer for indexed, vertex-colored meshes.
public enum MeshGLBWriter {
    public static func write(_ source: MeshAsset, to url: URL) throws {
        let mesh = source.normals == nil ? try source.withGeneratedNormals() : source
        var binary = Data()
        var bufferViews: [[String: Any]] = []
        var accessors: [[String: Any]] = []

        func appendView(_ data: Data, target: Int) -> Int {
            while !binary.count.isMultiple(of: 4) { binary.append(0) }
            let offset = binary.count
            binary.append(data)
            let index = bufferViews.count
            bufferViews.append([
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": data.count,
                "target": target,
            ])
            return index
        }

        let positionView = appendView(floatData(mesh.vertices), target: 34_962)
        accessors.append([
            "bufferView": positionView,
            "componentType": 5_126,
            "count": mesh.vertexCount,
            "type": "VEC3",
            "min": mesh.bounds.minimum,
            "max": mesh.bounds.maximum,
        ])
        let positionAccessor = accessors.count - 1

        let normalView = appendView(floatData(mesh.normals!), target: 34_962)
        accessors.append([
            "bufferView": normalView,
            "componentType": 5_126,
            "count": mesh.vertexCount,
            "type": "VEC3",
        ])
        let normalAccessor = accessors.count - 1

        var colorAccessor: Int?
        if let colors = mesh.colorsRGBA8 {
            // glTF defines COLOR_0 as linear; colorsRGBA8 is sRGB-encoded.
            let linear = VertexColorTransfer.linearRGBA16(fromSRGBA8: colors)
            let view = appendView(uint16Data(linear), target: 34_962)
            accessors.append([
                "bufferView": view,
                "componentType": 5_123,
                "normalized": true,
                "count": mesh.vertexCount,
                "type": "VEC4",
            ])
            colorAccessor = accessors.count - 1
        }

        var uvAccessor: Int?
        if let uv = mesh.textureCoordinates {
            let view = appendView(floatData(uv), target: 34_962)
            accessors.append([
                "bufferView": view,
                "componentType": 5_126,
                "count": mesh.vertexCount,
                "type": "VEC2",
            ])
            uvAccessor = accessors.count - 1
        }

        let indexView = appendView(uint32Data(mesh.indices), target: 34_963)
        accessors.append([
            "bufferView": indexView,
            "componentType": 5_125,
            "count": mesh.indices.count,
            "type": "SCALAR",
        ])
        let indexAccessor = accessors.count - 1

        var attributes: [String: Any] = [
            "POSITION": positionAccessor,
            "NORMAL": normalAccessor,
        ]
        if let colorAccessor { attributes["COLOR_0"] = colorAccessor }
        if let uvAccessor { attributes["TEXCOORD_0"] = uvAccessor }

        let json: [String: Any] = [
            "asset": ["version": "2.0", "generator": "mere.run native Asset3D"],
            "scene": 0,
            "scenes": [["nodes": [0]]],
            "nodes": [["mesh": 0, "name": "Reconstructed Asset"]],
            "meshes": [[
                "name": "Reconstructed Mesh",
                "primitives": [[
                    "attributes": attributes,
                    "indices": indexAccessor,
                    "material": 0,
                    "mode": 4,
                ]],
            ]],
            "materials": [[
                "name": "Vertex Colors",
                "doubleSided": true,
                "pbrMetallicRoughness": [
                    "baseColorFactor": [1, 1, 1, 1],
                    "metallicFactor": 0,
                    "roughnessFactor": 1,
                ],
            ]],
            "buffers": [["byteLength": binary.count]],
            "bufferViews": bufferViews,
            "accessors": accessors,
            "extras": [
                "coordinateSystem": mesh.coordinateSystem.rawValue,
                "units": mesh.units.rawValue,
                "inferredUnseenGeometry": mesh.inferredUnseenGeometry,
            ],
        ]

        var jsonData = try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
        while !jsonData.count.isMultiple(of: 4) { jsonData.append(0x20) }
        while !binary.count.isMultiple(of: 4) { binary.append(0) }

        let totalLength = 12 + 8 + jsonData.count + 8 + binary.count
        var glb = Data()
        appendUInt32(0x4654_6C67, to: &glb) // glTF
        appendUInt32(2, to: &glb)
        appendUInt32(UInt32(totalLength), to: &glb)
        appendUInt32(UInt32(jsonData.count), to: &glb)
        appendUInt32(0x4E4F_534A, to: &glb) // JSON
        glb.append(jsonData)
        appendUInt32(UInt32(binary.count), to: &glb)
        appendUInt32(0x004E_4942, to: &glb) // BIN
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

    private static func uint16Data(_ values: [UInt16]) -> Data {
        var data = Data()
        data.reserveCapacity(values.count * 2)
        for value in values {
            var value = value.littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        return data
    }

    private static func uint32Data(_ values: [UInt32]) -> Data {
        var data = Data()
        data.reserveCapacity(values.count * 4)
        for value in values { appendUInt32(value, to: &data) }
        return data
    }

    private static func appendUInt32(_ value: UInt32, to data: inout Data) {
        var value = value.littleEndian
        withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
}
