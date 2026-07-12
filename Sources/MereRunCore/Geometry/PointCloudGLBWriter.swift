import Foundation

/// Dependency-free glTF 2.0 binary point-cloud writer (`mode = POINTS`).
public enum PointCloudGLBWriter {
    public static func write(_ cloud: PointCloudAsset, to url: URL) throws {
        var binary = Data()
        var bufferViews: [[String: Any]] = []
        var accessors: [[String: Any]] = []

        func appendView(_ data: Data) -> Int {
            while !binary.count.isMultiple(of: 4) { binary.append(0) }
            let offset = binary.count
            binary.append(data)
            let index = bufferViews.count
            bufferViews.append([
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": data.count,
                "target": 34_962,
            ])
            return index
        }

        let positionView = appendView(floatData(cloud.positions))
        accessors.append([
            "bufferView": positionView,
            "componentType": 5_126,
            "count": cloud.pointCount,
            "type": "VEC3",
            "min": cloud.bounds.minimum,
            "max": cloud.bounds.maximum,
        ])
        var attributes: [String: Any] = ["POSITION": accessors.count - 1]
        if let colors = cloud.colorsRGBA8 {
            let view = appendView(Data(colors))
            accessors.append([
                "bufferView": view,
                "componentType": 5_121,
                "normalized": true,
                "count": cloud.pointCount,
                "type": "VEC4",
            ])
            attributes["COLOR_0"] = accessors.count - 1
        }
        if let confidence = cloud.confidence {
            let view = appendView(floatData(confidence))
            accessors.append([
                "bufferView": view,
                "componentType": 5_126,
                "count": cloud.pointCount,
                "type": "SCALAR",
            ])
            attributes["_CONFIDENCE"] = accessors.count - 1
        }
        if let viewIndices = cloud.viewIndices {
            let view = appendView(uint32Data(viewIndices))
            accessors.append([
                "bufferView": view,
                "componentType": 5_125,
                "count": cloud.pointCount,
                "type": "SCALAR",
            ])
            attributes["_VIEW_INDEX"] = accessors.count - 1
        }

        let json: [String: Any] = [
            "asset": ["version": "2.0", "generator": "mere.run native geometry"],
            "scene": 0,
            "scenes": [["nodes": [0]]],
            "nodes": [["mesh": 0, "name": "Multi-view point cloud"]],
            "meshes": [[
                "name": "Multi-view point cloud",
                "primitives": [["attributes": attributes, "mode": 0]],
            ]],
            "buffers": [["byteLength": binary.count]],
            "bufferViews": bufferViews,
            "accessors": accessors,
            "extras": [
                "coordinateSystem": cloud.coordinateSystem.rawValue,
                "units": cloud.units.rawValue,
            ],
        ]
        var jsonData = try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
        while !jsonData.count.isMultiple(of: 4) { jsonData.append(0x20) }
        while !binary.count.isMultiple(of: 4) { binary.append(0) }

        var glb = Data()
        appendUInt32(0x4654_6C67, to: &glb)
        appendUInt32(2, to: &glb)
        appendUInt32(UInt32(12 + 8 + jsonData.count + 8 + binary.count), to: &glb)
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
