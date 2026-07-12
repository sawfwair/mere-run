import Foundation

/// Minimal, standards-compliant uncompressed scanline OpenEXR writer.
///
/// Geometry interchange only needs full-precision float channels. Keeping this
/// writer in core avoids an external command, Python sidecar, or platform-only
/// image API in the inference path.
public enum OpenEXRWriter {
    public static func writeFloatChannels(
        _ channels: [(name: String, values: [Float])],
        width: Int,
        height: Int,
        to url: URL
    ) throws {
        guard width > 0, height > 0 else {
            throw GeometryError.invalidDimensions(width: width, height: height)
        }
        let pixelCount = width * height
        for channel in channels {
            guard !channel.name.isEmpty, channel.name.utf8.allSatisfy({ $0 < 128 && $0 != 0 }) else {
                throw GeometryError.invalidIntrinsics("EXR channel names must be non-empty ASCII")
            }
            guard channel.values.count == pixelCount else {
                throw GeometryError.invalidElementCount(
                    field: "EXR channel \(channel.name)",
                    expected: pixelCount,
                    actual: channel.values.count
                )
            }
        }
        guard !channels.isEmpty else {
            throw GeometryError.invalidElementCount(field: "EXR channels", expected: 1, actual: 0)
        }

        var header = Data()
        appendUInt32(20_000_630, to: &header) // OpenEXR magic number.
        appendUInt32(2, to: &header) // Version 2, scanline image.
        appendAttribute(name: "channels", type: "chlist", value: channelList(channels.map(\.name)), to: &header)
        appendAttribute(name: "compression", type: "compression", value: Data([0]), to: &header)
        appendAttribute(name: "dataWindow", type: "box2i", value: box(width: width, height: height), to: &header)
        appendAttribute(name: "displayWindow", type: "box2i", value: box(width: width, height: height), to: &header)
        appendAttribute(name: "lineOrder", type: "lineOrder", value: Data([0]), to: &header)
        appendAttribute(name: "pixelAspectRatio", type: "float", value: floatData(1), to: &header)
        var center = Data()
        appendFloat(0, to: &center)
        appendFloat(0, to: &center)
        appendAttribute(name: "screenWindowCenter", type: "v2f", value: center, to: &header)
        appendAttribute(name: "screenWindowWidth", type: "float", value: floatData(1), to: &header)
        header.append(0) // End of header attributes.

        let bytesPerScanline = 8 + width * channels.count * MemoryLayout<Float>.size
        let scanlineDataOffset = header.count + height * MemoryLayout<UInt64>.size
        var output = header
        output.reserveCapacity(scanlineDataOffset + height * bytesPerScanline)
        for y in 0..<height {
            appendUInt64(UInt64(scanlineDataOffset + y * bytesPerScanline), to: &output)
        }
        for y in 0..<height {
            appendInt32(Int32(y), to: &output)
            appendInt32(Int32(width * channels.count * MemoryLayout<Float>.size), to: &output)
            let rowStart = y * width
            for channel in channels {
                for x in 0..<width {
                    appendFloat(channel.values[rowStart + x], to: &output)
                }
            }
        }

        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try output.write(to: url, options: .atomic)
    }

    private static func channelList(_ names: [String]) -> Data {
        var data = Data()
        for name in names {
            appendCString(name, to: &data)
            appendInt32(2, to: &data) // FLOAT pixel type.
            data.append(0) // pLinear.
            data.append(contentsOf: [0, 0, 0]) // Reserved.
            appendInt32(1, to: &data) // xSampling.
            appendInt32(1, to: &data) // ySampling.
        }
        data.append(0)
        return data
    }

    private static func box(width: Int, height: Int) -> Data {
        var data = Data()
        appendInt32(0, to: &data)
        appendInt32(0, to: &data)
        appendInt32(Int32(width - 1), to: &data)
        appendInt32(Int32(height - 1), to: &data)
        return data
    }

    private static func floatData(_ value: Float) -> Data {
        var data = Data()
        appendFloat(value, to: &data)
        return data
    }

    private static func appendAttribute(name: String, type: String, value: Data, to data: inout Data) {
        appendCString(name, to: &data)
        appendCString(type, to: &data)
        appendInt32(Int32(value.count), to: &data)
        data.append(value)
    }

    private static func appendCString(_ value: String, to data: inout Data) {
        data.append(contentsOf: value.utf8)
        data.append(0)
    }

    private static func appendInt32(_ value: Int32, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private static func appendUInt32(_ value: UInt32, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private static func appendUInt64(_ value: UInt64, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private static func appendFloat(_ value: Float, to data: inout Data) {
        appendUInt32(value.bitPattern, to: &data)
    }
}
