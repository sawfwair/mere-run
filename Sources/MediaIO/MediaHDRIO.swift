import Foundation
#if canImport(CoreImage)
import CoreImage
#endif

public struct MediaEXRColorMetadata: Sendable, Hashable {
    public let chromaticities: [Float]
    public let colorSpace: String

    public init(chromaticities: [Float], colorSpace: String) {
        precondition(chromaticities.count == 8)
        self.chromaticities = chromaticities
        self.colorSpace = colorSpace
    }

    public static let rec709Linear = MediaEXRColorMetadata(
        chromaticities: [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290],
        colorSpace: "sRGB"
    )
    public static let acescgLinear = MediaEXRColorMetadata(
        chromaticities: [0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767],
        colorSpace: "ACEScg"
    )
    public static let acescct = MediaEXRColorMetadata(
        chromaticities: Self.acescgLinear.chromaticities,
        colorSpace: "ACEScct"
    )
    public static let logC3 = MediaEXRColorMetadata(
        chromaticities: Self.rec709Linear.chromaticities,
        colorSpace: "LogC3"
    )
}

public enum MediaHDRImageIO {
    public static func isEXR(_ url: URL) -> Bool {
        url.pathExtension.lowercased() == "exr"
    }

    public static func exrFrameURLs(in directory: URL) throws -> [URL] {
        let values = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        return values.filter(isEXR).sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    public static func isEXRDirectory(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return false
        }
        return (try? exrFrameURLs(in: url).isEmpty == false) ?? false
    }

    public static func decodeEXR(_ url: URL) throws -> MediaFloatImage {
        #if canImport(CoreImage)
        return try decodeEXRWithCoreImage(url)
        #else
        try FFmpegMediaIO.decodeEXR(url)
        #endif
    }

    #if canImport(CoreImage)
    private static func decodeEXRWithCoreImage(_ url: URL) throws -> MediaFloatImage {
        guard let source = CIImage(
            contentsOf: url,
            options: [.applyOrientationProperty: false]
        ) else {
            throw MediaIOError.imageMetadataFailed(url)
        }
        let extent = source.extent.integral
        let width = Int(extent.width)
        let height = Int(extent.height)
        guard width > 0, height > 0 else {
            throw MediaIOError.imageMetadataFailed(url)
        }
        let context = CIContext(options: [
            .workingColorSpace: NSNull(),
            .outputColorSpace: NSNull(),
        ])
        var rgba = [Float](repeating: 0, count: width * height * 4)
        context.render(
            source,
            toBitmap: &rgba,
            rowBytes: width * 4 * MemoryLayout<Float>.size,
            bounds: extent,
            format: .RGBAf,
            colorSpace: nil
        )
        var rgb = [Float](repeating: 0, count: width * height * 3)
        for pixel in 0..<(width * height) {
            rgb[pixel * 3] = rgba[pixel * 4]
            rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1]
            rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2]
        }
        return try MediaFloatImage(width: width, height: height, rgb: rgb)
    }
    #endif

    public static func writeEXR(
        _ image: MediaFloatImage,
        metadata: MediaEXRColorMetadata = .rec709Linear,
        to url: URL
    ) throws {
        try MediaOpenEXRWriter.write(image, metadata: metadata, to: url)
    }

    public static func resized(
        _ image: MediaFloatImage,
        width: Int,
        height: Int
    ) throws -> MediaFloatImage {
        guard width > 0, height > 0 else {
            throw MediaIOError.invalidImageDimensions(width: width, height: height)
        }
        if image.width == width, image.height == height { return image }
        var output = [Float](repeating: 0, count: width * height * 3)
        let xScale = Float(image.width) / Float(width)
        let yScale = Float(image.height) / Float(height)
        for y in 0..<height {
            let sourceY = (Float(y) + 0.5) * yScale - 0.5
            let y0 = max(0, min(image.height - 1, Int(floor(sourceY))))
            let y1 = min(image.height - 1, y0 + 1)
            let yWeight = max(0, min(1, sourceY - Float(y0)))
            for x in 0..<width {
                let sourceX = (Float(x) + 0.5) * xScale - 0.5
                let x0 = max(0, min(image.width - 1, Int(floor(sourceX))))
                let x1 = min(image.width - 1, x0 + 1)
                let xWeight = max(0, min(1, sourceX - Float(x0)))
                for channel in 0..<3 {
                    let top = image.rgb[((y0 * image.width + x0) * 3) + channel]
                        * (1 - xWeight)
                        + image.rgb[((y0 * image.width + x1) * 3) + channel] * xWeight
                    let bottom = image.rgb[((y1 * image.width + x0) * 3) + channel]
                        * (1 - xWeight)
                        + image.rgb[((y1 * image.width + x1) * 3) + channel] * xWeight
                    output[((y * width + x) * 3) + channel] = top * (1 - yWeight) + bottom * yWeight
                }
            }
        }
        return try MediaFloatImage(width: width, height: height, rgb: output)
    }

    public static func centerCropped(
        _ image: MediaFloatImage,
        width: Int,
        height: Int
    ) throws -> MediaFloatImage {
        let scale = max(Float(width) / Float(image.width), Float(height) / Float(image.height))
        let scaledWidth = max(width, Int(ceil(Float(image.width) * scale)))
        let scaledHeight = max(height, Int(ceil(Float(image.height) * scale)))
        let resizedImage = try resized(image, width: scaledWidth, height: scaledHeight)
        let originX = (scaledWidth - width) / 2
        let originY = (scaledHeight - height) / 2
        var output = [Float](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                let source = ((y + originY) * scaledWidth + x + originX) * 3
                let destination = (y * width + x) * 3
                output[destination..<(destination + 3)] = resizedImage.rgb[source..<(source + 3)]
            }
        }
        return try MediaFloatImage(width: width, height: height, rgb: output)
    }

    public static func reflectPadded(
        _ image: MediaFloatImage,
        width: Int,
        height: Int
    ) throws -> MediaFloatImage {
        guard width > 0, height > 0 else {
            throw MediaIOError.invalidImageDimensions(width: width, height: height)
        }
        let fitted: MediaFloatImage
        if width >= image.width, height >= image.height {
            fitted = image
        } else {
            let scale = min(Float(width) / Float(image.width), Float(height) / Float(image.height))
            fitted = try resized(
                image,
                width: max(1, Int((Float(image.width) * scale).rounded())),
                height: max(1, Int((Float(image.height) * scale).rounded()))
            )
        }
        var output = [Float](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            let sourceY = reflectedIndex(y, count: fitted.height)
            for x in 0..<width {
                let sourceX = reflectedIndex(x, count: fitted.width)
                let source = (sourceY * fitted.width + sourceX) * 3
                let destination = (y * width + x) * 3
                output[destination..<(destination + 3)] = fitted.rgb[source..<(source + 3)]
            }
        }
        return try MediaFloatImage(width: width, height: height, rgb: output)
    }

    private static func reflectedIndex(_ index: Int, count: Int) -> Int {
        guard count > 1 else { return 0 }
        let period = 2 * count - 2
        let folded = index % period
        return folded < count ? folded : period - folded
    }
}

private enum MediaOpenEXRWriter {
    static func write(
        _ image: MediaFloatImage,
        metadata: MediaEXRColorMetadata,
        to url: URL
    ) throws {
        var red = [Float](repeating: 0, count: image.width * image.height)
        var green = red
        var blue = red
        for pixel in red.indices {
            red[pixel] = image.rgb[pixel * 3]
            green[pixel] = image.rgb[pixel * 3 + 1]
            blue[pixel] = image.rgb[pixel * 3 + 2]
        }

        let channels = [("R", red), ("G", green), ("B", blue)]
        var header = Data()
        appendUInt32(20_000_630, to: &header)
        appendUInt32(2, to: &header)
        appendAttribute(name: "channels", type: "chlist", value: channelList(channels.map(\.0)), to: &header)
        appendAttribute(name: "compression", type: "compression", value: Data([0]), to: &header)
        appendAttribute(name: "dataWindow", type: "box2i", value: box(image), to: &header)
        appendAttribute(name: "displayWindow", type: "box2i", value: box(image), to: &header)
        appendAttribute(name: "lineOrder", type: "lineOrder", value: Data([0]), to: &header)
        appendAttribute(name: "pixelAspectRatio", type: "float", value: floatData(1), to: &header)
        var center = Data()
        appendFloat(0, to: &center)
        appendFloat(0, to: &center)
        appendAttribute(name: "screenWindowCenter", type: "v2f", value: center, to: &header)
        appendAttribute(name: "screenWindowWidth", type: "float", value: floatData(1), to: &header)
        var chromaticities = Data()
        for value in metadata.chromaticities {
            appendFloat(value, to: &chromaticities)
        }
        appendAttribute(
            name: "chromaticities",
            type: "chromaticities",
            value: chromaticities,
            to: &header
        )
        appendAttribute(
            name: "colorSpace",
            type: "string",
            value: Data(metadata.colorSpace.utf8),
            to: &header
        )
        header.append(0)

        let scanlinePayloadBytes = image.width * channels.count * MemoryLayout<Float16>.size
        let bytesPerScanline = 8 + scanlinePayloadBytes
        let scanlineDataOffset = header.count + image.height * MemoryLayout<UInt64>.size
        var output = header
        output.reserveCapacity(scanlineDataOffset + image.height * bytesPerScanline)
        for y in 0..<image.height {
            appendUInt64(UInt64(scanlineDataOffset + y * bytesPerScanline), to: &output)
        }
        for y in 0..<image.height {
            appendInt32(Int32(y), to: &output)
            appendInt32(Int32(scanlinePayloadBytes), to: &output)
            let rowStart = y * image.width
            for channel in channels {
                for x in 0..<image.width {
                    appendUInt16(Float16(channel.1[rowStart + x]).bitPattern, to: &output)
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
            appendInt32(1, to: &data)
            data.append(contentsOf: [0, 0, 0, 0])
            appendInt32(1, to: &data)
            appendInt32(1, to: &data)
        }
        data.append(0)
        return data
    }

    private static func box(_ image: MediaFloatImage) -> Data {
        var data = Data()
        appendInt32(0, to: &data)
        appendInt32(0, to: &data)
        appendInt32(Int32(image.width - 1), to: &data)
        appendInt32(Int32(image.height - 1), to: &data)
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
        appendUInt32(UInt32(bitPattern: value), to: &data)
    }

    private static func appendUInt16(_ value: UInt16, to data: inout Data) {
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

public enum MediaHDRVideoIO {
    public typealias RGBFloatFrameProvider = (_ frameIndex: Int) throws -> [Float]

    /// Encodes already BT.2020/HLG-transfer RGB floats into a tagged Main10 HEVC MP4.
    public static func writeHLGMP4(
        rgbFloatFrameAt frameProvider: RGBFloatFrameProvider,
        width: Int,
        height: Int,
        frameCount: Int,
        fps: Double,
        to outputURL: URL
    ) throws {
        try FFmpegMediaIO.writeHLGMP4(
            rgbFloatFrameAt: frameProvider,
            width: width,
            height: height,
            frameCount: frameCount,
            fps: fps,
            to: outputURL
        )
    }
}
