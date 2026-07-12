import Foundation
import MediaIO

public enum GeometryPreviewWriter {
    public static func writeDepth(_ frame: DenseGeometryFrame, to url: URL) throws {
        let validDepth = (0..<frame.pixelCount)
            .filter { frame.isValid(pixel: $0) }
            .map { frame.depth[$0] }
            .sorted()
        guard !validDepth.isEmpty else { throw GeometryError.noValidGeometry }
        let near = percentile(validDepth, 0.02)
        let far = percentile(validDepth, 0.98)
        try writeDepth(
            frame.depth,
            width: frame.width,
            height: frame.height,
            validity: frame.validity,
            near: near,
            far: far,
            to: url
        )
    }

    public static func writeDepth(
        _ depth: [Float],
        width: Int,
        height: Int,
        validity: [UInt8]? = nil,
        near: Float,
        far: Float,
        to url: URL
    ) throws {
        let pixelCount = width * height
        guard depth.count == pixelCount else {
            throw GeometryError.invalidElementCount(field: "preview depth", expected: pixelCount, actual: depth.count)
        }
        if let validity, validity.count != pixelCount {
            throw GeometryError.invalidElementCount(field: "preview validity", expected: pixelCount, actual: validity.count)
        }
        let range = max(far - near, Float.ulpOfOne)
        var rgba = [UInt8](repeating: 0, count: pixelCount * 4)
        for pixel in 0..<pixelCount {
            let destination = pixel * 4
            let value = depth[pixel]
            let isMarkedValid = validity.map { $0[pixel] != 0 } ?? true
            guard isMarkedValid, value.isFinite, value > 0 else {
                rgba[destination + 3] = 0
                continue
            }
            // Near is bright and far is dark, matching common compositor depth previews.
            let normalized = 1 - min(1, max(0, (value - near) / range))
            let byte = UInt8((normalized * 255).rounded())
            rgba[destination] = byte
            rgba[destination + 1] = byte
            rgba[destination + 2] = byte
            rgba[destination + 3] = 255
        }
        try write(width: width, height: height, rgba: rgba, to: url)
    }

    public static func writeNormals(_ frame: DenseGeometryFrame, to url: URL) throws {
        guard let normals = frame.normals else {
            throw GeometryError.invalidElementCount(field: "normals", expected: frame.pixelCount * 3, actual: 0)
        }
        var rgba = [UInt8](repeating: 0, count: frame.pixelCount * 4)
        for pixel in 0..<frame.pixelCount {
            let destination = pixel * 4
            guard frame.isValid(pixel: pixel) else {
                rgba[destination + 3] = 0
                continue
            }
            let vector = pixel * 3
            for channel in 0..<3 {
                let normalized = min(1, max(0, normals[vector + channel] * 0.5 + 0.5))
                rgba[destination + channel] = UInt8((normalized * 255).rounded())
            }
            rgba[destination + 3] = 255
        }
        try write(width: frame.width, height: frame.height, rgba: rgba, to: url)
    }

    public static func writeValidity(_ frame: DenseGeometryFrame, to url: URL) throws {
        var rgba = [UInt8](repeating: 255, count: frame.pixelCount * 4)
        for pixel in 0..<frame.pixelCount {
            let value: UInt8 = frame.isValid(pixel: pixel) ? 255 : 0
            let destination = pixel * 4
            rgba[destination] = value
            rgba[destination + 1] = value
            rgba[destination + 2] = value
            rgba[destination + 3] = 255
        }
        try write(width: frame.width, height: frame.height, rgba: rgba, to: url)
    }

    private static func percentile(_ sorted: [Float], _ fraction: Double) -> Float {
        let index = min(sorted.count - 1, max(0, Int((Double(sorted.count - 1) * fraction).rounded())))
        return sorted[index]
    }

    private static func write(width: Int, height: Int, rgba: [UInt8], to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try MediaImageIO.writePNG(try MediaImage(width: width, height: height, rgba8: rgba), to: url)
    }
}
