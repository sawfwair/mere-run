import Foundation

public enum GeometryProjection {
    /// Reprojects a depth map using normalized pinhole intrinsics.
    public static func pointMap(
        depth: [Float],
        validity: [UInt8],
        intrinsics: GeometryCameraIntrinsics
    ) throws -> [Float] {
        let width = intrinsics.imageWidth
        let height = intrinsics.imageHeight
        let pixels = width * height
        guard depth.count == pixels else {
            throw GeometryError.invalidElementCount(field: "depth", expected: pixels, actual: depth.count)
        }
        guard validity.count == pixels else {
            throw GeometryError.invalidElementCount(field: "validity", expected: pixels, actual: validity.count)
        }

        let fx = Float(intrinsics.normalizedFX)
        let fy = Float(intrinsics.normalizedFY)
        let cx = Float(intrinsics.normalizedCX)
        let cy = Float(intrinsics.normalizedCY)
        var points = [Float](repeating: .infinity, count: pixels * 3)
        for y in 0..<height {
            let v = (Float(y) + 0.5) / Float(height)
            for x in 0..<width {
                let pixel = y * width + x
                let z = depth[pixel]
                guard validity[pixel] != 0, z.isFinite, z > 0 else { continue }
                let u = (Float(x) + 0.5) / Float(width)
                let base = pixel * 3
                points[base] = (u - cx) / fx * z
                points[base + 1] = (v - cy) / fy * z
                points[base + 2] = z
            }
        }
        return points
    }

    public static func depthStatistics(for frame: DenseGeometryFrame) throws -> GeometryDepthStatistics {
        var validCount = 0
        var minimum = Double.infinity
        var maximum = -Double.infinity
        var sum = 0.0
        for pixel in 0..<frame.pixelCount where frame.isValid(pixel: pixel) {
            let value = Double(frame.depth[pixel])
            validCount += 1
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            sum += value
        }
        guard validCount > 0 else { throw GeometryError.noValidGeometry }
        return GeometryDepthStatistics(
            validPixelCount: validCount,
            invalidPixelCount: frame.pixelCount - validCount,
            minimum: minimum,
            maximum: maximum,
            mean: sum / Double(validCount)
        )
    }
}
