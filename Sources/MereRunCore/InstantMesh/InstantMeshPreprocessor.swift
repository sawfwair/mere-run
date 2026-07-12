import Foundation
import MediaIO
@preconcurrency import MLX

public enum InstantMeshPreprocessingError: Error, Equatable, LocalizedError, Sendable {
    case invalidViewCount(Int)
    case invalidImageSize(Int)
    case invalidCameraRadius(Float)
    case invalidCameraFieldOfView(Float)
    case invalidCameraBasis
    case cameraCountMismatch(expected: Int, actual: Int)
    case invalidCamera(index: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidViewCount(let count):
            "InstantMesh Base reconstruction requires exactly 4 or 6 ordered views; received \(count)."
        case .invalidImageSize(let size):
            "InstantMesh conditioning image size must be positive; received \(size)."
        case .invalidCameraRadius(let radius):
            "InstantMesh camera radius must be positive and finite; received \(radius)."
        case .invalidCameraFieldOfView(let degrees):
            "InstantMesh camera field of view must produce a finite focal length in (0, 180) degrees; received \(degrees)."
        case .invalidCameraBasis:
            "InstantMesh could not normalize the official camera basis."
        case .cameraCountMismatch(let expected, let actual):
            "InstantMesh camera count must match the \(expected) input views; received \(actual)."
        case .invalidCamera(let index):
            "InstantMesh camera \(index) must contain 16 finite C2W/intrinsic values."
        }
    }
}

public struct InstantMeshPreprocessedBatch {
    /// `[1, view, 320, 320, 3]`, white-composited RGB in `[0, 1]`.
    public let images: MLXArray
    /// `[1, view, 16]`, row-major C2W(3x4) + `(fx, fy, cx, cy)`.
    public let cameras: MLXArray
    public let sourceImages: [MediaImage]
    public let processedImages: [MediaImage]
    /// Exact row-major C2W(3x4) + normalized intrinsics used by inference.
    public let cameraValues: [[Float]]
    public let usedOfficialCameraRig: Bool
}

public enum InstantMeshCameraRig {
    /// Camera conditioning used by the released reconstruction checkpoint.
    /// This is only a deterministic camera convention; it does not run or
    /// package the excluded Zero123++ view-generation model.
    public static func official(viewCount: Int, radius: Float = 4, fovDegrees: Float = 30) throws -> [[Float]] {
        guard viewCount == 4 || viewCount == 6 else {
            throw InstantMeshPreprocessingError.invalidViewCount(viewCount)
        }
        guard radius.isFinite, radius > 0 else {
            throw InstantMeshPreprocessingError.invalidCameraRadius(radius)
        }
        guard fovDegrees.isFinite, fovDegrees > 0, fovDegrees < 180 else {
            throw InstantMeshPreprocessingError.invalidCameraFieldOfView(fovDegrees)
        }
        let azimuths: [Float] = [30, 90, 150, 210, 270, 330]
        let elevations: [Float] = [20, -10, 20, -10, 20, -10]
        let focal64 = 0.5 / tan(Double(fovDegrees) * .pi / 360)
        guard focal64.isFinite, focal64 > 0, focal64 <= Double(Float.greatestFiniteMagnitude) else {
            throw InstantMeshPreprocessingError.invalidCameraFieldOfView(fovDegrees)
        }
        let focal = Float(focal64)
        let six = try zip(azimuths, elevations).map { azimuth, elevation -> [Float] in
            let azimuthRadians = azimuth * .pi / 180
            let elevationRadians = elevation * .pi / 180
            let position = Vector3(
                x: radius * cos(elevationRadians) * cos(azimuthRadians),
                y: radius * cos(elevationRadians) * sin(azimuthRadians),
                z: radius * sin(elevationRadians)
            )
            let zAxis = try position.normalized()
            let xAxis = try Vector3(x: 0, y: 0, z: 1).cross(zAxis).normalized()
            let yAxis = try zAxis.cross(xAxis).normalized()
            return [
                xAxis.x, yAxis.x, zAxis.x, position.x,
                xAxis.y, yAxis.y, zAxis.y, position.y,
                xAxis.z, yAxis.z, zAxis.z, position.z,
                focal, focal, 0.5, 0.5,
            ]
        }
        if viewCount == 6 { return six }
        return [six[0], six[2], six[4], six[5]]
    }

    private struct Vector3 {
        let x: Float
        let y: Float
        let z: Float

        func normalized() throws -> Vector3 {
            let scale = max(abs(x), abs(y), abs(z))
            guard scale.isFinite, scale > 0 else {
                throw InstantMeshPreprocessingError.invalidCameraBasis
            }
            let scaledX = x / scale
            let scaledY = y / scale
            let scaledZ = z / scale
            let length = sqrt(scaledX * scaledX + scaledY * scaledY + scaledZ * scaledZ)
            guard length.isFinite, length > 0 else {
                throw InstantMeshPreprocessingError.invalidCameraBasis
            }
            return Vector3(x: scaledX / length, y: scaledY / length, z: scaledZ / length)
        }

        func cross(_ other: Vector3) -> Vector3 {
            Vector3(
                x: y * other.z - z * other.y,
                y: z * other.x - x * other.z,
                z: x * other.y - y * other.x
            )
        }
    }
}

public enum InstantMeshPreprocessor {
    public static func prepare(
        sourceImages: [MediaImage],
        cameras suppliedCameras: [[Float]]? = nil,
        size: Int = InstantMeshConfiguration.production.conditioningImageSize
    ) throws -> InstantMeshPreprocessedBatch {
        guard sourceImages.count == 4 || sourceImages.count == 6 else {
            throw InstantMeshPreprocessingError.invalidViewCount(sourceImages.count)
        }
        guard size > 0 else {
            throw InstantMeshPreprocessingError.invalidImageSize(size)
        }
        let cameraValues: [[Float]]
        let usedOfficialCameraRig: Bool
        if let suppliedCameras {
            guard suppliedCameras.count == sourceImages.count else {
                throw InstantMeshPreprocessingError.cameraCountMismatch(
                    expected: sourceImages.count,
                    actual: suppliedCameras.count
                )
            }
            for (index, camera) in suppliedCameras.enumerated()
                where camera.count != 16 || !camera.allSatisfy(\.isFinite) {
                throw InstantMeshPreprocessingError.invalidCamera(index: index)
            }
            cameraValues = suppliedCameras
            usedOfficialCameraRig = false
        } else {
            cameraValues = try InstantMeshCameraRig.official(viewCount: sourceImages.count)
            usedOfficialCameraRig = true
        }

        var processedImages: [MediaImage] = []
        var imageValues: [Float] = []
        processedImages.reserveCapacity(sourceImages.count)
        imageValues.reserveCapacity(sourceImages.count * size * size * 3)
        for image in sourceImages {
            let white = try compositeOnWhite(image)
            let sourceRGB = rgbFloat(white)
            let resizedRGB = if white.width == size, white.height == size {
                sourceRGB
            } else {
                antialiasedBicubicResize(
                    sourceRGB,
                    sourceWidth: white.width,
                    sourceHeight: white.height,
                    targetWidth: size,
                    targetHeight: size
                )
            }
            imageValues.append(contentsOf: resizedRGB)
            processedImages.append(try previewImage(resizedRGB, width: size, height: size))
        }
        return InstantMeshPreprocessedBatch(
            images: MLXArray(imageValues).reshaped(1, sourceImages.count, size, size, 3),
            cameras: MLXArray(cameraValues.flatMap { $0 }).reshaped(1, sourceImages.count, 16),
            sourceImages: sourceImages,
            processedImages: processedImages,
            cameraValues: cameraValues,
            usedOfficialCameraRig: usedOfficialCameraRig
        )
    }

    private static func compositeOnWhite(_ image: MediaImage) throws -> MediaImage {
        var rgba = image.rgba8
        for pixel in 0..<(image.width * image.height) {
            let offset = pixel * 4
            let alpha = UInt16(rgba[offset + 3])
            let inverse = UInt16(255) - alpha
            for channel in 0..<3 {
                let value = UInt16(rgba[offset + channel]) * alpha + UInt16(255) * inverse
                rgba[offset + channel] = UInt8((value + 127) / 255)
            }
            rgba[offset + 3] = 255
        }
        return try MediaImage(width: image.width, height: image.height, rgba8: rgba)
    }

    private static func rgbFloat(_ image: MediaImage) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(image.width * image.height * 3)
        for pixel in 0..<(image.width * image.height) {
            let offset = pixel * 4
            values.append(Float(image.rgba8[offset]) / 255)
            values.append(Float(image.rgba8[offset + 1]) / 255)
            values.append(Float(image.rgba8[offset + 2]) / 255)
        }
        return values
    }

    private static func previewImage(
        _ rgb: [Float],
        width: Int,
        height: Int
    ) throws -> MediaImage {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0..<(width * height) {
            for channel in 0..<3 {
                let value = min(1, max(0, rgb[pixel * 3 + channel]))
                rgba[pixel * 4 + channel] = UInt8(
                    clamping: Int((value * 255).rounded())
                )
            }
        }
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }

    /// Separable port of torchvision/PyTorch tensor resize with
    /// `mode="bicubic"`, `align_corners=false`, and `antialias=true`.
    /// PyTorch's antialiased cubic kernel uses `a = -0.5`, widens support while
    /// downsampling, clips support at image edges, and renormalizes each output
    /// sample. The released InstantMesh path clamps the result to `[0, 1]`.
    static func antialiasedBicubicResize(
        _ source: [Float],
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> [Float] {
        precondition(sourceWidth > 0 && sourceHeight > 0)
        precondition(targetWidth > 0 && targetHeight > 0)
        precondition(source.count == sourceWidth * sourceHeight * 3)
        if sourceWidth == targetWidth, sourceHeight == targetHeight { return source }

        let horizontal = bicubicResamplingWeights(
            sourceSize: sourceWidth,
            targetSize: targetWidth
        )
        var intermediate = [Float](repeating: 0, count: sourceHeight * targetWidth * 3)
        for y in 0..<sourceHeight {
            for targetX in 0..<targetWidth {
                let targetOffset = (y * targetWidth + targetX) * 3
                for (sourceX, weight) in horizontal[targetX] {
                    let sourceOffset = (y * sourceWidth + sourceX) * 3
                    intermediate[targetOffset] += source[sourceOffset] * weight
                    intermediate[targetOffset + 1] += source[sourceOffset + 1] * weight
                    intermediate[targetOffset + 2] += source[sourceOffset + 2] * weight
                }
            }
        }

        let vertical = bicubicResamplingWeights(
            sourceSize: sourceHeight,
            targetSize: targetHeight
        )
        var output = [Float](repeating: 0, count: targetHeight * targetWidth * 3)
        for targetY in 0..<targetHeight {
            for (sourceY, weight) in vertical[targetY] {
                for x in 0..<targetWidth {
                    let sourceOffset = (sourceY * targetWidth + x) * 3
                    let targetOffset = (targetY * targetWidth + x) * 3
                    output[targetOffset] += intermediate[sourceOffset] * weight
                    output[targetOffset + 1] += intermediate[sourceOffset + 1] * weight
                    output[targetOffset + 2] += intermediate[sourceOffset + 2] * weight
                }
            }
        }
        for index in output.indices {
            output[index] = min(1, max(0, output[index]))
        }
        return output
    }

    private static func bicubicResamplingWeights(
        sourceSize: Int,
        targetSize: Int
    ) -> [[(index: Int, weight: Float)]] {
        let scale = Float(sourceSize) / Float(targetSize)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        var result: [[(index: Int, weight: Float)]] = []
        result.reserveCapacity(targetSize)
        for destination in 0..<targetSize {
            let center = scale * (Float(destination) + 0.5)
            let first = max(0, Int(ceil(center - support - 0.5)))
            let last = min(sourceSize - 1, Int(floor(center + support - 0.5)))
            var weights: [(Int, Float)] = []
            weights.reserveCapacity(max(0, last - first + 1))
            var total: Float = 0
            if first <= last {
                for index in first...last {
                    let distance = (Float(index) + 0.5 - center) / filterScale
                    let weight = cubicKernel(distance)
                    if weight != 0 {
                        weights.append((index, weight))
                        total += weight
                    }
                }
            }
            if total != 0 {
                weights = weights.map { ($0.0, $0.1 / total) }
            }
            result.append(weights)
        }
        return result
    }

    private static func cubicKernel(_ value: Float) -> Float {
        let x = abs(value)
        let coefficient: Float = -0.5
        if x < 1 {
            return ((coefficient + 2) * x - (coefficient + 3)) * x * x + 1
        }
        if x < 2 {
            return ((coefficient * x - 5 * coefficient) * x + 8 * coefficient) * x
                - 4 * coefficient
        }
        return 0
    }
}
