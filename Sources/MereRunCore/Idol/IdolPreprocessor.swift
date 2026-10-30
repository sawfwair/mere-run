import Foundation
import MediaIO

public enum IdolInputMode: String, Codable, Sendable {
    case synthetic, real
    public var cropMargin: Double { self == .synthetic ? 1.2 : 1.7 }
}

public enum IdolPreprocessor {
    public static func prepare(imageURL: URL, size: Int = 1024, mode: IdolInputMode = .synthetic,
                               alreadyFramed: Bool = false,
                               backgroundRemover: ((MediaImage) throws -> MediaImage)? = nil) throws -> MediaImage {
        let admitted = try VFXImageInputSnapshotBatch.capture([imageURL])
        defer { admitted.cleanup() }
        let image = try MediaImageIO.decode(admitted.snapshotURLs[0])
        _ = try VFXImageInputValidator.validate(width: image.width, height: image.height, path: imageURL.path)
        return try prepare(image, size: size, mode: mode, alreadyFramed: alreadyFramed, backgroundRemover: backgroundRemover)
    }

    /// Resize the short side, infer missing alpha, erode it, frame the foreground,
    /// and composite on black in the same order as the publisher's pipeline.
    public static func prepare(_ image: MediaImage, size: Int = 1024, mode: IdolInputMode = .synthetic,
                               alreadyFramed: Bool = false,
                               backgroundRemover: ((MediaImage) throws -> MediaImage)? = nil) throws -> MediaImage {
        guard size > 0, size <= 1024 else { throw IdolError.invalidInput("Invalid conditioning image size.") }
        var working = image
        if !alreadyFramed {
            let scale = Double(size) / Double(min(image.width, image.height))
            let width = max(1, Int((Double(image.width) * scale).rounded(.toNearestOrEven)))
            let height = max(1, Int((Double(image.height) * scale).rounded(.toNearestOrEven)))
            guard width <= 8192, height <= 8192, width * height <= 8_388_608 else {
                throw IdolError.invalidInput("Conditioning resize exceeds the 8-megapixel working limit; crop the source image first.")
            }
            working = try IdolImageResize.resize(image, width: width, height: height)
            let transparent = stride(from: 3, to: working.rgba8.count, by: 4).contains { working.rgba8[$0] < 255 }
            if !transparent {
                guard let backgroundRemover else {
                    throw IdolError.invalidInput("Opaque input needs the BiRefNet checkpoint or explicit --already-framed input on black.")
                }
                working = try backgroundRemover(working)
            }
            var eroded = working.rgba8, left = width, top = height, right = -1, bottom = -1
            for y in 0..<height {
                for x in 0..<width {
                    var alpha: UInt8 = 255
                    for dy in -1...1 {
                        for dx in -1...1 {
                            alpha = min(alpha, working.rgba8[(min(max(y + dy, 0), height - 1) * width
                                + min(max(x + dx, 0), width - 1)) * 4 + 3])
                        }
                    }
                    eroded[(y * width + x) * 4 + 3] = alpha
                    if alpha > 0 { left = min(left, x); right = max(right, x); top = min(top, y); bottom = max(bottom, y) }
                }
            }
            guard right >= left, bottom >= top else { throw IdolError.invalidInput("Input alpha contains no foreground after erosion.") }
            let centerX = Double(left + right) / 2, centerY = Double(top + bottom) / 2
            let half = Double(max(right - left, bottom - top)) / 2 * mode.cropMargin
            let originX = Int(centerX - half), originY = Int(centerY - half)
            let cropWidth = Int(centerX + half) - originX, cropHeight = Int(centerY + half) - originY
            guard cropWidth > 0, cropHeight > 0 else { throw IdolError.invalidInput("Foreground crop is empty.") }
            var crop = [UInt8](repeating: 0, count: cropWidth * cropHeight * 4)
            for y in 0..<cropHeight {
                for x in 0..<cropWidth {
                    let sourceX = originX + x, sourceY = originY + y
                    if sourceX >= 0, sourceX < width, sourceY >= 0, sourceY < height {
                        for channel in 0..<4 { crop[(y * cropWidth + x) * 4 + channel] = eroded[(sourceY * width + sourceX) * 4 + channel] }
                    }
                }
            }
            working = try MediaImage(width: cropWidth, height: cropHeight, rgba8: crop)
        }
        let resized = try IdolImageResize.resize(working, width: size, height: size)
        var bytes = resized.rgba8
        for pixel in 0..<(size * size) {
            for channel in 0..<3 { bytes[pixel * 4 + channel] = IdolImageResize.multiply255(bytes[pixel * 4 + channel], bytes[pixel * 4 + 3]) }
            bytes[pixel * 4 + 3] = 255
        }
        return try MediaImage(width: size, height: size, rgba8: bytes)
    }
}
