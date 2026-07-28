import ArgumentParser
import Foundation
import MediaIO

struct ImageOutpaintInsets: Codable, Equatable {
    let top: Int
    let right: Int
    let bottom: Int
    let left: Int

    static func parse(_ value: String) throws -> Self {
        let parts = value
            .split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard parts.count == 4,
              let top = Int(parts[0]),
              let right = Int(parts[1]),
              let bottom = Int(parts[2]),
              let left = Int(parts[3]),
              [top, right, bottom, left].allSatisfy({ $0 >= 0 }) else {
            throw ValidationError("--outpaint must be top,right,bottom,left using nonnegative pixel values.")
        }
        guard top + right + bottom + left > 0 else {
            throw ValidationError("--outpaint must expand at least one edge.")
        }
        return Self(top: top, right: right, bottom: bottom, left: left)
    }

    var description: String {
        "\(top),\(right),\(bottom),\(left)"
    }
}

/// Prepares a portable img2img canvas and deterministically restores every unmasked pixel after
/// generation. This lets every supported image backend offer the same mask/outpaint contract,
/// even when its native pipeline only exposes whole-image conditioning.
struct ImageEditPreparation {
    let generationInputURL: URL
    let baseImage: MediaImage
    let editMask: [UInt8]
    let featherPixels: Int
    let temporaryDirectory: URL

    static func make(
        inputURL: URL,
        maskURL: URL?,
        outpaint: ImageOutpaintInsets?,
        width: Int,
        height: Int,
        featherPixels: Int,
        fileManager: FileManager = .default
    ) throws -> Self {
        guard featherPixels >= 0 else {
            throw ValidationError("--mask-feather must be >= 0")
        }
        let source = try MediaImageIO.decode(inputURL)
        let insets = outpaint ?? ImageOutpaintInsets(top: 0, right: 0, bottom: 0, left: 0)
        let availableWidth = width - insets.left - insets.right
        let availableHeight = height - insets.top - insets.bottom
        guard availableWidth > 0, availableHeight > 0 else {
            throw ValidationError("--outpaint padding must leave a positive source region inside --width/--height.")
        }

        // The padding contract is exact: each requested edge remains the specified number of
        // pixels. As with ordinary image-to-image, the source is normalized to the remaining
        // requested generation dimensions.
        let placedWidth = availableWidth
        let placedHeight = availableHeight
        let resizedSource = try MediaImageIO.resized(source, width: placedWidth, height: placedHeight)
        let originX = insets.left
        let originY = insets.top

        var baseRGBA = [UInt8](repeating: 127, count: width * height * 4)
        var mask = [UInt8](repeating: outpaint == nil ? 0 : 255, count: width * height)
        for pixel in 0..<(width * height) {
            baseRGBA[(pixel * 4) + 3] = 255
        }

        let sourceMask: MediaImage?
        if let maskURL {
            sourceMask = try MediaImageIO.resized(
                MediaImageIO.decode(maskURL),
                width: placedWidth,
                height: placedHeight
            )
        } else {
            sourceMask = nil
        }

        for y in 0..<placedHeight {
            for x in 0..<placedWidth {
                let sourceIndex = ((y * placedWidth) + x) * 4
                let canvasPixel = ((y + originY) * width) + x + originX
                let canvasIndex = canvasPixel * 4
                baseRGBA[canvasIndex] = resizedSource.rgba8[sourceIndex]
                baseRGBA[canvasIndex + 1] = resizedSource.rgba8[sourceIndex + 1]
                baseRGBA[canvasIndex + 2] = resizedSource.rgba8[sourceIndex + 2]
                baseRGBA[canvasIndex + 3] = 255
                if let sourceMask {
                    let luminance = (
                        Int(sourceMask.rgba8[sourceIndex])
                            + Int(sourceMask.rgba8[sourceIndex + 1])
                            + Int(sourceMask.rgba8[sourceIndex + 2])
                    ) / 3
                    mask[canvasPixel] = UInt8(clamping: luminance)
                } else {
                    mask[canvasPixel] = 0
                }
            }
        }

        let baseImage = try MediaImage(width: width, height: height, rgba8: baseRGBA)
        let temporaryDirectory = fileManager.temporaryDirectory
            .appendingPathComponent("mererun-image-edit-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
        let generationInputURL = temporaryDirectory.appendingPathComponent("conditioning.png")
        try MediaImageIO.writePNG(baseImage, to: generationInputURL)

        return Self(
            generationInputURL: generationInputURL,
            baseImage: baseImage,
            editMask: mask,
            featherPixels: featherPixels,
            temporaryDirectory: temporaryDirectory
        )
    }

    func finish(generatedURL: URL) throws {
        var generated = try MediaImageIO.decode(generatedURL)
        if generated.width != baseImage.width || generated.height != baseImage.height {
            generated = try MediaImageIO.resized(
                generated,
                width: baseImage.width,
                height: baseImage.height
            )
        }
        let mask = Self.feather(
            editMask,
            width: baseImage.width,
            height: baseImage.height,
            radius: featherPixels
        )
        var rgba = baseImage.rgba8
        for pixel in 0..<(baseImage.width * baseImage.height) {
            let alpha = Int(mask[pixel])
            guard alpha > 0 else { continue }
            let inverse = 255 - alpha
            let offset = pixel * 4
            for channel in 0..<3 {
                rgba[offset + channel] = UInt8(
                    clamping: (
                        (Int(generated.rgba8[offset + channel]) * alpha)
                            + (Int(baseImage.rgba8[offset + channel]) * inverse)
                            + 127
                    ) / 255
                )
            }
            rgba[offset + 3] = 255
        }
        try MediaImageIO.writePNG(
            try MediaImage(width: baseImage.width, height: baseImage.height, rgba8: rgba),
            to: generatedURL
        )
    }

    func cleanup(fileManager: FileManager = .default) {
        try? fileManager.removeItem(at: temporaryDirectory)
    }

    static func feather(_ source: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        guard radius > 0, source.count == width * height else { return source }
        var horizontal = [UInt8](repeating: 0, count: source.count)
        var output = [UInt8](repeating: 0, count: source.count)
        for y in 0..<height {
            var sum = 0
            for x in -radius...radius {
                sum += Int(source[(y * width) + min(max(x, 0), width - 1)])
            }
            for x in 0..<width {
                horizontal[(y * width) + x] = UInt8(clamping: sum / ((radius * 2) + 1))
                let removingX = min(max(x - radius, 0), width - 1)
                let addingX = min(max(x + radius + 1, 0), width - 1)
                sum += Int(source[(y * width) + addingX]) - Int(source[(y * width) + removingX])
            }
        }
        for x in 0..<width {
            var sum = 0
            for y in -radius...radius {
                sum += Int(horizontal[(min(max(y, 0), height - 1) * width) + x])
            }
            for y in 0..<height {
                output[(y * width) + x] = UInt8(clamping: sum / ((radius * 2) + 1))
                let removingY = min(max(y - radius, 0), height - 1)
                let addingY = min(max(y + radius + 1, 0), height - 1)
                sum += Int(horizontal[(addingY * width) + x]) - Int(horizontal[(removingY * width) + x])
            }
        }
        return output
    }
}
