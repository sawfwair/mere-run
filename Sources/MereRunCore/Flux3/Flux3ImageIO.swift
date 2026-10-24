import Foundation
import MediaIO
import MLX
#if canImport(ImageIO)
import ImageIO
#endif

enum Flux3ImageIO {
    struct Reference {
        let pixels: MLXArray
        let height: Int
        let width: Int
    }

    static func prepare(_ url: URL, multiple: Bool) throws -> Reference {
        var image = try MediaImageIO.decode(url)
        #if canImport(ImageIO)
        if let source = CGImageSourceCreateWithURL(url as CFURL, nil),
           let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
           let orientation = properties[kCGImagePropertyOrientation] as? Int {
            image = try oriented(image, orientation: orientation)
        }
        #endif
        let size = canvas(width: image.width, height: image.height, multiple: multiple)
        guard !multiple || min(size.fitWidth, size.fitHeight) >= 16 else {
            throw Flux3Error.invalidConfiguration("Each resized multi-reference image needs both sides at least 16 pixels.")
        }
        image = try MuseGlimmerImageProcessor.resizedLanczos(image, width: size.fitWidth, height: size.fitHeight)
        let pixels = try paddedOrCropped(image, width: size.width, height: size.height, reflect: !multiple)
        return Reference(pixels: MLXArray(MediaImageIO.rgbCHWFloat(pixels, normalizedToMinusOneToOne: true))
            .reshaped(1, 3, size.height, size.width), height: size.height, width: size.width)
    }

    static func canvas(width: Int, height: Int, multiple: Bool) -> (fitWidth: Int, fitHeight: Int, width: Int, height: Int) {
        precondition(width > 0 && height > 0)
        let budget = multiple ? 1024 * 1024 : 4 * 1024 * 1024
        var scale = min(1, sqrt(Double(budget) / Double(width) / Double(height)))
        if !multiple { scale = min(scale, 4096 / Double(max(width, height))) }
        while true {
            let fittedWidth = max(1, multiple ? Int(Double(width) * scale) : Int((Double(width) * scale).rounded(.toNearestOrEven)))
            let fittedHeight = max(1, multiple ? Int(Double(height) * scale) : Int((Double(height) * scale).rounded(.toNearestOrEven)))
            let canvasWidth = multiple ? max(16, fittedWidth / 16 * 16) : max(64, (fittedWidth + 15) / 16 * 16)
            let canvasHeight = multiple ? max(16, fittedHeight / 16 * 16) : max(64, (fittedHeight + 15) / 16 * 16)
            if canvasWidth * canvasHeight <= budget { return (fittedWidth, fittedHeight, canvasWidth, canvasHeight) }
            scale *= 0.995
        }
    }

    static func oriented(_ image: MediaImage, orientation: Int) throws -> MediaImage {
        guard (2...8).contains(orientation) else { return image }
        let swapped = orientation >= 5
        let width = swapped ? image.height : image.width, height = swapped ? image.width : image.height
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let source: (Int, Int)
                switch orientation {
                case 2: source = (image.width - 1 - x, y)
                case 3: source = (image.width - 1 - x, image.height - 1 - y)
                case 4: source = (x, image.height - 1 - y)
                case 5: source = (y, x)
                case 6: source = (y, image.height - 1 - x)
                case 7: source = (image.width - 1 - y, image.height - 1 - x)
                default: source = (image.width - 1 - y, x)
                }
                for channel in 0..<4 { bytes[(y * width + x) * 4 + channel] = image.rgba8[(source.1 * image.width + source.0) * 4 + channel] }
            }
        }
        return try MediaImage(width: width, height: height, rgba8: bytes)
    }

    static func paddedOrCropped(_ image: MediaImage, width: Int, height: Int, reflect: Bool) throws -> MediaImage {
        let left = (width - image.width) / 2, top = (height - image.height) / 2
        func coordinate(_ value: Int, count: Int) -> Int {
            let symmetric = min(image.width, image.height) <= 16
            if count == 1 { return 0 }
            let period = symmetric ? 2 * count : 2 * (count - 1)
            let wrapped = (value % period + period) % period
            return wrapped < count ? wrapped : period - wrapped - (symmetric ? 1 : 0)
        }
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let sourceX = reflect ? coordinate(x - left, count: image.width) : x + (image.width - width) / 2
                let sourceY = reflect ? coordinate(y - top, count: image.height) : y + (image.height - height) / 2
                for channel in 0..<3 { bytes[(y * width + x) * 4 + channel] = image.rgba8[(sourceY * image.width + sourceX) * 4 + channel] }
            }
        }
        return try MediaImage(width: width, height: height, rgba8: bytes)
    }
}
