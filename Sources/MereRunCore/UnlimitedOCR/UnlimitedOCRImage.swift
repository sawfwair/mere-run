#if !os(iOS)
import Foundation
import MediaIO
import MLX

enum UnlimitedOCRImage {
    struct Page {
        let global: MLXArray
        let tiles: [MLXArray]
        let columns: Int
        let rows: Int
        var tokenCount: Int { 273 + (tiles.isEmpty ? 0 : rows * 10 * (columns * 10 + 1)) }
    }

    static func tileGrid(width: Int, height: Int) -> (columns: Int, rows: Int) {
        if width <= 640 && height <= 640 { return (1, 1) }
        let aspect = Double(width) / Double(height)
        var ratios: [(Int, Int)] = []
        for x in 1...16 {
            for y in 1...16 where (2...16).contains(x * y) { ratios.append((x, y)) }
        }
        ratios.sort { left, right in
            left.0 * left.1 == right.0 * right.1 ? left.0 < right.0 : left.0 * left.1 < right.0 * right.1
        }
        var best = (1, 1)
        var distance = Double.infinity
        for ratio in ratios {
            let delta = abs(aspect - Double(ratio.0) / Double(ratio.1))
            if delta < distance || (delta == distance && Double(width * height) > 0.5 * 640 * 640 * Double(ratio.0 * ratio.1)) {
                best = ratio
                distance = delta
            }
        }
        return best
    }

    static func prepare(_ url: URL, crop: Bool) throws -> Page {
        let image = try MediaImageIO.decodeWithPillowJPEG(url)
        let global = try letterbox(image)
        let grid = crop ? tileGrid(width: image.width, height: image.height) : (columns: 1, rows: 1)
        var tiles: [MLXArray] = []
        if grid.columns * grid.rows > 1 {
            let resized = try MediaImageIO.bicubicResizedRGB(image, width: grid.columns * 640, height: grid.rows * 640)
            for row in 0..<grid.rows {
                for column in 0..<grid.columns {
                    var rgba = [UInt8](repeating: 255, count: 640 * 640 * 4)
                    for y in 0..<640 {
                        let start = ((row * 640 + y) * resized.width + column * 640) * 4
                        rgba.replaceSubrange((y * 640 * 4)..<((y + 1) * 640 * 4), with: resized.rgba8[start..<(start + 640 * 4)])
                    }
                    tiles.append(pixels(try MediaImage(width: 640, height: 640, rgba8: rgba)))
                }
            }
        }
        return Page(global: global, tiles: tiles, columns: grid.columns, rows: grid.rows)
    }

    static func letterbox(_ image: MediaImage) throws -> MLXArray {
        // Pillow ImageOps.contain rounds the constrained dimension using ties-to-even.
        let width: Int
        let height: Int
        if image.width > image.height {
            width = 1024
            height = max(1, Int((Double(image.height) * 1024 / Double(image.width)).rounded(.toNearestOrEven)))
        } else {
            height = 1024
            width = max(1, Int((Double(image.width) * 1024 / Double(image.height)).rounded(.toNearestOrEven)))
        }
        let resized = try MediaImageIO.bicubicResizedRGB(image, width: width, height: height)
        var rgba = [UInt8](repeating: 127, count: 1024 * 1024 * 4)
        let x = Int((Double(1024 - width) / 2).rounded(.toNearestOrEven))
        let y = Int((Double(1024 - height) / 2).rounded(.toNearestOrEven))
        for row in 0..<height {
            let start = ((row + y) * 1024 + x) * 4
            rgba.replaceSubrange(start..<(start + width * 4), with: resized.rgba8[(row * width * 4)..<((row + 1) * width * 4)])
        }
        return pixels(try MediaImage(width: 1024, height: 1024, rgba8: rgba))
    }

    private static func pixels(_ image: MediaImage) -> MLXArray {
        MLXArray(MediaImageIO.rgbCHWFloat(image, normalizedToMinusOneToOne: true))
            .reshaped(1, 3, image.height, image.width).transposed(0, 2, 3, 1).asType(.bfloat16)
    }
}
#endif
