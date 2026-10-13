#if os(macOS) && arch(arm64)
import Foundation

public struct CoreMLVisionInputs: Sendable {
    public let patches: [Float16]
    public let positions: [Int32]
    public let positionEmbeddings: [Float16]
    public let valid: [Float16]
    public let pool: [Float16]
    public let tokenCount: Int
    public let capacity: Int
}

public enum ImagePreprocessor {
    public static func targetSize(height: Int, width: Int, tokens: Int = 280) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0, [70, 140, 280].contains(tokens) else { throw EmbedderError(description: "Invalid image dimensions or vision token budget.") }
        let factor = sqrt(Double(tokens * 9 * 256) / Double(height * width))
        var h = Int(floor(Double(height) * factor / 48)) * 48
        var w = Int(floor(Double(width) * factor / 48)) * 48
        if h == 0 { h = 48; w = min(Int(floor(Double(width) / Double(height))) * 48, tokens * 48) }
        if w == 0 { w = 48; h = min(Int(floor(Double(height) / Double(width))) * 48, tokens * 48) }
        guard h > 0, w > 0, h * w <= tokens * 9 * 256 else { throw EmbedderError(description: "Image exceeds the vision patch budget.") }
        return (h, w)
    }

    public static func prepare(url: URL, positionTable: Data, tokens: Int = 280) throws -> CoreMLVisionInputs {
        let image = try ImageDecoder.decodeRGB(url: url, applyEXIFOrientation: false)
        let size = try targetSize(height: image.height, width: image.width, tokens: tokens)
        let rgb = ImageResampler.resize(image.rgb, width: image.width, height: image.height, channels: 3, toWidth: size.width, toHeight: size.height)
        return try patchify(rgb: rgb, width: size.width, height: size.height, positionTable: positionTable, tokens: tokens)
    }

    public static func patchify(rgb: [UInt8], width: Int, height: Int, positionTable: Data, tokens: Int = 280) throws -> CoreMLVisionInputs {
        guard [70, 140, 280].contains(tokens), width > 0, height > 0, width % 48 == 0, height % 48 == 0,
              rgb.count == width * height * 3, width * height / 256 <= tokens * 9,
              positionTable.count == 2 * 1024 * 768 * 2 else { throw EmbedderError(description: "Invalid Core ML vision patch inputs.") }
        let n = tokens * 9, rows = height / 16, columns = width / 16
        var patches = [Float16](repeating: 0, count: n * 768)
        var positions = [Int32](repeating: -1, count: n * 2)
        var embeds = [Float16](repeating: 0, count: n * 768)
        var valid = [Float16](repeating: 0, count: n)
        var pool = [Float16](repeating: 0, count: tokens * n)
        try positionTable.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: UInt16.self)
            for y in 0..<rows { for x in 0..<columns {
                guard x < 1024, y < 1024 else { throw EmbedderError(description: "Vision position exceeds the published position table.") }
                let index = y * columns + x
                positions[index * 2] = Int32(x); positions[index * 2 + 1] = Int32(y); valid[index] = 1
                let group = (y / 3) * (columns / 3) + x / 3
                pool[group * n + index] = Float16(1.0 / 9.0)
                for c in 0..<768 {
                    embeds[index * 768 + c] = Float16(bitPattern: UInt16(littleEndian: table[x * 768 + c]))
                        + Float16(bitPattern: UInt16(littleEndian: table[(1024 + y) * 768 + c]))
                }
                for py in 0..<16 { for px in 0..<16 { for c in 0..<3 {
                    let source = ((y * 16 + py) * width + x * 16 + px) * 3 + c
                    patches[index * 768 + (py * 16 + px) * 3 + c] = Float16(Float(rgb[source]) / 255)
                } } }
            } }
        }
        return CoreMLVisionInputs(patches: patches, positions: positions, positionEmbeddings: embeds, valid: valid,
                                  pool: pool, tokenCount: rows * columns / 9, capacity: tokens)
    }
}
#endif
