#if !os(iOS)
// Native Swift/MLX reimplementation of the pinned LiquidAI D1 references.
// Modified for mere.run; see THIRD_PARTY_NOTICES.md and licenses/d1-LFM-OPEN-LICENSE.txt.
import Foundation
import MediaIO
import MereRunD1Model

struct D1PreparedMedia {
    let patches: [[Float]]
    let grids: [D1VLImageGrid]
    let markup: String
    let samples: [Float]?
    var tokens: Int {
        if let samples { return ((max(8_000, samples.count) / 160 + 7) / 8) }
        return grids.reduce(0) { $0 + $1.patchCount / 4 }
    }
    static func prepare(_ request: D1DecisionRequest, config: D1Configuration) throws -> D1PreparedMedia {
        if let audio = request.audio {
            guard config.isOmni else { throw D1Error.invalid("D1-3B accepts text and images; audio requires D1 omni.") }
            let url = try localURL(audio)
            let buffer = try MediaAudioIO.decodeSegment(url, startTime: 0, duration: 30, targetSampleRate: 16_000, channels: 1)
            guard !buffer.samples.isEmpty, buffer.samples.allSatisfy(\.isFinite) else { throw D1Error.invalid("D1 audio is empty or nonfinite.") }
            return D1PreparedMedia(patches: [], grids: [], markup: "", samples: Array(buffer.samples.prefix(480_000)))
        }
        var patches: [[Float]] = [], grids: [D1VLImageGrid] = [], markup = ""
        for path in request.images {
            var image = try MediaImageIO.decode(localURL(path))
            if !config.isOmni, image.width * image.height > 1_024 * 1_024 {
                let scale = sqrt(Double(1_024 * 1_024) / Double(image.width * image.height))
                image = try MediaImageIO.bicubicResizedRGB(image, width: max(1, Int(Double(image.width) * scale)), height: max(1, Int(Double(image.height) * scale)))
            }
            let plan = try layout(width: image.width, height: image.height)
            var crops: [MediaImage] = []
            if plan.tiled {
                let big = try resized(image, width: plan.columns * 512, height: plan.rows * 512, bicubic: !config.isOmni)
                for row in 0..<plan.rows { for column in 0..<plan.columns {
                    var rgba: [UInt8] = []
                    for y in 0..<512 {
                        let start = ((row * 512 + y) * big.width + column * 512) * 4
                        rgba.append(contentsOf: big.rgba8[start..<(start + 512 * 4)])
                    }
                    crops.append(try MediaImage(width: 512, height: 512, rgba8: rgba))
                } }
            }
            crops.append(try resized(image, width: plan.width, height: plan.height, bicubic: !config.isOmni))
            if !config.isOmni { markup += "<|image_start|>" }
            for (index, crop) in crops.enumerated() {
                let grid = D1VLImageGrid(rows: crop.height / 16, columns: crop.width / 16)
                var packed: [Float] = []
                packed.reserveCapacity(grid.patchCount * 768)
                for row in 0..<grid.rows { for column in 0..<grid.columns {
                    for y in 0..<16 { for x in 0..<16 { for channel in 0..<3 {
                        let byte = crop.rgba8[((row * 16 + y) * crop.width + column * 16 + x) * 4 + channel]
                        packed.append((Float(byte) - 127.5) / 127.5)
                    } } }
                } }
                patches.append(packed); grids.append(grid)
                if !config.isOmni {
                    if plan.tiled { markup += index == crops.count - 1 ? "<|img_thumbnail|>" : "<|img_row_\(index / plan.columns + 1)_col_\(index % plan.columns + 1)|>" }
                    markup += String(repeating: "<image>", count: grid.patchCount / 4)
                }
            }
            if !config.isOmni { markup += "<|image_end|>" }
        }
        return D1PreparedMedia(patches: patches, grids: grids, markup: markup, samples: nil)
    }
    static func localURL(_ path: String) throws -> URL {
        guard !path.contains("://"), FileManager.default.fileExists(atPath: path) else { throw D1Error.invalid("D1 media must be existing local files: \(path)") }
        return URL(fileURLWithPath: path)
    }
    struct Layout { let width: Int; let height: Int; let rows: Int; let columns: Int; let tiled: Bool }
    static func layout(width: Int, height: Int) throws -> Layout {
        let area = Double(width * height), maximum = 262_144.0, minimum = 65_536.0
        func rounded(_ value: Int) -> Int { max(32, Int((Double(value) / 32).rounded(.toNearestOrEven)) * 32) }
        var w = rounded(width), h = rounded(height)
        if Double(w * h) > maximum {
            let scale = sqrt(area / maximum)
            w = max(32, Int(floor(Double(width) / scale / 32)) * 32)
            h = max(32, Int(floor(Double(height) / scale / 32)) * 32)
        } else if Double(w * h) < minimum {
            let scale = sqrt(minimum / area)
            w = Int(ceil(Double(width) * scale / 32)) * 32; h = Int(ceil(Double(height) * scale / 32)) * 32
        }
        guard w * h <= 262_144 else { throw D1Error.invalid("D1 image aspect ratio exceeds the checkpoint’s 1,024-patch crop budget.") }
        let tiled = max(16, Int((Double(width) / 32).rounded(.toNearestOrEven)) * 32)
            * max(16, Int((Double(height) / 32).rounded(.toNearestOrEven)) * 32) > 524_288
        var columns = 1, rows = 1, best = Double.infinity
        if tiled {
            var ratios: [(Int, Int)] = []
            for x in 1...10 { for y in 1...10 where (2...10).contains(x * y) { ratios.append((x, y)) } }
            ratios.sort { lhs, rhs in
                let leftArea = lhs.0 * lhs.1, rightArea = rhs.0 * rhs.1
                return leftArea == rightArea ? lhs.0 < rhs.0 : leftArea < rightArea
            }
            for (x, y) in ratios {
                let difference = abs(Double(width) / Double(height) - Double(x) / Double(y))
                if difference < best || (difference == best && area > 0.5 * 512 * 512 * Double(x * y)) { columns = x; rows = y; best = difference }
            }
        }
        return Layout(width: w, height: h, rows: rows, columns: columns, tiled: tiled)
    }
    /// Released torchvision uint8 antialiasing: bilinear for omni, bicubic for D1-3B.
    /// Match the adaptive int16 coefficient precision and intermediate uint8 rounding.
    static func resized(_ image: MediaImage, width: Int, height: Int, bicubic: Bool = false) throws -> MediaImage {
        if image.width == width, image.height == height { return image }
        func kernel(_ distance: Double) -> Double {
            let x = abs(distance)
            if !bicubic { return max(0, 1 - x) }
            if x < 1 { return ((1.5 * x - 2.5) * x * x) + 1 }
            if x < 2 { return (((x - 5) * x + 8) * x - 4) * -0.5 }
            return 0
        }
        func coefficients(source: Int, target: Int) -> (values: [[(Int, Int)]], precision: Int) {
            let scale = Double(source) / Double(target), filterScale = max(1, scale)
            let support = (bicubic ? 2.0 : 1.0) * filterScale
            let rows = (0..<target).map { index -> [(Int, Double)] in
                let center = (Double(index) + 0.5) * scale
                let start = max(0, Int(center - support + 0.5)), end = min(source, Int(center + support + 0.5))
                let values = (start..<end).map { ($0, kernel((Double($0) + 0.5 - center) / filterScale)) }
                let sum = values.reduce(0) { $0 + $1.1 }
                return values.map { ($0.0, $0.1 / sum) }
            }
            let maximum = rows.flatMap { $0.map(\.1) }.max() ?? 1
            var precision = 0
            while precision < 22, Int(0.5 + maximum * Double(1 << (precision + 1))) < (1 << 15) { precision += 1 }
            let unit = Double(1 << precision)
            return (rows.map { row in row.map { ($0.0, Int($0.1 * unit + ($0.1 < 0 ? -0.5 : 0.5))) } }, precision)
        }
        let horizontal = coefficients(source: image.width, target: width), vertical = coefficients(source: image.height, target: height)
        var intermediate = [UInt8](repeating: 0, count: image.height * width * 3)
        for y in 0..<image.height { for x in 0..<width { for channel in 0..<3 {
            let total = horizontal.values[x].reduce(1 << (horizontal.precision - 1)) {
                $0 + Int(image.rgba8[(y * image.width + $1.0) * 4 + channel]) * $1.1
            }
            intermediate[(y * width + x) * 3 + channel] = UInt8(clamping: total >> horizontal.precision)
        } } }
        var rgba = [UInt8](repeating: 255, count: height * width * 4)
        for y in 0..<height { for x in 0..<width { for channel in 0..<3 {
            let total = vertical.values[y].reduce(1 << (vertical.precision - 1)) {
                $0 + Int(intermediate[($1.0 * width + x) * 3 + channel]) * $1.1
            }
            rgba[(y * width + x) * 4 + channel] = UInt8(clamping: total >> vertical.precision)
        } } }
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }
}
#endif
