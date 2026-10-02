import Foundation
import MediaIO
import MLX
import MereRunQwenModel

struct ClefPreparedMedia {
    struct Item {
        let pixels: [Float]
        let width: Int
        let height: Int
        let frameCount: Int
        let grid: (Int, Int, Int)
        let isVideo: Bool
    }
    let text: String
    let items: [Item]

    static func prepare(_ request: ClefDecisionRequest, processor: ClefProcessorConfiguration) throws -> ClefPreparedMedia {
        var text = ""
        var items: [Item] = []
        var tokenCount = 0
        func load(_ path: String) throws -> MediaImage {
            if let scheme = URL(string: path)?.scheme, scheme != "file" {
                throw ClefError.invalidInput("Clef media must use local file paths.")
            }
            let url: URL
            if path.hasPrefix("file://") {
                guard let parsed = URL(string: path), parsed.isFileURL else { throw ClefError.invalidInput("Invalid Clef file URL.") }
                url = parsed
            } else { url = URL(fileURLWithPath: path) }
            return try MediaImageIO.decode(url)
        }
        func reserve(_ grid: (Int, Int, Int)) throws {
            tokenCount += grid.0 * grid.1 * grid.2 / 4
            guard tokenCount < request.maxTokens else {
                throw ClefError.invalidInput("Clef media exceeds the context budget before schema and state.")
            }
        }
        for path in request.images {
            try Task.checkCancellation()
            let image = try load(path)
            let size = try Q35Generator.qwen3VLTargetSize(
                originalWidth: image.width, originalHeight: image.height, patchSize: 16, spatialMergeSize: 2,
                minPixels: processor.image_processor.size.shortest_edge, maxPixels: processor.image_processor.size.longest_edge)
            let grid = (1, size.height / 16, size.width / 16)
            try reserve(grid)
            let resized = try MediaImageIO.bicubicResizedRGB(image, width: size.width, height: size.height)
            items.append(Item(pixels: MediaImageIO.rescaledRGBCHWFloat(resized, rescaleFactor: Float(processor.image_processor.rescale_factor),
                                                                      normalizedToMinusOneToOne: true),
                              width: size.width, height: size.height, frameCount: 1, grid: grid, isVideo: false))
            text += "<|vision_start|>" + String(repeating: "<|image_pad|>", count: grid.1 * grid.2 / 4) + "<|vision_end|>"
        }
        for paths in request.videos {
            try Task.checkCancellation()
            let indices = sampleFrameIndices(count: paths.count)
            let first = try load(paths[indices[0]])
            let size = try videoSize(width: first.width, height: first.height, frames: indices.count,
                                     minPixels: processor.video_processor.size.shortest_edge,
                                     maxPixels: processor.video_processor.size.longest_edge)
            let paddedCount = indices.count + indices.count % 2
            let grid = (paddedCount / 2, size.height / 16, size.width / 16)
            try reserve(grid)
            var pixels: [Float] = []
            var paddedIndices = indices
            if paddedIndices.count % 2 != 0 { paddedIndices.append(indices[indices.count - 1]) }
            for index in paddedIndices {
                try Task.checkCancellation()
                let image = try load(paths[index])
                guard image.width == first.width, image.height == first.height else {
                    throw ClefError.invalidInput("Frames in each Clef video must share dimensions.")
                }
                let resized = try MediaImageIO.bicubicResizedRGB(image, width: size.width, height: size.height)
                pixels += MediaImageIO.rescaledRGBCHWFloat(resized, rescaleFactor: Float(processor.video_processor.rescale_factor),
                                                           normalizedToMinusOneToOne: true)
            }
            items.append(Item(pixels: pixels, width: size.width, height: size.height, frameCount: paddedCount, grid: grid, isVideo: true))
            text += "<|vision_start|>"
            for temporal in 0..<grid.0 {
                let timestamp = Double(paddedIndices[2 * temporal] + paddedIndices[2 * temporal + 1]) / 48
                let label = String(format: "%.1f", locale: Locale(identifier: "en_US_POSIX"), timestamp)
                text += "<\(label) seconds><|vision_start|>"
                    + String(repeating: "<|video_pad|>", count: grid.1 * grid.2 / 4) + "<|vision_end|>"
            }
            text += "<|vision_end|>"
        }
        if !items.isEmpty { text += "\n" }
        return ClefPreparedMedia(text: text, items: items)
    }

    static func sampleFrameIndices(count: Int) -> [Int] {
        let samples = min(max(Int(Double(count) / 24 * 2), 4), 768, count)
        if samples == 1 { return [0] }
        return (0..<samples).map { Int((Double($0) * Double(count - 1) / Double(samples - 1)).rounded(.toNearestOrEven)) }
    }

    static func videoSize(width: Int, height: Int, frames: Int, minPixels: Int, maxPixels: Int) throws -> (width: Int, height: Int) {
        guard Double(max(width, height)) / Double(min(width, height)) <= 200 else {
            throw ClefError.invalidInput("Clef video aspect ratio must not exceed 200.")
        }
        var width = width
        var height = height
        if min(width, height) < 32 {
            let scale = max(32 / Double(width), 32 / Double(height))
            width = Int(Double(width) * scale)
            height = Int(Double(height) * scale)
        }
        let temporal = max(2, frames + frames % 2)
        var resizedWidth = Int((Double(width) / 32).rounded(.toNearestOrEven)) * 32
        var resizedHeight = Int((Double(height) / 32).rounded(.toNearestOrEven)) * 32
        if temporal * resizedWidth * resizedHeight > maxPixels {
            let beta = sqrt(Double(frames * width * height) / Double(maxPixels))
            resizedWidth = max(32, Int(floor(Double(width) / beta / 32)) * 32)
            resizedHeight = max(32, Int(floor(Double(height) / beta / 32)) * 32)
        } else if temporal * resizedWidth * resizedHeight < minPixels {
            let beta = sqrt(Double(minPixels) / Double(frames * width * height))
            resizedWidth = Int(ceil(Double(width) * beta / 32)) * 32
            resizedHeight = Int(ceil(Double(height) * beta / 32)) * 32
        }
        return (resizedWidth, resizedHeight)
    }

    func replacements(tower: Q35VisionTower) throws -> [Q35VisionReplacement] {
        try items.flatMap { item in
            try Task.checkCancellation()
            let pixels = MLXArray(item.pixels, [item.frameCount, 3, item.height, item.width])
            if item.isVideo {
                let embeddings = try tower.encodeVideoFrames(pixels, gridTHW: item.grid)
                let perFrame = item.grid.1 * item.grid.2 / 4
                return (0..<item.grid.0).map { temporal in
                    Q35VisionReplacement(embeddings: embeddings[(temporal * perFrame)..<((temporal + 1) * perFrame)],
                                         gridTHW: (1, item.grid.1, item.grid.2))
                }
            }
            return [Q35VisionReplacement(embeddings: try tower.encodeImage(pixelValues: pixels, gridTHW: item.grid), gridTHW: item.grid)]
        }
    }
}
