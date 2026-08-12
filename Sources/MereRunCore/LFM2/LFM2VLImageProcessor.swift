import Foundation
import MediaIO
import MLX

struct LFM2VLImageGrid: Hashable, Sendable {
    let rows: Int
    let columns: Int

    var patchCount: Int { rows * columns }

    func visualTokenCount(downsampleFactor: Int) -> Int {
        let factor = max(1, downsampleFactor)
        let paddedRows = rows + ((factor - (rows % factor)) % factor)
        let paddedColumns = columns + ((factor - (columns % factor)) % factor)
        return (paddedRows / factor) * (paddedColumns / factor)
    }
}

struct LFM2VLImageBatch {
    let pixelValues: MLXArray
    let grids: [LFM2VLImageGrid]
}

enum LFM2VLImageProcessor {
    static func makeBatch(
        imageReferences: [String],
        config: LFM2VLConfig,
        processorConfig: LFM2VLProcessorConfig?
    ) throws -> LFM2VLImageBatch {
        guard !imageReferences.isEmpty else {
            throw LFM2Error.generationFailed("LFM2-VL image batch is empty.")
        }

        let patchSize = processorConfig?.imageProcessor.encoderPatchSize ?? config.encoderPatchSize
        let downsample = processorConfig?.imageProcessor.downsampleFactor ?? config.downsampleFactor
        let minTokens = processorConfig?.imageProcessor.minImageTokens ?? config.minImageTokens
        let maxTokens = processorConfig?.imageProcessor.maxImageTokens ?? config.maxImageTokens
        let maxPatches = processorConfig?.imageProcessor.maxNumPatches ?? config.maxNumPatches
        let patchDimension = patchSize * patchSize * config.visionConfig.numChannels
        var packedImages: [[Float]] = []
        var grids: [LFM2VLImageGrid] = []

        for reference in imageReferences {
            let image = try decodeImage(reference)
            let size = smartResize(
                width: image.width,
                height: image.height,
                encoderPatchSize: patchSize,
                downsampleFactor: downsample,
                minImageTokens: minTokens,
                maxImageTokens: maxTokens
            )
            let resized = try MediaImageIO.resized(image, width: size.width, height: size.height)
            let chw = MediaImageIO.rgbCHWFloat(resized, normalizedToMinusOneToOne: true)
            let rows = size.height / patchSize
            let columns = size.width / patchSize
            let grid = LFM2VLImageGrid(rows: rows, columns: columns)
            guard grid.patchCount <= maxPatches else {
                throw LFM2Error.generationFailed(
                    "LFM2-VL image produced \(grid.patchCount) patches; maximum is \(maxPatches)."
                )
            }

            var patches = [Float](repeating: 0, count: maxPatches * patchDimension)
            let channelPlane = size.width * size.height
            var patchIndex = 0
            for patchRow in 0..<rows {
                for patchColumn in 0..<columns {
                    var outputIndex = patchIndex * patchDimension
                    for row in 0..<patchSize {
                        let sourceRow = (patchRow * patchSize) + row
                        for column in 0..<patchSize {
                            let sourceColumn = (patchColumn * patchSize) + column
                            let pixelIndex = (sourceRow * size.width) + sourceColumn
                            for channel in 0..<config.visionConfig.numChannels {
                                patches[outputIndex] = chw[(channel * channelPlane) + pixelIndex]
                                outputIndex += 1
                            }
                        }
                    }
                    patchIndex += 1
                }
            }
            packedImages.append(patches)
            grids.append(grid)
        }

        return LFM2VLImageBatch(
            pixelValues: MLXArray(
                packedImages.flatMap { $0 },
                [imageReferences.count, maxPatches, patchDimension]
            ),
            grids: grids
        )
    }

    static func expandedPromptTokens(
        _ tokens: [Int],
        grids: [LFM2VLImageGrid],
        downsampleFactor: Int,
        imageTokenId: Int,
        imageStartTokenId: Int,
        imageEndTokenId: Int
    ) throws -> [Int] {
        let placeholderCount = tokens.filter { $0 == imageTokenId }.count
        guard placeholderCount == grids.count else {
            throw LFM2Error.generationFailed(
                "LFM2-VL prompt has \(placeholderCount) image placeholders for \(grids.count) image(s)."
            )
        }

        var gridIndex = 0
        var expanded: [Int] = []
        for token in tokens {
            guard token == imageTokenId else {
                expanded.append(token)
                continue
            }
            let count = grids[gridIndex].visualTokenCount(downsampleFactor: downsampleFactor)
            expanded.append(imageStartTokenId)
            expanded.append(contentsOf: repeatElement(imageTokenId, count: count))
            expanded.append(imageEndTokenId)
            gridIndex += 1
        }
        return expanded
    }

    static func smartResize(
        width: Int,
        height: Int,
        encoderPatchSize: Int,
        downsampleFactor: Int,
        minImageTokens: Int,
        maxImageTokens: Int
    ) -> (width: Int, height: Int) {
        let factor = max(1, encoderPatchSize * downsampleFactor)
        let minPixels = minImageTokens * factor * factor
        let maxPixels = maxImageTokens * factor * factor
        var targetHeight = max(factor, Int((Double(height) / Double(factor)).rounded()) * factor)
        var targetWidth = max(factor, Int((Double(width) / Double(factor)).rounded()) * factor)

        if targetHeight * targetWidth > maxPixels {
            let scale = sqrt(Double(max(1, height * width)) / Double(maxPixels))
            targetHeight = max(factor, Int(floor(Double(height) / scale / Double(factor))) * factor)
            targetWidth = max(factor, Int(floor(Double(width) / scale / Double(factor))) * factor)
        } else if targetHeight * targetWidth < minPixels {
            let scale = sqrt(Double(minPixels) / Double(max(1, height * width)))
            targetHeight = Int(ceil(Double(height) * scale / Double(factor))) * factor
            targetWidth = Int(ceil(Double(width) * scale / Double(factor))) * factor
        }
        return (targetWidth, targetHeight)
    }

    private static func decodeImage(_ reference: String) throws -> MediaImage {
        let trimmed = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw LFM2Error.generationFailed("Image reference is empty.")
        }
        if trimmed.lowercased().hasPrefix("data:") {
            guard let comma = trimmed.firstIndex(of: ","),
                  trimmed[..<comma].lowercased().contains(";base64"),
                  let data = Data(
                      base64Encoded: String(trimmed[trimmed.index(after: comma)...]),
                      options: [.ignoreUnknownCharacters]
                  ) else {
                throw LFM2Error.generationFailed("Invalid base64 image data URL.")
            }
            return try MediaImageIO.decode(data: data)
        }
        if let url = URL(string: trimmed), url.scheme?.lowercased() == "file" {
            return try MediaImageIO.decode(url)
        }
        if let scheme = URL(string: trimmed)?.scheme?.lowercased(),
           scheme == "http" || scheme == "https" {
            throw LFM2Error.generationFailed(
                "Remote image URLs are not fetched by the local LFM2-VL runtime; use a file path or data URL."
            )
        }
        return try MediaImageIO.decode(URL(fileURLWithPath: trimmed).standardizedFileURL)
    }
}
