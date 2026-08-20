import Foundation
import MediaIO
import MLX

struct NemotronOmniPreparedImage: @unchecked Sendable {
    let pixelValues: MLXArray
    let sourcePatchCount: Int

    var languageTokenCount: Int {
        NemotronOmniPlaceholderPlanner.imageTokenCount(sourcePatchCount: sourcePatchCount)
    }
}

enum NemotronOmniImageProcessor {
    static func prepare(
        reference: String,
        config: NemotronOmniPreprocessorConfig,
        contextLength: Int,
        videoMode: Bool = false
    ) throws -> NemotronOmniPreparedImage {
        let image = try decode(reference: reference)
        let patchGrid: (width: Int, height: Int)
        if videoMode {
            patchGrid = videoPatchGrid(width: image.width, height: image.height)
        } else {
            let available = max(1, contextLength - 4) * 4
            let budget = max(
                config.minNumPatches,
                min(config.maxNumPatches, available)
            )
            patchGrid = imagePatchGrid(
                width: image.width,
                height: image.height,
                patchSize: config.patchSize,
                minimumPatches: config.minNumPatches,
                budget: budget
            )
        }

        let targetWidth = patchGrid.width * config.patchSize
        let targetHeight = patchGrid.height * config.patchSize
        let pixels = bicubicResizeRGB(
            image,
            targetWidth: targetWidth,
            targetHeight: targetHeight
        )
        var normalized = pixels
        let pixelCount = targetWidth * targetHeight
        for pixel in 0..<pixelCount {
            for channel in 0..<3 {
                let index = pixel * 3 + channel
                normalized[index] = (
                    normalized[index] - config.normalizationMean[channel]
                ) / config.normalizationStandardDeviation[channel]
            }
        }
        return NemotronOmniPreparedImage(
            pixelValues: MLXArray(
                normalized,
                [1, targetHeight, targetWidth, 3]
            ).asType(.bfloat16),
            sourcePatchCount: patchGrid.width * patchGrid.height
        )
    }

    static func imagePatchGrid(
        width: Int,
        height: Int,
        patchSize: Int = 16,
        minimumPatches: Int = 1_024,
        budget: Int = 13_312
    ) -> (width: Int, height: Int) {
        precondition(width > 0 && height > 0 && patchSize > 0 && budget > 0)
        let closestHeight = (height + patchSize - 1) / patchSize
        let closestWidth = (width + patchSize - 1) / patchSize
        let patchCount = max(1, closestHeight * closestWidth)
        let factor = min(sqrt(Double(budget) / Double(patchCount)), 1)
        var targetHeight = max(1, Int(floor(factor * Double(closestHeight))))
        var targetWidth = max(1, Int(floor(factor * Double(closestWidth))))

        if budget > minimumPatches, targetHeight * targetWidth < minimumPatches {
            let upscale = sqrt(
                Double(minimumPatches) / Double(max(1, targetHeight * targetWidth))
            )
            targetHeight = max(1, Int(ceil(upscale * Double(targetHeight))))
            targetWidth = max(1, Int(ceil(upscale * Double(targetWidth))))
        }

        targetHeight = roundedPatchDimension(
            targetHeight,
            other: targetWidth,
            budget: budget
        )
        targetWidth = roundedPatchDimension(
            targetWidth,
            other: targetHeight,
            budget: budget
        )
        return (targetWidth, targetHeight)
    }

    static func videoPatchGrid(
        width: Int,
        height: Int,
        targetPatches: Int = 1_024
    ) -> (width: Int, height: Int) {
        precondition(width > 0 && height > 0)
        let aspect = Double(width) / Double(height)
        var patchHeight = max(1, Int(sqrt(Double(targetPatches) / aspect).rounded()))
        var patchWidth = max(1, Int(sqrt(Double(targetPatches) * aspect).rounded()))
        let heightUp = patchHeight + (patchHeight.isMultiple(of: 2) ? 0 : 1)
        let widthUp = patchWidth + (patchWidth.isMultiple(of: 2) ? 0 : 1)
        if heightUp * widthUp <= targetPatches {
            patchHeight = heightUp
            patchWidth = widthUp
        } else {
            patchHeight = max(2, patchHeight - patchHeight % 2)
            patchWidth = max(2, patchWidth - patchWidth % 2)
        }
        return (patchWidth, patchHeight)
    }

    private static func roundedPatchDimension(
        _ value: Int,
        other: Int,
        budget: Int
    ) -> Int {
        let remainder = value % 2
        guard remainder != 0 else { return value }
        let increased = value + 1
        if increased * other <= budget {
            return increased
        }
        return max(2, value - remainder)
    }

    private static func decode(reference: String) throws -> MediaImage {
        let trimmed = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw NemotronOmniError.unsupportedMedia("image reference is empty")
        }
        if trimmed.lowercased().hasPrefix("data:") {
            guard let comma = trimmed.firstIndex(of: ","),
                  trimmed[..<comma].lowercased().contains(";base64"),
                  let data = Data(
                    base64Encoded: String(trimmed[trimmed.index(after: comma)...]),
                    options: [.ignoreUnknownCharacters]
                  ) else {
                throw NemotronOmniError.unsupportedMedia("invalid base64 image data URL")
            }
            return try MediaImageIO.decode(data: data)
        }
        if let parsed = URL(string: trimmed), parsed.scheme?.lowercased() == "file" {
            return try MediaImageIO.decode(parsed)
        }
        if let scheme = URL(string: trimmed)?.scheme?.lowercased(),
           scheme == "http" || scheme == "https" {
            let data = try Data(contentsOf: URL(string: trimmed)!)
            return try MediaImageIO.decode(data: data)
        }
        return try MediaImageIO.decode(URL(fileURLWithPath: trimmed).standardizedFileURL)
    }

    /// Separable Keys-cubic resize with antialiasing on reduction. The geometry
    /// matches PyTorch's align_corners=false media preprocessing contract.
    private static func bicubicResizeRGB(
        _ image: MediaImage,
        targetWidth: Int,
        targetHeight: Int
    ) -> [Float] {
        let horizontal = cubicWeights(source: image.width, target: targetWidth)
        var intermediate = [Float](
            repeating: 0,
            count: image.height * targetWidth * 3
        )
        for y in 0..<image.height {
            for targetX in 0..<targetWidth {
                let destination = (y * targetWidth + targetX) * 3
                for (sourceX, weight) in horizontal[targetX] {
                    let source = (y * image.width + sourceX) * 4
                    intermediate[destination] += Float(image.rgba8[source]) / 255 * weight
                    intermediate[destination + 1] += Float(image.rgba8[source + 1]) / 255 * weight
                    intermediate[destination + 2] += Float(image.rgba8[source + 2]) / 255 * weight
                }
            }
        }

        let vertical = cubicWeights(source: image.height, target: targetHeight)
        var output = [Float](repeating: 0, count: targetHeight * targetWidth * 3)
        for targetY in 0..<targetHeight {
            for (sourceY, weight) in vertical[targetY] {
                for x in 0..<targetWidth {
                    let source = (sourceY * targetWidth + x) * 3
                    let destination = (targetY * targetWidth + x) * 3
                    output[destination] += intermediate[source] * weight
                    output[destination + 1] += intermediate[source + 1] * weight
                    output[destination + 2] += intermediate[source + 2] * weight
                }
            }
        }
        return output
    }

    private static func cubicWeights(
        source: Int,
        target: Int
    ) -> [[(index: Int, weight: Float)]] {
        let scale = Float(source) / Float(target)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        return (0..<target).map { destination in
            let center = scale * (Float(destination) + 0.5)
            let first = max(0, Int(ceil(center - support - 0.5)))
            let last = min(source - 1, Int(floor(center + support - 0.5)))
            var values: [(Int, Float)] = []
            var total: Float = 0
            if first <= last {
                for index in first...last {
                    let distance = (Float(index) + 0.5 - center) / filterScale
                    let weight = cubicKernel(distance)
                    if weight != 0 {
                        values.append((index, weight))
                        total += weight
                    }
                }
            }
            return total == 0 ? values : values.map { ($0.0, $0.1 / total) }
        }
    }

    private static func cubicKernel(_ value: Float) -> Float {
        let distance = abs(value)
        let coefficient: Float = -0.5
        if distance < 1 {
            return ((coefficient + 2) * distance - (coefficient + 3))
                * distance * distance + 1
        }
        if distance < 2 {
            return (((distance - 5) * distance + 8) * distance - 4) * coefficient
        }
        return 0
    }
}
