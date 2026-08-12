import Foundation
import MediaIO
import MLX

struct MuseGlimmerImageBatch {
    let pixelValues: MLXArray
    let grids: [(temporal: Int, height: Int, width: Int)]
    let tokenCounts: [Int]
}

enum MuseGlimmerImageProcessor {
    static func makeBatch(
        imageReferences: [String],
        config: MuseGlimmerVisionConfig,
        maxImageTokens: Int = 4_096
    ) throws -> MuseGlimmerImageBatch {
        guard !imageReferences.isEmpty else {
            throw MuseGlimmerError.unsupportedConfiguration("Muse Glimmer image batch is empty.")
        }
        let patchDimension = config.patchTemporal * 3 * config.patchSize * config.patchSize
        var flattened: [Float] = []
        var grids: [(Int, Int, Int)] = []
        var tokenCounts: [Int] = []

        for reference in imageReferences {
            let image = try decodeImage(reference)
            let target = targetSize(
                originalWidth: image.width,
                originalHeight: image.height,
                patchSize: config.patchSize,
                mergeSize: config.mergeSize,
                maxImageTokens: maxImageTokens
            )
            // The published processor resizes uint8 RGB with Lanczos-3 before
            // normalization. Keep this model-specific so other MediaIO callers
            // retain their existing interpolation contract.
            let resized = try resizedLanczos(image, width: target.width, height: target.height)
            let chw = MediaImageIO.rgbCHWFloat(resized, normalizedToMinusOneToOne: true)
            let gridHeight = resized.height / config.patchSize
            let gridWidth = resized.width / config.patchSize
            appendStaticImagePatches(
                chw: chw,
                width: resized.width,
                height: resized.height,
                patchSize: config.patchSize,
                patchTemporal: config.patchTemporal,
                to: &flattened
            )
            grids.append((1, gridHeight, gridWidth))
            tokenCounts.append((gridHeight * gridWidth) / (config.mergeSize * config.mergeSize))
        }

        return MuseGlimmerImageBatch(
            pixelValues: MLXArray(flattened, [flattened.count / patchDimension, patchDimension]),
            grids: grids,
            tokenCounts: tokenCounts
        )
    }

    static func expandedPromptTokens(
        _ tokens: [Int],
        tokenCounts: [Int],
        imageTokenId: Int,
        imageStartTokenId: Int,
        imageEndTokenId: Int
    ) throws -> [Int] {
        guard !tokenCounts.isEmpty else { return tokens }
        let occurrences = tokens.filter { $0 == imageTokenId }.count
        if occurrences == tokenCounts.reduce(0, +) {
            return tokens
        }
        guard occurrences == tokenCounts.count else {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Muse Glimmer prompt has \(occurrences) image placeholders for \(tokenCounts.count) image(s)."
            )
        }
        var imageIndex = 0
        var expanded: [Int] = []
        for token in tokens {
            guard token == imageTokenId else {
                expanded.append(token)
                continue
            }
            expanded.append(imageStartTokenId)
            expanded.append(contentsOf: repeatElement(imageTokenId, count: tokenCounts[imageIndex]))
            expanded.append(imageEndTokenId)
            imageIndex += 1
        }
        return expanded
    }

    static func targetSize(
        originalWidth: Int,
        originalHeight: Int,
        patchSize: Int,
        mergeSize: Int,
        maxImageTokens: Int
    ) -> (width: Int, height: Int) {
        let factor = max(1, patchSize * mergeSize)
        let width = max(1, originalWidth)
        let height = max(1, originalHeight)
        var idealPatchHeight = Double(height) / Double(factor)
        var idealPatchWidth = Double(width) / Double(factor)
        let ratio = idealPatchWidth / idealPatchHeight
        if idealPatchHeight * idealPatchWidth > Double(maxImageTokens) {
            idealPatchHeight = sqrt(Double(maxImageTokens) / ratio)
            idealPatchWidth = idealPatchHeight * ratio
        }

        let heightCandidates = integerCandidates(idealPatchHeight)
        let widthCandidates = integerCandidates(idealPatchWidth)
        var best = (
            width: 1,
            height: 1,
            ratioError: Double.greatestFiniteMagnitude,
            sizeError: Double.greatestFiniteMagnitude
        )
        for candidateHeight in heightCandidates {
            for candidateWidth in widthCandidates {
                guard candidateHeight * candidateWidth <= maxImageTokens else { continue }
                let ratioError = abs(
                    Double(candidateHeight) / Double(candidateWidth)
                        - Double(height) / Double(width)
                )
                let sizeError = abs(Double(candidateHeight) - idealPatchHeight)
                    + abs(Double(candidateWidth) - idealPatchWidth)
                if ratioError < best.ratioError
                    || (ratioError == best.ratioError && sizeError < best.sizeError) {
                    best = (candidateWidth, candidateHeight, ratioError, sizeError)
                }
            }
        }
        return (best.width * factor, best.height * factor)
    }

    private static func integerCandidates(_ value: Double) -> [Int] {
        let lower = max(1, Int(floor(value)))
        let upper = max(1, Int(ceil(value)))
        return lower == upper ? [lower] : [lower, upper]
    }

    private static func decodeImage(_ reference: String) throws -> MediaImage {
        let trimmed = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw MuseGlimmerError.unsupportedConfiguration("Muse Glimmer image reference is empty.")
        }
        if trimmed.lowercased().hasPrefix("data:") {
            guard let comma = trimmed.firstIndex(of: ","),
                  trimmed[..<comma].lowercased().contains(";base64"),
                  let data = Data(base64Encoded: String(trimmed[trimmed.index(after: comma)...]), options: .ignoreUnknownCharacters) else {
                throw MuseGlimmerError.unsupportedConfiguration("Invalid base64 Muse Glimmer image data URL.")
            }
            return try MediaImageIO.decode(data: data)
        }
        if let url = URL(string: trimmed), url.scheme?.lowercased() == "file" {
            return try MediaImageIO.decode(url)
        }
        if let scheme = URL(string: trimmed)?.scheme?.lowercased(), scheme == "http" || scheme == "https" {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Remote image URLs are not fetched by the local Muse Glimmer runtime; use a file path or data URL."
            )
        }
        return try MediaImageIO.decode(URL(fileURLWithPath: trimmed).standardizedFileURL)
    }

    private struct LanczosContribution {
        let start: Int
        let coefficients: [Int32]
    }

    static func resizedLanczos(
        _ image: MediaImage,
        width: Int,
        height: Int
    ) throws -> MediaImage {
        guard width > 0, height > 0 else {
            throw MediaIOError.invalidImageDimensions(width: width, height: height)
        }
        if image.width == width, image.height == height {
            return image
        }

        let precisionBits = 22
        let roundingOffset = Int64(1 << (precisionBits - 1))
        let horizontal = lanczosContributions(
            sourceSize: image.width,
            targetSize: width,
            precisionBits: precisionBits
        )
        let vertical = lanczosContributions(
            sourceSize: image.height,
            targetSize: height,
            precisionBits: precisionBits
        )

        // Match Pillow's uint8 path: round and clamp after each separable pass.
        // Muse consumes RGB, so alpha is deliberately regenerated as opaque.
        var intermediate = [UInt8](repeating: 0, count: image.height * width * 3)
        for sourceY in 0..<image.height {
            for targetX in 0..<width {
                let contribution = horizontal[targetX]
                for channel in 0..<3 {
                    var sum = roundingOffset
                    for (offset, coefficient) in contribution.coefficients.enumerated() {
                        let sourceX = contribution.start + offset
                        let source = (sourceY * image.width + sourceX) * 4 + channel
                        sum += Int64(image.rgba8[source]) * Int64(coefficient)
                    }
                    intermediate[(sourceY * width + targetX) * 3 + channel] = clippedUInt8(
                        sum,
                        precisionBits: precisionBits
                    )
                }
            }
        }

        var output = [UInt8](repeating: 255, count: width * height * 4)
        for targetY in 0..<height {
            let contribution = vertical[targetY]
            for targetX in 0..<width {
                for channel in 0..<3 {
                    var sum = roundingOffset
                    for (offset, coefficient) in contribution.coefficients.enumerated() {
                        let sourceY = contribution.start + offset
                        let source = (sourceY * width + targetX) * 3 + channel
                        sum += Int64(intermediate[source]) * Int64(coefficient)
                    }
                    output[(targetY * width + targetX) * 4 + channel] = clippedUInt8(
                        sum,
                        precisionBits: precisionBits
                    )
                }
            }
        }
        return try MediaImage(width: width, height: height, rgba8: output)
    }

    private static func lanczosContributions(
        sourceSize: Int,
        targetSize: Int,
        precisionBits: Int
    ) -> [LanczosContribution] {
        let scale = Double(sourceSize) / Double(targetSize)
        let filterScale = max(1, scale)
        let support = 3 * filterScale
        let fixedScale = Double(1 << precisionBits)
        return (0..<targetSize).map { destination in
            let center = (Double(destination) + 0.5) * scale
            let first = max(0, Int(center - support + 0.5))
            let last = min(sourceSize, Int(center + support + 0.5))
            var floating: [Double] = []
            floating.reserveCapacity(max(1, last - first))
            var total = 0.0
            for index in first..<last {
                let distance = (Double(index) - center + 0.5) / filterScale
                let value = lanczos(distance)
                floating.append(value)
                total += value
            }
            let coefficients = floating.map { value -> Int32 in
                let scaled = value / total * fixedScale
                let rounded = scaled < 0 ? scaled - 0.5 : scaled + 0.5
                return Int32(rounded.rounded(.towardZero))
            }
            return LanczosContribution(start: first, coefficients: coefficients)
        }
    }

    @inline(__always)
    private static func lanczos(_ value: Double) -> Double {
        let magnitude = abs(value)
        if magnitude < Double.ulpOfOne { return 1 }
        if magnitude >= 3 { return 0 }
        let piValue = Double.pi * value
        return sin(piValue) / piValue * sin(piValue / 3) / (piValue / 3)
    }

    @inline(__always)
    private static func clippedUInt8(_ value: Int64, precisionBits: Int) -> UInt8 {
        let shifted = value >> precisionBits
        return UInt8(clamping: shifted)
    }

    private static func appendStaticImagePatches(
        chw: [Float],
        width: Int,
        height: Int,
        patchSize: Int,
        patchTemporal: Int,
        to output: inout [Float]
    ) {
        let channelPlane = width * height
        for patchY in 0..<(height / patchSize) {
            for patchX in 0..<(width / patchSize) {
                for _ in 0..<patchTemporal {
                    for channel in 0..<3 {
                        for y in 0..<patchSize {
                            let sourceY = patchY * patchSize + y
                            for x in 0..<patchSize {
                                let sourceX = patchX * patchSize + x
                                output.append(chw[channel * channelPlane + sourceY * width + sourceX])
                            }
                        }
                    }
                }
            }
        }
    }
}
