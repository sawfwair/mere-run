import Foundation
import MediaIO
import MLX

struct Gemma4UnifiedImageBatch {
    let pixelValues: MLXArray
    let imagePositionIds: MLXArray
    let softTokenCounts: [Int]
}

enum Gemma4UnifiedImageProcessor {
    static func makeBatch(
        imageReferences: [String],
        visionConfig: Gemma4UnifiedVisionConfig
    ) throws -> Gemma4UnifiedImageBatch {
        guard !imageReferences.isEmpty else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 unified image batch is empty.")
        }

        let maxSoftTokens = max(1, visionConfig.numSoftTokens)
        let modelPatchSize = max(1, visionConfig.modelPatchSize)
        let patchDim = modelPatchSize * modelPatchSize * 3

        var allPatches: [Float] = []
        var allPositions: [Int32] = []
        var softTokenCounts: [Int] = []
        allPatches.reserveCapacity(imageReferences.count * maxSoftTokens * patchDim)
        allPositions.reserveCapacity(imageReferences.count * maxSoftTokens * 2)
        softTokenCounts.reserveCapacity(imageReferences.count)

        for reference in imageReferences {
            let image = try decodeImage(reference)
            let processed = try preprocess(
                image,
                visionConfig: visionConfig,
                maxSoftTokens: maxSoftTokens,
                modelPatchSize: modelPatchSize
            )
            softTokenCounts.append(processed.softTokenCount)
            allPatches.append(contentsOf: processed.patches)
            allPositions.append(contentsOf: processed.positions)
        }

        return Gemma4UnifiedImageBatch(
            pixelValues: MLXArray(allPatches, [imageReferences.count, maxSoftTokens, patchDim]),
            imagePositionIds: MLXArray(allPositions, [imageReferences.count, maxSoftTokens, 2]),
            softTokenCounts: softTokenCounts
        )
    }

    static func expandedPromptTokens(
        _ tokens: [Int],
        softTokenCounts: [Int],
        imageTokenId: Int,
        boiTokenId: Int,
        eoiTokenId: Int
    ) throws -> [Int] {
        guard !softTokenCounts.isEmpty else { return tokens }
        let imageOccurrences = tokens.filter { $0 == imageTokenId }.count
        let expandedImageTokenCount = softTokenCounts.reduce(0, +)
        if imageOccurrences == expandedImageTokenCount {
            return tokens
        }
        guard imageOccurrences == softTokenCounts.count else {
            throw Gemma4Error.unsupportedConfiguration(
                "Gemma4 unified prompt has \(imageOccurrences) image placeholders for \(softTokenCounts.count) image(s)."
            )
        }

        var countIndex = 0
        var expanded: [Int] = []
        expanded.reserveCapacity(tokens.count + expandedImageTokenCount + (softTokenCounts.count * 2))
        for token in tokens {
            guard token == imageTokenId else {
                expanded.append(token)
                continue
            }
            let softTokenCount = softTokenCounts[countIndex]
            expanded.append(boiTokenId)
            expanded.append(contentsOf: Array(repeating: imageTokenId, count: softTokenCount))
            expanded.append(eoiTokenId)
            countIndex += 1
        }
        return expanded
    }

    static func mmTokenTypeIds(tokens: [Int], imageTokenId: Int) -> MLXArray {
        let values = tokens.map { token -> Int32 in
            token == imageTokenId ? 1 : 0
        }
        return MLXArray(values, [1, values.count])
    }

    private static func decodeImage(_ reference: String) throws -> MediaImage {
        let trimmed = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw Gemma4Error.unsupportedConfiguration("Image URL is empty.")
        }
        if trimmed.lowercased().hasPrefix("data:") {
            guard let comma = trimmed.firstIndex(of: ",") else {
                throw Gemma4Error.unsupportedConfiguration("Invalid image data URL.")
            }
            let metadata = trimmed[..<comma].lowercased()
            guard metadata.contains(";base64") else {
                throw Gemma4Error.unsupportedConfiguration("Only base64 image data URLs are supported.")
            }
            let payload = String(trimmed[trimmed.index(after: comma)...])
            guard let data = Data(base64Encoded: payload, options: [.ignoreUnknownCharacters]) else {
                throw Gemma4Error.unsupportedConfiguration("Invalid base64 image data URL.")
            }
            return try MediaImageIO.decode(data: data)
        }

        if let url = URL(string: trimmed), url.scheme?.lowercased() == "file" {
            return try MediaImageIO.decode(url)
        }
        if let scheme = URL(string: trimmed)?.scheme?.lowercased(),
           scheme == "http" || scheme == "https" {
            throw Gemma4Error.unsupportedConfiguration("Remote image URLs are not fetched by the local Gemma4 runtime; use a file path or data URL.")
        }

        return try MediaImageIO.decode(URL(fileURLWithPath: trimmed).standardizedFileURL)
    }

    private static func preprocess(
        _ image: MediaImage,
        visionConfig: Gemma4UnifiedVisionConfig,
        maxSoftTokens: Int,
        modelPatchSize: Int
    ) throws -> (patches: [Float], positions: [Int32], softTokenCount: Int) {
        let (targetWidth, targetHeight) = targetSize(
            originalWidth: image.width,
            originalHeight: image.height,
            patchSize: max(1, visionConfig.patchSize),
            poolingKernelSize: max(1, visionConfig.poolingKernelSize),
            maxSoftTokens: maxSoftTokens
        )
        let resized = try MediaImageIO.resized(image, width: targetWidth, height: targetHeight)
        let chw = MediaImageIO.rgbCHWFloat(resized, normalizedToMinusOneToOne: false)
        let patchRows = max(1, resized.height / modelPatchSize)
        let patchCols = max(1, resized.width / modelPatchSize)
        let patchDim = modelPatchSize * modelPatchSize * 3
        let actualPatches = min(maxSoftTokens, patchRows * patchCols)

        var patches = [Float](repeating: 0, count: maxSoftTokens * patchDim)
        var positions = [Int32](repeating: -1, count: maxSoftTokens * 2)
        let channelPlane = resized.width * resized.height

        var patchIndex = 0
        for patchY in 0..<patchRows {
            for patchX in 0..<patchCols {
                guard patchIndex < maxSoftTokens else { break }
                var writeOffset = patchIndex * patchDim
                for y in 0..<modelPatchSize {
                    let sourceY = (patchY * modelPatchSize) + y
                    for x in 0..<modelPatchSize {
                        let sourceX = (patchX * modelPatchSize) + x
                        let pixelOffset = (sourceY * resized.width) + sourceX
                        for channel in 0..<3 {
                            patches[writeOffset] = chw[(channel * channelPlane) + pixelOffset]
                            writeOffset += 1
                        }
                    }
                }
                positions[patchIndex * 2] = Int32(patchX)
                positions[(patchIndex * 2) + 1] = Int32(patchY)
                patchIndex += 1
            }
        }

        return (patches, positions, actualPatches)
    }

    private static func targetSize(
        originalWidth width: Int,
        originalHeight height: Int,
        patchSize: Int,
        poolingKernelSize: Int,
        maxSoftTokens: Int
    ) -> (width: Int, height: Int) {
        let maxPatches = maxSoftTokens * poolingKernelSize * poolingKernelSize
        let targetPixels = maxPatches * patchSize * patchSize
        let factor = sqrt(Double(targetPixels) / Double(max(1, width * height)))
        let sideMultiple = max(1, patchSize * poolingKernelSize)

        var targetHeight = Int(floor((factor * Double(height)) / Double(sideMultiple))) * sideMultiple
        var targetWidth = Int(floor((factor * Double(width)) / Double(sideMultiple))) * sideMultiple

        if targetHeight == 0 && targetWidth == 0 {
            targetHeight = sideMultiple
            targetWidth = sideMultiple
        } else if targetHeight == 0 {
            targetHeight = sideMultiple
            let aspect = max(1, Int(floor(Double(width) / Double(max(1, height)))))
            let maxSideLength = max(1, (maxPatches / max(1, poolingKernelSize * poolingKernelSize)) * sideMultiple)
            targetWidth = min(aspect * sideMultiple, maxSideLength)
        } else if targetWidth == 0 {
            targetWidth = sideMultiple
            let aspect = max(1, Int(floor(Double(height) / Double(max(1, width)))))
            let maxSideLength = max(1, (maxPatches / max(1, poolingKernelSize * poolingKernelSize)) * sideMultiple)
            targetHeight = min(aspect * sideMultiple, maxSideLength)
        }

        return (max(sideMultiple, targetWidth), max(sideMultiple, targetHeight))
    }
}
