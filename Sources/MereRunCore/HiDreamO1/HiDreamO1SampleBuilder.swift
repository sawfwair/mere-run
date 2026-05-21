import Foundation
import MLX

public enum HiDreamO1SampleBuilder {
    public static let timestepTokenCount = 1
    public static let patchSize = 32
    public static let conditionImageSize = 384
    public static let predefinedResolutions: [Resolution] = [
        .init(width: 2048, height: 2048),
        .init(width: 2304, height: 1728),
        .init(width: 1728, height: 2304),
        .init(width: 2560, height: 1440),
        .init(width: 1440, height: 2560),
        .init(width: 2496, height: 1664),
        .init(width: 1664, height: 2496),
        .init(width: 3104, height: 1312),
        .init(width: 1312, height: 3104),
        .init(width: 2304, height: 1792),
        .init(width: 1792, height: 2304),
    ]

    public struct Resolution: Sendable, Hashable {
        public var width: Int
        public var height: Int
    }

    public struct Sample: Sendable, Hashable {
        public var inputIds: [Int]
        public var paddedInputIds: [Int]
        public var positionIds: [[Int]]
        public var tokenTypes: [Int]
        public var vinputMask: [Bool]
        public var targetImageLength: Int
        public var referenceImageLengths: [Int]
    }

    public static func closestResolution(width: Int, height: Int) -> Resolution {
        guard width > 0, height > 0 else {
            return predefinedResolutions[0]
        }
        let ratio = Double(width) / Double(height)
        return predefinedResolutions.min { lhs, rhs in
            let lhsRatio = Double(lhs.width) / Double(lhs.height)
            let rhsRatio = Double(rhs.width) / Double(rhs.height)
            let lhsDistance = Darwin.fabs(lhsRatio - ratio)
            let rhsDistance = Darwin.fabs(rhsRatio - ratio)
            return lhsDistance < rhsDistance
        } ?? predefinedResolutions[0]
    }

    public static func resizedReferenceSize(
        originalWidth: Int,
        originalHeight: Int,
        maxSize: Int
    ) -> Resolution {
        guard originalWidth > 0, originalHeight > 0 else {
            return Resolution(width: patchSize, height: patchSize)
        }
        let maxArea = Double(maxSize * maxSize)
        let scale = sqrt(maxArea / Double(originalWidth * originalHeight))
        let widthScaled = Double(originalWidth) * scale
        let heightScaled = Double(originalHeight) * scale
        let candidates = [
            Resolution(width: roundedDown(widthScaled.rounded()), height: roundedDown(heightScaled.rounded())),
            Resolution(width: roundedDown(widthScaled.rounded()), height: roundedDown(floor(heightScaled))),
            Resolution(width: roundedDown(floor(widthScaled)), height: roundedDown(heightScaled.rounded())),
            Resolution(width: roundedDown(floor(widthScaled)), height: roundedDown(floor(heightScaled))),
        ].sorted { $0.width * $0.height > $1.width * $1.height }
        return candidates.first { $0.width * $0.height <= maxSize * maxSize }
            ?? Resolution(width: patchSize, height: patchSize)
    }

    public static func targetResolution(
        width: Int,
        height: Int,
        referenceOriginalSizes: [Resolution],
        keepOriginalAspect: Bool
    ) -> Resolution {
        if keepOriginalAspect, referenceOriginalSizes.count == 1 {
            let original = referenceOriginalSizes[0]
            return resizedReferenceSize(originalWidth: original.width, originalHeight: original.height, maxSize: 2_048)
        }
        return closestResolution(width: width, height: height)
    }

    public static func referenceMaxSize(targetWidth: Int, targetHeight: Int, referenceCount: Int) -> Int {
        let maxTarget = max(targetWidth, targetHeight)
        switch referenceCount {
        case 1:
            return maxTarget
        case 2:
            return maxTarget * 48 / 64
        case 3...4:
            return maxTarget / 2
        case 5...8:
            return maxTarget * 24 / 64
        default:
            return maxTarget / 4
        }
    }

    public static func conditionSize(referenceCount: Int) -> Int {
        switch referenceCount {
        case ...4:
            return conditionImageSize
        case 5...8:
            return conditionImageSize * 48 / 64
        default:
            return conditionImageSize / 2
        }
    }

    public static func vlmConditionSize(originalWidth: Int, originalHeight: Int, maxSize: Int) -> Resolution {
        guard originalWidth > 0, originalHeight > 0 else {
            return Resolution(width: patchSize, height: patchSize)
        }
        let ratio = Double(originalWidth) / Double(originalHeight)
        let width = sqrt(Double(maxSize * maxSize) * ratio)
        let height = width / ratio
        return Resolution(width: roundedDown(width), height: roundedDown(height))
    }

    public static func textToImageSample(
        textTokenIds: [Int],
        height: Int,
        width: Int,
        config: HiDreamO1Config
    ) -> Sample {
        let imageLength = (height / patchSize) * (width / patchSize)
        let paddedInputIds = paddedVisionInputIds(
            textTokenIds: textTokenIds,
            visionLengths: [imageLength],
            config: config
        )
        let imageGrid = [Resolution(width: width / patchSize, height: height / patchSize)]
        let positionIds = ropeIndexFixPoint(
            inputIds: paddedInputIds,
            imageGrids: imageGrid,
            skipVisionStartToken: [true],
            config: config
        )

        let textLength = textTokenIds.count
        let totalLength = positionIds.first?.count ?? textLength + imageLength
        var tokenTypes = Array(repeating: 0, count: totalLength)
        let generationStart = max(0, textLength - timestepTokenCount)
        if generationStart < totalLength {
            for index in generationStart..<totalLength {
                tokenTypes[index] = 1
            }
        }
        if generationStart < textLength {
            for index in generationStart..<textLength {
                tokenTypes[index] = 3
            }
        }
        let vinputMask = tokenTypes.map { $0 == 1 }
        let tokenTypesBin = tokenTypes.map { $0 > 0 ? 1 : 0 }

        return Sample(
            inputIds: textTokenIds,
            paddedInputIds: paddedInputIds,
            positionIds: positionIds,
            tokenTypes: tokenTypesBin,
            vinputMask: vinputMask,
            targetImageLength: imageLength,
            referenceImageLengths: []
        )
    }

    public static func referenceSample(
        textTokenIds: [Int],
        targetHeight: Int,
        targetWidth: Int,
        referenceSizes: [Resolution],
        conditionGrids: [Resolution] = [],
        config: HiDreamO1Config
    ) -> Sample {
        let targetLength = (targetHeight / patchSize) * (targetWidth / patchSize)
        let referenceLengths = referenceSizes.map { ($0.height / patchSize) * ($0.width / patchSize) }
        let totalReferenceLength = referenceLengths.reduce(0, +)
        let visionLengths = [targetLength] + referenceLengths
        let paddedInputIds = paddedVisionInputIds(
            textTokenIds: textTokenIds,
            visionLengths: visionLengths,
            config: config
        )
        let imageGrids = [Resolution(width: targetWidth / patchSize, height: targetHeight / patchSize)]
            + referenceSizes.map { Resolution(width: $0.width / patchSize, height: $0.height / patchSize) }
        let positionIds = ropeIndexFixPoint(
            inputIds: paddedInputIds,
            imageGrids: conditionGrids + imageGrids,
            skipVisionStartToken: Array(repeating: false, count: conditionGrids.count)
                + Array(repeating: true, count: imageGrids.count),
            config: config
        )

        let textLength = textTokenIds.count
        let totalLength = positionIds.first?.count ?? textLength + targetLength + totalReferenceLength
        var tokenTypes = Array(repeating: 0, count: totalLength)
        let generationStart = max(0, textLength - timestepTokenCount)
        let generationEnd = min(totalLength, generationStart + targetLength + timestepTokenCount)
        if generationStart < generationEnd {
            for index in generationStart..<generationEnd {
                tokenTypes[index] = 1
            }
        }
        if generationEnd < totalLength {
            for index in generationEnd..<totalLength {
                tokenTypes[index] = 2
            }
        }
        if generationStart < textLength {
            for index in generationStart..<textLength {
                tokenTypes[index] = 3
            }
        }
        let vinputMask = tokenTypes.map { $0 == 1 || $0 == 2 }
        let tokenTypesBin = tokenTypes.map { $0 > 0 ? 1 : 0 }

        return Sample(
            inputIds: textTokenIds,
            paddedInputIds: paddedInputIds,
            positionIds: positionIds,
            tokenTypes: tokenTypesBin,
            vinputMask: vinputMask,
            targetImageLength: targetLength,
            referenceImageLengths: referenceLengths
        )
    }

    public static func patchifyCHW(_ image: MLXArray, patchSize: Int = patchSize) -> MLXArray {
        precondition(image.ndim == 3, "Expected CHW image.")
        let channels = image.dim(0)
        let height = image.dim(1)
        let width = image.dim(2)
        let patchH = height / patchSize
        let patchW = width / patchSize
        return image
            .reshaped(channels, patchH, patchSize, patchW, patchSize)
            .transposed(1, 3, 0, 2, 4)
            .reshaped(patchH * patchW, channels * patchSize * patchSize)
    }

    public static func unpatchifyCHW(_ patches: MLXArray, height: Int, width: Int, patchSize: Int = patchSize) -> MLXArray {
        let channels = patches.dim(1) / (patchSize * patchSize)
        let patchH = height / patchSize
        let patchW = width / patchSize
        return patches
            .reshaped(patchH, patchW, channels, patchSize, patchSize)
            .transposed(2, 0, 3, 1, 4)
            .reshaped(channels, height, width)
    }

    private static func paddedVisionInputIds(
        textTokenIds: [Int],
        visionLengths: [Int],
        config: HiDreamO1Config
    ) -> [Int] {
        var inputIds = textTokenIds
        for length in visionLengths {
            guard length > 0 else {
                continue
            }
            inputIds.append(config.visionStartTokenId)
            if length > 1 {
                inputIds.append(contentsOf: Array(repeating: config.imageTokenId, count: length - 1))
            }
        }
        return inputIds
    }

    private static func ropeIndexFixPoint(
        inputIds: [Int],
        imageGrids: [Resolution],
        skipVisionStartToken: [Bool],
        config: HiDreamO1Config,
        fixPoint: Int = 4_096
    ) -> [[Int]] {
        var imageIndex = 0
        var remainingImages = imageGrids.count
        var tokens = inputIds
        var positionGroups: [[[Int]]] = []
        var start = 0
        var nextFixPoint = fixPoint

        while remainingImages > 0 {
            let imageEnd = tokens[start...].firstIndex(of: config.imageTokenId) ?? (tokens.count + 1)
            let grid = imageGrids[imageIndex]
            let skipVisionStart = imageIndex < skipVisionStartToken.count && skipVisionStartToken[imageIndex]
            imageIndex += 1
            remainingImages -= 1

            let gridT = 1
            let gridH = grid.height
            let gridW = grid.width
            let rawTextLength = imageEnd - start
            let textLength = max(0, rawTextLength - (skipVisionStart ? 1 : 0))
            let startIndex = (positionGroups.flatMap { $0 }.flatMap { $0 }.max() ?? -1) + 1
            if textLength > 0 {
                positionGroups.append(repeatedPositions(start: startIndex, count: textLength))
            }

            let visionStart = skipVisionStart ? startIndex + max(0, nextFixPoint - startIndex) : startIndex + textLength
            positionGroups.append(visionPositions(t: gridT, h: gridH, w: gridW, start: visionStart))
            if skipVisionStart {
                nextFixPoint = 0
            }
            start = imageEnd + gridT * gridH * gridW
            tokens = inputIds
        }

        if start < inputIds.count {
            let startIndex = (positionGroups.flatMap { $0 }.flatMap { $0 }.max() ?? -1) + 1
            let textLength = inputIds.count - start
            positionGroups.append(repeatedPositions(start: startIndex, count: textLength))
        }

        let emptyPositions: [[Int]] = [[], [], []]
        let flattened = positionGroups.reduce(into: emptyPositions) { partial, group in
            partial[0].append(contentsOf: group[0])
            partial[1].append(contentsOf: group[1])
            partial[2].append(contentsOf: group[2])
        }
        return flattened
    }

    private static func repeatedPositions(start: Int, count: Int) -> [[Int]] {
        let positions = Array(start..<(start + count))
        return [positions, positions, positions]
    }

    private static func visionPositions(t: Int, h: Int, w: Int, start: Int) -> [[Int]] {
        var tPositions: [Int] = []
        var hPositions: [Int] = []
        var wPositions: [Int] = []
        for tIndex in 0..<t {
            for hIndex in 0..<h {
                for wIndex in 0..<w {
                    tPositions.append(start + tIndex)
                    hPositions.append(start + hIndex)
                    wPositions.append(start + wIndex)
                }
            }
        }
        return [tPositions, hPositions, wPositions]
    }

    private static func roundedDown(_ value: Double) -> Int {
        max(patchSize, Int(value) / patchSize * patchSize)
    }
}
