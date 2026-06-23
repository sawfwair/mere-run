import Foundation
import MLX

public struct Krea2PreparedSample {
    public let imageTokens: MLXArray
    public let positionIds: MLXArray
    public let validMask: MLXArray
    public let imageTokenHeight: Int
    public let imageTokenWidth: Int
    public let textTokenCount: Int
    public let imageTokenCount: Int
}

public enum Krea2SampleBuilder {
    public static let vaeCompression = 8
    public static let patchSize = 2
    public static let latentChannels = 16
    public static let defaultMu: Float = 1.15

    public static func alignedResolution(width: Int, height: Int) -> (width: Int, height: Int) {
        let align = vaeCompression * patchSize
        return (roundUp(width, multiple: align), roundUp(height, multiple: align))
    }

    public static func patchify(_ latents: MLXArray, patch: Int = patchSize) -> MLXArray {
        let batch = latents.dim(0)
        let channels = latents.dim(1)
        let height = latents.dim(2)
        let width = latents.dim(3)
        var x = latents.reshaped(batch, channels, height / patch, patch, width / patch, patch)
        x = x.transposed(0, 2, 4, 1, 3, 5)
        return x.reshaped(batch, (height / patch) * (width / patch), channels * patch * patch)
    }

    public static func unpatchify(
        _ tokens: MLXArray,
        tokenHeight: Int,
        tokenWidth: Int,
        channels: Int = latentChannels,
        patch: Int = patchSize
    ) -> MLXArray {
        let batch = tokens.dim(0)
        var x = tokens.reshaped(batch, tokenHeight, tokenWidth, channels, patch, patch)
        x = x.transposed(0, 3, 1, 4, 2, 5)
        return x.reshaped(batch, channels, tokenHeight * patch, tokenWidth * patch)
    }

    public static func prepare(
        latents: MLXArray,
        textLength: Int,
        textMask: MLXArray,
        patch: Int = patchSize
    ) -> Krea2PreparedSample {
        let batch = latents.dim(0)
        let latentHeight = latents.dim(2)
        let latentWidth = latents.dim(3)
        let tokenHeight = latentHeight / patch
        let tokenWidth = latentWidth / patch
        let imageTokenCount = tokenHeight * tokenWidth
        let imageTokens = patchify(latents, patch: patch)
        let imageMask = MLX.ones([batch, imageTokenCount], dtype: .int32)
        let validMask = MLX.concatenated([textMask.asType(.int32), imageMask], axis: 1)

        var positions = [Float]()
        positions.reserveCapacity(batch * (textLength + imageTokenCount) * 3)
        for _ in 0..<batch {
            for _ in 0..<textLength {
                positions.append(0)
                positions.append(0)
                positions.append(0)
            }
            for y in 0..<tokenHeight {
                for x in 0..<tokenWidth {
                    positions.append(0)
                    positions.append(Float(y))
                    positions.append(Float(x))
                }
            }
        }

        return Krea2PreparedSample(
            imageTokens: imageTokens,
            positionIds: MLXArray(positions).reshaped(batch, textLength + imageTokenCount, 3),
            validMask: validMask,
            imageTokenHeight: tokenHeight,
            imageTokenWidth: tokenWidth,
            textTokenCount: textLength,
            imageTokenCount: imageTokenCount
        )
    }

    public static func timesteps(
        imageTokenCount: Int,
        steps: Int,
        mu: Float? = defaultMu,
        minResolution: Int = 256,
        maxResolution: Int = 1280,
        y1: Float = 0.5,
        y2: Float = 1.15,
        sigma: Float = 1.0
    ) -> [Float] {
        let x1 = Float((minResolution / (vaeCompression * patchSize)) * (minResolution / (vaeCompression * patchSize)))
        let x2 = Float((maxResolution / (vaeCompression * patchSize)) * (maxResolution / (vaeCompression * patchSize)))
        let resolvedMu: Float
        if let mu {
            resolvedMu = mu
        } else {
            let slope = (y2 - y1) / (x2 - x1)
            resolvedMu = slope * Float(imageTokenCount) + (y1 - slope * x1)
        }
        let expMu = Foundation.exp(resolvedMu)
        return (0...steps).map { index in
            let t = 1.0 - Float(index) / Float(steps)
            guard t > 0 else { return 0 }
            let shifted = expMu / (expMu + Foundation.pow(1.0 / t - 1.0, sigma))
            return shifted
        }
    }

    public static func paddedTextTokenInputs(
        promptTokenIds: [Int],
        suffixTokenIds: [Int],
        padTokenId: Int,
        maxLength: Int,
        prefixDropCount: Int = 34
    ) -> (tokenIds: [Int], attentionMask: [Int32]) {
        let promptBlockLength = maxLength + prefixDropCount - suffixTokenIds.count
        var promptIds = Array(promptTokenIds.prefix(promptBlockLength))
        var promptMask = Array(repeating: Int32(1), count: promptIds.count)
        if promptIds.count < promptBlockLength {
            let paddingCount = promptBlockLength - promptIds.count
            promptIds.append(contentsOf: Array(repeating: padTokenId, count: paddingCount))
            promptMask.append(contentsOf: Array(repeating: Int32(0), count: paddingCount))
        }
        return (
            tokenIds: promptIds + suffixTokenIds,
            attentionMask: promptMask + Array(repeating: Int32(1), count: suffixTokenIds.count)
        )
    }

    public static func qwenActivationLayerIndices(from hiddenStateIndices: [Int]) -> [Int] {
        hiddenStateIndices.map { max(0, $0 - 1) }
    }

    static func attentionMask(validMask: MLXArray, dtype: DType) -> MLXArray {
        let batch = validMask.dim(0)
        let length = validMask.dim(1)
        let columns = validMask.reshaped(batch, 1, 1, length)
        let keep = columns .== MLXArray(Int32(1))
        let zeros = MLX.zeros([batch, 1, length, length], dtype: dtype)
        return MLX.where(keep, zeros, zeros + MLXArray(-1.0e9).asType(dtype))
    }

    static func roundUp(_ value: Int, multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
    }
}
