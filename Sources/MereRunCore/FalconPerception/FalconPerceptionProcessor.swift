import Foundation
import MediaIO
@preconcurrency import MLX

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

public struct FalconPerceptionProcessedInput: @unchecked Sendable {
    public let inputIDs: MLXArray
    public let pixelValues: MLXArray
    public let imageGridHW: MLXArray
    public let processedSize: (width: Int, height: Int)

    public init(inputIDs: MLXArray, pixelValues: MLXArray, imageGridHW: MLXArray, processedSize: (width: Int, height: Int)) {
        self.inputIDs = inputIDs
        self.pixelValues = pixelValues
        self.imageGridHW = imageGridHW
        self.processedSize = processedSize
    }
}

public enum FalconPerceptionProcessorError: LocalizedError {
    case unsupportedPlatform
    case invalidImage(URL)

    public var errorDescription: String? {
        switch self {
        case .unsupportedPlatform:
            return "Falcon Perception preprocessing requires a supported MediaIO image backend."
        case .invalidImage(let url):
            return "Failed to load image: \(url.path)"
        }
    }
}

public struct FalconPerceptionProcessor: @unchecked Sendable {
    public static let imageMean: [Float] = [0.5, 0.5, 0.5]
    public static let imageStd: [Float] = [0.5, 0.5, 0.5]

    public let tokenizer: FalconPerceptionTokenizer
    public let config: FalconPerceptionModelConfig

    public init(tokenizer: FalconPerceptionTokenizer, config: FalconPerceptionModelConfig) {
        self.tokenizer = tokenizer
        self.config = config
    }

    public func makePrompt(for query: String) -> String {
        "<|image|>Segment these expressions in the image:<|start_of_query|>\(query)<|REF_SEG|>"
    }

    public func process(imageURL: URL, query: String) throws -> FalconPerceptionProcessedInput {
        let image: MediaImage
        do {
            image = try MediaImageIO.decode(imageURL)
        } catch {
            throw FalconPerceptionProcessorError.invalidImage(imageURL)
        }
        return process(imageRGBA: image, query: query)
    }

    public func process(imageRGBA: MediaImage, query: String) -> FalconPerceptionProcessedInput {
        let resized = Self.resizeIfNecessary(imageRGBA, shortest: 256, longest: 1024)
        let smartResized = Self.smartResize(resized, factor: config.visionConfig.spatialPatchSize)
        let pixels = Self.normalizedPixels(from: smartResized)
        let height = smartResized.height
        let width = smartResized.width
        let gridH = height / config.visionConfig.spatialPatchSize
        let gridW = width / config.visionConfig.spatialPatchSize

        let prompt = makePrompt(for: query)
        let tokenIDs = tokenizer.encode(prompt, addSpecialTokens: false)
        let expanded = expandImageTokens(tokenIDs, gridH: gridH, gridW: gridW)

        let inputIDs = MLXArray(expanded.map(Int32.init), [1, expanded.count])
        let pixelValues = MLXArray(pixels, [1, height, width, 3])
        let imageGridHW = MLXArray([Int32(gridH), Int32(gridW)], [1, 2])

        return FalconPerceptionProcessedInput(
            inputIDs: inputIDs,
            pixelValues: pixelValues,
            imageGridHW: imageGridHW,
            processedSize: (width, height)
        )
    }

    #if canImport(CoreGraphics)
    public func process(imageRGBA: CGImage, query: String) -> FalconPerceptionProcessedInput {
        let resized = Self.resizeIfNecessary(imageRGBA, shortest: 256, longest: 1024)
        let smartResized = Self.smartResize(resized, factor: config.visionConfig.spatialPatchSize)
        let pixels = Self.normalizedPixels(from: smartResized)
        let height = smartResized.height
        let width = smartResized.width
        let gridH = height / config.visionConfig.spatialPatchSize
        let gridW = width / config.visionConfig.spatialPatchSize

        let prompt = makePrompt(for: query)
        let tokenIDs = tokenizer.encode(prompt, addSpecialTokens: false)
        let expanded = expandImageTokens(tokenIDs, gridH: gridH, gridW: gridW)

        let inputIDs = MLXArray(expanded.map(Int32.init), [1, expanded.count])
        let pixelValues = MLXArray(pixels, [1, height, width, 3])
        let imageGridHW = MLXArray([Int32(gridH), Int32(gridW)], [1, 2])

        return FalconPerceptionProcessedInput(
            inputIDs: inputIDs,
            pixelValues: pixelValues,
            imageGridHW: imageGridHW,
            processedSize: (width, height)
        )
    }
    #endif

    public func expandImageTokens(_ tokenIDs: [Int], gridH: Int, gridW: Int) -> [Int] {
        let imagePrefixIDs = [
            config.imageCLSTokenID,
            config.imageReg1TokenID,
            config.imageReg2TokenID,
            config.imageReg3TokenID,
            config.imageReg4TokenID,
        ]

        var expanded: [Int] = []
        expanded.reserveCapacity(tokenIDs.count + gridH * gridW + imagePrefixIDs.count + 1)
        for tokenID in tokenIDs {
            if tokenID == config.imgID {
                expanded.append(contentsOf: imagePrefixIDs)
                expanded.append(contentsOf: Array(repeating: config.imgID, count: gridH * gridW))
                expanded.append(config.imgEndID)
            } else {
                expanded.append(tokenID)
            }
        }
        return expanded
    }

    public static func resizeIfNecessary(_ image: MediaImage, shortest: Int, longest: Int) -> MediaImage {
        let width = image.width
        let height = image.height
        if shortest <= width && width <= longest && shortest <= height && height <= longest {
            return image
        }

        let aspectRatio = Double(width) / Double(height)
        let isVertical = width < height

        var newWidth: Int
        var newHeight: Int
        if width < shortest || height < shortest {
            if isVertical {
                newWidth = shortest
                newHeight = Int(Double(shortest) / aspectRatio)
            } else {
                newHeight = shortest
                newWidth = Int(Double(shortest) * aspectRatio)
            }
        } else if isVertical {
            newWidth = longest
            newHeight = Int(Double(newWidth) / aspectRatio)
        } else {
            newHeight = longest
            newWidth = Int(Double(newHeight) * aspectRatio)
        }

        if newWidth > longest {
            newWidth = longest
            newHeight = Int(Double(newWidth) / aspectRatio)
        }
        if newHeight > longest {
            newHeight = longest
            newWidth = Int(Double(newHeight) * aspectRatio)
        }

        return (try? MediaImageIO.resized(image, width: newWidth, height: newHeight)) ?? image
    }

    public static func smartResize(_ image: MediaImage, factor: Int, minPixels: Int = 56 * 56, maxPixels: Int = 28 * 28 * 1280) -> MediaImage {
        let width = image.width
        let height = image.height

        var roundedHeight = max(factor, Int((Double(height) / Double(factor)).rounded()) * factor)
        var roundedWidth = max(factor, Int((Double(width) / Double(factor)).rounded()) * factor)

        if roundedHeight * roundedWidth > maxPixels {
            let beta = sqrt(Double(height * width) / Double(maxPixels))
            roundedHeight = max(factor, Int(floor(Double(height) / beta / Double(factor))) * factor)
            roundedWidth = max(factor, Int(floor(Double(width) / beta / Double(factor))) * factor)
        } else if roundedHeight * roundedWidth < minPixels {
            let beta = sqrt(Double(minPixels) / Double(height * width))
            roundedHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
            roundedWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }

        if roundedWidth == width && roundedHeight == height {
            return image
        }
        return (try? MediaImageIO.resized(image, width: roundedWidth, height: roundedHeight)) ?? image
    }

    private static func normalizedPixels(from image: MediaImage) -> [Float] {
        let width = image.width
        let height = image.height
        var normalized: [Float] = []
        normalized.reserveCapacity(width * height * 3)
        for pixel in 0..<(width * height) {
            let offset = pixel * 4
            for channel in 0..<3 {
                let value = Float(image.rgba8[offset + channel]) / 255.0
                normalized.append((value - imageMean[channel]) / imageStd[channel])
            }
        }
        return normalized
    }

    #if canImport(CoreGraphics)
    public static func resizeIfNecessary(_ image: CGImage, shortest: Int, longest: Int) -> CGImage {
        let width = image.width
        let height = image.height
        if shortest <= width && width <= longest && shortest <= height && height <= longest {
            return image
        }

        let aspectRatio = Double(width) / Double(height)
        let isVertical = width < height

        var newWidth: Int
        var newHeight: Int
        if width < shortest || height < shortest {
            if isVertical {
                newWidth = shortest
                newHeight = Int(Double(shortest) / aspectRatio)
            } else {
                newHeight = shortest
                newWidth = Int(Double(shortest) * aspectRatio)
            }
        } else if isVertical {
            newWidth = longest
            newHeight = Int(Double(newWidth) / aspectRatio)
        } else {
            newHeight = longest
            newWidth = Int(Double(newHeight) * aspectRatio)
        }

        if newWidth > longest {
            newWidth = longest
            newHeight = Int(Double(newWidth) / aspectRatio)
        }
        if newHeight > longest {
            newHeight = longest
            newWidth = Int(Double(newHeight) * aspectRatio)
        }

        return resize(image, width: newWidth, height: newHeight)
    }

    public static func smartResize(_ image: CGImage, factor: Int, minPixels: Int = 56 * 56, maxPixels: Int = 28 * 28 * 1280) -> CGImage {
        let width = image.width
        let height = image.height

        var roundedHeight = max(factor, Int((Double(height) / Double(factor)).rounded()) * factor)
        var roundedWidth = max(factor, Int((Double(width) / Double(factor)).rounded()) * factor)

        if roundedHeight * roundedWidth > maxPixels {
            let beta = sqrt(Double(height * width) / Double(maxPixels))
            roundedHeight = max(factor, Int(floor(Double(height) / beta / Double(factor))) * factor)
            roundedWidth = max(factor, Int(floor(Double(width) / beta / Double(factor))) * factor)
        } else if roundedHeight * roundedWidth < minPixels {
            let beta = sqrt(Double(minPixels) / Double(height * width))
            roundedHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
            roundedWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }

        if roundedWidth == width && roundedHeight == height {
            return image
        }
        return resize(image, width: roundedWidth, height: roundedHeight)
    }

    private static func loadImageRGBA(url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    private static func resize(_ image: CGImage, width: Int, height: Int) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return image
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage() ?? image
    }

    private static func normalizedPixels(from image: CGImage) -> [Float] {
        let width = image.width
        let height = image.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = width * 4
        var raw = [UInt8](repeating: 0, count: bytesPerRow * height)
        raw.withUnsafeMutableBytes { bytes in
            guard let context = CGContext(
                data: bytes.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                return
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        var normalized: [Float] = []
        normalized.reserveCapacity(width * height * 3)
        for offset in stride(from: 0, to: raw.count, by: 4) {
            for channel in 0..<3 {
                let value = Float(raw[offset + channel]) / 255.0
                normalized.append((value - imageMean[channel]) / imageStd[channel])
            }
        }
        return normalized
    }
    #endif
}
