import Foundation
import MediaIO
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
#endif

public enum HiDreamO1ImagePreprocessorError: LocalizedError, Sendable {
    case imageLoadFailed(URL)
    case imageResizeFailed
    case pixelBufferFailed
    case invalidSize(width: Int, height: Int)
    case imageWriteFailed(URL)

    public var errorDescription: String? {
        switch self {
        case .imageLoadFailed(let url):
            return "Failed to load HiDream O1 image: \(url.path)"
        case .imageResizeFailed:
            return "Failed to resize HiDream O1 image."
        case .pixelBufferFailed:
            return "Failed to convert HiDream O1 image pixels."
        case .invalidSize(let width, let height):
            return "HiDream O1 image dimensions must be positive multiples of 32, got \(width)x\(height)."
        case .imageWriteFailed(let url):
            return "Failed to write HiDream O1 image: \(url.path)"
        }
    }
}

public enum HiDreamO1ImagePreprocessor {
    public struct PatchTensor {
        public var resolution: HiDreamO1SampleBuilder.Resolution
        public var imageCHW: MLXArray
        public var patches: MLXArray
    }

    public struct VisionConditionTensor {
        public var resolution: HiDreamO1SampleBuilder.Resolution
        public var grid: QwenVisionGrid
        public var mergedGrid: HiDreamO1SampleBuilder.Resolution
        public var pixelValues: MLXArray

        public var tokenCount: Int {
            mergedGrid.width * mergedGrid.height
        }
    }

    public static func patchTensor(
        from url: URL,
        resolution: HiDreamO1SampleBuilder.Resolution,
        dtype: DType = .float32
    ) throws -> PatchTensor {
        let image = try loadImage(from: url)
        return try patchTensor(from: image, resolution: resolution, dtype: dtype)
    }

    public static func visionConditionTensor(
        from url: URL,
        resolution: HiDreamO1SampleBuilder.Resolution,
        config: HiDreamO1Config,
        dtype: DType = .bfloat16
    ) throws -> VisionConditionTensor {
        let image = try loadImage(from: url)
        return try visionConditionTensor(from: image, resolution: resolution, config: config, dtype: dtype)
    }

    public static func imageSize(_ url: URL) throws -> HiDreamO1SampleBuilder.Resolution {
        do {
            let size = try MediaImageIO.size(of: url)
            return .init(width: size.width, height: size.height)
        } catch {
            throw HiDreamO1ImagePreprocessorError.imageLoadFailed(url)
        }
    }

    public static func patchTensor(
        from image: MediaImage,
        resolution: HiDreamO1SampleBuilder.Resolution,
        dtype: DType = .float32
    ) throws -> PatchTensor {
        let imageCHW = try normalizedCHW(from: image, resolution: resolution, dtype: dtype)
        let patches = HiDreamO1SampleBuilder.patchifyCHW(imageCHW)
        return PatchTensor(resolution: resolution, imageCHW: imageCHW, patches: patches)
    }

    public static func normalizedCHW(
        from image: MediaImage,
        resolution: HiDreamO1SampleBuilder.Resolution,
        dtype: DType = .float32
    ) throws -> MLXArray {
        try validate(resolution)
        let resized = try MediaImageIO.resized(image, width: resolution.width, height: resolution.height)
        let values = MediaImageIO.rgbCHWFloat(resized, normalizedToMinusOneToOne: true)
        return MLXArray(values, [3, resized.height, resized.width]).asType(dtype)
    }

    public static func visionConditionTensor(
        from image: MediaImage,
        resolution: HiDreamO1SampleBuilder.Resolution,
        config: HiDreamO1Config,
        dtype: DType = .bfloat16
    ) throws -> VisionConditionTensor {
        let patchSize = config.visionConfig.patchSize
        let mergeSize = config.visionConfig.spatialMergeSize
        try validateVisionResolution(resolution, patchSize: patchSize, mergeSize: mergeSize)
        let imageCHW = try normalizedCHW(from: image, resolution: resolution, dtype: dtype)
        let patchInputs = prepareVisionPatchInputs(
            imageCHW: imageCHW,
            patchSize: patchSize,
            temporalPatchSize: config.visionConfig.temporalPatchSize,
            mergeSize: mergeSize
        )
        let gridHeight = resolution.height / patchSize
        let gridWidth = resolution.width / patchSize
        return VisionConditionTensor(
            resolution: resolution,
            grid: QwenVisionGrid(temporal: 1, height: gridHeight, width: gridWidth),
            mergedGrid: .init(width: gridWidth / mergeSize, height: gridHeight / mergeSize),
            pixelValues: patchInputs
        )
    }

    public static func saveNormalizedCHW(_ image: MLXArray, to url: URL) throws {
        do {
            try QwenImageIO.saveImage(array: (image.asType(.float32) + 1.0) / 2.0, to: url)
        } catch {
            throw HiDreamO1ImagePreprocessorError.imageWriteFailed(url)
        }
    }

    private static func loadImage(from url: URL) throws -> MediaImage {
        do {
            return try MediaImageIO.decode(url)
        } catch {
            throw HiDreamO1ImagePreprocessorError.imageLoadFailed(url)
        }
    }

    #if canImport(CoreGraphics)

    public static func patchTensor(
        from image: CGImage,
        resolution: HiDreamO1SampleBuilder.Resolution,
        dtype: DType = .float32
    ) throws -> PatchTensor {
        let imageCHW = try normalizedCHW(from: image, resolution: resolution, dtype: dtype)
        let patches = HiDreamO1SampleBuilder.patchifyCHW(imageCHW)
        return PatchTensor(resolution: resolution, imageCHW: imageCHW, patches: patches)
    }

    public static func normalizedCHW(
        from image: CGImage,
        resolution: HiDreamO1SampleBuilder.Resolution,
        dtype: DType = .float32
    ) throws -> MLXArray {
        try validate(resolution)
        let resized = try resizedCGImage(from: image, resolution: resolution)
        return try normalizedCHW(from: resized, dtype: dtype)
    }

    public static func visionConditionTensor(
        from image: CGImage,
        resolution: HiDreamO1SampleBuilder.Resolution,
        config: HiDreamO1Config,
        dtype: DType = .bfloat16
    ) throws -> VisionConditionTensor {
        let patchSize = config.visionConfig.patchSize
        let mergeSize = config.visionConfig.spatialMergeSize
        try validateVisionResolution(resolution, patchSize: patchSize, mergeSize: mergeSize)
        let imageCHW = try normalizedCHW(from: image, resolution: resolution, dtype: dtype)
        let patchInputs = prepareVisionPatchInputs(
            imageCHW: imageCHW,
            patchSize: patchSize,
            temporalPatchSize: config.visionConfig.temporalPatchSize,
            mergeSize: mergeSize
        )
        let gridHeight = resolution.height / patchSize
        let gridWidth = resolution.width / patchSize
        return VisionConditionTensor(
            resolution: resolution,
            grid: QwenVisionGrid(temporal: 1, height: gridHeight, width: gridWidth),
            mergedGrid: .init(width: gridWidth / mergeSize, height: gridHeight / mergeSize),
            pixelValues: patchInputs
        )
    }

    private static func loadCGImage(from url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw HiDreamO1ImagePreprocessorError.imageLoadFailed(url)
        }
        return image
    }

    private static func resizedCGImage(
        from image: CGImage,
        resolution: HiDreamO1SampleBuilder.Resolution
    ) throws -> CGImage {
        guard resolution.width > 0, resolution.height > 0 else {
            throw HiDreamO1ImagePreprocessorError.imageResizeFailed
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let context = CGContext(
            data: nil,
            width: resolution.width,
            height: resolution.height,
            bitsPerComponent: 8,
            bytesPerRow: resolution.width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            throw HiDreamO1ImagePreprocessorError.imageResizeFailed
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: resolution.width, height: resolution.height))
        guard let resized = context.makeImage() else {
            throw HiDreamO1ImagePreprocessorError.imageResizeFailed
        }
        return resized
    }

    private static func normalizedCHW(from image: CGImage, dtype: DType) throws -> MLXArray {
        let width = image.width
        let height = image.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var buffer = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        let succeeded = buffer.withUnsafeMutableBytes { pointer -> Bool in
            guard let baseAddress = pointer.baseAddress,
                  let context = CGContext(
                      data: baseAddress,
                      width: width,
                      height: height,
                      bitsPerComponent: 8,
                      bytesPerRow: bytesPerRow,
                      space: colorSpace,
                      bitmapInfo: bitmapInfo
                  ) else {
                return false
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }

        guard succeeded else {
            throw HiDreamO1ImagePreprocessorError.pixelBufferFailed
        }

        var values = [Float](repeating: 0, count: width * height * 3)
        for yIndex in 0..<height {
            for xIndex in 0..<width {
                let pixelIndex = yIndex * width + xIndex
                let sourceIndex = pixelIndex * bytesPerPixel
                values[pixelIndex] = Float(buffer[sourceIndex]) / 127.5 - 1.0
                values[pixelIndex + width * height] = Float(buffer[sourceIndex + 1]) / 127.5 - 1.0
                values[pixelIndex + 2 * width * height] = Float(buffer[sourceIndex + 2]) / 127.5 - 1.0
            }
        }

        return MLXArray(values, [3, height, width]).asType(dtype)
    }

    private static func cgImage(fromNormalizedCHW image: MLXArray) throws -> CGImage {
        precondition(image.ndim == 3 && image.dim(0) == 3, "Expected [3,H,W] normalized image.")
        let height = image.dim(1)
        let width = image.dim(2)
        let pixelCount = height * width
        let normalized = MLX.clip((image.asType(.float32) + 1.0) / 2.0, min: 0, max: 1)
        let scaled = (normalized * 255.0).asType(.uint8)
        MLX.eval(scaled)
        let data = scaled.asData().data

        var bytes = [UInt8](repeating: 255, count: pixelCount * 4)
        data.withUnsafeBytes { pointer in
            let source = pointer.bindMemory(to: UInt8.self)
            for pixelIndex in 0..<pixelCount {
                let destinationIndex = pixelIndex * 4
                bytes[destinationIndex] = source[pixelIndex]
                bytes[destinationIndex + 1] = source[pixelIndex + pixelCount]
                bytes[destinationIndex + 2] = source[pixelIndex + pixelCount * 2]
            }
        }

        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else {
            throw HiDreamO1ImagePreprocessorError.pixelBufferFailed
        }
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw HiDreamO1ImagePreprocessorError.pixelBufferFailed
        }
        return image
    }
    #endif

    private static func prepareVisionPatchInputs(
        imageCHW: MLXArray,
        patchSize: Int,
        temporalPatchSize: Int,
        mergeSize: Int
    ) -> MLXArray {
        let pixelValues = imageCHW.expandedDimensions(axis: 0)
        let batch = pixelValues.dim(0)
        let channels = pixelValues.dim(1)
        let height = pixelValues.dim(2)
        let width = pixelValues.dim(3)

        let patchH = height / patchSize
        let patchW = width / patchSize
        let blockH = patchH / max(1, mergeSize)
        let blockW = patchW / max(1, mergeSize)
        let numPatches = patchH * patchW

        precondition(blockH > 0 && blockW > 0, "Invalid HiDream O1 vision condition grid.")

        var patches = pixelValues.reshaped(
            batch,
            channels,
            blockH,
            mergeSize,
            patchSize,
            blockW,
            mergeSize,
            patchSize
        )
        patches = patches.transposed(0, 2, 5, 3, 6, 1, 4, 7)
        patches = patches.reshaped(batch, numPatches, channels, patchSize * patchSize)

        let repeats = max(1, temporalPatchSize)
        let temporalSlices = (0..<repeats).map { _ in
            patches.expandedDimensions(axis: 3)
        }
        let temporal = temporalSlices.count == 1
            ? temporalSlices[0]
            : MLX.concatenated(temporalSlices, axis: 3)

        return temporal.reshaped(batch, numPatches, channels * repeats * patchSize * patchSize)
    }

    private static func validate(_ resolution: HiDreamO1SampleBuilder.Resolution) throws {
        let patchSize = HiDreamO1SampleBuilder.patchSize
        guard resolution.width > 0,
              resolution.height > 0,
              resolution.width.isMultiple(of: patchSize),
              resolution.height.isMultiple(of: patchSize) else {
            throw HiDreamO1ImagePreprocessorError.invalidSize(
                width: resolution.width,
                height: resolution.height
            )
        }
    }

    private static func validateVisionResolution(
        _ resolution: HiDreamO1SampleBuilder.Resolution,
        patchSize: Int,
        mergeSize: Int
    ) throws {
        let alignment = patchSize * max(1, mergeSize)
        guard resolution.width > 0,
              resolution.height > 0,
              resolution.width.isMultiple(of: alignment),
              resolution.height.isMultiple(of: alignment) else {
            throw HiDreamO1ImagePreprocessorError.invalidSize(
                width: resolution.width,
                height: resolution.height
            )
        }
    }
}
