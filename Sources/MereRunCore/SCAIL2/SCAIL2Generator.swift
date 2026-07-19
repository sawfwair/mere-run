import Foundation
import MediaIO
import MLX
import MLXRandom

public enum SCAIL2GenerationError: LocalizedError, Sendable {
    case invalidResolution(width: Int, height: Int)
    case invalidStepCount(Int)
    case invalidFrameRate(Int)
    case invalidGuidanceScale(Float)
    case invalidScheduleShift(Float)
    case invalidSegment(length: Int, overlap: Int)
    case inputNotFound(URL)
    case inputDecodeFailed(URL)
    case mismatchedDrivingFrames(video: Int, mask: Int)
    case noUsableSegment(frameCount: Int, segmentLength: Int)
    case missingModelFiles([URL])
    case invalidReferenceCount(Int)
    case invalidReferenceMask(URL, colors: [SCAIL2SubjectColor])
    case duplicateReferenceColor(SCAIL2SubjectColor)

    public var errorDescription: String? {
        switch self {
        case .invalidResolution(let width, let height):
            return "SCAIL-2 resolution must be positive and divisible by 32; received \(width)x\(height)."
        case .invalidStepCount(let count):
            return "SCAIL-2 sampling steps must be positive; received \(count)."
        case .invalidFrameRate(let fps):
            return "SCAIL-2 frame rate must be positive; received \(fps)."
        case .invalidGuidanceScale(let scale):
            return "SCAIL-2 guidance scale must be positive; received \(scale)."
        case .invalidScheduleShift(let shift):
            return "SCAIL-2 schedule shift must be positive; received \(shift)."
        case .invalidSegment(let length, let overlap):
            return "SCAIL-2 segment length and overlap must equal 1 modulo 4, with 0 < overlap < length; received length=\(length), overlap=\(overlap)."
        case .inputNotFound(let url):
            return "SCAIL-2 input not found: \(url.path)"
        case .inputDecodeFailed(let url):
            return "SCAIL-2 could not decode input: \(url.path)"
        case .mismatchedDrivingFrames(let video, let mask):
            return "SCAIL-2 driving video and mask must have the same frame count; received \(video) and \(mask)."
        case .noUsableSegment(let frameCount, let segmentLength):
            return "SCAIL-2 could not form a complete \(segmentLength)-frame segment from \(frameCount) driving frames."
        case .missingModelFiles(let urls):
            return "SCAIL-2 model root is missing: \(urls.map(\.lastPathComponent).joined(separator: ", "))."
        case .invalidReferenceCount(let count):
            return "SCAIL-2 requires one to six reference subjects; received \(count)."
        case .invalidReferenceMask(let url, let colors):
            return "SCAIL-2 reference mask \(url.path) must contain exactly one non-background palette color; found \(colors.map(\.rawValue).sorted())."
        case .duplicateReferenceColor(let color):
            return "SCAIL-2 reference masks must use unique palette colors; duplicate: \(color.rawValue)."
        }
    }
}

public struct SCAIL2ReferenceInput: Hashable, Sendable {
    public let imageURL: URL
    public let maskURL: URL

    public init(imageURL: URL, maskURL: URL) {
        self.imageURL = imageURL
        self.maskURL = maskURL
    }
}

public enum SCAIL2TailPolicy: String, Codable, CaseIterable, Hashable, Sendable {
    case drop
    case padTrim = "pad-trim"
}

public struct SCAIL2GenerationOptions: Hashable, Sendable {
    public let prompt: String
    public let negativePrompt: String
    public let reference: SCAIL2ReferenceInput
    public let additionalReferences: [SCAIL2ReferenceInput]
    public let drivingVideoURL: URL
    public let drivingMaskVideoURL: URL
    public let outputURL: URL
    public let mode: SCAIL2Mode
    public let width: Int
    public let height: Int
    public let steps: Int
    public let guidanceScale: Float
    public let shift: Float
    public let seed: UInt64
    public let fps: Int
    public let segmentLength: Int
    public let segmentOverlap: Int
    public let tailPolicy: SCAIL2TailPolicy

    public init(
        prompt: String,
        negativePrompt: String = "",
        reference: SCAIL2ReferenceInput,
        additionalReferences: [SCAIL2ReferenceInput] = [],
        drivingVideoURL: URL,
        drivingMaskVideoURL: URL,
        outputURL: URL,
        mode: SCAIL2Mode = .animation,
        width: Int = 896,
        height: Int = 512,
        steps: Int = 40,
        guidanceScale: Float = 5,
        shift: Float = 3,
        seed: UInt64 = 42,
        fps: Int = 16,
        segmentLength: Int = 81,
        segmentOverlap: Int = 5,
        tailPolicy: SCAIL2TailPolicy = .drop
    ) throws {
        guard width > 0, height > 0, width.isMultiple(of: 32), height.isMultiple(of: 32) else {
            throw SCAIL2GenerationError.invalidResolution(width: width, height: height)
        }
        guard steps > 0 else { throw SCAIL2GenerationError.invalidStepCount(steps) }
        guard fps > 0 else { throw SCAIL2GenerationError.invalidFrameRate(fps) }
        guard guidanceScale > 0 else { throw SCAIL2GenerationError.invalidGuidanceScale(guidanceScale) }
        guard shift > 0 else { throw SCAIL2GenerationError.invalidScheduleShift(shift) }
        guard segmentLength > 0,
              segmentOverlap > 0,
              segmentOverlap < segmentLength,
              (segmentLength - 1).isMultiple(of: 4),
              (segmentOverlap - 1).isMultiple(of: 4) else {
            throw SCAIL2GenerationError.invalidSegment(length: segmentLength, overlap: segmentOverlap)
        }
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.reference = reference
        self.additionalReferences = additionalReferences
        self.drivingVideoURL = drivingVideoURL
        self.drivingMaskVideoURL = drivingMaskVideoURL
        self.outputURL = outputURL
        self.mode = mode
        self.width = width
        self.height = height
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.shift = shift
        self.seed = seed
        self.fps = fps
        self.segmentLength = segmentLength
        self.segmentOverlap = segmentOverlap
        self.tailPolicy = tailPolicy
    }
}

public struct SCAIL2GenerationResult: Hashable, Sendable {
    public let outputURL: URL
    public let seed: UInt64
    public let frameCount: Int
    public let width: Int
    public let height: Int
    public let segmentCount: Int
}

public enum SCAIL2SegmentBuilder {
    public static func build(
        frameCount: Int,
        segmentLength: Int,
        segmentOverlap: Int,
        temporalStride: Int = 4
    ) -> [Range<Int>] {
        precondition(segmentLength > 0 && segmentOverlap > 0 && segmentOverlap < segmentLength)
        precondition(temporalStride > 0)
        guard frameCount > 0 else { return [] }
        if frameCount <= segmentLength {
            let keep = ((frameCount - 1) / temporalStride) * temporalStride + 1
            return [0..<keep]
        }
        let stride = segmentLength - segmentOverlap
        var result: [Range<Int>] = []
        var start = 0
        while start + segmentLength <= frameCount {
            result.append(start..<(start + segmentLength))
            start += stride
        }
        return result
    }

    public static func paddedFrameCount(
        frameCount: Int,
        segmentLength: Int,
        segmentOverlap: Int
    ) -> Int {
        guard frameCount > 0 else { return 0 }
        guard frameCount > segmentLength else { return segmentLength }
        let stride = segmentLength - segmentOverlap
        let remainder = frameCount - segmentLength
        let additionalSegments = (remainder + stride - 1) / stride
        return segmentLength + additionalSegments * stride
    }
}

public final class SCAIL2Generator: @unchecked Sendable {
    private struct DrivingSegment {
        let latent: MLXArray
        let mask: MLXArray
    }

    public init() {}

    public func generate(
        options: SCAIL2GenerationOptions,
        resources: SCAIL2Resources,
        progressHandler: (@Sendable (GenerationProgress) -> Void)? = nil
    ) async throws -> SCAIL2GenerationResult {
        let missingModelFiles = resources.validate()
        guard missingModelFiles.isEmpty else {
            throw SCAIL2GenerationError.missingModelFiles(missingModelFiles)
        }
        try validateInputs(options)
        let configuration = try resources.loadConfiguration()

        let referenceImage = try decodeImage(options.reference.imageURL)
        let dimensions = Self.resolvedDimensions(
            sourceWidth: referenceImage.width,
            sourceHeight: referenceImage.height,
            requestedWidth: options.width,
            requestedHeight: options.height
        )
        let referenceMaskImage = try Self.normalizedMaskForMode(
            try validatedReferenceMask(options.reference.maskURL),
            mode: options.mode,
            role: .mainReference
        )
        let additionalImages = try options.additionalReferences.map {
            (
                try decodeImage($0.imageURL),
                try Self.normalizedMaskForMode(
                    try validatedReferenceMask($0.maskURL),
                    mode: options.mode,
                    role: .additionalSubjectReference
                )
            )
        }
        try validateReferenceColors(
            [referenceMaskImage] + additionalImages.map(\.1),
            urls: [options.reference.maskURL] + options.additionalReferences.map(\.maskURL)
        )

        let temporaryRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-scail2-\(UUID().uuidString)", isDirectory: true)
        let drivingFramesDirectory = temporaryRoot.appendingPathComponent("driving", isDirectory: true)
        let maskFramesDirectory = temporaryRoot.appendingPathComponent("mask", isDirectory: true)
        try FileManager.default.createDirectory(at: drivingFramesDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: maskFramesDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporaryRoot) }

        let drivingSequence = try MediaVideoIO.extractFrames(
            from: options.drivingVideoURL,
            into: drivingFramesDirectory
        )
        let maskSequence = try MediaVideoIO.extractFrames(
            from: options.drivingMaskVideoURL,
            into: maskFramesDirectory
        )
        guard drivingSequence.frameURLs.count == maskSequence.frameURLs.count else {
            throw SCAIL2GenerationError.mismatchedDrivingFrames(
                video: drivingSequence.frameURLs.count,
                mask: maskSequence.frameURLs.count
            )
        }
        let originalFrameCount = drivingSequence.frameURLs.count
        let drivingFrameURLs: [URL]
        let maskFrameURLs: [URL]
        if options.tailPolicy == .padTrim,
           let lastDriving = drivingSequence.frameURLs.last,
           let lastMask = maskSequence.frameURLs.last {
            let paddedCount = SCAIL2SegmentBuilder.paddedFrameCount(
                frameCount: originalFrameCount,
                segmentLength: options.segmentLength,
                segmentOverlap: options.segmentOverlap
            )
            drivingFrameURLs = drivingSequence.frameURLs
                + [URL](repeating: lastDriving, count: paddedCount - originalFrameCount)
            maskFrameURLs = maskSequence.frameURLs
                + [URL](repeating: lastMask, count: paddedCount - originalFrameCount)
        } else {
            drivingFrameURLs = drivingSequence.frameURLs
            maskFrameURLs = maskSequence.frameURLs
        }
        let segments = SCAIL2SegmentBuilder.build(
            frameCount: drivingFrameURLs.count,
            segmentLength: options.segmentLength,
            segmentOverlap: options.segmentOverlap
        )
        guard !segments.isEmpty else {
            throw SCAIL2GenerationError.noUsableSegment(
                frameCount: drivingSequence.frameURLs.count,
                segmentLength: options.segmentLength
            )
        }

        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: options.steps))
        let text = try encodeText(
            options: options,
            resources: resources
        )
        eval(text.prompt, text.negativePrompt)
        Memory.clearCache()

        let referencePixels = SCAIL2InputPreprocessor.centerCroppedTensor(
            image: referenceImage,
            width: dimensions.width,
            height: dimensions.height
        )
        let referenceMaskPixels = SCAIL2InputPreprocessor.centerCroppedTensor(
            image: referenceMaskImage,
            width: dimensions.width,
            height: dimensions.height
        )
        let processedAdditional = additionalImages.map { pair in
            (
                SCAIL2InputPreprocessor.centerCroppedTensor(
                    image: pair.0,
                    width: dimensions.width,
                    height: dimensions.height
                ),
                SCAIL2InputPreprocessor.centerCroppedTensor(
                    image: pair.1,
                    width: dimensions.width,
                    height: dimensions.height
                )
            )
        }

        progressHandler?(GenerationProgress(
            stage: .encodingReferenceImages,
            stepIndex: 0,
            totalSteps: options.steps
        ))
        let imageEmbeddings = try encodeReferenceImage(
            croppedReference: referencePixels,
            resources: resources
        )
        eval(imageEmbeddings)
        Memory.clearCache()

        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: options.steps))
        let vae = try SCAIL2ModelLoader.loadVAE(resources: resources)
        let referenceLatent = encodePixels(referencePixels, with: vae)
        let referenceMask = SCAIL2MaskEncoder.encode(referenceMaskPixels)
        let additionalLatents = processedAdditional.map { encodePixels($0.0, with: vae) }
        let additionalMasks = processedAdditional.map { SCAIL2MaskEncoder.encode($0.1) }
        eval([referenceLatent, referenceMask] + additionalLatents + additionalMasks)

        var drivingSegments: [DrivingSegment] = []
        drivingSegments.reserveCapacity(segments.count)
        for range in segments {
            try Task.checkCancellation()
            let drivingImages = try decodeImages(drivingFrameURLs[range])
            let maskImages = try decodeMaskImages(maskFrameURLs[range]).map {
                try Self.normalizedMaskForMode($0, mode: options.mode, role: .driving)
            }
            let drivingPixels = SCAIL2InputPreprocessor.centerCroppedTensor(
                images: drivingImages,
                width: dimensions.width,
                height: dimensions.height
            )
            let maskPixels = SCAIL2InputPreprocessor.centerCroppedTensor(
                images: maskImages,
                width: dimensions.width,
                height: dimensions.height
            )
            let halfDriving = SCAIL2InputPreprocessor.halfResolutionBilinear(drivingPixels)
            let halfMask = SCAIL2InputPreprocessor.halfResolutionBilinear(maskPixels)
            let latent = encodePixels(halfDriving, with: vae)
            let mask = SCAIL2MaskEncoder.encode(halfMask)
            eval(latent, mask)
            drivingSegments.append(DrivingSegment(latent: latent, mask: mask))
            Memory.clearCache()
        }

        progressHandler?(GenerationProgress(
            stage: .loadingTransformer,
            stepIndex: 0,
            totalSteps: options.steps
        ))
        let transformer = try SCAIL2ModelLoader.loadTransformer(
            resources: resources,
            configuration: configuration
        )
        let prompt = text.prompt.expandedDimensions(axis: 0)
        let negativePrompt = text.negativePrompt.expandedDimensions(axis: 0)
        let positiveConditioning = transformer.prepareConditioning(
            textEmbeddings: prompt,
            imageEmbeddings: imageEmbeddings
        )
        let negativeConditioning = transformer.prepareConditioning(
            textEmbeddings: negativePrompt,
            imageEmbeddings: imageEmbeddings
        )
        eval(positiveConditioning.arrays + negativeConditioning.arrays)
        Memory.clearCache()

        MLXRandom.seed(options.seed)
        var outputSegments: [MLXArray] = []
        var historyPixels: MLXArray?
        for (segmentIndex, driving) in drivingSegments.enumerated() {
            try Task.checkCancellation()
            let historyLatent = historyPixels.map { encodePixels($0, with: vae) }
            if let historyLatent { eval(historyLatent) }
            let latentFrames = driving.latent.dim(1)
            let latentHeight = referenceLatent.dim(2)
            let latentWidth = referenceLatent.dim(3)
            let historyMask: MLXArray? = historyLatent.map {
                let historyFrames = min($0.dim(1), latentFrames)
                return MLX.concatenated([
                    MLX.ones([4, historyFrames, latentHeight, latentWidth]),
                    MLX.zeros([4, latentFrames - historyFrames, latentHeight, latentWidth]),
                ], axis: 1)
            }
            let noise = MLXRandom.normal([
                configuration.vaeLatentChannels, latentFrames, latentHeight, latentWidth,
            ]).asType(.float32)
            var latent = Self.applyCleanHistory(noise, history: historyLatent)
            eval(latent)
            var scheduler = Wan2UniPCScheduler(steps: options.steps, shift: options.shift)
            for (stepIndex, timestep) in scheduler.timesteps.enumerated() {
                try Task.checkCancellation()
                progressHandler?(GenerationProgress(
                    stage: .denoising,
                    stepIndex: stepIndex,
                    totalSteps: options.steps
                ))
                let cleanInput = Self.applyCleanHistory(latent, history: historyLatent)
                let modelInput = SCAIL2TransformerInput(
                    videoLatent: cleanInput,
                    referenceLatent: referenceLatent,
                    referenceMask: referenceMask,
                    drivingLatent: driving.latent,
                    drivingMask: driving.mask,
                    historyMask: historyMask,
                    additionalReferenceLatents: additionalLatents,
                    additionalReferenceMasks: additionalMasks,
                    textEmbeddings: prompt,
                    imageEmbeddings: imageEmbeddings,
                    timestep: MLXArray(timestep),
                    mode: options.mode
                )
                let conditional = transformer(modelInput, conditioning: positiveConditioning)
                eval(conditional)
                let velocity: MLXArray
                if options.guidanceScale <= 1 {
                    velocity = conditional
                } else {
                    let unconditional = transformer(modelInput, conditioning: negativeConditioning)
                    velocity = unconditional + options.guidanceScale * (conditional - unconditional)
                }
                latent = scheduler.step(modelOutput: velocity, sample: latent)
                latent = Self.applyCleanHistory(latent, history: historyLatent)
                eval(latent)
                Memory.clearCache()
            }

            progressHandler?(GenerationProgress(
                stage: .decoding,
                stepIndex: options.steps,
                totalSteps: options.steps
            ))
            let decoded = decodeLatent(latent, with: vae)
            eval(decoded)
            outputSegments.append(
                segmentIndex == 0 ? decoded : decoded[0..., options.segmentOverlap...]
            )
            if segmentIndex + 1 < drivingSegments.count {
                historyPixels = Self.historyPixels(
                    from: decoded,
                    overlap: options.segmentOverlap
                )
                eval(historyPixels!)
            }
            Memory.clearCache()
        }

        let combinedPixels = MLX.concatenated(outputSegments, axis: 1)
        let pixels = options.tailPolicy == .padTrim && combinedPixels.dim(1) > originalFrameCount
            ? combinedPixels[0..., 0..<originalFrameCount, 0..., 0..., 0...]
            : combinedPixels
        let frames = MLX.clip((pixels + 1) * 127.5, min: 0, max: 255).asType(.uint8)
        eval(frames)
        progressHandler?(GenerationProgress(stage: .saving, stepIndex: options.steps, totalSteps: options.steps))
        try LTXVideoMP4Writer.writeMP4(frames: frames, fps: options.fps, to: options.outputURL)
        return SCAIL2GenerationResult(
            outputURL: options.outputURL,
            seed: options.seed,
            frameCount: frames.dim(1),
            width: dimensions.width,
            height: dimensions.height,
            segmentCount: segments.count
        )
    }

    static func resolvedDimensions(
        sourceWidth: Int,
        sourceHeight: Int,
        requestedWidth: Int,
        requestedHeight: Int
    ) -> (width: Int, height: Int) {
        let mismatchedOrientation =
            (sourceHeight < sourceWidth && requestedHeight > requestedWidth)
            || (sourceHeight > sourceWidth && requestedHeight < requestedWidth)
        return mismatchedOrientation
            ? (requestedHeight, requestedWidth)
            : (requestedWidth, requestedHeight)
    }

    static func applyCleanHistory(_ latent: MLXArray, history: MLXArray?) -> MLXArray {
        guard let history else { return latent }
        let historyFrames = min(history.dim(1), latent.dim(1))
        return MLX.concatenated([
            history[0..., 0..<historyFrames],
            latent[0..., historyFrames...],
        ], axis: 1)
    }

    static func historyPixels(from decoded: MLXArray, overlap: Int) -> MLXArray {
        precondition(decoded.ndim == 5 && decoded.dim(0) == 1)
        precondition(overlap > 0 && overlap <= decoded.dim(1))
        return decoded[
            0,
            (decoded.dim(1) - overlap)...,
            0...,
            0...,
            0...
        ]
    }

    static func normalizedMaskForMode(
        _ image: MediaImage,
        mode: SCAIL2Mode,
        role: SCAIL2MaskRole
    ) throws -> MediaImage {
        var rgba = image.rgba8
        let background = role.background(mode: mode)
        for pixelIndex in 0..<(image.width * image.height) {
            let offset = pixelIndex * 4
            let rgb = (rgba[offset], rgba[offset + 1], rgba[offset + 2])
            guard rgb == (255, 255, 255) || rgb == (0, 0, 0) else { continue }
            rgba[offset] = background.0
            rgba[offset + 1] = background.1
            rgba[offset + 2] = background.2
            rgba[offset + 3] = background.3
        }
        return try MediaImage(width: image.width, height: image.height, rgba8: rgba)
    }

    private func validateInputs(_ options: SCAIL2GenerationOptions) throws {
        let referenceCount = 1 + options.additionalReferences.count
        guard (1...6).contains(referenceCount) else {
            throw SCAIL2GenerationError.invalidReferenceCount(referenceCount)
        }
        let urls = [
            options.reference.imageURL,
            options.reference.maskURL,
            options.drivingVideoURL,
            options.drivingMaskVideoURL,
        ] + options.additionalReferences.flatMap { [$0.imageURL, $0.maskURL] }
        for url in urls where !FileManager.default.fileExists(atPath: url.path) {
            throw SCAIL2GenerationError.inputNotFound(url)
        }
    }

    private func decodeImage(_ url: URL) throws -> MediaImage {
        do {
            return try MediaImageIO.decode(url)
        } catch {
            throw SCAIL2GenerationError.inputDecodeFailed(url)
        }
    }

    private func decodeImages(_ urls: ArraySlice<URL>) throws -> [MediaImage] {
        try urls.map(decodeImage)
    }

    private func decodeMaskImages(_ urls: ArraySlice<URL>) throws -> [MediaImage] {
        try urls.map {
            try SCAIL2Palette.snapped(
                try decodeImage($0),
                tolerance: SCAIL2Palette.codecTolerance
            )
        }
    }

    private func validatedReferenceMask(_ url: URL) throws -> MediaImage {
        let image = try decodeImage(url)
        let colors = try SCAIL2Palette.subjectColors(
            in: image,
            tolerance: SCAIL2Palette.codecTolerance
        )
        guard colors.count == 1 else {
            throw SCAIL2GenerationError.invalidReferenceMask(
                url,
                colors: colors.sorted { $0.rawValue < $1.rawValue }
            )
        }
        return try SCAIL2Palette.snapped(
            image,
            tolerance: SCAIL2Palette.codecTolerance
        )
    }

    private func validateReferenceColors(
        _ images: [MediaImage],
        urls: [URL]
    ) throws {
        var colors = Set<SCAIL2SubjectColor>()
        for (image, url) in zip(images, urls) {
            let maskColors = try SCAIL2Palette.subjectColors(in: image, tolerance: 0)
            guard let color = maskColors.first, maskColors.count == 1 else {
                throw SCAIL2GenerationError.invalidReferenceMask(
                    url,
                    colors: maskColors.sorted { $0.rawValue < $1.rawValue }
                )
            }
            guard colors.insert(color).inserted else {
                throw SCAIL2GenerationError.duplicateReferenceColor(color)
            }
        }
    }

    private func encodeText(
        options: SCAIL2GenerationOptions,
        resources: SCAIL2Resources
    ) throws -> SCAIL2TextConditioning {
        let tokenizer = try SCAIL2ModelLoader.loadTokenizer(resources: resources)
        let encoder = try SCAIL2ModelLoader.loadTextEncoder(resources: resources)
        return try SCAIL2ModelLoader.encodePrompts(
            tokenizer: tokenizer,
            encoder: encoder,
            prompt: options.prompt,
            negativePrompt: options.negativePrompt
        )
    }

    private func encodeReferenceImage(
        croppedReference: MLXArray,
        resources: SCAIL2Resources
    ) throws -> MLXArray {
        let clip = try SCAIL2ModelLoader.loadCLIP(resources: resources)
        return clip(SCAIL2CLIPPreprocessor.normalizedNHWC(croppedImages: croppedReference))
    }

    private func encodePixels(_ pixels: MLXArray, with vae: Wan2VAEModel) -> MLXArray {
        vae.encodeVideo(pixels.expandedDimensions(axis: 0))[0]
            .transposed(3, 0, 1, 2)
            .asType(.float32)
    }

    private func decodeLatent(_ latent: MLXArray, with vae: Wan2VAEModel) -> MLXArray {
        vae.decode(latent.transposed(1, 2, 3, 0).expandedDimensions(axis: 0))
    }
}
