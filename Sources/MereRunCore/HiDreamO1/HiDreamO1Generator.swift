import Foundation
import MLX

public enum HiDreamO1GeneratorError: LocalizedError, Sendable {
    case missingModelFiles([URL])

    public var errorDescription: String? {
        switch self {
        case .missingModelFiles(let urls):
            return "Missing HiDream O1 model files: \(urls.map(\.path).joined(separator: ", "))"
        }
    }
}

public final class HiDreamO1Generator: ImageGenerator {
    public init() {}

    public func generate(
        _ request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 0, totalSteps: 1))
        let rootURL = try resolveModelRoot(request)
        let resources = HiDreamO1Resources(rootURL: rootURL)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw HiDreamO1GeneratorError.missingModelFiles(missing)
        }

        let config = try HiDreamO1Config.load(from: resources)
        let tokenizer = try await HiDreamO1TokenizerAndTemplate.load(from: resources)
        let manifest = try MereRunModelManifest.loadRequired(from: rootURL)
        let variant = manifest.variant ?? .distilled
        let shift: Float = variant == .distilled ? 1.0 : 3.0
        let scheduler = HiDreamO1Scheduler(steps: request.steps, variant: variant, shift: shift)

        let references = effectiveReferenceImages(for: request)
        let referenceOriginalSizes = try references.map { try HiDreamO1ImagePreprocessor.imageSize($0) }
        let targetResolution = HiDreamO1SampleBuilder.targetResolution(
            width: request.width,
            height: request.height,
            referenceOriginalSizes: referenceOriginalSizes,
            keepOriginalAspect: request.keepOriginalAspect
        )
        let referenceSizes = resizedReferenceSizes(
            originalSizes: referenceOriginalSizes,
            targetResolution: targetResolution,
            keepOriginalAspect: request.keepOriginalAspect
        )
        let referenceTensors = try zip(references, referenceSizes).map { url, size in
            try HiDreamO1ImagePreprocessor.patchTensor(from: url, resolution: size)
        }
        let conditionSize = HiDreamO1SampleBuilder.conditionSize(referenceCount: references.count)
        let visionConditionSizes = referenceSizes.map {
            HiDreamO1SampleBuilder.vlmConditionSize(
                originalWidth: $0.width,
                originalHeight: $0.height,
                maxSize: conditionSize
            )
        }
        let visionConditions = try zip(references, visionConditionSizes).map { url, size in
            try HiDreamO1ImagePreprocessor.visionConditionTensor(
                from: url,
                resolution: size,
                config: config
            )
        }
        let textTokenIds = references.isEmpty
            ? try tokenizer.encodeTextToImagePrompt(request.prompt)
            : try tokenizer.encodeReferencePrompt(
                request.prompt,
                referenceImageTokenCounts: visionConditions.map(\.tokenCount)
            )
        let usesCFG = request.guidanceScale > 1.0
        let unconditionalPrompt = request.negativePrompt ?? " "
        let unconditionalTextTokenIds: [Int]?
        if usesCFG {
            unconditionalTextTokenIds = references.isEmpty
                ? try tokenizer.encodeTextToImagePrompt(unconditionalPrompt)
                : try tokenizer.encodeReferencePrompt(
                    unconditionalPrompt,
                    referenceImageTokenCounts: visionConditions.map(\.tokenCount)
                )
        } else {
            unconditionalTextTokenIds = nil
        }
        let sample = references.isEmpty
            ? HiDreamO1SampleBuilder.textToImageSample(
                textTokenIds: textTokenIds,
                height: targetResolution.height,
                width: targetResolution.width,
                config: config
            )
            : HiDreamO1SampleBuilder.referenceSample(
                textTokenIds: textTokenIds,
                targetHeight: targetResolution.height,
                targetWidth: targetResolution.width,
                referenceSizes: referenceSizes,
                conditionGrids: visionConditions.map(\.mergedGrid),
                config: config
            )
        let unconditionalSample = unconditionalTextTokenIds.map { tokenIds in
            references.isEmpty
                ? HiDreamO1SampleBuilder.textToImageSample(
                    textTokenIds: tokenIds,
                    height: targetResolution.height,
                    width: targetResolution.width,
                    config: config
                )
                : HiDreamO1SampleBuilder.referenceSample(
                    textTokenIds: tokenIds,
                    targetHeight: targetResolution.height,
                    targetWidth: targetResolution.width,
                    referenceSizes: referenceSizes,
                    conditionGrids: visionConditions.map(\.mergedGrid),
                    config: config
                )
        }
        let conditioning = HiDreamO1Denoiser.Conditioning(
            sample: sample,
            visionConditions: visionConditions
        )
        let unconditionalConditioning = unconditionalSample.map {
            HiDreamO1Denoiser.Conditioning(sample: $0, visionConditions: visionConditions)
        }

        let seed = request.seed ?? deterministicSeed(prompt: request.prompt)
        let model = try HiDreamO1ModelLoader.load(
            resources: resources,
            config: config,
            progressHandler: progressHandler
        )
        let image: MLXArray
        if references.isEmpty {
            let patches = try HiDreamO1Denoiser.runTextOnly(
                model: model,
                conditioning: conditioning,
                unconditionalConditioning: unconditionalConditioning,
                scheduler: scheduler,
                height: targetResolution.height,
                width: targetResolution.width,
                seed: seed,
                guidanceScale: Float(request.guidanceScale),
                progressHandler: progressHandler
            )
            image = HiDreamO1SampleBuilder.unpatchifyCHW(
                patches,
                height: targetResolution.height,
                width: targetResolution.width
            )
        } else {
            let patches = try HiDreamO1Denoiser.runWithReferences(
                model: model,
                conditioning: conditioning,
                unconditionalConditioning: unconditionalConditioning,
                scheduler: scheduler,
                height: targetResolution.height,
                width: targetResolution.width,
                seed: seed,
                referenceTensors: referenceTensors,
                guidanceScale: Float(request.guidanceScale),
                progressHandler: progressHandler
            )
            image = HiDreamO1SampleBuilder.unpatchifyCHW(
                patches,
                height: targetResolution.height,
                width: targetResolution.width
            )
        }
        progressHandler?(GenerationProgress(stage: .saving, stepIndex: 0, totalSteps: 1))
        try ensureOutputDirectory(request.outputURL)
        try HiDreamO1ImagePreprocessor.saveNormalizedCHW(image, to: request.outputURL)
        progressHandler?(GenerationProgress(stage: .saving, stepIndex: 1, totalSteps: 1))
        return GenerationResult(outputURL: request.outputURL, seed: seed)
    }

    private func resolveModelRoot(_ request: GenerationRequest) throws -> URL {
        if let model = request.model {
            let url = URL(fileURLWithPath: model).standardizedFileURL
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        if let resolved = ManagedModelResolver.resolveInstalledModel(id: ModelResolver.ModelID.hidreamO1Dev.rawValue) {
            return resolved
        }
        throw ModelResolver.ResolverError.modelNotFound(
            .hidreamO1Dev,
            searched: [MereRunModelPaths.modelDir(ModelResolver.ModelID.hidreamO1Dev.rawValue)],
            upstreamRepoId: "HiDream-ai/HiDream-O1-Image-Dev"
        )
    }

    private func effectiveReferenceImages(for request: GenerationRequest) -> [URL] {
        if !request.referenceImages.isEmpty {
            return request.referenceImages
        }
        if let inputImage = request.inputImage {
            return [inputImage]
        }
        return []
    }

    private func ensureOutputDirectory(_ outputURL: URL) throws {
        let directory = outputURL.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }

    private func deterministicSeed(prompt: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in prompt.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x0000_0100_0000_01b3
        }
        return hash
    }

    private func resizedReferenceSizes(
        originalSizes: [HiDreamO1SampleBuilder.Resolution],
        targetResolution: HiDreamO1SampleBuilder.Resolution,
        keepOriginalAspect: Bool
    ) -> [HiDreamO1SampleBuilder.Resolution] {
        if keepOriginalAspect, originalSizes.count == 1 {
            return [targetResolution]
        }
        let maxSize = HiDreamO1SampleBuilder.referenceMaxSize(
            targetWidth: targetResolution.width,
            targetHeight: targetResolution.height,
            referenceCount: originalSizes.count
        )
        return originalSizes.map {
            HiDreamO1SampleBuilder.resizedReferenceSize(
                originalWidth: $0.width,
                originalHeight: $0.height,
                maxSize: maxSize
            )
        }
    }
}
