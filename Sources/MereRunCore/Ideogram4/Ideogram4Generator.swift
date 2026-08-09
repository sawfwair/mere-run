import Foundation
import MLX
import MLXRandom
@preconcurrency import Tokenizers

public enum Ideogram4GeneratorError: LocalizedError, Sendable {
    case missingModelFiles([URL])
    case unsupportedMode(String)
    case modelsNotLoaded

    public var errorDescription: String? {
        switch self {
        case .missingModelFiles(let urls):
            return "Missing Ideogram 4 model files: \(urls.map(\.path).joined(separator: ", "))"
        case .unsupportedMode(let mode):
            return "Ideogram 4 SDNQ native generation does not support \(mode) yet."
        case .modelsNotLoaded:
            return "Ideogram 4 models were not loaded."
        }
    }
}

public final class Ideogram4Generator: ImageGenerator {
    private var loadedModelPath: String?
    private var conditionalTransformer: Ideogram4Transformer?
    private var unconditionalTransformer: Ideogram4Transformer?
    private var textEncoder: QwenEncoder?
    private var tokenizer: QwenTokenizer?
    private var vae: AutoencoderKL?

    public init() {}

    deinit {
        unload()
    }

    public func unload() {
        loadedModelPath = nil
        conditionalTransformer = nil
        unconditionalTransformer = nil
        textEncoder = nil
        tokenizer = nil
        vae = nil
        clearGPUMemory()
    }

    public func generate(
        _ request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        defer {
            clearGPUMemory()
        }

        guard request.referenceImages.isEmpty else {
            throw Ideogram4GeneratorError.unsupportedMode("reference images")
        }
        guard request.inputImage == nil else {
            throw Ideogram4GeneratorError.unsupportedMode("image-to-image")
        }
        guard request.lora == nil else {
            throw Ideogram4GeneratorError.unsupportedMode("LoRA")
        }

        let rootURL = try resolveModelRoot(request)
        let resources = Ideogram4Resources(rootURL: rootURL)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw Ideogram4GeneratorError.missingModelFiles(missing)
        }

        if loadedModelPath != rootURL.path || conditionalTransformer == nil {
            do {
                unload()
                try loadModels(from: resources, progressHandler: progressHandler)
            } catch {
                unload()
                throw error
            }
            loadedModelPath = rootURL.path
        }

        guard let conditionalTransformer,
              let unconditionalTransformer,
              let textEncoder,
              let tokenizer,
              let vae else {
            throw Ideogram4GeneratorError.modelsNotLoaded
        }

        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1))
        let tokenIds = try encodePrompt(request.prompt, tokenizer: tokenizer, maxLength: min(request.maxSequenceLength, 2_048))
        let inputIds = MLXArray(tokenIds.map(Int32.init)).reshaped([1, tokenIds.count])
        let attentionMask = MLXArray([Int32](repeating: 1, count: tokenIds.count)).reshaped([1, tokenIds.count])
        let llmFeatures = Ideogram4TextFeatures.encode(
            inputIds: inputIds,
            attentionMask: attentionMask,
            using: textEncoder
        ).asType(.bfloat16)
        eval(llmFeatures)
        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 1, totalSteps: 1))

        let seed = request.seed ?? deterministicSeed(prompt: request.prompt)
        let gridSample = try Ideogram4SampleBuilder.textToImageSample(
            llmFeatures: llmFeatures,
            imageWidth: request.width,
            imageHeight: request.height
        )
        let imageTokenCount = gridSample.imageTokenCount
        var z = MLXRandom.normal(
            [1, imageTokenCount, conditionalTransformer.configuration.inChannels],
            key: MLXRandom.key(seed)
        ).asType(.float32)

        let scheduler = Ideogram4Scheduler.preset(
            steps: request.steps,
            width: request.width,
            height: request.height,
            guidanceScale: request.guidanceScale
        )

        let imageStart = gridSample.textTokenCount
        let imageEnd = imageStart + gridSample.imageTokenCount
        let negativePositionIds = gridSample.positionIds[0..., 0..., imageStart..<imageEnd]
        let negativeSegmentIds = gridSample.segmentIds[0..., imageStart..<imageEnd]
        let negativeIndicator = gridSample.indicator[0..., imageStart..<imageEnd]
        let negativeFeatures = MLX.zeros(
            [1, gridSample.imageTokenCount, llmFeatures.dim(2)],
            dtype: llmFeatures.dtype
        )

        for step in stride(from: request.steps - 1, through: 0, by: -1) {
            try Task.checkCancellation()
            let completed = request.steps - 1 - step
            conditionalTransformer.beginDenoisingStep(index: completed, count: request.steps)
            unconditionalTransformer.beginDenoisingStep(index: completed, count: request.steps)
            progressHandler?(GenerationProgress(stage: .denoising, stepIndex: completed, totalSteps: request.steps))

            let tValue = scheduler.value(at: scheduler.interval(step + 1))
            let sValue = scheduler.value(at: scheduler.interval(step))
            let timestep = MLXArray([tValue])
            let modelLatents = z.asType(.bfloat16)

            let positiveSample = try Ideogram4SampleBuilder.pack(
                llmFeatures: llmFeatures,
                imageLatents: modelLatents,
                imageWidth: request.width,
                imageHeight: request.height
            )
            let positiveOutput = conditionalTransformer(sample: positiveSample, timestep: timestep)
            let positiveVelocity = Ideogram4SampleBuilder.imageTokens(from: positiveOutput, sample: positiveSample)

            let negativeVelocity = unconditionalTransformer(
                llmFeatures: negativeFeatures,
                x: modelLatents,
                timestep: timestep,
                positionIds: negativePositionIds,
                segmentIds: negativeSegmentIds,
                indicator: negativeIndicator,
                segmentsAreUniform: true
            )

            let guidance = MLXArray(scheduler.guidanceSchedule[step])
            let velocity = guidance * positiveVelocity + (MLXArray(1.0) - guidance) * negativeVelocity
            z = z + velocity * MLXArray(sValue - tValue)
            eval(z)
        }
        progressHandler?(GenerationProgress(stage: .denoising, stepIndex: request.steps, totalSteps: request.steps))

        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 0, totalSteps: 1))
        let decoded = decodeLatents(
            z,
            gridHeight: gridSample.imageTokenHeight,
            gridWidth: gridSample.imageTokenWidth,
            vae: vae
        )
        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 1, totalSteps: 1))

        progressHandler?(GenerationProgress(stage: .saving, stepIndex: 0, totalSteps: 1))
        try ensureOutputDirectory(request.outputURL)
        try QwenImageIO.saveImage(array: decoded, to: request.outputURL)
        progressHandler?(GenerationProgress(stage: .saving, stepIndex: 1, totalSteps: 1))
        return GenerationResult(outputURL: request.outputURL, seed: seed)
    }

    private func loadModels(
        from resources: Ideogram4Resources,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws {
        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 2))
        conditionalTransformer = try Ideogram4ModelLoader.loadConditionalTransformer(from: resources)
        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 1, totalSteps: 2))
        unconditionalTransformer = try Ideogram4ModelLoader.loadUnconditionalTransformer(from: resources)
        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 2, totalSteps: 2))

        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 0, totalSteps: 2))
        tokenizer = try QwenTokenizer.load(from: resources.tokenizerURL, maxLengthOverride: 2_048)
        textEncoder = try Ideogram4ModelLoader.loadTextEncoder(
            from: resources,
            progressHandler: { progress in
                progressHandler?(GenerationProgress(
                    stage: .loadingEncoder,
                    stepIndex: progress.shardIndex + 1,
                    totalSteps: max(progress.shardCount, 1)
                ))
            }
        )
        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 2, totalSteps: 2))

        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        vae = try Ideogram4ModelLoader.loadVAE(from: resources)
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 1, totalSteps: 1))
    }

    private func encodePrompt(_ prompt: String, tokenizer: QwenTokenizer, maxLength: Int) throws -> [Int] {
        let messages: [Message] = [
            ["role": "user", "content": prompt]
        ]
        return try tokenizer.encodeChatTemplate(
            messages: messages,
            addGenerationPrompt: true,
            includeThinking: false,
            maxLength: maxLength
        )
    }

    private func decodeLatents(
        _ latents: MLXArray,
        gridHeight: Int,
        gridWidth: Int,
        vae: AutoencoderKL
    ) -> MLXArray {
        let patch = Ideogram4SampleBuilder.latentPatchSize
        let channels = Ideogram4SampleBuilder.latentChannels
        var z = latents * Ideogram4LatentNorm.scaleTensor(dtype: latents.dtype)
            + Ideogram4LatentNorm.shiftTensor(dtype: latents.dtype)
        z = z.reshaped([1, gridHeight, gridWidth, patch, patch, channels])
        z = z.transposed(0, 5, 1, 3, 2, 4)
            .reshaped([1, channels, gridHeight * patch, gridWidth * patch])
            .asType(.bfloat16)
        var decoded = vae.decode(z).0
        decoded = MLX.clip(decoded.asType(.float32), min: -1.0, max: 1.0)
        return MLX.clip(QwenImageIO.denormalizeFromDecoder(decoded), min: 0, max: 1)
    }

    private func resolveModelRoot(_ request: GenerationRequest) throws -> URL {
        if let model = request.model {
            let url = URL(fileURLWithPath: model).standardizedFileURL
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        if let resolved = ManagedModelResolver.resolveInstalledModel(id: ModelResolver.ModelID.ideogram4SDNQUInt4.rawValue) {
            return resolved
        }
        throw ModelResolver.ResolverError.modelNotFound(
            .ideogram4SDNQUInt4,
            searched: [MereRunModelPaths.modelDir(ModelResolver.ModelID.ideogram4SDNQUInt4.rawValue)],
            upstreamRepoId: Ideogram4Resources.upstreamRepoId
        )
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

    private func clearGPUMemory(synchronize: Bool = true) {
        if synchronize {
            Stream.gpu.synchronize()
        }
        MLX.eval(MLXArray([]))
        Memory.clearCache()
    }
}
