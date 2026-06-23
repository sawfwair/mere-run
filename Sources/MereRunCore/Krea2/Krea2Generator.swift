import Foundation
import MLX
import MLXRandom

public enum Krea2GeneratorError: LocalizedError, Sendable {
    case missingModelFiles([URL])
    case unsupportedMode(String)
    case modelsNotLoaded

    public var errorDescription: String? {
        switch self {
        case .missingModelFiles(let urls):
            return "Missing Krea 2 model files: \(urls.map(\.path).joined(separator: ", "))"
        case .unsupportedMode(let mode):
            return "Krea 2 Turbo native generation does not support \(mode) yet."
        case .modelsNotLoaded:
            return "Krea 2 models were not loaded."
        }
    }
}

public final class Krea2Generator: ImageGenerator {
    private var loadedModelPath: String?
    private var configs: Krea2ModelConfigs?
    private var transformer: Krea2Transformer?
    private var textEncoder: QwenEncoder?
    private var tokenizer: QwenTokenizer?
    private var vae: QwenImageEditVAE?

    public init() {}

    deinit {
        unload()
    }

    public func unload() {
        loadedModelPath = nil
        configs = nil
        transformer = nil
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
            throw Krea2GeneratorError.unsupportedMode("reference images")
        }
        guard request.inputImage == nil else {
            throw Krea2GeneratorError.unsupportedMode("image-to-image")
        }
        guard request.lora == nil else {
            throw Krea2GeneratorError.unsupportedMode("LoRA")
        }

        let rootURL = try resolveModelRoot(request)
        let resources = Krea2Resources(rootURL: rootURL)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw Krea2GeneratorError.missingModelFiles(missing)
        }

        if loadedModelPath != rootURL.path || transformer == nil {
            do {
                unload()
                try loadModels(from: resources, progressHandler: progressHandler)
            } catch {
                unload()
                throw error
            }
            loadedModelPath = rootURL.path
        }

        guard let configs, let transformer, let textEncoder, let tokenizer, let vae else {
            throw Krea2GeneratorError.modelsNotLoaded
        }

        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1))
        let text = try encodePrompt(
            request.prompt,
            tokenizer: tokenizer,
            encoder: textEncoder,
            selectedLayers: configs.selectedTextLayers,
            maxLength: min(request.maxSequenceLength, 512)
        )
        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 1, totalSteps: 1))

        let seed = request.seed ?? deterministicSeed(prompt: request.prompt)
        let aligned = Krea2SampleBuilder.alignedResolution(width: request.width, height: request.height)
        var latents = MLXRandom.normal(
            [
                1,
                configs.transformer.latentChannels,
                aligned.height / Krea2SampleBuilder.vaeCompression,
                aligned.width / Krea2SampleBuilder.vaeCompression,
            ],
            key: MLXRandom.key(seed)
        ).asType(.bfloat16)

        let prepared = Krea2SampleBuilder.prepare(
            latents: latents,
            textLength: text.hiddenStates.dim(1),
            textMask: text.attentionMask,
            patch: configs.transformer.patchSize
        )
        var imageTokens = prepared.imageTokens.asType(.float32)
        let mu = request.sigmaShift ?? Krea2SampleBuilder.defaultMu
        let timesteps = Krea2SampleBuilder.timesteps(
            imageTokenCount: prepared.imageTokenCount,
            steps: request.steps,
            mu: mu
        )

        for step in 0..<request.steps {
            try Task.checkCancellation()
            progressHandler?(GenerationProgress(stage: .denoising, stepIndex: step, totalSteps: request.steps))
            let tCurrent = timesteps[step]
            let tPrevious = timesteps[step + 1]
            let timestep = MLXArray([tCurrent]).asType(.bfloat16)
            let velocity = transformer(
                imageTokens: imageTokens.asType(.bfloat16),
                textContext: text.hiddenStates.asType(.bfloat16),
                timestep: timestep,
                positionIds: prepared.positionIds,
                validMask: prepared.validMask
            )
            imageTokens = imageTokens + velocity.asType(.float32) * MLXArray(tPrevious - tCurrent)
            MLX.eval(imageTokens)
        }
        progressHandler?(GenerationProgress(stage: .denoising, stepIndex: request.steps, totalSteps: request.steps))

        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 0, totalSteps: 1))
        latents = Krea2SampleBuilder.unpatchify(
            imageTokens.asType(.bfloat16),
            tokenHeight: prepared.imageTokenHeight,
            tokenWidth: prepared.imageTokenWidth,
            channels: configs.transformer.latentChannels,
            patch: configs.transformer.patchSize
        )
        let decoded = decodeLatents(latents, vae: vae, config: configs.vae, height: request.height, width: request.width)
        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 1, totalSteps: 1))

        progressHandler?(GenerationProgress(stage: .saving, stepIndex: 0, totalSteps: 1))
        try ensureOutputDirectory(request.outputURL)
        try QwenImageIO.saveImage(array: decoded, to: request.outputURL)
        progressHandler?(GenerationProgress(stage: .saving, stepIndex: 1, totalSteps: 1))
        return GenerationResult(outputURL: request.outputURL, seed: seed)
    }

    private func loadModels(
        from resources: Krea2Resources,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws {
        let loadedConfigs = try Krea2ModelConfigs.load(from: resources)
        configs = loadedConfigs

        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 3))
        transformer = try Krea2ModelLoader.loadTransformer(
            from: resources,
            configuration: loadedConfigs.transformer,
            progressHandler: { shard in
                progressHandler?(GenerationProgress(
                    stage: .loadingTransformer,
                    stepIndex: shard.shardIndex + 1,
                    totalSteps: max(shard.shardCount, 1)
                ))
            }
        )
        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 3, totalSteps: 3))

        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 0, totalSteps: 2))
        tokenizer = try QwenTokenizer.load(from: resources.tokenizerURL, maxLengthOverride: 512)
        textEncoder = try Krea2ModelLoader.loadTextEncoder(
            from: resources,
            configuration: loadedConfigs.textEncoder
        )
        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 2, totalSteps: 2))

        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        vae = try Krea2ModelLoader.loadVAE(from: resources, configuration: loadedConfigs.vae)
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 1, totalSteps: 1))
    }

    private func encodePrompt(
        _ prompt: String,
        tokenizer: QwenTokenizer,
        encoder: QwenEncoder,
        selectedLayers: [Int],
        maxLength: Int
    ) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray) {
        let prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        let suffix = "<|im_end|>\n<|im_start|>assistant\n"
        let prefixDropCount = 34
        let suffixIds = tokenizer.encodeText(suffix)
        let inputs = Krea2SampleBuilder.paddedTextTokenInputs(
            promptTokenIds: tokenizer.encodeText(prefix + prompt),
            suffixTokenIds: suffixIds,
            padTokenId: tokenizer.padTokenId,
            maxLength: maxLength,
            prefixDropCount: prefixDropCount
        )
        let tokenIds = inputs.tokenIds
        let inputIds = MLXArray(tokenIds.map(Int32.init)).reshaped(1, tokenIds.count)
        let attentionMask = MLXArray(inputs.attentionMask).reshaped(1, inputs.attentionMask.count)
        let hiddenStates = encoder.forwardActivationHiddenStates(
            inputIds: inputIds,
            attentionMask: attentionMask,
            activationLayers: Krea2SampleBuilder.qwenActivationLayerIndices(from: selectedLayers)
        )
        let stacked = MLX.stacked(hiddenStates, axis: 2).asType(.bfloat16)
        let croppedStates = stacked[0..., prefixDropCount..., 0..., 0...]
        let croppedMask = attentionMask[0..., prefixDropCount...]
        MLX.eval(croppedStates, croppedMask)
        return (croppedStates, croppedMask)
    }

    private func decodeLatents(
        _ latents: MLXArray,
        vae: QwenImageEditVAE,
        config: QwenImageEditVAEConfig,
        height: Int,
        width: Int
    ) -> MLXArray {
        var z = latents
        if let mean = config.latentsMean, let std = config.latentsStd {
            let meanTensor = MLXArray(mean).reshaped(1, mean.count, 1, 1).asType(z.dtype)
            let stdTensor = MLXArray(std).reshaped(1, std.count, 1, 1).asType(z.dtype)
            z = z * stdTensor + meanTensor
        }
        var vaeInput = z
        if let shift = config.shiftFactor, shift != 0 {
            vaeInput = vaeInput - MLXArray(shift).asType(vaeInput.dtype)
        }
        vaeInput = vaeInput * MLXArray(config.scalingFactor).asType(vaeInput.dtype)
        var decoded = vae.decode(vaeInput.asType(.bfloat16))
        decoded = MLX.clip(decoded.asType(.float32), min: -1.0, max: 1.0)
        var image = QwenImageIO.denormalizeFromDecoder(decoded)
        if image.dim(2) != height || image.dim(3) != width {
            image = image[0..., 0..., 0..<height, 0..<width]
        }
        return MLX.clip(image, min: 0, max: 1)
    }

    private func resolveModelRoot(_ request: GenerationRequest) throws -> URL {
        if let model = request.model {
            let url = URL(fileURLWithPath: model).standardizedFileURL
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        if let resolved = ManagedModelResolver.resolveInstalledModel(id: ModelResolver.ModelID.krea2Turbo.rawValue) {
            return resolved
        }
        throw ModelResolver.ResolverError.modelNotFound(
            .krea2Turbo,
            searched: [MereRunModelPaths.modelDir(ModelResolver.ModelID.krea2Turbo.rawValue)],
            upstreamRepoId: Krea2Resources.upstreamRepoId
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
