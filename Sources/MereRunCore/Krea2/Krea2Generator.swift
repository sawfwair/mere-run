import Foundation
import MLX
import MLXRandom

public enum Krea2GeneratorError: LocalizedError, Sendable {
    case missingModelFiles([URL])
    case unsupportedMode(String)
    case modelsNotLoaded
    case invalidConditioningRebalanceWeights(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .missingModelFiles(let urls):
            return "Missing Krea 2 model files: \(urls.map(\.path).joined(separator: ", "))"
        case .unsupportedMode(let mode):
            return "Krea 2 Turbo native generation does not support \(mode) yet."
        case .modelsNotLoaded:
            return "Krea 2 models were not loaded."
        case .invalidConditioningRebalanceWeights(let expected, let actual):
            return "Krea 2 conditioning rebalance expected \(expected) layer weights, received \(actual)."
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
    private var currentLoRA: LoRA?
    private var transformerLoRALayers: [String: TrainableLoRALayer]?
    private var transformerLoRARank: Int?

    private static let loraDebugEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_LORA_DEBUG"]?.lowercased()
        return raw == "1" || raw == "true" || raw == "yes"
    }()

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
        currentLoRA = nil
        transformerLoRALayers = nil
        transformerLoRARank = nil
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

        let rootURL = try resolveModelRoot(request)
        let resources = Krea2Resources(rootURL: rootURL)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw Krea2GeneratorError.missingModelFiles(missing)
        }
        if let bits = request.kreaBaseQuantizationBits {
            return try await generateMemoryEfficient(
                request,
                resources: resources,
                bits: bits,
                progressHandler: progressHandler
            )
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

        try await applyLoRAIfNeeded(
            request.lora,
            resources: resources,
            configuration: configs?.transformer,
            progressHandler: progressHandler
        )

        guard let configs, let transformer, let textEncoder, let tokenizer, let vae else {
            throw Krea2GeneratorError.modelsNotLoaded
        }

        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1))
        let text = try encodePrompt(
            request.prompt,
            tokenizer: tokenizer,
            encoder: textEncoder,
            selectedLayers: configs.selectedTextLayers,
            maxLength: min(request.maxSequenceLength, 512),
            conditioningRebalance: request.kreaConditioningRebalance
        )
        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 1, totalSteps: 1))

        let denoised = try denoise(
            request,
            configs: configs,
            transformer: transformer,
            text: text,
            progressHandler: progressHandler
        )
        return try decodeAndSave(
            denoised.latents,
            seed: denoised.seed,
            request: request,
            configs: configs,
            vae: vae,
            progressHandler: progressHandler
        )
    }

    private func generateMemoryEfficient(
        _ request: GenerationRequest,
        resources: Krea2Resources,
        bits: Int,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        unload()
        let loadedConfigs = try Krea2ModelConfigs.load(from: resources)
        configs = loadedConfigs

        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 0, totalSteps: 2))
        let text: (hiddenStates: MLXArray, attentionMask: MLXArray)
        do {
            let activeTokenizer = try QwenTokenizer.load(from: resources.tokenizerURL, maxLengthOverride: 512)
            let activeTextEncoder = try Krea2ModelLoader.loadTextEncoder(
                from: resources,
                configuration: loadedConfigs.textEncoder
            )
            progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 2, totalSteps: 2))
            progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1))
            text = try encodePrompt(
                request.prompt,
                tokenizer: activeTokenizer,
                encoder: activeTextEncoder,
                selectedLayers: loadedConfigs.selectedTextLayers,
                maxLength: min(request.maxSequenceLength, 512),
                conditioningRebalance: request.kreaConditioningRebalance
            )
            MLX.eval(text.hiddenStates, text.attentionMask)
            progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 1, totalSteps: 1))
        }
        MLX.Memory.clearCache()

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
        guard let transformer else {
            throw Krea2GeneratorError.modelsNotLoaded
        }
        Krea2LoRATrainer.quantizeTransformerBase(transformer, bits: bits)
        MLX.eval(transformer)
        MLX.Memory.clearCache()
        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 3, totalSteps: 3))

        try await applyLoRAIfNeeded(
            request.lora,
            resources: resources,
            configuration: loadedConfigs.transformer,
            progressHandler: progressHandler
        )
        let denoised = try denoise(
            request,
            configs: loadedConfigs,
            transformer: transformer,
            text: text,
            progressHandler: progressHandler
        )
        MLX.eval(denoised.latents)
        self.transformer = nil
        transformerLoRALayers = nil
        transformerLoRARank = nil
        currentLoRA = nil
        MLX.Memory.clearCache()

        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        let activeVAE = try Krea2ModelLoader.loadVAE(from: resources, configuration: loadedConfigs.vae)
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 1, totalSteps: 1))
        return try decodeAndSave(
            denoised.latents,
            seed: denoised.seed,
            request: request,
            configs: loadedConfigs,
            vae: activeVAE,
            progressHandler: progressHandler
        )
    }

    private func denoise(
        _ request: GenerationRequest,
        configs: Krea2ModelConfigs,
        transformer: Krea2Transformer,
        text: (hiddenStates: MLXArray, attentionMask: MLXArray),
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws -> (latents: MLXArray, seed: UInt64) {
        let seed = request.seed ?? deterministicSeed(prompt: request.prompt)
        let aligned = Krea2SampleBuilder.alignedResolution(width: request.width, height: request.height)
        let initialLatents = MLXRandom.normal(
            [
                1,
                configs.transformer.latentChannels,
                aligned.height / Krea2SampleBuilder.vaeCompression,
                aligned.width / Krea2SampleBuilder.vaeCompression,
            ],
            key: MLXRandom.key(seed)
        ).asType(.bfloat16)
        let prepared = Krea2SampleBuilder.prepare(
            latents: initialLatents,
            textLength: text.hiddenStates.dim(1),
            textMask: text.attentionMask,
            patch: configs.transformer.patchSize
        )
        var imageTokens = prepared.imageTokens.asType(.float32)
        let timesteps = Krea2SampleBuilder.timesteps(
            imageTokenCount: prepared.imageTokenCount,
            steps: request.steps,
            mu: request.sigmaShift ?? Krea2SampleBuilder.defaultMu
        )
        let textContextBF16 = text.hiddenStates.asType(.bfloat16)
        for step in 0..<request.steps {
            try Task.checkCancellation()
            progressHandler?(GenerationProgress(stage: .denoising, stepIndex: step, totalSteps: request.steps))
            let tCurrent = timesteps[step]
            let tPrevious = timesteps[step + 1]
            let velocity = transformer(
                imageTokens: imageTokens.asType(.bfloat16),
                textContext: textContextBF16,
                timestep: MLXArray([tCurrent]).asType(.bfloat16),
                positionIds: prepared.positionIds,
                validMask: prepared.validMask
            )
            imageTokens = imageTokens + velocity.asType(.float32) * MLXArray(tPrevious - tCurrent)
            MLX.eval(imageTokens)
        }
        progressHandler?(GenerationProgress(stage: .denoising, stepIndex: request.steps, totalSteps: request.steps))
        let latents = Krea2SampleBuilder.unpatchify(
            imageTokens.asType(.bfloat16),
            tokenHeight: prepared.imageTokenHeight,
            tokenWidth: prepared.imageTokenWidth,
            channels: configs.transformer.latentChannels,
            patch: configs.transformer.patchSize
        )
        return (latents, seed)
    }

    private func decodeAndSave(
        _ latents: MLXArray,
        seed: UInt64,
        request: GenerationRequest,
        configs: Krea2ModelConfigs,
        vae: QwenImageEditVAE,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws -> GenerationResult {
        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 0, totalSteps: 1))
        let decoded = decodeLatents(
            latents,
            vae: vae,
            config: configs.vae,
            height: request.height,
            width: request.width
        )
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

    private func applyLoRAIfNeeded(
        _ lora: LoRA?,
        resources: Krea2Resources,
        configuration: Krea2TransformerConfiguration?,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws {
        guard lora != currentLoRA else { return }

        if lora == nil {
            if let transformerLoRALayers {
                for layer in transformerLoRALayers.values {
                    layer.isActive = false
                }
            }
            currentLoRA = nil
            return
        }

        guard let lora else { return }
        guard let configuration else {
            throw Krea2GeneratorError.modelsNotLoaded
        }

        progressHandler?(GenerationProgress(stage: .loadingLoRA, stepIndex: 0, totalSteps: 2))
        let loraURL = try await LoRAWeightLoader.resolveURL(for: lora)
        let loraWeights = try LoRAWeightLoader.load(from: loraURL)
        let targetRank = loraWeights.rank

        if let existingRank = transformerLoRARank, existingRank != targetRank {
            transformer = try Krea2ModelLoader.loadTransformer(
                from: resources,
                configuration: configuration,
                progressHandler: nil
            )
            transformerLoRALayers = nil
            transformerLoRARank = nil
            currentLoRA = nil
        }

        guard let transformer else {
            throw Krea2GeneratorError.modelsNotLoaded
        }

        if transformerLoRALayers == nil {
            if Self.loraDebugEnabled {
                FileHandle.standardError.write(
                    Data("[Krea2 LoRA] Injecting with rank=\(targetRank), alpha=\(loraWeights.alpha)\n".utf8)
                )
            }
            transformerLoRALayers = try Krea2LoRAInjector.inject(
                into: transformer,
                rank: targetRank,
                alpha: loraWeights.alpha,
                targetSuffixes: Krea2LoRAInjector.defaultTargetSuffixes,
                zeroInitUp: true
            )
            transformerLoRARank = targetRank
            if let transformerLoRALayers {
                for layer in transformerLoRALayers.values {
                    layer.isActive = false
                }
            }
        }

        guard let layers = transformerLoRALayers else {
            throw LoRAError.invalidFormat("Failed to inject Krea 2 LoRA layers.")
        }

        let updatedCount = Krea2LoRAInjector.applyWeights(
            loraWeights,
            to: layers,
            debug: Self.loraDebugEnabled
        )
        for (path, layer) in layers {
            layer.isActive = loraWeights.weights[path] != nil
        }
        guard updatedCount > 0 else {
            throw LoRAError.invalidFormat("No matching Krea 2 transformer layers found for this LoRA.")
        }

        let userScale = Self.loraScale(for: lora)
        if userScale != 1 {
            let scale = MLXArray(userScale)
            for layer in layers.values where layer.isActive {
                layer.loraDown = layer.loraDown * scale
            }
        }

        currentLoRA = lora
        progressHandler?(GenerationProgress(stage: .loadingLoRA, stepIndex: 2, totalSteps: 2))
    }

    private static func loraScale(for lora: LoRA) -> Float {
        switch lora {
        case .local(_, let scale):
            return Float(scale)
        case .remote(_, let scale):
            return Float(scale)
        }
    }

    private func encodePrompt(
        _ prompt: String,
        tokenizer: QwenTokenizer,
        encoder: QwenEncoder,
        selectedLayers: [Int],
        maxLength: Int,
        conditioningRebalance: Krea2ConditioningRebalance?
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
        let rebalancedStates = try rebalanceTextConditioning(croppedStates, conditioningRebalance)
        let croppedMask = attentionMask[0..., prefixDropCount...]
        MLX.eval(rebalancedStates, croppedMask)
        return (rebalancedStates, croppedMask)
    }

    private func rebalanceTextConditioning(
        _ hiddenStates: MLXArray,
        _ conditioningRebalance: Krea2ConditioningRebalance?
    ) throws -> MLXArray {
        guard let conditioningRebalance else { return hiddenStates }

        var rebalanced = hiddenStates.asType(.float32)
        if !conditioningRebalance.layerWeights.isEmpty {
            let layerCount = rebalanced.dim(2)
            guard conditioningRebalance.layerWeights.count == layerCount else {
                throw Krea2GeneratorError.invalidConditioningRebalanceWeights(
                    expected: layerCount,
                    actual: conditioningRebalance.layerWeights.count
                )
            }

            let layerWeights = MLXArray(conditioningRebalance.layerWeights)
                .reshaped(1, 1, layerCount, 1)
                .asType(.float32)
            rebalanced = rebalanced * layerWeights
        }

        if conditioningRebalance.multiplier != 1 {
            rebalanced = rebalanced * MLXArray(conditioningRebalance.multiplier).asType(.float32)
        }

        return rebalanced.asType(hiddenStates.dtype)
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
