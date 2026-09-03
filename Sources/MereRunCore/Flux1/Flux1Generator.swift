import Foundation
import MLX
import MLXRandom

public enum Flux1Error: LocalizedError {
    case missingFiles([URL])
    case modelNotFound(String)
    case unsupportedMode(String)
    case invalidConfiguration(String)
    case incompatibleAdapter(path: String, actual: [Int], expected: [Int])

    public var errorDescription: String? {
        switch self {
        case .missingFiles(let files):
            return "Missing FLUX.1 model files: \(files.map(\.path).joined(separator: ", "))"
        case .modelNotFound(let model):
            return "FLUX.1 model not found: \(model)"
        case .unsupportedMode(let mode):
            return "FLUX.1-dev native generation does not support \(mode)."
        case .invalidConfiguration(let message):
            return "Invalid FLUX.1 configuration: \(message)"
        case .incompatibleAdapter(let path, let actual, let expected):
            return "FLUX.1 adapter target \(path) has shape \(actual); expected \(expected). "
                + "Verify that the adapter was trained for FLUX.1-dev, not FLUX.2 or Klein."
        }
    }
}

public actor Flux1Generator: ImageGenerator {
    public init() {}

    public func unload() {
        Memory.clearCache()
    }

    public func generate(
        _ request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        guard request.referenceImages.isEmpty else {
            throw Flux1Error.unsupportedMode("reference images")
        }
        guard request.inputImage == nil else {
            throw Flux1Error.unsupportedMode("image-to-image")
        }
        guard request.negativePrompt?.isEmpty != false else {
            throw Flux1Error.unsupportedMode("negative prompts")
        }

        let rootURL = try resolveModelRoot(request.model)
        let resources = Flux1Resources(rootURL: rootURL)
        let missing = resources.validate()
        guard missing.isEmpty else { throw Flux1Error.missingFiles(missing) }
        let configurations = try loadConfigurations(resources: resources)
        try validate(configurations: configurations)

        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 0, totalSteps: 2))
        let pooled = try encodeCLIP(
            prompt: request.prompt,
            resources: resources,
            configuration: configurations.clip
        )
        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 1, totalSteps: 2))
        Memory.clearCache()

        let context = try encodeT5(
            prompt: request.prompt,
            maximumLength: min(max(request.maxSequenceLength, 1), 512),
            resources: resources,
            configuration: configurations.t5
        )
        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 2, totalSteps: 2))
        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 1, totalSteps: 1))
        Memory.clearCache()

        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 3))
        let transformer = Flux1Transformer(configuration: configurations.transformer)
        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: resources.transformerWeightsIndexURL,
            to: transformer,
            dtype: .bfloat16,
            verify: [.shapeMismatch],
            mapper: Flux1TransformerWeightMapper.map,
            progressHandler: { progress in
                progressHandler?(GenerationProgress(
                    stage: .loadingTransformer,
                    stepIndex: progress.shardIndex + 1,
                    totalSteps: progress.shardCount
                ))
            }
        )
        try await applyAdapters(request.loras, to: transformer, progressHandler: progressHandler)
        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 3, totalSteps: 3))

        let seed = request.seed ?? deterministicSeed(prompt: request.prompt)
        let resolution = Flux1SampleBuilder.alignedResolution(width: request.width, height: request.height)
        let latentHeight = resolution.height / Flux1SampleBuilder.vaeCompression
        let latentWidth = resolution.width / Flux1SampleBuilder.vaeCompression
        let noise = MLXRandom.normal(
            [1, configurations.vae.latentChannels, latentHeight, latentWidth],
            key: MLXRandom.key(seed)
        ).asType(.bfloat16)
        var tokens = Flux1SampleBuilder.pack(noise)
        let imageIDs = Flux1SampleBuilder.imageIDs(height: latentHeight, width: latentWidth)
        let textIDs = Flux1SampleBuilder.textIDs(length: context.dim(1))
        let scheduler = Flux1Scheduler(
            steps: request.steps,
            imageSequenceLength: tokens.dim(1),
            configuration: configurations.scheduler
        )
        let guidance = configurations.transformer.guidanceEmbeds
            ? MLXArray([Float(request.guidanceScale)]).asType(.bfloat16)
            : nil

        for index in 0..<max(request.steps, 1) {
            try Task.checkCancellation()
            progressHandler?(GenerationProgress(
                stage: .denoising,
                stepIndex: index,
                totalSteps: max(request.steps, 1)
            ))
            let prediction = transformer(
                image: tokens,
                context: context,
                pooledProjection: pooled,
                timestep: MLXArray([scheduler.sigmas[index]]).asType(.bfloat16),
                imageIDs: imageIDs,
                textIDs: textIDs,
                guidance: guidance
            )
            tokens = scheduler.step(modelOutput: prediction, index: index, sample: tokens)
            MLX.eval(tokens)
        }
        progressHandler?(GenerationProgress(
            stage: .denoising,
            stepIndex: max(request.steps, 1),
            totalSteps: max(request.steps, 1)
        ))

        let latents = Flux1SampleBuilder.unpack(tokens, height: latentHeight, width: latentWidth)
        MLX.eval(latents)
        Memory.clearCache()

        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        let vae = AutoencoderKL(configuration: configurations.vae.coreConfiguration)
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.vaeWeightsURL,
            to: vae,
            dtype: .bfloat16,
            verify: [.shapeMismatch],
            mapper: { key, value in
                let mapped = value.ndim == 4 && key.contains("conv")
                    ? HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value)
                    : value
                return [(key, mapped)]
            }
        )
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 1, totalSteps: 1))
        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 0, totalSteps: 1))
        let (decoded, _) = vae.decode(latents)
        var image = MLX.clip(QwenImageIO.denormalizeFromDecoder(decoded), min: 0, max: 1)
        if resolution.width != request.width || resolution.height != request.height {
            image = image[0..., 0..., 0..<request.height, 0..<request.width]
        }
        MLX.eval(image)
        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: 1, totalSteps: 1))

        progressHandler?(GenerationProgress(stage: .saving, stepIndex: 0, totalSteps: 1))
        let outputDirectory = request.outputURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        try QwenImageIO.saveImage(array: image, to: request.outputURL)
        progressHandler?(GenerationProgress(stage: .saving, stepIndex: 1, totalSteps: 1))
        Memory.clearCache()
        return GenerationResult(outputURL: request.outputURL, seed: seed)
    }

    private struct Configurations {
        let transformer: Flux1TransformerConfiguration
        let clip: Flux1CLIPConfiguration
        let t5: Flux1T5Configuration
        let vae: Flux1VAEConfiguration
        let scheduler: Flux1SchedulerConfiguration
    }

    private func loadConfigurations(resources: Flux1Resources) throws -> Configurations {
        let decoder = JSONDecoder()
        return try Configurations(
            transformer: decoder.decode(
                Flux1TransformerConfiguration.self,
                from: Data(contentsOf: resources.transformerConfigURL)
            ),
            clip: decoder.decode(
                Flux1CLIPConfiguration.self,
                from: Data(contentsOf: resources.clipConfigURL)
            ),
            t5: decoder.decode(
                Flux1T5Configuration.self,
                from: Data(contentsOf: resources.t5ConfigURL)
            ),
            vae: decoder.decode(
                Flux1VAEConfiguration.self,
                from: Data(contentsOf: resources.vaeConfigURL)
            ),
            scheduler: decoder.decode(
                Flux1SchedulerConfiguration.self,
                from: Data(contentsOf: resources.schedulerConfigURL)
            )
        )
    }

    private func validate(configurations: Configurations) throws {
        guard configurations.transformer.hiddenSize == 3_072,
              configurations.transformer.inChannels == 64,
              configurations.transformer.axesDimsRope == [16, 56, 56] else {
            throw Flux1Error.invalidConfiguration("expected the FLUX.1-dev transformer shape")
        }
        guard configurations.clip.hiddenSize == 768,
              configurations.t5.dModel == 4_096,
              configurations.vae.latentChannels == 16 else {
            throw Flux1Error.invalidConfiguration("expected CLIP-L, T5-XXL, and the 16-channel FLUX.1 VAE")
        }
    }

    private func encodeCLIP(
        prompt: String,
        resources: Flux1Resources,
        configuration: Flux1CLIPConfiguration
    ) throws -> MLXArray {
        let tokenizer = try Flux1Tokenizer.load(
            from: resources.clipTokenizerURL,
            maximumLength: configuration.maxPositionEmbeddings,
            fallbackPadTokenID: 49_407
        )
        let encoded = tokenizer.encode(prompt)
        let inputIDs = MLXArray(encoded.ids.map(Int32.init)).reshaped(1, encoded.ids.count)
        let model = try Flux1TextEncoderLoader.loadCLIP(resources: resources, configuration: configuration)
        let pooled = model.pooled(inputIDs: inputIDs, index: encoded.pooledIndex).asType(.bfloat16)
        MLX.eval(pooled)
        return pooled
    }

    private func encodeT5(
        prompt: String,
        maximumLength: Int,
        resources: Flux1Resources,
        configuration: Flux1T5Configuration
    ) throws -> MLXArray {
        let tokenizer = try Flux1Tokenizer.load(
            from: resources.t5TokenizerURL,
            maximumLength: maximumLength,
            fallbackPadTokenID: 0
        )
        let encoded = tokenizer.encode(prompt)
        let inputIDs = MLXArray(encoded.ids.map(Int32.init)).reshaped(1, encoded.ids.count)
        let mask = MLXArray(encoded.mask.map(Int32.init)).reshaped(1, encoded.mask.count)
        let model = try Flux1TextEncoderLoader.loadT5(
            resources: resources,
            configuration: configuration
        )
        let context = model(tokenIDs: inputIDs, mask: mask).asType(.bfloat16)
        MLX.eval(context)
        return context
    }

    private func applyAdapters(
        _ adapters: [LoRA],
        to transformer: Flux1Transformer,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws {
        guard !adapters.isEmpty else { return }
        var inputs: [Flux2LoRAStackInput] = []
        for (index, adapter) in adapters.enumerated() {
            let weights = try await LoRAWeightLoader.load(from: adapter, architecture: .flux1)
            inputs.append(Flux2LoRAStackInput(
                label: Self.adapterLabel(adapter),
                scale: Self.adapterScale(adapter),
                weights: weights
            ))
            progressHandler?(GenerationProgress(
                stage: .loadingLoRA,
                stepIndex: index + 1,
                totalSteps: adapters.count
            ))
        }
        let stacked = try Flux2LoRAStacker.stack(inputs)
        let layers = try Flux2LoRAInjector.inject(
            into: transformer,
            rank: stacked.rank,
            targetRanks: stacked.targetRanks,
            zeroInitUp: true
        )
        var applied = 0
        for (path, layer) in layers {
            guard let weights = stacked.weights[path] else { continue }
            guard weights.down.shape == layer.loraDown.shape else {
                throw Flux1Error.incompatibleAdapter(
                    path: path,
                    actual: weights.down.shape,
                    expected: layer.loraDown.shape
                )
            }
            guard weights.up.shape == layer.loraUp.shape else {
                throw Flux1Error.incompatibleAdapter(
                    path: path,
                    actual: weights.up.shape,
                    expected: layer.loraUp.shape
                )
            }
            layer.loraDown = weights.down.asType(.bfloat16)
            layer.loraUp = weights.up.asType(.bfloat16)
            layer.isActive = true
            applied += 1
        }
        guard applied > 0 else {
            throw LoRAError.invalidFormat("No FLUX.1 transformer layers matched this adapter.")
        }
    }

    private func resolveModelRoot(_ model: String?) throws -> URL {
        let spec = model ?? Flux1Resources.modelID
        let localURL = URL(fileURLWithPath: spec).standardizedFileURL
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: localURL.path, isDirectory: &isDirectory), isDirectory.boolValue {
            return localURL
        }
        if let modelID = ModelResolver.ModelID(rawValue: spec) {
            return try ModelResolver().resolve(modelID).rootURL
        }
        throw Flux1Error.modelNotFound(spec)
    }

    private func deterministicSeed(prompt: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in prompt.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x0000_0100_0000_01b3
        }
        return hash
    }

    private static func adapterScale(_ adapter: LoRA) -> Float {
        switch adapter {
        case .local(_, let scale), .remote(_, let scale): return Float(scale)
        }
    }

    private static func adapterLabel(_ adapter: LoRA) -> String {
        switch adapter {
        case .local(let path, _): return path
        case .remote(let reference, _): return reference
        }
    }
}
