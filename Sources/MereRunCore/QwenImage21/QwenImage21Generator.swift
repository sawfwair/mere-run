import Foundation
import MLX
import MLXRandom

public final class QwenImage21Generator: ImageGenerator {
    public init() {}

    public func generate(
        _ request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        guard request.width > 0, request.height > 0, request.width.isMultiple(of: 32), request.height.isMultiple(of: 32),
              request.steps >= 2, request.guidanceScale.isFinite, request.loras.isEmpty,
              request.sigmas == nil, request.outputURL.pathExtension.lowercased() == "png" else {
            throw QwenImage21Error.invalidConfiguration("Use PNG output, dimensions divisible by 32, and at least two steps; LoRA and custom sigma lists are unsupported.")
        }
        let references = [request.inputImage].compactMap { $0 } + request.referenceImages
        guard references.count <= 10 else { throw QwenImage21Error.invalidLayout("At most ten reference images are supported.") }
        let root = try ImageGenerationModelSelection(request.model ?? QwenImage21Resources.modelID).resolveRoot()
        let resources = QwenImage21Resources(rootURL: root)
        let missing = resources.validate()
        guard missing.isEmpty else { throw QwenImage21Error.invalidWeights("Missing or invalid resources: \(missing.map(\.path).joined(separator: ", "))") }
        defer { Memory.clearCache() }
        func progress(_ stage: GenerationStage, _ step: Int = 0, _ total: Int = 1) {
            progressHandler?(GenerationProgress(stage: stage, stepIndex: step, totalSteps: total))
        }
        try Task.checkCancellation()
        progress(.encodingReferenceImages)
        let images = try references.map(QwenImage21ImageIO.prepare)
        progress(.loadingEncoder)
        var conditioner: QwenImage21Conditioner? = try QwenImage21Conditioner(resources: resources)
        progress(.encodingText)
        let height = request.height / 16, width = request.width / 16
        let (text, layout) = try conditioner!.encode(prompt: request.prompt, images: images, targetHeight: height, targetWidth: width)
        var negative: (MLXArray, QwenImage21Layout)?
        if request.guidanceScale > 1, let prompt = request.negativePrompt {
            negative = try conditioner!.encode(prompt: prompt, images: images, targetHeight: height, targetWidth: width)
        }
        conditioner = nil
        Memory.clearCache()
        progress(.loadingVAE)
        let vaeConfig = try resources.decode("vae/config.json", as: QwenImage21VAEConfig.self)
        var vae: QwenImage21VAE? = images.isEmpty ? nil : try QwenImage21VAE(config: vaeConfig, arrays: resources.arrays("vae", stem: "diffusion_pytorch_model"))
        var referenceLatents: [MLXArray] = []
        for (index, image) in images.enumerated() {
            let latent = try vae!.encode(image.rgba).reshaped(1, -1, vaeConfig.zDim)
            eval(latent)
            referenceLatents.append(latent)
            progress(.encodingReferenceImages, index + 1, images.count)
        }
        vae = nil
        Memory.clearCache()
        progress(.loadingTransformer)
        let config = try resources.decode("transformer/config.json", as: QwenImage21TransformerConfig.self)
        var transformer: QwenImage21Transformer? = try QwenImage21Transformer(config: config, arrays: resources.arrays("transformer", stem: "diffusion_pytorch_model"))
        let schedule = try resources.decode("scheduler/scheduler_config.json", as: QwenImage21Scheduler.self)
            .sigmas(steps: request.steps, tokenCount: height * width, shift: request.sigmaShift)
        let seed = ImageGenerationSeed.resolve(request.seed, prompt: request.prompt, backend: .qwenImage21)
        // Generate in channel-first order, then flatten spatial tokens like the reference pipeline.
        var latents = MLXRandom.normal([1, vaeConfig.zDim, height, width], key: MLXRandom.key(seed))
            .asType(.bfloat16).transposed(0, 2, 3, 1).reshaped(1, height * width, vaeConfig.zDim)
        var positiveCache: QwenImage21PrefixCache? = config.causalCondition ? QwenImage21PrefixCache(layerCount: config.numLayers) : nil
        var negativeCache: QwenImage21PrefixCache? = negative == nil || !config.causalCondition ? nil : QwenImage21PrefixCache(layerCount: config.numLayers)
        for step in 0..<request.steps {
            try Task.checkCancellation()
            let input = concatenated(referenceLatents + [latents], axis: 1)
            // The reference casts scheduler t to BF16 before dividing by 1000.
            let timestep = (MLXArray(schedule[step] * 1000).asType(.bfloat16) / 1000).item(Float.self)
            var prediction = try transformer!(latents: input, text: text, timestep: timestep, layout: layout, cache: positiveCache)
            if let (negativeText, negativeLayout) = negative {
                let unconditioned = try transformer!(latents: input, text: negativeText, timestep: timestep, layout: negativeLayout, cache: negativeCache)
                prediction = unconditioned + Float(request.guidanceScale) * (prediction - unconditioned)
            }
            let delta: Float = schedule[step + 1] - schedule[step]
            let update = prediction.asType(.float32) * delta
            latents = (latents.asType(.float32) + update).asType(.bfloat16)
            eval(latents)
            progress(.denoising, step + 1, request.steps)
        }
        transformer = nil
        positiveCache = nil
        negativeCache = nil
        Memory.clearCache()
        progress(.loadingVAE)
        vae = try QwenImage21VAE(config: vaeConfig, arrays: resources.arrays("vae", stem: "diffusion_pytorch_model"))
        progress(.decoding)
        let pixels = try vae!.decode(latents.reshaped(1, height, width, vaeConfig.zDim))
        eval(pixels)
        try Task.checkCancellation()
        progress(.saving)
        try QwenImage21ImageIO.save(pixels, to: request.outputURL)
        progress(.saving, 1)
        return GenerationResult(outputURL: request.outputURL, seed: seed)
    }
}
