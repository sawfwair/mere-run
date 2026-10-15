import Foundation
import MLX
import MLXRandom

public final class Flux3Generator: ImageGenerator {
    public init() {}

    public func generate(_ request: GenerationRequest,
                         progressHandler: (@Sendable (GenerationProgress) -> Void)?) async throws -> GenerationResult {
        guard (64...4096).contains(request.width), (64...4096).contains(request.height),
              request.width.isMultiple(of: 16), request.height.isMultiple(of: 16),
              request.guidanceScale == 1, request.negativePrompt?.isEmpty != false,
              request.loras.isEmpty, request.sigmas == nil, request.sigmaShift == nil,
              request.outputURL.pathExtension.lowercased() == "png" else {
            throw Flux3Error.invalidConfiguration("Use PNG, 64–4096 pixel sides divisible by 16 and CFG 1; negative prompts, LoRA and custom schedules are unsupported.")
        }
        let urls = [request.inputImage].compactMap { $0 } + request.referenceImages
        guard urls.count <= 10 else { throw Flux3Error.invalidConfiguration("At most ten ordered reference images are supported.") }
        let schedule = try Flux3Sampling.schedule(steps: request.steps)
        let root = try ImageGenerationModelSelection(request.model ?? Flux3Resources.modelID).resolveRoot()
        let resources = Flux3Resources(rootURL: root)
        let missing = resources.validate()
        guard missing.isEmpty else { throw Flux3Error.invalidWeights("Missing resources: \(missing.map(\.path).joined(separator: ", "))") }
        let manifest = try MereRunModelManifest.loadRequired(from: root)
        guard manifest.engine == .flux3, manifest.variant == .distilled, manifest.precision == .bf16 else {
            throw Flux3Error.invalidConfiguration("Expected the original distilled BF16 FLUX 3 Klein package.")
        }
        defer { Memory.clearCache() }
        func progress(_ stage: GenerationStage, _ step: Int = 0, _ total: Int = 1) {
            progressHandler?(GenerationProgress(stage: stage, stepIndex: step, totalSteps: total))
        }
        try Task.checkCancellation()
        progress(.loadingEncoder)
        var conditioner: Flux3Conditioner? = try Flux3Conditioner(resources: resources)
        progress(.encodingText)
        let text = try conditioner!.encode(request.prompt)
        conditioner = nil
        Memory.clearCache()
        var referenceTokens: [MLXArray] = [], referenceIDs: [MLXArray] = []
        if !urls.isEmpty {
            progress(.loadingVAE)
            let vae = try Flux3VAE(resources: resources)
            for (index, url) in urls.enumerated() {
                try Task.checkCancellation()
                let image = try Flux3ImageIO.prepare(url, multiple: urls.count > 1)
                let tokens = vae.encode(image.pixels)
                eval(tokens)
                referenceTokens.append(tokens)
                referenceIDs.append(Flux3Latents.imageIDs(height: image.height / 16, width: image.width / 16, referenceIndex: index))
                progress(.encodingReferenceImages, index + 1, urls.count)
            }
        }
        Memory.clearCache()
        progress(.loadingTransformer)
        var transformer: Flux3Transformer? = try Flux3Transformer(weights: MLX.loadArrays(url: root.appending(path: "model.safetensors")))
        let seed = ImageGenerationSeed.resolve(request.seed, prompt: request.prompt, backend: .flux3)
        let height = request.height / 16, width = request.width / 16
        // The upstream keeps FP32 Euler state, initialized with BF16-valued noise.
        var tokens = MLXRandom.normal([height * width, 128], key: MLXRandom.key(seed)).asType(.bfloat16).asType(.float32)
        let imageIDs = Flux3Latents.imageIDs(height: height, width: width)
        let textIDs = Flux3Latents.textIDs(length: text.dim(0))
        let references = referenceTokens.isEmpty ? nil : concatenated(referenceTokens, axis: 0)
        let refIDs = referenceIDs.isEmpty ? nil : concatenated(referenceIDs, axis: 0)
        for step in 0..<request.steps {
            try Task.checkCancellation()
            let prediction = try transformer!(image: tokens, imageIDs: imageIDs, text: text, textIDs: textIDs,
                references: references, referenceIDs: refIDs, timestep: schedule[step], nextTimestep: schedule[step + 1])
            tokens = tokens + (schedule[step + 1] - schedule[step]) * prediction.asType(.float32)
            eval(tokens)
            progress(.denoising, step + 1, request.steps)
        }
        transformer = nil
        Memory.clearCache()
        try Task.checkCancellation()
        progress(.loadingVAE)
        let vae = try Flux3VAE(resources: resources)
        progress(.decoding)
        let image = vae.decode(tokens, height: height, width: width)
        eval(image)
        try Task.checkCancellation()
        progress(.saving)
        try QwenImageIO.saveImage(array: image, to: request.outputURL)
        progress(.saving, 1)
        return GenerationResult(outputURL: request.outputURL, seed: seed)
    }
}
