import Foundation
import MediaIO
import MLX
import MLXRandom

/// Native Swift/MLX execution, optionally retaining models for a sequential session.
public actor MiniMaxH3ImageGenerator: ImageGenerator {
    private let retainModels: Bool
    private let residentBF16: Bool
    private let attentionHeads: Int?
    private let attentionQueries: Int?
    private let vaeTileBatch: Int
    private var retainedRoot: URL?
    private var retainedManifest: MereRunModelManifest?
    private var retainedTransformer: MiniMaxH3ImageTransformer?
    private var retainedVAE: MiniMaxH3ImageVAE?
    private struct Conditioning {
        let prompt: String
        let states: MLXArray
        let tags: [Int32]
    }
    private var retainedConditioning: Conditioning?

    public init(retainModels: Bool = false, residentBF16: Bool = false, attentionHeads: Int? = nil,
                attentionQueries: Int? = nil, vaeTileBatch: Int = 32) {
        self.retainModels = retainModels
        self.residentBF16 = residentBF16
        self.attentionHeads = attentionHeads
        self.attentionQueries = attentionQueries
        self.vaeTileBatch = vaeTileBatch
    }

    static func defaultAttentionSchedule(nativeQ8: Bool, architecture: String) -> (heads: Int, queries: Int) {
        nativeQ8 && architecture == "applegpu_g17s" ? (28, 1_024) : (8, 640)
    }

    public func unload() {
        retainedTransformer = nil
        retainedVAE = nil
        retainedConditioning = nil
        retainedRoot = nil
        retainedManifest = nil
        Memory.clearCache()
    }

    private func loadVAE(_ resources: MiniMaxH3ImageResources) throws -> MiniMaxH3ImageVAE {
        if let retainedVAE { return retainedVAE }
        let vae = try MiniMaxH3ImageLoader.vae(resources)
        vae.maximumDecodeTileBatchSize = vaeTileBatch
        if retainModels { retainedVAE = vae }
        return vae
    }

    public func generate(_ request: GenerationRequest,
                         progressHandler: (@Sendable (GenerationProgress) -> Void)?) async throws -> GenerationResult {
        try Self.validate(request)
        let root = try ImageGenerationModelSelection(request.model ?? MiniMaxH3ImageResources.modelID).resolveRoot()
        let resources = MiniMaxH3ImageResources(rootURL: root)
        let missing = resources.validate()
        guard missing.isEmpty else { throw resources.invalid("Missing or invalid H3-Image resources: \(missing.map(\.path).joined(separator: ", "))") }
        let manifest = try MereRunModelManifest.loadRequired(from: root)
        let native = try resources.mlxFormat()
        guard manifest.engine == .miniMaxH3Image, manifest.family == .h3Image,
              (manifest.precision == .bf16 && native == nil)
                || (manifest.precision == .int8 && native != nil) else {
            throw resources.invalid("Expected an original BF16 or native Q8 H3-Image manifest.")
        }
        if retainModels, retainedRoot != root || retainedManifest != manifest {
            unload()
            retainedRoot = root
            retainedManifest = manifest
        }
        defer { Memory.clearCache() }
        func progress(_ stage: GenerationStage, _ step: Int = 0, _ total: Int = 1) {
            progressHandler?(.init(stage: stage, stepIndex: step, totalSteps: total))
        }
        try Task.checkCancellation()
        let urls = [request.inputImage].compactMap { $0 } + request.referenceImages
        let references = try urls.map(MiniMaxH3ImageConditioner.prepare)
        progress(.loadingEncoder)
        let conditioning: (states: MLXArray, tags: [Int32])
        if references.isEmpty, let cached = retainedConditioning, cached.prompt == request.prompt {
            progress(.encodingText)
            conditioning = (cached.states, cached.tags)
        } else {
            conditioning = try MiniMaxH3ImageConditioner.encode(prompt: request.prompt, references: references,
                resources: resources, didLoadEncoder: { progress(.encodingText) })
            if retainModels, references.isEmpty {
                retainedConditioning = Conditioning(prompt: request.prompt, states: conditioning.states, tags: conditioning.tags)
            }
        }
        progress(.encodingText, 1)
        Memory.clearCache()
        let seed = ImageGenerationSeed.resolve(request.seed, prompt: request.prompt, backend: .miniMaxH3Image)
        var draw: UInt64 = 0
        func noise(_ shape: [Int]) -> MLXArray {
            defer { draw += 1 }
            return MLXRandom.normal(shape, key: MLXRandom.key(seed &+ draw))
        }
        var referenceLatents: [MLXArray] = []
        if !references.isEmpty {
            progress(.loadingVAE)
            let vae = try loadVAE(resources)
            for (index, reference) in references.enumerated() {
                try Task.checkCancellation()
                let image = reference.vaeImage
                let pixels = MLXArray(MediaImageIO.rgbCHWFloat(image, normalizedToMinusOneToOne: false))
                    .reshaped(1, 3, image.height, image.width).transposed(0, 2, 3, 1)
                // Crop at the top left to whole VAE cells; patch alignment happens after posterior sampling.
                let moments = vae.encodeMoments(pixels[0..., 0..<(image.height / 16 * 16), 0..<(image.width / 16 * 16), 0...])
                let latent = vae.sample(moments, noise: noise([1, 24, moments.dim(2), moments.dim(3)]))
                MLX.eval(latent)
                referenceLatents.append(latent)
                progress(.encodingReferenceImages, index + 1, references.count)
            }
        }
        Memory.clearCache()
        let canvas = MiniMaxH3ImageGeometry.canvas(width: request.width, height: request.height)
        let height = canvas.height / 16, width = canvas.width / 16
        var target = MiniMaxH3Geometry.patchifyVideo(noise([1, 24, 1, height, width]))
        let referenceRows = referenceLatents.map { latent in
            let t = MiniMaxH3ImageSampling.referenceTimestep
            return MiniMaxH3Geometry.patchifyVideo((t * latent + (1 - t) * noise(latent.shape)).expandedDimensions(axis: 2))
        }
        let layout = MiniMaxH3ImageLayout(textTags: conditioning.tags,
            referenceGrids: referenceLatents.map { ($0.dim(2), $0.dim(3)) }, height: height, width: width)
        progress(.loadingTransformer)
        var transformer: MiniMaxH3ImageTransformer?
        if let retainedTransformer { transformer = retainedTransformer }
        else {
            transformer = try MiniMaxH3ImageLoader.transformer(resources)
            if residentBF16, native != nil { transformer!.materializeResidentBF16() }
        }
        if retainModels { retainedTransformer = transformer }
        #if os(macOS) || os(iOS)
        let architecture = GPU.deviceInfo().architecture
        #else
        let architecture = ""
        #endif
        let schedule = Self.defaultAttentionSchedule(nativeQ8: native != nil, architecture: architecture)
        transformer!.maximumAttentionHeadsPerKernel = attentionHeads ?? schedule.heads
        transformer!.maximumAttentionQueryTokens = attentionQueries ?? schedule.queries
        MiniMaxH3ImageProfile.install(on: transformer!)
        let text = transformer!.prepareText(conditioning.states)
        let preparedLayout = transformer!.prepareLayout(layout)
        progress(.denoising, 0, 6)
        for step in 0..<6 {
            try Task.checkCancellation()
            let rows = MLX.concatenated(referenceRows + [target], axis: 1)
            let velocity = try transformer!(imageRows: rows, text: text, preparedLayout: preparedLayout, step: step)
            target = MiniMaxH3ImageSampling.step(sample: target, velocity: velocity, index: step)
            MLX.eval(target)
            progress(.denoising, step + 1, 6)
        }
        transformer = nil
        Memory.clearCache()
        try Task.checkCancellation()
        progress(.loadingVAE)
        let vae = try loadVAE(resources)
        progress(.decoding)
        let latents = MiniMaxH3Geometry.unpatchifyVideo(target, frames: 1, height: height, width: width).squeezed(axis: 2)
        let rgb = MLX.clip(vae.decode(latents) * 255, min: 0, max: 255).asType(.uint8)
        MLX.eval(rgb)
        var image = try MediaImageIO.imageFromRGBHWC(rgb.asArray(UInt8.self), width: canvas.width, height: canvas.height)
        image = try MuseGlimmerImageProcessor.resizedLanczos(image, width: request.width, height: request.height)
        try Task.checkCancellation()
        progress(.saving)
        try MediaImageIO.writePNG(image, to: request.outputURL)
        progress(.saving, 1)
        return GenerationResult(outputURL: request.outputURL, seed: seed)
    }

    static func validate(_ request: GenerationRequest) throws {
        guard request.steps == 6, request.guidanceScale == 1, request.negativePrompt?.isEmpty != false,
              request.loras.isEmpty, request.sigmas == nil, request.sigmaShift == nil,
              request.width >= 32, request.height >= 32, request.width <= 8_192, request.height <= 8_192,
              request.width * request.height <= 20_971_520,
              request.outputURL.pathExtension.lowercased() == "png",
              !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ImageGenerationIssue("h3_image_options_invalid", "H3-Image requires six steps, CFG 1, PNG, 32–8192 pixel sides and at most 20 MP; negative prompts, LoRAs and custom schedules are unsupported.")
        }
        guard request.referenceImages.count + (request.inputImage == nil ? 0 : 1) <= 16 else {
            throw ImageGenerationIssue("reference_count_invalid", "H3-Image supports up to 16 ordered reference images, including --input.")
        }
    }
}
