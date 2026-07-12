import Foundation
import MediaIO
import MLX
import MLXRandom

public enum Wan2TI2VGeneratorError: LocalizedError {
    case sourceImageNotFound(URL)
    case sourceImageDecodeFailed(URL)
    case missingModelFiles([URL])

    public var errorDescription: String? {
        switch self {
        case .sourceImageNotFound(let url):
            return "Wan2.2 source image not found: \(url.path)"
        case .sourceImageDecodeFailed(let url):
            return "Wan2.2 could not decode source image: \(url.path)"
        case .missingModelFiles(let urls):
            return "Wan2.2 model root is missing: \(urls.map(\.lastPathComponent).joined(separator: ", "))"
        }
    }
}

public struct Wan2VideoGenerationResult: @unchecked Sendable {
    public let frames: MLXArray
    public let seed: UInt64
    public let terminalFrameLatent: MLXArray?

    public init(frames: MLXArray, seed: UInt64, terminalFrameLatent: MLXArray? = nil) {
        self.frames = frames
        self.seed = seed
        self.terminalFrameLatent = terminalFrameLatent
    }
}

public final class Wan2TI2VGenerator: @unchecked Sendable {
    private struct PromptCacheKey: Hashable {
        let prompt: String
        let negativePrompt: String
    }

    private var loadedRootURL: URL?
    private var tokenizer: Wan2Tokenizer?
    private var textEncoder: Wan2TextEncoderModel?
    private var vae: Wan2VAEModel?
    private var transformer: Wan2TransformerModel?
    private var promptCache: [PromptCacheKey: Wan2TextConditioning] = [:]
    private var chainedTerminalFrameLatent: MLXArray?
    private let cameraWeightsURL: URL?

    public init(cameraWeightsURL: URL? = nil) {
        self.cameraWeightsURL = cameraWeightsURL?.standardizedFileURL
    }

    public var conditioningMode: Wan2WorldConditioningMode {
        cameraWeightsURL == nil ? .textAndFirstFrame : .projectiveCameraLatents
    }

    public var isWarm: Bool {
        tokenizer != nil && textEncoder != nil && vae != nil && transformer != nil
    }

    public var hasChainedTerminalFrameLatent: Bool {
        chainedTerminalFrameLatent != nil
    }

    public func clearChain() {
        chainedTerminalFrameLatent = nil
    }

    public func prepare(
        resources: Wan2Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws {
        try ensureRoot(resources)
        _ = try loadedTextComponents(resources: resources, progress: progress)
        _ = try loadedVAE(resources: resources, progress: progress)
        _ = try loadedTransformer(resources: resources, progress: progress)
    }

    public func unload() {
        tokenizer = nil
        textEncoder = nil
        vae = nil
        transformer = nil
        loadedRootURL = nil
        promptCache.removeAll(keepingCapacity: false)
        chainedTerminalFrameLatent = nil
        Memory.clearCache()
    }

    public func generate(
        options: Wan2GenerationOptions,
        resources: Wan2Resources,
        useChainedSourceLatent: Bool = false,
        captureTerminalFrameLatent: Bool = false,
        progressHandler: (@Sendable (GenerationProgress) -> Void)? = nil
    ) async throws -> Wan2VideoGenerationResult {
        guard FileManager.default.fileExists(atPath: options.sourceImageURL.path) else {
            throw Wan2TI2VGeneratorError.sourceImageNotFound(options.sourceImageURL)
        }
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw Wan2TI2VGeneratorError.missingModelFiles(missing)
        }
        try ensureRoot(resources)

        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: options.steps))
        let conditioning = try conditioning(options: options, resources: resources)
        eval(conditioning.prompt, conditioning.negativePrompt)
        Memory.clearCache()

        progressHandler?(GenerationProgress(stage: .encodingReferenceImages, stepIndex: 0, totalSteps: options.steps))
        let imageLatent: MLXArray
        if useChainedSourceLatent, let chainedTerminalFrameLatent {
            imageLatent = chainedTerminalFrameLatent
        } else {
            imageLatent = try encodeSourceImage(options: options, resources: resources)
        }
        Memory.clearCache()

        MLXRandom.seed(options.seed)
        let noise = MLXRandom.normal(options.latentShape).asType(.float32)
        let latentMask = Wan2TI2VConditioning.latentMask(shape: options.latentShape)
        let tokenMask = Wan2TI2VConditioning.tokenMask(latentShape: options.latentShape)
        var latent = Wan2TI2VConditioning.blend(imageLatent: imageLatent, noise: noise, mask: latentMask)
        eval(latent, latentMask, tokenMask)

        progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: options.steps))
        latent = try await denoise(
            latent: latent,
            imageLatent: imageLatent,
            latentMask: latentMask,
            tokenMask: tokenMask,
            conditioning: conditioning,
            options: options,
            resources: resources,
            progressHandler: progressHandler
        )
        eval(latent)
        Memory.clearCache()

        progressHandler?(GenerationProgress(stage: .decoding, stepIndex: options.steps, totalSteps: options.steps))
        let frames = try decode(latent: latent, resources: resources)
        eval(frames)
        let terminalFrameLatent = captureTerminalFrameLatent
            ? try encodeTerminalFrame(frames, resources: resources)
            : nil
        if let terminalFrameLatent {
            eval(terminalFrameLatent)
            chainedTerminalFrameLatent = terminalFrameLatent
        }
        Memory.clearCache()
        return Wan2VideoGenerationResult(
            frames: frames,
            seed: options.seed,
            terminalFrameLatent: terminalFrameLatent
        )
    }

    private func encodeSourceImage(
        options: Wan2GenerationOptions,
        resources: Wan2Resources
    ) throws -> MLXArray {
        let decoded: MediaImage
        do {
            decoded = try MediaImageIO.decode(options.sourceImageURL)
        } catch {
            throw Wan2TI2VGeneratorError.sourceImageDecodeFailed(options.sourceImageURL)
        }
        let image = try MediaImageIO.centerCropped(decoded, width: options.width, height: options.height)
        let channels = MediaImageIO.rgbCHWFloat(image, normalizedToMinusOneToOne: true)
        let tensor = MLXArray(channels)
            .reshaped(3, options.height, options.width)
            .transposed(1, 2, 0)
            .reshaped(1, 1, options.height, options.width, 3)

        let vae = try loadedVAE(resources: resources)
        let encoded = vae.encodeImage(tensor)
        let latent = encoded[0].transposed(3, 0, 1, 2).asType(.float32)
        eval(latent)
        return latent
    }

    private func denoise(
        latent initialLatent: MLXArray,
        imageLatent: MLXArray,
        latentMask: MLXArray,
        tokenMask: MLXArray,
        conditioning: Wan2TextConditioning,
        options: Wan2GenerationOptions,
        resources: Wan2Resources,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> MLXArray {
        let transformer = try loadedTransformer(resources: resources)
        let positiveContext = transformer.embedText(conditioning.prompt.expandedDimensions(axis: 0))
        let negativeContext = transformer.embedText(conditioning.negativePrompt.expandedDimensions(axis: 0))
        let positiveCaches = transformer.prepareCrossAttentionCaches(context: positiveContext)
        let negativeCaches = transformer.prepareCrossAttentionCaches(context: negativeContext)
        eval(
            positiveContext,
            negativeContext,
            positiveCaches.flatMap { [$0.key, $0.value] },
            negativeCaches.flatMap { [$0.key, $0.value] }
        )

        var eulerScheduler = Wan2FlowMatchEulerScheduler(steps: options.steps, shift: options.shift)
        var uniPCScheduler = Wan2UniPCScheduler(steps: options.steps, shift: options.shift)
        let usesDreamXSampler = options.cameraConditioning != nil
        let timesteps = usesDreamXSampler ? eulerScheduler.timesteps : uniPCScheduler.timesteps
        var latent = initialLatent
        let debugStats = ProcessInfo.processInfo.environment["MERERUN_WAN2_DEBUG_STATS"] == "1"
        for (stepIndex, timestep) in timesteps.enumerated() {
            try Task.checkCancellation()
            progressHandler?(GenerationProgress(
                stage: .denoising,
                stepIndex: stepIndex,
                totalSteps: options.steps
            ))
            let tokenTimesteps = tokenMask * timestep
            let conditional = transformer(
                latents: [latent],
                timesteps: tokenTimesteps,
                embeddedContext: positiveContext,
                crossCaches: positiveCaches,
                cameraConditioning: options.cameraConditioning
            )[0]
            eval(conditional)
            Memory.clearCache()

            let unconditional = transformer(
                latents: [latent],
                timesteps: tokenTimesteps,
                embeddedContext: negativeContext,
                crossCaches: negativeCaches,
                cameraConditioning: options.cameraConditioning
            )[0]
            let velocity = unconditional + options.guidanceScale * (conditional - unconditional)
            latent = usesDreamXSampler
                ? eulerScheduler.step(velocity: velocity, sample: latent)
                : uniPCScheduler.step(modelOutput: velocity, sample: latent)
            latent = Wan2TI2VConditioning.blend(
                imageLatent: imageLatent,
                noise: latent,
                mask: latentMask
            )
            eval(latent)
            if debugStats {
                Self.writeDebugStats(
                    stepIndex: stepIndex,
                    timestep: timestep,
                    conditional: conditional,
                    unconditional: unconditional,
                    velocity: velocity,
                    latent: latent
                )
            }
            Memory.clearCache()
        }
        return latent
    }

    private func decode(latent: MLXArray, resources: Wan2Resources) throws -> MLXArray {
        let vae = try loadedVAE(resources: resources)
        let channelsLast = latent.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
        let decoded = vae.decode(channelsLast)
        return MLX.clip((decoded + 1) * 127.5, min: 0, max: 255).asType(.uint8)
    }

    private func encodeTerminalFrame(
        _ frames: MLXArray,
        resources: Wan2Resources
    ) throws -> MLXArray {
        precondition(frames.ndim == 5 && frames.dim(0) == 1 && frames.dim(4) == 3)
        let terminal = frames[0, frames.dim(1) - 1]
            .asType(.float32) / 127.5 - 1
        let image = terminal.reshaped(1, 1, frames.dim(2), frames.dim(3), 3)
        let encoded = try loadedVAE(resources: resources).encodeImage(image)
        return encoded[0].transposed(3, 0, 1, 2).asType(.float32)
    }

    private func conditioning(
        options: Wan2GenerationOptions,
        resources: Wan2Resources
    ) throws -> Wan2TextConditioning {
        let key = PromptCacheKey(prompt: options.prompt, negativePrompt: options.negativePrompt)
        if let cached = promptCache[key] {
            return cached
        }
        let components = try loadedTextComponents(resources: resources)
        let value = try Wan2ModelLoader.encodePrompts(
            tokenizer: components.tokenizer,
            encoder: components.encoder,
            prompt: options.prompt,
            negativePrompt: options.negativePrompt
        )
        promptCache[key] = value
        return value
    }

    private func ensureRoot(_ resources: Wan2Resources) throws {
        let root = resources.rootURL.standardizedFileURL
        if let loadedRootURL, loadedRootURL != root {
            unload()
        }
        loadedRootURL = root
    }

    private func loadedTextComponents(
        resources: Wan2Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> (tokenizer: Wan2Tokenizer, encoder: Wan2TextEncoderModel) {
        try ensureRoot(resources)
        if tokenizer == nil {
            progress?("Loading UMT5 tokenizer")
            tokenizer = try Wan2ModelLoader.loadTokenizer(resources: resources)
        }
        if textEncoder == nil {
            textEncoder = try Wan2ModelLoader.loadTextEncoder(resources: resources, progress: progress)
        }
        return (tokenizer!, textEncoder!)
    }

    private func loadedVAE(
        resources: Wan2Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Wan2VAEModel {
        try ensureRoot(resources)
        if vae == nil {
            vae = try Wan2ModelLoader.loadVAE(resources: resources, progress: progress)
        }
        return vae!
    }

    private func loadedTransformer(
        resources: Wan2Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Wan2TransformerModel {
        try ensureRoot(resources)
        if transformer == nil {
            if let cameraWeightsURL {
                transformer = try Wan2ModelLoader.loadDreamXCameraTransformer(
                    resources: resources,
                    cameraWeightsURL: cameraWeightsURL,
                    progress: progress
                )
            } else {
                transformer = try Wan2ModelLoader.loadTransformer(resources: resources, progress: progress)
            }
        }
        return transformer!
    }

    private static func writeDebugStats(
        stepIndex: Int,
        timestep: Float,
        conditional: MLXArray,
        unconditional: MLXArray,
        velocity: MLXArray,
        latent: MLXArray
    ) {
        func summary(_ array: MLXArray) -> String {
            let value = array.asType(.float32)
            let mean = MLX.mean(value).item(Float.self)
            let standardDeviation = MLX.std(value).item(Float.self)
            let minimum = MLX.min(value).item(Float.self)
            let maximum = MLX.max(value).item(Float.self)
            return String(format: "mean=%.5f std=%.5f min=%.5f max=%.5f", mean, standardDeviation, minimum, maximum)
        }
        let line = "[Wan2 debug] step=\(stepIndex + 1) t=\(timestep) cond{\(summary(conditional))} uncond{\(summary(unconditional))} velocity{\(summary(velocity))} latent{\(summary(latent))}\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}
