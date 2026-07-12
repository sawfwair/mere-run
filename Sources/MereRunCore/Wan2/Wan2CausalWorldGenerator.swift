import Foundation
import MediaIO
import MLX
import MLXRandom

public enum Wan2CausalWorldGeneratorError: LocalizedError {
    case sourceImageRequired
    case sourceImageNotFound(URL)
    case invalidResolution(width: Int, height: Int)

    public var errorDescription: String? {
        switch self {
        case .sourceImageRequired:
            return "The first DreamX causal block requires a source image."
        case .sourceImageNotFound(let url):
            return "DreamX causal source image not found: \(url.path)"
        case .invalidResolution(let width, let height):
            return "DreamX causal resolution must be positive and divisible by 32; received \(width)x\(height)."
        }
    }
}

struct Wan2CausalDecodePlan: Equatable {
    let latentStart: Int
    let latentCount: Int
    let transitionStartPixelFrame: Int

    static func make(
        previousLatentFrames: Int,
        currentLatentFrames: Int,
        maximumContextLatentFrames: Int = 3
    ) -> Self {
        precondition(previousLatentFrames >= 0)
        precondition(currentLatentFrames > 0)
        precondition(maximumContextLatentFrames >= 0)
        let contextFrames = min(previousLatentFrames, maximumContextLatentFrames)
        return Self(
            latentStart: previousLatentFrames - contextFrames,
            latentCount: contextFrames + currentLatentFrames,
            transitionStartPixelFrame: contextFrames == 0 ? 0 : (contextFrames - 1) * 4
        )
    }
}

public final class Wan2CausalWorldGenerator: @unchecked Sendable {
    private let weightsURL: URL
    private var loadedRootURL: URL?
    private var tokenizer: Wan2Tokenizer?
    private var textEncoder: Wan2TextEncoderModel?
    private var vae: Wan2VAEModel?
    private var transformer: Wan2TransformerModel?
    private var causalState = Wan2CausalTransformerState()
    private var accumulatedLatents: MLXArray?
    private var promptKey: String?
    private var embeddedContext: MLXArray?
    private var crossCaches: [Wan2AttentionKVCache]?

    public init(weightsURL: URL) {
        self.weightsURL = weightsURL.standardizedFileURL
    }

    public var isWarm: Bool {
        tokenizer != nil && textEncoder != nil && vae != nil && transformer != nil
    }

    public var latentFrameCount: Int { accumulatedLatents?.dim(1) ?? 0 }

    public func prepare(
        resources: Wan2Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws {
        try ensureRoot(resources)
        _ = try loadedTextComponents(resources: resources, progress: progress)
        _ = try loadedVAE(resources: resources, progress: progress)
        _ = try loadedTransformer(progress: progress)
    }

    public func reset() {
        causalState.reset()
        accumulatedLatents = nil
    }

    public func unload() {
        tokenizer = nil
        textEncoder = nil
        vae = nil
        transformer = nil
        loadedRootURL = nil
        accumulatedLatents = nil
        promptKey = nil
        embeddedContext = nil
        crossCaches = nil
        causalState = Wan2CausalTransformerState()
        Memory.clearCache()
    }

    public func generateBlock(
        prompt: String,
        camera: Wan2WorldCameraControl,
        sourceImageURL: URL? = nil,
        resources: Wan2Resources,
        width: Int = 1_280,
        height: Int = 704,
        seed: UInt64 = 42,
        progressHandler: (@Sendable (GenerationProgress) -> Void)? = nil
    ) async throws -> Wan2VideoGenerationResult {
        guard width > 0, height > 0, width.isMultiple(of: 32), height.isMultiple(of: 32) else {
            throw Wan2CausalWorldGeneratorError.invalidResolution(width: width, height: height)
        }
        try ensureRoot(resources)
        let model = try loadedTransformer()
        let vae = try loadedVAE(resources: resources)
        let context = try conditioning(prompt: prompt, resources: resources, model: model)

        let previousLatentFrames = latentFrameCount
        let isFirstBlock = previousLatentFrames == 0
        let latentHeight = height / 16
        let latentWidth = width / 16
        MLXRandom.seed(seed)
        var noisyBlock = MLXRandom.normal([48, 3, latentHeight, latentWidth]).asType(.bfloat16)
        if isFirstBlock {
            guard let sourceImageURL else { throw Wan2CausalWorldGeneratorError.sourceImageRequired }
            guard FileManager.default.fileExists(atPath: sourceImageURL.path) else {
                throw Wan2CausalWorldGeneratorError.sourceImageNotFound(sourceImageURL)
            }
            let source = try encodeSourceImage(
                sourceImageURL,
                width: width,
                height: height,
                vae: vae
            )
            noisyBlock = MLX.concatenated([source, noisyBlock[0..., 1...]], axis: 1)
        }
        eval(noisyBlock)

        let spatialTokens = (latentHeight / 2) * (latentWidth / 2)
        let currentStartToken = previousLatentFrames * spatialTokens
        let cameraConditioning = Wan2DreamXARTrajectory.compile(
            control: camera,
            pixelFrameCount: 9,
            chunkRelative: true
        )
        let scheduler = Wan2CausalForcingScheduler()
        let frozenMask = isFirstBlock
            ? Wan2TI2VConditioning.latentMask(shape: noisyBlock.shape).asType(.bfloat16)
            : MLX.ones(noisyBlock.shape, dtype: .bfloat16)
        var latents = noisyBlock
        for (index, timestep) in scheduler.timesteps.enumerated() {
            try Task.checkCancellation()
            progressHandler?(GenerationProgress(
                stage: .denoising,
                stepIndex: index,
                totalSteps: scheduler.timesteps.count + 1
            ))
            let tokenTimesteps = causalTokenTimesteps(
                timestep: timestep,
                spatialTokens: spatialTokens,
                freezesFirstFrame: isFirstBlock
            )
            let flow = model(
                latents: [latents],
                timesteps: tokenTimesteps,
                embeddedContext: context,
                crossCaches: crossCaches,
                cameraConditioning: cameraConditioning,
                causalState: causalState,
                currentStartToken: currentStartToken
            )[0]
            var clean = scheduler.predictClean(flow: flow, sample: latents, timestep: timestep)
            clean = frozenMask * clean + (1 - frozenMask) * noisyBlock
            if index + 1 < scheduler.timesteps.count {
                let nextNoise = MLXRandom.normal(clean.shape).asType(.bfloat16)
                latents = scheduler.addNoise(
                    clean: clean,
                    noise: nextNoise,
                    timestep: scheduler.timesteps[index + 1]
                )
                latents = frozenMask * latents + (1 - frozenMask) * noisyBlock
            } else {
                latents = clean
            }
            eval(latents)
            Memory.clearCache()
        }

        progressHandler?(GenerationProgress(
            stage: .denoising,
            stepIndex: scheduler.timesteps.count,
            totalSteps: scheduler.timesteps.count + 1
        ))
        let contextTimesteps = MLX.ones([1, 3 * spatialTokens]) * Float(0.1)
        let contextFlow = model(
            latents: [latents],
            timesteps: contextTimesteps,
            embeddedContext: context,
            crossCaches: crossCaches,
            cameraConditioning: cameraConditioning,
            causalState: causalState,
            currentStartToken: currentStartToken
        )
        eval(contextFlow)
        accumulatedLatents = accumulatedLatents.map {
            MLX.concatenated([$0, latents], axis: 1)
        } ?? latents
        eval(accumulatedLatents!)
        Memory.clearCache()

        progressHandler?(GenerationProgress(
            stage: .decoding,
            stepIndex: scheduler.timesteps.count + 1,
            totalSteps: scheduler.timesteps.count + 1
        ))
        let decodePlan = Wan2CausalDecodePlan.make(
            previousLatentFrames: previousLatentFrames,
            currentLatentFrames: latents.dim(1)
        )
        let decodeLatents = accumulatedLatents![
            0...,
            decodePlan.latentStart..<(decodePlan.latentStart + decodePlan.latentCount),
            0...,
            0...
        ]
        let decodedFrames = decode(decodeLatents, vae: vae)
        let allFrames = Wan2DreamXColorStabilizer.process(decodedFrames)
        eval(allFrames)
        let transitionFrames = allFrames[0..., decodePlan.transitionStartPixelFrame...]
        eval(transitionFrames)
        let terminalLatent = accumulatedLatents![0..., (accumulatedLatents!.dim(1) - 1)...]
        return Wan2VideoGenerationResult(
            frames: transitionFrames,
            seed: seed,
            terminalFrameLatent: terminalLatent
        )
    }

    private func causalTokenTimesteps(
        timestep: Float,
        spatialTokens: Int,
        freezesFirstFrame: Bool
    ) -> MLXArray {
        guard freezesFirstFrame else {
            return MLX.ones([1, 3 * spatialTokens]) * timestep
        }
        return MLX.concatenated([
            MLX.zeros([1, spatialTokens]),
            MLX.ones([1, 2 * spatialTokens]) * timestep
        ], axis: 1)
    }

    private func conditioning(
        prompt: String,
        resources: Wan2Resources,
        model: Wan2TransformerModel
    ) throws -> MLXArray {
        if promptKey == prompt, let embeddedContext { return embeddedContext }
        let components = try loadedTextComponents(resources: resources)
        let encoded = try Wan2ModelLoader.encodePrompts(
            tokenizer: components.tokenizer,
            encoder: components.encoder,
            prompt: prompt,
            negativePrompt: Wan2Resources.defaultNegativePrompt
        )
        let context = model.embedText(encoded.prompt.expandedDimensions(axis: 0))
        let caches = model.prepareCrossAttentionCaches(context: context)
        eval(context, caches.flatMap { [$0.key, $0.value] })
        promptKey = prompt
        embeddedContext = context
        crossCaches = caches
        return context
    }

    private func encodeSourceImage(
        _ url: URL,
        width: Int,
        height: Int,
        vae: Wan2VAEModel
    ) throws -> MLXArray {
        let decoded = try MediaImageIO.decode(url)
        let image = try MediaImageIO.centerCropped(decoded, width: width, height: height)
        let channels = MediaImageIO.rgbCHWFloat(image, normalizedToMinusOneToOne: true)
        let tensor = MLXArray(channels)
            .reshaped(3, height, width)
            .transposed(1, 2, 0)
            .reshaped(1, 1, height, width, 3)
        let encoded = vae.encodeImage(tensor)
        let latent = encoded[0].transposed(3, 0, 1, 2).asType(.bfloat16)
        eval(latent)
        return latent
    }

    private func decode(_ latents: MLXArray, vae: Wan2VAEModel) -> MLXArray {
        let decoded = vae.decode(latents.transposed(1, 2, 3, 0).expandedDimensions(axis: 0))
        return MLX.clip((decoded + 1) * 127.5, min: 0, max: 255).asType(.uint8)
    }

    private func loadedTextComponents(
        resources: Wan2Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> (tokenizer: Wan2Tokenizer, encoder: Wan2TextEncoderModel) {
        try ensureRoot(resources)
        if tokenizer == nil {
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
        progress: (@Sendable (String) -> Void)? = nil
    ) throws -> Wan2TransformerModel {
        if transformer == nil {
            transformer = try Wan2ModelLoader.loadDreamXCausalTransformer(
                weightsURL: weightsURL,
                progress: progress
            )
        }
        return transformer!
    }

    private func ensureRoot(_ resources: Wan2Resources) throws {
        let root = resources.rootURL.standardizedFileURL
        if loadedRootURL != root {
            unload()
            loadedRootURL = root
        }
    }
}
