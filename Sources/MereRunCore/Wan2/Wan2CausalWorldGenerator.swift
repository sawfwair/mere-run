import Foundation
import MediaIO
import MLX
import MLXRandom

public enum Wan2CausalWorldGeneratorError: LocalizedError {
    case sourceImageRequired
    case sourceImageNotFound(URL)
    case invalidResolution(width: Int, height: Int)
    case resolutionExceedsMaximum(width: Int, height: Int)
    case invalidLatentFrameCount(Int)
    case latentFrameCountExceedsMaximum(requested: Int, maximum: Int)
    case emptyActionSequence
    case invalidActionSequence
    case invalidSpeed(Float)

    public var errorDescription: String? {
        switch self {
        case .sourceImageRequired:
            return "The first DreamX causal block requires a source image."
        case .sourceImageNotFound(let url):
            return "DreamX causal source image not found: \(url.path)"
        case .invalidResolution(let width, let height):
            return "DreamX causal resolution must be positive and divisible by 32; received \(width)x\(height)."
        case .resolutionExceedsMaximum(let width, let height):
            return "DreamX causal rollout resolution may not exceed 1280x704; received \(width)x\(height)."
        case .invalidLatentFrameCount(let count):
            return "DreamX causal rollout latent-frame count must be positive and divisible by 3; received \(count)."
        case .latentFrameCountExceedsMaximum(let requested, let maximum):
            return "DreamX causal rollout supports at most \(maximum) latent frames; received \(requested)."
        case .emptyActionSequence:
            return "DreamX causal rollout requires at least one keyboard action segment."
        case .invalidActionSequence:
            return "DreamX causal rollout action weights must have a finite positive sum and fit within the pixel-frame count."
        case .invalidSpeed(let speed):
            return "DreamX causal rollout speed must be finite and positive; received \(speed)."
        }
    }
}

public struct Wan2CausalRolloutChunk: @unchecked Sendable {
    public let blockIndex: Int
    public let blockCount: Int
    public let pixelFrameStart: Int
    public let frames: MLXArray

    public init(blockIndex: Int, blockCount: Int, pixelFrameStart: Int, frames: MLXArray) {
        self.blockIndex = blockIndex
        self.blockCount = blockCount
        self.pixelFrameStart = pixelFrameStart
        self.frames = frames
    }
}

public struct Wan2CausalWorldCheckpoint: @unchecked Sendable {
    fileprivate let causalState: Wan2CausalTransformerCheckpoint
    fileprivate let accumulatedLatents: MLXArray?
    fileprivate let promptKey: String?
    fileprivate let embeddedContext: MLXArray?
    fileprivate let crossCaches: [Wan2AttentionKVCache]?
    let sceneMemory: Wan2DreamXSceneMemoryCheckpoint
    public let currentWorldPose: Wan2DreamXWorldPose
    public let generatedLatentFrameCount: Int
    public let retainedLatentFrameCount: Int
    public let sceneMemoryFrameCount: Int
    public let sceneMemoryRetrievalCount: Int
    public let sceneMemoryRecycledFrameCount: Int
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
    /// Current upstream one-minute recipe: 252 latent frames -> 1,005 pixels.
    public static let maximumRolloutLatentFrameCount = 252
    private let weightsURL: URL
    private var loadedRootURL: URL?
    private var tokenizer: Wan2Tokenizer?
    private var textEncoder: Wan2TextEncoderModel?
    private var vae: Wan2VAEModel?
    private var transformer: Wan2TransformerModel?
    private var causalState = Wan2CausalTransformerState()
    private var accumulatedLatents: MLXArray?
    private var generatedLatentFrames = 0
    private var promptKey: String?
    private var embeddedContext: MLXArray?
    private var crossCaches: [Wan2AttentionKVCache]?
    private let sceneMemoryPolicy: Wan2DreamXSceneMemoryPolicy
    private var sceneMemory: Wan2DreamXSceneMemory
    private var currentWorldPose: Wan2DreamXWorldPose = .identity
    private var sceneMemoryRetrievals = 0
    private var sceneMemoryRecycledFrames = 0

    public init(
        weightsURL: URL,
        sceneMemoryPolicy: Wan2DreamXSceneMemoryPolicy = .init()
    ) {
        self.weightsURL = weightsURL.standardizedFileURL
        self.sceneMemoryPolicy = sceneMemoryPolicy
        self.sceneMemory = Wan2DreamXSceneMemory(policy: sceneMemoryPolicy)
    }

    public var isWarm: Bool {
        tokenizer != nil && textEncoder != nil && vae != nil && transformer != nil
    }

    public var latentFrameCount: Int { generatedLatentFrames }
    public var retainedLatentFrameCount: Int { accumulatedLatents?.dim(1) ?? 0 }
    public var worldPose: Wan2DreamXWorldPose { currentWorldPose }
    public var sceneMemoryMode: Wan2DreamXSceneMemoryMode { sceneMemoryPolicy.mode }
    public var sceneMemoryConfiguration: Wan2DreamXSceneMemoryPolicy { sceneMemoryPolicy }
    public var sceneMemoryFrameCount: Int { sceneMemory.frameCount }
    public var sceneMemoryRetrievalCount: Int { sceneMemoryRetrievals }
    public var sceneMemoryRecycledFrameCount: Int { sceneMemoryRecycledFrames }

    public func checkpoint() -> Wan2CausalWorldCheckpoint {
        Wan2CausalWorldCheckpoint(
            causalState: causalState.checkpoint(),
            accumulatedLatents: accumulatedLatents,
            promptKey: promptKey,
            embeddedContext: embeddedContext,
            crossCaches: crossCaches,
            sceneMemory: sceneMemory.checkpoint(),
            currentWorldPose: currentWorldPose,
            generatedLatentFrameCount: generatedLatentFrames,
            retainedLatentFrameCount: retainedLatentFrameCount,
            sceneMemoryFrameCount: sceneMemory.frameCount,
            sceneMemoryRetrievalCount: sceneMemoryRetrievals,
            sceneMemoryRecycledFrameCount: sceneMemoryRecycledFrames
        )
    }

    public func restore(_ checkpoint: Wan2CausalWorldCheckpoint) {
        precondition(checkpoint.generatedLatentFrameCount >= 0)
        precondition(checkpoint.retainedLatentFrameCount >= 0)
        precondition(checkpoint.retainedLatentFrameCount <= 3)
        precondition(
            (checkpoint.accumulatedLatents?.dim(1) ?? 0) == checkpoint.retainedLatentFrameCount
        )
        causalState.restore(checkpoint.causalState)
        accumulatedLatents = checkpoint.accumulatedLatents
        generatedLatentFrames = checkpoint.generatedLatentFrameCount
        promptKey = checkpoint.promptKey
        embeddedContext = checkpoint.embeddedContext
        crossCaches = checkpoint.crossCaches
        sceneMemory.restore(checkpoint.sceneMemory)
        precondition(sceneMemory.frameCount == checkpoint.sceneMemoryFrameCount)
        currentWorldPose = checkpoint.currentWorldPose
        sceneMemoryRetrievals = checkpoint.sceneMemoryRetrievalCount
        sceneMemoryRecycledFrames = checkpoint.sceneMemoryRecycledFrameCount
    }

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
        generatedLatentFrames = 0
        sceneMemory.reset()
        currentWorldPose = .identity
        sceneMemoryRetrievals = 0
        sceneMemoryRecycledFrames = 0
    }

    public func unload() {
        tokenizer = nil
        textEncoder = nil
        vae = nil
        transformer = nil
        loadedRootURL = nil
        accumulatedLatents = nil
        generatedLatentFrames = 0
        promptKey = nil
        embeddedContext = nil
        crossCaches = nil
        sceneMemory = Wan2DreamXSceneMemory(policy: sceneMemoryPolicy)
        currentWorldPose = .identity
        sceneMemoryRetrievals = 0
        sceneMemoryRecycledFrames = 0
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
        guard width <= 1_280, height <= 704 else {
            throw Wan2CausalWorldGeneratorError.resolutionExceedsMaximum(width: width, height: height)
        }
        try ensureRoot(resources)
        let latentHeight = height / 16
        let latentWidth = width / 16
        MLXRandom.seed(seed)
        let noisyBlock = MLXRandom.normal([48, 3, latentHeight, latentWidth]).asType(.bfloat16)
        let cameraConditioning = Wan2DreamXARTrajectory.compile(
            control: camera,
            pixelFrameCount: 9,
            speed: Wan2DreamXARTrajectory.defaultSpeed,
            chunkRelative: true
        )
        let worldPoses = Wan2DreamXWorldTrajectory.compile(
            segments: Wan2DreamXARTrajectory.segments(for: camera),
            pixelFrameCount: 9,
            speed: Wan2DreamXARTrajectory.defaultSpeed,
            startingAt: currentWorldPose
        )
        return try await generatePreparedBlock(
            prompt: prompt,
            cameraConditioning: cameraConditioning,
            targetWorldPoses: worldPoses,
            noisyBlock: noisyBlock,
            sourceImageURL: sourceImageURL,
            resources: resources,
            width: width,
            height: height,
            seed: seed,
            progressHandler: progressHandler
        )
    }

    public func generateRollout(
        prompt: String,
        actionSequence: [Wan2DreamXARTrajectorySegment],
        latentFrameCount requestedLatentFrameCount: Int = 21,
        speed: Float = Wan2DreamXARTrajectory.defaultSpeed,
        sourceImageURL: URL? = nil,
        resources: Wan2Resources,
        width: Int = 1_280,
        height: Int = 704,
        seed: UInt64 = 42,
        progressHandler: (@Sendable (GenerationProgress) -> Void)? = nil,
        chunkHandler: ((Wan2CausalRolloutChunk) async throws -> Void)? = nil
    ) async throws -> Wan2VideoGenerationResult {
        guard width > 0, height > 0, width.isMultiple(of: 32), height.isMultiple(of: 32) else {
            throw Wan2CausalWorldGeneratorError.invalidResolution(width: width, height: height)
        }
        guard width <= 1_280, height <= 704 else {
            throw Wan2CausalWorldGeneratorError.resolutionExceedsMaximum(width: width, height: height)
        }
        guard requestedLatentFrameCount > 0, requestedLatentFrameCount.isMultiple(of: 3) else {
            throw Wan2CausalWorldGeneratorError.invalidLatentFrameCount(requestedLatentFrameCount)
        }
        guard requestedLatentFrameCount <= Self.maximumRolloutLatentFrameCount else {
            throw Wan2CausalWorldGeneratorError.latentFrameCountExceedsMaximum(
                requested: requestedLatentFrameCount,
                maximum: Self.maximumRolloutLatentFrameCount
            )
        }
        guard !actionSequence.isEmpty else {
            throw Wan2CausalWorldGeneratorError.emptyActionSequence
        }
        let pixelFrameCount = (requestedLatentFrameCount - 1) * 4 + 1
        let totalActionWeight = actionSequence.reduce(Float(0)) { $0 + $1.weight }
        guard totalActionWeight.isFinite, totalActionWeight > 0,
              actionSequence.count <= pixelFrameCount else {
            throw Wan2CausalWorldGeneratorError.invalidActionSequence
        }
        guard speed.isFinite, speed > 0 else {
            throw Wan2CausalWorldGeneratorError.invalidSpeed(speed)
        }
        try ensureRoot(resources)

        let trajectory = Wan2DreamXARTrajectory.compile(
            segments: actionSequence,
            pixelFrameCount: pixelFrameCount,
            speed: speed,
            chunkRelative: true
        )
        let worldPoses = Wan2DreamXWorldTrajectory.compile(
            segments: actionSequence,
            pixelFrameCount: pixelFrameCount,
            speed: speed,
            startingAt: currentWorldPose
        )
        let latentHeight = height / 16
        let latentWidth = width / 16
        MLXRandom.seed(seed)
        let rolloutNoise = MLXRandom.normal([
            48, requestedLatentFrameCount, latentHeight, latentWidth
        ]).asType(.bfloat16)
        eval(rolloutNoise)

        let blockCount = requestedLatentFrameCount / 3
        let progressStepsPerBlock = 6
        var emittedPixelFrames = 0
        var emittedChunks: [MLXArray] = []
        var terminalFrameLatent: MLXArray?
        emittedChunks.reserveCapacity(blockCount)

        for blockIndex in 0..<blockCount {
            try Task.checkCancellation()
            let latentStart = blockIndex * 3
            let latentEnd = latentStart + 3
            let blockNoise = rolloutNoise[0..., latentStart..<latentEnd, 0..., 0...]
            let blockCamera = trajectory.frames(latentStart..<latentEnd)
            let blockWorldPoses = Array(worldPoses[latentStart..<latentEnd])
            let blockProgress: @Sendable (GenerationProgress) -> Void = { update in
                progressHandler?(GenerationProgress(
                    stage: update.stage,
                    stepIndex: blockIndex * progressStepsPerBlock + update.stepIndex,
                    totalSteps: blockCount * progressStepsPerBlock
                ))
            }
            let result = try await generatePreparedBlock(
                prompt: prompt,
                cameraConditioning: blockCamera,
                targetWorldPoses: blockWorldPoses,
                noisyBlock: blockNoise,
                sourceImageURL: blockIndex == 0 ? sourceImageURL : nil,
                resources: resources,
                width: width,
                height: height,
                seed: seed,
                progressHandler: blockProgress
            )
            terminalFrameLatent = result.terminalFrameLatent
            let emittedFrames = blockIndex == 0
                ? result.frames
                : result.frames[0..., 1..., 0..., 0..., 0...]
            eval(emittedFrames)
            try await chunkHandler?(Wan2CausalRolloutChunk(
                blockIndex: blockIndex,
                blockCount: blockCount,
                pixelFrameStart: emittedPixelFrames,
                frames: emittedFrames
            ))
            emittedPixelFrames += emittedFrames.dim(1)
            emittedChunks.append(emittedFrames)
        }

        let frames = MLX.concatenated(emittedChunks, axis: 1)
        eval(frames)
        progressHandler?(GenerationProgress(
            stage: .decoding,
            stepIndex: blockCount * progressStepsPerBlock,
            totalSteps: blockCount * progressStepsPerBlock
        ))
        return Wan2VideoGenerationResult(
            frames: frames,
            seed: seed,
            terminalFrameLatent: terminalFrameLatent
        )
    }

    private func generatePreparedBlock(
        prompt: String,
        cameraConditioning: Wan2ProjectiveCameraConditioning,
        targetWorldPoses: [Wan2DreamXWorldPose],
        noisyBlock initialNoise: MLXArray,
        sourceImageURL: URL?,
        resources: Wan2Resources,
        width: Int,
        height: Int,
        seed: UInt64,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> Wan2VideoGenerationResult {
        precondition(targetWorldPoses.count == 3)
        let model = try loadedTransformer()
        let vae = try loadedVAE(resources: resources)
        let context = try conditioning(prompt: prompt, resources: resources, model: model)

        let previousLatentFrames = generatedLatentFrames
        let retainedPreviousLatentFrames = retainedLatentFrameCount
        let isFirstBlock = previousLatentFrames == 0
        let sceneMemoryMatches = targetWorldPoses.enumerated().map { offset, pose in
            if isFirstBlock && offset == 0 { return nil as Wan2DreamXSceneMemoryMatch? }
            return sceneMemory.retrieve(
                for: pose,
                targetFrameIndex: previousLatentFrames + offset
            )
        }
        let retrievedFrameCount = sceneMemoryMatches.compactMap { $0 }.count
        sceneMemoryRetrievals += retrievedFrameCount
        if sceneMemoryPolicy.recyclingStrength > 0 {
            sceneMemoryRecycledFrames += retrievedFrameCount
        }
        var noisyBlock = initialNoise
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

        let latentHeight = height / 16
        let latentWidth = width / 16
        let spatialTokens = (latentHeight / 2) * (latentWidth / 2)
        let currentStartToken = previousLatentFrames * spatialTokens
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
            clean = applySceneMemoryAnchors(clean, matches: sceneMemoryMatches)
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
        let decodeLatents = accumulatedLatents.map {
            MLX.concatenated([$0, latents], axis: 1)
        } ?? latents
        eval(decodeLatents)
        sceneMemory.record(
            cleanLatents: latents,
            poses: targetWorldPoses,
            startingAt: previousLatentFrames
        )
        currentWorldPose = targetWorldPoses.last!
        generatedLatentFrames += latents.dim(1)
        accumulatedLatents = decodeLatents[
            0...,
            max(0, decodeLatents.dim(1) - 3)...,
            0...,
            0...
        ]
        eval(accumulatedLatents!)
        Memory.clearCache()

        progressHandler?(GenerationProgress(
            stage: .decoding,
            stepIndex: scheduler.timesteps.count + 1,
            totalSteps: scheduler.timesteps.count + 1
        ))
        let decodePlan = Wan2CausalDecodePlan.make(
            previousLatentFrames: retainedPreviousLatentFrames,
            currentLatentFrames: latents.dim(1)
        )
        let decodeWindow = decodeLatents[
            0...,
            decodePlan.latentStart..<(decodePlan.latentStart + decodePlan.latentCount),
            0...,
            0...
        ]
        let decodedFrames = decode(decodeWindow, vae: vae)
        let allFrames = Wan2DreamXColorStabilizer.process(decodedFrames)
        eval(allFrames)
        let transitionFrames = allFrames[0..., decodePlan.transitionStartPixelFrame...]
        eval(transitionFrames)
        let terminalLatent = decodeLatents[0..., (decodeLatents.dim(1) - 1)...]
        return Wan2VideoGenerationResult(
            frames: transitionFrames,
            seed: seed,
            terminalFrameLatent: terminalLatent
        )
    }

    private func applySceneMemoryAnchors(
        _ clean: MLXArray,
        matches: [Wan2DreamXSceneMemoryMatch?]
    ) -> MLXArray {
        let strength = sceneMemoryPolicy.recyclingStrength
        guard strength > 0, matches.contains(where: { $0 != nil }) else { return clean }
        precondition(clean.dim(1) == matches.count)
        let frames = matches.enumerated().map { index, match -> MLXArray in
            let predicted = clean[0..., index..<(index + 1), 0..., 0...]
            guard let match else { return predicted }
            let effectiveStrength = sceneMemoryPolicy.recyclingStrength(for: match.metadata)
            return predicted * (1 - effectiveStrength)
                + match.cleanLatent.asType(clean.dtype) * effectiveStrength
        }
        let anchored = MLX.concatenated(frames, axis: 1)
        eval(anchored)
        return anchored
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
        let image = try Wan2DreamXImagePreprocessor.resized(decoded, width: width, height: height)
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

private extension Wan2ProjectiveCameraConditioning {
    func frames(_ range: Range<Int>) -> Wan2ProjectiveCameraConditioning {
        precondition(range.lowerBound >= 0 && range.upperBound <= frameCount)
        let viewStart = range.lowerBound * 16
        let viewEnd = range.upperBound * 16
        let intrinsicStart = range.lowerBound * 9
        let intrinsicEnd = range.upperBound * 9
        return Wan2ProjectiveCameraConditioning(
            frameCount: range.count,
            viewMatrices: Array(viewMatrices[viewStart..<viewEnd]),
            intrinsics: Array(intrinsics[intrinsicStart..<intrinsicEnd])
        )
    }
}
