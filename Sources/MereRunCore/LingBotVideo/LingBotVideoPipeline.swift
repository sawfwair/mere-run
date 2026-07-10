import Foundation
import MLX
import MLXNN

public struct LingBotVideoGenerationOptions: Sendable {
    public let prompt: String
    public let negativePrompt: String
    public let width: Int
    public let height: Int
    public let numFrames: Int
    public let fps: Int
    public let steps: Int
    public let guidanceScale: Float
    public let shift: Float
    public let batchCFG: Bool
    public let seed: Int
    public let temporalProbe: Bool
    public let temporalProbeStep: Int
    public let runRefiner: Bool
    public let refinerWidth: Int?
    public let refinerHeight: Int?
    public let refinerSteps: Int
    public let refinerGuidanceScale: Float
    public let refinerShift: Float
    public let refinerThreshold: Float
    public let refinerSigmaTailSteps: Int
    public let refinerBatchCFG: Bool

    public init(
        prompt: String,
        negativePrompt: String = LingBotVideoPipeline.defaultNegativePrompt,
        width: Int = 320,
        height: Int = 192,
        numFrames: Int = 9,
        fps: Int = 24,
        steps: Int = 40,
        guidanceScale: Float = 3,
        shift: Float = 3,
        batchCFG: Bool = false,
        seed: Int = 42,
        temporalProbe: Bool = false,
        temporalProbeStep: Int = 4,
        runRefiner: Bool = false,
        refinerWidth: Int? = nil,
        refinerHeight: Int? = nil,
        refinerSteps: Int = 8,
        refinerGuidanceScale: Float = 3,
        refinerShift: Float = 3,
        refinerThreshold: Float = 0.85,
        refinerSigmaTailSteps: Int = 2,
        refinerBatchCFG: Bool = false
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.shift = shift
        self.batchCFG = batchCFG
        self.seed = seed
        self.temporalProbe = temporalProbe
        self.temporalProbeStep = temporalProbeStep
        self.runRefiner = runRefiner
        self.refinerWidth = refinerWidth
        self.refinerHeight = refinerHeight
        self.refinerSteps = refinerSteps
        self.refinerGuidanceScale = refinerGuidanceScale
        self.refinerShift = refinerShift
        self.refinerThreshold = refinerThreshold
        self.refinerSigmaTailSteps = refinerSigmaTailSteps
        self.refinerBatchCFG = refinerBatchCFG
    }

    public var resolvedRefinerWidth: Int {
        refinerWidth ?? 1_920
    }

    public var resolvedRefinerHeight: Int {
        refinerHeight ?? 1_088
    }
}

public struct LingBotVideoGenerationResult: @unchecked Sendable {
    public let frames: MLXArray
    public let latents: MLXArray
    public let refined: Bool
    public let temporalProbe: Bool

    public init(
        frames: MLXArray,
        latents: MLXArray,
        refined: Bool = false,
        temporalProbe: Bool = false
    ) {
        self.frames = frames
        self.latents = latents
        self.refined = refined
        self.temporalProbe = temporalProbe
    }
}

public struct LingBotVideoTemporalMetrics: Sendable, Hashable {
    public static let unstableMeanLumaDelta: Float = 18
    public static let unstablePeakLumaDelta: Float = 45

    public let meanLumaDelta: Float
    public let peakLumaDelta: Float
    public let lumaStandardDeviation: Float

    public var isInformative: Bool {
        lumaStandardDeviation >= 12
    }

    public var isLikelyUnstable: Bool {
        meanLumaDelta > Self.unstableMeanLumaDelta
            || peakLumaDelta > Self.unstablePeakLumaDelta
    }

    public static func analyze(frames: MLXArray) -> LingBotVideoTemporalMetrics {
        precondition(frames.ndim == 5, "LingBot temporal metrics expect [B,T,H,W,C] frames.")
        precondition(frames.dim(0) == 1, "LingBot temporal metrics expect batch size one.")
        precondition(frames.dim(4) == 3, "LingBot temporal metrics expect RGB frames.")
        guard frames.dim(1) > 1 else {
            return LingBotVideoTemporalMetrics(
                meanLumaDelta: 0,
                peakLumaDelta: 0,
                lumaStandardDeviation: 0
            )
        }

        let values = frames.asType(.float32)
        let luma = values[0..., 0..., 0..., 0..., 0] * 0.2126
            + values[0..., 0..., 0..., 0..., 1] * 0.7152
            + values[0..., 0..., 0..., 0..., 2] * 0.0722
        let current = luma[0, 1..., 0..., 0...]
        let previous = luma[0, 0..<(frames.dim(1) - 1), 0..., 0...]
        let transitionDeltas = MLX.abs(current - previous)
            .reshaped(frames.dim(1) - 1, -1)
            .mean(axis: 1)
        MLX.eval(transitionDeltas)
        return LingBotVideoTemporalMetrics(
            meanLumaDelta: transitionDeltas.mean().item(Float.self),
            peakLumaDelta: MLX.max(transitionDeltas).item(Float.self),
            lumaStandardDeviation: MLX.std(luma).item(Float.self)
        )
    }
}

public struct LingBotVideoGenerationProgress: Sendable, Hashable {
    public enum Branch: String, Sendable, Hashable {
        case conditional
        case unconditional
        case batchedCFG = "batched-cfg"
    }

    public enum Stage: String, Sendable, Hashable {
        case encodingPrompt
        case loadingTransformer
        case denoising
        case loadingVAE
        case preparingRefiner
        case loadingRefiner
        case refining
        case decodingTemporalProbe
        case decoding
    }

    public let stage: Stage
    public let stepIndex: Int
    public let totalSteps: Int
    public let branch: Branch?
    public let blockIndex: Int
    public let totalBlocks: Int

    public init(
        stage: Stage,
        stepIndex: Int,
        totalSteps: Int,
        branch: Branch? = nil,
        blockIndex: Int = 0,
        totalBlocks: Int = 0
    ) {
        self.stage = stage
        self.stepIndex = stepIndex
        self.totalSteps = totalSteps
        self.branch = branch
        self.blockIndex = blockIndex
        self.totalBlocks = totalBlocks
    }
}

public actor LingBotVideoPipeline {
    public enum PipelineError: LocalizedError {
        case emptyPrompt
        case invalidDimensions(Int, Int)
        case invalidFrameCount(Int)
        case invalidFPS(Int)
        case invalidGuidance(Float)
        case invalidShift(Float)
        case invalidTemporalProbeStep(Int)
        case invalidRefinerDimensions(Int, Int)
        case invalidRefinerSteps(Int)
        case invalidRefinerGuidance(Float)
        case invalidRefinerShift(Float)
        case invalidRefinerThreshold(Float)
        case invalidRefinerTailSteps(Int)
        case incompatibleTransformerWeights(String)
        case refinerResizeFailed([Int], expectedHeight: Int, expectedWidth: Int)

        public var errorDescription: String? {
            switch self {
            case .emptyPrompt:
                return "LingBot-Video prompt cannot be empty."
            case .invalidDimensions(let width, let height):
                return "LingBot-Video width and height must be positive multiples of 16 (got \(width)x\(height))."
            case .invalidFrameCount(let count):
                return "LingBot-Video frame count must be 4n+1 and at least 5 (got \(count))."
            case .invalidFPS(let fps):
                return "LingBot-Video FPS must be >= 1 (got \(fps))."
            case .invalidGuidance(let guidance):
                return "LingBot-Video guidance scale must be > 0 (got \(guidance))."
            case .invalidShift(let shift):
                return "LingBot-Video scheduler shift must be > 0 (got \(shift))."
            case .invalidTemporalProbeStep(let step):
                return "LingBot-Video temporal probe step must be between 1 and --steps (got \(step))."
            case .invalidRefinerDimensions(let width, let height):
                return "LingBot-Video refiner width and height must be positive multiples of 16 (got \(width)x\(height))."
            case .invalidRefinerSteps(let steps):
                return "LingBot-Video refiner steps must be >= 1 (got \(steps))."
            case .invalidRefinerGuidance(let guidance):
                return "LingBot-Video refiner guidance scale must be > 0 (got \(guidance))."
            case .invalidRefinerShift(let shift):
                return "LingBot-Video refiner scheduler shift must be > 0 (got \(shift))."
            case .invalidRefinerThreshold(let threshold):
                return "LingBot-Video refiner threshold must be in (0, 1] (got \(threshold))."
            case .invalidRefinerTailSteps(let steps):
                return "LingBot-Video refiner sigma tail steps must be >= 0 (got \(steps))."
            case .incompatibleTransformerWeights(let message):
                return "LingBot-Video transformer checkpoint does not match the native module: \(message)"
            case .refinerResizeFailed(let shape, let height, let width):
                return "LingBot-Video refiner resize produced \(shape), expected spatial size \(width)x\(height)."
            }
        }
    }

    public static let defaultNegativePrompt = #"{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}"#

    public init() {}

    public func generate(
        modelRoot: URL,
        options: LingBotVideoGenerationOptions,
        progressHandler: (@Sendable (LingBotVideoGenerationProgress) -> Void)? = nil
    ) throws -> LingBotVideoGenerationResult {
        try Self.validate(options)
        let resources = try LingBotVideoResources(rootURL: modelRoot)
        try resources.validateForInference()
        if options.runRefiner, !options.temporalProbe {
            try resources.validateForRefiner()
        }

        progressHandler?(.init(stage: .encodingPrompt, stepIndex: 0, totalSteps: 1))
        let promptEmbeddings: MLXArray
        let negativeEmbeddings: MLXArray?
        do {
            let encoder = try LingBotVideoPromptEncoder(resources: resources)
            promptEmbeddings = try encoder.encode(options.prompt)
            negativeEmbeddings = options.guidanceScale > 1
                ? try encoder.encode(options.negativePrompt)
                : nil
            if let negativeEmbeddings {
                MLX.eval(promptEmbeddings, negativeEmbeddings)
            } else {
                MLX.eval(promptEmbeddings)
            }
        }
        Memory.clearCache()

        let probeVAE: AutoencoderKL3D?
        if options.temporalProbe {
            progressHandler?(.init(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
            let vae = try Self.loadVAE(resources: resources)
            MLX.eval(vae)
            Memory.clearCache()
            probeVAE = vae
        } else {
            probeVAE = nil
        }

        let latentFrames = (options.numFrames - 1) / 4 + 1
        let latentHeight = options.height / 8
        let latentWidth = options.width / 8
        var latents = MLXRandom.normal(
            [1, resources.transformerConfig.inChannels, latentFrames, latentHeight, latentWidth],
            key: MLXRandom.key(UInt64(bitPattern: Int64(options.seed)))
        ).asType(.float32)
        MLX.eval(latents)

        progressHandler?(.init(stage: .loadingTransformer, stepIndex: 0, totalSteps: 1))
        do {
            let transformer = LingBotVideoTransformer(
                config: resources.transformerConfig,
                quantization: resources.quantizationConfig
            )
            try Self.loadTransformerWeights(
                transformer,
                componentURL: resources.transformerURL,
                config: resources.transformerConfig
            )
            transformer.prepareInferenceCaches()
            MLX.eval(transformer)
            Memory.clearCache()

            let scheduler = LingBotVideoFlowUniPCScheduler()
            try scheduler.setTimesteps(stepCount: options.steps, shift: options.shift)
            latents = try Self.denoise(
                transformer: transformer,
                latents: latents,
                promptEmbeddings: promptEmbeddings,
                negativeEmbeddings: negativeEmbeddings,
                guidanceScale: options.guidanceScale,
                batchCFG: options.batchCFG,
                scheduler: scheduler,
                stage: .denoising,
                stopAfterStep: options.temporalProbe ? options.temporalProbeStep : nil,
                progressHandler: progressHandler
            )
        }
        MLX.eval(latents)
        Memory.clearCache()

        if let probeVAE {
            progressHandler?(.init(stage: .decodingTemporalProbe, stepIndex: 1, totalSteps: 1))
            let temporalProbeFrames = Self.decodeFrames(
                latents: latents,
                vae: probeVAE,
                vaeConfig: resources.vaeConfig
            )
            MLX.eval(temporalProbeFrames)
            Memory.clearCache()
            return LingBotVideoGenerationResult(
                frames: temporalProbeFrames,
                latents: latents,
                temporalProbe: true
            )
        }

        progressHandler?(.init(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        let vae = try Self.loadVAE(resources: resources)
        MLX.eval(vae)
        Memory.clearCache()

        if options.runRefiner {
            progressHandler?(.init(stage: .preparingRefiner, stepIndex: 0, totalSteps: 1))
            let refinerKeys = MLXRandom.split(
                key: MLXRandom.key(UInt64(bitPattern: Int64(options.seed))),
                into: 2
            )
            latents = try Self.prepareRefinerLatents(
                baseLatents: latents,
                vae: vae,
                vaeConfig: resources.vaeConfig,
                targetHeight: options.resolvedRefinerHeight,
                targetWidth: options.resolvedRefinerWidth,
                threshold: options.refinerThreshold,
                posteriorKey: refinerKeys[0],
                noiseKey: refinerKeys[1]
            )
            MLX.eval(latents)
            Memory.clearCache()

            guard let refinerURL = resources.refinerURL,
                  let refinerConfig = resources.refinerConfig
            else {
                throw LingBotVideoResources.ResourceError.missingFile(
                    resources.rootURL.appendingPathComponent("refiner/config.json")
                )
            }
            progressHandler?(.init(stage: .loadingRefiner, stepIndex: 0, totalSteps: 1))
            do {
                let refiner = LingBotVideoTransformer(
                    config: refinerConfig,
                    quantization: resources.quantizationConfig
                )
                try Self.loadTransformerWeights(
                    refiner,
                    componentURL: refinerURL,
                    config: refinerConfig
                )
                refiner.prepareInferenceCaches()
                MLX.eval(refiner)
                Memory.clearCache()

                let scheduler = LingBotVideoFlowUniPCScheduler()
                let sigmas = try LingBotVideoFlowUniPCScheduler.refinerSigmas(
                    stepCount: options.refinerSteps,
                    shift: options.refinerShift,
                    threshold: options.refinerThreshold,
                    tailStepCount: options.refinerSigmaTailSteps
                )
                try scheduler.setTimesteps(sigmas: sigmas)
                let nullEmbeddings = options.refinerGuidanceScale > 1
                    ? MLX.zeros(promptEmbeddings.shape, dtype: promptEmbeddings.dtype)
                    : nil
                latents = try Self.denoise(
                    transformer: refiner,
                    latents: latents,
                    promptEmbeddings: promptEmbeddings,
                    negativeEmbeddings: nullEmbeddings,
                    guidanceScale: options.refinerGuidanceScale,
                    batchCFG: options.refinerBatchCFG,
                    scheduler: scheduler,
                    stage: .refining,
                    stopAfterStep: nil,
                    progressHandler: progressHandler
                )
            }
            MLX.eval(latents)
            Memory.clearCache()
        }

        progressHandler?(.init(stage: .decoding, stepIndex: 0, totalSteps: 1))
        let frames = Self.decodeFrames(latents: latents, vae: vae, vaeConfig: resources.vaeConfig)
        MLX.eval(frames)
        Memory.clearCache()
        return LingBotVideoGenerationResult(
            frames: frames,
            latents: latents,
            refined: options.runRefiner
        )
    }

    private static func validate(_ options: LingBotVideoGenerationOptions) throws {
        guard !options.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw PipelineError.emptyPrompt
        }
        guard options.width > 0,
              options.height > 0,
              options.width % 16 == 0,
              options.height % 16 == 0
        else {
            throw PipelineError.invalidDimensions(options.width, options.height)
        }
        guard options.numFrames >= 5, (options.numFrames - 1) % 4 == 0 else {
            throw PipelineError.invalidFrameCount(options.numFrames)
        }
        guard options.fps >= 1 else {
            throw PipelineError.invalidFPS(options.fps)
        }
        guard options.guidanceScale > 0 else {
            throw PipelineError.invalidGuidance(options.guidanceScale)
        }
        guard options.shift > 0 else {
            throw PipelineError.invalidShift(options.shift)
        }
        guard !options.temporalProbe
                || (options.temporalProbeStep >= 1 && options.temporalProbeStep <= options.steps)
        else {
            throw PipelineError.invalidTemporalProbeStep(options.temporalProbeStep)
        }
        guard !options.runRefiner || (
            options.resolvedRefinerWidth > 0
                && options.resolvedRefinerHeight > 0
                && options.resolvedRefinerWidth % 16 == 0
                && options.resolvedRefinerHeight % 16 == 0
        ) else {
            throw PipelineError.invalidRefinerDimensions(
                options.resolvedRefinerWidth,
                options.resolvedRefinerHeight
            )
        }
        guard !options.runRefiner || options.refinerSteps >= 1 else {
            throw PipelineError.invalidRefinerSteps(options.refinerSteps)
        }
        guard !options.runRefiner || options.refinerGuidanceScale > 0 else {
            throw PipelineError.invalidRefinerGuidance(options.refinerGuidanceScale)
        }
        guard !options.runRefiner || options.refinerShift > 0 else {
            throw PipelineError.invalidRefinerShift(options.refinerShift)
        }
        guard !options.runRefiner || (options.refinerThreshold > 0 && options.refinerThreshold <= 1) else {
            throw PipelineError.invalidRefinerThreshold(options.refinerThreshold)
        }
        guard !options.runRefiner || options.refinerSigmaTailSteps >= 0 else {
            throw PipelineError.invalidRefinerTailSteps(options.refinerSigmaTailSteps)
        }
    }

    private static func denoise(
        transformer: LingBotVideoTransformer,
        latents initialLatents: MLXArray,
        promptEmbeddings: MLXArray,
        negativeEmbeddings: MLXArray?,
        guidanceScale: Float,
        batchCFG: Bool,
        scheduler: LingBotVideoFlowUniPCScheduler,
        stage: LingBotVideoGenerationProgress.Stage,
        stopAfterStep: Int?,
        progressHandler: (@Sendable (LingBotVideoGenerationProgress) -> Void)?
    ) throws -> MLXArray {
        var latents = initialLatents
        let totalSteps = scheduler.timesteps.count
        let batchedConditioning = batchCFG
            ? negativeEmbeddings.map {
                batchedCFGConditioning(positive: promptEmbeddings, negative: $0)
            }
            : nil
        for (index, timestep) in scheduler.timesteps.enumerated() {
            progressHandler?(.init(stage: stage, stepIndex: index + 1, totalSteps: totalSteps))
            let timestepArray = (
                MLXArray([timestep / 1000]).asType(.bfloat16) * 1000
            ).asType(.float32)
            let prediction: MLXArray
            if let batchedConditioning {
                let batchedLatents = MLX.concatenated([latents, latents], axis: 0)
                let batchedTimestep = MLX.concatenated([timestepArray, timestepArray], axis: 0)
                let batchedPrediction = transformer(
                    hiddenStates: batchedLatents,
                    timestep: batchedTimestep,
                    encoderHiddenStates: batchedConditioning.embeddings,
                    encoderAttentionMask: batchedConditioning.mask,
                    encoderTextLengths: batchedConditioning.textLengths,
                    blockProgressInterval: 4,
                    blockProgressHandler: progressHandler.map { handler in
                        { blockIndex, totalBlocks in
                            handler(.init(
                                stage: stage,
                                stepIndex: index + 1,
                                totalSteps: totalSteps,
                                branch: .batchedCFG,
                                blockIndex: blockIndex,
                                totalBlocks: totalBlocks
                            ))
                        }
                    }
                )
                MLX.eval(batchedPrediction)
                let conditional = batchedPrediction[0..<1]
                let unconditional = batchedPrediction[1..<2]
                prediction = unconditional + guidanceScale * (conditional - unconditional)
            } else {
                let conditional = transformer(
                    hiddenStates: latents,
                    timestep: timestepArray,
                    encoderHiddenStates: promptEmbeddings,
                    blockProgressInterval: 4,
                    blockProgressHandler: progressHandler.map { handler in
                        { blockIndex, totalBlocks in
                            handler(.init(
                                stage: stage,
                                stepIndex: index + 1,
                                totalSteps: totalSteps,
                                branch: .conditional,
                                blockIndex: blockIndex,
                                totalBlocks: totalBlocks
                            ))
                        }
                    }
                )
                MLX.eval(conditional)

                if let negativeEmbeddings {
                let unconditional = transformer(
                    hiddenStates: latents,
                    timestep: timestepArray,
                    encoderHiddenStates: negativeEmbeddings,
                    blockProgressInterval: 4,
                    blockProgressHandler: progressHandler.map { handler in
                        { blockIndex, totalBlocks in
                            handler(.init(
                                stage: stage,
                                stepIndex: index + 1,
                                totalSteps: totalSteps,
                                branch: .unconditional,
                                blockIndex: blockIndex,
                                totalBlocks: totalBlocks
                            ))
                        }
                    }
                )
                MLX.eval(unconditional)
                prediction = unconditional + guidanceScale * (conditional - unconditional)
                } else {
                    prediction = conditional
                }
            }
            MLX.eval(prediction)
            if let stopAfterStep, index + 1 == stopAfterStep {
                let estimate = latents - Float(scheduler.sigmas[index]) * prediction
                MLX.eval(estimate)
                Memory.clearCache()
                return estimate
            }
            latents = try scheduler.step(modelOutput: prediction, sample: latents)
            MLX.eval(latents)
            Memory.clearCache()
        }
        return latents
    }

    private static func batchedCFGConditioning(
        positive: MLXArray,
        negative: MLXArray
    ) -> (embeddings: MLXArray, mask: MLXArray, textLengths: [Int]) {
        precondition(positive.dim(0) == 1 && negative.dim(0) == 1)
        precondition(positive.dim(2) == negative.dim(2))
        let positiveLength = positive.dim(1)
        let negativeLength = negative.dim(1)
        let maximumLength = max(positiveLength, negativeLength)

        func paddedEmbeddings(_ value: MLXArray, length: Int) -> MLXArray {
            guard length < maximumLength else { return value }
            return MLX.padded(
                value,
                widths: [[0, 0], [0, maximumLength - length], [0, 0]]
            )
        }

        let embeddings = MLX.concatenated([
            paddedEmbeddings(positive, length: positiveLength),
            paddedEmbeddings(negative, length: negativeLength),
        ], axis: 0)
        let positiveMask = [Int32](repeating: 1, count: positiveLength)
            + [Int32](repeating: 0, count: maximumLength - positiveLength)
        let negativeMask = [Int32](repeating: 1, count: negativeLength)
            + [Int32](repeating: 0, count: maximumLength - negativeLength)
        let mask = MLXArray(positiveMask + negativeMask, [2, maximumLength])
        return (embeddings, mask, [positiveLength, negativeLength])
    }

    private static func prepareRefinerLatents(
        baseLatents: MLXArray,
        vae: AutoencoderKL3D,
        vaeConfig: LingBotVideoVAEConfig,
        targetHeight: Int,
        targetWidth: Int,
        threshold: Float,
        posteriorKey: MLXArray,
        noiseKey: MLXArray
    ) throws -> MLXArray {
        let decoded = decodeNormalized(latents: baseLatents, vae: vae, vaeConfig: vaeConfig)
        MLX.eval(decoded)

        let batch = decoded.dim(0)
        let frames = decoded.dim(2)
        let sourceHeight = decoded.dim(3)
        let sourceWidth = decoded.dim(4)
        let flattened = decoded
            .transposed(0, 2, 3, 4, 1)
            .reshaped(batch * frames, sourceHeight, sourceWidth, 3)
        let resizedFlat = Upsample(
            scaleFactor: .array([
                Float(targetHeight) / Float(sourceHeight),
                Float(targetWidth) / Float(sourceWidth),
            ]),
            mode: .cubic(alignCorners: false)
        )(flattened)
        guard resizedFlat.dim(1) == targetHeight, resizedFlat.dim(2) == targetWidth else {
            throw PipelineError.refinerResizeFailed(
                resizedFlat.shape,
                expectedHeight: targetHeight,
                expectedWidth: targetWidth
            )
        }
        let resized = MLX.clip(resizedFlat, min: 0, max: 1)
            .reshaped(batch, frames, targetHeight, targetWidth, 3)
            .transposed(0, 4, 1, 2, 3)
        let vaeLatents = vae.encodeSampled(resized * 2 - 1, key: posteriorKey).asType(.float32)
        let mean = MLXArray(vaeConfig.latentsMean).reshaped(1, vaeConfig.zDim, 1, 1, 1)
        let standardDeviation = MLXArray(vaeConfig.latentsStd).reshaped(1, vaeConfig.zDim, 1, 1, 1)
        let upscaledLatents = (vaeLatents - mean) / standardDeviation
        let noise = MLXRandom.normal(upscaledLatents.shape, key: noiseKey).asType(.float32)
        let initialLatents = (1 - threshold) * upscaledLatents + threshold * noise
        MLX.eval(initialLatents)
        return initialLatents
    }

    private static func loadVAE(resources: LingBotVideoResources) throws -> AutoencoderKL3D {
        let vaeConfig = resources.vaeConfig
        let vae = AutoencoderKL3D(config: VAE3DConfig(
            inChannels: 3,
            outChannels: 3,
            latentChannels: vaeConfig.zDim,
            blockOutChannels: vaeConfig.blockOutChannels,
            layersPerBlock: vaeConfig.numResBlocks,
            normNumGroups: 32,
            scalingFactor: 1,
            shiftFactor: 0,
            temporalCompressionRatio: vaeConfig.temporalCompressionRatio,
            midBlockAddAttention: true
        ))
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: resources.vaeURL.appendingPathComponent("diffusion_pytorch_model.safetensors"),
            to: vae,
            dtype: nil,
            verify: [.shapeMismatch],
            include: { _ in true },
            mapper: QwenImageEditVAE.weightMapper,
            batchSize: 8
        )
        return vae
    }

    private static func decodeNormalized(
        latents: MLXArray,
        vae: AutoencoderKL3D,
        vaeConfig: LingBotVideoVAEConfig
    ) -> MLXArray {
        let mean = MLXArray(vaeConfig.latentsMean).reshaped(1, vaeConfig.zDim, 1, 1, 1)
        let standardDeviation = MLXArray(vaeConfig.latentsStd).reshaped(1, vaeConfig.zDim, 1, 1, 1)
        let vaeLatents = latents.asType(.float32) * standardDeviation + mean
        let decoded = vae.decodeStreaming(vaeLatents)
        return MLX.clip((decoded.asType(.float32) + 1) / 2, min: 0, max: 1)
    }

    private static func decodeFrames(
        latents: MLXArray,
        vae: AutoencoderKL3D,
        vaeConfig: LingBotVideoVAEConfig
    ) -> MLXArray {
        let normalized = decodeNormalized(latents: latents, vae: vae, vaeConfig: vaeConfig)
        return (normalized.transposed(0, 2, 3, 4, 1) * 255).asType(.uint8)
    }

    private static func loadTransformerWeights(
        _ transformer: LingBotVideoTransformer,
        componentURL: URL,
        config: LingBotVideoTransformerConfig
    ) throws {
        let mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in
            LingBotVideoTransformer.mapWeight(key: key, value: value, config: config)
        }
        let single = componentURL.appendingPathComponent("diffusion_pytorch_model.safetensors")
        if FileManager.default.fileExists(atPath: single.path) {
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: single,
                to: transformer,
                dtype: nil,
                verify: [.shapeMismatch],
                include: { _ in true },
                mapper: mapper,
                batchSize: 8
            )
            return
        }

        let indexURL = componentURL.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: Data(contentsOf: indexURL))
        let checkpointKeys = Set(index.weightMap.keys.map(LingBotVideoTransformer.mapWeightKey))
        let parameterKeys = Set(transformer.parameters().flattened().map(\.0))
        let missing = parameterKeys.subtracting(checkpointKeys).sorted()
        let unexpected = checkpointKeys.subtracting(parameterKeys).sorted()
        if !missing.isEmpty || !unexpected.isEmpty {
            let missingSummary = missing.prefix(3).joined(separator: ", ")
            let unexpectedSummary = unexpected.prefix(3).joined(separator: ", ")
            throw PipelineError.incompatibleTransformerWeights(
                "missing=[\(missingSummary)] unexpected=[\(unexpectedSummary)]"
            )
        }
        for shard in index.shardFilenames {
            try SafetensorsStreamingLoader.applyWeightsStreaming(
                url: componentURL.appendingPathComponent(shard),
                to: transformer,
                dtype: nil,
                verify: .none,
                include: { _ in true },
                mapper: mapper,
                batchSize: 8
            )
            Memory.clearCache()
        }
    }
}
