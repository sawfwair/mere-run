import Foundation
import MediaIO
import MLX
import MLXRandom

public enum Cosmos3GenerationMode: String, Codable, CaseIterable, Sendable {
    case textToImage = "text_to_image"
    case imageToImage = "image_to_image"
    case textToVideo = "text_to_video"
    case imageToVideo = "image_to_video"
    case videoToVideo = "video_to_video"
    case policy
    case forwardDynamics = "forward_dynamics"
    case inverseDynamics = "inverse_dynamics"
}

public enum Cosmos3EdgeGeneratorError: LocalizedError, Sendable {
    case invalidResolution(width: Int, height: Int)
    case invalidFrameCount(Int)
    case invalidFPS(Int)
    case invalidStepCount(Int)
    case conflictingConditioning
    case missingFile(URL)
    case emptyVideo(URL)
    case missingModelFiles([URL])

    public var errorDescription: String? {
        switch self {
        case .invalidResolution(let width, let height):
            return "Cosmos3 resolution must be positive and divisible by 16; received \(width)x\(height)."
        case .invalidFrameCount(let count):
            return "Cosmos3 video frame count must be 4n+1 (or 1 for image generation); received \(count)."
        case .invalidFPS(let fps):
            return "Cosmos3 frames per second must be positive; received \(fps)."
        case .invalidStepCount(let count):
            return "Cosmos3 denoising steps must be positive; received \(count)."
        case .conflictingConditioning:
            return "Cosmos3 accepts either image, video, or action conditioning in one request."
        case .missingFile(let url):
            return "Cosmos3 conditioning file does not exist: \(url.path)"
        case .emptyVideo(let url):
            return "Cosmos3 conditioning video has no decodable frames: \(url.path)"
        case .missingModelFiles(let urls):
            return "Cosmos3 model root is incomplete: \(urls.map(\.path).joined(separator: ", "))."
        }
    }
}

public struct Cosmos3GenerationOptions: Hashable, Sendable {
    public let prompt: String
    public let negativePrompt: String
    public let outputURL: URL?
    public let imageURL: URL?
    public let videoURL: URL?
    public let action: Cosmos3ActionCondition?
    public let conditionedVideoLatentFrames: [Int]
    public let keepsVideoTail: Bool
    public let width: Int
    public let height: Int
    public let numFrames: Int
    public let steps: Int
    public let guidanceScale: Float
    public let shift: Float
    public let schedule: Cosmos3UniPCSchedule
    public let seed: UInt64
    public let fps: Int
    public let useSystemPrompt: Bool

    public init(
        prompt: String,
        negativePrompt: String = "",
        outputURL: URL? = nil,
        imageURL: URL? = nil,
        videoURL: URL? = nil,
        action: Cosmos3ActionCondition? = nil,
        conditionedVideoLatentFrames: [Int] = [0, 1],
        keepsVideoTail: Bool = false,
        width: Int = 1_280,
        height: Int = 720,
        numFrames: Int = 189,
        steps: Int? = nil,
        guidanceScale: Float? = nil,
        shift: Float? = nil,
        schedule: Cosmos3UniPCSchedule = .nvidiaShiftedFlow,
        seed: UInt64 = 0,
        fps: Int? = nil,
        useSystemPrompt: Bool = false
    ) throws {
        guard width > 0, height > 0, width % 16 == 0, height % 16 == 0 else {
            throw Cosmos3EdgeGeneratorError.invalidResolution(width: width, height: height)
        }
        guard numFrames == 1 || (numFrames > 1 && (numFrames - 1) % 4 == 0) else {
            throw Cosmos3EdgeGeneratorError.invalidFrameCount(numFrames)
        }
        let isImageEdit = action == nil && imageURL != nil && numFrames == 1
        let resolvedSteps = steps ?? (action == nil
            ? (isImageEdit ? 35 : (numFrames == 1 ? 50 : 35))
            : 30)
        let resolvedGuidance = guidanceScale ?? (action == nil
            ? (isImageEdit ? 6 : (numFrames == 1 ? 4 : 6))
            : 1)
        let resolvedShift = shift ?? (action == nil
            ? (isImageEdit ? 5 : (numFrames == 1 ? 3 : 10))
            : 3)
        let resolvedFPS = fps ?? (action == nil ? 24 : 30)
        guard resolvedSteps > 0 else {
            throw Cosmos3EdgeGeneratorError.invalidStepCount(resolvedSteps)
        }
        guard resolvedFPS > 0 else {
            throw Cosmos3EdgeGeneratorError.invalidFPS(resolvedFPS)
        }
        let topLevelConditions = [imageURL != nil, videoURL != nil].filter { $0 }.count
        guard action == nil || topLevelConditions == 0 else {
            throw Cosmos3EdgeGeneratorError.conflictingConditioning
        }
        guard topLevelConditions <= 1 else {
            throw Cosmos3EdgeGeneratorError.conflictingConditioning
        }
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.outputURL = outputURL
        self.imageURL = imageURL
        self.videoURL = videoURL
        self.action = action
        self.conditionedVideoLatentFrames = conditionedVideoLatentFrames
        self.keepsVideoTail = keepsVideoTail
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.steps = resolvedSteps
        self.guidanceScale = resolvedGuidance
        self.shift = resolvedShift
        self.schedule = schedule
        self.seed = seed
        self.fps = resolvedFPS
        self.useSystemPrompt = useSystemPrompt
    }

    public var mode: Cosmos3GenerationMode {
        if let action {
            switch action.mode {
            case .policy: return .policy
            case .forwardDynamics: return .forwardDynamics
            case .inverseDynamics: return .inverseDynamics
            }
        }
        if videoURL != nil { return .videoToVideo }
        if imageURL != nil && numFrames > 1 { return .imageToVideo }
        if imageURL != nil { return .imageToImage }
        return numFrames == 1 ? .textToImage : .textToVideo
    }
}

public struct Cosmos3GenerationResult: @unchecked Sendable {
    public let mode: Cosmos3GenerationMode
    public let frames: MLXArray
    public let actions: MLXArray?
    public let terminalFrameLatent: MLXArray
    public let seed: UInt64
    public let width: Int
    public let height: Int

    public init(
        mode: Cosmos3GenerationMode,
        frames: MLXArray,
        actions: MLXArray?,
        terminalFrameLatent: MLXArray,
        seed: UInt64,
        width: Int,
        height: Int
    ) {
        self.mode = mode
        self.frames = frames
        self.actions = actions
        self.terminalFrameLatent = terminalFrameLatent
        self.seed = seed
        self.width = width
        self.height = height
    }
}

public final class Cosmos3EdgeGenerator: @unchecked Sendable {
    private struct PreparedConditioning {
        let cleanVision: MLXArray
        let conditionedVisionFrames: [Int]
        let initialActions: MLXArray?
        let cleanActions: MLXArray?
        let conditionedActionFrames: [Int]
        let actionDomain: Cosmos3ActionDomain?
        let rawActionDimension: Int?
        let width: Int
        let height: Int
    }

    private var loadedRootURL: URL?
    private var tokenizer: Cosmos3Tokenizer?
    private var transformer: Cosmos3OmniTransformerModel?
    private var vae: Wan2VAEModel?
    private var chainedTerminalFrameLatent: MLXArray?

    public init() {}

    public var isWarm: Bool {
        tokenizer != nil && transformer != nil && vae != nil
    }

    public var hasChainedTerminalFrameLatent: Bool {
        chainedTerminalFrameLatent != nil
    }

    public func clearChain() {
        chainedTerminalFrameLatent = nil
    }

    public func prepare(
        resources: Cosmos3Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) async throws {
        try ensureRoot(resources)
        if tokenizer == nil {
            progress?("Loading Cosmos3-Edge tokenizer")
            tokenizer = try await Cosmos3Tokenizer.load(resources: resources)
        }
        if vae == nil {
            vae = try Cosmos3ModelLoader.loadVAE(resources: resources, progress: progress)
        }
        if transformer == nil {
            transformer = try Cosmos3ModelLoader.loadTransformer(
                resources: resources,
                progress: progress
            )
        }
    }

    public func unload() {
        tokenizer = nil
        transformer = nil
        vae = nil
        loadedRootURL = nil
        chainedTerminalFrameLatent = nil
        Memory.clearCache()
    }

    public func generate(
        options: Cosmos3GenerationOptions,
        resources: Cosmos3Resources,
        useChainedSourceLatent: Bool = false,
        progressHandler: (@Sendable (GenerationProgress) -> Void)? = nil
    ) async throws -> Cosmos3GenerationResult {
        try ensureRoot(resources)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw Cosmos3EdgeGeneratorError.missingModelFiles(missing)
        }
        try await prepare(resources: resources)
        MLXRandom.seed(options.seed)
        let effectiveFrames = options.action.map { $0.chunkSize + 1 } ?? options.numFrames
        let conditioning = try prepareConditioning(
            options: options,
            effectiveFrames: effectiveFrames,
            resources: resources,
            useChainedSourceLatent: useChainedSourceLatent
        )
        if options.mode == .imageToImage {
            return try generateImageEdit(
                options: options,
                conditioning: conditioning,
                progressHandler: progressHandler
            )
        }
        let promptPair = try tokenizer!.encode(
            prompt: options.prompt,
            negativePrompt: options.negativePrompt,
            numFrames: effectiveFrames,
            height: conditioning.height,
            width: conditioning.width,
            fps: Float(options.fps),
            action: options.action,
            useSystemPrompt: options.useSystemPrompt
        )
        let conditionalTokens = MLXArray(promptPair.conditionalTokenIDs.map(Int32.init))
        let unconditionalTokens = MLXArray(promptPair.unconditionalTokenIDs.map(Int32.init))

        let visionNoise = MLXRandom.normal(conditioning.cleanVision.shape).asType(.float32)
        let visionConditionMask = Self.frameMask(
            frameCount: conditioning.cleanVision.dim(1),
            conditionedFrames: conditioning.conditionedVisionFrames,
            rank: 4
        )
        var visionLatents = visionConditionMask * conditioning.cleanVision
            + (1 - visionConditionMask) * visionNoise

        var actionLatents = conditioning.initialActions
        let actionConditionMask: MLXArray?
        if let initialActions = conditioning.initialActions,
           let cleanActions = conditioning.cleanActions {
            let mask = Self.frameMask(
                frameCount: initialActions.dim(0),
                conditionedFrames: conditioning.conditionedActionFrames,
                rank: 2
            )
            actionConditionMask = mask
            actionLatents = mask * cleanActions + (1 - mask) * initialActions
        } else {
            actionConditionMask = nil
        }
        eval(visionLatents)
        if let actionLatents { eval(actionLatents) }

        var visionScheduler = Cosmos3UniPCScheduler(
            steps: options.steps,
            shift: options.shift,
            schedule: options.schedule
        )
        var actionScheduler = Cosmos3UniPCScheduler(
            steps: options.steps,
            shift: options.shift,
            schedule: options.schedule
        )
        for (stepIndex, timestep) in visionScheduler.timesteps.enumerated() {
            try Task.checkCancellation()
            transformer!.beginDenoisingStep(
                index: stepIndex,
                count: visionScheduler.timesteps.count
            )
            progressHandler?(GenerationProgress(
                stage: .denoising,
                stepIndex: stepIndex,
                totalSteps: options.steps
            ))
            let conditional = transformer!.predict(try Cosmos3DenoisingInput(
                tokenIDs: conditionalTokens,
                visionLatents: visionLatents,
                conditionedVisionFrames: conditioning.conditionedVisionFrames,
                timestep: timestep,
                fps: Float(options.fps),
                actionLatents: actionLatents,
                conditionedActionFrames: conditioning.conditionedActionFrames,
                actionDomain: conditioning.actionDomain,
                rawActionDimension: conditioning.rawActionDimension
            ))
            let visionVelocity: MLXArray
            var actionVelocity: MLXArray?
            if options.guidanceScale == 1 {
                visionVelocity = conditional.visionVelocity
                actionVelocity = conditional.actionVelocity
            } else {
                let unconditional = transformer!.predict(try Cosmos3DenoisingInput(
                    tokenIDs: unconditionalTokens,
                    visionLatents: visionLatents,
                    conditionedVisionFrames: conditioning.conditionedVisionFrames,
                    timestep: timestep,
                    fps: Float(options.fps),
                    actionLatents: actionLatents,
                    conditionedActionFrames: conditioning.conditionedActionFrames,
                    actionDomain: conditioning.actionDomain,
                    rawActionDimension: conditioning.rawActionDimension
                ))
                visionVelocity = unconditional.visionVelocity
                    + options.guidanceScale
                    * (conditional.visionVelocity - unconditional.visionVelocity)
                if let conditionalAction = conditional.actionVelocity,
                   let unconditionalAction = unconditional.actionVelocity {
                    actionVelocity = unconditionalAction
                        + options.guidanceScale * (conditionalAction - unconditionalAction)
                }
            }
            visionLatents = visionScheduler.step(
                modelOutput: visionVelocity,
                sample: visionLatents
            )
            visionLatents = visionConditionMask * conditioning.cleanVision
                + (1 - visionConditionMask) * visionLatents
            if let velocity = actionVelocity,
               var currentActions = actionLatents,
               let mask = actionConditionMask,
               let cleanActions = conditioning.cleanActions,
               conditioning.conditionedActionFrames.count < currentActions.dim(0) {
                currentActions = actionScheduler.step(
                    modelOutput: velocity,
                    sample: currentActions
                )
                actionLatents = mask * cleanActions + (1 - mask) * currentActions
            }
            eval(visionLatents)
            if let actionLatents { eval(actionLatents) }
            Memory.clearCache()
        }

        progressHandler?(GenerationProgress(
            stage: .decoding,
            stepIndex: options.steps,
            totalSteps: options.steps
        ))
        let decoded = vae!.decode(
            visionLatents.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
        )
        let frames = MLX.clip((decoded + 1) * 127.5, min: 0, max: 255).asType(.uint8)
        let terminalFrameLatent = visionLatents[0..., visionLatents.dim(1) - 1]
            .expandedDimensions(axis: 1)
        eval(frames, terminalFrameLatent)
        chainedTerminalFrameLatent = terminalFrameLatent
        if let outputURL = options.outputURL {
            if frames.dim(1) == 1 {
                let first = frames[0, 0]
                eval(first)
                let image = try MediaImageIO.imageFromRGBHWC(
                    first.asArray(UInt8.self),
                    width: conditioning.width,
                    height: conditioning.height
                )
                try MediaImageIO.writePNG(image, to: outputURL)
            } else {
                try LTXVideoMP4Writer.writeMP4(
                    frames: frames,
                    fps: options.fps,
                    to: outputURL
                )
            }
        }
        let predictedActions: MLXArray?
        if options.action?.mode == .policy || options.action?.mode == .inverseDynamics {
            if let actionLatents, let width = conditioning.rawActionDimension {
                predictedActions = actionLatents[0..., 0..<width]
            } else {
                predictedActions = nil
            }
        } else {
            predictedActions = nil
        }
        return Cosmos3GenerationResult(
            mode: options.mode,
            frames: frames,
            actions: predictedActions,
            terminalFrameLatent: terminalFrameLatent,
            seed: options.seed,
            width: conditioning.width,
            height: conditioning.height
        )
    }

    private func generateImageEdit(
        options: Cosmos3GenerationOptions,
        conditioning: PreparedConditioning,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws -> Cosmos3GenerationResult {
        let systemPrompt =
            "You are a helpful assistant who will edit images based on the user's instructions."
        let prompts = try tokenizer!.encode(
            prompt: options.prompt,
            negativePrompt: options.negativePrompt,
            numFrames: 1,
            height: conditioning.height,
            width: conditioning.width,
            fps: Float(options.fps),
            systemPrompt: systemPrompt
        )
        let conditionalTokens = MLXArray(prompts.conditionalTokenIDs.map(Int32.init))
        let unconditionalTokens = MLXArray(prompts.unconditionalTokenIDs.map(Int32.init))
        let source = conditioning.cleanVision
        var target = MLXRandom.normal(source.shape).asType(.float32)
        eval(source, target)
        var scheduler = Cosmos3UniPCScheduler(
            steps: options.steps,
            shift: options.shift,
            schedule: options.schedule
        )
        for (stepIndex, timestep) in scheduler.timesteps.enumerated() {
            try Task.checkCancellation()
            transformer!.beginDenoisingStep(index: stepIndex, count: scheduler.timesteps.count)
            progressHandler?(GenerationProgress(
                stage: .denoising,
                stepIndex: stepIndex,
                totalSteps: options.steps
            ))
            let items = [
                try Cosmos3VisionDenoisingItem(
                    latents: source,
                    conditionedFrames: Array(0..<source.dim(1))
                ),
                try Cosmos3VisionDenoisingItem(latents: target),
            ]
            let conditional = transformer!.predictVisionItems(
                tokenIDs: conditionalTokens,
                items: items,
                timestep: timestep,
                fps: Float(options.fps)
            )[1]
            let velocity: MLXArray
            if options.guidanceScale == 1 {
                velocity = conditional
            } else {
                let unconditional = transformer!.predictVisionItems(
                    tokenIDs: unconditionalTokens,
                    items: items,
                    timestep: timestep,
                    fps: Float(options.fps)
                )[1]
                velocity = unconditional
                    + options.guidanceScale * (conditional - unconditional)
            }
            target = scheduler.step(modelOutput: velocity, sample: target)
            eval(target)
            Memory.clearCache()
        }
        progressHandler?(GenerationProgress(
            stage: .decoding,
            stepIndex: options.steps,
            totalSteps: options.steps
        ))
        let decoded = vae!.decode(
            target.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
        )
        let frames = MLX.clip((decoded + 1) * 127.5, min: 0, max: 255).asType(.uint8)
        let terminal = target[0..., target.dim(1) - 1].expandedDimensions(axis: 1)
        eval(frames, terminal)
        chainedTerminalFrameLatent = terminal
        if let outputURL = options.outputURL {
            let imageArray = frames[0, 0]
            eval(imageArray)
            let image = try MediaImageIO.imageFromRGBHWC(
                imageArray.asArray(UInt8.self),
                width: conditioning.width,
                height: conditioning.height
            )
            try MediaImageIO.writePNG(image, to: outputURL)
        }
        return Cosmos3GenerationResult(
            mode: .imageToImage,
            frames: frames,
            actions: nil,
            terminalFrameLatent: terminal,
            seed: options.seed,
            width: conditioning.width,
            height: conditioning.height
        )
    }

    private func prepareConditioning(
        options: Cosmos3GenerationOptions,
        effectiveFrames: Int,
        resources: Cosmos3Resources,
        useChainedSourceLatent: Bool
    ) throws -> PreparedConditioning {
        if useChainedSourceLatent, let chainedTerminalFrameLatent, let action = options.action {
            let latentFrames = (effectiveFrames - 1) / 4 + 1
            let latentHeight = chainedTerminalFrameLatent.dim(2)
            let latentWidth = chainedTerminalFrameLatent.dim(3)
            var cleanVision = MLX.zeros(
                [transformer!.configuration.latentChannels, latentFrames, latentHeight, latentWidth],
                dtype: chainedTerminalFrameLatent.dtype
            )
            cleanVision = cleanVision.at[0..., 0, 0..., 0...].add(
                chainedTerminalFrameLatent[0..., 0, 0..., 0...]
            )
            return try actionConditioning(
                cleanVision: cleanVision,
                action: action,
                width: latentWidth * 16,
                height: latentHeight * 16
            )
        }

        if let action = options.action {
            let prepared = try prepareActionFrames(action, targetFrames: effectiveFrames)
            let encoded = try encode(
                images: prepared.images,
                width: prepared.canvasWidth,
                height: prepared.canvasHeight,
                resources: resources,
                cropWidth: prepared.contentWidth,
                cropHeight: prepared.contentHeight
            )
            return try actionConditioning(
                cleanVision: encoded,
                action: action,
                width: encoded.dim(3) * 16,
                height: encoded.dim(2) * 16
            )
        }

        let images: [MediaImage]
        var conditionedFrames: [Int] = []
        if let imageURL = options.imageURL {
            try requireFile(imageURL)
            let image = try MediaImageIO.centerCropped(
                MediaImageIO.decode(imageURL),
                width: options.width,
                height: options.height
            )
            images = Array(repeating: image, count: effectiveFrames)
            if effectiveFrames > 1 { conditionedFrames = [0] }
        } else if let videoURL = options.videoURL {
            try requireFile(videoURL)
            let latentFrames = (effectiveFrames - 1) / 4 + 1
            let validConditioned = options.conditionedVideoLatentFrames.filter {
                $0 >= 0 && $0 < latentFrames
            }
            let pixelCount = (validConditioned.max() ?? 0) * 4 + 1
            var decoded = try decodeVideoFrames(
                videoURL,
                maximumCount: max(pixelCount, 1),
                keepsTail: options.keepsVideoTail
            ).map {
                try MediaImageIO.centerCropped($0, width: options.width, height: options.height)
            }
            while decoded.count < effectiveFrames {
                decoded.append(decoded.last!)
            }
            images = Array(decoded.prefix(effectiveFrames))
            conditionedFrames = validConditioned
        } else {
            var rgba = [UInt8](repeating: 0, count: options.width * options.height * 4)
            for alpha in stride(from: 3, to: rgba.count, by: 4) {
                rgba[alpha] = 255
            }
            let black = try MediaImage(width: options.width, height: options.height, rgba8: rgba)
            images = Array(repeating: black, count: effectiveFrames)
        }
        return PreparedConditioning(
            cleanVision: try encode(
                images: images,
                width: options.width,
                height: options.height,
                resources: resources
            ),
            conditionedVisionFrames: conditionedFrames,
            initialActions: nil,
            cleanActions: nil,
            conditionedActionFrames: [],
            actionDomain: nil,
            rawActionDimension: nil,
            width: options.width,
            height: options.height
        )
    }

    private func actionConditioning(
        cleanVision: MLXArray,
        action: Cosmos3ActionCondition,
        width: Int,
        height: Int
    ) throws -> PreparedConditioning {
        let actionDimension = transformer!.configuration.actionDimension!
        let cleanActions: MLXArray
        let initialActions: MLXArray
        let conditionedActions: [Int]
        switch action.mode {
        case .forwardDynamics:
            let actions = action.modelActions(actionDimension: actionDimension)!
            cleanActions = MLXArray(actions.flatMap { $0 }).reshaped(
                action.chunkSize,
                actionDimension
            )
            initialActions = cleanActions
            conditionedActions = Array(0..<action.chunkSize)
        case .policy, .inverseDynamics:
            cleanActions = MLX.zeros([action.chunkSize, actionDimension])
            var noise = MLXRandom.normal([action.chunkSize, actionDimension]).asType(.float32)
            if action.domain.rawActionDimension < actionDimension {
                let mask = MLXArray(
                    Array(repeating: Float(1), count: action.domain.rawActionDimension)
                        + Array(
                            repeating: Float(0),
                            count: actionDimension - action.domain.rawActionDimension
                        )
                )
                noise = noise * mask
            }
            initialActions = noise
            conditionedActions = []
        }
        let conditionedVision = action.mode == .inverseDynamics
            ? Array(0..<cleanVision.dim(1))
            : [0]
        return PreparedConditioning(
            cleanVision: cleanVision,
            conditionedVisionFrames: conditionedVision,
            initialActions: initialActions,
            cleanActions: cleanActions,
            conditionedActionFrames: conditionedActions,
            actionDomain: action.domain,
            rawActionDimension: action.domain.rawActionDimension,
            width: width,
            height: height
        )
    }

    private func encode(
        images: [MediaImage],
        width: Int,
        height: Int,
        resources: Cosmos3Resources,
        cropWidth: Int? = nil,
        cropHeight: Int? = nil
    ) throws -> MLXArray {
        let frames = images.map { image -> MLXArray in
            let channels = MediaImageIO.rgbCHWFloat(
                image,
                normalizedToMinusOneToOne: true
            )
            return MLXArray(channels).reshaped(3, height, width).transposed(1, 2, 0)
        }
        let video = MLX.stacked(frames, axis: 0).expandedDimensions(axis: 0)
        let encoded = vae!.encodeVideo(video)[0].transposed(3, 0, 1, 2).asType(.float32)
        guard let cropWidth, let cropHeight else { return encoded }
        let latentWidth = max(cropWidth / 16, 1)
        let latentHeight = max(cropHeight / 16, 1)
        return encoded[0..., 0..., 0..<latentHeight, 0..<latentWidth]
    }

    private func prepareActionFrames(
        _ action: Cosmos3ActionCondition,
        targetFrames: Int
    ) throws -> (
        images: [MediaImage],
        canvasWidth: Int,
        canvasHeight: Int,
        contentWidth: Int,
        contentHeight: Int
    ) {
        let sourceImages: [MediaImage]
        if let imageURL = action.imageURL {
            try requireFile(imageURL)
            sourceImages = [try MediaImageIO.decode(imageURL)]
        } else if let videoURL = action.videoURL {
            try requireFile(videoURL)
            sourceImages = try decodeVideoFrames(
                videoURL,
                maximumCount: targetFrames,
                keepsTail: false
            )
        } else {
            preconditionFailure("Validated action condition has no media")
        }
        guard let first = sourceImages.first else {
            throw Cosmos3EdgeGeneratorError.emptyVideo(action.videoURL!)
        }
        let canvas = action.resolutionTier.nearestCanvas(
            sourceHeight: first.height,
            sourceWidth: first.width
        )
        let scale = min(
            Double(canvas.width) / Double(first.width),
            Double(canvas.height) / Double(first.height),
            1
        )
        let contentWidth = max(1, Int(Double(first.width) * scale + 0.5))
        let contentHeight = max(1, Int(Double(first.height) * scale + 0.5))
        var prepared = try sourceImages.map {
            try Self.paddedActionImage(
                $0,
                canvasWidth: canvas.width,
                canvasHeight: canvas.height,
                contentWidth: contentWidth,
                contentHeight: contentHeight
            )
        }
        while prepared.count < targetFrames {
            prepared.append(prepared.last!)
        }
        prepared = Array(prepared.prefix(targetFrames))
        return (prepared, canvas.width, canvas.height, contentWidth, contentHeight)
    }

    private static func paddedActionImage(
        _ image: MediaImage,
        canvasWidth: Int,
        canvasHeight: Int,
        contentWidth: Int,
        contentHeight: Int
    ) throws -> MediaImage {
        let content = try MediaImageIO.resized(
            image,
            width: contentWidth,
            height: contentHeight
        )
        let padRight = canvasWidth - contentWidth
        let padBottom = canvasHeight - contentHeight
        let replicate = padRight >= contentWidth || padBottom >= contentHeight
        var rgba = [UInt8](repeating: 0, count: canvasWidth * canvasHeight * 4)
        for y in 0..<canvasHeight {
            let sourceY = y < contentHeight
                ? y
                : (replicate ? contentHeight - 1 : 2 * contentHeight - 2 - y)
            for x in 0..<canvasWidth {
                let sourceX = x < contentWidth
                    ? x
                    : (replicate ? contentWidth - 1 : 2 * contentWidth - 2 - x)
                let source = (sourceY * contentWidth + sourceX) * 4
                let target = (y * canvasWidth + x) * 4
                rgba[target..<(target + 4)] = content.rgba8[source..<(source + 4)]
            }
        }
        return try MediaImage(width: canvasWidth, height: canvasHeight, rgba8: rgba)
    }

    private func decodeVideoFrames(
        _ url: URL,
        maximumCount: Int,
        keepsTail: Bool
    ) throws -> [MediaImage] {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-cosmos3-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let sequence = try MediaVideoIO.extractFrames(
            from: url,
            into: root,
            endFrame: keepsTail ? nil : maximumCount - 1
        )
        guard !sequence.frameURLs.isEmpty else {
            throw Cosmos3EdgeGeneratorError.emptyVideo(url)
        }
        let selected = keepsTail
            ? Array(sequence.frameURLs.suffix(maximumCount))
            : Array(sequence.frameURLs.prefix(maximumCount))
        return try selected.map(MediaImageIO.decode)
    }

    private static func frameMask(
        frameCount: Int,
        conditionedFrames: [Int],
        rank: Int
    ) -> MLXArray {
        let conditioned = Set(conditionedFrames)
        let values = (0..<frameCount).map { conditioned.contains($0) ? Float(1) : 0 }
        switch rank {
        case 4:
            return MLXArray(values).reshaped(1, frameCount, 1, 1)
        case 2:
            return MLXArray(values).reshaped(frameCount, 1)
        default:
            preconditionFailure("Unsupported Cosmos3 condition-mask rank")
        }
    }

    private func ensureRoot(_ resources: Cosmos3Resources) throws {
        let root = resources.rootURL.standardizedFileURL
        if let loadedRootURL, loadedRootURL != root {
            unload()
        }
        loadedRootURL = root
    }

    private func requireFile(_ url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Cosmos3EdgeGeneratorError.missingFile(url)
        }
    }
}
