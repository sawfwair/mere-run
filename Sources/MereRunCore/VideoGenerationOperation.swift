import Foundation

public enum VideoGenerationError: Error, LocalizedError, Equatable, Sendable {
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidInput(let message): message
        }
    }
}

public enum VideoGenerationEvent: Sendable {
    case diagnostic(String)
    case progress(stage: String, step: Int, totalSteps: Int)
    case progressFinished
}

public struct VideoGenerationOutcome: Sendable {
    public let primaryURL: URL
    public let isDirectory: Bool
    public let timings: LTXVideoTimingReport?
    public let includesTimings: Bool

    public init(primaryURL: URL, isDirectory: Bool = false, timings: LTXVideoTimingReport? = nil, includesTimings: Bool = false) {
        self.primaryURL = primaryURL
        self.isDirectory = isDirectory
        self.timings = timings
        self.includesTimings = includesTimings
    }
}

/// Resolved inputs for one native video execution. Preparation does not retain a model.
public struct VideoGenerationPreparedRequest: Sendable {
    public enum Input: Sendable {
        case ltx(VideoGenerationLTXRequest)
        case audioToVideo(LTXAudioToVideoGenerationOptions)
        case wan(Wan2GenerationOptions)
        case h3(MiniMaxH3GenerationOptions)
    }

    public let plan: VideoGenerationPlan
    public let modelRoot: URL
    public let input: Input
}

/// Owns preparation, native execution, unloading, and media output. The caller
/// owns admission and binary resource setup; this operation acquires no permit.
public enum VideoGenerationOperation {
    public typealias EventHandler = @Sendable (VideoGenerationEvent) -> Void
    public typealias Executor = @Sendable (VideoGenerationPreparedRequest, EventHandler?) async throws -> VideoGenerationOutcome

    public static func execute(
        _ options: VideoGenerationOptions,
        allowAutoDownload: Bool = true,
        prepareRuntime: @Sendable () throws -> Void = {},
        eventHandler: EventHandler? = nil,
        executor: Executor = generate
    ) async throws -> VideoGenerationOutcome {
        try Task.checkCancellation()
        let request = try await prepare(options, allowAutoDownload: allowAutoDownload, eventHandler: eventHandler)
        try Task.checkCancellation()
        try prepareRuntime()
        let outcome = try await executor(request, eventHandler)
        try Task.checkCancellation()
        return outcome
    }

    public static func prepare(
        _ options: VideoGenerationOptions,
        allowAutoDownload: Bool = true,
        eventHandler: EventHandler? = nil
    ) async throws -> VideoGenerationPreparedRequest {
        try Task.checkCancellation()
        if let issue = options.validationIssues(profile: options.observedProfile()).first(where: { $0.severity == .blocker }) {
            throw issue
        }
        if !options.autoDuration.isEmpty, options.numFrames != nil {
            eventHandler?(.diagnostic("Warning: --auto-duration is ignored because --num-frames was supplied.\n"))
        }
        let sourceImage = try inputURL(options.image, name: "Image")
        let endImage = try inputURL(options.endImage, name: "End image")
        let sourceAudio = try inputURL(options.audio, name: "Audio")
        let root = try await VideoGenerationModelResolver.resolve(
            explicitModelRoot: options.modelRoot,
            requestedModel: options.resolvedRequestedModel,
            variant: options.variant,
            allowAutoDownload: allowAutoDownload
        )
        try Task.checkCancellation()
        let profile = VideoGenerationModelProfile.observe(root: root)
        if let issue = options.validationIssues(profile: profile).first(where: { $0.severity == .blocker }) {
            throw issue
        }
        let arguments = VideoGenerationArgumentParser(options: options)
        let baseModelID = profile == .ltx25Full ? ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
            : profile.isLTX25 ? ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
            : options.resolvedRequestedModel
        let loras = try arguments.parseLTXLoRAConfigurations(options.loras, optionName: "--lora", baseModelID: baseModelID)
        let detailingLoRAs = try arguments.parseLTXLoRAConfigurations(
            options.detailingLoRAs, optionName: "--detailing-lora", baseModelID: ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
        )
        let preparation = try VideoGenerationLTXPreparation(options: options, profile: profile, loras: loras, detailingLoRAs: detailingLoRAs)
        let referenceVideos = try arguments.parseLTXReferenceVideoConditionings(
            downscaleFactor: preparation.referenceDownscaleFactor, temporalScaleFactor: preparation.referenceTemporalScaleFactor
        )
        let plan = try VideoGenerationPlan(options: options, profile: profile, preparation: preparation)
        _ = try inputURL(options.textEmbeddings, name: "LTX text embeddings")
        let images = try arguments.parseLTXImageConditionings()
        let trimmedPrompt = options.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let prompt: String
        if options.enhancePrompt {
            let mode = sourceImage == nil ? "text-to-video" : "image-to-video"
            eventHandler?(.diagnostic("Enhancing prompt with native Gemma-4 (\(mode))\n"))
            prompt = try await LTXPromptEnhancer.enhance(
                prompt: trimmedPrompt, modelID: options.promptEnhancerModel,
                modelRoot: options.promptEnhancerModelRoot.map { URL(fileURLWithPath: $0).standardizedFileURL }, referenceImage: sourceImage
            )
            eventHandler?(.diagnostic("Enhanced prompt: \(prompt)\n"))
        } else {
            prompt = trimmedPrompt
        }
        try Task.checkCancellation()
        let input: VideoGenerationPreparedRequest.Input
        if profile.isH3 {
            let h3 = try VideoGenerationH3Preparation(options: options, profile: profile, modelRoot: root)
            input = .h3(try plan.miniMaxH3Options(
                prompt: prompt, adapterURL: h3.adapterURL, firstFrameURL: sourceImage, lastFrameURL: endImage,
                frames: VideoGenerationArgumentParser.h3Frames(options.h3FrameInputs),
                references: VideoGenerationArgumentParser.h3References(options.references)
            ))
        } else {
            let ltx = VideoGenerationLTXRequest(
                plan: plan, preparation: preparation, prompt: prompt, sourceImageURL: sourceImage,
                endImageURL: endImage, imageConditionings: images, referenceVideos: referenceVideos
            )
            if let sourceAudio {
                try VideoGenerationModelResolver.validateAudioToVideo(root)
                input = .audioToVideo(ltx.audioToVideoOptions(audioURL: sourceAudio))
            } else if profile == .wan {
                guard let sourceImage else { throw VideoGenerationError.invalidInput("Wan2.2 TI2V requires --image.") }
                input = .wan(try plan.wanOptions(prompt: prompt, sourceImageURL: sourceImage))
            } else {
                try VideoGenerationModelResolver.validate(root)
                input = .ltx(ltx)
            }
        }
        return VideoGenerationPreparedRequest(plan: plan, modelRoot: root, input: input)
    }

    public static func generate(_ request: VideoGenerationPreparedRequest, eventHandler: EventHandler?) async throws -> VideoGenerationOutcome {
        try await NativeVideoGeneration(eventHandler: eventHandler).execute(request)
    }

    private static func inputURL(_ path: String?, name: String) throws -> URL? {
        guard let path else { return nil }
        let url = URL(fileURLWithPath: path).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw VideoGenerationError.invalidInput("\(name) file not found: \(url.path)")
        }
        return url
    }
}
