import Foundation
import MereRunContract

// MARK: - Video templates

extension CommandCatalog {
    package static let videoTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .videoGenerate,
            category: .media,
            title: "Generate video",
            subtitle: "Full-power LTX-2.5, Wan, or synchronized MiniMax-H3 generation",
            systemImage: "film",
            promptLabel: "Prompt",
            secondaryLabel: "Negative prompt",
            inputKind: .image,
            outputKind: .file("mp4"),
            defaultPrompt: "a cinematic drone flythrough over snowy mountains",
            defaultModel: "video-ltx25-full-bf16"
        ),
        CommandTemplate(
            id: .videoRetake,
            category: .media,
            title: "Retake video",
            subtitle: "Regenerate a timed LTX-2.5 video or audio region",
            systemImage: "timeline.selection",
            promptLabel: "Replacement prompt",
            secondaryLabel: "Negative prompt",
            inputKind: .video,
            outputKind: .file("mp4"),
            defaultPrompt: "continue the performance with natural synchronized motion",
            defaultModel: "video-ltx25-distilled-bf16"
        ),
        CommandTemplate(
            id: .videoDubIt,
            category: .media,
            title: "Dub-It",
            subtitle: "Transfer synchronized video and audio identity with LTX-2.5 IC-LoRA",
            systemImage: "person.wave.2",
            promptLabel: "Scene prompt",
            inputKind: .video,
            outputKind: .file("mp4"),
            defaultPrompt: "the speaker performs on a rain-lit street",
            defaultModel: "video-ltx25-distilled-bf16"
        ),
        CommandTemplate(
            id: .videoAnimate,
            category: .media,
            title: "Animate subject",
            subtitle: "SCAIL-2 animation and replacement",
            systemImage: "figure.walk.motion",
            promptLabel: "Prompt",
            secondaryLabel: "Negative prompt",
            inputKind: .image,
            outputKind: .file("mp4"),
            defaultPrompt: "a dancer in a red silk dress",
            defaultModel: "video-scail2-14b-mlx"
        ),
        CommandTemplate(
            id: .videoCosmos3,
            category: .media,
            title: "Cosmos3",
            subtitle: "Generation, dynamics, policy, and reasoning",
            systemImage: "sparkles.tv",
            promptLabel: "Prompt or action task",
            secondaryLabel: "Negative prompt",
            outputKind: .file("mp4"),
            defaultPrompt: "a cinematic rover crossing a windswept alien plain",
            defaultModel: "video-cosmos3-edge-mlx"
        ),
        CommandTemplate(
            id: .videoPrepareMasks,
            category: .media,
            title: "Prepare SCAIL-2 masks",
            subtitle: "SAM 3.1 mask-plan preparation",
            systemImage: "square.stack.3d.up",
            inputKind: .file([.json]),
            outputKind: .directory,
            defaultModel: "vision-segment-sam31"
        ),
        CommandTemplate(
            id: .videoExportLatents,
            category: .media,
            title: "Export video latents",
            subtitle: "Write native LTX final latents",
            systemImage: "shippingbox",
            promptLabel: "Prompt",
            outputKind: .file("safetensors"),
            defaultPrompt: "a cinematic drone flythrough over snowy mountains",
            defaultModel: "video-ltx-av"
        ),
        CommandTemplate(
            id: .videoSession,
            category: .media,
            title: "Resident LTX session",
            subtitle: "Keep LTX 2.3 warm for JSONL requests",
            systemImage: "bolt.horizontal.circle",
            defaultModel: "video-ltx23-full-mlx"
        )
    ]
}

// MARK: - Video arguments

extension CommandArguments {
    /// `video generate` for `draft`: every value the draft holds, which `CommandTemplate.arguments`
    /// then scopes to the family the run gets. The family decides one thing here: MiniMax-H3's
    /// step count lives in its schedule override, not the step slider. `scope` defaults to the
    /// app's scope for the draft, which asks the CLI about a local folder or an install-dependent id.
    package static func videoGenerate(_ draft: CommandDraft, scope: StudioOptionScope? = nil) -> [String] {
        typealias F = CommandFlags.VideoGenerate
        let hasAudio = !draft.audioPath.isBlank
        let quality = hasAudio ? LTXVideoQuality.final : draft.videoQuality
        let outputMode = hasAudio ? LTXVideoOutputMode.audioVideo : draft.videoOutputMode
        let scope = scope ?? StudioOptionScope.videoGenerate(draft)
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.modelRoot.isBlank { args.option(F.modelRoot, draft.modelRoot) }
        args.option(F.quality, quality.rawValue)
        args.option(F.outputMode, outputMode.rawValue)
        args.option(F.width, String(draft.width))
        args.option(F.height, String(draft.height))
        if draft.useDuration {
            args.option(F.duration, format(draft.durationSeconds))
        } else {
            args.option(F.numFrames, String(draft.numFrames))
        }
        args.option(F.fps, String(draft.fps))
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if !draft.secondaryText.isBlank { args.option(F.negativePrompt, draft.secondaryText) }
        // MiniMax-H3 picks its schedule from the packed geometry unless the draft overrides it.
        if scope.uses(F.h3Acceleration) {
            if let h3Steps = draft.h3Steps { args.option(F.steps, String(h3Steps)) }
        } else {
            args.option(F.steps, String(draft.steps))
        }
        args.option(F.guidanceScale, format(draft.cfgScale))
        args.option(F.shift, format(draft.scheduleShift))
        if let weightMode = draft.h3WeightMode, !weightMode.isBlank { args.option(F.h3WeightMode, weightMode) }
        if let accelerationMode = draft.h3AccelerationMode, !accelerationMode.isBlank {
            args.option(F.h3Acceleration, accelerationMode)
        }
        for reference in draft.h3ReferenceInputs ?? [] where !reference.isBlank {
            args.option(F.reference, reference)
        }
        if hasAudio {
            args.option(F.audio, draft.audioPath)
            args.option(F.audioStartTime, format(draft.audioStartTime))
            args.option(F.a2vGuidanceScale, format(draft.a2vGuidanceScale))
            args.option(F.videoCfgGuidanceScale, format(draft.videoCFGGuidanceScale))
            args.option(F.audioCfgGuidanceScale, format(draft.audioCFGGuidanceScale))
            args.option(F.v2aGuidanceScale, format(draft.v2aGuidanceScale))
            args.option(F.a2vSteps, String(draft.a2vSteps))
            if let audioMaxDuration = draft.audioMaxDuration, audioMaxDuration > 0 {
                args.option(F.audioMaxDuration, format(audioMaxDuration))
            }
        }
        if !draft.inputPath.isBlank {
            args.option(F.image, draft.inputPath)
            args.option(F.imageStrength, format(draft.strength))
        }
        if !draft.endImagePath.isBlank {
            args.option(F.endImage, draft.endImagePath)
            args.option(F.endImageStrength, format(draft.endImageStrength))
        }
        if draft.preflight {
            args.flag(F.preflight)
            if draft.json { args.flag(F.json) }
        }
        if draft.timings { args.flag(F.timings) }
        if !draft.timingsOutputPath.isBlank { args.option(F.timingsOutput, draft.timingsOutputPath) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func videoRetake(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VideoRetake
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        args.option(F.source, draft.inputPath)
        args.option(F.startTime, format(draft.retakeStartTime))
        args.option(F.endTime, format(draft.retakeEndTime))
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.modelRoot.isBlank { args.option(F.modelRoot, draft.modelRoot) }
        if !draft.secondaryText.isBlank { args.option(F.negativePrompt, draft.secondaryText) }
        args.option(F.steps, String(draft.steps))
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if draft.retakePreserveVideo { args.flag(F.preserveVideo) }
        if draft.retakePreserveAudio { args.flag(F.preserveAudio) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func videoDubIt(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VideoDubIt
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        args.option(F.referenceVideo, draft.inputPath)
        args.option(F.icLoRA, draft.loraPath)
        args.option(F.icLoRAStrength, format(draft.loraScale))
        args.option(F.referenceStrength, format(draft.strength))
        args.option(F.width, String(draft.width))
        args.option(F.height, String(draft.height))
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.modelRoot.isBlank { args.option(F.modelRoot, draft.modelRoot) }
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func videoAnimate(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VideoAnimate
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        args.option(F.reference, draft.inputPath)
        args.option(F.referenceMask, draft.referenceMaskPath)
        args.option(F.drivingVideo, draft.drivingVideoPath)
        args.option(F.drivingMask, draft.drivingMaskPath)
        args.option(F.output, draft.outputPath)
        args.option(F.mode, draft.videoTaskMode)
        args.option(F.profile, draft.renderProfile)
        args.option(F.width, String(draft.width))
        args.option(F.height, String(draft.height))
        args.option(F.fps, String(draft.fps))
        args.option(F.segmentLength, String(draft.segmentLength))
        args.option(F.segmentOverlap, String(draft.segmentOverlap))
        args.option(F.tailPolicy, draft.tailPolicy)
        args.option(F.audioSource, draft.audioSource)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.modelRoot.isBlank { args.option(F.modelRoot, draft.modelRoot) }
        let additionalReferences = pathList(draft.referenceImagePaths)
        let additionalMasks = pathList(draft.scailAdditionalReferenceMaskPaths ?? "")
        for (reference, mask) in zip(additionalReferences, additionalMasks) {
            args.option(F.additionalReference, reference)
            args.option(F.additionalReferenceMask, mask)
        }
        if !draft.loraPath.isBlank {
            args.option(F.distilledAdapter, draft.loraPath)
            args.option(F.distilledAdapterStrength, format(draft.loraScale))
        }
        if !draft.secondaryText.isBlank { args.option(F.negativePrompt, draft.secondaryText) }
        if draft.renderProfile == "quality" {
            args.option(F.steps, String(draft.steps))
            args.option(F.guidanceScale, format(draft.cfgScale))
            args.option(F.shift, format(draft.scheduleShift))
            args.option(F.sampler, draft.sampler)
        }
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if draft.preflight {
            args.flag(F.preflight)
            if draft.json { args.flag(F.json) }
        }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func videoCosmos3(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VideoCosmos3
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        args.option(F.mode, draft.cosmosMode)
        args.option(F.output, draft.outputPath)
        args.option(F.width, String(draft.width))
        args.option(F.height, String(draft.height))
        args.option(F.numFrames, String(draft.numFrames))
        args.option(F.schedule, draft.schedule)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.cosmosImagePath.isBlank { args.option(F.image, draft.cosmosImagePath) }
        if !draft.cosmosVideoPath.isBlank { args.option(F.video, draft.cosmosVideoPath) }
        if !draft.actionsOutputPath.isBlank { args.option(F.actionsOutput, draft.actionsOutputPath) }
        if !draft.secondaryText.isBlank { args.option(F.negativePrompt, draft.secondaryText) }
        if draft.steps > 0 { args.option(F.steps, String(draft.steps)) }
        if draft.cfgScale > 0 { args.option(F.guidanceScale, format(draft.cfgScale)) }
        if draft.scheduleShift > 0 { args.option(F.shift, format(draft.scheduleShift)) }
        if draft.fps > 0 { args.option(F.fps, String(draft.fps)) }
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func videoPrepareMasks(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VideoPrepareMasks
        var args = ArgumentBuilder(F.self)
        args.option(F.plan, draft.inputPath)
        args.option(F.outputDir, draft.outputPath)
        if !draft.previewFrame.isBlank { args.option(F.previewFrame, draft.previewFrame) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.preflight {
            args.flag(F.preflight)
            if draft.json { args.flag(F.json) }
        }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func videoExportLatents(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VideoExportLatents
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.width, String(draft.width))
        args.option(F.height, String(draft.height))
        args.option(F.numFrames, String(draft.numFrames))
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func videoSession(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VideoSession
        var args = ArgumentBuilder(F.self)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.modelRoot.isBlank { args.option(F.modelRoot, draft.modelRoot) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }
}

// MARK: - Video validation

extension CommandCatalog {
    /// The reason a video template's draft cannot run, beyond the prompt and input checks
    /// every template shares; nil for a draft that can, and for every other template.
    package static func videoValidationMessage(for id: CommandTemplateID, draft: CommandDraft) -> String? {
        switch id {
        case .videoGenerate:
            typealias F = CommandFlags.VideoGenerate
            let scope = StudioOptionScope.videoGenerate(draft)
            if let refusal = scope.blockingReason { return refusal }
            let model = scope.family?.title ?? "This model"
            if scope.requires(F.reference), (draft.h3ReferenceInputs ?? []).allSatisfy(\.isBlank) {
                return "\(model) requires at least one ordered reference."
            }
            if scope.requires(F.image), draft.inputPath.isBlank {
                return "\(model) requires a start image."
            }
            if scope.uses(F.endImage), !draft.endImagePath.isBlank, draft.inputPath.isBlank {
                return "A start image is required when an end keyframe is selected."
            }
        case .videoRetake:
            if let refusal = StudioOptionScope.videoRetake(draft).blockingReason { return refusal }
            if draft.retakeStartTime < 0 || draft.retakeStartTime >= draft.retakeEndTime {
                return "Retake requires a nonnegative start before the end time."
            }
            if draft.retakePreserveVideo && draft.retakePreserveAudio {
                return "Retake must regenerate video, audio, or both."
            }
        case .videoDubIt:
            if draft.loraPath.isBlank {
                return "Dub-It requires an IC-LoRA file."
            }
        case .videoAnimate:
            if draft.referenceMaskPath.isBlank {
                return "Reference mask path is required."
            }
            if draft.drivingVideoPath.isBlank {
                return "Driving video path is required."
            }
            if draft.drivingMaskPath.isBlank {
                return "Driving mask path is required."
            }
            let additionalReferences = CommandArguments.pathList(draft.referenceImagePaths)
            let additionalMasks = CommandArguments.pathList(draft.scailAdditionalReferenceMaskPaths ?? "")
            if additionalReferences.count != additionalMasks.count {
                return "Each additional SCAIL reference needs one matching reference mask."
            }
            if additionalReferences.count > 5 {
                return "SCAIL supports at most six subjects total."
            }
        case .videoPrepareMasks:
            if draft.outputPath.isBlank {
                return "Output directory is required."
            }
        default:
            break
        }
        return nil
    }
}

// MARK: - Video runtime family

extension StudioOptionScope {
    /// `video generate`'s scope from the fields that pick its family, which the builder needs
    /// before the rest of the command line exists: `--model-root` over `--model`, for a blank
    /// model the source audio and quality that select the default checkpoint, and the output
    /// mode, which decides the folder `video-ltx-av` runs. The CLI answers for a local folder or
    /// an install-dependent id through `source`; until it does, every option is offered.
    package static func videoGenerate(_ draft: CommandDraft, source: StudioScopeSource = .live) -> StudioOptionScope {
        typealias F = CommandFlags.VideoGenerate
        let hasAudio = !draft.audioPath.isBlank
        var arguments = modelArguments(draft, flags: (F.model, F.modelRoot))
        if hasAudio { arguments += [F.audio, draft.audioPath] }
        arguments += [F.quality, (hasAudio ? LTXVideoQuality.final : draft.videoQuality).rawValue]
        arguments += [F.outputMode, (hasAudio ? LTXVideoOutputMode.audioVideo : draft.videoOutputMode).rawValue]
        return source.scope(capability: MereRunCapabilityCatalog.videoGenerate, commandLine: arguments)
    }

    package static func videoRetake(_ draft: CommandDraft, source: StudioScopeSource = .live) -> StudioOptionScope {
        typealias F = CommandFlags.VideoRetake
        return source.scope(
            capability: MereRunCapabilityCatalog.videoRetake, commandLine: modelArguments(draft, flags: (F.model, F.modelRoot))
        )
    }

    private static func modelArguments(_ draft: CommandDraft, flags: (model: String, modelRoot: String)) -> [String] {
        (draft.modelRoot.isBlank ? [] : [flags.modelRoot, draft.modelRoot]) + (draft.model.isBlank ? [] : [flags.model, draft.model])
    }
}
