import Foundation

// MARK: - Music & Video templates

extension CommandCatalog {
    package static let mediaTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .audioEnhance,
            category: .media,
            title: "Enhance audio",
            subtitle: "AP-BWE speech extension or UniverSR restoration",
            systemImage: "waveform.badge.plus",
            inputKind: .audio,
            outputKind: .file("wav"),
            defaultModel: "audio-enhance-ap-bwe-16kto48k"
        ),
        CommandTemplate(
            id: .audioGenerate,
            category: .media,
            title: "Generate audio",
            subtitle: "Native LTX-2.5 text-to-audio generation",
            systemImage: "waveform.badge.sparkles",
            promptLabel: "Audio prompt",
            secondaryLabel: "Negative prompt",
            outputKind: .file("wav"),
            defaultPrompt: "a quiet forest at dawn with distant birds",
            defaultModel: "video-ltx25-full-bf16"
        ),
        CommandTemplate(
            id: .musicGenerate,
            category: .media,
            title: "Generate music",
            subtitle: "ACE-Step or Magenta RT2 music generation",
            systemImage: "music.note",
            promptLabel: "Caption",
            secondaryLabel: "Lyrics",
            outputKind: .file("wav"),
            defaultPrompt: "upbeat electronic groove",
            defaultModel: "music-acestep"
        ),
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
        ),
        CommandTemplate(
            id: .musicAnalyze,
            category: .media,
            title: "Analyze music",
            subtitle: "ACE-Step audio understanding (JSON)",
            systemImage: "waveform.badge.magnifyingglass",
            inputKind: .audio,
            defaultModel: "music-acestep"
        ),
        CommandTemplate(
            id: .musicTranscribe,
            category: .media,
            title: "Transcribe to MIDI",
            subtitle: "MuScriptor full-mix audio to instrument tracks",
            systemImage: "pianokeys",
            inputKind: .audio,
            outputKind: .file("mid"),
            defaultModel: "music-muscriptor-medium"
        ),
        CommandTemplate(
            id: .musicSeparate,
            category: .media,
            title: "Separate or restore",
            subtitle: "RoFormer stems, dereverb, and denoise",
            systemImage: "slider.horizontal.3",
            inputKind: .audio,
            outputKind: .directory,
            defaultModel: "music-separate-bs-roformer-viperx-1297"
        ),
        CommandTemplate(
            id: .musicRealtime,
            category: .media,
            title: "Realtime music",
            subtitle: "Magenta RT2 playback, capture, and MIDI steering",
            systemImage: "dot.radiowaves.left.and.right",
            promptLabel: "Prompt",
            outputKind: .file("wav"),
            defaultPrompt: "warm ambient pads with a slow build",
            defaultModel: "music-magenta-rt2-small"
        ),
        CommandTemplate(
            id: .musicTrainAdapter,
            category: .media,
            title: "Train music adapter",
            subtitle: "Native ACE-Step LoRA or LoKr training",
            systemImage: "tuningfork",
            inputKind: .file([.json]),
            outputKind: .file("safetensors"),
            defaultModel: "music-acestep"
        ),
        CommandTemplate(
            id: .musicServe,
            category: .media,
            title: "Resident music API",
            subtitle: "Keep ACE-Step, LM, and adapters warm",
            systemImage: "server.rack",
            defaultModel: "music-acestep"
        )
    ]
}

// MARK: - Music & Video arguments

extension CommandArguments {
    package static func musicGenerate(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.MusicGenerate
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.musicInstrumental {
            if !draft.secondaryText.isBlank { args.option(F.lyrics, draft.secondaryText) }
            if !draft.musicLyricsFile.isBlank { args.option(F.lyricsFile, draft.musicLyricsFile) }
            if !draft.musicLRCFile.isBlank { args.option(F.lrcFile, draft.musicLRCFile) }
        }
        if !draft.musicLRCOutput.isBlank { args.option(F.lrcOutput, draft.musicLRCOutput) }
        args.option(F.exportFormat, draft.musicExportFormat)
        args.option(F.normalize, draft.musicNormalization)
        args.option(F.targetPeakDb, format(draft.musicTargetPeakDB))
        args.option(F.fadeInMs, format(draft.musicFadeInMS))
        args.option(F.fadeOutMs, format(draft.musicFadeOutMS))
        if draft.musicNoDither { args.flag(F.noDither) }
        if !draft.musicRecipeOutput.isBlank { args.option(F.recipeOutput, draft.musicRecipeOutput) }
        if draft.musicNoRecipe { args.flag(F.noRecipe) }
        if !draft.musicDAWBundle.isBlank { args.option(F.dawBundle, draft.musicDAWBundle) }
        if !draft.musicStems.isBlank { args.option(F.stems, draft.musicStems) }
        args.repeated(F.adapter, pathList(draft.musicAdapterPaths))
        if !draft.musicAdapterPaths.isBlank {
            args.option(F.adapterKind, draft.musicAdapterKind)
            args.repeated(F.adapterScale, pathList(draft.musicAdapterScales))
        }
        if !draft.musicCheckpointsRoot.isBlank {
            args.option(F.checkpointsRoot, draft.musicCheckpointsRoot)
        }
        if !draft.musicDecoderSubdirectory.isBlank {
            args.option(F.decoderSubdirectory, draft.musicDecoderSubdirectory)
        }
        if !draft.musicVAESubdirectory.isBlank {
            args.option(F.vaeSubdirectory, draft.musicVAESubdirectory)
        }
        if !draft.musicLMSubdirectory.isBlank {
            args.option(F.lmSubdirectory, draft.musicLMSubdirectory)
        }
        if !draft.musicLMModel.isBlank {
            args.option(F.lmModel, draft.musicLMModel)
        }
        if !draft.musicTextSubdirectory.isBlank {
            args.option(F.textSubdirectory, draft.musicTextSubdirectory)
        }
        // Any other mode passes neither half of the pair and leaves the choice to the CLI.
        args.pair(F.useLM, F.noLM, ["use": true, "disable": false][draft.musicLMMode])
        if draft.musicAnalyzeSourceAudio { args.flag(F.analyzeSourceAudio) }
        if draft.useDuration { args.option(F.duration, format(draft.durationSeconds)) }
        args.option(F.quality, draft.musicQuality)
        if draft.musicOverrideSteps { args.option(F.steps, String(draft.steps)) }
        if !draft.musicShift.isBlank { args.option(F.shift, draft.musicShift) }
        if !draft.musicInferMethod.isBlank { args.option(F.inferMethod, draft.musicInferMethod) }
        if !draft.musicSampler.isBlank { args.option(F.sampler, draft.musicSampler) }
        if !draft.musicGuidanceScale.isBlank { args.option(F.guidanceScale, draft.musicGuidanceScale) }
        if !draft.musicGuidanceMode.isBlank { args.option(F.guidanceMode, draft.musicGuidanceMode) }
        if !draft.musicCFGIntervalStart.isBlank {
            args.option(F.cfgIntervalStart, draft.musicCFGIntervalStart)
        }
        if !draft.musicCFGIntervalEnd.isBlank {
            args.option(F.cfgIntervalEnd, draft.musicCFGIntervalEnd)
        }
        if !draft.musicVelocityNormThreshold.isBlank {
            args.option(F.velocityNormThreshold, draft.musicVelocityNormThreshold)
        }
        if !draft.musicVelocityEMAFactor.isBlank {
            args.option(F.velocityEmaFactor, draft.musicVelocityEMAFactor)
        }
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if draft.musicCandidates > 0 { args.option(F.candidates, String(draft.musicCandidates)) }
        if draft.musicKeepCandidates { args.flag(F.keepCandidates) }
        args.option(F.audioCoverStrength, format(draft.musicCoverStrength))
        args.option(F.coverNoiseStrength, format(draft.musicCoverNoiseStrength))
        args.option(F.retakeVariance, format(draft.musicRetakeVariance))
        args.option(F.vocalLanguage, draft.musicVocalLanguage)
        args.option(F.instruction, draft.musicInstruction)
        args.option(F.taskType, draft.musicTask)
        if !draft.musicRetakeSeed.isBlank { args.option(F.retakeSeed, draft.musicRetakeSeed) }
        if !draft.musicSourceAudio.isBlank { args.option(F.sourceAudio, draft.musicSourceAudio) }
        args.repeated(F.referenceAudio, pathList(draft.musicReferenceAudioPaths))
        if !draft.musicTrackName.isBlank { args.option(F.trackName, draft.musicTrackName) }
        if !draft.musicCompleteTrackClasses.isBlank {
            args.option(F.completeTrackClasses, draft.musicCompleteTrackClasses)
        }
        if draft.musicNonCover { args.flag(F.nonCover) }
        if ["repaint", "lego"].contains(draft.musicTask) {
            args.option(F.repaintStart, format(draft.musicRepaintStart))
            args.option(F.repaintEnd, format(draft.musicRepaintEnd))
            args.option(F.chunkMaskMode, draft.musicChunkMaskMode)
            args.option(F.repaintMode, draft.musicRepaintMode)
            args.option(F.repaintStrength, format(draft.musicRepaintStrength))
        }
        if draft.musicFlowEdit {
            args.flag(F.flowEdit)
            args.option(F.flowEditNMin, format(draft.musicFlowEditNMin))
            args.option(F.flowEditNMax, format(draft.musicFlowEditNMax))
            args.option(F.flowEditNAverage, String(draft.musicFlowEditNAverage))
            if !draft.musicSourceCaption.isBlank {
                args.option(F.sourceCaption, draft.musicSourceCaption)
            }
            if !draft.musicSourceLyrics.isBlank {
                args.option(F.sourceLyrics, draft.musicSourceLyrics)
            }
        }
        if !draft.musicBPM.isBlank { args.option(F.bpm, draft.musicBPM) }
        if !draft.musicKey.isBlank { args.option(F.keyscale, draft.musicKey) }
        if !draft.musicTimeSignature.isBlank {
            args.option(F.timesignature, draft.musicTimeSignature)
        }
        args.option(F.lmTemperature, format(draft.musicLMTemperature))
        args.option(F.lmTopK, String(draft.musicLMTopK))
        args.option(F.lmTopP, format(draft.musicLMTopP))
        args.option(F.lmRepetitionPenalty, format(draft.musicLMRepetitionPenalty))
        args.option(F.lmCfgScale, format(draft.musicLMCFGScale))
        args.option(F.lmNegativePrompt, draft.musicLMNegativePrompt)
        if draft.musicInstrumental { args.flag(F.instrumental) }
        if !draft.musicMetadataDuration.isBlank {
            args.option(F.metadataDuration, draft.musicMetadataDuration)
        }
        if !draft.musicMetadataLanguage.isBlank {
            args.option(F.metadataLanguage, draft.musicMetadataLanguage)
        }
        if draft.musicNoTiledVAE { args.flag(F.noTiledVAE) }
        args.option(F.vaeChunkSize, String(draft.musicVAEChunkSize))
        args.option(F.vaeOverlap, String(draft.musicVAEOverlap))
        if draft.model.localizedCaseInsensitiveContains("magenta") {
            args.option(F.temperature, format(draft.musicTemperature))
            args.option(F.styleConditioning, draft.musicStyleConditioning)
            args.option(F.topK, String(draft.musicTopK))
            args.option(F.cfgMusiccoca, format(draft.musicCFGMusicCoCa))
            args.option(F.cfgNotes, format(draft.musicCFGNotes))
            args.option(F.cfgDrums, format(draft.musicCFGDrums))
            args.option(F.unmaskWidth, String(draft.musicUnmaskWidth))
            args.option(F.seedRotation, String(draft.musicSeedRotation))
            args.option(F.prefillDuration, format(draft.musicPrefillDuration))
            if draft.musicDrumless { args.flag(F.drumless) }
            if draft.musicPrefillSilence { args.flag(F.prefillSilence) }
        }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func videoGenerate(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.VideoGenerate
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.modelRoot.isBlank { args.option(F.modelRoot, draft.modelRoot) }
        let family = StudioVideoModelFamily(model: draft.modelRoot.isBlank ? draft.model : draft.modelRoot)
        if family == .ltx {
            let quality = draft.audioPath.isBlank ? draft.videoQuality : .final
            let outputMode = draft.audioPath.isBlank ? draft.videoOutputMode : .audioVideo
            args.option(F.quality, quality.rawValue)
            args.option(F.outputMode, outputMode.rawValue)
        }
        args.option(F.width, String(draft.width))
        args.option(F.height, String(draft.height))
        if draft.useDuration {
            args.option(F.duration, format(draft.durationSeconds))
        } else {
            args.option(F.numFrames, String(draft.numFrames))
        }
        if !family.isMiniMaxH3 { args.option(F.fps, String(draft.fps)) }
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if !family.isMiniMaxH3, !draft.secondaryText.isBlank {
            args.option(F.negativePrompt, draft.secondaryText)
        }
        if family == .wan {
            args.option(F.steps, String(draft.steps))
            args.option(F.guidanceScale, format(draft.cfgScale))
            args.option(F.shift, format(draft.scheduleShift))
        }
        if family.isMiniMaxH3 {
            if let h3Steps = draft.h3Steps { args.option(F.steps, String(h3Steps)) }
            if let weightMode = draft.h3WeightMode, !weightMode.isBlank {
                args.option(F.h3WeightMode, weightMode)
            }
            if let accelerationMode = draft.h3AccelerationMode,
               !accelerationMode.isBlank {
                args.option(F.h3Acceleration, accelerationMode)
            }
            for reference in draft.h3ReferenceInputs ?? [] where !reference.isBlank {
                args.option(F.reference, reference)
            }
        } else if !draft.audioPath.isBlank {
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
        if !family.isMiniMaxH3, draft.timings { args.flag(F.timings) }
        if !family.isMiniMaxH3, !draft.timingsOutputPath.isBlank {
            args.option(F.timingsOutput, draft.timingsOutputPath)
        }
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
        if !draft.secondaryText.isBlank {
            args.option(F.negativePrompt, draft.secondaryText)
        }
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

    package static func audioEnhance(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.AudioEnhance
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.modelRoot.isBlank { args.option(F.modelPath, draft.modelRoot) }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if let overlap = draft.audioOverlap { args.option(F.overlap, String(overlap)) }
        if let inputRate = draft.audioInputRate { args.option(F.inputRate, String(inputRate)) }
        if let method = draft.audioODEMethod, !method.isBlank {
            args.option(F.odeMethod, method)
        }
        if let steps = draft.audioODESteps { args.option(F.odeSteps, String(steps)) }
        if let guidance = draft.audioGuidanceScale {
            args.option(F.guidanceScale, format(guidance))
        }
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if let seconds = draft.audioChunkSeconds {
            args.option(F.chunkSeconds, String(seconds))
        }
        if let dtype = draft.audioDType, !dtype.isBlank { args.option(F.dtype, dtype) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func audioGenerate(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.AudioGenerate
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.modelRoot.isBlank { args.option(F.modelRoot, draft.modelRoot) }
        if !draft.secondaryText.isBlank {
            args.option(F.negativePrompt, draft.secondaryText)
        }
        if draft.useDuration {
            args.option(F.duration, format(draft.durationSeconds))
        } else {
            args.option(F.numFrames, String(draft.numFrames))
            args.option(F.fps, String(draft.fps))
        }
        args.option(F.steps, String(draft.steps))
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func musicAnalyze(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.MusicAnalyze
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.musicCheckpointsRoot.isBlank {
            args.option(F.checkpointsRoot, draft.musicCheckpointsRoot)
        }
        if !draft.musicDecoderSubdirectory.isBlank {
            args.option(F.decoderSubdirectory, draft.musicDecoderSubdirectory)
        }
        if !draft.musicVAESubdirectory.isBlank {
            args.option(F.vaeSubdirectory, draft.musicVAESubdirectory)
        }
        if !draft.musicLMSubdirectory.isBlank {
            args.option(F.lmSubdirectory, draft.musicLMSubdirectory)
        }
        if !draft.musicLMModel.isBlank {
            args.option(F.lmModel, draft.musicLMModel)
        }
        if draft.useDuration { args.option(F.duration, format(draft.durationSeconds)) }
        args.option(F.maxNewTokens, String(draft.musicAnalysisMaxTokens))
        args.option(F.lmTemperature, format(draft.musicAnalysisTemperature))
        args.option(F.lmTopK, String(draft.musicLMTopK))
        args.option(F.lmTopP, format(draft.musicLMTopP))
        if draft.musicIncludeRawLM { args.flag(F.includeRawLM) }
        if draft.musicIncludeAudioCodes { args.flag(F.includeAudioCodes) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func musicTranscribe(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.MusicTranscribe
        var args = ArgumentBuilder(F.self)
        if !draft.inputPath.isBlank { args.value(draft.inputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.musicTranscribeModelPath.isBlank {
            args.option(F.modelPath, draft.musicTranscribeModelPath)
        }
        if !draft.musicTranscribeVariant.isBlank {
            args.option(F.variant, draft.musicTranscribeVariant)
        }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        args.option(F.format, draft.musicTranscribeFormat)
        if !draft.musicInstruments.isBlank {
            args.option(F.instruments, draft.musicInstruments)
        }
        if draft.musicListInstruments { args.flag(F.listInstruments) }
        if draft.musicSampling { args.flag(F.sampling) }
        args.option(F.temperature, format(draft.temperature))
        args.option(F.maxTokensPerChunk, String(draft.musicMaxTokensPerChunk))
        args.option(F.beamSize, String(draft.musicBeamSize))
        args.option(F.chunkBatchSize, String(draft.musicChunkBatchSize))
        args.option(F.dtype, draft.musicDType)
        if draft.musicStrictEOS { args.flag(F.strictEos) }
        if draft.musicNoMusicalContext { args.flag(F.noMusicalContext) }
        if !draft.musicContextOutput.isBlank {
            args.option(F.contextOutput, draft.musicContextOutput)
        }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func musicSeparate(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.MusicSeparate
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.modelRoot.isBlank { args.option(F.modelPath, draft.modelRoot) }
        if !draft.outputPath.isBlank { args.option(F.outputDir, draft.outputPath) }
        if let overlap = draft.audioOverlap { args.option(F.overlap, String(overlap)) }
        if let dtype = draft.audioDType, !dtype.isBlank { args.option(F.dtype, dtype) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func musicRealtime(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.MusicRealtime
        var args = ArgumentBuilder(F.self)
        if !draft.prompt.isBlank { args.value(draft.prompt) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.duration, format(draft.durationSeconds))
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.musicPlay { args.flag(F.noPlay) }
        args.option(F.styleConditioning, draft.musicStyleConditioning)
        args.option(F.temperature, format(draft.musicTemperature))
        args.option(F.topK, String(draft.musicTopK))
        args.option(F.cfgMusiccoca, format(draft.musicCFGMusicCoCa))
        args.option(F.cfgNotes, format(draft.musicCFGNotes))
        args.option(F.cfgDrums, format(draft.musicCFGDrums))
        args.option(F.unmaskWidth, String(draft.musicUnmaskWidth))
        args.option(F.seedRotation, String(draft.musicSeedRotation))
        args.option(F.prefillDuration, format(draft.musicPrefillDuration))
        args.option(F.midiChannel, draft.musicMIDIChannel)
        args.option(F.midiNoteOffset, String(draft.musicMIDINoteOffset))
        if draft.musicDrumless { args.flag(F.drumless) }
        if draft.musicPrefillSilence { args.flag(F.prefillSilence) }
        if draft.musicInteractive { args.flag(F.interactive) }
        if draft.musicListMIDIInputs { args.flag(F.listMidiInputs) }
        if draft.musicMIDIMonitor { args.flag(F.midiMonitor) }
        if draft.musicMIDILogEvents { args.flag(F.midiLogEvents) }
        if draft.musicMIDILogRaw { args.flag(F.midiLogRaw) }
        if !draft.musicMIDIInput.isBlank { args.option(F.midiInput, draft.musicMIDIInput) }
        args.repeated(F.midiCc, pathList(draft.musicMIDICCMappings))
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func musicTrainAdapter(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.MusicTrainAdapter
        var args = ArgumentBuilder(F.self)
        args.option(F.dataset, draft.inputPath)
        args.option(F.output, draft.outputPath)
        args.option(F.kind, draft.musicTrainingKind)
        args.option(F.rank, String(draft.rank))
        args.option(F.alpha, format(draft.alpha))
        args.option(F.factor, String(draft.musicTrainingFactor))
        args.option(F.steps, String(draft.steps))
        args.option(F.learningRate, format(draft.learningRate))
        args.option(F.weightDecay, format(draft.musicTrainingWeightDecay))
        args.option(F.seed, draft.seed)
        args.option(F.maxDuration, format(draft.musicTrainingMaxDuration))
        args.option(F.decoderSubdirectory, draft.musicDecoderSubdirectory)
        args.option(F.vaeSubdirectory, draft.musicVAESubdirectory)
        args.option(F.logEvery, String(draft.musicTrainingLogEvery))
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.musicCheckpointsRoot.isBlank {
            args.option(F.checkpointsRoot, draft.musicCheckpointsRoot)
        }
        if !draft.musicTextSubdirectory.isBlank {
            args.option(F.textSubdirectory, draft.musicTextSubdirectory)
        }
        return args.arguments
    }

    package static func musicServe(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.MusicServe
        var args = ArgumentBuilder(F.self)
        args.option(F.host, draft.host)
        args.option(F.port, String(draft.port))
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.musicCheckpointsRoot.isBlank {
            args.option(F.checkpointsRoot, draft.musicCheckpointsRoot)
        }
        if !draft.musicDecoderSubdirectory.isBlank {
            args.option(F.decoderSubdirectory, draft.musicDecoderSubdirectory)
        }
        if !draft.musicVAESubdirectory.isBlank {
            args.option(F.vaeSubdirectory, draft.musicVAESubdirectory)
        }
        if !draft.musicLMSubdirectory.isBlank {
            args.option(F.lmSubdirectory, draft.musicLMSubdirectory)
        }
        if !draft.musicLMModel.isBlank {
            args.option(F.lmModel, draft.musicLMModel)
        }
        if !draft.musicTextSubdirectory.isBlank {
            args.option(F.textSubdirectory, draft.musicTextSubdirectory)
        }
        args.repeated(F.adapter, pathList(draft.musicAdapterPaths))
        if !draft.musicAdapterPaths.isBlank {
            args.option(F.adapterKind, draft.musicAdapterKind)
            args.repeated(F.adapterScale, pathList(draft.musicAdapterScales))
        }
        return args.arguments
    }
}
