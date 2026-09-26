import Foundation

// MARK: - Music templates

extension CommandCatalog {
    package static let musicTemplates: [CommandTemplate] = [
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

// MARK: - Music arguments

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
        // ACE-Step's own defaults stay off the command line: MiniMax Music 3 and YuE2 accept them
        // only to ignore them, and a decoder subdirectory is how the CLI picks the checkpoint.
        if !draft.musicDecoderSubdirectory.isBlank {
            args.optionUnlessDefault(F.decoderSubdirectory, draft.musicDecoderSubdirectory)
        }
        if !draft.musicVAESubdirectory.isBlank {
            args.optionUnlessDefault(F.vaeSubdirectory, draft.musicVAESubdirectory)
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
        // The scope drops the flag where the family does not take it, and "song", which every
        // ACE-Step checkpoint runs when it is left off (the contract's family default).
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
        args.number(F.audioCoverStrength, draft.musicCoverStrength, unlessDefault: F.defaultValues)
        args.number(F.coverNoiseStrength, draft.musicCoverNoiseStrength, unlessDefault: F.defaultValues)
        args.number(F.retakeVariance, draft.musicRetakeVariance, unlessDefault: F.defaultValues)
        args.optionUnlessDefault(F.vocalLanguage, draft.musicVocalLanguage)
        args.optionUnlessDefault(F.instruction, draft.musicInstruction)
        args.optionUnlessDefault(F.taskType, draft.musicTask)
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
        args.number(F.lmTemperature, draft.musicLMTemperature, unlessDefault: F.defaultValues)
        args.optionUnlessDefault(F.lmTopK, String(draft.musicLMTopK))
        args.number(F.lmTopP, draft.musicLMTopP, unlessDefault: F.defaultValues)
        args.number(F.lmRepetitionPenalty, draft.musicLMRepetitionPenalty, unlessDefault: F.defaultValues)
        args.number(F.lmCfgScale, draft.musicLMCFGScale, unlessDefault: F.defaultValues)
        args.optionUnlessDefault(F.lmNegativePrompt, draft.musicLMNegativePrompt)
        if draft.musicInstrumental { args.flag(F.instrumental) }
        if !draft.musicMetadataDuration.isBlank {
            args.option(F.metadataDuration, draft.musicMetadataDuration)
        }
        if !draft.musicMetadataLanguage.isBlank {
            args.option(F.metadataLanguage, draft.musicMetadataLanguage)
        }
        if draft.musicNoTiledVAE { args.flag(F.noTiledVAE) }
        args.optionUnlessDefault(F.vaeChunkSize, String(draft.musicVAEChunkSize))
        args.optionUnlessDefault(F.vaeOverlap, String(draft.musicVAEOverlap))
        // Magenta RealTime 2's controls, also only when they differ from the CLI's defaults.
        args.number(F.temperature, draft.musicTemperature, unlessDefault: F.defaultValues)
        args.optionUnlessDefault(F.styleConditioning, draft.musicStyleConditioning)
        args.optionUnlessDefault(F.topK, String(draft.musicTopK))
        args.number(F.cfgMusiccoca, draft.musicCFGMusicCoCa, unlessDefault: F.defaultValues)
        args.number(F.cfgNotes, draft.musicCFGNotes, unlessDefault: F.defaultValues)
        args.number(F.cfgDrums, draft.musicCFGDrums, unlessDefault: F.defaultValues)
        args.optionUnlessDefault(F.unmaskWidth, String(draft.musicUnmaskWidth))
        args.optionUnlessDefault(F.seedRotation, String(draft.musicSeedRotation))
        args.number(F.prefillDuration, draft.musicPrefillDuration, unlessDefault: F.defaultValues)
        if draft.musicDrumless { args.flag(F.drumless) }
        if draft.musicPrefillSilence { args.flag(F.prefillSilence) }
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
        // `--factor` is LoKr's factorization target; the CLI's -1 picks balanced factors itself.
        if draft.musicTrainingKind == "lokr", draft.musicTrainingFactor > 0 {
            args.option(F.factor, String(draft.musicTrainingFactor))
        }
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
            args.optionUnlessDefault(F.decoderSubdirectory, draft.musicDecoderSubdirectory)
        }
        if !draft.musicVAESubdirectory.isBlank {
            args.optionUnlessDefault(F.vaeSubdirectory, draft.musicVAESubdirectory)
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

private extension ArgumentBuilder {
    /// Appends `--flag value` unless the value is, as a number, the contract's `default_value`,
    /// which the CLI runs anyway: `1` and `1.0` are one value there.
    mutating func number(_ flag: String, _ value: Double, unlessDefault defaults: [String: String]) {
        let rendered = CommandArguments.format(value)
        guard defaults[flag].flatMap(Double.init) != Double(rendered) else { return }
        option(flag, rendered)
    }
}

// MARK: - Music validation

extension CommandCatalog {
    /// The reason a music template's draft cannot run, beyond the prompt and input checks
    /// every template shares; nil for a draft that can, and for every other template.
    package static func musicValidationMessage(for id: CommandTemplateID, draft: CommandDraft) -> String? {
        switch id {
        case .musicGenerate:
            if draft.musicInstrumental,
               !draft.secondaryText.isBlank || !draft.musicLyricsFile.isBlank
                    || !draft.musicLRCFile.isBlank
            {
                return "Instrumental cannot be combined with lyrics."
            }
            if !draft.musicLRCFile.isBlank
                && (!draft.secondaryText.isBlank || !draft.musicLyricsFile.isBlank) {
                return "Use synchronized LRC or plain lyrics, not both."
            }
            let sourceTasks = ["repaint", "cover", "cover-nofsq", "extract", "lego", "complete"]
            if (sourceTasks.contains(draft.musicTask) || draft.musicFlowEdit)
                && draft.musicSourceAudio.isBlank {
                return "Source audio is required for \(draft.musicTask) and flow-edit workflows."
            }
        case .musicTranscribe:
            if draft.inputPath.isBlank && !draft.musicListInstruments {
                return "Audio path is required unless listing instruments."
            }
        case .musicRealtime:
            if draft.prompt.isBlank && !draft.musicListMIDIInputs && !draft.musicMIDIMonitor {
                return "A prompt is required unless listing or monitoring MIDI inputs."
            }
            if !draft.musicPlay && draft.outputPath.isBlank && !draft.musicListMIDIInputs
                && !draft.musicMIDIMonitor {
                return "Enable playback or choose an output file."
            }
        case .musicTrainAdapter:
            if draft.inputPath.isBlank {
                return "Dataset manifest is required."
            }
            if draft.outputPath.isBlank {
                return "Adapter output is required."
            }
        case .musicServe:
            if draft.host != "127.0.0.1" && draft.host != "localhost"
                && draft.host != "::1" && draft.apiKey.isBlank {
                return "An API key is required for non-loopback music servers."
            }
        case .musicSeparate:
            if let overlap = draft.audioOverlap, overlap <= 0 {
                return "Overlap must be positive."
            }
        default:
            break
        }
        return nil
    }
}
