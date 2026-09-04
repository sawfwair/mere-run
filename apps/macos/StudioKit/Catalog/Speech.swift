import Foundation

// MARK: - Speech templates

extension CommandCatalog {
    package static let speechTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .speechSynthesize,
            category: .speech,
            title: "Synthesize",
            subtitle: "Text to speech",
            systemImage: "waveform",
            promptLabel: "Text",
            secondaryLabel: "Voice",
            outputKind: .file("wav"),
            defaultPrompt: "Hello from mere.run.",
            defaultSecondaryText: "A calm female voice with clear pronunciation",
            defaultModel: "speech-tts-qwen3-nano"
        ),
        CommandTemplate(
            id: .speechTranscribe,
            category: .speech,
            title: "Transcribe",
            subtitle: "Speech to text",
            systemImage: "captions.bubble",
            inputKind: .audio,
            outputKind: .file("txt")
        ),
        CommandTemplate(
            id: .speechDiarize,
            category: .speech,
            title: "Diarize speakers",
            subtitle: "Identify who spoke when",
            systemImage: "person.2.fill",
            inputKind: .audio,
            outputKind: .file("json"),
            defaultModel: "speech-diarization-sortformer"
        ),
        CommandTemplate(id: .speechProfileList, category: .speech, title: "Voice profiles", subtitle: "List saved clone profiles", systemImage: "person.wave.2"),
        CommandTemplate(
            id: .speechProfileCreate,
            category: .speech,
            title: "Create voice profile",
            subtitle: "Save reference audio for clone mode",
            systemImage: "person.badge.plus",
            promptLabel: "Profile name",
            secondaryLabel: "Transcript override",
            inputKind: .audio,
            defaultPrompt: "Narration profile"
        ),
        CommandTemplate(
            id: .speechProfileDelete,
            category: .speech,
            title: "Delete voice profile",
            subtitle: "Remove a saved clone profile",
            systemImage: "person.badge.minus",
            promptLabel: "Profile UUID"
        ),
        CommandTemplate(
            id: .speechListen,
            category: .speech,
            title: "Live transcription",
            subtitle: "Stream microphone audio through live Qwen ASR",
            systemImage: "waveform.badge.mic"
        )
    ]
}

// MARK: - Speech arguments

extension CommandArguments {
    package static func speechSynthesize(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SpeechSynthesize
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        args.option(F.output, draft.outputPath)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.secondaryText.isBlank { args.option(F.voice, draft.secondaryText) }
        if draft.voiceMode == "clone" { args.option(F.mode, "clone") }
        if !draft.voiceProfile.isBlank { args.option(F.profile, draft.voiceProfile) }
        if !draft.refAudioPath.isBlank { args.option(F.refAudio, draft.refAudioPath) }
        if !draft.refText.isBlank { args.option(F.refText, draft.refText) }
        if !draft.saveProfileName.isBlank { args.option(F.saveProfile, draft.saveProfileName) }
        if !draft.language.isBlank, draft.language != "auto" { args.option(F.language, draft.language) }
        args.option(F.temperature, format(draft.temperature))
        if draft.stream {
            args.flag(F.stream)
            args.option(F.streamChunkTokens, String(draft.speechStreamChunkTokens))
        }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func speechTranscribe(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SpeechTranscribe
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.backend, draft.backend)
        args.option(F.task, draft.task)
        args.option(F.maxTokens, String(draft.maxTokens))
        if !draft.language.isBlank, draft.language != "auto" { args.option(F.language, draft.language) }
        if draft.stream {
            args.flag(F.stream)
            args.option(F.streamChunkMs, String(draft.speechStreamChunkMS))
            args.option(F.streamDecodeMs, String(draft.speechStreamDecodeMS))
        }
        if !draft.speechInputFormat.isBlank {
            args.option(F.inputFormat, draft.speechInputFormat)
        }
        if draft.speechSampleRate != 16_000 {
            args.option(F.sampleRate, String(draft.speechSampleRate))
        }
        if draft.speechJSONL { args.flag(F.jsonl) }
        if !draft.timestamps { args.flag(F.noTimestamps) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func speechDiarize(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SpeechDiarize
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if let outputFormat = draft.speechDiarizationFormat {
            args.option(F.format, outputFormat)
        }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if let threshold = draft.speechDiarizationThreshold {
            args.option(F.threshold, format(threshold))
        }
        if let minimumDuration = draft.speechDiarizationMinDuration {
            args.option(F.minDuration, format(minimumDuration))
        }
        if let mergeGap = draft.speechDiarizationMergeGap {
            args.option(F.mergeGap, format(mergeGap))
        }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func speechProfileList(_ draft: CommandDraft) -> [String] {
        CommandFlags.SpeechProfileList.command
    }

    package static func speechProfileCreate(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SpeechProfileCreate
        var args = ArgumentBuilder(F.self)
        args.option(F.name, draft.prompt)
        args.option(F.audio, draft.inputPath)
        if !draft.secondaryText.isBlank { args.option(F.text, draft.secondaryText) }
        if !draft.language.isBlank { args.option(F.language, draft.language) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func speechProfileDelete(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SpeechProfileDelete
        var args = ArgumentBuilder(F.self)
        args.option(F.id, draft.prompt)
        return args.arguments
    }

    package static func speechListen(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SpeechListen
        var args = ArgumentBuilder(F.self)
        if draft.speechListenListDevices { args.flag(F.listDevices) }
        if !draft.speechListenDevice.isBlank { args.option(F.device, draft.speechListenDevice) }
        if !draft.language.isBlank { args.option(F.language, draft.language) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.speechListenDecodeMS > 0 {
            args.option(F.decodeMs, String(draft.speechListenDecodeMS))
        }
        if draft.speechListenSilenceMS > 0 {
            args.option(F.silenceMs, String(draft.speechListenSilenceMS))
        }
        if draft.quiet { args.flag(F.quiet) }
        if draft.speechJSONL { args.flag(F.jsonl) }
        return args.arguments
    }
}
