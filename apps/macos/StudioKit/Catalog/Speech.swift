import Foundation
import MereRunContract

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
        ),
        CommandTemplate(
            id: .speechDiarizeLive,
            category: .speech,
            title: "Live speakers",
            subtitle: "Stream Nemotron 3 speaker activity from the microphone",
            systemImage: "person.2.wave.2",
            defaultModel: "speech-diarization-nemotron3"
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
        // `--mode` picks what the CLI reads: a voice description (and, on CustomVoice, a named
        // speaker) in style mode, a reference in clone mode.
        if draft.voiceMode == "clone" {
            args.option(F.mode, "clone")
            if !draft.voiceProfile.isBlank { args.option(F.profile, draft.voiceProfile) }
            if !draft.refAudioPath.isBlank { args.option(F.refAudio, draft.refAudioPath) }
            if !draft.refText.isBlank { args.option(F.refText, draft.refText) }
            if !draft.saveProfileName.isBlank { args.option(F.saveProfile, draft.saveProfileName) }
            if draft.model == "speech-tts-breeze-2", !draft.secondaryText.isBlank {
                args.option(F.voice, draft.secondaryText)
            }
        } else {
            if let speaker = draft.voiceSpeaker, !speaker.isBlank { args.option(F.speaker, speaker) }
            if !draft.secondaryText.isBlank { args.option(F.voice, draft.secondaryText) }
        }
        if !draft.language.isBlank, draft.language != "auto" { args.option(F.language, draft.language) }
        args.option(F.temperature, format(draft.temperature))
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if let scale = draft.speechCFGScale { args.option(F.cfgScale, format(scale)) }
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
        // Only Qwen3-ASR reads a token budget; Parakeet warns about one it would ignore.
        args.optionUnlessDefault(F.maxTokens, String(draft.maxTokens))
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
        if let latency = draft.speechDiarizationLatency, latency != "offline" {
            args.option(F.latency, latency)
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

    package static func speechDiarizeLive(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SpeechDiarizeLive
        var args = ArgumentBuilder(F.self)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.speechListenDevice.isBlank { args.option(F.device, draft.speechListenDevice) }
        if draft.speechListenListDevices { args.flag(F.listDevices) }
        if draft.speechDiarizationLiveStdin { args.flag(F.stdin) }
        if let latency = draft.speechDiarizationLatency { args.option(F.latency, latency) }
        if let threshold = draft.speechDiarizationThreshold {
            args.option(F.threshold, format(threshold))
        }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }
}

// MARK: - Speech validation

extension CommandCatalog {
    /// The reason a speech template's draft cannot run, beyond the prompt and input checks
    /// every template shares; nil for a draft that can, and for every other template.
    package static func speechValidationMessage(for id: CommandTemplateID, draft: CommandDraft) -> String? {
        switch id {
        case .speechSynthesize:
            if draft.stream && draft.speechStreamChunkTokens < 1 {
                return "Streaming chunk tokens must be greater than zero."
            }
        case .speechTranscribe:
            if draft.stream && (draft.speechStreamChunkMS < 1 || draft.speechStreamDecodeMS < 1) {
                return "Streaming feed and decode intervals must be greater than zero."
            }
        case .speechDiarize:
            let format = draft.speechDiarizationFormat ?? "json"
            if !["json", "rttm"].contains(format) {
                return "Diarization format must be JSON or RTTM."
            }
            let latency = draft.speechDiarizationLatency ?? "offline"
            if !["offline", "1.04", "0.64", "0.32"].contains(latency) {
                return "Nemotron 3 latency must be offline, 1.04, 0.64, or 0.32 seconds."
            }
            if !(0...1).contains(draft.speechDiarizationThreshold ?? 0.5) {
                return "Diarization threshold must be between zero and one."
            }
            if (draft.speechDiarizationMinDuration ?? 0.25) < 0 {
                return "Minimum speaker duration must be zero or greater."
            }
            if (draft.speechDiarizationMergeGap ?? 0.25) < 0 {
                return "Speaker merge gap must be zero or greater."
            }
        case .speechDiarizeLive:
            let latency = draft.speechDiarizationLatency ?? "1.04"
            if !["1.04", "0.64", "0.32"].contains(latency) {
                return "Live diarization latency must be 1.04, 0.64, or 0.32 seconds."
            }
            if !draft.model.isBlank && draft.model != "speech-diarization-nemotron3" {
                return "Live diarization requires speech-diarization-nemotron3."
            }
            if !(0...1).contains(draft.speechDiarizationThreshold ?? 0.5) {
                return "Diarization threshold must be between zero and one."
            }
        default:
            break
        }
        return nil
    }
}
