import Foundation

// MARK: - Audio templates

extension CommandCatalog {
    package static let audioTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .audioEdit, category: .media, title: "AuK speech editing",
            subtitle: "Experimental instruction-based speech generation and editing",
            systemImage: "waveform", promptLabel: "Instruction",
            secondaryLabel: "Qwen encoder directory (optional)", inputKind: .audio,
            outputKind: .file("wav"), defaultPrompt: "Say hello in a calm voice.",
            defaultModel: "audio-auk-base"
        ),
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
        )
    ]
}

// MARK: - Audio arguments

extension CommandArguments {
    package static func audioEdit(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.AudioEdit
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        args.option(F.model, draft.model)
        if !draft.inputPath.isBlank { args.option(F.audio, draft.inputPath) }
        if !draft.modelRoot.isBlank { args.option(F.modelPath, draft.modelRoot) }
        if !draft.secondaryText.isBlank { args.option(F.thinkerPath, draft.secondaryText) }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if draft.useDuration { args.option(F.duration, format(draft.durationSeconds)) }
        args.option(F.steps, String(draft.steps))
        if let guidance = draft.audioGuidanceScale { args.option(F.guidance, format(guidance)) }
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
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
}
