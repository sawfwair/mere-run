import Foundation

// MARK: - Sound FX templates

extension CommandCatalog {
    static let soundFXTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .sfxGenerate,
            category: .sfx,
            title: "Generate sound effect",
            subtitle: "Woosh or MMAudio text-to-audio",
            systemImage: "speaker.wave.2",
            promptLabel: "Prompt",
            secondaryLabel: "Negative prompt",
            outputKind: .file("wav"),
            defaultPrompt: "a heavy wooden door creaking open",
            defaultModel: "sfx-woosh-dflow"
        ),
        CommandTemplate(
            id: .sfxVideo,
            category: .sfx,
            title: "Video foley",
            subtitle: "Generate sound effects from a video",
            systemImage: "video.badge.waveform",
            promptLabel: "Prompt",
            secondaryLabel: "Negative prompt",
            inputKind: .video,
            outputKind: .file("wav"),
            defaultPrompt: "footsteps on gravel",
            defaultModel: "sfx-woosh-dvflow-8s"
        ),
        CommandTemplate(
            id: .sfxAEEncode,
            category: .sfx,
            title: "Autoencoder · encode",
            subtitle: "Audio → Woosh latents (.npy)",
            systemImage: "arrow.down.doc",
            inputKind: .audio,
            outputKind: .file("npy"),
            defaultModel: "sfx-woosh-dflow"
        ),
        CommandTemplate(
            id: .sfxAEDecode,
            category: .sfx,
            title: "Autoencoder · decode",
            subtitle: "Woosh latents (.npy) → audio",
            systemImage: "arrow.up.doc",
            inputKind: .file([.data]),
            outputKind: .file("wav"),
            defaultModel: "sfx-woosh-dflow"
        ),
        CommandTemplate(
            id: .sfxClapScore,
            category: .sfx,
            title: "CLAP score",
            subtitle: "Score audio against a prompt (JSON)",
            systemImage: "checkmark.seal",
            promptLabel: "Prompt",
            inputKind: .audio,
            defaultPrompt: "a heavy wooden door creaking open",
            defaultModel: "sfx-woosh-clap"
        ),
        CommandTemplate(
            id: .sfxConditionText,
            category: .sfx,
            title: "Conditioning · text",
            subtitle: "Export Woosh conditioning tensors",
            systemImage: "function",
            promptLabel: "Prompt",
            outputKind: .file("safetensors"),
            defaultPrompt: "a heavy wooden door creaking open",
            defaultModel: "sfx-woosh-dflow"
        )
    ]
}

// MARK: - Sound FX arguments

extension CommandArguments {
    static func sfxGenerate(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SFXGenerate
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if !draft.secondaryText.isBlank {
            args.option(F.negativePrompt, draft.secondaryText)
        }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.duration, format(draft.durationSeconds))
        args.option(F.steps, String(draft.steps))
        if draft.cfgScale != 1.0 { args.option(F.cfg, format(draft.cfgScale)) }
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if !draft.sfxRenoise.isBlank { args.option(F.renoise, draft.sfxRenoise) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    static func sfxVideo(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SFXVideoGenerate
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        args.value(draft.inputPath)
        if !draft.secondaryText.isBlank {
            args.option(F.negativePrompt, draft.secondaryText)
        }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.duration, format(draft.durationSeconds))
        args.option(F.steps, String(draft.steps))
        if draft.cfgScale > 0 { args.option(F.cfg, format(draft.cfgScale)) }
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if !draft.sfxRenoise.isBlank { args.option(F.renoise, draft.sfxRenoise) }
        if !draft.sfxSynchformerModel.isBlank {
            args.option(F.synchformerModel, draft.sfxSynchformerModel)
        }
        args.option(F.syncBatchSize, String(draft.sfxSyncBatchSize))
        args.option(F.clipBatchSize, String(draft.sfxClipBatchSize))
        if draft.preflight { args.flag(F.preflight) }
        if draft.preflight, draft.json { args.flag(F.json) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    static func sfxAEEncode(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SFXAEEncode
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    static func sfxAEDecode(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SFXAEDecode
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    static func sfxClapScore(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SFXClapScore
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        args.value(draft.inputPath)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    static func sfxConditionText(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.SFXConditionText
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }
}
