import Foundation
import MereRunContract

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
        if let guidance = draft.audioGuidanceScale {
            args.option(F.guidance, format(guidance))
        }
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
        if let inputRate = draft.audioInputRate {
            args.option(F.inputRate, String(inputRate))
        }
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

// MARK: - Model arguments

extension CommandArguments {
    /// `[flag, model]` for a chosen model, or nothing, so a blank model resolves to the
    /// contract's default the way the CLI's own default does.
    static func modelArguments(_ flag: String, _ draft: CommandDraft) -> [String] {
        draft.model.isBlank ? [] : [flag, draft.model]
    }
}

// MARK: - Audio validation

extension CommandCatalog {
    /// The reason an audio template's draft cannot run, beyond the prompt and input checks
    /// every template shares; nil for a draft that can, and for every other template.
    package static func audioValidationMessage(
        for id: CommandTemplateID,
        draft: CommandDraft,
        source: StudioScopeSource
    ) -> String? {
        switch id {
        case .audioEdit:
            if draft.inputPath.isBlank && !draft.useDuration { return "Choose a duration or reference audio." }
            if draft.useDuration && (!draft.durationSeconds.isFinite || draft.durationSeconds <= 0 || draft.durationSeconds > 300) {
                return "Duration must be in (0, 300] seconds."
            }
            if !["audio-auk-base", "audio-auk-flash"].contains(draft.model) { return "Choose an AuK base or Flash model." }
            let scope = source.scope(capability: MereRunCapabilityCatalog.audioEdit, commandLine: [CommandFlags.AudioEdit.model, draft.model]
            )
            if scope.fixedValue(CommandFlags.AudioEdit.steps) == nil, !(1...1000).contains(draft.steps) {
                return "Steps must be in 1...1000."
            }
            if let guidance = draft.audioGuidanceScale, !guidance.isFinite || guidance < 0 {
                return "Guidance must be finite and non-negative."
            }
        case .audioEnhance:
            if let overlap = draft.audioOverlap, overlap <= 0 {
                return "Overlap must be positive."
            }
            let scope = source.scope(capability: MereRunCapabilityCatalog.audioEnhance, commandLine: CommandArguments.modelArguments(CommandFlags.AudioEnhance.model, draft)
            )
            // UniverSR's controls, checked when the selected model's family reads them.
            if scope.allows(CommandFlags.AudioEnhance.odeSteps) {
                if let inputRate = draft.audioInputRate,
                   ![8_000, 12_000, 16_000, 24_000].contains(inputRate) {
                    return "UniverSR input bandwidth must be 8000, 12000, 16000, or 24000 Hz."
                }
                if (draft.audioODESteps ?? 4) <= 0 {
                    return "UniverSR ODE steps must be positive."
                }
                if (draft.audioChunkSeconds ?? 10) < 3 {
                    return "UniverSR chunks must be at least 3 seconds."
                }
            }
        default:
            break
        }
        return nil
    }
}
