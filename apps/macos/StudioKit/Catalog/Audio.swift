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
        let scope = ContractFamilyScope(MereRunCapabilityCatalog.audioEdit, arguments: [F.model, draft.model])
        var args = ArgumentBuilder(F.self)
        args.value(draft.prompt)
        args.option(F.model, draft.model)
        if !draft.inputPath.isBlank { args.option(F.audio, draft.inputPath) }
        if !draft.modelRoot.isBlank { args.option(F.modelPath, draft.modelRoot) }
        if !draft.secondaryText.isBlank { args.option(F.thinkerPath, draft.secondaryText) }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if draft.useDuration { args.option(F.duration, format(draft.durationSeconds)) }
        if !scope.fixes(F.steps) { args.option(F.steps, String(draft.steps)) }
        if let guidance = draft.audioGuidanceScale, !scope.fixes(F.guidance) {
            args.option(F.guidance, format(guidance))
        }
        if !draft.seed.isBlank { args.option(F.seed, draft.seed) }
        if draft.quiet { args.flag(F.quiet) }
        return args.arguments
    }

    package static func audioEnhance(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.AudioEnhance
        let scope = ContractFamilyScope(MereRunCapabilityCatalog.audioEnhance, arguments: modelArguments(F.model, draft))
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.modelRoot.isBlank { args.option(F.modelPath, draft.modelRoot) }
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if let overlap = draft.audioOverlap, scope.reads(F.overlap) { args.option(F.overlap, String(overlap)) }
        if let inputRate = draft.audioInputRate, !scope.fixes(F.inputRate) {
            args.option(F.inputRate, String(inputRate))
        }
        if let method = draft.audioODEMethod, !method.isBlank, scope.reads(F.odeMethod) {
            args.option(F.odeMethod, method)
        }
        if let steps = draft.audioODESteps, scope.reads(F.odeSteps) { args.option(F.odeSteps, String(steps)) }
        if let guidance = draft.audioGuidanceScale, scope.reads(F.guidanceScale) {
            args.option(F.guidanceScale, format(guidance))
        }
        if !draft.seed.isBlank, scope.reads(F.seed) { args.option(F.seed, draft.seed) }
        if let seconds = draft.audioChunkSeconds, scope.reads(F.chunkSeconds) {
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

// MARK: - Runtime family scope

/// A capability's options as the contract scopes them for the runtime family a command line
/// resolves to. When the contract can't name the family without inspecting a local folder, every
/// option counts as read, so the builder sends what the draft holds and the CLI decides.
struct ContractFamilyScope {
    let family: String?
    private let options: [String: MereRunCapabilityOption]

    /// `arguments` are the ones after the command path that choose the family: the model flag,
    /// or the selector flags.
    init(_ capability: MereRunCommandCapability, arguments: [String]) {
        let invocation = MereRunCommandInvocation(capability: capability, arguments: arguments)
        if case .family(let id, _, _) = capability.resolveFamily(invocation) {
            family = id
        } else {
            family = nil
        }
        options = Dictionary(uniqueKeysWithValues: capability.options(forFamily: family).map { ($0.flag, $0) })
    }

    /// The family reads `flag`: it neither refuses nor ignores it.
    func reads(_ flag: String) -> Bool {
        options[flag] != nil
    }

    /// The family runs one value of `flag` whatever is passed, so the builder leaves it off.
    func fixes(_ flag: String) -> Bool {
        family != nil && options[flag]?.familyRules.first?.values?.count == 1
    }
}

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
    package static func audioValidationMessage(for id: CommandTemplateID, draft: CommandDraft) -> String? {
        switch id {
        case .audioEdit:
            if draft.inputPath.isBlank && !draft.useDuration { return "Choose a duration or reference audio." }
            if draft.useDuration && (!draft.durationSeconds.isFinite || draft.durationSeconds <= 0 || draft.durationSeconds > 300) {
                return "Duration must be in (0, 300] seconds."
            }
            if !["audio-auk-base", "audio-auk-flash"].contains(draft.model) { return "Choose an AuK base or Flash model." }
            let scope = ContractFamilyScope(
                MereRunCapabilityCatalog.audioEdit, arguments: [CommandFlags.AudioEdit.model, draft.model]
            )
            if !scope.fixes(CommandFlags.AudioEdit.steps), !(1...1000).contains(draft.steps) {
                return "Steps must be in 1...1000."
            }
            if let guidance = draft.audioGuidanceScale, !guidance.isFinite || guidance < 0 {
                return "Guidance must be finite and non-negative."
            }
        case .audioEnhance:
            if let overlap = draft.audioOverlap, overlap <= 0 {
                return "Overlap must be positive."
            }
            let scope = ContractFamilyScope(
                MereRunCapabilityCatalog.audioEnhance,
                arguments: CommandArguments.modelArguments(CommandFlags.AudioEnhance.model, draft)
            )
            // UniverSR's controls, checked when the selected model's family reads them.
            if scope.reads(CommandFlags.AudioEnhance.odeSteps) {
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
