import Foundation

extension MereRunCapabilityCatalog {
    public static let audioEdit: MereRunCommandCapability = {
        typealias F = AudioEditFamily
        return MereRunCommandCapability(
            id: "audio.edit", command: ["audio", "edit"], title: "Edit or generate AuK speech",
            summary: "Generate or edit speech from instructions with experimental native AuK.",
            arguments: [.init(name: "instruction", label: "Instruction", kind: .string, required: true)],
            options: [
                .init(flag: "--audio", label: "Reference audio", kind: .file),
                .init(flag: "--model", label: "Model", kind: .string),
                .init(flag: "--model-path", label: "AuK checkpoint", kind: .directory),
                .init(flag: "--thinker-path", label: "Qwen encoder", kind: .directory),
                .init(flag: "--output", aliases: ["-o"], label: "Output WAV", kind: .file),
                .init(flag: "--duration", label: "Duration", kind: .number),
                // Flash validates the base range, then runs its fixed four-step schedule.
                .init(
                    flag: "--steps", label: "Base steps", kind: .integer,
                    defaultValue: "32", range: .init(min: 1, max: 1_000, step: 1)
                ).scoped(F.rule(.aukFlash, values: ["4"], severity: .warning)),
                // Flash validates the value, then runs with guidance off.
                .init(flag: "--guidance", label: "Base guidance", kind: .number, defaultValue: "2.0")
                    .scoped(F.rule(.aukFlash, values: ["0"], severity: .warning)),
                .init(flag: "--seed", label: "Seed", kind: .integer, defaultValue: "42"),
                .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean)
            ],
            output: .init(kind: .file, fileExtension: "wav", flag: "--output"),
            routing: audioEditRouting
        )
    }()

    public static let audioEnhance: MereRunCommandCapability = {
        typealias F = AudioEnhanceFamily
        return MereRunCommandCapability(
            id: "audio.enhance",
            command: ["audio", "enhance"],
            title: "Enhance audio",
            summary: "Extend speech or general-audio bandwidth to 48 kHz with native AP-BWE or UniverSR.",
            arguments: [
                .init(name: "audio", label: "Audio", kind: .file, required: true)
            ],
            options: [
                .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
                .init(flag: "--model-path", label: "Model path", kind: .directory, group: Group.modelAndAdapters, tier: .expert),
                .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file, group: Group.output, tier: .standard),
                // The CLI also requires a divisor of AP-BWE's 96000-sample chunk; that arithmetic
                // stays in the command.
                .init(
                    flag: "--overlap", label: "AP-BWE overlap", kind: .integer,
                    tier: .standard, range: .init(min: 1, max: 64, step: 1)
                ).scoped(F.only(.apBWE)),
                // AP-BWE always reads 16 kHz and refuses any other rate.
                .init(
                    flag: "--input-rate", label: "UniverSR input bandwidth", kind: .choice,
                    choices: ["8000", "12000", "16000", "24000"], tier: .standard
                ).scoped(F.rule(.apBWE, values: ["16000"])),
                .init(
                    flag: "--ode-method",
                    label: "UniverSR ODE method",
                    kind: .choice,
                    choices: ["euler", "midpoint", "rk4"],
                    defaultValue: "midpoint",
                    tier: .standard
                ).scoped(F.only(.univerSR, ignoredBy: [.apBWE])),
                .init(
                    flag: "--ode-steps", label: "UniverSR ODE steps", kind: .integer,
                    defaultValue: "4", tier: .standard, range: .init(min: 1, max: 100, step: 1)
                ).scoped(F.only(.univerSR, ignoredBy: [.apBWE])),
                .init(
                    flag: "--guidance-scale", label: "UniverSR guidance", kind: .number,
                    defaultValue: "1.5", tier: .standard, range: .init(min: 0, max: 10, step: 0.1)
                ).scoped(F.only(.univerSR, ignoredBy: [.apBWE])),
                .init(flag: "--seed", label: "Seed", kind: .integer, defaultValue: "42", tier: .standard)
                    .scoped(F.only(.univerSR, ignoredBy: [.apBWE])),
                .init(
                    flag: "--chunk-seconds", label: "UniverSR chunk (s)", kind: .integer,
                    defaultValue: "10", tier: .standard, range: .init(min: 3, max: 600, step: 1)
                ).scoped(F.only(.univerSR, ignoredBy: [.apBWE])),
                .init(
                    flag: "--dtype",
                    label: "Compute",
                    kind: .choice,
                    choices: ["float16", "float32"],
                    defaultValue: "float32",
                    tier: .essential
                ),
                .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert)
            ],
            output: .init(kind: .file, fileExtension: "wav", flag: "--output"),
            routing: audioEnhanceRouting
        )
    }()

    public static let audioGenerate = MereRunCommandCapability(
        id: "audio.generate",
        command: ["audio", "generate"],
        title: "Generate LTX audio",
        summary: "Generate audio only with the native LTX-2.5 text-to-audio pipeline.",
        arguments: [
            .init(name: "prompt", label: "Audio prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--output", aliases: ["-o"], label: "WAV output", kind: .file),
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string),
            .init(flag: "--enhance-prompt", label: "Enhance prompt", kind: .boolean),
            .init(flag: "--prompt-enhancer-model", label: "Prompt enhancer", kind: .string),
            .init(flag: "--prompt-enhancer-model-root", label: "Prompt enhancer root", kind: .directory),
            .init(flag: "--duration", label: "Duration", kind: .number),
            .init(flag: "--auto-duration", label: "Auto duration range", kind: .string, repeatable: true),
            .init(flag: "--num-frames", label: "Video-clock frames", kind: .integer),
            .init(flag: "--fps", label: "Video-clock rate", kind: .integer),
            .init(flag: "--steps", label: "Denoising steps", kind: .integer),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--audio-cfg-guidance-scale", label: "Audio CFG", kind: .number),
            .init(flag: "--audio-stg-guidance-scale", label: "Audio STG", kind: .number),
            .init(flag: "--audio-rescale", label: "Guidance rescale", kind: .number),
            .init(flag: "--audio-stg-block", label: "STG block", kind: .integer, repeatable: true),
            .init(flag: "--audio-skip-step", label: "Guidance skip", kind: .integer),
            .init(flag: "--sigmas", label: "Sigma schedule", kind: .string),
            .init(flag: "--lora", label: "Audio LoRA", kind: .string, repeatable: true),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )
}
