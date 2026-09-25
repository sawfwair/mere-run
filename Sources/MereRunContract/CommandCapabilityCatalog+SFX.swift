import Foundation

extension MereRunCapabilityCatalog {
    public static let sfxGenerate: MereRunCommandCapability = {
        typealias F = SFXGenerateFamily
        return MereRunCommandCapability(
            id: "sfx.generate",
            command: ["sfx", "generate"],
            title: "Generate sound effect",
            summary: "Generate a Woosh or MMAudio sound effect from text.",
            arguments: [
                .init(name: "prompt", label: "Prompt", kind: .string, required: true)
            ],
            options: [
                .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string, group: Group.prompt, tier: .standard)
                    .scoped(F.only(.mmaudio)),
                .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file, group: Group.output, tier: .standard),
                .init(
                    flag: "--model", aliases: ["-m"], label: "Model", kind: .string,
                    defaultValue: "sfx-woosh-dflow", group: Group.modelAndAdapters, tier: .essential
                ),
                .init(
                    flag: "--duration", label: "Duration", kind: .number,
                    group: Group.sampling, tier: .essential, range: .init(min: 0.5, max: 30, step: 0.5)
                ).scoped(F.rule(.wooshDFlow, defaultValue: "5"), .rule(.wooshFlow, defaultValue: "5"),
                         .rule(.mmaudio, defaultValue: "8")),
                // Woosh Flow's own schedule is 32 steps, but the command runs 4 unless told.
                .init(
                    flag: "--steps", aliases: ["-s"], label: "Steps", kind: .integer,
                    group: Group.sampling, tier: .standard, range: .init(min: 1, max: 100, step: 1)
                ).scoped(F.rule(.wooshDFlow, defaultValue: "4"), .rule(.wooshFlow, defaultValue: "4"),
                         .rule(.mmaudio, defaultValue: "25")),
                .init(
                    flag: "--cfg", label: "CFG", kind: .number,
                    defaultValue: "4.5", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 20, step: 0.1)
                ),
                .init(flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .essential, range: .init(min: 0, step: 1)),
                // Woosh Flow validates the schedule, then samples without renoising.
                .init(flag: "--renoise", label: "Renoise", kind: .string, group: Group.sampling, tier: .expert)
                    .scoped(F.only(.wooshDFlow, ignoredBy: [.wooshFlow])),
                .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
                progressJSONOption,
                receiptOption
            ],
            output: .init(kind: .file, fileExtension: "wav", flag: "--output"),
            routing: sfxGenerateRouting
        )
    }()

    public static let sfxVideoGenerate: MereRunCommandCapability = {
        typealias F = SFXVideoGenerateFamily
        return MereRunCommandCapability(
            id: "sfx.video.generate",
            command: ["sfx", "video", "generate"],
            title: "Video foley",
            summary: "Generate synchronized sound effects from video conditioning.",
            arguments: [
                .init(name: "prompt", label: "Prompt", kind: .string, required: true),
                .init(name: "input", label: "Video or features", kind: .file, required: true)
            ],
            options: [
                .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string).scoped(F.only(.mmaudio)),
                .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file),
                .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
                // MMAudio loads the Synchformer in its own snapshot.
                .init(flag: "--synchformer-model", label: "Synchformer", kind: .string)
                    .scoped(F.only(.wooshDVFlow, .wooshVFlow, ignoredBy: [.mmaudio])),
                .init(flag: "--duration", label: "Duration", kind: .number, defaultValue: "8.0", group: Group.sampling, tier: .essential),
                .init(flag: "--steps", aliases: ["-s"], label: "Steps", kind: .integer, group: Group.sampling, tier: .essential)
                    .scoped(F.rule(.wooshDVFlow, defaultValue: "4"), .rule(.wooshVFlow, defaultValue: "32"),
                            .rule(.mmaudio, defaultValue: "25")),
                .init(flag: "--cfg", label: "CFG", kind: .number)
                    .scoped(F.rule(.wooshDVFlow, defaultValue: "3"), .rule(.wooshVFlow, defaultValue: "4.5"),
                            .rule(.mmaudio, defaultValue: "4.5")),
                .init(flag: "--seed", label: "Seed", kind: .integer),
                // Woosh VFlow validates the schedule, then samples without renoising.
                .init(flag: "--renoise", label: "Renoise", kind: .string)
                    .scoped(F.only(.wooshDVFlow, ignoredBy: [.wooshVFlow])),
                .init(flag: "--sync-batch-size", label: "Sync batch", kind: .integer),
                // Woosh validates the size, then never reads it.
                .init(flag: "--clip-batch-size", label: "CLIP batch", kind: .integer)
                    .scoped(F.only(.mmaudio, ignoredBy: [.wooshDVFlow, .wooshVFlow])),
                .init(flag: "--preflight", label: "Preflight", kind: .boolean),
                .init(flag: "--json", label: "JSON", kind: .boolean),
                .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean)
            ],
            output: .init(kind: .file, fileExtension: "wav", flag: "--output"),
            routing: sfxVideoGenerateRouting
        )
    }()

    public static let sfxAEEncode = MereRunCommandCapability(
        id: "sfx.ae.encode",
        command: ["sfx", "ae", "encode"],
        title: "Encode SFX latents",
        summary: "Encode audio into Woosh latent arrays.",
        arguments: [.init(name: "input", label: "Audio", kind: .file, required: true)],
        options: [
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file),
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "npy", flag: "--output")
    )

    public static let sfxAEDecode = MereRunCommandCapability(
        id: "sfx.ae.decode",
        command: ["sfx", "ae", "decode"],
        title: "Decode SFX latents",
        summary: "Decode Woosh latent arrays into audio.",
        arguments: [.init(name: "input", label: "Latents", kind: .file, required: true)],
        options: [
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file),
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let sfxCLAPScore = MereRunCommandCapability(
        id: "sfx.clap.score",
        command: ["sfx", "clap", "score"],
        title: "CLAP score",
        summary: "Score semantic alignment between a prompt and audio.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true),
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text),
        routing: sfxCLAPScoreRouting
    )

    public static let sfxConditionText = MereRunCommandCapability(
        id: "sfx.condition.text",
        command: ["sfx", "condition", "text"],
        title: "SFX text conditioning",
        summary: "Export text-conditioning tensors for Woosh.",
        arguments: [.init(name: "prompt", label: "Prompt", kind: .string, required: true)],
        options: [
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file),
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )
}
