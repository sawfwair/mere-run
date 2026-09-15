import Foundation

extension MereRunCapabilityCatalog {
    public static let sfxGenerate = MereRunCommandCapability(
        id: "sfx.generate",
        command: ["sfx", "generate"],
        title: "Generate sound effect",
        summary: "Generate a Woosh or MMAudio sound effect from text.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string, group: Group.prompt, tier: .standard),
            .init(flag: "--output", label: "Output", kind: .file, group: Group.output, tier: .standard),
            .init(
                flag: "--model", label: "Model", kind: .string,
                defaultValue: "sfx-woosh-dflow", group: Group.modelAndAdapters, tier: .essential
            ),
            .init(
                flag: "--duration", label: "Duration", kind: .number,
                group: Group.sampling, tier: .essential, range: .init(min: 0.5, max: 30, step: 0.5)
            ),
            .init(
                flag: "--steps", label: "Steps", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 100, step: 1)
            ),
            .init(
                flag: "--cfg", label: "CFG", kind: .number,
                defaultValue: "4.5", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .essential, range: .init(min: 0, step: 1)),
            .init(flag: "--renoise", label: "Renoise", kind: .string, group: Group.sampling, tier: .expert),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let sfxVideoGenerate = MereRunCommandCapability(
        id: "sfx.video.generate",
        command: ["sfx", "video", "generate"],
        title: "Video foley",
        summary: "Generate synchronized sound effects from video conditioning.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true),
            .init(name: "input", label: "Video or features", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--synchformer-model", label: "Synchformer", kind: .string),
            .init(flag: "--duration", label: "Duration", kind: .number),
            .init(flag: "--steps", label: "Steps", kind: .integer),
            .init(flag: "--cfg", label: "CFG", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--renoise", label: "Renoise", kind: .string),
            .init(flag: "--sync-batch-size", label: "Sync batch", kind: .integer),
            .init(flag: "--clip-batch-size", label: "CLIP batch", kind: .integer),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let sfxAEEncode = MereRunCommandCapability(
        id: "sfx.ae.encode",
        command: ["sfx", "ae", "encode"],
        title: "Encode SFX latents",
        summary: "Encode audio into Woosh latent arrays.",
        arguments: [.init(name: "input", label: "Audio", kind: .file, required: true)],
        options: [
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
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
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
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
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let sfxConditionText = MereRunCommandCapability(
        id: "sfx.condition.text",
        command: ["sfx", "condition", "text"],
        title: "SFX text conditioning",
        summary: "Export text-conditioning tensors for Woosh.",
        arguments: [.init(name: "prompt", label: "Prompt", kind: .string, required: true)],
        options: [
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )
}
