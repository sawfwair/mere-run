import Foundation

extension MereRunCapabilityCatalog {
    public static let audioEnhance = MereRunCommandCapability(
        id: "audio.enhance",
        command: ["audio", "enhance"],
        title: "Enhance audio",
        summary: "Extend speech or general-audio bandwidth to 48 kHz with native AP-BWE or UniverSR.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-path", label: "Model path", kind: .directory),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--overlap", label: "AP-BWE overlap", kind: .integer),
            .init(flag: "--input-rate", label: "Input bandwidth", kind: .integer),
            .init(
                flag: "--ode-method",
                label: "UniverSR ODE method",
                kind: .choice,
                choices: ["euler", "midpoint", "rk4"]
            ),
            .init(flag: "--ode-steps", label: "UniverSR ODE steps", kind: .integer),
            .init(flag: "--guidance-scale", label: "UniverSR guidance", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--chunk-seconds", label: "Chunk seconds", kind: .integer),
            .init(
                flag: "--dtype",
                label: "Compute type",
                kind: .choice,
                choices: ["float16", "float32"]
            ),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let audioGenerate = MereRunCommandCapability(
        id: "audio.generate",
        command: ["audio", "generate"],
        title: "Generate LTX audio",
        summary: "Generate audio only with the native LTX-2.5 text-to-audio pipeline.",
        arguments: [
            .init(name: "prompt", label: "Audio prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--output", label: "WAV output", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
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
