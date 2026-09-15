import Foundation

extension MereRunCapabilityCatalog {
    public static let speechSynthesize = MereRunCommandCapability(
        id: "speech.synthesize",
        command: ["speech", "synthesize"],
        title: "Synthesize speech",
        summary: "Create styled or cloned speech, including streaming output.",
        arguments: [
            .init(name: "text", label: "Text", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Output", kind: .file, required: true, group: Group.output, tier: .essential),
            .init(
                flag: "--model", label: "Model", kind: .string,
                defaultValue: "speech-tts-qwen3-nano", group: Group.modelAndAdapters, tier: .essential
            ),
            .init(
                flag: "--voice", label: "Voice", kind: .string,
                defaultValue: "A calm female voice with clear pronunciation", group: Group.prompt, tier: .essential
            ),
            .init(
                flag: "--mode", label: "Mode", kind: .choice, choices: ["style", "clone"],
                defaultValue: "style", group: Group.inputs, tier: .standard
            ),
            .init(flag: "--profile", label: "Profile", kind: .string, group: Group.inputs, tier: .standard),
            .init(flag: "--ref-audio", label: "Reference audio", kind: .file, group: Group.inputs, tier: .standard),
            .init(
                flag: "--ref-text", label: "Reference text", kind: .string,
                group: Group.inputs, tier: .standard, dependsOn: "--ref-audio"
            ),
            .init(flag: "--language", label: "Language", kind: .string, defaultValue: "auto", group: Group.prompt, tier: .standard),
            .init(
                flag: "--save-profile", label: "Save profile", kind: .string,
                group: Group.inputs, tier: .expert, dependsOn: "--ref-audio"
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                defaultValue: "0.6", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(flag: "--stream", label: "Stream", kind: .boolean, group: Group.output, tier: .expert),
            .init(
                flag: "--stream-chunk-tokens", label: "Chunk tokens", kind: .integer,
                defaultValue: "25", group: Group.output, tier: .expert, range: .init(min: 1, max: 500, step: 1), dependsOn: "--stream"
            ),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let speechTranscribe = MereRunCommandCapability(
        id: "speech.transcribe",
        command: ["speech", "transcribe"],
        title: "Transcribe speech",
        summary: "Transcribe files or raw streaming audio with optional JSONL events.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--run-dir", label: "Run directory", kind: .directory, group: Group.output, tier: .expert),
            .init(flag: "--timestamps", label: "Include timestamps", kind: .boolean),
            .init(flag: "--output", label: "Output", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .standard),
            .init(
                flag: "--backend", label: "Backend", kind: .choice, choices: ["auto", "parakeet", "qwen"],
                defaultValue: "auto", group: Group.modelAndAdapters, tier: .essential
            ),
            .init(
                flag: "--provider", label: "Parakeet provider", kind: .choice, choices: ["mlx", "coreml"],
                defaultValue: "mlx", group: Group.modelAndAdapters, tier: .standard
            ),
            .init(
                flag: "--coreml-encoder", label: "Core ML artifact", kind: .directory,
                group: Group.modelAndAdapters, tier: .expert, dependsOn: "--provider"
            ),
            .init(
                flag: "--task", label: "Task", kind: .choice, choices: ["transcribe", "translate"],
                defaultValue: "transcribe", group: Group.prompt, tier: .essential
            ),
            .init(flag: "--language", label: "Language", kind: .string, group: Group.prompt, tier: .standard),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "448", group: Group.sampling, tier: .standard, range: .init(min: 1, max: 8_192, step: 1)
            ),
            .init(flag: "--stream", label: "Stream", kind: .boolean, group: Group.run, tier: .expert),
            .init(
                flag: "--stream-chunk-ms", label: "Feed interval", kind: .integer,
                defaultValue: "200", group: Group.run, tier: .expert, range: .init(min: 10, max: 5_000, step: 10), dependsOn: "--stream"
            ),
            .init(
                flag: "--stream-decode-ms", label: "Decode interval", kind: .integer,
                defaultValue: "2000", group: Group.run, tier: .expert, range: .init(min: 100, max: 10_000, step: 100), dependsOn: "--stream"
            ),
            .init(flag: "--input-format", label: "Input format", kind: .string, group: Group.inputs, tier: .expert, dependsOn: "--stream"),
            // Raw stdin protocol v1 accepts exactly 16 kHz, so the range pins
            // the single value a shell may send rather than a span.
            .init(
                flag: "--sample-rate", label: "Sample rate", kind: .integer,
                group: Group.inputs, tier: .expert, range: .init(min: 16_000, max: 16_000, step: 1),
                dependsOn: "--stream"
            ),
            .init(flag: "--jsonl", label: "JSON Lines", kind: .boolean, group: Group.output, tier: .expert, dependsOn: "--stream"),
            .init(flag: "--no-timestamps", label: "No timestamps", kind: .boolean, group: Group.output, tier: .standard),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            receiptOption
        ],
        output: .init(kind: .text, fileExtension: "txt", flag: "--output", optional: true)
    )

    public static let speechDiarize = MereRunCommandCapability(
        id: "speech.diarize",
        command: ["speech", "diarize"],
        title: "Diarize speech",
        summary: "Identify speaker activity in a local audio file as JSON or RTTM.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--format", label: "Format", kind: .choice, choices: ["json", "rttm"]),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--threshold", label: "Threshold", kind: .number),
            .init(flag: "--min-duration", label: "Minimum duration", kind: .number),
            .init(flag: "--merge-gap", label: "Merge gap", kind: .number),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text, flag: "--output", optional: true)
    )

    public static let speechProfileList = MereRunCommandCapability(
        id: "speech.profile.list",
        command: ["speech", "profile", "list"],
        title: "Voice profiles",
        summary: "List saved voice-cloning profiles.",
        options: [],
        output: .init(kind: .text)
    )

    public static let speechProfileCreate = MereRunCommandCapability(
        id: "speech.profile.create",
        command: ["speech", "profile", "create"],
        title: "Create voice profile",
        summary: "Create a reusable voice-cloning profile.",
        options: [
            .init(flag: "--name", label: "Name", kind: .string, required: true),
            .init(flag: "--audio", label: "Audio", kind: .file, required: true),
            .init(flag: "--text", label: "Transcript", kind: .string),
            .init(flag: "--language", label: "Language", kind: .string),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let speechProfileDelete = MereRunCommandCapability(
        id: "speech.profile.delete",
        command: ["speech", "profile", "delete"],
        title: "Delete voice profile",
        summary: "Delete one saved voice profile.",
        options: [
            .init(flag: "--id", label: "Profile id", kind: .string, required: true)
        ],
        output: .init(kind: .text)
    )

    public static let speechListen = MereRunCommandCapability(
        id: "speech.listen",
        command: ["speech", "listen"],
        title: "Live transcription",
        summary: "Transcribe a macOS microphone with live Qwen ASR.",
        options: [
            .init(flag: "--device", label: "Input device", kind: .string),
            .init(flag: "--list-devices", label: "List devices", kind: .boolean),
            .init(flag: "--language", label: "Language", kind: .string),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--decode-ms", label: "Decode window", kind: .integer),
            .init(flag: "--silence-ms", label: "Silence window", kind: .integer),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean),
            .init(flag: "--jsonl", label: "JSONL", kind: .boolean)
        ],
        output: .init(kind: .text)
    )
}
