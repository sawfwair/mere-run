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
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file, required: true, group: Group.output, tier: .essential),
            .init(
                flag: "--model", aliases: ["-m"], label: "Model", kind: .string,
                defaultValue: "speech-tts-qwen3-nano", group: Group.modelAndAdapters, tier: .essential
            ),
            .init(
                flag: "--voice", aliases: ["-v"], label: "Voice", kind: .string,
                defaultValue: "A calm female voice with clear pronunciation", group: Group.prompt, tier: .essential
            ).scoped(SpeechSynthesizeFamily.only(.style, .customVoice, ignoredBy: [.clone])),
            // The published CustomVoice checkpoint's speakers (`talker_config.spk_id`). The gate
            // compares no value: the command checks the name against the checkpoint that runs, so
            // a local CustomVoice folder can name its own.
            .init(
                flag: "--speaker", label: "Speaker", kind: .choice,
                choices: ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"],
                group: Group.prompt, tier: .essential, choiceSpellings: .caseInsensitive
            ).scoped(SpeechSynthesizeFamily.only(.customVoice, ignoredBy: [.style, .clone])),
            .init(
                flag: "--mode", label: "Mode", kind: .choice, choices: ["style", "clone"],
                defaultValue: "style", group: Group.inputs, tier: .standard, choiceSpellings: .exact
            ),
            .init(flag: "--profile", label: "Profile", kind: .string, group: Group.inputs, tier: .standard)
                .scoped(SpeechSynthesizeFamily.only(.clone, ignoredBy: [.style])),
            .init(flag: "--ref-audio", label: "Reference audio", kind: .file, group: Group.inputs, tier: .standard)
                .scoped(SpeechSynthesizeFamily.only(.clone, ignoredBy: [.style])),
            .init(
                flag: "--ref-text", label: "Reference text", kind: .string,
                group: Group.inputs, tier: .standard, dependsOn: "--ref-audio"
            ).scoped(SpeechSynthesizeFamily.only(.clone, ignoredBy: [.style])),
            .init(flag: "--language", label: "Language", kind: .string, defaultValue: "auto", group: Group.prompt, tier: .standard),
            .init(
                flag: "--save-profile", label: "Save profile", kind: .string,
                group: Group.inputs, tier: .expert, dependsOn: "--ref-audio"
            ).scoped(SpeechSynthesizeFamily.only(.clone, ignoredBy: [.style])),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                defaultValue: "0.6", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(flag: "--stream", label: "Stream", kind: .boolean, group: Group.output, tier: .expert),
            .init(
                flag: "--stream-chunk-tokens", label: "Chunk tokens", kind: .integer,
                defaultValue: "25", group: Group.output, tier: .expert, range: .init(min: 1, max: 500, step: 1), dependsOn: "--stream"
            ),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output"),
        routing: speechSynthesizeRouting
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
            .init(flag: "--timestamps", label: "Include timestamps", kind: .boolean)
                .scoped(SpeechTranscribeFamily.only(.parakeet, ignoredBy: [.qwen3ASR])),
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .standard),
            .init(
                flag: "--backend", label: "Backend", kind: .choice, choices: ["auto", "parakeet", "qwen"],
                defaultValue: "auto", group: Group.modelAndAdapters, tier: .essential, choiceSpellings: .exact
            ).scoped(SpeechTranscribeFamily.rule(.qwen3ASR, values: ["auto", "qwen"], severity: .warning)),
            .init(
                flag: "--provider", label: "Parakeet provider", kind: .choice, choices: ["mlx", "coreml"],
                defaultValue: "mlx", group: Group.modelAndAdapters, tier: .standard, choiceSpellings: .exact
            ).scoped(
                // Qwen3-ASR always runs MLX: the default passes with no effect, Core ML is refused.
                SpeechTranscribeFamily.only(.parakeet, ignoredBy: [.qwen3ASR]), .rule(.qwen3ASR, values: ["mlx"])
            ),
            .init(
                flag: "--coreml-encoder", label: "Core ML artifact", kind: .directory,
                group: Group.modelAndAdapters, tier: .expert, dependsOn: "--provider"
            ).scoped(SpeechTranscribeFamily.only(.parakeet)),
            .init(
                flag: "--task", label: "Task", kind: .choice, choices: ["transcribe", "translate"],
                defaultValue: "transcribe", group: Group.prompt, tier: .essential, choiceSpellings: .exact
            ),
            // Translation always targets English, whatever language the audio is in.
            .init(
                flag: "--language", label: "Language", kind: .string, group: Group.prompt, tier: .standard,
                overriddenBy: [.init(flag: "--task", values: ["translate"])]
            ),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "448", group: Group.sampling, tier: .standard, range: .init(min: 1, max: 8_192, step: 1)
            ).scoped(SpeechTranscribeFamily.only(.qwen3ASR, ignoredBy: [.parakeet])),
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
            .init(flag: "--no-timestamps", label: "No timestamps", kind: .boolean, group: Group.output, tier: .standard)
                .scoped(SpeechTranscribeFamily.only(.parakeet, ignoredBy: [.qwen3ASR])),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            receiptOption
        ],
        output: .init(kind: .text, fileExtension: "txt", flag: "--output", optional: true),
        routing: speechTranscribeRouting
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
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(
                flag: "--format", aliases: ["-f"], label: "Format", kind: .choice, choices: ["json", "rttm"],
                defaultValue: "json", tier: .essential
            ),
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file, group: Group.output, tier: .standard),
            .init(
                flag: "--threshold", label: "Activity threshold", kind: .number,
                defaultValue: "0.5", tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--min-duration", label: "Minimum segment", kind: .number,
                defaultValue: "0.25", tier: .standard, range: .init(min: 0, max: 5, step: 0.05)
            ),
            .init(
                flag: "--merge-gap", label: "Merge gap", kind: .number,
                defaultValue: "0.25", tier: .standard, range: .init(min: 0, max: 5, step: 0.05)
            ),
            .init(
                flag: "--latency",
                label: "Input buffer",
                kind: .choice,
                choices: ["offline", "1.04", "0.64", "0.32"],
                defaultValue: "offline",
                tier: .essential
            ).scoped(
                // Sortformer runs offline: the default passes with no effect, a buffer is refused.
                SpeechDiarizeFamily.only(.nemotron3, ignoredBy: [.sortformer]), .rule(.sortformer, values: ["offline"])
            ),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert)
        ],
        output: .init(kind: .text, flag: "--output", optional: true),
        routing: speechDiarizeRouting
    )

    public static let speechDiarizeLive = MereRunCommandCapability(
        id: "speech.diarize-live",
        command: ["speech", "diarize-live"],
        title: "Live speaker diarization",
        summary: "Stream Nemotron 3 speaker activity from a microphone or 16 kHz PCM stdin.",
        options: [
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--device", label: "Input device", kind: .string, group: Group.inputs, tier: .standard),
            .init(flag: "--list-devices", label: "List devices", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--stdin", label: "PCM stdin", kind: .boolean, group: Group.run, tier: .expert),
            .init(
                flag: "--latency", label: "Input buffer", kind: .choice,
                choices: ["1.04", "0.64", "0.32"], defaultValue: "1.04", tier: .essential
            ),
            .init(
                flag: "--threshold", label: "Activity threshold", kind: .number,
                defaultValue: "0.5", tier: .essential, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert)
        ],
        output: .init(kind: .text),
        routing: speechDiarizeLiveRouting
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
            .init(flag: "--name", label: "Name", kind: .string, required: true, group: Group.prompt, tier: .essential),
            .init(flag: "--audio", label: "Reference audio", kind: .file, required: true, group: Group.inputs, tier: .essential),
            .init(flag: "--text", label: "Transcript", kind: .string, group: Group.prompt, tier: .standard),
            .init(flag: "--language", label: "Language", kind: .string, defaultValue: "auto", tier: .standard),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert)
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
            .init(flag: "--device", label: "Input device", kind: .string, group: Group.inputs, tier: .standard),
            .init(flag: "--list-devices", label: "List devices", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--language", label: "Language", kind: .string, tier: .essential),
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(
                flag: "--decode-ms", label: "Decode window (ms)", kind: .integer,
                defaultValue: "2000", tier: .standard, range: .init(min: 1, max: 10_000, step: 100)
            ),
            .init(
                flag: "--silence-ms", label: "Silence to commit (ms)", kind: .integer,
                defaultValue: "900", tier: .standard, range: .init(min: 1, max: 5_000, step: 100)
            ),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--jsonl", label: "JSONL", kind: .boolean, group: Group.run, tier: .expert)
        ],
        output: .init(kind: .text),
        routing: speechListenRouting
    )
}
