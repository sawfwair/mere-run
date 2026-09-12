import Foundation

extension MereRunCapabilityCatalog {
    public static let musicGenerate = MereRunCommandCapability(
        id: "music.generate",
        command: ["music", "generate"],
        title: "Generate and edit music",
        summary: "Create, cover, repaint, flow-edit, rank, stem, and export production music.",
        arguments: [
            .init(name: "caption", label: "Music prompt", kind: .string, required: true)
        ],
        options: [
            .init(
                flag: "--compose", label: "Compose with a chat model", kind: .boolean, group: Group.prompt, tier: .expert
            ),
            .init(
                // Defaults to the hardware-aware chat model the CLI picks for the
                // current machine, so the catalog advertises no fixed value.
                flag: "--composer-model", label: "Composer model", kind: .string,
                group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--composer-model-root", label: "Composer model root", kind: .directory,
                group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--require-composer-installed", label: "Require installed composer", kind: .boolean,
                group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--composition-output", label: "Composition output", kind: .file, group: Group.output, tier: .expert
            ),
            .init(
                flag: "--lyrics-preflight", label: "Lyric duration checks", kind: .choice,
                choices: ["off", "warn", "strict"], defaultValue: "warn", group: Group.prompt, tier: .expert
            ),
            .init(
                flag: "--minimum-duration", label: "Minimum duration", kind: .number, group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--min-frames", label: "Minimum acoustic frames", kind: .integer, group: Group.sampling,
                tier: .expert
            ),
            .init(
                flag: "--max-frames", label: "Maximum acoustic frames", kind: .integer, group: Group.sampling,
                tier: .expert
            ),
            .init(
                flag: "--sample-rate", label: "Sample rate", kind: .integer, group: Group.output, tier: .expert
            ),
            .init(
                flag: "--memory-mode", label: "Memory mode", kind: .choice, choices: ["staged", "resident"],
                group: Group.run, tier: .expert
            ),
            .init(
                flag: "--performance-mode", label: "Performance mode", kind: .choice,
                choices: ["reference", "optimized", "q8", "q4", "q8-lm", "q4-lm"], group: Group.run, tier: .expert
            ),
            .init(
                flag: "--sampling-tier", label: "Sampling tier", kind: .choice, choices: ["quality", "fast", "draft"],
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--flow-strategy", label: "Flow strategy", kind: .choice, choices: ["sequential", "overlap-average"],
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--flow-solver", label: "Flow solver", kind: .choice, choices: ["euler", "ab2"],
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--ar-cfg-frames", label: "Autoregressive CFG frames", kind: .integer, group: Group.sampling,
                tier: .expert
            ),
            .init(
                flag: "--flow-cfg-end", label: "Flow CFG cutoff", kind: .number, group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--seed-strategy", label: "Seed strategy", kind: .choice, choices: ["legacy", "stage-separated-v1"],
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--profile-output", label: "Profile output", kind: .file, group: Group.output, tier: .expert
            ),
            .init(
                flag: "--no-lm-caption-rewrite", label: "Preserve input caption", kind: .boolean, group: Group.prompt,
                tier: .expert
            ),
            .init(flag: "--lyrics", label: "Lyrics", kind: .string, group: Group.prompt, tier: .standard),
            .init(flag: "--lyrics-file", label: "Lyrics file", kind: .file, group: Group.prompt, tier: .expert),
            .init(flag: "--instrumental", label: "Instrumental", kind: .boolean, group: Group.prompt, tier: .standard),
            .init(flag: "--lrc-file", label: "LRC file", kind: .file, group: Group.prompt, tier: .expert),
            .init(flag: "--lrc-output", label: "LRC output", kind: .file, group: Group.output, tier: .expert),
            .init(flag: "--output", label: "Audio output", kind: .file, group: Group.output, tier: .standard),
            .init(
                flag: "--export-format", label: "Audio format", kind: .choice, choices: ["pcm16", "pcm24", "float32"],
                defaultValue: "pcm24", group: Group.output, tier: .expert
            ),
            .init(
                flag: "--normalize", label: "Normalization", kind: .choice, choices: ["none", "peak"],
                defaultValue: "peak", group: Group.output, tier: .expert
            ),
            .init(
                flag: "--target-peak-db", label: "Peak target", kind: .number,
                defaultValue: "-1.0", group: Group.output, tier: .expert, range: .init(min: -24, max: 0, step: 0.5)
            ),
            .init(
                flag: "--fade-in-ms", label: "Fade in", kind: .number,
                defaultValue: "5.0", group: Group.output, tier: .expert, range: .init(min: 0, max: 5_000, step: 1)
            ),
            .init(
                flag: "--fade-out-ms", label: "Fade out", kind: .number,
                defaultValue: "20.0", group: Group.output, tier: .expert, range: .init(min: 0, max: 5_000, step: 1)
            ),
            .init(flag: "--no-dither", label: "Disable dither", kind: .boolean, group: Group.output, tier: .expert),
            .init(flag: "--recipe-output", label: "Recipe output", kind: .file, group: Group.output, tier: .expert),
            .init(flag: "--no-recipe", label: "Disable recipe", kind: .boolean, group: Group.output, tier: .expert),
            .init(flag: "--daw-bundle", label: "DAW bundle", kind: .directory, group: Group.output, tier: .expert),
            .init(flag: "--stems", label: "Stems", kind: .string, group: Group.output, tier: .expert),
            .init(flag: "--adapter", label: "Adapter", kind: .file, repeatable: true, group: Group.modelAndAdapters, tier: .standard),
            .init(
                flag: "--adapter-kind", label: "Adapter kind", kind: .choice, choices: ["auto", "lora", "lokr"],
                defaultValue: "auto", group: Group.modelAndAdapters, tier: .expert, dependsOn: "--adapter"
            ),
            .init(
                flag: "--adapter-scale", label: "Adapter scale", kind: .number, repeatable: true,
                group: Group.modelAndAdapters, tier: .standard, range: .init(min: 0, max: 2, step: 0.05), dependsOn: "--adapter"
            ),
            .init(
                flag: "--model", label: "Model", kind: .string,
                defaultValue: "music-acestep", group: Group.modelAndAdapters, tier: .essential
            ),
            .init(flag: "--checkpoints-root", label: "Checkpoints root", kind: .directory, group: Group.modelAndAdapters, tier: .expert),
            .init(
                flag: "--decoder-subdirectory", label: "Decoder", kind: .string,
                defaultValue: "acestep-v15-turbo", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--vae-subdirectory", label: "VAE", kind: .string,
                defaultValue: "vae", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(flag: "--lm-subdirectory", label: "Language model", kind: .string, group: Group.modelAndAdapters, tier: .expert),
            .init(flag: "--lm-model", label: "LM model", kind: .string, group: Group.modelAndAdapters, tier: .expert),
            .init(flag: "--text-subdirectory", label: "Text encoder", kind: .string, group: Group.modelAndAdapters, tier: .expert),
            .init(flag: "--use-lm", label: "Use LM planning", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(flag: "--no-lm", label: "Disable LM planning", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(
                flag: "--analyze-source-audio", label: "Analyze source", kind: .boolean,
                group: Group.inputs, tier: .expert, dependsOn: "--source-audio"
            ),
            .init(
                flag: "--duration", label: "Duration", kind: .number,
                group: Group.sampling, tier: .essential, range: .init(min: 1, max: 600, step: 1)
            ),
            .init(
                flag: "--quality", label: "Quality", kind: .choice, choices: ["draft", "song", "final", "edit"],
                group: Group.sampling, tier: .essential
            ),
            .init(
                flag: "--steps", label: "Steps", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 200, step: 1)
            ),
            .init(
                flag: "--shift", label: "Scheduler shift", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            ),
            .init(
                flag: "--infer-method", label: "Inference method", kind: .choice, choices: ["ode", "sde"],
                group: Group.sampling, tier: .expert
            ),
            .init(flag: "--sampler", label: "Sampler", kind: .choice, choices: ["euler", "heun"], group: Group.sampling, tier: .expert),
            .init(
                flag: "--guidance-scale", label: "Guidance scale", kind: .number,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 20, step: 0.1)
            ),
            .init(
                flag: "--guidance-mode", label: "Guidance mode", kind: .choice, choices: ["apg", "adg", "cfg"],
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--cfg-interval-start", label: "CFG start", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--cfg-interval-end", label: "CFG end", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--velocity-norm-threshold", label: "Velocity clamp", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, step: 0.1)
            ),
            .init(
                flag: "--velocity-ema-factor", label: "Velocity EMA", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .essential, range: .init(min: 0, step: 1)),
            .init(
                flag: "--candidates", label: "Candidate count", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 16, step: 1)
            ),
            .init(
                flag: "--keep-candidates", label: "Keep candidates", kind: .boolean,
                group: Group.output, tier: .expert, dependsOn: "--candidates"
            ),
            .init(
                flag: "--audio-cover-strength", label: "Cover strength", kind: .number,
                defaultValue: "1.0", group: Group.inputs, tier: .standard, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--source-audio"
            ),
            .init(
                flag: "--cover-noise-strength", label: "Cover noise", kind: .number,
                defaultValue: "0.0", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--source-audio"
            ),
            .init(flag: "--retake-seed", label: "Retake seed", kind: .integer, group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)),
            .init(
                flag: "--retake-variance", label: "Retake variance", kind: .number,
                defaultValue: "0.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--retake-seed"
            ),
            .init(flag: "--vocal-language", label: "Vocal language", kind: .string, defaultValue: "en", group: Group.prompt, tier: .standard),
            .init(
                flag: "--instruction", label: "Instruction", kind: .string,
                defaultValue: "Fill the audio semantic mask based on the given conditions:", group: Group.prompt, tier: .expert
            ),
            .init(
                flag: "--task-type",
                label: "Task",
                kind: .choice,
                choices: ["text2music", "repaint", "cover", "cover-nofsq", "extract", "lego", "complete"],
                defaultValue: "text2music", group: Group.inputs, tier: .standard
            ),
            .init(flag: "--source-audio", label: "Source audio", kind: .file, group: Group.inputs, tier: .standard),
            .init(
                flag: "--reference-audio", label: "Reference audio", kind: .file, repeatable: true,
                group: Group.inputs, tier: .standard
            ),
            .init(flag: "--track-name", label: "Track name", kind: .string, group: Group.inputs, tier: .expert),
            .init(flag: "--complete-track-classes", label: "Track classes", kind: .string, group: Group.inputs, tier: .expert),
            .init(flag: "--non-cover", label: "No FSQ cover", kind: .boolean, group: Group.inputs, tier: .expert, dependsOn: "--source-audio"),
            .init(
                flag: "--repaint-start", label: "Repaint start", kind: .number,
                defaultValue: "0.0", group: Group.inputs, tier: .standard, range: .init(min: 0, step: 0.1), dependsOn: "--source-audio"
            ),
            .init(
                flag: "--repaint-end", label: "Repaint end", kind: .number,
                defaultValue: "-1.0", group: Group.inputs, tier: .standard, range: .init(min: -1, step: 0.1), dependsOn: "--source-audio"
            ),
            .init(
                flag: "--chunk-mask-mode", label: "Chunk mask", kind: .choice, choices: ["auto", "explicit"],
                defaultValue: "auto", group: Group.inputs, tier: .expert, dependsOn: "--source-audio"
            ),
            .init(
                flag: "--repaint-mode", label: "Repaint mode", kind: .choice, choices: ["conservative", "balanced", "aggressive"],
                defaultValue: "balanced", group: Group.inputs, tier: .expert, dependsOn: "--source-audio"
            ),
            .init(
                flag: "--repaint-strength", label: "Repaint strength", kind: .number,
                defaultValue: "0.5", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--source-audio"
            ),
            .init(flag: "--flow-edit", label: "Flow edit", kind: .boolean, group: Group.inputs, tier: .standard, dependsOn: "--source-audio"),
            .init(flag: "--source-caption", label: "Source caption", kind: .string, group: Group.inputs, tier: .standard, dependsOn: "--flow-edit"),
            .init(flag: "--source-lyrics", label: "Source lyrics", kind: .string, group: Group.inputs, tier: .expert, dependsOn: "--flow-edit"),
            .init(
                flag: "--flow-edit-n-min", label: "Flow start", kind: .number,
                defaultValue: "0.0", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05), dependsOn: "--flow-edit"
            ),
            .init(
                flag: "--flow-edit-n-max", label: "Flow end", kind: .number,
                defaultValue: "1.0", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05), dependsOn: "--flow-edit"
            ),
            .init(
                flag: "--flow-edit-n-average", label: "Flow samples", kind: .integer,
                defaultValue: "1", group: Group.inputs, tier: .expert, range: .init(min: 1, max: 16, step: 1), dependsOn: "--flow-edit"
            ),
            .init(flag: "--bpm", label: "BPM", kind: .integer, group: Group.prompt, tier: .standard, range: .init(min: 20, max: 300, step: 1)),
            .init(flag: "--keyscale", label: "Key", kind: .string, group: Group.prompt, tier: .standard),
            .init(flag: "--timesignature", label: "Time signature", kind: .string, group: Group.prompt, tier: .standard),
            .init(
                flag: "--lm-temperature", label: "LM temperature", kind: .number,
                defaultValue: "0.85", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--lm-top-k", label: "LM top-k", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1_000, step: 1)
            ),
            .init(
                flag: "--lm-top-p", label: "LM top-p", kind: .number,
                defaultValue: "0.9", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--lm-repetition-penalty", label: "LM repetition penalty", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0.5, max: 2, step: 0.05)
            ),
            .init(
                flag: "--lm-cfg-scale", label: "LM CFG scale", kind: .number,
                defaultValue: "2.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            ),
            .init(
                flag: "--lm-negative-prompt", label: "LM negative prompt", kind: .string,
                defaultValue: "NO USER INPUT", group: Group.sampling, tier: .expert
            ),
            .init(flag: "--metadata-duration", label: "Metadata duration", kind: .string, group: Group.prompt, tier: .expert),
            .init(flag: "--metadata-language", label: "Metadata language", kind: .string, group: Group.prompt, tier: .expert),
            .init(flag: "--no-tiled-vae", label: "Disable tiled VAE", kind: .boolean, group: Group.run, tier: .expert),
            .init(
                flag: "--vae-chunk-size", label: "VAE chunk", kind: .integer,
                defaultValue: "512", group: Group.run, tier: .expert, range: .init(min: 64, max: 4_096, step: 64)
            ),
            .init(
                flag: "--vae-overlap", label: "VAE overlap", kind: .integer,
                defaultValue: "64", group: Group.run, tier: .expert, range: .init(min: 0, max: 1_024, step: 8)
            ),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            .init(
                flag: "--temperature", label: "RT2 temperature", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--style-conditioning", label: "RT2 style", kind: .choice, choices: ["streaming", "full"],
                defaultValue: "streaming", group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--top-k", label: "RT2 top-k", kind: .integer,
                defaultValue: "100", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1_000, step: 1)
            ),
            .init(
                flag: "--cfg-musiccoca", label: "MusicCoCa CFG", kind: .number,
                defaultValue: "3.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--cfg-notes", label: "Notes CFG", kind: .number,
                defaultValue: "5.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--cfg-drums", label: "Drums CFG", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(flag: "--drumless", label: "Drumless", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(
                flag: "--unmask-width", label: "Unmask width", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)
            ),
            .init(
                flag: "--seed-rotation", label: "Seed rotation", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)
            ),
            .init(flag: "--prefill-silence", label: "Prefill silence", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(
                flag: "--prefill-duration", label: "Prefill duration", kind: .number,
                defaultValue: "1.64", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 30, step: 0.01),
                dependsOn: "--prefill-silence"
            ),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let musicAnalyze = MereRunCommandCapability(
        id: "music.analyze",
        command: ["music", "analyze"],
        title: "Analyze music",
        summary: "Extract structured music metadata and optional ACE-Step audio codes.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--checkpoints-root", label: "Checkpoints root", kind: .directory),
            .init(flag: "--decoder-subdirectory", label: "Decoder", kind: .string),
            .init(flag: "--vae-subdirectory", label: "VAE", kind: .string),
            .init(flag: "--lm-subdirectory", label: "Language model", kind: .string),
            .init(flag: "--lm-model", label: "LM model", kind: .string),
            .init(flag: "--duration", label: "Duration", kind: .number),
            .init(flag: "--max-new-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--lm-temperature", label: "LM temperature", kind: .number),
            .init(flag: "--lm-top-k", label: "LM top-k", kind: .integer),
            .init(flag: "--lm-top-p", label: "LM top-p", kind: .number),
            .init(flag: "--include-raw-lm", label: "Raw LM", kind: .boolean),
            .init(flag: "--include-audio-codes", label: "Audio codes", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let musicTranscribe = MereRunCommandCapability(
        id: "music.transcribe",
        command: ["music", "transcribe"],
        title: "Transcribe music",
        summary: "Turn a full mix into separated MIDI or structured events with musical context.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: false)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-path", label: "Model path", kind: .directory),
            .init(flag: "--variant", label: "Variant", kind: .choice, choices: ["small", "medium", "large"]),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--format", label: "Format", kind: .choice, choices: ["midi", "json", "jsonl"]),
            .init(flag: "--instruments", label: "Instruments", kind: .string),
            .init(flag: "--list-instruments", label: "List instruments", kind: .boolean),
            .init(flag: "--sampling", label: "Sampling", kind: .boolean),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--max-tokens-per-chunk", label: "Tokens per chunk", kind: .integer),
            .init(flag: "--strict-eos", label: "Strict EOS", kind: .boolean),
            .init(flag: "--beam-size", label: "Beam size", kind: .integer),
            .init(flag: "--chunk-batch-size", label: "Chunk batch", kind: .integer),
            .init(flag: "--dtype", label: "Compute type", kind: .choice, choices: ["bfloat16", "float16", "float32"]),
            .init(flag: "--no-musical-context", label: "Disable context", kind: .boolean),
            .init(flag: "--context-output", label: "Context output", kind: .file),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, flag: "--output")
    )

    public static let musicSeparate = MereRunCommandCapability(
        id: "music.separate",
        command: ["music", "separate"],
        title: "Separate or restore music",
        summary: "Create stems, remove reverb, or denoise audio with native RoFormer models.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-path", label: "Model path", kind: .directory),
            .init(flag: "--output-dir", label: "Output directory", kind: .directory),
            .init(flag: "--overlap", label: "Overlap", kind: .integer),
            .init(
                flag: "--dtype",
                label: "Compute type",
                kind: .choice,
                choices: ["float16", "float32"]
            ),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output-dir")
    )

    public static let musicRealtime = MereRunCommandCapability(
        id: "music.realtime",
        command: ["music", "realtime"],
        title: "Realtime music",
        summary: "Run live Magenta RT2 generation with text, interactive, and MIDI steering.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: false)
        ],
        options: [
            .init(flag: "--play", label: "Play audio", kind: .boolean),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--duration", label: "Duration", kind: .number),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--no-play", label: "Disable playback", kind: .boolean),
            .init(flag: "--style-conditioning", label: "Style", kind: .choice, choices: ["streaming", "full"]),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-k", label: "Top-k", kind: .integer),
            .init(flag: "--cfg-musiccoca", label: "MusicCoCa CFG", kind: .number),
            .init(flag: "--cfg-notes", label: "Notes CFG", kind: .number),
            .init(flag: "--cfg-drums", label: "Drums CFG", kind: .number),
            .init(flag: "--drumless", label: "Drumless", kind: .boolean),
            .init(flag: "--unmask-width", label: "Unmask width", kind: .integer),
            .init(flag: "--seed-rotation", label: "Seed rotation", kind: .integer),
            .init(flag: "--prefill-silence", label: "Prefill silence", kind: .boolean),
            .init(flag: "--prefill-duration", label: "Prefill duration", kind: .number),
            .init(flag: "--interactive", label: "Interactive", kind: .boolean),
            .init(flag: "--list-midi-inputs", label: "List MIDI", kind: .boolean),
            .init(flag: "--midi-monitor", label: "MIDI monitor", kind: .boolean),
            .init(flag: "--midi-log-events", label: "Log MIDI", kind: .boolean),
            .init(flag: "--midi-log-raw", label: "Log raw MIDI", kind: .boolean),
            .init(flag: "--midi-input", label: "MIDI input", kind: .string),
            .init(flag: "--midi-channel", label: "MIDI channel", kind: .string),
            .init(flag: "--midi-note-offset", label: "MIDI transpose", kind: .integer),
            .init(flag: "--midi-cc", label: "MIDI CC map", kind: .string, repeatable: true),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .service, fileExtension: "wav", flag: "--output", optional: true)
    )

    public static let musicTrainAdapter = MereRunCommandCapability(
        id: "music.train-adapter",
        command: ["music", "train-adapter"],
        title: "Train music adapter",
        summary: "Train a native ACE-Step LoRA or LoKr adapter.",
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--dataset", label: "Dataset", kind: .file, required: true),
            .init(flag: "--output", label: "Output", kind: .file, required: true),
            .init(flag: "--kind", label: "Adapter kind", kind: .choice, choices: ["auto", "lora", "lokr"]),
            .init(flag: "--rank", label: "Rank", kind: .integer),
            .init(flag: "--alpha", label: "Alpha", kind: .number),
            .init(flag: "--factor", label: "LoKr factor", kind: .integer),
            .init(flag: "--steps", label: "Steps", kind: .integer),
            .init(flag: "--learning-rate", label: "Learning rate", kind: .number),
            .init(flag: "--weight-decay", label: "Weight decay", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--max-duration", label: "Max duration", kind: .number),
            .init(flag: "--checkpoints-root", label: "Checkpoints root", kind: .directory),
            .init(flag: "--decoder-subdirectory", label: "Decoder", kind: .string),
            .init(flag: "--vae-subdirectory", label: "VAE", kind: .string),
            .init(flag: "--text-subdirectory", label: "Text encoder", kind: .string),
            .init(flag: "--log-every", label: "Progress interval", kind: .integer)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let musicServe = MereRunCommandCapability(
        id: "music.serve",
        command: ["music", "serve"],
        title: "Serve resident music",
        summary: "Keep ACE-Step, its language model, and adapter stack warm behind a local API.",
        options: [
            .init(
                flag: "--memory-mode", label: "Memory mode", kind: .choice, choices: ["staged", "resident"],
                group: Group.run, tier: .expert
            ),
            .init(
                flag: "--performance-mode", label: "Performance mode", kind: .choice,
                choices: ["reference", "optimized", "q8", "q4", "q8-lm", "q4-lm"], group: Group.run, tier: .expert
            ),
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--checkpoints-root", label: "Checkpoints root", kind: .directory),
            .init(flag: "--decoder-subdirectory", label: "Decoder", kind: .string),
            .init(flag: "--vae-subdirectory", label: "VAE", kind: .string),
            .init(flag: "--lm-subdirectory", label: "Language model", kind: .string),
            .init(flag: "--lm-model", label: "LM model", kind: .string),
            .init(flag: "--text-subdirectory", label: "Text encoder", kind: .string),
            .init(flag: "--adapter", label: "Adapter", kind: .file, repeatable: true),
            .init(flag: "--adapter-kind", label: "Adapter kind", kind: .choice, choices: ["auto", "lora", "lokr"]),
            .init(flag: "--adapter-scale", label: "Adapter scale", kind: .number, repeatable: true),
            .init(flag: "--api-key", label: "API key", kind: .string)
        ],
        output: .init(kind: .service)
    )
}
