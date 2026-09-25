import Foundation

private typealias G = MereRunCapabilityCatalog.MusicGenerateFamily
private typealias P = MereRunCapabilityCatalog.MusicSeparateFamily
private typealias S = MereRunCapabilityCatalog.MusicServeFamily

extension MereRunCapabilityCatalog {
    /// Option scopes follow `MusicGenerateCommand`: MiniMax Music 3 and YuE2 refuse an ACE-Step or
    /// Magenta option unless it carries that runtime's own default (`onlyDefaultFor`), while
    /// ACE-Step never reads Magenta's controls and Magenta never reads most of ACE-Step's.
    public static let musicGenerate = MereRunCommandCapability(
        id: "music.generate",
        command: ["music", "generate"],
        title: "Generate and edit music",
        summary: "Create, cover, repaint, flow-edit, rank, stem, and export production music.",
        arguments: [
            .init(name: "caption", label: "Music prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--score-mode", label: "YuE2 score planning", kind: .choice,
                  choices: ["full", "melody", "off"], group: Group.prompt, tier: .expert)
                .scoped(G.only(.yue2)),
            .init(flag: "--abc-file", label: "YuE2 input score", kind: .file, group: Group.inputs, tier: .expert)
                .scoped(G.only(.yue2)),
            .init(flag: "--abc-output", label: "YuE2 score output", kind: .file, group: Group.output, tier: .expert)
                .scoped(G.only(.yue2)),
            .init(flag: "--abc-max-tokens", label: "YuE2 score token budget", kind: .integer, group: Group.sampling, tier: .expert)
                .scoped(G.only(.yue2)),
            .init(flag: "--semantic-temperature", label: "YuE2 semantic temperature", kind: .number, group: Group.sampling, tier: .expert)
                .scoped(G.only(.yue2)),
            .init(flag: "--semantic-top-p", label: "YuE2 semantic top-p", kind: .number, group: Group.sampling, tier: .expert)
                .scoped(G.only(.yue2)),
            .init(flag: "--semantic-top-k", label: "YuE2 semantic top-k", kind: .integer, group: Group.sampling, tier: .expert)
                .scoped(G.only(.yue2)),
            .init(flag: "--semantic-repetition-penalty", label: "YuE2 semantic repetition penalty", kind: .number,
                  group: Group.sampling, tier: .expert)
                .scoped(G.only(.yue2)),
            .init(
                flag: "--compose", label: "Compose with a chat model", kind: .boolean, group: Group.prompt, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                // Defaults to the hardware-aware chat model the CLI picks for the
                // current machine, so the catalog advertises no fixed value.
                flag: "--composer-model", label: "Composer model", kind: .string,
                group: Group.modelAndAdapters, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3, ignoredBy: G.aceStep + [.yue2, .magentaRT2])),
            .init(
                flag: "--composer-model-root", label: "Composer model root", kind: .directory,
                group: Group.modelAndAdapters, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                flag: "--require-composer-installed", label: "Require installed composer", kind: .boolean,
                group: Group.modelAndAdapters, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                flag: "--composition-output", label: "Composition output", kind: .file, group: Group.output, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                flag: "--lyrics-preflight", label: "Lyric duration checks", kind: .choice,
                choices: ["off", "warn", "strict"], defaultValue: "warn", group: Group.prompt, tier: .expert
            )
                .scoped(
                    G.only(.miniMaxMusic3, ignoredBy: G.aceStep + [.yue2, .magentaRT2]),
                    onlyDefaultFor: G.aceStep + [.yue2, .magentaRT2]
                ),
            .init(
                flag: "--minimum-duration", label: "Minimum duration", kind: .number, group: Group.sampling, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3, .yue2)),
            .init(
                flag: "--min-frames", label: "Minimum acoustic frames", kind: .integer, group: Group.sampling,
                tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3, .yue2)),
            .init(
                flag: "--max-frames", label: "Maximum acoustic frames", kind: .integer, group: Group.sampling,
                tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3, .yue2)),
            .init(
                flag: "--sample-rate", label: "Sample rate", kind: .integer, group: Group.output, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3), .rule(.miniMaxMusic3, values: ["32000", "44100"])),
            .init(
                flag: "--memory-mode", label: "Memory mode", kind: .choice, choices: ["staged", "resident"],
                group: Group.run, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                flag: "--performance-mode", label: "Performance mode", kind: .choice,
                choices: ["reference", "optimized", "q8", "q4", "q8-lm", "q4-lm"], group: Group.run, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                flag: "--sampling-tier", label: "Sampling tier", kind: .choice, choices: ["quality", "fast", "draft"],
                group: Group.sampling, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                flag: "--flow-strategy", label: "Flow strategy", kind: .choice, choices: ["sequential", "overlap-average"],
                group: Group.sampling, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                flag: "--flow-solver", label: "Flow solver", kind: .choice, choices: ["euler", "ab2"],
                group: Group.sampling, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                flag: "--ar-cfg-frames", label: "Autoregressive CFG frames", kind: .integer, group: Group.sampling,
                tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                flag: "--flow-cfg-end", label: "Flow CFG cutoff", kind: .number, group: Group.sampling, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                flag: "--seed-strategy", label: "Seed strategy", kind: .choice, choices: ["legacy", "stage-separated-v1"],
                group: Group.sampling, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                flag: "--profile-output", label: "Profile output", kind: .file, group: Group.output, tier: .expert
            )
                .scoped(G.only(.miniMaxMusic3)),
            .init(
                flag: "--no-lm-caption-rewrite", label: "Preserve input caption", kind: .boolean, group: Group.prompt,
                tier: .expert
            )
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(flag: "--lyrics", label: "Lyrics", kind: .string, group: Group.prompt, tier: .standard)
                .scoped(G.except(.magentaRT2)),
            .init(flag: "--lyrics-file", label: "Lyrics file", kind: .file, group: Group.prompt, tier: .expert)
                .scoped(G.except(.magentaRT2)),
            .init(flag: "--instrumental", label: "Instrumental", kind: .boolean, group: Group.prompt, tier: .standard)
                .scoped(G.except(ignoredBy: [.yue2, .magentaRT2])),
            .init(flag: "--lrc-file", label: "LRC file", kind: .file, group: Group.prompt, tier: .expert)
                .scoped(G.except(.yue2, ignoredBy: [.magentaRT2])),
            .init(flag: "--lrc-output", label: "LRC output", kind: .file, group: Group.output, tier: .expert)
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(flag: "--output", aliases: ["-o"], label: "Audio output", kind: .file, group: Group.output, tier: .standard),
            .init(
                flag: "--export-format", label: "Audio format", kind: .choice, choices: ["pcm16", "pcm24", "float32"],
                defaultValue: "pcm24", group: Group.output, tier: .expert
            )
                .scoped(G.except(ignoredBy: [.magentaRT2])),
            .init(
                flag: "--normalize", label: "Normalization", kind: .choice, choices: ["none", "peak"],
                defaultValue: "peak", group: Group.output, tier: .expert
            )
                .scoped(G.except(ignoredBy: [.magentaRT2])),
            .init(
                flag: "--target-peak-db", label: "Peak target", kind: .number,
                defaultValue: "-1.0", group: Group.output, tier: .expert, range: .init(min: -24, max: 0, step: 0.5)
            )
                .scoped(G.except(ignoredBy: [.magentaRT2])),
            .init(
                flag: "--fade-in-ms", label: "Fade in", kind: .number,
                defaultValue: "5.0", group: Group.output, tier: .expert, range: .init(min: 0, max: 5_000, step: 1)
            )
                .scoped(G.except(ignoredBy: [.magentaRT2])),
            .init(
                flag: "--fade-out-ms", label: "Fade out", kind: .number,
                defaultValue: "20.0", group: Group.output, tier: .expert, range: .init(min: 0, max: 5_000, step: 1)
            )
                .scoped(G.except(ignoredBy: [.magentaRT2])),
            .init(flag: "--no-dither", label: "Disable dither", kind: .boolean, group: Group.output, tier: .expert)
                .scoped(G.except(ignoredBy: [.magentaRT2])),
            .init(flag: "--recipe-output", label: "Recipe output", kind: .file, group: Group.output, tier: .expert)
                .scoped(G.except(ignoredBy: [.magentaRT2])),
            .init(flag: "--no-recipe", label: "Disable recipe", kind: .boolean, group: Group.output, tier: .expert)
                .scoped(G.except(ignoredBy: [.magentaRT2])),
            .init(flag: "--daw-bundle", label: "DAW bundle", kind: .directory, group: Group.output, tier: .expert)
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(flag: "--stems", label: "Stems", kind: .string, group: Group.output, tier: .expert)
                .scoped(G.only(.aceStepBase, ignoredBy: [.magentaRT2])),
            .init(flag: "--adapter", label: "Adapter", kind: .file, repeatable: true, group: Group.modelAndAdapters, tier: .standard)
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(
                flag: "--adapter-kind", label: "Adapter kind", kind: .choice, choices: ["auto", "lora", "lokr"],
                defaultValue: "auto", group: Group.modelAndAdapters, tier: .expert, dependsOn: "--adapter"
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--adapter-scale", label: "Adapter scale", kind: .number, repeatable: true,
                group: Group.modelAndAdapters, tier: .standard, range: .init(min: 0, max: 2, step: 0.05), dependsOn: "--adapter"
            )
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(
                flag: "--model", aliases: ["-m"], label: "Model", kind: .string,
                defaultValue: "music-acestep", group: Group.modelAndAdapters, tier: .essential
            ),
            .init(flag: "--checkpoints-root", label: "Checkpoints root", kind: .directory, group: Group.modelAndAdapters, tier: .expert)
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(
                flag: "--decoder-subdirectory", aliases: ["--turbo-subdirectory"], label: "Decoder", kind: .string,
                defaultValue: "acestep-v15-turbo", group: Group.modelAndAdapters, tier: .expert
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--vae-subdirectory", label: "VAE", kind: .string,
                defaultValue: "vae", group: Group.modelAndAdapters, tier: .expert
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(flag: "--lm-subdirectory", label: "Language model", kind: .string, group: Group.modelAndAdapters, tier: .expert)
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(flag: "--lm-model", label: "LM model", kind: .string, group: Group.modelAndAdapters, tier: .expert)
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(flag: "--text-subdirectory", label: "Text encoder", kind: .string, group: Group.modelAndAdapters, tier: .expert)
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(flag: "--use-lm", label: "Use LM planning", kind: .boolean, group: Group.sampling, tier: .expert)
                .scoped(G.except(.miniMaxMusic3, .yue2, .magentaRT2)),
            .init(flag: "--no-lm", label: "Disable LM planning", kind: .boolean, group: Group.sampling, tier: .expert)
                .scoped(G.except(.miniMaxMusic3, .yue2, .magentaRT2)),
            .init(
                flag: "--analyze-source-audio", label: "Analyze source", kind: .boolean,
                group: Group.inputs, tier: .expert, dependsOn: "--source-audio"
            )
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(
                flag: "--duration", label: "Duration", kind: .number,
                group: Group.sampling, tier: .essential, range: .init(min: 1, max: 600, step: 1)
            )
                .scoped(
                    G.rule(.miniMaxMusic3, range: .init(min: 0, max: 360, step: 1)),
                    .rule(.yue2, range: .init(min: 0, max: 360, step: 1))
                ),
            .init(
                flag: "--quality", label: "Quality", kind: .choice, choices: ["draft", "song", "final", "edit"],
                group: Group.sampling, tier: .essential
            )
                .scoped(
                    G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2]),
                    .rule(.aceStepTurbo, defaultValue: "song"),
                    .rule(.aceStepSFT, defaultValue: "song"),
                    .rule(.aceStepBase, defaultValue: "song")
                ),
            .init(
                flag: "--steps", aliases: ["-s"], label: "Steps", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 200, step: 1)
            )
                .scoped(
                    G.except(.magentaRT2),
                    .rule(.aceStepTurbo, defaultValue: "8"),
                    .rule(.aceStepSFT, defaultValue: "50"),
                    .rule(.aceStepBase, defaultValue: "50"),
                    .rule(.miniMaxMusic3, defaultValue: "30"),
                    .rule(.yue2, defaultValue: "32", range: .init(min: 1, max: 1_000, step: 1))
                ),
            .init(
                flag: "--shift", label: "Scheduler shift", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            )
                .scoped(
                    G.except(.miniMaxMusic3, .yue2, .magentaRT2),
                    .rule(.aceStepTurbo, defaultValue: "3"),
                    .rule(.aceStepSFT, defaultValue: "1"),
                    .rule(.aceStepBase, defaultValue: "1")
                ),
            .init(
                flag: "--infer-method", label: "Inference method", kind: .choice, choices: ["ode", "sde"],
                group: Group.sampling, tier: .expert
            )
                .scoped(G.except(.miniMaxMusic3, .yue2, .magentaRT2)),
            .init(flag: "--sampler", label: "Sampler", kind: .choice, choices: ["euler", "heun"], group: Group.sampling, tier: .expert)
                .scoped(G.except(.miniMaxMusic3, .yue2, .magentaRT2)),
            .init(
                flag: "--guidance-scale", label: "Guidance scale", kind: .number,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 20, step: 0.1)
            )
                .scoped(
                    G.except(.magentaRT2),
                    .rule(.aceStepTurbo, values: ["1"], severity: .warning),
                    .rule(.aceStepSFT, defaultValue: "7"),
                    .rule(.aceStepBase, defaultValue: "7"),
                    .rule(.miniMaxMusic3, defaultValue: "1.7"),
                    .rule(.yue2, range: .init(min: 0, max: 20, step: 0.1))
                ),
            .init(
                flag: "--guidance-mode", label: "Guidance mode", kind: .choice, choices: ["apg", "adg", "cfg"],
                group: Group.sampling, tier: .expert
            )
                .scoped(G.only(.aceStepSFT, .aceStepBase, ignoredBy: [.aceStepTurbo])),
            .init(
                flag: "--cfg-interval-start", label: "CFG start", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            )
                .scoped(G.only(.aceStepSFT, .aceStepBase, ignoredBy: [.aceStepTurbo])),
            .init(
                flag: "--cfg-interval-end", label: "CFG end", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            )
                .scoped(G.only(.aceStepSFT, .aceStepBase, ignoredBy: [.aceStepTurbo])),
            .init(
                flag: "--velocity-norm-threshold", label: "Velocity clamp", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, step: 0.1)
            )
                .scoped(G.except(.miniMaxMusic3, .yue2, .magentaRT2)),
            .init(
                flag: "--velocity-ema-factor", label: "Velocity EMA", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            )
                .scoped(G.except(.miniMaxMusic3, .yue2, .magentaRT2)),
            .init(flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .essential, range: .init(min: 0, step: 1))
                .scoped(G.except(.magentaRT2)),
            .init(
                flag: "--candidates", aliases: ["--best-of"], label: "Candidate count", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 16, step: 1)
            )
                .scoped(
                    G.except(ignoredBy: [.magentaRT2]),
                    .rule(.miniMaxMusic3, values: ["1"]),
                    .rule(.yue2, values: ["1"])
                ),
            .init(
                flag: "--keep-candidates", label: "Keep candidates", kind: .boolean,
                group: Group.output, tier: .expert, dependsOn: "--candidates"
            )
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(
                flag: "--audio-cover-strength", label: "Cover strength", kind: .number,
                defaultValue: "1.0", group: Group.inputs, tier: .standard, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--source-audio"
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--cover-noise-strength", label: "Cover noise", kind: .number,
                defaultValue: "0.0", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--source-audio"
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2, .magentaRT2]
                ),
            .init(flag: "--retake-seed", label: "Retake seed", kind: .integer, group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1))
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(
                flag: "--retake-variance", label: "Retake variance", kind: .number,
                defaultValue: "0.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--retake-seed"
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(flag: "--vocal-language", label: "Vocal language", kind: .string, defaultValue: "en", group: Group.prompt, tier: .standard)
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--instruction", label: "Instruction", kind: .string,
                defaultValue: "Fill the audio semantic mask based on the given conditions:", group: Group.prompt, tier: .expert
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--task-type", aliases: ["--task"],
                label: "Task",
                kind: .choice,
                choices: ["text2music", "repaint", "cover", "cover-nofsq", "extract", "lego", "complete"],
                defaultValue: "text2music", group: Group.inputs, tier: .standard
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2, .magentaRT2],
                    .rule(.aceStepTurbo, values: ["text2music", "repaint", "cover", "cover-nofsq"]),
                    .rule(.aceStepSFT, values: ["text2music", "repaint", "cover", "cover-nofsq"])
                ),
            .init(flag: "--source-audio", label: "Source audio", kind: .file, group: Group.inputs, tier: .standard)
                .scoped(G.except(.miniMaxMusic3, .yue2, .magentaRT2)),
            .init(
                flag: "--reference-audio", label: "Reference audio", kind: .file, repeatable: true,
                group: Group.inputs, tier: .standard
            )
                .scoped(G.except(.miniMaxMusic3, .yue2, .magentaRT2)),
            .init(flag: "--track-name", label: "Track name", kind: .string, group: Group.inputs, tier: .expert)
                .scoped(G.only(.aceStepBase, ignoredBy: [.aceStepTurbo, .aceStepSFT])),
            .init(flag: "--complete-track-classes", label: "Track classes", kind: .string, group: Group.inputs, tier: .expert)
                .scoped(G.only(.aceStepBase, ignoredBy: [.aceStepTurbo, .aceStepSFT])),
            .init(flag: "--non-cover", label: "No FSQ cover", kind: .boolean, group: Group.inputs, tier: .expert, dependsOn: "--source-audio")
                .scoped(G.except(.miniMaxMusic3, .yue2, .magentaRT2)),
            .init(
                flag: "--repaint-start", label: "Repaint start", kind: .number,
                defaultValue: "0.0", group: Group.inputs, tier: .standard, range: .init(min: 0, step: 0.1), dependsOn: "--source-audio"
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--repaint-end", label: "Repaint end", kind: .number,
                defaultValue: "-1.0", group: Group.inputs, tier: .standard, range: .init(min: -1, step: 0.1), dependsOn: "--source-audio"
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--chunk-mask-mode", label: "Chunk mask", kind: .choice, choices: ["auto", "explicit"],
                defaultValue: "auto", group: Group.inputs, tier: .expert, dependsOn: "--source-audio"
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--repaint-mode", label: "Repaint mode", kind: .choice, choices: ["conservative", "balanced", "aggressive"],
                defaultValue: "balanced", group: Group.inputs, tier: .expert, dependsOn: "--source-audio"
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--repaint-strength", label: "Repaint strength", kind: .number,
                defaultValue: "0.5", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--source-audio"
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(flag: "--flow-edit", label: "Flow edit", kind: .boolean, group: Group.inputs, tier: .standard, dependsOn: "--source-audio")
                .scoped(G.except(.miniMaxMusic3, .yue2, .magentaRT2)),
            .init(flag: "--source-caption", label: "Source caption", kind: .string, group: Group.inputs, tier: .standard, dependsOn: "--flow-edit")
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(flag: "--source-lyrics", label: "Source lyrics", kind: .string, group: Group.inputs, tier: .expert, dependsOn: "--flow-edit")
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(
                flag: "--flow-edit-n-min", label: "Flow start", kind: .number,
                defaultValue: "0.0", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05), dependsOn: "--flow-edit"
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--flow-edit-n-max", label: "Flow end", kind: .number,
                defaultValue: "1.0", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05), dependsOn: "--flow-edit"
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--flow-edit-n-average", label: "Flow samples", kind: .integer,
                defaultValue: "1", group: Group.inputs, tier: .expert, range: .init(min: 1, max: 16, step: 1), dependsOn: "--flow-edit"
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(flag: "--bpm", label: "BPM", kind: .integer, group: Group.prompt, tier: .standard, range: .init(min: 20, max: 300, step: 1))
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(flag: "--keyscale", aliases: ["--key"], label: "Key", kind: .string, group: Group.prompt, tier: .standard)
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(flag: "--timesignature", aliases: ["--timesig"], label: "Time signature", kind: .string, group: Group.prompt, tier: .standard)
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(
                flag: "--lm-temperature", label: "LM temperature", kind: .number,
                defaultValue: "0.85", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 2, step: 0.05)
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--lm-top-k", label: "LM top-k", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1_000, step: 1)
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--lm-top-p", label: "LM top-p", kind: .number,
                defaultValue: "0.9", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--lm-repetition-penalty", label: "LM repetition penalty", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0.5, max: 2, step: 0.05)
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--lm-cfg-scale", label: "LM CFG scale", kind: .number,
                defaultValue: "2.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--lm-negative-prompt", label: "LM negative prompt", kind: .string,
                defaultValue: "NO USER INPUT", group: Group.sampling, tier: .expert
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(flag: "--metadata-duration", label: "Metadata duration", kind: .string, group: Group.prompt, tier: .expert)
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(flag: "--metadata-language", label: "Metadata language", kind: .string, group: Group.prompt, tier: .expert)
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(flag: "--no-tiled-vae", label: "Disable tiled VAE", kind: .boolean, group: Group.run, tier: .expert)
                .scoped(G.except(.miniMaxMusic3, .yue2, ignoredBy: [.magentaRT2])),
            .init(
                flag: "--vae-chunk-size", label: "VAE chunk", kind: .integer,
                defaultValue: "512", group: Group.run, tier: .expert, range: .init(min: 64, max: 4_096, step: 64)
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--vae-overlap", label: "VAE overlap", kind: .integer,
                defaultValue: "64", group: Group.run, tier: .expert, range: .init(min: 0, max: 1_024, step: 8)
            )
                .scoped(
                    G.except(ignoredBy: [.miniMaxMusic3, .yue2, .magentaRT2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            .init(
                flag: "--temperature", label: "RT2 temperature", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 2, step: 0.05)
            )
                .scoped(
                    G.only(.magentaRT2, ignoredBy: G.aceStep + [.miniMaxMusic3, .yue2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--style-conditioning", label: "RT2 style", kind: .choice, choices: ["streaming", "full"],
                defaultValue: "streaming", group: Group.sampling, tier: .expert
            )
                .scoped(
                    G.only(.magentaRT2, ignoredBy: G.aceStep + [.miniMaxMusic3, .yue2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--top-k", label: "RT2 top-k", kind: .integer,
                defaultValue: "100", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1_000, step: 1)
            )
                .scoped(
                    G.only(.magentaRT2, ignoredBy: G.aceStep + [.miniMaxMusic3, .yue2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--cfg-musiccoca", label: "MusicCoCa CFG", kind: .number,
                defaultValue: "3.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            )
                .scoped(
                    G.only(.magentaRT2, ignoredBy: G.aceStep + [.miniMaxMusic3, .yue2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--cfg-notes", label: "Notes CFG", kind: .number,
                defaultValue: "5.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            )
                .scoped(
                    G.only(.magentaRT2, ignoredBy: G.aceStep + [.miniMaxMusic3, .yue2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--cfg-drums", label: "Drums CFG", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            )
                .scoped(
                    G.only(.magentaRT2, ignoredBy: G.aceStep + [.miniMaxMusic3, .yue2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(flag: "--drumless", label: "Drumless", kind: .boolean, group: Group.sampling, tier: .expert)
                .scoped(G.only(.magentaRT2, ignoredBy: G.aceStep)),
            .init(
                flag: "--unmask-width", label: "Unmask width", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)
            )
                .scoped(
                    G.only(.magentaRT2, ignoredBy: G.aceStep + [.miniMaxMusic3, .yue2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(
                flag: "--seed-rotation", label: "Seed rotation", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)
            )
                .scoped(
                    G.only(.magentaRT2, ignoredBy: G.aceStep + [.miniMaxMusic3, .yue2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            .init(flag: "--prefill-silence", label: "Prefill silence", kind: .boolean, group: Group.sampling, tier: .expert)
                .scoped(G.only(.magentaRT2, ignoredBy: G.aceStep)),
            .init(
                flag: "--prefill-duration", label: "Prefill duration", kind: .number,
                defaultValue: "1.64", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 30, step: 0.01),
                dependsOn: "--prefill-silence"
            )
                .scoped(
                    G.only(.magentaRT2, ignoredBy: G.aceStep + [.miniMaxMusic3, .yue2]),
                    onlyDefaultFor: [.miniMaxMusic3, .yue2]
                ),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output"),
        routing: musicGenerateRouting
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
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            .init(flag: "--checkpoints-root", label: "Checkpoints root", kind: .directory),
            .init(flag: "--decoder-subdirectory", aliases: ["--turbo-subdirectory"], label: "Decoder", kind: .string),
            .init(flag: "--vae-subdirectory", label: "VAE", kind: .string),
            .init(flag: "--lm-subdirectory", label: "Language model", kind: .string),
            .init(flag: "--lm-model", label: "LM model", kind: .string),
            .init(flag: "--duration", label: "Duration", kind: .number, group: Group.sampling, tier: .essential),
            .init(flag: "--max-new-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--lm-temperature", label: "LM temperature", kind: .number),
            .init(flag: "--lm-top-k", label: "LM top-k", kind: .integer),
            .init(flag: "--lm-top-p", label: "LM top-p", kind: .number),
            .init(flag: "--include-raw-lm", label: "Raw LM", kind: .boolean),
            .init(flag: "--include-audio-codes", label: "Audio codes", kind: .boolean),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text),
        routing: musicAnalyzeRouting
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
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            .init(flag: "--model-path", label: "Model path", kind: .directory),
            // Managed MuScriptor models ship `config.json`, which wins over the variant; it only
            // sizes a local checkpoint that has none, which the contract does not identify.
            .init(flag: "--variant", label: "Variant", kind: .choice, choices: ["small", "medium", "large"],
                  group: Group.run, tier: .expert)
                .scoped(MuScriptorFamily.only(ignoredBy: [.muScriptor])),
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file),
            .init(flag: "--format", aliases: ["-f"], label: "Format", kind: .choice, choices: ["midi", "json", "jsonl"],
                  group: Group.output, tier: .essential),
            .init(flag: "--instruments", label: "Instruments", kind: .string),
            .init(flag: "--list-instruments", label: "List instruments", kind: .boolean),
            .init(flag: "--sampling", label: "Sampling", kind: .boolean),
            .init(flag: "--temperature", aliases: ["-t"], label: "Temperature", kind: .number),
            .init(flag: "--max-tokens-per-chunk", label: "Tokens per chunk", kind: .integer),
            .init(flag: "--strict-eos", label: "Strict EOS", kind: .boolean),
            .init(flag: "--beam-size", label: "Beam size", kind: .integer),
            .init(flag: "--chunk-batch-size", label: "Chunk batch", kind: .integer),
            .init(flag: "--dtype", label: "Compute type", kind: .choice, choices: ["bfloat16", "float16", "float32"]),
            .init(flag: "--no-musical-context", label: "Disable context", kind: .boolean),
            .init(flag: "--context-output", label: "Context output", kind: .file),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, flag: "--output"),
        routing: musicTranscribeRouting
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
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--model-path", label: "Model path", kind: .directory, group: Group.modelAndAdapters, tier: .expert),
            .init(flag: "--output-dir", aliases: ["-o"], label: "Output directory", kind: .directory, group: Group.output, tier: .standard),
            .init(flag: "--overlap", label: "Chunk overlap", kind: .integer, tier: .standard, range: .init(min: 1, step: 1))
                .scoped(
                    P.rule(.bsRoFormer2Stem, values: roFormerOverlaps(chunkSize: 352_800), defaultValue: "2"),
                    .rule(.bsRoFormer4Stem, values: roFormerOverlaps(chunkSize: 485_100), defaultValue: "2"),
                    .rule(.melRoFormerDereverb, values: roFormerOverlaps(chunkSize: 352_800), defaultValue: "2"),
                    .rule(.melRoFormerDenoise, values: roFormerOverlaps(chunkSize: 352_800), defaultValue: "4")
                ),
            .init(
                flag: "--dtype",
                label: "Compute",
                kind: .choice,
                choices: ["float16", "float32"],
                defaultValue: "float16",
                tier: .essential
            ),
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean, group: Group.run, tier: .expert)
        ],
        output: .init(kind: .directory, flag: "--output-dir"),
        routing: musicSeparateRouting
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
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            .init(flag: "--duration", label: "Duration", kind: .number),
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file),
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
            .init(flag: "--quiet", aliases: ["-q"], label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .service, fileExtension: "wav", flag: "--output", optional: true),
        routing: musicRealtimeRouting
    )

    public static let musicTrainAdapter = MereRunCommandCapability(
        id: "music.train-adapter",
        command: ["music", "train-adapter"],
        title: "Train music adapter",
        summary: "Train a native ACE-Step LoRA or LoKr adapter.",
        options: [
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            .init(flag: "--dataset", label: "Dataset", kind: .file, required: true),
            .init(flag: "--output", aliases: ["-o"], label: "Output", kind: .file, required: true),
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
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output"),
        routing: musicTrainAdapterRouting
    )

    public static let musicServe = MereRunCommandCapability(
        id: "music.serve",
        command: ["music", "serve"],
        title: "Serve resident music",
        summary: "Keep ACE-Step, with its language model and adapters, or MiniMax Music 3 warm behind a local API.",
        options: [
            .init(
                flag: "--memory-mode", label: "Memory mode", kind: .choice, choices: ["staged", "resident"],
                group: Group.run, tier: .expert
            )
                .scoped(S.only(.miniMaxMusic3)),
            .init(
                flag: "--performance-mode", label: "Performance mode", kind: .choice,
                choices: ["reference", "optimized", "q8", "q4", "q8-lm", "q4-lm"], group: Group.run, tier: .expert
            )
                .scoped(S.only(.miniMaxMusic3)),
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", aliases: ["-p"], label: "Port", kind: .integer),
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            .init(flag: "--checkpoints-root", label: "Checkpoints root", kind: .directory)
                .scoped(S.only(.aceStep)),
            .init(flag: "--decoder-subdirectory", label: "Decoder", kind: .string, defaultValue: "acestep-v15-turbo")
                .scoped(S.only(.aceStep, ignoredBy: [.miniMaxMusic3]), onlyDefaultFor: [.miniMaxMusic3]),
            .init(flag: "--vae-subdirectory", label: "VAE", kind: .string, defaultValue: "vae")
                .scoped(S.only(.aceStep, ignoredBy: [.miniMaxMusic3]), onlyDefaultFor: [.miniMaxMusic3]),
            .init(flag: "--lm-subdirectory", label: "Language model", kind: .string)
                .scoped(S.only(.aceStep)),
            .init(flag: "--lm-model", label: "LM model", kind: .string)
                .scoped(S.only(.aceStep)),
            .init(flag: "--text-subdirectory", label: "Text encoder", kind: .string)
                .scoped(S.only(.aceStep)),
            .init(flag: "--adapter", label: "Adapter", kind: .file, repeatable: true)
                .scoped(S.only(.aceStep)),
            .init(flag: "--adapter-kind", label: "Adapter kind", kind: .choice, choices: ["auto", "lora", "lokr"],
                  defaultValue: "auto")
                .scoped(S.only(.aceStep, ignoredBy: [.miniMaxMusic3]), onlyDefaultFor: [.miniMaxMusic3]),
            .init(flag: "--adapter-scale", label: "Adapter scale", kind: .number, repeatable: true)
                .scoped(S.only(.aceStep)),
            .init(flag: "--api-key", label: "API key", kind: .string)
        ],
        output: .init(kind: .service),
        routing: musicServeRouting
    )
}
