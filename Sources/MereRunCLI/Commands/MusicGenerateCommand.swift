import ArgumentParser
import AudioCodecs
import Foundation
import MLX
import MereRunCore

struct MusicGenerate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate audio from a music prompt.",
        discussion: """
        Prints the output WAV path to stdout.
        Progress and diagnostics are printed to stderr.

        Example:
          mere.run music generate "upbeat electronic groove" \
            --model music-acestep \
            --lyrics "[verse]\\nwe dance all night" \
            --text-subdirectory Qwen3-Embedding-0.6B \
            -o out.wav

          mere.run music generate "ambient modular synths with brushed drums" \
            --model music-magenta-rt2-small \
            --duration 4 \
            -o out.wav

          mere.run music generate "dream-pop cover with soft vocals" \
            --source-audio ~/Downloads/song.mp3 \
            --lyrics "[verse]\\nnew words over the old shape" \
            --audio-cover-strength 1.0 \
            -o cover.wav
        """
    )

    @Argument(help: "Caption prompt describing the target music.")
    var caption: String

    @Option(name: [.long], help: "Lyrics text (optional).")
    var lyrics: String = ""

    @Option(name: [.customLong("lyrics-file")], help: "Path to lyrics text file. Cannot be used with --lyrics.")
    var lyricsFile: String?

    @Option(name: [.customLong("lrc-file")], help: "Synchronized LRC lyrics input. Cannot be combined with plain lyrics options.")
    var lrcFile: String?

    @Option(name: [.customLong("lrc-output")], help: "Write synchronized LRC; plain lyrics receive clearly marked approximate line timing.")
    var lrcOutput: String?

    @Option(name: [.customShort("o"), .long], help: "Output WAV path (default: ./mererun-music-<timestamp>.wav).")
    var output: String?

    @Option(name: [.customLong("export-format")], help: "WAV encoding: pcm16, pcm24, or float32.")
    var exportFormat: ACEStepAudioFormat = .pcm24

    @Option(name: [.customLong("normalize")], help: "Output normalization: none or peak.")
    var normalization: ACEStepNormalizationMode = .peak

    @Option(name: [.customLong("target-peak-db")], help: "Peak-normalization target in dBFS.")
    var targetPeakDB: Float = -1

    @Option(name: [.customLong("fade-in-ms")], help: "Output fade-in in milliseconds.")
    var fadeInMilliseconds: Float = 5

    @Option(name: [.customLong("fade-out-ms")], help: "Output fade-out in milliseconds.")
    var fadeOutMilliseconds: Float = 20

    @Flag(name: [.customLong("no-dither")], help: "Disable deterministic TPDF dither for integer PCM export.")
    var noDither: Bool = false

    @Option(name: [.customLong("recipe-output")], help: "Generation recipe JSON path (default: <output>.recipe.json).")
    var recipeOutput: String?

    @Flag(name: [.customLong("no-recipe")], help: "Do not write the reproducible generation recipe sidecar.")
    var noRecipe: Bool = false

    @Option(name: [.customLong("daw-bundle")], help: "Write a portable DAW bundle with WAVs, recipe, LRC markers, and a REAPER project.")
    var dawBundle: String?

    @Option(name: [.customLong("stems")], help: "Comma-separated stems to extract after generation (Base checkpoints only).")
    var stems: String?

    @Option(
        name: [.customLong("adapter")],
        parsing: .upToNextOption,
        help: "PEFT LoRA or LyCORIS LoKr safetensors file(s) to stack on the ACE-Step decoder."
    )
    var adapters: [String] = []

    @Option(
        name: [.customLong("adapter-kind")],
        help: "Adapter format: auto, lora, or lokr."
    )
    var adapterKind: ACEStepAdapterKind = .auto

    @Option(
        name: [.customLong("adapter-scale")],
        parsing: .upToNextOption,
        help: "Adapter scale(s): one value for all adapters, or one per --adapter path."
    )
    var adapterScales: [Float] = []

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local path to the ACE-Step model root/checkpoints root.")
    var model: String = ModelResolver.ModelID.aceStep.rawValue

    @Option(name: [.customLong("checkpoints-root")], help: "Root directory containing ACE-Step checkpoint subdirectories. Auto-discovered if not set.")
    var checkpointsRoot: String?

    @Option(
        name: [.customLong("decoder-subdirectory"), .customLong("turbo-subdirectory")],
        help: "ACE-Step decoder subdirectory under checkpoints root."
    )
    var turboSubdirectory: String = "acestep-v15-turbo"

    @Option(name: [.customLong("vae-subdirectory")], help: "VAE subdirectory under checkpoints root.")
    var vaeSubdirectory: String = "vae"

    @Option(name: [.customLong("lm-subdirectory")], help: "Optional 5Hz LM subdirectory under checkpoints root. Auto-detected as 'acestep-5Hz-lm-1.7B' when omitted.")
    var lmSubdirectory: String?

    @Option(name: [.customLong("text-subdirectory")], help: "Text encoder subdirectory under checkpoints root. Auto-detected as 'Qwen3-Embedding-0.6B' when omitted.")
    var textSubdirectory: String?

    @Flag(name: [.customLong("use-lm")], help: "Use 5Hz LM constrained decoding for supported ACE-Step tasks.")
    var useLM: Bool = false

    @Flag(name: [.customLong("no-lm")], help: "Disable ACE-Step LM planning and semantic-code generation selected by a quality preset.")
    var noLM: Bool = false

    @Flag(
        name: [.customLong("analyze-source-audio")],
        help: "Use ACE-Step 5Hz LM audio understanding to fill missing cover metadata from --source-audio."
    )
    var analyzeSourceAudio: Bool = false

    @Option(name: [.customLong("duration")], help: "Output duration in seconds (omitting it lets song/final quality plan duration).")
    var durationSeconds: Float?

    @Option(name: [.customLong("quality")], help: "Adaptive ACE-Step quality: draft, song, final, or edit.")
    var quality: ACEStepQualityPreset = .song

    @Option(name: [.customShort("s"), .long], help: "Denoise steps (default: Turbo 8; Base/SFT 50).")
    var steps: Int?

    @Option(name: [.customLong("shift")], help: "Flow scheduler shift (default: Turbo 3; Base/SFT 1).")
    var shift: Float?

    @Option(name: [.customLong("infer-method")], help: "Diffusion method: ode or sde (default: ode).")
    var inferMethod: ACEStepInferenceMethod?

    @Option(name: [.customLong("sampler")], help: "ODE sampler: euler or heun (default: euler).")
    var samplerMode: ACEStepSamplerMode?

    @Option(name: [.customLong("guidance-scale")], help: "Base/SFT diffusion guidance (default: 7; Turbo always uses 1).")
    var guidanceScale: Float?

    @Option(name: [.customLong("guidance-mode")], help: "Base/SFT guidance: apg, adg, or cfg (default: apg).")
    var guidanceMode: ACEStepGuidanceMode?

    @Option(name: [.customLong("cfg-interval-start")], help: "Enable guidance at or above this timestep in [0, 1].")
    var cfgIntervalStart: Float?

    @Option(name: [.customLong("cfg-interval-end")], help: "Enable guidance at or below this timestep in [0, 1].")
    var cfgIntervalEnd: Float?

    @Option(name: [.customLong("velocity-norm-threshold")], help: "Clamp velocity norm relative to the latent norm (0 disables).")
    var velocityNormThreshold: Float?

    @Option(name: [.customLong("velocity-ema-factor")], help: "Blend each velocity with its predecessor in [0, 1) (0 disables).")
    var velocityEMAFactor: Float?

    @Option(name: [.long], help: "Seed for deterministic generation.")
    var seed: UInt64?

    @Option(
        name: [.customLong("candidates"), .customLong("best-of")],
        help: "Generate and rank this many warm-session candidates (preset default: draft 1, song/edit 2, final 4)."
    )
    var candidateCount: Int?

    @Flag(
        name: [.customLong("keep-candidates")],
        help: "Save every ranked candidate next to the selected output."
    )
    var keepCandidates: Bool = false

    @Option(name: [.customLong("audio-cover-strength")], help: "Cover-conditioning strength in [0, 1].")
    var audioCoverStrength: Float = 1.0

    @Option(name: [.customLong("cover-noise-strength")], help: "Source-latent noise initialization strength in [0, 1] for ACE-Step covers.")
    var coverNoiseStrength: Float = 0.0

    @Option(name: [.customLong("retake-seed")], help: "Independent seed used to vary a reproducible retake.")
    var retakeSeed: UInt64?

    @Option(name: [.customLong("retake-variance")], help: "Retake interpolation in [0, 1]; 0 keeps --seed and 1 uses --retake-seed.")
    var retakeVariance: Float = 0

    @Option(name: [.customLong("vocal-language")], help: "Language tag used in lyric prompt formatting.")
    var vocalLanguage: String = "en"

    @Option(name: [.customLong("instruction")], help: "Caption instruction prefix.")
    var instruction: String = "Fill the audio semantic mask based on the given conditions:"

    @Option(
        name: [.customLong("task-type"), .customLong("task")],
        help: "ACE-Step task: text2music, repaint, cover, cover-nofsq, extract, lego, or complete."
    )
    var taskType: ACEStepTask = .textToMusic

    @Option(name: [.customLong("source-audio")], help: "Source audio for ACE-Step cover, repaint, extract, lego, or complete. With the default task, implies cover.")
    var sourceAudio: String?

    @Option(name: [.customLong("reference-audio")], parsing: .upToNextOption, help: "Optional reference audio file(s) for ACE-Step timbre conditioning.")
    var referenceAudio: [String] = []

    @Option(name: [.customLong("track-name")], help: "Track name for extract/lego tasks.")
    var trackName: String?

    @Option(name: [.customLong("complete-track-classes")], help: "Comma-separated track classes for complete task (e.g. Drums,Bass).")
    var completeTrackClasses: String?

    @Flag(name: [.customLong("non-cover")], help: "Compatibility alias that selects cover-nofsq instead of FSQ cover conditioning.")
    var nonCover: Bool = false

    @Option(name: [.customLong("repaint-start")], help: "Repaint/lego range start in seconds.")
    var repaintStartSeconds: Float = 0

    @Option(name: [.customLong("repaint-end")], help: "Repaint/lego range end in seconds (-1 uses the source end).")
    var repaintEndSeconds: Float = -1

    @Option(name: [.customLong("chunk-mask-mode")], help: "Repaint chunk-mask mode: auto or explicit.")
    var chunkMaskMode: ACEStepChunkMaskMode = .auto

    @Option(name: [.customLong("repaint-mode")], help: "Source preservation: conservative, balanced, or aggressive.")
    var repaintMode: ACEStepRepaintMode = .balanced

    @Option(name: [.customLong("repaint-strength")], help: "Balanced repaint aggressiveness in [0, 1].")
    var repaintStrength: Float = 0.5

    @Flag(name: [.customLong("flow-edit")], help: "Morph --source-audio from its source caption/lyrics toward the target prompt.")
    var flowEdit: Bool = false

    @Option(name: [.customLong("source-caption")], help: "Caption describing --source-audio before flow edit.")
    var sourceCaption: String?

    @Option(name: [.customLong("source-lyrics")], help: "Original lyrics used by flow edit.")
    var sourceLyrics: String = ""

    @Option(name: [.customLong("flow-edit-n-min")], help: "Normalized flow-edit integration window start.")
    var flowEditNMin: Float = 0

    @Option(name: [.customLong("flow-edit-n-max")], help: "Normalized flow-edit integration window end.")
    var flowEditNMax: Float = 1

    @Option(name: [.customLong("flow-edit-n-average")], help: "Monte Carlo forward-noise draws per flow-edit step.")
    var flowEditNAverage: Int = 1

    @Option(name: [.customLong("bpm")], help: "Optional BPM metadata for constrained LM / prompt metadata.")
    var bpm: Int?

    @Option(name: [.customLong("keyscale"), .customLong("key")], help: "Optional key metadata (e.g. C major).")
    var keyscale: String?

    @Option(name: [.customLong("timesignature"), .customLong("timesig")], help: "Optional time signature metadata (e.g. 4).")
    var timesignature: String?

    @Option(name: [.customLong("lm-top-k")], help: "LM top-k sampling (0 = disabled).")
    var lmTopK: Int = 0

    @Option(name: [.customLong("lm-top-p")], help: "LM top-p nucleus sampling (1.0 = disabled).")
    var lmTopP: Float = 0.9

    @Option(name: [.customLong("metadata-duration")], help: "Optional metadata duration string for constrained LM.")
    var metadataDuration: String?

    @Option(name: [.customLong("metadata-language")], help: "Optional language metadata for constrained LM.")
    var metadataLanguage: String?

    @Flag(name: [.customLong("no-tiled-vae")], help: "Disable tiled VAE decode.")
    var noTiledVAE: Bool = false

    @Option(name: [.customLong("vae-chunk-size")], help: "VAE tiled decode chunk size.")
    var vaeChunkSize: Int = 512

    @Option(name: [.customLong("vae-overlap")], help: "VAE tiled decode overlap.")
    var vaeOverlap: Int = 64

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    @Option(name: [.customLong("temperature")], help: "Magenta RT2 sampling temperature.")
    var magentaTemperature: Float = 1.0

    @Option(name: [.customLong("style-conditioning")], help: "Magenta RT2 style conditioning detail: streaming or full.")
    var magentaStyleConditioning: MagentaRT2StyleConditioning = .streaming

    @Option(name: [.customLong("top-k")], help: "Magenta RT2 top-k sampling.")
    var magentaTopK: Int = 100

    @Option(name: [.customLong("cfg-musiccoca")], help: "Magenta RT2 MusicCoCa guidance scale.")
    var magentaCFGMusicCoCa: Float = 3.0

    @Option(name: [.customLong("cfg-notes")], help: "Magenta RT2 MIDI-notes guidance scale.")
    var magentaCFGNotes: Float = 5.0

    @Option(name: [.customLong("cfg-drums")], help: "Magenta RT2 drums guidance scale.")
    var magentaCFGDrums: Float = 1.0

    @Flag(name: [.customLong("drumless")], help: "Enable Magenta RT2 drumless mode.")
    var magentaDrumless: Bool = false

    @Option(name: [.customLong("unmask-width")], help: "Magenta RT2 token unmask width.")
    var magentaUnmaskWidth: Int = 0

    @Option(name: [.customLong("seed-rotation")], help: "Magenta RT2 seed rotation.")
    var magentaSeedRotation: Int = 0

    @Flag(name: [.customLong("prefill-silence")], help: "Prefill Magenta RT2 state with silence before generation.")
    var magentaPrefillSilence: Bool = false

    @Option(name: [.customLong("prefill-duration")], help: "Magenta RT2 silent prefill duration in seconds.")
    var magentaPrefillDuration: Float = 1.64

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        if let durationSeconds, durationSeconds <= 0 {
            throw ValidationError("--duration must be > 0")
        }
        if useLM && noLM {
            throw ValidationError("Pass either --use-lm or --no-lm, not both.")
        }
        if let steps, steps < 1 {
            throw ValidationError("--steps must be >= 1")
        }
        if let shift, shift <= 0 {
            throw ValidationError("--shift must be > 0")
        }
        if let guidanceScale, guidanceScale < 1 {
            throw ValidationError("--guidance-scale must be >= 1")
        }
        if let cfgIntervalStart, !(0...1).contains(cfgIntervalStart) {
            throw ValidationError("--cfg-interval-start must be between 0 and 1")
        }
        if let cfgIntervalEnd, !(0...1).contains(cfgIntervalEnd) {
            throw ValidationError("--cfg-interval-end must be between 0 and 1")
        }
        let resolvedCFGStart = cfgIntervalStart ?? 0
        let resolvedCFGEnd = cfgIntervalEnd ?? 1
        guard resolvedCFGStart <= resolvedCFGEnd else {
            throw ValidationError("--cfg-interval-start must be <= --cfg-interval-end")
        }
        if let velocityNormThreshold, velocityNormThreshold < 0 {
            throw ValidationError("--velocity-norm-threshold must be >= 0")
        }
        if let velocityEMAFactor, !(0..<1).contains(velocityEMAFactor) {
            throw ValidationError("--velocity-ema-factor must be between 0 (inclusive) and 1 (exclusive)")
        }
        guard (0.0...1.0).contains(audioCoverStrength) else {
            throw ValidationError("--audio-cover-strength must be between 0.0 and 1.0")
        }
        if let candidateCount, !(1...16).contains(candidateCount) {
            throw ValidationError("--candidates must be between 1 and 16")
        }
        guard (0.0...1.0).contains(coverNoiseStrength) else {
            throw ValidationError("--cover-noise-strength must be between 0.0 and 1.0")
        }
        guard (0.0...1.0).contains(retakeVariance) else {
            throw ValidationError("--retake-variance must be between 0.0 and 1.0")
        }
        guard repaintStartSeconds >= 0 else {
            throw ValidationError("--repaint-start must be >= 0")
        }
        guard repaintEndSeconds == -1 || repaintEndSeconds > repaintStartSeconds else {
            throw ValidationError("--repaint-end must be -1 or greater than --repaint-start")
        }
        guard (0.0...1.0).contains(repaintStrength) else {
            throw ValidationError("--repaint-strength must be between 0.0 and 1.0")
        }
        if flowEdit {
            guard sourceAudio != nil else {
                throw ValidationError("--flow-edit requires --source-audio")
            }
            guard sourceCaption?.trimmingCharacters(
                in: .whitespacesAndNewlines
            ).isEmpty == false else {
                throw ValidationError("--flow-edit requires --source-caption")
            }
            guard 0 <= flowEditNMin,
                  flowEditNMin <= flowEditNMax,
                  flowEditNMax <= 1
            else {
                throw ValidationError(
                    "--flow-edit requires 0 <= n-min <= n-max <= 1"
                )
            }
            guard flowEditNAverage >= 1 else {
                throw ValidationError("--flow-edit-n-average must be >= 1")
            }
        }
        guard lmTopK >= 0 else {
            throw ValidationError("--lm-top-k must be >= 0")
        }
        guard (0.0...1.0).contains(lmTopP) else {
            throw ValidationError("--lm-top-p must be between 0.0 and 1.0")
        }
        guard vaeChunkSize > 0 else {
            throw ValidationError("--vae-chunk-size must be > 0")
        }
        guard vaeOverlap >= 0 else {
            throw ValidationError("--vae-overlap must be >= 0")
        }
        if lyricsFile != nil, !lyrics.isEmpty {
            throw ValidationError("Pass either --lyrics or --lyrics-file, not both.")
        }
        if lrcFile != nil, lyricsFile != nil || !lyrics.isEmpty {
            throw ValidationError(
                "Pass --lrc-file or plain --lyrics/--lyrics-file, not both."
            )
        }
        guard targetPeakDB <= 0 else {
            throw ValidationError("--target-peak-db must be <= 0")
        }
        guard fadeInMilliseconds >= 0, fadeOutMilliseconds >= 0 else {
            throw ValidationError("Output fades must be >= 0")
        }
        if dawBundle != nil, noRecipe {
            throw ValidationError(
                "--daw-bundle requires recipe metadata; remove --no-recipe"
            )
        }

        if isMagentaRT2Request {
            try await runMagentaRT2()
            return
        }

        let effectiveTask = resolvedACEStepTask
        var effectiveUseLM = flowEdit
            ? false
            : resolvedACEStepUsesLM(task: effectiveTask)
        if analyzeSourceAudio && effectiveTask != .cover && effectiveTask != .coverNoFSQ {
            throw ValidationError("--analyze-source-audio requires ACE-Step cover mode with --source-audio.")
        }
        let wantsLMResources = effectiveUseLM || analyzeSourceAudio

        let checkpointsRootURL = try await resolveACEStepCheckpointsRoot()
        let resolvedTurboSubdirectory = try resolveACEStepTurboSubdirectory(
            at: checkpointsRootURL,
            explicit: turboSubdirectory
        )
        let checkpointVariant = try ACEStepCheckpointVariant.load(
            modelRootURL: checkpointsRootURL.appendingPathComponent(
                resolvedTurboSubdirectory,
                isDirectory: true
            )
        )
        do {
            try checkpointVariant.validate(effectiveTask)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        let stemNames = parsedStemNames()
        if !stemNames.isEmpty {
            do {
                try checkpointVariant.validate(.extract)
            } catch {
                throw ValidationError(
                    "--stems requires an ACE-Step Base checkpoint: "
                        + error.localizedDescription
                )
            }
        }
        let qualityDefaults = quality.defaults(
            for: checkpointVariant,
            task: effectiveTask
        )
        let resolvedLMSubdirectory = try wantsLMResources
            ? resolveACEStepLMSubdirectory(at: checkpointsRootURL, explicit: lmSubdirectory)
            : nil
        let resolvedTextSubdirectory = try resolveACEStepTextSubdirectory(
            at: checkpointsRootURL,
            explicit: textSubdirectory
        )
        if wantsLMResources && resolvedLMSubdirectory == nil {
            if useLM || analyzeSourceAudio || lmSubdirectory != nil {
                throw ValidationError(
                    "ACE-Step LM planning/generation and --analyze-source-audio require --lm-subdirectory. "
                        + "Set --lm-subdirectory or use a managed model that includes a 5Hz LM."
                )
            }
            effectiveUseLM = false
            if !quiet {
                CLIStderr.write(
                    "No 5Hz LM is installed with this checkpoint; "
                        + "continuing with direct DiT generation. Use --no-lm to make this choice explicit.\n"
                )
            }
        }
        let needsLMResources = effectiveUseLM || analyzeSourceAudio
        if resolvedTextSubdirectory == nil {
            throw ValidationError("ACE-Step text encoder not found. Set --text-subdirectory or keep a default layout like 'Qwen3-Embedding-0.6B'.")
        }
        if quality == .final,
           let resolvedLMSubdirectory,
           !resolvedLMSubdirectory.lowercased().contains("4b"),
           !quiet
        {
            CLIStderr.write(
                "Final quality is using \(resolvedLMSubdirectory); "
                    + "use music-acestep-xl-turbo-lm4b or --lm-subdirectory with the 4B LM for the strongest planning.\n"
            )
        }

        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-music", defaultExtension: "wav")
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let inputLRC = try loadLRC()
        let resolvedLyrics = try inputLRC?.lyrics ?? loadLyrics()
        let sourceAudio48kHz = try loadACEStepSourceAudio48kHz()
        let referenceAudio48kHz = try loadACEStepReferenceAudio48kHz()
        if effectiveTask.requiresSourceAudio && sourceAudio48kHz == nil {
            throw ValidationError("--source-audio is required for ACE-Step \(effectiveTask.rawValue).")
        }
        var effectiveDurationSeconds = resolvedACEStepDurationSeconds(
            task: effectiveTask,
            sourceAudio48kHz: sourceAudio48kHz,
            fallback: durationSeconds ?? qualityDefaults.fallbackDurationSeconds
        )
        let resolvedInstruction = resolveInstruction(
            task: effectiveTask,
            explicitInstruction: instruction,
            trackName: trackName,
            completeTrackClasses: completeTrackClasses
        )

        let shouldPlanDuration = effectiveUseLM
            && durationSeconds == nil
            && qualityDefaults.automaticDuration
            && !effectiveTask.locksDurationToSource
        var effectiveCaption = caption
        var userMetadata = ACEStep5HzLMConstrainedSampler.UserMetadata(
            bpm: bpm.map(String.init),
            caption: caption,
            duration: shouldPlanDuration
                ? nil
                : resolvedLMMetadataDuration(
                    effectiveDurationSeconds: effectiveDurationSeconds
                ),
            keyscale: keyscale,
            language: metadataLanguage,
            timesignature: timesignature
        )

        if !quiet {
            CLIStderr.write("Loading ACE-Step checkpoints from \(checkpointsRootURL.path)\n")
            if useLM && !effectiveUseLM {
                CLIStderr.write("Skipping 5Hz LM for ACE-Step \(effectiveTask.rawValue) task; upstream uses direct DiT conditioning for this task.\n")
            }
        }

        let container = ACEStepModelContainer(
            checkpointsRootURL: checkpointsRootURL,
            turboSubdirectory: resolvedTurboSubdirectory,
            vaeSubdirectory: vaeSubdirectory,
            lmSubdirectory: needsLMResources ? resolvedLMSubdirectory : nil,
            textEncoderSubdirectory: resolvedTextSubdirectory
        )
        let resources = try await container.resources()
        let pipeline = try ACEStepPipeline(
            decoderResources: resources.decoderResources,
            vaeResources: resources.vaeResources,
            lmResources: resources.lmResources,
            textEncoderResources: resources.textEncoderResources
        )
        let loadedAdapters = try loadAdapters(into: pipeline)

        if analyzeSourceAudio {
            guard let sourceAudio48kHz else {
                throw ValidationError("--analyze-source-audio requires --source-audio.")
            }
            if !quiet {
                CLIStderr.write("Analyzing ACE-Step source audio with 5Hz LM\n")
            }
            let sourceAnalysis = try pipeline.understandSourceAudio(
                sourceAudio48kHz: sourceAudio48kHz,
                durationSeconds: effectiveDurationSeconds,
                lmConfig: .init(maxNewTokens: 2048, temperature: 0.3, topK: lmTopK, topP: lmTopP)
            )
            let merge = mergedMetadataWithSourceAnalysis(userMetadata, sourceAnalysis.metadata)
            userMetadata = merge.metadata
            if !quiet {
                let summary = sourceAnalysis.metadata.understandingSummary
                CLIStderr.write("ACE-Step source analysis: \(summary)\n")
                if merge.filledFields.isEmpty {
                    CLIStderr.write("No missing ACE-Step metadata fields were filled from source analysis\n")
                } else {
                    CLIStderr.write("Filled ACE-Step metadata from source analysis: \(merge.filledFields.joined(separator: ", "))\n")
                }
            }
        }

        if effectiveUseLM && qualityDefaults.plansMetadata {
            if !quiet {
                CLIStderr.write("Planning ACE-Step \(quality.rawValue) metadata with the 5Hz LM\n")
            }
            let planningMetadata = ACEStep5HzLMConstrainedSampler.UserMetadata(
                bpm: userMetadata.bpm,
                caption: nil,
                duration: shouldPlanDuration ? nil : userMetadata.duration,
                keyscale: userMetadata.keyscale,
                language: userMetadata.language,
                timesignature: userMetadata.timesignature
            )
            let plan = try pipeline.planMusic(
                caption: caption,
                lyrics: resolvedLyrics,
                instruction: resolvedInstruction,
                userMetadata: planningMetadata,
                lmConfig: .init(
                    maxNewTokens: 1_024,
                    temperature: 0.85,
                    topK: lmTopK,
                    topP: lmTopP,
                    seed: seed
                )
            )
            if let plannedCaption = nonEmpty(plan.metadata.caption) {
                effectiveCaption = plannedCaption
            }
            if shouldPlanDuration, let plannedDuration = plan.metadata.durationSeconds {
                effectiveDurationSeconds = clampedAutomaticDuration(plannedDuration)
            }
            userMetadata = mergePlanMetadata(
                existing: userMetadata,
                plan: plan.metadata,
                caption: effectiveCaption,
                durationSeconds: effectiveDurationSeconds
            )
            if !quiet {
                CLIStderr.write(
                    "ACE-Step plan: \(plan.metadata.understandingSummary); "
                        + "duration=\(Int(effectiveDurationSeconds))s\n"
                )
            }
        }

        let inference = ACEStepInferenceConfig(
            durationSeconds: effectiveDurationSeconds,
            fixNFE: steps ?? qualityDefaults.inferenceSteps,
            shift: shift ?? qualityDefaults.shift,
            timesteps: nil,
            coverNoiseStrength: coverNoiseStrength,
            retakeSeed: retakeSeed,
            retakeVariance: retakeVariance,
            inferMethod: inferMethod ?? .ode,
            samplerMode: samplerMode ?? qualityDefaults.samplerMode,
            guidanceScale: guidanceScale ?? qualityDefaults.guidanceScale,
            guidanceMode: guidanceMode ?? .apg,
            cfgIntervalStart: resolvedCFGStart,
            cfgIntervalEnd: resolvedCFGEnd,
            velocityNormThreshold: velocityNormThreshold
                ?? qualityDefaults.velocityNormThreshold,
            velocityEMAFactor: velocityEMAFactor
                ?? qualityDefaults.velocityEMAFactor,
            useTiledVaeDecode: !noTiledVAE,
            vaeChunkSize: vaeChunkSize,
            vaeOverlap: vaeOverlap,
            seed: seed
        )
        let repaintConfiguration = effectiveTask == .repaint || effectiveTask == .lego
            ? ACEStepRepaintConfiguration(
                startSeconds: repaintStartSeconds,
                endSeconds: repaintEndSeconds,
                chunkMaskMode: chunkMaskMode,
                mode: repaintMode,
                strength: repaintStrength
            )
            : nil
        let flowEditConfiguration = flowEdit
            ? ACEStepFlowEditConfiguration(
                sourceCaption: sourceCaption ?? "",
                sourceLyrics: sourceLyrics,
                nMin: flowEditNMin,
                nMax: flowEditNMax,
                nAverage: flowEditNAverage,
                retakeSeed: retakeSeed
            )
            : nil

        let resolvedCandidateCount = candidateCount ?? qualityDefaults.candidateCount
        if !quiet {
            let mode = effectiveUseLM ? "constrained 5Hz LM + diffusion" : "direct prompt-to-audio diffusion"
            CLIStderr.write(
                "Running \(mode) in one warm session; candidates=\(resolvedCandidateCount)\n"
            )
        }
        let session = ACEStepGenerationSession(pipeline: pipeline)
        let ranked = try session.generateBest(
            ACEStepSessionRequest(
                caption: effectiveCaption,
                lyrics: resolvedLyrics,
                config: inference,
                lmConfig: .init(
                    maxNewTokens: 4_096,
                    temperature: 0.85,
                    topK: lmTopK,
                    topP: lmTopP
                ),
                lmUserMetadata: userMetadata,
                sourceAudio48kHz: sourceAudio48kHz,
                referenceTimbreAudio48kHz: referenceAudio48kHz,
                audioCoverStrength: audioCoverStrength,
                vocalLanguage: vocalLanguage,
                instruction: resolvedInstruction,
                task: effectiveTask,
                repaintConfiguration: repaintConfiguration,
                flowEditConfiguration: flowEditConfiguration,
                useLanguageModel: effectiveUseLM
            ),
            candidateCount: resolvedCandidateCount
        )

        let exportOptions = ACEStepAudioExportOptions(
            format: exportFormat,
            normalization: normalization,
            targetPeakDB: targetPeakDB,
            fadeInMilliseconds: fadeInMilliseconds,
            fadeOutMilliseconds: fadeOutMilliseconds,
            dither: !noDither
        )
        try ACEStepWAVWriter.writeWAV(
            ranked.best.audio,
            to: outputURL,
            sampleRate: 48_000,
            options: exportOptions
        )
        if keepCandidates {
            for candidate in ranked.candidates {
                let candidateURL = candidateOutputURL(
                    selectedOutputURL: outputURL,
                    rank: ranked.candidates.firstIndex { $0.index == candidate.index } ?? 0,
                    candidate: candidate
                )
                try ACEStepWAVWriter.writeWAV(
                    candidate.audio,
                    to: candidateURL,
                    sampleRate: 48_000,
                    options: exportOptions
                )
            }
        }

        var stemTracks: [ACEStepDAWBundleWriter.Track] = []
        for stemName in stemNames {
            if !quiet {
                CLIStderr.write("Extracting ACE-Step stem: \(stemName)\n")
            }
            var stemConfig = inference
            stemConfig.seed = ranked.best.seed
            let extracted = try session.generateBest(
                ACEStepSessionRequest(
                    caption: effectiveCaption,
                    lyrics: "",
                    config: stemConfig,
                    lmUserMetadata: userMetadata,
                    sourceAudio48kHz: ranked.best.audio,
                    vocalLanguage: vocalLanguage,
                    instruction: ACEStepTask.extract.instruction(
                        trackName: stemName
                    ),
                    task: .extract,
                    useLanguageModel: false
                ),
                candidateCount: 1
            ).best
            let stemURL = stemOutputURL(
                selectedOutputURL: outputURL,
                stemName: stemName
            )
            try ACEStepWAVWriter.writeWAV(
                extracted.audio,
                to: stemURL,
                sampleRate: 48_000,
                options: exportOptions
            )
            stemTracks.append(
                ACEStepDAWBundleWriter.Track(
                    name: stemName,
                    audio: extracted.audio
                )
            )
        }

        let synchronizedLyrics: ACEStepLRCDocument? = {
            if let inputLRC {
                return inputLRC
            }
            if lrcOutput != nil || dawBundle != nil, !resolvedLyrics.isEmpty {
                return .approximate(
                    lyrics: resolvedLyrics,
                    durationSeconds: Double(effectiveDurationSeconds)
                )
            }
            return nil
        }()
        let synchronizedLyricsURL: URL? = try {
            guard let synchronizedLyrics else {
                return nil
            }
            let url = lrcOutput.map(resolveUserPath)
                ?? outputURL.deletingPathExtension().appendingPathExtension("lrc")
            try synchronizedLyrics.rendered().write(
                to: url,
                atomically: true,
                encoding: .utf8
            )
            return url
        }()

        let recipeURL: URL? = try {
            guard !noRecipe else {
                return nil
            }
            let url = recipeOutput.map(resolveUserPath)
                ?? outputURL.deletingPathExtension()
                    .appendingPathExtension("recipe.json")
            let manifest = try MereRunModelManifest.loadIfPresent(
                from: checkpointsRootURL
            )
            let sourceSHA256 = try sourceAudio.map {
                try ModelArtifactPin.fileSHA256(resolveUserPath($0))
            }
            let recipe = ACEStepGenerationRecipe(
                schemaVersion: ACEStepGenerationRecipe.currentSchemaVersion,
                createdAt: Date(),
                modelID: model,
                checkpointVariant: checkpointVariant,
                decoderSubdirectory: resolvedTurboSubdirectory,
                checkpointSources:
                    ACEStepGenerationRecipe.checkpointProvenance(
                        modelID: model,
                        manifest: manifest
                    ),
                languageModelSubdirectory: resolvedLMSubdirectory,
                textEncoderSubdirectory: resolvedTextSubdirectory ?? "",
                adapters: loadedAdapters,
                task: effectiveTask,
                quality: quality,
                caption: effectiveCaption,
                lyrics: resolvedLyrics,
                instruction: resolvedInstruction,
                conditioningMetadata:
                    ACEStepRecipeConditioningMetadata(userMetadata),
                inference: inference,
                repaint: repaintConfiguration,
                flowEdit: flowEditConfiguration,
                languageModelUsed: effectiveUseLM,
                candidates: ranked.candidates.enumerated().map {
                    rank,
                    candidate in
                    ACEStepRecipeCandidate(
                        rank: rank + 1,
                        index: candidate.index,
                        seed: candidate.seed,
                        score: candidate.score,
                        metrics: candidate.metrics,
                        lmAudioCodeCount: candidate.lmAudioCodeCount,
                        selected: candidate.index == ranked.best.index
                    )
                },
                export: exportOptions,
                sourceAudioSHA256: sourceSHA256,
                outputFilename: outputURL.lastPathComponent,
                outputSHA256: try ModelArtifactPin.fileSHA256(outputURL),
                lrcFilename: synchronizedLyricsURL?.lastPathComponent,
                lrcTimingIsApproximate:
                    synchronizedLyrics?.timingIsApproximate
            )
            try recipe.write(to: url)
            return url
        }()

        if let dawBundle {
            guard let recipeURL else {
                throw ValidationError("DAW bundle requires a recipe.")
            }
            let directory = resolveUserPath(dawBundle)
            try ACEStepDAWBundleWriter.write(
                directory: directory,
                mixURL: outputURL,
                recipeURL: recipeURL,
                lrcURL: synchronizedLyricsURL,
                candidates: ranked.candidates.enumerated().map {
                    rank,
                    candidate in
                    ACEStepDAWBundleWriter.Track(
                        name: "Candidate \(rank + 1) seed \(candidate.seed)",
                        audio: candidate.audio
                    )
                },
                stems: stemTracks,
                lrc: synchronizedLyrics,
                exportOptions: exportOptions
            )
            if !quiet {
                CLIStderr.write("Saved DAW bundle: \(directory.path)\n")
            }
        }

        if !quiet {
            for (rank, candidate) in ranked.candidates.enumerated() {
                let selected = candidate.index == ranked.best.index ? " selected" : ""
                CLIStderr.write(
                    String(
                        format: "Candidate %d: seed=%llu score=%.2f%@\n",
                        rank + 1,
                        candidate.seed,
                        candidate.score,
                        selected
                    )
                )
            }
            CLIStderr.write("Saved audio: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    private func loadAdapters(
        into pipeline: ACEStepPipeline
    ) throws -> [ACEStepAdapterDescriptor] {
        guard !adapters.isEmpty else {
            if !adapterScales.isEmpty {
                throw ValidationError("--adapter-scale requires --adapter.")
            }
            return []
        }
        guard adapterScales.isEmpty
            || adapterScales.count == 1
            || adapterScales.count == adapters.count
        else {
            throw ValidationError(
                "--adapter-scale accepts one value for every adapter or one value per adapter."
            )
        }

        return try adapters.enumerated().map { index, path in
            let scale = adapterScales.isEmpty
                ? 1
                : adapterScales.count == 1
                    ? adapterScales[0]
                    : adapterScales[index]
            let report = try pipeline.loadAdapter(
                from: resolveUserPath(path),
                kind: adapterKind,
                scale: scale
            )
            if !quiet {
                CLIStderr.write(
                    "Loaded ACE-Step \(report.kind.rawValue) adapter "
                        + "\(report.filename) at scale \(report.scale) "
                        + "on \(report.matchedLayerCount) layers\n"
                )
            }
            return report
        }
    }

    private var isMagentaRT2Request: Bool {
        if MagentaRT2Resources.isMagentaRT2Model(model) {
            return true
        }
        let url = URL(fileURLWithPath: model).standardizedFileURL
        return MagentaRT2Resources.looksLikeMagentaRT2Root(url)
    }

    private func candidateOutputURL(
        selectedOutputURL: URL,
        rank: Int,
        candidate: ACEStepGeneratedCandidate
    ) -> URL {
        let stem = selectedOutputURL.deletingPathExtension().lastPathComponent
        let filename = "\(stem).candidate-\(rank + 1).seed-\(candidate.seed).wav"
        return selectedOutputURL
            .deletingLastPathComponent()
            .appendingPathComponent(filename)
    }

    var resolvedACEStepTask: ACEStepTask {
        if flowEdit {
            return .textToMusic
        }
        if nonCover && (taskType == .cover || taskType == .textToMusic) {
            return .coverNoFSQ
        }
        if taskType == .textToMusic,
           sourceAudio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
        {
            return .cover
        }
        return taskType
    }

    var resolvedACEStepIsCover: Bool {
        resolvedACEStepTask.usesFSQCoverHints
    }

    func resolvedACEStepUsesLM(task: ACEStepTask) -> Bool {
        guard !noLM, !task.skipsLanguageModel else {
            return false
        }
        return useLM || quality.defaults(for: .turbo, task: task).usesLanguageModel
    }

    private func runMagentaRT2() async throws {
        try validateMagentaRT2Options()

        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-magenta-rt2", defaultExtension: "wav")
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let resources = try await MagentaRT2Resources.resolve(
            requestedModel: model,
            progress: { event in
                guard !quiet else { return }
                switch event {
                case .downloading(let percent):
                    CLIStderr.write("Downloading Magenta RT2 assets... \(percent)%\n")
                case .extracting:
                    CLIStderr.write("Extracting Magenta RT2 assets...\n")
                }
            }
        )
        let controls = try magentaControls()
        if !quiet {
            CLIStderr.write("Loading Magenta RT2 model from \(resources.modelURL.path)\n")
        }
        let audio = try await MagentaRT2Renderer.render(
            MagentaRT2RenderRequest(
                prompt: caption,
                resources: resources,
                durationSeconds: durationSeconds ?? 10,
                controls: controls
            ),
            progress: { frame, total in
                guard !quiet, frame % 10 == 0 || frame + 1 == total else { return }
                CLIStderr.write("Generated Magenta RT2 frame \(frame + 1)/\(total)\n")
            }
        )
        let writer = try StreamingWAVWriter(
            outputURL: outputURL,
            sampleRate: MagentaRT2Resources.sampleRate,
            channels: MagentaRT2Resources.channels
        )
        try writer.append(samples: audio)
        try writer.close()
        if !quiet {
            CLIStderr.write("Saved audio: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    private func validateMagentaRT2Options() throws {
        if !lyrics.isEmpty || lyricsFile != nil {
            throw ValidationError("Magenta RT2 does not support --lyrics or --lyrics-file. Put musical direction in the prompt.")
        }
        if useLM {
            throw ValidationError("Magenta RT2 does not support --use-lm; that option is ACE-Step only.")
        }
        if noLM {
            throw ValidationError("Magenta RT2 does not support --no-lm; that option is ACE-Step only.")
        }
        if taskType != .textToMusic {
            throw ValidationError("Magenta RT2 does not support --task-type; that option is ACE-Step only.")
        }
        if trackName != nil
            || completeTrackClasses != nil
            || nonCover
            || sourceAudio != nil
            || coverNoiseStrength != 0.0
            || !referenceAudio.isEmpty
        {
            throw ValidationError("Magenta RT2 does not support ACE-Step cover/extract/lego options.")
        }
        if seed != nil {
            throw ValidationError("Magenta RT2 uses --seed-rotation instead of --seed.")
        }
        if steps != nil
            || shift != nil
            || inferMethod != nil
            || samplerMode != nil
            || guidanceScale != nil
            || guidanceMode != nil
            || cfgIntervalStart != nil
            || cfgIntervalEnd != nil
            || velocityNormThreshold != nil
            || velocityEMAFactor != nil
        {
            throw ValidationError("Magenta RT2 does not use ACE-Step diffusion or guidance options.")
        }
        _ = try magentaControls()
    }

    private func magentaControls() throws -> MagentaRT2Controls {
        guard magentaTopK >= 0 else {
            throw ValidationError("--top-k must be >= 0")
        }
        guard magentaUnmaskWidth >= 0 else {
            throw ValidationError("--unmask-width must be >= 0")
        }
        guard magentaPrefillDuration > 0 else {
            throw ValidationError("--prefill-duration must be > 0")
        }
        return MagentaRT2Controls(
            styleConditioning: magentaStyleConditioning,
            temperature: magentaTemperature,
            topK: Int32(magentaTopK),
            cfgMusicCoCa: magentaCFGMusicCoCa,
            cfgNotes: magentaCFGNotes,
            cfgDrums: magentaCFGDrums,
            drumless: magentaDrumless,
            unmaskWidth: Int32(magentaUnmaskWidth),
            seedRotation: Int32(magentaSeedRotation),
            prefillSilence: magentaPrefillSilence,
            prefillDurationSeconds: magentaPrefillDuration
        )
    }

    private func loadLyrics() throws -> String {
        if let lyricsFile {
            let lyricsURL = resolveUserPath(lyricsFile)
            guard FileManager.default.fileExists(atPath: lyricsURL.path) else {
                throw ValidationError("Lyrics file not found: \(lyricsURL.path)")
            }
            return try String(contentsOf: lyricsURL, encoding: .utf8)
        }
        return lyrics
    }

    private func loadLRC() throws -> ACEStepLRCDocument? {
        guard let lrcFile else {
            return nil
        }
        let url = resolveUserPath(lrcFile)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("LRC file not found: \(url.path)")
        }
        do {
            return try ACEStepLRCDocument.parse(
                String(contentsOf: url, encoding: .utf8)
            )
        } catch {
            throw ValidationError(
                "Invalid LRC file \(url.path): \(error.localizedDescription)"
            )
        }
    }

    private func parsedStemNames() -> [String] {
        stems?
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            ?? []
    }

    private func stemOutputURL(
        selectedOutputURL: URL,
        stemName: String
    ) -> URL {
        let safe = stemName.lowercased().map { character -> Character in
            character.isLetter || character.isNumber ? character : "-"
        }
        let stem = selectedOutputURL.deletingPathExtension().lastPathComponent
        return selectedOutputURL.deletingLastPathComponent()
            .appendingPathComponent("\(stem).stem-\(String(safe)).wav")
    }

    private func loadACEStepSourceAudio48kHz() throws -> MLXArray? {
        guard let sourceAudio,
              !sourceAudio.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return nil
        }
        let audio = try loadACEStepAudio48kHz(sourceAudio, label: "--source-audio")
        if !quiet {
            CLIStderr.write("Using ACE-Step source audio: \(resolveUserPath(sourceAudio).path)\n")
        }
        return audio
    }

    private func loadACEStepReferenceAudio48kHz() throws -> [MLXArray]? {
        let paths = referenceAudio
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !paths.isEmpty else {
            return nil
        }
        let audios = try paths.map { try loadACEStepAudio48kHz($0, label: "--reference-audio") }
        if !quiet {
            CLIStderr.write("Using \(audios.count) ACE-Step reference audio file(s)\n")
        }
        return audios
    }

    private func loadACEStepAudio48kHz(_ path: String, label: String) throws -> MLXArray {
        try ACEStepCLIHelper.loadAudio48kHz(path, label: label)
    }

    private func resolveUserPath(_ path: String) -> URL {
        ACEStepCLIHelper.resolveUserPath(path)
    }

    func resolvedACEStepDurationSeconds(
        task: ACEStepTask,
        sourceAudio48kHz: MLXArray?,
        fallback: Float? = nil
    ) -> Float {
        let requestedDuration = fallback ?? durationSeconds
            ?? quality.defaults(for: .turbo, task: task).fallbackDurationSeconds
        guard task.locksDurationToSource,
              let sourceAudio48kHz,
              sourceAudio48kHz.ndim >= 2
        else {
            return requestedDuration
        }
        let sourceFrames = sourceAudio48kHz.dim(1)
        guard sourceFrames > 0 else {
            return requestedDuration
        }
        return Float(sourceFrames) / 48_000.0
    }

    func resolvedMetadataDuration(effectiveDurationSeconds: Float) -> String {
        if let metadataDuration, !metadataDuration.isEmpty {
            return metadataDuration
        }
        let seconds = max(1, Int(effectiveDurationSeconds))
        return "\(seconds) seconds"
    }

    func resolvedLMMetadataDuration(effectiveDurationSeconds: Float) -> String {
        if let metadataDuration, !metadataDuration.isEmpty {
            return metadataDuration
        }
        return String(max(1, Int(effectiveDurationSeconds)))
    }

    func clampedAutomaticDuration(_ duration: Float) -> Float {
        let upperBound: Float = quality == .song ? 240 : 600
        return min(max(duration, 10), upperBound)
    }

    private func mergePlanMetadata(
        existing: ACEStep5HzLMConstrainedSampler.UserMetadata,
        plan: ACEStepMusicUnderstandingMetadata,
        caption: String,
        durationSeconds: Float
    ) -> ACEStep5HzLMConstrainedSampler.UserMetadata {
        ACEStep5HzLMConstrainedSampler.UserMetadata(
            bpm: nonEmpty(existing.bpm) ?? plan.bpm.map(String.init),
            caption: caption,
            duration: resolvedLMMetadataDuration(
                effectiveDurationSeconds: durationSeconds
            ),
            keyscale: nonEmpty(existing.keyscale) ?? nonEmpty(plan.keyscale),
            language: nonEmpty(existing.language) ?? nonEmpty(plan.language),
            timesignature: nonEmpty(existing.timesignature)
                ?? nonEmpty(plan.timesignature)
        )
    }

    private func nonEmpty(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }

    func mergedMetadataWithSourceAnalysis(
        _ metadata: ACEStep5HzLMConstrainedSampler.UserMetadata,
        _ analysis: ACEStepMusicUnderstandingMetadata
    ) -> (metadata: ACEStep5HzLMConstrainedSampler.UserMetadata, filledFields: [String]) {
        var filledFields: [String] = []

        let analyzedBPM = analysis.bpm.map(String.init)
        let mergedBPM = fillMissing(metadata.bpm, with: analyzedBPM, field: "bpm", filledFields: &filledFields)
        let mergedKeyscale = fillMissing(metadata.keyscale, with: analysis.keyscale, field: "keyscale", filledFields: &filledFields)
        let mergedLanguage = fillMissing(metadata.language, with: analysis.language, field: "language", filledFields: &filledFields)
        let mergedTimeSignature = fillMissing(
            metadata.timesignature,
            with: analysis.timesignature,
            field: "timesignature",
            filledFields: &filledFields
        )

        return (
            ACEStep5HzLMConstrainedSampler.UserMetadata(
                bpm: mergedBPM,
                caption: metadata.caption,
                duration: metadata.duration,
                keyscale: mergedKeyscale,
                language: mergedLanguage,
                timesignature: mergedTimeSignature
            ),
            filledFields
        )
    }

    private func fillMissing(
        _ existing: String?,
        with analyzed: String?,
        field: String,
        filledFields: inout [String]
    ) -> String? {
        let trimmedExisting = existing?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !trimmedExisting.isEmpty {
            return existing
        }
        let trimmedAnalyzed = analyzed?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmedAnalyzed.isEmpty else {
            return existing
        }
        filledFields.append(field)
        return trimmedAnalyzed
    }

    private func resolveInstruction(
        task: ACEStepTask,
        explicitInstruction: String,
        trackName: String?,
        completeTrackClasses: String?
    ) -> String {
        let defaultInstruction = "Fill the audio semantic mask based on the given conditions:"
        let trimmed = explicitInstruction.trimmingCharacters(in: .whitespacesAndNewlines)
        let taskInstruction = task.instruction(
            trackName: trackName,
            completeTrackClasses: parseCompleteTrackClasses(completeTrackClasses)
        )

        let shouldAutoMapInstruction = trimmed.isEmpty || trimmed == defaultInstruction
        return shouldAutoMapInstruction ? taskInstruction : trimmed
    }

    private func parseCompleteTrackClasses(_ value: String?) -> [String] {
        let rawItems = value?
            .split { $0 == "," || $0 == "|" || $0 == "/" }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            ?? []
        return rawItems.filter { !$0.isEmpty }
    }

    private func resolveACEStepCheckpointsRoot() async throws -> URL {
        try await ACEStepCLIHelper.resolveCheckpointsRoot(
            model: model,
            checkpointsRoot: checkpointsRoot,
            turboSubdirectory: turboSubdirectory,
            vaeSubdirectory: vaeSubdirectory,
            lmSubdirectory: useLM ? lmSubdirectory : nil,
            textSubdirectory: textSubdirectory
        )
    }

    private func resolveACEStepLMSubdirectory(at root: URL, explicit: String?) throws -> String? {
        try ACEStepCLIHelper.resolveLMSubdirectory(at: root, explicit: explicit)
    }

    private func resolveACEStepTextSubdirectory(at root: URL, explicit: String?) throws -> String? {
        try ACEStepCLIHelper.resolveTextSubdirectory(at: root, explicit: explicit)
    }

    func buildAcestepCheckpointCandidates() -> [URL] {
        ACEStepCLIHelper.buildCheckpointCandidates(model: model, checkpointsRoot: checkpointsRoot)
    }

    private func resolveACEStepTurboSubdirectory(at root: URL, explicit: String) throws -> String {
        try ACEStepCLIHelper.resolveTurboSubdirectory(at: root, explicit: explicit)
    }
}
