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

          mere.run music generate "cinematic synth-pop, soaring female vocal" \
            --model music-minimax-music3 \
            --lyrics "[verse]\nwe turn the dark into gold\n[chorus]\nwe are electric" \
            --duration 30 \
            -o minimax.wav

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

    @Flag(name: [.customLong("instrumental")], help: "Use upstream's [Instrumental] lyric marker; cannot be combined with lyrics.")
    var instrumental: Bool = false

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

    @Option(name: [.customShort("m"), .long], help: "Managed music model id or local model root.")
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

    @Option(name: [.customLong("lm-subdirectory")], help: "Legacy 5Hz LM subdirectory under the selected checkpoint root.")
    var lmSubdirectory: String?

    @Option(
        name: [.customLong("lm-model")],
        help: "Managed 5Hz LM model id or local planner root. Defaults to music-acestep-lm-1.7b when the selected checkpoint has no LM."
    )
    var lmModel: String?

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

    @Option(name: [.customShort("s"), .long], help: "Denoise steps (MiniMax Music 3: 30; ACE-Step Turbo: 8; Base/SFT: 50).")
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
        help: "Generate and rank this many warm-session candidates (default: 1; upstream auto-scoring is opt-in)."
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

    @Option(name: [.customLong("lm-temperature")], help: "LM sampling temperature from 0.0 to 2.0.")
    var lmTemperature: Float = 0.85

    @Option(
        name: [.customLong("lm-repetition-penalty")],
        help: "LM repetition penalty (> 0; 1.0 = disabled)."
    )
    var lmRepetitionPenalty: Float = 1.0

    @Option(name: [.customLong("lm-cfg-scale")], help: "LM classifier-free guidance for semantic audio codes (upstream default: 2.0).")
    var lmCFGScale: Float = 2.0

    @Option(name: [.customLong("lm-negative-prompt")], help: "LM unconditional prompt used when --lm-cfg-scale is above 1.")
    var lmNegativePrompt: String = "NO USER INPUT"

    @Flag(
        name: [.customLong("no-lm-caption-rewrite")],
        help: "Keep the input caption authoritative while still using the LM for metadata and semantic audio codes (upstream use_cot_caption=false)."
    )
    var noLMCaptionRewrite: Bool = false

    @Option(
        name: [.customLong("metadata-duration")],
        help: "Compatibility alias for --duration; explicit duration always controls planning and rendering."
    )
    var metadataDuration: String?

    @Option(name: [.customLong("metadata-language")], help: "Compatibility override for --vocal-language.")
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

        let explicitDurationSeconds = try resolvedExplicitDurationSeconds()
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
        guard (0.0...2.0).contains(lmTemperature) else {
            throw ValidationError("--lm-temperature must be between 0.0 and 2.0")
        }
        guard lmRepetitionPenalty > 0 else {
            throw ValidationError("--lm-repetition-penalty must be > 0")
        }
        guard lmCFGScale >= 1, lmCFGScale.isFinite else {
            throw ValidationError("--lm-cfg-scale must be >= 1")
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
        if instrumental, lrcFile != nil || lyricsFile != nil || !lyrics.isEmpty {
            throw ValidationError("--instrumental cannot be combined with lyrics options.")
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

        if isMiniMaxMusic3Request {
            try runMiniMaxMusic3(explicitDurationSeconds: explicitDurationSeconds)
            return
        }

        if isMagentaRT2Request {
            try await runMagentaRT2()
            return
        }

        let effectiveTask = resolvedACEStepTask
        let effectiveUseLM = flowEdit
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
        let resolvedLM = try await wantsLMResources
            ? ACEStepCLIHelper.resolveLMResources(
                checkpointsRoot: checkpointsRootURL,
                lmModel: lmModel,
                lmSubdirectory: lmSubdirectory
            )
            : nil
        let resolvedTextSubdirectory = try resolveACEStepTextSubdirectory(
            at: checkpointsRootURL,
            explicit: textSubdirectory
        )
        let needsLMResources = effectiveUseLM || analyzeSourceAudio
        if resolvedTextSubdirectory == nil {
            throw ValidationError("ACE-Step text encoder not found. Set --text-subdirectory or keep a default layout like 'Qwen3-Embedding-0.6B'.")
        }
        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-music", defaultExtension: "wav")
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let inputLRC = try loadLRC()
        let resolvedLyrics = instrumental
            ? "[Instrumental]"
            : try inputLRC?.lyrics ?? loadLyrics()
        let effectiveLMRepetitionPenalty = lmRepetitionPenalty == 1
            ? nil
            : lmRepetitionPenalty
        let sourceAudio48kHz = try loadACEStepSourceAudio48kHz()
        let referenceAudio48kHz = try loadACEStepReferenceAudio48kHz()
        if effectiveTask.requiresSourceAudio && sourceAudio48kHz == nil {
            throw ValidationError("--source-audio is required for ACE-Step \(effectiveTask.rawValue).")
        }
        var effectiveDurationSeconds = resolvedACEStepDurationSeconds(
            task: effectiveTask,
            sourceAudio48kHz: sourceAudio48kHz,
            fallback: explicitDurationSeconds ?? qualityDefaults.fallbackDurationSeconds
        )
        let resolvedInstruction = resolveInstruction(
            task: effectiveTask,
            explicitInstruction: instruction,
            trackName: trackName,
            completeTrackClasses: completeTrackClasses
        )

        let shouldPlanDuration = effectiveUseLM
            && explicitDurationSeconds == nil
            && qualityDefaults.automaticDuration
            && !effectiveTask.locksDurationToSource
        var effectiveCaption = caption
        var lmCodeGenerationContext: ACEStepLMCodeGenerationContext?
        let effectiveLanguage = ACEStepPlanningPolicy.effectiveLanguage(
            vocalLanguage: vocalLanguage,
            metadataLanguage: metadataLanguage
        )
        var userMetadata = ACEStep5HzLMConstrainedSampler.UserMetadata(
            bpm: bpm.map(String.init),
            caption: caption,
            duration: shouldPlanDuration
                ? nil
                : resolvedLMMetadataDuration(
                    effectiveDurationSeconds: effectiveDurationSeconds
                ),
            keyscale: keyscale,
            language: effectiveLanguage,
            timesignature: timesignature
        )

        if !quiet {
            CLIStderr.write("Loading ACE-Step checkpoints from \(checkpointsRootURL.path)\n")
            if let resolvedLM {
                CLIStderr.write(
                    "Using ACE-Step planner \(resolvedLM.source) from \(resolvedLM.rootURL.path)\n"
                )
            }
            if useLM && !effectiveUseLM {
                CLIStderr.write("Skipping 5Hz LM for ACE-Step \(effectiveTask.rawValue) task; upstream uses direct DiT conditioning for this task.\n")
            }
        }

        let container = ACEStepModelContainer(
            decoderRootURL: checkpointsRootURL.appendingPathComponent(
                resolvedTurboSubdirectory,
                isDirectory: true
            ),
            vaeRootURL: checkpointsRootURL.appendingPathComponent(
                vaeSubdirectory,
                isDirectory: true
            ),
            lmRootURL: needsLMResources ? resolvedLM?.rootURL : nil,
            textEncoderRootURL: resolvedTextSubdirectory.map {
                checkpointsRootURL.appendingPathComponent($0, isDirectory: true)
            }
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
                lmConfig: .init(
                    maxNewTokens: 2_048,
                    temperature: 0.3,
                    topK: lmTopK,
                    topP: lmTopP,
                    repetitionPenalty: effectiveLMRepetitionPenalty
                )
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

        if effectiveUseLM {
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
                useCotCaption: !noLMCaptionRewrite,
                lmConfig: .init(
                    maxNewTokens: 1_024,
                    temperature: lmTemperature,
                    topK: lmTopK,
                    topP: lmTopP,
                    repetitionPenalty: effectiveLMRepetitionPenalty,
                    seed: seed
                )
            )
            lmCodeGenerationContext = plan.codeGenerationContext
            if !noLMCaptionRewrite,
               let plannedCaption = nonEmpty(plan.metadata.caption)
            {
                effectiveCaption = plannedCaption
            }
            if shouldPlanDuration, let plannedDuration = plan.metadata.durationSeconds {
                effectiveDurationSeconds = clampedAutomaticDuration(plannedDuration)
            }
            userMetadata = ACEStepPlanningPolicy.merge(
                userMetadata: userMetadata,
                plan: plan.metadata,
                caption: effectiveCaption,
                durationSeconds: effectiveDurationSeconds
            )
            lmCodeGenerationContext = lmCodeGenerationContext?.applying(
                userMetadata: userMetadata
            )
            if !quiet {
                CLIStderr.write(
                    "ACE-Step effective plan: "
                        + "\(ACEStepPlanningPolicy.summary(userMetadata))\n"
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
                    temperature: lmTemperature,
                    topK: lmTopK,
                    topP: lmTopP,
                    repetitionPenalty: effectiveLMRepetitionPenalty,
                    cfgScale: lmCFGScale,
                    negativePrompt: lmNegativePrompt
                ),
                lmUserMetadata: userMetadata,
                lmCodeGenerationContext: lmCodeGenerationContext,
                sourceAudio48kHz: sourceAudio48kHz,
                referenceTimbreAudio48kHz: referenceAudio48kHz,
                audioCoverStrength: audioCoverStrength,
                vocalLanguage: effectiveLanguage,
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
                    vocalLanguage: effectiveLanguage,
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
                languageModelSubdirectory: resolvedLM?.rootURL.lastPathComponent,
                languageModelSource: resolvedLM?.source,
                languageModelRoot: resolvedLM?.rootURL.path,
                languageModelSources: resolvedLM.map {
                    ACEStepGenerationRecipe.languageModelProvenance(
                        source: $0.source,
                        subdirectory: $0.rootURL.lastPathComponent,
                        checkpointModelID: model,
                        checkpointManifest: manifest
                    )
                } ?? [],
                textEncoderSubdirectory: resolvedTextSubdirectory ?? "",
                adapters: loadedAdapters,
                task: effectiveTask,
                quality: quality,
                inputCaption: caption,
                caption: effectiveCaption,
                lyrics: resolvedLyrics,
                instruction: resolvedInstruction,
                languageModelReasoning: lmCodeGenerationContext?.reasoning,
                conditioningMetadata:
                    ACEStepRecipeConditioningMetadata(userMetadata),
                languageModelSampling: effectiveUseLM
                    ? ACEStepRecipeLMSampling(
                        temperature: lmTemperature,
                        topK: lmTopK,
                        topP: lmTopP,
                        repetitionPenalty: lmRepetitionPenalty,
                        cfgScale: lmCFGScale,
                        negativePrompt: lmNegativePrompt,
                        useCotCaption: !noLMCaptionRewrite
                    )
                    : nil,
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

    private var isMiniMaxMusic3Request: Bool {
        if model == ModelResolver.ModelID.miniMaxMusic3.rawValue {
            return true
        }
        return MiniMaxMusic3Resources.looksLikeRoot(resolveUserPath(model))
    }

    private func runMiniMaxMusic3(explicitDurationSeconds: Float?) throws {
        try validateMiniMaxMusic3Options(explicitDurationSeconds: explicitDurationSeconds)
        let inputLRC = try loadLRC()
        let resolvedLyrics = instrumental
            ? "[Instrumental]"
            : try inputLRC?.lyrics ?? loadLyrics()
        guard !resolvedLyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError(
                "MiniMax Music 3 requires --lyrics, --lyrics-file, --lrc-file, or --instrumental."
            )
        }

        let rootURL: URL
        if model == ModelResolver.ModelID.miniMaxMusic3.rawValue {
            do {
                rootURL = try ModelResolver().resolve(.miniMaxMusic3).rootURL
            } catch {
                throw ValidationError(
                    "MiniMax Music 3 is not installed. Review its license, then run "
                        + "`mere.run model pull \(MiniMaxMusic3Resources.modelID) --accept-model-license`."
                )
            }
        } else {
            rootURL = resolveUserPath(model)
        }
        let resources = MiniMaxMusic3Resources(rootURL: rootURL)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw ValidationError(
                "Incomplete MiniMax Music 3 root at \(rootURL.path): "
                    + missing.map(\.lastPathComponent).joined(separator: ", ")
            )
        }

        let outputURL = CLIOutput.resolveOutputURL(
            output,
            defaultPrefix: "mererun-minimax-music3",
            defaultExtension: "wav"
        )
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if !quiet {
            CLIStderr.write("Loading MiniMax Music 3 from \(rootURL.path)\n")
        }
        let pipeline = try MiniMaxMusic3Pipeline(resources: resources)
        let result = try pipeline.generate(
            options: MiniMaxMusic3GenerationOptions(
                caption: caption,
                lyrics: resolvedLyrics,
                durationSeconds: explicitDurationSeconds ?? 60,
                inferenceSteps: steps ?? 30,
                seed: seed ?? 0,
                guidanceScale: guidanceScale ?? 1.7
            ),
            progress: { event in
                guard !quiet else { return }
                switch event {
                case .semantic(let frame, let maximum):
                    if frame == 1 || frame == maximum || frame % 25 == 0 {
                        CLIStderr.write("MiniMax semantic frames: \(frame)/\(maximum)\n")
                    }
                case .denoise(let chunk, let chunkCount, let step, let stepCount):
                    if step == 1 || step == stepCount || step % 5 == 0 {
                        CLIStderr.write(
                            "MiniMax flow chunk \(chunk)/\(chunkCount): step \(step)/\(stepCount)\n"
                        )
                    }
                case .decode(let chunk, let chunkCount):
                    CLIStderr.write("MiniMax vocoder chunk: \(chunk)/\(chunkCount)\n")
                }
            }
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
            result.waveform.transposed(0, 2, 1),
            to: outputURL,
            sampleRate: result.sampleRate,
            options: exportOptions
        )
        if !noRecipe {
            let recipeURL = recipeOutput.map(resolveUserPath)
                ?? outputURL.deletingPathExtension().appendingPathExtension("recipe.json")
            let recipe = MiniMaxMusic3GenerationRecipe(
                schemaVersion: MiniMaxMusic3GenerationRecipe.currentSchemaVersion,
                createdAt: Date(),
                modelID: model,
                sourceRepository: MiniMaxMusic3Resources.repository,
                sourceRevision: MiniMaxMusic3Resources.revision,
                caption: caption,
                lyrics: resolvedLyrics,
                durationSeconds: explicitDurationSeconds ?? 60,
                generatedFrameCount: result.frameCount,
                inferenceSteps: steps ?? 30,
                seed: seed ?? 0,
                guidanceScale: guidanceScale ?? 1.7,
                sampleRate: result.sampleRate,
                export: exportOptions,
                outputFilename: outputURL.lastPathComponent,
                outputSHA256: try ModelArtifactPin.fileSHA256(outputURL)
            )
            try recipe.write(to: recipeURL)
            if !quiet {
                CLIStderr.write("Saved recipe: \(recipeURL.path)\n")
            }
        }
        if !quiet {
            CLIStderr.write(
                "Generated \(result.frameCount) MiniMax frames at \(result.sampleRate) Hz stereo\n"
            )
            CLIStderr.write("Saved audio: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    private func validateMiniMaxMusic3Options(explicitDurationSeconds: Float?) throws {
        if let explicitDurationSeconds, explicitDurationSeconds > 360 {
            throw ValidationError("MiniMax Music 3 supports at most 360 seconds (9,000 frames at 25 Hz).")
        }
        if useLM || noLM || analyzeSourceAudio || lmModel != nil || lmSubdirectory != nil {
            throw ValidationError("MiniMax Music 3 uses its built-in autoregressive stage; ACE-Step LM options do not apply.")
        }
        if taskType != .textToMusic
            || sourceAudio != nil
            || !referenceAudio.isEmpty
            || nonCover
            || flowEdit
            || trackName != nil
            || completeTrackClasses != nil
        {
            throw ValidationError("MiniMax Music 3 currently supports text-and-lyrics generation only.")
        }
        if !adapters.isEmpty || !adapterScales.isEmpty || stems != nil {
            throw ValidationError("MiniMax Music 3 does not support ACE-Step adapters or stem extraction.")
        }
        if shift != nil
            || inferMethod != nil
            || samplerMode != nil
            || guidanceMode != nil
            || cfgIntervalStart != nil
            || cfgIntervalEnd != nil
            || velocityNormThreshold != nil
            || velocityEMAFactor != nil
        {
            throw ValidationError("MiniMax Music 3 uses its fixed flow-matching Euler schedule.")
        }
        if let candidateCount, candidateCount != 1 {
            throw ValidationError("MiniMax Music 3 currently supports one candidate per invocation.")
        }
        if keepCandidates || dawBundle != nil || lrcOutput != nil {
            throw ValidationError("Candidate, DAW-bundle, and LRC export are not yet available for MiniMax Music 3.")
        }
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
        let seconds = max(1, Int(effectiveDurationSeconds))
        return "\(seconds) seconds"
    }

    func resolvedLMMetadataDuration(effectiveDurationSeconds: Float) -> String {
        return String(max(1, Int(effectiveDurationSeconds)))
    }

    func resolvedExplicitDurationSeconds() throws -> Float? {
        let legacyDuration: Float? = try metadataDuration.flatMap { value in
            let parts = value
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
                .split(whereSeparator: { $0.isWhitespace })
            guard let first = parts.first, let parsed = Float(first) else {
                throw ValidationError("--metadata-duration must be a number of seconds.")
            }
            let units = parts.dropFirst().joined(separator: " ")
            guard units.isEmpty || ["s", "sec", "second", "seconds"].contains(units) else {
                throw ValidationError("--metadata-duration must be a number of seconds.")
            }
            return parsed
        }
        if let durationSeconds, let legacyDuration,
           abs(durationSeconds - legacyDuration) > 0.001
        {
            throw ValidationError("--duration and --metadata-duration must match when both are provided.")
        }
        let resolved = durationSeconds ?? legacyDuration
        if let resolved, resolved <= 0 || resolved > 600 {
            throw ValidationError("--duration must be greater than 0 and at most 600 seconds.")
        }
        return resolved
    }

    func clampedAutomaticDuration(_ duration: Float) -> Float {
        let upperBound: Float = quality == .song ? 240 : 600
        return min(max(duration, 10), upperBound)
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
