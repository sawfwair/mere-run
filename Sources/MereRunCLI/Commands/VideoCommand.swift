import ArgumentParser
import Foundation
import MLX
import MereRunContract
import MereRunCore

enum MiniMaxH3CLITransformerWeightMode: String, CaseIterable, ExpressibleByArgument {
    case automatic = "auto"
    case quantized
    case residentBF16 = "resident-bf16"

    var generationMode: MiniMaxH3TransformerWeightMode {
        switch self {
        case .automatic: .automatic
        case .quantized: .quantized
        case .residentBF16: .residentBF16
        }
    }
}

enum MiniMaxH3CLIAccelerationMode: String, CaseIterable, ExpressibleByArgument {
    case quality
    case balanced
    case maximum
    case layers45 = "layers-45"
    case layers40 = "layers-40"
    case velocityReuse2 = "velocity-reuse-2"
    case tokenReduction = "token-reduction"

    var generationMode: MiniMaxH3AccelerationMode {
        switch self {
        case .quality: .quality
        case .balanced: .balanced
        case .maximum: .maximum
        case .layers45: .layers45
        case .layers40: .layers40
        case .velocityReuse2: .velocityReuse2
        case .tokenReduction: .tokenReduction
        }
    }
}

struct Video: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "video",
        abstract: "Generate and understand video with native Swift/MLX pipelines.",
        subcommands: [
            VideoAnimate.self,
            VideoCosmos3.self,
            VideoDubIt.self,
            VideoExportLatents.self,
            VideoGenerate.self,
            VideoPrepareMasks.self,
            VideoRetake.self,
            VideoSession.self
        ]
    )
}

struct VideoExportLatents: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "export-latents",
        abstract: "Run native Swift/MLX distilled LTX denoising and export final latents.",
        discussion: """
        Generates stage-2 distilled latents using native Swift/MLX.

        Expected model layout:
          <model-root>/text_encoder/config.json
          <model-root>/text_encoder/model.safetensors.index.json
          <model-root>/tokenizer/*
          <model-root>/ltx-2-19b-distilled.safetensors
          <model-root>/ltx-2-spatial-upscaler-x2-1.0.safetensors

        Example:
          swift run mere.run video export-latents \\
            --model video-ltx-av \\
            -o out.safetensors \\
            "a cinematic drone flyover at sunrise"
        """
    )

    @Argument(help: "Prompt for latent generation.")
    var prompt: String

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local path to the LTX model root.")
    var model: String = ModelResolver.ModelID.ltxVideoAV.rawValue

    @Option(name: [.customLong("model-root")], help: "Local path to the distilled LTX model root. Takes precedence over --model.")
    var modelRoot: String?

    @Option(name: [.customShort("o"), .long], help: "Output safetensors path for final stage latents.")
    var output: String?

    @Option(name: [.long], help: "Output width (must be divisible by 64).")
    var width: Int = 768

    @Option(name: [.long], help: "Output height (must be divisible by 64).")
    var height: Int = 512

    @Option(name: [.customLong("num-frames")], help: "Frame count (must satisfy 8n+1).")
    var numFrames: Int = 65

    @Option(name: [.long], help: "Seed value.")
    var seed: Int = 42

    @Flag(name: [.short, .long], help: "Quiet mode.")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        let rootURL = try await resolveVideoModelRoot(
            explicitModelRoot: modelRoot,
            requestedModel: model,
            variant: .distilled,
            allowAutoDownload: true
        )
        try validateNativeModelRoot(rootURL)
        guard !isLTX23SplitModelRoot(rootURL) else {
            throw ValidationError(
                """
                LTX 2.3 split MLX model roots are recognized, but `video export-latents` still requires the older \
                `video-ltx-av` merged LTX layout.
                """
            )
        }

        let upsamplerWeights = rootURL.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors", isDirectory: false)
        guard FileManager.default.fileExists(atPath: upsamplerWeights.path) else {
            throw ValidationError("Missing upsampler weights: \(upsamplerWeights.path)")
        }

        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-video-latents", defaultExtension: "safetensors")
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let generator = LTXDistilledLatentGenerator()
        try await generator.load(modelRoot: rootURL)
        let result = try await generator.generate(
            options: LTXDistilledLatentGenerationOptions(
                prompt: prompt,
                width: width,
                height: height,
                numFrames: numFrames,
                fps: 24,
                seed: seed
            )
        )
        await generator.unload()

        try MLX.save(array: result.latents, url: outputURL)

        if !quiet {
            CLIStderr.write("Model root: \(rootURL.path)\n")
            CLIStderr.write("Final latent shape: \(result.latents.shape)\n")
            CLIStderr.write("Stage1 latent shape: \(result.stage1Latents.shape)\n")
            CLIStderr.write("Saved: \(outputURL.path)\n")
        }

        print(outputURL.path)
    }
}

struct VideoGenerate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate MP4 video with native Swift/MLX video models.",
        discussion: """
        Prints the output MP4 path to stdout.
        Progress and diagnostics are printed to stderr.

        Quality and output are separate choices. The default is a fast draft
        checkpoint with video-only output. Use an official LTX 2.5 catalog id
        for distilled, full, HQ, keyframe, DFR, IC-LoRA, HDR, and generated-AV
        workflows. --output-mode audio-video requests synchronized generated
        audio. Supplying --audio selects native two-stage A2Vid and preserves
        the chosen source segment as the soundtrack.

        --variant distilled|unified-av remains available for compatibility. Do
        not combine it with --quality or --output-mode.

        Examples:
          swift run mere.run video generate "a cinematic drone flythrough over snowy mountains" --num-frames 65
          swift run mere.run video generate "woman walking in neon rain" --image frame.png
          swift run mere.run video generate "a car drives from dawn into sunset" --image start.png --end-image end.png
          swift run mere.run video generate "a cinematic final shot" --quality final --duration 4
          swift run mere.run video generate "dialogue with clean background music" --quality final --output-mode audio-video --duration 15 --fps 24
          swift run mere.run video generate "a kinetic live performance" --audio song.wav --audio-start-time 30 --duration 5 --image performer.png
          swift run mere.run video generate "the camera walks forward" --model video-wan22-ti2v-5b-mlx --image frame.png --num-frames 41 --width 1280 --height 704
          swift run mere.run video generate "use this subject and motion" --model video-minimax-h3-ref2va-mlx --reference image:subject.png --reference video:motion.mp4
          swift run mere.run video generate "one continuous tracking shot" --model minimax-h3-fl2va-bf16-mlx --duration 15 --h3-window-frames 124 --h3-window-overlap 35
          swift run mere.run video generate "the actor crosses three sets" --model minimax-h3-fl2va-bf16-mlx --h3-frame 72:second-set.png --h3-frame 144:third-set.png
        """
    )

    @Argument(help: "Prompt for video generation.")
    var prompt: String

    @Option(name: [.customShort("o"), .long], help: "Output MP4 path (default: ./mererun-video-<timestamp>.mp4).")
    var output: String?

    @Option(name: [.customShort("m"), .long], help: "Managed video model id or local model root. Defaults by operation.")
    var model: String = ""

    @Option(name: [.customLong("quality")], help: "LTX checkpoint quality: draft uses LTX 2.3 standalone distilled; final uses LTX 2.3 dev + distilled-LoRA or LTX 2.5 distilled.")
    var quality: LTXVideoQuality?

    @Option(name: [.customLong("output-mode")], help: "LTX deliverable: video-only or synchronized audio-video.")
    var outputMode: LTXVideoOutputMode?

    @Option(name: [.customLong("variant")], help: "Compatibility selector: distilled defaults to draft video-only; unified-av defaults to final audio-video.")
    var legacyVariant: LTXVideoVariant?

    @Option(name: [.customLong("model-root")], help: "Local video model root. Takes precedence over --model.")
    var modelRoot: String?

    @Option(
        name: [.long],
        help: "Output width. Omitted LTX 2.5 requests use the selected upstream recipe's native canvas."
    )
    var width: Int?

    @Option(
        name: [.long],
        help: "Output height. Omitted LTX 2.5 requests use the selected upstream recipe's native canvas."
    )
    var height: Int?

    @Option(
        name: [.customLong("num-frames")],
        help: "Frame count. LTX 2.5 predicts duration in the official 1...20s range when omitted."
    )
    var numFrames: Int?

    @Option(name: [.long], help: "Target output duration in seconds. Overrides --num-frames using the selected model's native cadence.")
    var duration: Double?

    @Option(
        name: [.customLong("auto-duration")],
        parsing: .upToNextOption,
        help: "LTX 2.5 DurationHead range as MIN_SECONDS MAX_SECONDS."
    )
    var autoDuration: [Double] = []

    @Option(
        name: [.customLong("video-decoder")],
        help: "LTX 2.5 VAE decoder: diffusion for maximum fidelity or convolutional for lower memory and faster decode."
    )
    var videoDecoder: LTXVideoDecoderKind?

    @Option(
        name: [.customLong("ltx-transformer-execution")],
        help: "LTX 2.5 transformer blocks: eager or opt-in shared-graph compiled execution; fusion can change floating-point results slightly."
    )
    var ltxTransformerExecution: LTXTransformerExecution = .eager

    @Option(
        name: [.customLong("ltx-guidance-projection-cache")],
        help: "Reuse positive-prompt attention projections across full-model guidance passes: automatic, disabled, or enabled."
    )
    var ltxGuidanceProjectionCache: LTXGuidanceProjectionCacheMode = .disabled

    @Flag(
        name: [.customLong("ltx-teacache")],
        help: "Enable calibrated TeaCache block-residual reuse for full LTX 2.5 Euler or HQ Res2S generation."
    )
    var ltxTeaCache = false

    @Option(
        name: [.customLong("ltx-teacache-threshold")],
        help: "Override the calibrated TeaCache reuse threshold; larger values skip more transformer stacks."
    )
    var ltxTeaCacheThreshold: Float?

    @Option(
        name: [.customLong("ltx-teacache-calibration-output")],
        help: "Run full-compute TeaCache instrumentation and write native LTX 2.5 drift samples as JSON."
    )
    var ltxTeaCacheCalibrationOutput: String?

    @Option(
        name: [.customLong("hdr")],
        help: "LTX 2.5 HDR source/output space: srgb-linear, acescg, or acescct. Writes half-float EXR frames plus a tagged BT.2020/HLG Main10 MP4."
    )
    var hdrColorSpace: LTXHDRColorSpace?

    @Option(
        name: [.customLong("hdr-transfer")],
        help: "LTX HDR VAE working-space transfer: acescct for native EXR workflows or logc3 for HDR IC-LoRA."
    )
    var hdrTransfer: LTXHDRTransfer?

    @Flag(
        name: [.customLong("high-quality-hdr")],
        help: "HDR IC-LoRA temporal-quality mode: generate 2*N-1 internal frames, duplicate reference frames, then retain every other output frame."
    )
    var highQualityHDR = false

    @Option(
        name: [.customLong("text-embeddings")],
        help: "Upstream HDR IC-LoRA safetensors containing video_context and audio_context; skips Gemma loading."
    )
    var textEmbeddings: String?

    @Option(
        name: [.customLong("spatial-tile")],
        help: "HDR IC-LoRA/convolutional-VAE spatial decode tile size in pixels (upstream default: 1280)."
    )
    var vaeSpatialTileSize: Int?

    @Option(
        name: [.customLong("spatial-overlap")],
        help: "Convolutional-VAE spatial decode overlap in pixels."
    )
    var vaeSpatialTileOverlap: Int = 256

    @Flag(
        name: [.customLong("skip-mp4")],
        help: "Dedicated HDR IC-LoRA only: write half-float EXR frames without an HLG MP4 master."
    )
    var skipHDRMP4 = false

    @Option(name: [.long], help: "Frames per second. LTX accepts fractional rates such as 23.976.")
    var fps: Double = 24

    @Option(name: [.long], help: "Seed value.")
    var seed: Int?

    @Option(
        name: [.long],
        help: "Denoising schedule points. Defaults to 40 for Wan; MiniMax-H3 selects 9, 16, or 21 from packed geometry."
    )
    var steps: Int?

    @Option(
        name: [.customLong("h3-weight-mode")],
        help: "MiniMax-H3 transformer compute: auto, quantized, or resident-bf16."
    )
    var h3WeightMode: MiniMaxH3CLITransformerWeightMode = .automatic

    @Option(
        name: [.customLong("h3-acceleration")],
        help: "MiniMax-H3 acceleration: quality is exact; balanced/maximum add sparse attention and caching; layers-45/layers-40, velocity-reuse-2, and token-reduction are experimental research arms."
    )
    var h3Acceleration: MiniMaxH3CLIAccelerationMode = .quality

    @Option(
        name: [.customLong("h3-render-width")],
        help: "MiniMax-H3 internal render width. Set with --h3-render-height; output is upscaled."
    )
    var h3RenderWidth: Int?

    @Option(
        name: [.customLong("h3-render-height")],
        help: "MiniMax-H3 internal render height. Set with --h3-render-width; output is upscaled."
    )
    var h3RenderHeight: Int?

    @Option(
        name: [.customLong("h3-adapter")],
        help: "Installed MiniMax-H3 adapter catalog id or local safetensors path. Published Turbo steps and shifts are selected from the adapter recipe."
    )
    var h3Adapter: String?

    @Option(name: [.customLong("h3-adapter-strength")], help: "MiniMax-H3 runtime adapter multiplier.")
    var h3AdapterStrength: Float = 1

    @Option(
        name: [.customLong("h3-frame")],
        help: "MiniMax-H3 FL2VA image at zero-based FRAME:PATH. Repeat for arbitrary timed frame injection."
    )
    var h3FrameArguments: [String] = []

    @Option(
        name: [.customLong("h3-window-frames")],
        help: "MiniMax-H3 resident sliding-window size (17*n+5 frames). Enables long-form generation."
    )
    var h3WindowFrames: Int?

    @Option(
        name: [.customLong("h3-window-overlap")],
        help: "MiniMax-H3 sliding overlap (17*n+1 frames, default 18). Carries motion and matching audio."
    )
    var h3WindowOverlap: Int = 18

    @Option(name: [.customLong("guidance-scale")], help: "Wan classifier-free guidance scale.")
    var guidanceScale: Float = 5

    @Option(name: [.long], help: "Wan flow-schedule shift.")
    var shift: Float = 5

    @Option(name: [.customLong("negative-prompt")], help: "Negative prompt for Wan or full LTX generation. Defaults to the selected pipeline's official prompt.")
    var negativePrompt: String?

    @Flag(
        name: [.customLong("enhance-prompt")],
        help: "Expand the prompt with the native Gemma-4 LTX-2.5 caption enhancer."
    )
    var enhancePrompt: Bool = false

    @Option(
        name: [.customLong("prompt-enhancer-model")],
        help: "Managed Gemma-4 enhancer model id; defaults to text or vision Gemma-4 based on --image."
    )
    var promptEnhancerModel: String?

    @Option(
        name: [.customLong("prompt-enhancer-model-root")],
        help: "Explicit local generative Gemma-4 instruct checkpoint root for prompt enhancement."
    )
    var promptEnhancerModelRoot: String?

    @Option(name: [.customLong("audio")], help: "Source audio path. Automatically selects native LTX audio-to-video.")
    var audio: String?

    @Option(name: [.customLong("audio-start-time")], help: "Start time in seconds for the source audio segment.")
    var audioStartTime: Double = 0

    @Option(
        name: [.customLong("audio-max-duration")],
        help: "Maximum source-audio duration to decode; defaults to the generated video duration."
    )
    var audioMaxDuration: Double?

    @Option(name: [.customLong("a2v-guidance-scale")], help: "LTX audio-to-video modality guidance scale, including video guidance in full unified AV.")
    var a2vGuidanceScale: Float = 3

    @Option(name: [.customLong("video-cfg-guidance-scale")], help: "LTX full/A2Vid video classifier-free guidance scale.")
    var videoCFGGuidanceScale: Float = 3

    @Option(name: [.customLong("audio-cfg-guidance-scale")], help: "LTX full unified-AV audio classifier-free guidance scale.")
    var audioCFGGuidanceScale: Float = 7

    @Option(name: [.customLong("v2a-guidance-scale")], help: "LTX full unified-AV video-to-audio modality guidance scale.")
    var v2aGuidanceScale: Float = 3

    @Option(name: [.customLong("ltx-preset")], help: "LTX full-model recipe: standard or the official 15-step Res2s hq preset.")
    var ltxPreset: LTXGenerationPreset = .standard

    @Option(
        name: [.customLong("ltx-pipeline")],
        help: "LTX full-model topology: two-stage, keyframe-interpolation, or dev-one-stage."
    )
    var ltxPipeline: LTXGenerationPipeline = .twoStage

    @Option(name: [.customLong("ltx-sampler")], help: "LTX full-model sampler: euler, res2s, euler-ancestral, cfg-plus-plus, or gradient-estimating-euler.")
    var ltxSampler: LTXSamplerMode?

    @Option(
        name: [.customLong("ltx-sigmas")],
        parsing: .upToNextOption,
        help: "Explicit descending LTX stage-one sigma schedule ending at zero."
    )
    var ltxSigmas: [Float] = []

    @Option(
        name: [.customLong("ltx-stage-2-sigmas")],
        parsing: .upToNextOption,
        help: "Explicit descending LTX stage-two sigma schedule ending at zero."
    )
    var ltxStage2Sigmas: [Float] = []

    @Option(name: [.customLong("distilled-lora-strength-stage-1")], help: "Distilled LoRA strength in stage one; defaults to 0.25 for hq and 0 otherwise.")
    var distilledLoRAStrengthStage1: Float?

    @Option(name: [.customLong("distilled-lora-strength-stage-2")], help: "Distilled LoRA strength in stage two; defaults to 0.5 for hq and 1 otherwise.")
    var distilledLoRAStrengthStage2: Float?

    @Option(name: [.customLong("ltx-sampler-eta")], help: "LTX ancestral/Res2s stochasticity in [0, 1].")
    var ltxSamplerEta: Float = 0.5

    @Option(name: [.customLong("video-stg-scale")], help: "LTX video spatio-temporal guidance scale.")
    var videoSTGScale: Float = 1

    @Option(name: [.customLong("video-guidance-rescale")], help: "LTX video guidance rescale strength.")
    var videoGuidanceRescale: Float = 0.7

    @Option(name: [.customLong("video-stg-block")], help: "LTX video transformer block to perturb for STG. Repeatable.")
    var videoSTGBlocks: [Int] = []

    @Option(name: [.customLong("video-guidance-skip-step")], help: "Reuse the prior video denoised estimate for this many interleaved steps.")
    var videoGuidanceSkipStep: Int = 0

    @Option(name: [.customLong("audio-stg-scale")], help: "LTX audio spatio-temporal guidance scale.")
    var audioSTGScale: Float = 1

    @Option(name: [.customLong("audio-guidance-rescale")], help: "LTX audio guidance rescale strength.")
    var audioGuidanceRescale: Float = 0.7

    @Option(name: [.customLong("audio-stg-block")], help: "LTX audio transformer block to perturb for STG. Repeatable.")
    var audioSTGBlocks: [Int] = []

    @Option(name: [.customLong("audio-guidance-skip-step")], help: "Reuse the prior audio denoised estimate for this many interleaved steps.")
    var audioGuidanceSkipStep: Int = 0

    @Flag(name: [.customLong("no-res2s-bong-math")], help: "Disable official Res2s iterative anchor refinement.")
    var noRes2sBongMath = false

    @Option(name: [.customLong("res2s-bong-max-iterations")], help: "Maximum Res2s anchor-refinement iterations.")
    var res2sBongMaxIterations: Int = 100

    @Option(name: [.customLong("gradient-estimation-gamma")], help: "Velocity correction coefficient for gradient-estimating Euler.")
    var gradientEstimationGamma: Float = 2

    @Option(name: [.customLong("a2v-steps")], help: "LTX full/A2Vid stage-one inference steps.")
    var a2vSteps: Int = 30

    @Option(name: [.long], help: "Optional source image path (enables image-to-video).")
    var image: String?

    @Option(name: [.customLong("image-strength")], help: "Image conditioning strength in [0, 1].")
    var imageStrength: Float = 1.0

    @Option(name: [.customLong("end-image")], help: "Optional end keyframe path; conditions the last frame so the clip interpolates a directed start->end motion. Requires --image.")
    var endImage: String?

    @Option(name: [.customLong("reference")], help: "Ordered MiniMax-H3 Ref2VA input as image:path, video:path, or audio:path. Repeat to preserve semantic order.")
    var references: [String] = []

    @Option(name: [.customLong("end-image-strength")], help: "End keyframe conditioning strength in [0, 1].")
    var endImageStrength: Float = 1.0

    @Option(
        name: [.customLong("image-conditioning")],
        help: "LTX 2.5 timed image guide as PIXEL_FRAME:PATH[:STRENGTH[:CRF]]. Repeat for arbitrary reference frames."
    )
    var imageConditioningArguments: [String] = []

    @Option(
        name: [.customLong("generated-keyframe")],
        help: "LTX 2.5 pixel-frame position for a generated keyframe slot. Repeat in increasing order."
    )
    var generatedKeyframeIndices: [Int] = []

    @Option(
        name: [.customLong("num-generated-keyframes")],
        help: "Number of evenly spaced interior LTX 2.5 generated keyframe slots."
    )
    var numGeneratedKeyframes: Int = 0

    @Option(
        name: [.customLong("lora")],
        help: "LTX runtime LoRA as PATH[=STRENGTH]. Repeat to stack adapters."
    )
    var loraArguments: [String] = []

    @Option(
        name: [.customLong("video-conditioning")],
        help: "IC-LoRA reference video as PATH[=STRENGTH]. Repeat for multiple references."
    )
    var videoConditioningArguments: [String] = []

    @Option(
        name: [.customLong("conditioning-attention-strength")],
        help: "IC-LoRA target/reference attention strength in [0, 1]."
    )
    var conditioningAttentionStrength: Float = 1

    @Option(
        name: [.customLong("conditioning-attention-mask")],
        help: "Grayscale mask video controlling per-region IC-LoRA attention; combines with --conditioning-attention-strength."
    )
    var conditioningAttentionMask: String?

    @Flag(
        name: [.customLong("skip-stage-2")],
        help: "Skip IC-LoRA upsampling/refinement and emit the half-resolution stage-one preview."
    )
    var skipStage2 = false

    @Option(
        name: [.customLong("reference-downscale-factor")],
        help: "Override IC-LoRA reference spatial downscale; adapter metadata is preferred."
    )
    var referenceDownscaleFactor: Int?

    @Option(
        name: [.customLong("reference-temporal-scale-factor")],
        help: "Override IC-LoRA reference temporal subsampling; adapter metadata is preferred."
    )
    var referenceTemporalScaleFactor: Int?

    @Flag(
        name: [.customLong("dfr")],
        help: "Use the official LTX 2.5 Diffusion Fidelity Rendering pipeline."
    )
    var dfr: Bool = false

    @Option(
        name: [.customLong("temporal-upsample-rounds")],
        help: "LTX 2.5 DFR temporal x2 refinement rounds: 0, 1, or 2."
    )
    var temporalUpsampleRounds: Int = 0

    @Option(
        name: [.customLong("detailing-lora")],
        help: "DFR stage-two IC-LoRA as PATH[=STRENGTH]. Repeat to stack detailing adapters."
    )
    var detailingLoRAArguments: [String] = []

    @Option(
        name: [.customLong("detailing-reference-downscale-factor")],
        help: "Override the DFR detailing IC-LoRA reference downscale factor; metadata is preferred."
    )
    var detailingReferenceDownscaleFactor: Int?

    @Flag(name: [.customLong("preflight")], help: "Inspect the video generation request without running generation.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "With --preflight, emit a structured JSON report.")
    var json: Bool = false

    @Flag(name: [.customLong("timings")], help: "Print native LTX split-distilled/unified-AV/A2Vid phase timings to stderr.")
    var timings: Bool = false

    @Option(name: [.customLong("timings-output")], help: "Write native LTX split-distilled/unified-AV/A2Vid phase timings as JSON.")
    var timingsOutput: String?

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    @Flag(name: [.customLong(CLIGenerationProgressPrinter.flagName)], help: CLIGenerationProgressPrinter.flagHelp)
    var progressJson: Bool = false

    @Flag(name: [.customLong(RunReceipt.flagName)], help: RunReceipt.flagHelp)
    var receipt: Bool = false

    func validate() throws {
        try RunReceipt.validate(receipt: receipt, preflight: preflight)
    }

    private var generationOptions: VideoGenerationOptions {
        makeGenerationOptions(outputURL: URL(fileURLWithPath: output ?? "."))
    }

    var variant: LTXVideoVariant { generationOptions.variant }
    var autoDurationRange: LTX25AutoDuration? { generationOptions.autoDurationRange }
    var requestedQuality: LTXVideoQuality { generationOptions.requestedQuality }
    var effectiveOutputMode: LTXVideoOutputMode { generationOptions.effectiveOutputMode }
    var productSelectionValidationMessage: String? { generationOptions.productSelectionValidationMessage }
    var resolvedRequestedModel: String { generationOptions.resolvedRequestedModel }
    var usesLTX25RecipeGeometry: Bool { generationOptions.usesLTX25RecipeGeometry }
    var resolvedOutputWidth: Int { generationOptions.resolvedOutputWidth }
    var resolvedOutputHeight: Int { generationOptions.resolvedOutputHeight }

    func run() async throws {
        if json && !preflight {
            throw ValidationError("--json is only supported with --preflight for video generate.")
        }
        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-video", defaultExtension: "mp4")
        if preflight {
            try runPreflight(outputURL: outputURL)
            return
        }
        let options = makeGenerationOptions(outputURL: outputURL)
        let quiet = quiet
        let presentation = CLIVideoGenerationPresentation(quiet: quiet, progressJSON: progressJson)
        let outcome: VideoGenerationOutcome
        do {
            outcome = try await VideoGenerationOperation.execute(
                options,
                prepareRuntime: { try MLXBundleSupport.ensureAvailable(quiet: quiet) },
                eventHandler: presentation.eventHandler
            )
        } catch VideoGenerationError.invalidInput(let message) {
            throw ValidationError(message)
        } catch let issue as VideoGenerationIssue {
            throw ValidationError(issue.message)
        }
        if let report = outcome.timings {
            try emitLTXVideoTimingReport(report, printToStandardError: timings, outputPath: timingsOutput)
        }
        if !quiet { CLIStderr.write("Saved: \(outcome.primaryURL.path)\n") }
        print(outcome.primaryURL.path)
        try RunReceipt.emit(
            RunReceipt.generatedVideoOutputs(
                primary: outcome.primaryURL,
                kind: outcome.isDirectory ? .directory : .video,
                timings: outcome.includesTimings ? timingsOutput.map { URL(fileURLWithPath: $0) } : nil
            ),
            enabled: receipt
        )
    }

    func makeGenerationOptions(outputURL: URL) -> VideoGenerationOptions {
        VideoGenerationOptions(
            prompt: prompt,
            outputURL: outputURL,
            model: model,
            quality: quality,
            outputMode: outputMode,
            legacyVariant: legacyVariant,
            modelRoot: modelRoot,
            width: width,
            height: height,
            numFrames: numFrames,
            steps: steps,
            h3WeightMode: h3WeightMode.rawValue,
            h3AccelerationMode: h3Acceleration.rawValue,
            h3RenderWidth: h3RenderWidth,
            h3RenderHeight: h3RenderHeight,
            h3Adapter: h3Adapter,
            h3AdapterStrength: h3AdapterStrength,
            h3FrameInputs: h3FrameArguments,
            h3WindowFrames: h3WindowFrames,
            h3WindowOverlap: h3WindowOverlap,
            duration: duration,
            autoDuration: autoDuration,
            videoDecoder: videoDecoder,
            hdrColorSpace: hdrColorSpace,
            hdrTransfer: hdrTransfer,
            highQualityHDR: highQualityHDR,
            textEmbeddings: textEmbeddings,
            vaeSpatialTileSize: vaeSpatialTileSize,
            vaeSpatialTileOverlap: vaeSpatialTileOverlap,
            skipHDRMP4: skipHDRMP4,
            fps: fps,
            seed: seed,
            negativePrompt: negativePrompt,
            enhancePrompt: enhancePrompt,
            promptEnhancerModel: promptEnhancerModel,
            promptEnhancerModelRoot: promptEnhancerModelRoot,
            audio: audio,
            audioStartTime: audioStartTime,
            audioMaxDuration: audioMaxDuration,
            a2vGuidanceScale: a2vGuidanceScale,
            videoCFGGuidanceScale: videoCFGGuidanceScale,
            audioCFGGuidanceScale: audioCFGGuidanceScale,
            v2aGuidanceScale: v2aGuidanceScale,
            a2vSteps: a2vSteps,
            ltxPreset: ltxPreset,
            ltxPipeline: ltxPipeline,
            ltxSampler: ltxSampler,
            ltxSigmas: ltxSigmas,
            ltxStage2Sigmas: ltxStage2Sigmas,
            distilledLoRAStrengthStage1: distilledLoRAStrengthStage1,
            distilledLoRAStrengthStage2: distilledLoRAStrengthStage2,
            ltxSamplerEta: ltxSamplerEta,
            videoSTGScale: videoSTGScale,
            videoGuidanceRescale: videoGuidanceRescale,
            videoSTGBlocks: videoSTGBlocks,
            videoGuidanceSkipStep: videoGuidanceSkipStep,
            audioSTGScale: audioSTGScale,
            audioGuidanceRescale: audioGuidanceRescale,
            audioSTGBlocks: audioSTGBlocks,
            audioGuidanceSkipStep: audioGuidanceSkipStep,
            noRes2sBongMath: noRes2sBongMath,
            res2sBongMaxIterations: res2sBongMaxIterations,
            gradientEstimationGamma: gradientEstimationGamma,
            image: image,
            imageStrength: imageStrength,
            endImage: endImage,
            endImageStrength: endImageStrength,
            imageConditionings: imageConditioningArguments,
            numGeneratedKeyframes: numGeneratedKeyframes,
            generatedKeyframeIndices: generatedKeyframeIndices,
            loras: loraArguments,
            videoConditionings: videoConditioningArguments,
            conditioningAttentionStrength: conditioningAttentionStrength,
            conditioningAttentionMask: conditioningAttentionMask,
            skipStage2: skipStage2,
            referenceDownscaleFactor: referenceDownscaleFactor,
            referenceTemporalScaleFactor: referenceTemporalScaleFactor,
            dfr: dfr,
            temporalUpsampleRounds: temporalUpsampleRounds,
            detailingLoRAs: detailingLoRAArguments,
            detailingReferenceDownscaleFactor: detailingReferenceDownscaleFactor,
            references: references,
            timings: timings,
            timingsOutput: timingsOutput,
            guidanceScale: guidanceScale,
            shift: shift,
            ltxTransformerExecution: ltxTransformerExecution,
            ltxGuidanceProjectionCache: ltxGuidanceProjectionCache,
            ltxTeaCache: ltxTeaCache,
            ltxTeaCacheThreshold: ltxTeaCacheThreshold,
            ltxTeaCacheCalibrationOutput: ltxTeaCacheCalibrationOutput
        )
    }

    func makePreflightEnvelope(
        outputURL: URL,
        fileManager: FileManager = .default,
        adaptersRoot: URL = MereRunModelPaths.adaptersDir,
        now: @escaping () -> Date = Date.init
    ) -> VideoGenerationPreflightEnvelope {
        let input = makeGenerationOptions(outputURL: outputURL)
        return VideoGenerationPreflightAnalyzer(
            input: input,
            generationArgv: generationActionArguments(outputURL: outputURL),
            cwd: fileManager.currentDirectoryPath,
            fileManager: fileManager,
            adaptersRoot: adaptersRoot,
            now: now
        ).envelope()
    }

    static func encodePreflight(_ envelope: VideoGenerationPreflightEnvelope) throws -> String {
        let encoder = StructuredRunOutput.encoder()
        // Invalid floating-point arguments must still yield a JSON diagnostic.
        encoder.nonConformingFloatEncodingStrategy = .convertToString(
            positiveInfinity: "Infinity", negativeInfinity: "-Infinity", nan: "NaN"
        )
        return String(decoding: try encoder.encode(envelope), as: UTF8.self)
    }

    private func runPreflight(outputURL: URL) throws {
        let envelope = makePreflightEnvelope(outputURL: outputURL)
        if json {
            print(try Self.encodePreflight(envelope))
        } else {
            print(envelope.summary)
            for diagnostic in envelope.diagnostics {
                print("[\(diagnostic.severity.rawValue)] \(diagnostic.title): \(diagnostic.message)")
            }
        }
        if envelope.status == .blocked {
            throw ExitCode.failure
        }
    }

    private func generationActionArguments(outputURL: URL) -> [String] {
        var args = [
            "mere.run",
            "video",
            "generate",
            prompt,
            "--output",
            outputURL.path,
            "--model",
            resolvedRequestedModel,
            "--width",
            String(resolvedOutputWidth),
            "--height",
            String(resolvedOutputHeight),
            "--fps",
            String(fps),
        ]
        if let numFrames {
            args += ["--num-frames", String(numFrames)]
        }
        if let quality {
            args += ["--quality", quality.rawValue]
        }
        if let outputMode {
            args += ["--output-mode", outputMode.rawValue]
        }
        if let legacyVariant {
            args += ["--variant", legacyVariant.rawValue]
        }
        if isMiniMaxH3Request {
            args += [
                "--h3-weight-mode", h3WeightMode.rawValue,
                "--h3-acceleration", h3Acceleration.rawValue,
            ]
            if let h3RenderWidth, let h3RenderHeight {
                args += [
                    "--h3-render-width", String(h3RenderWidth),
                    "--h3-render-height", String(h3RenderHeight),
                ]
            }
            if let h3Adapter {
                args += ["--h3-adapter", h3Adapter, "--h3-adapter-strength", String(h3AdapterStrength)]
            }
            for frame in h3FrameArguments {
                args += ["--h3-frame", frame]
            }
            if let h3WindowFrames {
                args += [
                    "--h3-window-frames", String(h3WindowFrames),
                    "--h3-window-overlap", String(h3WindowOverlap),
                ]
            }
        }
        if let duration {
            args += ["--duration", String(duration)]
        }
        if autoDuration.count == 2 {
            args += [
                "--auto-duration",
                String(autoDuration[0]),
                String(autoDuration[1]),
            ]
        }
        if let videoDecoder {
            args += ["--video-decoder", videoDecoder.rawValue]
        }
        if ltxTransformerExecution != .eager {
            args += ["--ltx-transformer-execution", ltxTransformerExecution.rawValue]
        }
        if let hdrColorSpace {
            args += ["--hdr", hdrColorSpace.rawValue]
        }
        if let hdrTransfer {
            args += ["--hdr-transfer", hdrTransfer.rawValue]
        }
        if highQualityHDR {
            args.append("--high-quality-hdr")
        }
        if let textEmbeddings {
            args += ["--text-embeddings", textEmbeddings]
        }
        if let vaeSpatialTileSize {
            args += ["--spatial-tile", String(vaeSpatialTileSize)]
            args += ["--spatial-overlap", String(vaeSpatialTileOverlap)]
        }
        if skipHDRMP4 {
            args.append("--skip-mp4")
        }
        if let seed {
            args += ["--seed", String(seed)]
        }
        if let steps {
            args += ["--steps", String(steps)]
        }
        if let negativePrompt {
            args += ["--negative-prompt", negativePrompt]
        }
        if enhancePrompt {
            args.append("--enhance-prompt")
            if let promptEnhancerModel {
                args += ["--prompt-enhancer-model", promptEnhancerModel]
            }
            if let promptEnhancerModelRoot {
                args += ["--prompt-enhancer-model-root", promptEnhancerModelRoot]
            }
        }
        if audio != nil || requestedQuality == .final || effectiveOutputMode == .audioVideo {
            args += [
                "--a2v-guidance-scale", String(a2vGuidanceScale),
                "--video-cfg-guidance-scale", String(videoCFGGuidanceScale),
                "--audio-cfg-guidance-scale", String(audioCFGGuidanceScale),
                "--v2a-guidance-scale", String(v2aGuidanceScale),
                "--a2v-steps", String(a2vSteps),
            ]
        }
        if ltxPreset != .standard {
            args += ["--ltx-preset", ltxPreset.rawValue]
        }
        if ltxPipeline != .twoStage {
            args += ["--ltx-pipeline", ltxPipeline.rawValue]
        }
        if let ltxSampler {
            args += ["--ltx-sampler", ltxSampler.rawValue]
        }
        if !ltxSigmas.isEmpty {
            args.append("--ltx-sigmas")
            args.append(contentsOf: ltxSigmas.map { String($0) })
        }
        if !ltxStage2Sigmas.isEmpty {
            args.append("--ltx-stage-2-sigmas")
            args.append(contentsOf: ltxStage2Sigmas.map { String($0) })
        }
        if let distilledLoRAStrengthStage1 {
            args += ["--distilled-lora-strength-stage-1", String(distilledLoRAStrengthStage1)]
        }
        if let distilledLoRAStrengthStage2 {
            args += ["--distilled-lora-strength-stage-2", String(distilledLoRAStrengthStage2)]
        }
        if ltxSamplerEta != 0.5 {
            args += ["--ltx-sampler-eta", String(ltxSamplerEta)]
        }
        if videoSTGScale != 1 { args += ["--video-stg-scale", String(videoSTGScale)] }
        if videoGuidanceRescale != 0.7 {
            args += ["--video-guidance-rescale", String(videoGuidanceRescale)]
        }
        for block in videoSTGBlocks { args += ["--video-stg-block", String(block)] }
        if videoGuidanceSkipStep != 0 {
            args += ["--video-guidance-skip-step", String(videoGuidanceSkipStep)]
        }
        if audioSTGScale != 1 { args += ["--audio-stg-scale", String(audioSTGScale)] }
        if audioGuidanceRescale != 0.7 {
            args += ["--audio-guidance-rescale", String(audioGuidanceRescale)]
        }
        for block in audioSTGBlocks { args += ["--audio-stg-block", String(block)] }
        if audioGuidanceSkipStep != 0 {
            args += ["--audio-guidance-skip-step", String(audioGuidanceSkipStep)]
        }
        if noRes2sBongMath { args.append("--no-res2s-bong-math") }
        if res2sBongMaxIterations != 100 {
            args += ["--res2s-bong-max-iterations", String(res2sBongMaxIterations)]
        }
        if gradientEstimationGamma != 2 {
            args += ["--gradient-estimation-gamma", String(gradientEstimationGamma)]
        }
        if let audio {
            args += ["--audio", audio, "--audio-start-time", String(audioStartTime)]
            if let audioMaxDuration {
                args += ["--audio-max-duration", String(audioMaxDuration)]
            }
        }
        if let modelRoot {
            args += ["--model-root", modelRoot]
        }
        if let image {
            args += ["--image", image, "--image-strength", String(imageStrength)]
        }
        if let endImage {
            args += ["--end-image", endImage, "--end-image-strength", String(endImageStrength)]
        }
        for conditioning in imageConditioningArguments {
            args += ["--image-conditioning", conditioning]
        }
        if numGeneratedKeyframes > 0 {
            args += ["--num-generated-keyframes", String(numGeneratedKeyframes)]
        }
        for frame in generatedKeyframeIndices {
            args += ["--generated-keyframe", String(frame)]
        }
        for lora in loraArguments {
            args += ["--lora", lora]
        }
        for reference in videoConditioningArguments {
            args += ["--video-conditioning", reference]
        }
        if !videoConditioningArguments.isEmpty {
            args += [
                "--conditioning-attention-strength",
                String(conditioningAttentionStrength),
            ]
            if let referenceDownscaleFactor {
                args += ["--reference-downscale-factor", String(referenceDownscaleFactor)]
            }
            if let referenceTemporalScaleFactor {
                args += [
                    "--reference-temporal-scale-factor",
                    String(referenceTemporalScaleFactor),
                ]
            }
            if let conditioningAttentionMask {
                args += ["--conditioning-attention-mask", conditioningAttentionMask]
            }
            if skipStage2 {
                args.append("--skip-stage-2")
            }
        }
        if dfr {
            args.append("--dfr")
            args += ["--temporal-upsample-rounds", String(temporalUpsampleRounds)]
            for lora in detailingLoRAArguments {
                args += ["--detailing-lora", lora]
            }
            if let detailingReferenceDownscaleFactor {
                args += [
                    "--detailing-reference-downscale-factor",
                    String(detailingReferenceDownscaleFactor),
                ]
            }
        }
        for reference in references {
            args += ["--reference", reference]
        }
        if quiet {
            args.append("--quiet")
        }
        if timings {
            args.append("--timings")
        }
        if let timingsOutput {
            args += ["--timings-output", timingsOutput]
        }
        return args
    }

    private var isMiniMaxH3Request: Bool {
        let requested = resolvedRequestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if requested == ModelResolver.ModelID.miniMaxH3FL2VAMLX.rawValue
            || requested == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue
            || requested == ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue
            || requested == ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue
            || requested == ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue {
            return true
        }
        guard let modelRoot else { return false }
        let resources = MiniMaxH3Resources(
            rootURL: URL(fileURLWithPath: modelRoot).standardizedFileURL
        )
        return resources.validate().isEmpty && (try? resources.loadConfiguration()) != nil
    }
}

func nearestLTXFrameCount(duration: Double, fps: Double) -> Int {
    let targetFrames = max(9.0, duration * max(1, fps))
    let chunks = max(1, Int(((targetFrames - 1.0) / 8.0).rounded()))
    return chunks * 8 + 1
}

func nearestWanFrameCount(duration: Double, fps: Double) -> Int {
    let targetFrames = max(5.0, duration * max(1, fps))
    let chunks = max(1, Int(((targetFrames - 1.0) / 4.0).rounded()))
    return chunks * 4 + 1
}
