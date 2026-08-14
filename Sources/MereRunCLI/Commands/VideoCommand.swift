import ArgumentParser
import Foundation
import MediaIO
import MLX
import MereRunContract
import MereRunCore

enum LTXVideoGenerationRoute: String, Equatable {
    case legacyDistilledVideo = "legacy-distilled-video"
    case splitDistilledVideo = "split-distilled-video"
    case fullQualityVideo = "full-quality-video"
    case unifiedAV = "unified-av"

    var writesAudio: Bool {
        self == .unifiedAV
    }

    var supportsPhaseTimings: Bool {
        self != .legacyDistilledVideo
    }
}

func resolveLTXVideoGenerationRoute(
    outputMode: LTXVideoOutputMode,
    modelRoot: URL,
    fileManager: FileManager = .default
) -> LTXVideoGenerationRoute {
    switch outputMode {
    case .audioVideo:
        return .unifiedAV
    case .videoOnly:
        if isLTX23AudioToVideoModelRoot(modelRoot, fileManager: fileManager)
            || isLTX25FullModelRoot(modelRoot, fileManager: fileManager) {
            return .fullQualityVideo
        }
        return isLTX23SplitModelRoot(modelRoot, fileManager: fileManager)
            || isLTX25ModelRoot(modelRoot, fileManager: fileManager)
            ? .splitDistilledVideo
            : .legacyDistilledVideo
    }
}

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

private struct MiniMaxH3WiredMemoryPolicy: WiredMemoryPolicy, Hashable {
    func limit(baseline: Int, activeSizes: [Int]) -> Int {
        max(baseline, activeSizes.max() ?? baseline)
    }
}

private func miniMaxH3WiredMemoryTargetBytes() -> Int {
    let gibibyte = 1_073_741_824
    let desired = ProcessInfo.processInfo.physicalMemory >= UInt64(96 * gibibyte)
        ? 64 * gibibyte
        : 50 * gibibyte
#if os(macOS)
    let safetyMargin = 1_048_576
    guard let recommended = GPU.maxRecommendedWorkingSetBytes() else {
        return min(desired, Int(ProcessInfo.processInfo.physicalMemory / 2))
    }
    return min(desired, max(0, recommended - safetyMargin))
#else
    return min(desired, Int(ProcessInfo.processInfo.physicalMemory / 2))
#endif
}

func resolveLTXVideoGenerationRoute(
    variant: LTXVideoVariant,
    modelRoot: URL,
    fileManager: FileManager = .default
) -> LTXVideoGenerationRoute {
    resolveLTXVideoGenerationRoute(
        outputMode: variant == .unifiedAV ? .audioVideo : .videoOnly,
        modelRoot: modelRoot,
        fileManager: fileManager
    )
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
            CLIStderr.write("Final latent shape: \(shapeString(result.latents.shape))\n")
            CLIStderr.write("Stage1 latent shape: \(shapeString(result.stage1Latents.shape))\n")
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

    var variant: LTXVideoVariant {
        effectiveOutputMode.compatibilityVariant
    }

    var autoDurationRange: LTX25AutoDuration? {
        guard autoDuration.count == 2 else { return nil }
        return LTX25AutoDuration(
            minimumSeconds: autoDuration[0],
            maximumSeconds: autoDuration[1]
        )
    }

    func effectiveAutoDurationRange(
        modelRoot: URL,
        hasSourceAudio: Bool
    ) -> LTX25AutoDuration? {
        if numFrames != nil { return nil }
        if let autoDurationRange { return autoDurationRange }
        guard duration == nil,
              numFrames == nil,
              !hasSourceAudio,
              isLTX25ModelRoot(modelRoot) else {
            return nil
        }
        return LTX25AutoDuration(minimumSeconds: 1, maximumSeconds: 20)
    }

    var requestedQuality: LTXVideoQuality {
        if audio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false {
            return .final
        }
        if let modelRoot,
           isLTX25ModelRoot(URL(fileURLWithPath: modelRoot).standardizedFileURL) {
            return .final
        }
        if let quality {
            return quality
        }
        return legacyVariant == .unifiedAV ? .final : .draft
    }

    var effectiveOutputMode: LTXVideoOutputMode {
        if audio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false {
            return .audioVideo
        }
        if let outputMode {
            return outputMode
        }
        if dfr {
            return .audioVideo
        }
        return legacyVariant == .unifiedAV ? .audioVideo : .videoOnly
    }

    var productSelectionValidationMessage: String? {
        if legacyVariant != nil, quality != nil || outputMode != nil {
            return "Use --quality/--output-mode or the compatibility --variant option, not both."
        }
        let hasSourceAudio = audio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
        if hasSourceAudio, quality == .draft {
            return "--audio requires --quality final because source-audio conditioning uses the dev + distilled-LoRA checkpoint."
        }
        if hasSourceAudio, outputMode == .videoOnly {
            return "--audio preserves the selected soundtrack and requires --output-mode audio-video."
        }
        return nil
    }

    var resolvedRequestedModel: String {
        let requested = model.trimmingCharacters(in: .whitespacesAndNewlines)
        if !requested.isEmpty {
            return requested
        }
        if dfr
            || ltxPreset == .hq
            || ltxPipeline != .twoStage
            || ltxSampler != nil
            || distilledLoRAStrengthStage1 != nil
            || distilledLoRAStrengthStage2 != nil {
            return ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
        }
        if hdrColorSpace != nil
            || highQualityHDR
            || textEmbeddings != nil
            || enhancePrompt
            || !autoDuration.isEmpty
            || videoDecoder != nil
            || !imageConditioningArguments.isEmpty
            || numGeneratedKeyframes > 0
            || !generatedKeyframeIndices.isEmpty
            || !videoConditioningArguments.isEmpty {
            return ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
        }
        let hasAudio = audio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
        if hasAudio || requestedQuality == .final {
            return ModelResolver.ModelID.ltxVideo23FullMLX.rawValue
        }
        return ModelResolver.ModelID.ltxVideo23AVMLX.rawValue
    }

    var usesLTX25RecipeGeometry: Bool {
        if let modelRoot {
            let root = URL(fileURLWithPath: modelRoot).standardizedFileURL
            if isLTX25ModelRoot(root) {
                return true
            }
        }
        let requested = resolvedRequestedModel.lowercased()
        return requested == ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
            || requested == ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
            || requested.contains("ltx25")
            || requested.contains("ltx-2.5")
    }

    var resolvedOutputWidth: Int {
        if let width { return width }
        guard usesLTX25RecipeGeometry else { return 768 }
        if ltxPreset == .hq { return 1_920 }
        return ltxPipeline == .devOneStage ? 768 : 1_536
    }

    var resolvedOutputHeight: Int {
        if let height { return height }
        guard usesLTX25RecipeGeometry else { return 512 }
        if ltxPreset == .hq { return 1_088 }
        return ltxPipeline == .devOneStage ? 512 : 1_024
    }

    func run() async throws {
        let width = resolvedOutputWidth
        let height = resolvedOutputHeight
        if json && !preflight {
            throw ValidationError("--json is only supported with --preflight for video generate.")
        }

        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-video", defaultExtension: "mp4")
        if preflight {
            try runPreflight(outputURL: outputURL)
            return
        }

        if let productSelectionValidationMessage {
            throw ValidationError(productSelectionValidationMessage)
        }

        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else {
            throw ValidationError("Prompt cannot be empty.")
        }
        guard fps.isFinite, fps >= 1 else {
            throw ValidationError("--fps must be finite and >= 1")
        }
        if let duration {
            guard duration > 0 else {
                throw ValidationError("--duration must be > 0")
            }
        }
        guard autoDuration.isEmpty || autoDuration.count == 2 else {
            throw ValidationError("--auto-duration requires MIN_SECONDS MAX_SECONDS")
        }
        if autoDuration.count == 2 {
            guard autoDuration[0].isFinite,
                  autoDuration[1].isFinite,
                  autoDuration[0] > 0,
                  autoDuration[1] >= autoDuration[0] else {
                throw ValidationError("--auto-duration requires 0 < MIN_SECONDS <= MAX_SECONDS")
            }
            guard duration == nil else {
                throw ValidationError("Use --duration or --auto-duration, not both.")
            }
            guard audio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false else {
                throw ValidationError("--auto-duration is unavailable for source-audio A2Vid.")
            }
            if numFrames != nil, !quiet {
                CLIStderr.write("Warning: --auto-duration is ignored because --num-frames was supplied.\n")
            }
        }
        guard width >= 32 else {
            throw ValidationError("--width must be >= 32")
        }
        guard height >= 32 else {
            throw ValidationError("--height must be >= 32")
        }
        if let numFrames, numFrames < 5 {
            throw ValidationError("--num-frames must be >= 5")
        }
        if let steps, steps <= 0 {
            throw ValidationError("--steps must be >= 1")
        }
        guard guidanceScale >= 0 else {
            throw ValidationError("--guidance-scale must be >= 0")
        }
        guard shift > 0 else {
            throw ValidationError("--shift must be > 0")
        }
        guard audioStartTime.isFinite, audioStartTime >= 0 else {
            throw ValidationError("--audio-start-time must be finite and >= 0")
        }
        if let audioMaxDuration,
           !audioMaxDuration.isFinite || audioMaxDuration <= 0 {
            throw ValidationError("--audio-max-duration must be finite and > 0")
        }
        guard a2vGuidanceScale >= 0 else {
            throw ValidationError("--a2v-guidance-scale must be >= 0")
        }
        guard videoCFGGuidanceScale >= 0 else {
            throw ValidationError("--video-cfg-guidance-scale must be >= 0")
        }
        guard audioCFGGuidanceScale >= 0 else {
            throw ValidationError("--audio-cfg-guidance-scale must be >= 0")
        }
        guard v2aGuidanceScale >= 0 else {
            throw ValidationError("--v2a-guidance-scale must be >= 0")
        }
        guard (0...1).contains(ltxSamplerEta) else {
            throw ValidationError("--ltx-sampler-eta must be in [0, 1]")
        }
        guard videoSTGScale >= 0, audioSTGScale >= 0 else {
            throw ValidationError("--video-stg-scale and --audio-stg-scale must be >= 0")
        }
        guard (0...1).contains(videoGuidanceRescale),
              (0...1).contains(audioGuidanceRescale) else {
            throw ValidationError("LTX guidance rescale values must be in [0, 1]")
        }
        guard videoGuidanceSkipStep >= 0, audioGuidanceSkipStep >= 0 else {
            throw ValidationError("LTX guidance skip-step values must be >= 0")
        }
        guard videoSTGBlocks.allSatisfy({ $0 >= 0 }), audioSTGBlocks.allSatisfy({ $0 >= 0 }) else {
            throw ValidationError("LTX STG block indices must be >= 0")
        }
        guard res2sBongMaxIterations > 0 else {
            throw ValidationError("--res2s-bong-max-iterations must be >= 1")
        }
        guard gradientEstimationGamma.isFinite else {
            throw ValidationError("--gradient-estimation-gamma must be finite")
        }
        if !ltxSigmas.isEmpty {
            _ = try validatedLTXSigmaSchedule(ltxSigmas)
        }
        guard h3AdapterStrength > 0 else {
            throw ValidationError("--h3-adapter-strength must be > 0")
        }
        guard a2vSteps > 0 else {
            throw ValidationError("--a2v-steps must be >= 1")
        }
        guard (0...2).contains(temporalUpsampleRounds) else {
            throw ValidationError("--temporal-upsample-rounds must be 0, 1, or 2")
        }
        if temporalUpsampleRounds > 0, !dfr {
            throw ValidationError("--temporal-upsample-rounds requires --dfr")
        }
        if !detailingLoRAArguments.isEmpty, !dfr {
            throw ValidationError("--detailing-lora requires --dfr")
        }
        if detailingReferenceDownscaleFactor != nil, !dfr {
            throw ValidationError("--detailing-reference-downscale-factor requires --dfr")
        }
        if let detailingReferenceDownscaleFactor, detailingReferenceDownscaleFactor <= 0 {
            throw ValidationError("--detailing-reference-downscale-factor must be positive")
        }
        guard (0...1).contains(conditioningAttentionStrength) else {
            throw ValidationError("--conditioning-attention-strength must be in [0, 1]")
        }
        guard vaeSpatialTileOverlap >= 0 else {
            throw ValidationError("--spatial-overlap must be nonnegative")
        }
        if let vaeSpatialTileSize, vaeSpatialTileSize <= vaeSpatialTileOverlap {
            throw ValidationError("--spatial-tile must be larger than --spatial-overlap")
        }
        if let referenceDownscaleFactor, referenceDownscaleFactor <= 0 {
            throw ValidationError("--reference-downscale-factor must be positive")
        }
        if let referenceTemporalScaleFactor, referenceTemporalScaleFactor <= 0 {
            throw ValidationError("--reference-temporal-scale-factor must be positive")
        }
        if !videoConditioningArguments.isEmpty, loraArguments.isEmpty {
            throw ValidationError("--video-conditioning requires at least one IC-LoRA via --lora")
        }
        if dfr, audio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false {
            throw ValidationError("--dfr generates synchronized audio and cannot be combined with source --audio.")
        }
        if dfr,
           numGeneratedKeyframes > 0 || !generatedKeyframeIndices.isEmpty {
            throw ValidationError("--dfr derives its own generated-keyframe slots.")
        }
        if dfr, !videoConditioningArguments.isEmpty {
            throw ValidationError("--video-conditioning is an IC-LoRA workflow and cannot be combined with --dfr.")
        }
        if dfr, skipStage2 {
            throw ValidationError("--skip-stage-2 is an IC-LoRA workflow and cannot be combined with --dfr.")
        }
        if dfr, ltxPreset != .standard || ltxPipeline != .twoStage || ltxSampler != nil {
            throw ValidationError("--dfr owns its distilled two-stage sampler recipe.")
        }
        if dfr,
           distilledLoRAStrengthStage1 != nil || distilledLoRAStrengthStage2 != nil {
            throw ValidationError("--dfr owns its stage-one and stage-two distilled-LoRA strengths.")
        }
        guard (0...1).contains(imageStrength) else {
            throw ValidationError("--image-strength must be between 0 and 1")
        }
        guard (0...1).contains(endImageStrength) else {
            throw ValidationError("--end-image-strength must be between 0 and 1")
        }
        if endImage != nil, image == nil {
            throw ValidationError("--end-image requires --image (the start keyframe)")
        }
        guard numGeneratedKeyframes >= 0 else {
            throw ValidationError("--num-generated-keyframes must be nonnegative")
        }
        if numGeneratedKeyframes > 0, !generatedKeyframeIndices.isEmpty {
            throw ValidationError(
                "Use --num-generated-keyframes or explicit --generated-keyframe positions, not both."
            )
        }
        if ltxPipeline == .keyframeInterpolation {
            if audio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false {
                throw ValidationError("The keyframe-interpolation pipeline generates its own synchronized audio and cannot use source --audio.")
            }
            if numGeneratedKeyframes > 0 || !generatedKeyframeIndices.isEmpty {
                throw ValidationError("The keyframe-interpolation pipeline accepts timed image guides, not generated-keyframe slots.")
            }
            if !videoConditioningArguments.isEmpty || skipStage2 {
                throw ValidationError("The keyframe-interpolation pipeline cannot be combined with IC-LoRA reference-video controls.")
            }
        }

        let resolvedWidth = max(64, (width / 64) * 64)
        let resolvedHeight = max(64, (height / 64) * 64)
        var requestedNumFrames = duration.map { nearestLTXFrameCount(duration: $0, fps: fps) }
            ?? numFrames
            ?? 65
        var resolvedNumFrames = max(9, ((requestedNumFrames - 1) / 8) * 8 + 1)

        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let sourceImageURL: URL?
        if let image, !image.isEmpty {
            let url = URL(fileURLWithPath: image).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Image file not found: \(url.path)")
            }
            sourceImageURL = url
        } else {
            sourceImageURL = nil
        }

        let endImageURL: URL?
        if let endImage, !endImage.isEmpty {
            let url = URL(fileURLWithPath: endImage).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("End image file not found: \(url.path)")
            }
            endImageURL = url
        } else {
            endImageURL = nil
        }

        let sourceAudioURL: URL?
        if let audio, !audio.isEmpty {
            let url = URL(fileURLWithPath: audio).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Audio file not found: \(url.path)")
            }
            sourceAudioURL = url
        } else {
            sourceAudioURL = nil
        }

        let resolvedModelRoot = try await resolveVideoModelRoot(
            explicitModelRoot: modelRoot,
            requestedModel: resolvedRequestedModel,
            variant: variant
        ).path

        let resolvedRootURL = URL(fileURLWithPath: resolvedModelRoot).standardizedFileURL
        if sourceAudioURL != nil, duration == nil, numFrames == nil,
           isLTX25ModelRoot(resolvedRootURL) {
            requestedNumFrames = 121
            resolvedNumFrames = 121
        }
        let resolvedSeed = seed ?? (isLTX25ModelRoot(resolvedRootURL) ? 10 : 42)
        try validateProductSelection(modelRoot: resolvedRootURL)
        if videoDecoder != nil, !isLTX25ModelRoot(resolvedRootURL) {
            throw ValidationError("--video-decoder is available for official LTX 2.5 model roots.")
        }
        if hdrColorSpace != nil, !isLTX25ModelRoot(resolvedRootURL) {
            throw ValidationError("--hdr requires an official LTX 2.5 model root.")
        }
        let resolvedVideoDecoder = videoDecoder
            ?? (isLTX25FullModelRoot(resolvedRootURL) ? .diffusion : .convolutional)
        if enhancePrompt, !isLTX25ModelRoot(resolvedRootURL) {
            throw ValidationError("--enhance-prompt requires an official LTX 2.5 model root.")
        }
        if !autoDuration.isEmpty, !isLTX25ModelRoot(resolvedRootURL) {
            throw ValidationError("--auto-duration requires an official LTX 2.5 model root.")
        }
        if dfr, !isLTX25FullModelRoot(resolvedRootURL) {
            throw ValidationError(
                "--dfr requires the full \(ModelResolver.ModelID.ltxVideo25FullBF16.rawValue) checkpoint."
            )
        }
        let ltxAdapterBaseModelID = isLTX25FullModelRoot(resolvedRootURL)
            ? ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
            : isLTX25ModelRoot(resolvedRootURL)
                ? ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
                : resolvedRequestedModel
        let loras = try parseLTXLoRAConfigurations(
            loraArguments,
            optionName: "--lora",
            baseModelID: ltxAdapterBaseModelID
        )
        let detailingLoRAs = try parseLTXLoRAConfigurations(
            detailingLoRAArguments,
            optionName: "--detailing-lora",
            baseModelID: ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
        )
        let hdrLoRAConfigurations = try loras.compactMap(ltxHDRLoRAConfiguration)
        if Set(hdrLoRAConfigurations).count > 1 {
            throw ValidationError("Stacked HDR LoRAs must use the same HDR transform and reference downscale factor.")
        }
        let hdrLoRAConfiguration = hdrLoRAConfigurations.first
        let referenceScaleConfiguration = if videoConditioningArguments.isEmpty {
            LTXLoRAReferenceScaleConfiguration()
        } else {
            try ltxLoRAReferenceScaleConfiguration(loras)
        }
        let resolvedHDRColorSpace = hdrColorSpace
            ?? hdrLoRAConfiguration.map { _ in LTXHDRColorSpace.srgbLinear }
        let resolvedHDRTransfer = hdrTransfer
            ?? hdrLoRAConfiguration?.hdrTransform
            ?? .acesCCT
        if resolvedHDRColorSpace != nil, !isLTX25ModelRoot(resolvedRootURL) {
            throw ValidationError("HDR and HDR IC-LoRA workflows require an official LTX 2.5 model root.")
        }
        let referenceSpatialScale = referenceDownscaleFactor
            ?? hdrLoRAConfiguration?.referenceDownscaleFactor
            ?? referenceScaleConfiguration.downscaleFactor
        let referenceTemporalScale = referenceTemporalScaleFactor
            ?? referenceScaleConfiguration.temporalScaleFactor
        let referenceVideos = try parseLTXReferenceVideoConditionings(
            downscaleFactor: referenceSpatialScale,
            temporalScaleFactor: referenceTemporalScale
        )
        if skipStage2, referenceVideos.isEmpty {
            throw ValidationError("--skip-stage-2 requires --video-conditioning.")
        }
        if highQualityHDR, hdrLoRAConfiguration == nil {
            throw ValidationError("--high-quality-hdr requires an HDR IC-LoRA with hdr_transform metadata.")
        }
        let hdrICLoRA: LTXHDRICLoRAOptions?
        if hdrLoRAConfiguration != nil {
            guard !referenceVideos.isEmpty else {
                throw ValidationError("An HDR IC-LoRA requires at least one --video-conditioning reference.")
            }
            guard effectiveOutputMode == .videoOnly else {
                throw ValidationError("HDR IC-LoRA is a video-only pipeline; use --output-mode video-only.")
            }
            hdrICLoRA = LTXHDRICLoRAOptions(highQuality: highQualityHDR)
        } else {
            hdrICLoRA = nil
        }
        if skipHDRMP4, hdrICLoRA == nil {
            throw ValidationError("--skip-mp4 is available only for the dedicated HDR IC-LoRA pipeline.")
        }
        let resolvedLTXWidth = hdrICLoRA == nil ? resolvedWidth : width
        let resolvedLTXHeight = hdrICLoRA == nil ? resolvedHeight : height
        let textEmbeddingsURL = textEmbeddings.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        }
        if let textEmbeddingsURL {
            guard hdrICLoRA != nil else {
                throw ValidationError("--text-embeddings is available for the dedicated HDR IC-LoRA pipeline.")
            }
            guard FileManager.default.fileExists(atPath: textEmbeddingsURL.path) else {
                throw ValidationError("LTX text embeddings file not found: \(textEmbeddingsURL.path)")
            }
            if enhancePrompt || !autoDuration.isEmpty {
                throw ValidationError("--text-embeddings cannot be combined with --enhance-prompt or --auto-duration.")
            }
        }
        let generationPrompt: String
        if enhancePrompt {
            let enhancerRoot = promptEnhancerModelRoot.map {
                URL(fileURLWithPath: $0).standardizedFileURL
            }
            if !quiet {
                let mode = sourceImageURL == nil ? "text-to-video" : "image-to-video"
                CLIStderr.write("Enhancing prompt with native Gemma-4 (\(mode))\n")
            }
            generationPrompt = try await LTXPromptEnhancer.enhance(
                prompt: trimmedPrompt,
                modelID: promptEnhancerModel,
                modelRoot: enhancerRoot,
                referenceImage: sourceImageURL
            )
            if !quiet { CLIStderr.write("Enhanced prompt: \(generationPrompt)\n") }
        } else {
            generationPrompt = trimmedPrompt
        }
        if isMiniMaxH3ModelRoot(resolvedRootURL) {
            guard sourceAudioURL == nil else {
                throw ValidationError("MiniMax-H3 FL2VA generates its own synchronized audio and does not accept --audio.")
            }
            if timings || timingsOutput != nil {
                throw ValidationError("--timings and --timings-output are not available for MiniMax-H3 yet.")
            }
            let h3Resources = MiniMaxH3Resources(rootURL: resolvedRootURL)
            let h3Configuration = try h3Resources.loadConfiguration()
            let h3AdapterBaseModelID = h3Configuration.task == MiniMaxH3TurboAdapter.Task.ref2va.rawValue
                ? ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue
                : ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue
            let resolvedH3Adapter = try ManagedAdapterArgumentResolver.resolve(
                h3Adapter,
                baseModelID: h3AdapterBaseModelID
            ).map { URL(fileURLWithPath: $0).standardizedFileURL }
            let h3AdapterRecipe = resolvedH3Adapter.map(MiniMaxH3TurboAdapter.inferenceRecipe(for:))
            if let h3AdapterRecipe,
               !h3AdapterRecipe.supports(task: h3Configuration.task) {
                throw ValidationError(
                    "MiniMax-H3 adapter \(h3AdapterRecipe.name) requires \(h3AdapterRecipe.task.rawValue), not \(h3Configuration.task)."
                )
            }
            if h3AdapterRecipe?.task == .fl2va, !h3Resources.usesShardedBF16Transformer {
                throw ValidationError("MiniMax-H3 FL2VA adapters require the BF16 FL2VA model.")
            }
            if h3AdapterRecipe?.task == .ref2va, h3WeightMode == .quantized {
                throw ValidationError(
                    "MiniMax-H3 Ref2VA Turbo requires resident BF16 weights; use --h3-weight-mode resident-bf16."
                )
            }
            let parsedReferences = try parseMiniMaxH3References()
            let parsedFrameInputs = try parseMiniMaxH3FrameInputs()
            if h3Configuration.task == "fl2va", !parsedReferences.isEmpty {
                throw ValidationError("--reference requires a MiniMax-H3 Ref2VA model root.")
            }
            if h3Configuration.task == "ref2va" {
                guard sourceImageURL == nil,
                      endImageURL == nil,
                      parsedFrameInputs.isEmpty else {
                    throw ValidationError(
                        "MiniMax-H3 Ref2VA uses ordered --reference inputs, not FL2VA frame conditions."
                    )
                }
                guard !parsedReferences.isEmpty else {
                    throw ValidationError("MiniMax-H3 Ref2VA requires at least one --reference.")
                }
            }
            let h3Width = max(32, (width / 32) * 32)
            let h3Height = max(32, (height / 32) * 32)
            let requestedH3Frames = duration.map { Int(($0 * 24).rounded()) }
                ?? numFrames
                ?? 65
            let h3Frames = try MiniMaxH3Geometry.alignFrameCount(max(22, requestedH3Frames))
            let slidingWindowOptions = try h3WindowFrames.map {
                try MiniMaxH3SlidingWindowOptions(
                    totalFrameCount: h3Frames,
                    windowFrameCount: $0,
                    overlapFrameCount: h3WindowOverlap
                )
            }
            if !quiet {
                CLIStderr.write("Engine: native MiniMax-H3 \(h3Configuration.task.uppercased())\n")
                CLIStderr.write("Model root: \(resolvedRootURL.path)\n")
                if let resolvedH3Adapter {
                    CLIStderr.write("Adapter: \(resolvedH3Adapter.path) (strength \(h3AdapterStrength))\n")
                }
                if h3Width != width || h3Height != height {
                    CLIStderr.write("Adjusted MiniMax-H3 size to \(h3Width)x\(h3Height) (must be divisible by 32)\n")
                }
                if let h3RenderWidth, let h3RenderHeight {
                    CLIStderr.write(
                        "MiniMax-H3 internal render: \(h3RenderWidth)x\(h3RenderHeight); "
                            + "high-quality upscale to \(h3Width)x\(h3Height)\n"
                    )
                }
                if h3Frames != requestedH3Frames {
                    CLIStderr.write("Adjusted MiniMax-H3 frame count to \(h3Frames) (must have form 17*n+5)\n")
                }
                if let slidingWindowOptions {
                    let plan = MiniMaxH3SlidingWindowPlan(options: slidingWindowOptions)
                    CLIStderr.write(
                        "Sliding windows: \(plan.windows.count) x "
                            + "\(slidingWindowOptions.windowFrameCount) frames, "
                            + "overlap \(slidingWindowOptions.overlapFrameCount)\n"
                    )
                }
            }
            try MLXBundleSupport.ensureAvailable(quiet: quiet)
            let h3Options = try MiniMaxH3GenerationOptions(
                prompt: generationPrompt,
                width: h3Width,
                height: h3Height,
                renderWidth: h3RenderWidth,
                renderHeight: h3RenderHeight,
                numFrames: h3Frames,
                steps: steps,
                seed: UInt64(bitPattern: Int64(seed ?? 42)),
                transformerWeightMode: h3WeightMode.generationMode,
                accelerationMode: h3Acceleration.generationMode,
                adapterURL: resolvedH3Adapter,
                adapterStrength: h3AdapterStrength,
                firstFrameURL: sourceImageURL,
                lastFrameURL: endImageURL,
                frameInputs: parsedFrameInputs,
                references: parsedReferences
            )
            let generator = MiniMaxH3Generator(retainsRuntime: slidingWindowOptions != nil)
            let reportsProgress = !quiet
            let progressHandler: @Sendable (MiniMaxH3GenerationProgress) -> Void = { progress in
                guard reportsProgress else { return }
                if progress.stage == .denoising {
                    CLIStderr.write("Denoising \(progress.stepIndex + 1)/\(progress.totalSteps)\n")
                } else {
                    CLIStderr.write("\(progress.stage.rawValue)\n")
                }
            }
            let wiredMemoryTicket = MiniMaxH3WiredMemoryPolicy().ticket(
                size: miniMaxH3WiredMemoryTargetBytes()
            )
            let result = try await wiredMemoryTicket.withWiredLimit {
                try Stream.withNewDefaultStream {
                    if let slidingWindowOptions {
                        return try generator.generateSlidingWindows(
                            options: h3Options,
                            slidingWindowOptions: slidingWindowOptions,
                            resources: h3Resources,
                            windowHandler: { index, count in
                                guard reportsProgress else { return }
                                CLIStderr.write("H3 window \(index + 1)/\(count)\n")
                            },
                            progressHandler: progressHandler
                        )
                    }
                    return try generator.generate(
                        options: h3Options,
                        resources: h3Resources,
                        progressHandler: progressHandler
                    )
                }
            }
            try LTXVideoMP4Writer.writeMP4(
                frames: result.frames,
                fps: MiniMaxH3Geometry.framesPerSecond,
                to: outputURL,
                audioWaveform: result.audio,
                audioSampleRate: MiniMaxH3AudioVAE.samplingRate
            )
            if !quiet { CLIStderr.write("Saved: \(outputURL.path)\n") }
            print(outputURL.path)
            return
        }
        if h3Adapter != nil {
            throw ValidationError("--h3-adapter can only be used with a MiniMax-H3 model.")
        }
        if !h3FrameArguments.isEmpty || h3WindowFrames != nil {
            throw ValidationError(
                "--h3-frame and --h3-window-frames can only be used with a MiniMax-H3 model."
            )
        }
        if !references.isEmpty {
            throw ValidationError("--reference is only supported by MiniMax-H3 Ref2VA model roots.")
        }
        let ltxImageConditionings = try parseLTXImageConditionings()
        if (!ltxImageConditionings.isEmpty
            || numGeneratedKeyframes > 0
            || !generatedKeyframeIndices.isEmpty),
           !isLTX25ModelRoot(resolvedRootURL) {
            throw ValidationError(
                "LTX image conditioning and generated keyframes require an official LTX 2.5 model root."
            )
        }
        if let sourceAudioURL {
            if ltxPreset == .hq
                || ltxPipeline != .twoStage
                || ltxSampler != nil
                || !ltxSigmas.isEmpty
                || !ltxStage2Sigmas.isEmpty
                || distilledLoRAStrengthStage1 != nil
                || distilledLoRAStrengthStage2 != nil {
                throw ValidationError(
                    "LTX pipeline, preset, sampler, sigma, and distilled-LoRA controls select generated full-model AV, not source-audio A2Vid."
                )
            }
            try validateNativeAudioToVideoModelRoot(resolvedRootURL)
            try MLXBundleSupport.ensureAvailable(quiet: quiet)
            try await Stream.withNewDefaultStream {
                try await runNativeAudioToVideoGenerate(
                    prompt: generationPrompt,
                    negativePrompt: negativePrompt,
                    audioURL: sourceAudioURL,
                    audioStartTime: audioStartTime,
                    audioMaxDuration: audioMaxDuration,
                    width: resolvedWidth,
                    height: resolvedHeight,
                    numFrames: resolvedNumFrames,
                    fps: fps,
                    seed: resolvedSeed,
                    inferenceSteps: a2vSteps,
                    a2vGuidanceScale: a2vGuidanceScale,
                    videoCFGGuidanceScale: videoCFGGuidanceScale,
                    sourceImageURL: sourceImageURL,
                    imageStrength: imageStrength,
                    endImageURL: endImageURL,
                    endImageStrength: endImageStrength,
                    imageConditionings: ltxImageConditionings,
                    generatedKeyframeCount: numGeneratedKeyframes,
                    generatedKeyframeIndices: generatedKeyframeIndices,
                    hdrColorSpace: resolvedHDRColorSpace,
                    hdrTransfer: resolvedHDRTransfer,
                    videoDecoder: resolvedVideoDecoder,
                    modelRoot: resolvedRootURL,
                    outputURL: outputURL
                )
            }
            return
        }
        if isWan2ModelRoot(resolvedRootURL) {
            if timings || timingsOutput != nil {
                throw ValidationError(
                    "--timings and --timings-output are available for native LTX generation, not Wan2.2 TI2V."
                )
            }
            guard let sourceImageURL else {
                throw ValidationError("Wan2.2 TI2V requires --image.")
            }
            guard endImageURL == nil else {
                throw ValidationError("Wan2.2 TI2V does not support --end-image yet.")
            }
            guard fps.rounded() == fps else {
                throw ValidationError("Wan2.2 TI2V requires an integer --fps value.")
            }
            let wanFPS = Int(fps)
            let wanWidth = max(32, (width / 32) * 32)
            let wanHeight = max(32, (height / 32) * 32)
            let requestedWanFrames = duration.map { nearestWanFrameCount(duration: $0, fps: fps) }
                ?? numFrames
                ?? 65
            let wanFrames = max(5, ((requestedWanFrames - 1) / 4) * 4 + 1)
            if !quiet {
                if wanWidth != width || wanHeight != height {
                    CLIStderr.write("Adjusted Wan size to \(wanWidth)x\(wanHeight) (must be divisible by 32)\n")
                }
                if let duration {
                    let resolvedSeconds = Double(wanFrames) / Double(fps)
                    CLIStderr.write(
                        "Resolved Wan duration \(String(format: "%.2f", duration))s to \(wanFrames) frames at \(fps) fps (~\(String(format: "%.2f", resolvedSeconds))s; must satisfy 4n+1)\n"
                    )
                } else if let numFrames, wanFrames != numFrames {
                    CLIStderr.write("Adjusted Wan frame count to \(wanFrames) (must satisfy 4n+1)\n")
                }
            }
            try await runNativeWanGenerate(
                prompt: generationPrompt,
                negativePrompt: negativePrompt ?? Wan2Resources.defaultNegativePrompt,
                width: wanWidth,
                height: wanHeight,
                numFrames: wanFrames,
                steps: steps ?? 40,
                guidanceScale: guidanceScale,
                shift: shift,
                fps: wanFPS,
                seed: resolvedSeed,
                sourceImageURL: sourceImageURL,
                modelRoot: resolvedRootURL,
                outputURL: outputURL
            )
            return
        }

        if !quiet {
            if resolvedLTXWidth != width || resolvedLTXHeight != height {
                CLIStderr.write(
                    "Adjusted size to \(resolvedLTXWidth)x\(resolvedLTXHeight) (must be divisible by 64)\n"
                )
            }
            if let duration {
                let resolvedSeconds = Double(resolvedNumFrames) / Double(fps)
                CLIStderr.write(
                    "Resolved duration \(String(format: "%.2f", duration))s to \(resolvedNumFrames) frames at \(fps) fps (~\(String(format: "%.2f", resolvedSeconds))s; must satisfy 8n+1)\n"
                )
            } else if let numFrames, resolvedNumFrames != numFrames {
                CLIStderr.write("Adjusted frame count to \(resolvedNumFrames) (must satisfy 8n+1)\n")
            }
            if variant == .unifiedAV && fps != 24 {
                CLIStderr.write(
                    "Warning: LTX unified AV is trained for 24 fps; --fps \(fps) can make generated motion look time-stretched relative to audio.\n"
                )
            }
        }

        let nativeRoute = resolveLTXVideoGenerationRoute(
            variant: variant,
            modelRoot: resolvedRootURL
        )
        let usesAdvancedLTXPipelineControls = ltxPreset == .hq
            || ltxPipeline != .twoStage
            || ltxSampler != nil
            || !ltxSigmas.isEmpty
            || !ltxStage2Sigmas.isEmpty
            || distilledLoRAStrengthStage1 != nil
            || distilledLoRAStrengthStage2 != nil
        if usesAdvancedLTXPipelineControls,
           !isLTX25FullModelRoot(resolvedRootURL) {
            throw ValidationError(
                "LTX pipeline, sampler, preset, sigma, and distilled-LoRA controls require the full LTX 2.5 dev + distilled-LoRA model root."
            )
        }
        let usesHQPreset = ltxPreset == .hq
        if usesHQPreset, ltxPipeline != .twoStage {
            throw ValidationError("The hq preset is the official two-stage Res2s pipeline.")
        }
        let resolvedDistilledLoRAStrengthStage1 = distilledLoRAStrengthStage1
            ?? (dfr ? 1 : (usesHQPreset ? 0.25 : 0))
        let resolvedDistilledLoRAStrengthStage2 = distilledLoRAStrengthStage2
            ?? (usesHQPreset ? 0.5 : (ltxPipeline == .devOneStage ? 0 : 1))
        guard resolvedDistilledLoRAStrengthStage1.isFinite,
              resolvedDistilledLoRAStrengthStage2.isFinite else {
            throw ValidationError("Distilled LoRA strengths must be finite.")
        }
        if ltxPipeline == .devOneStage,
           resolvedDistilledLoRAStrengthStage1 != 0 || resolvedDistilledLoRAStrengthStage2 != 0 {
            throw ValidationError("The dev-one-stage pipeline runs without the distilled LoRA.")
        }
        let resolvedVideoSTGBlocks = Set(usesHQPreset ? [] : (videoSTGBlocks.isEmpty ? [28] : videoSTGBlocks))
        let resolvedAudioSTGBlocks = Set(usesHQPreset ? [] : (audioSTGBlocks.isEmpty ? [28] : audioSTGBlocks))
        let resolvedSampler = LTXSamplerConfiguration(
            mode: ltxSampler ?? (usesHQPreset ? .res2s : .euler),
            eta: ltxSamplerEta,
            noiseSeedOffset: usesHQPreset ? LTXSamplerConfiguration.hq.noiseSeedOffset : 10_000,
            substepNoiseSeedOffset: usesHQPreset
                ? LTXSamplerConfiguration.hq.substepNoiseSeedOffset
                : 20_000,
            res2sBongMath: !noRes2sBongMath,
            res2sBongMathMaxIterations: res2sBongMaxIterations,
            gradientEstimationGamma: gradientEstimationGamma
        )
        let resolvedVideoGuidance = LTXMultiModalGuidance(
            classifierFreeScale: videoCFGGuidanceScale,
            spatioTemporalScale: usesHQPreset ? 0 : videoSTGScale,
            rescale: usesHQPreset ? 0.45 : videoGuidanceRescale,
            modalityScale: a2vGuidanceScale,
            spatioTemporalBlocks: resolvedVideoSTGBlocks,
            skipStep: videoGuidanceSkipStep
        )
        let resolvedAudioGuidance = LTXMultiModalGuidance(
            classifierFreeScale: audioCFGGuidanceScale,
            spatioTemporalScale: usesHQPreset ? 0 : audioSTGScale,
            rescale: usesHQPreset ? 1 : audioGuidanceRescale,
            modalityScale: v2aGuidanceScale,
            spatioTemporalBlocks: resolvedAudioSTGBlocks,
            skipStep: audioGuidanceSkipStep
        )
        if (timings || timingsOutput != nil), !nativeRoute.supportsPhaseTimings {
            throw ValidationError(
                "--timings and --timings-output require a split LTX model, --quality final, --output-mode audio-video, or --audio."
            )
        }
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        try await Stream.withNewDefaultStream {
            try await runNativeGenerate(
                prompt: generationPrompt,
                width: resolvedLTXWidth,
                height: resolvedLTXHeight,
                numFrames: resolvedNumFrames,
                fps: fps,
                seed: resolvedSeed,
                outputMode: effectiveOutputMode,
                autoDurationRange: effectiveAutoDurationRange(
                    modelRoot: resolvedRootURL,
                    hasSourceAudio: false
                ),
                negativePrompt: negativePrompt,
                inferenceSteps: usesHQPreset ? 15 : a2vSteps,
                videoGuidance: resolvedVideoGuidance,
                audioGuidance: resolvedAudioGuidance,
                sigmas: ltxSigmas.isEmpty ? nil : ltxSigmas,
                stage2Sigmas: ltxStage2Sigmas.isEmpty ? nil : ltxStage2Sigmas,
                sampler: resolvedSampler,
                pipeline: ltxPipeline,
                distilledLoRAStrengthStage1: resolvedDistilledLoRAStrengthStage1,
                distilledLoRAStrengthStage2: resolvedDistilledLoRAStrengthStage2,
                sourceImageURL: sourceImageURL,
                imageStrength: imageStrength,
                endImageURL: endImageURL,
                endImageStrength: endImageStrength,
                imageConditionings: ltxImageConditionings,
                generatedKeyframeCount: numGeneratedKeyframes,
                generatedKeyframeIndices: generatedKeyframeIndices,
                dfrOptions: dfr
                    ? LTX25DFROptions(
                        temporalUpsampleRounds: temporalUpsampleRounds,
                        detailingLoRAs: detailingLoRAs,
                        detailingReferenceDownscaleFactor: detailingReferenceDownscaleFactor
                    )
                    : nil,
                referenceVideos: referenceVideos,
                loras: loras,
                skipStage2: skipStage2,
                precomputedTextEmbeddingsURL: textEmbeddingsURL,
                hdrColorSpace: resolvedHDRColorSpace,
                hdrTransfer: resolvedHDRTransfer,
                hdrICLoRA: hdrICLoRA,
                vaeSpatialTileSize: vaeSpatialTileSize,
                vaeSpatialTileOverlap: vaeSpatialTileOverlap,
                skipHDRMP4: skipHDRMP4,
                videoDecoder: resolvedVideoDecoder,
                modelRoot: resolvedModelRoot,
                outputURL: outputURL
            )
        }
    }

    private func validateProductSelection(modelRoot: URL) throws {
        if isMiniMaxH3ModelRoot(modelRoot) {
            if quality != nil || outputMode != nil || legacyVariant != nil {
                throw ValidationError("--quality, --output-mode, and --variant select LTX behavior and cannot be combined with MiniMax-H3.")
            }
            return
        }
        if isWan2ModelRoot(modelRoot) {
            if quality != nil || outputMode != nil {
                throw ValidationError("--quality and --output-mode currently select native LTX generation, not Wan2.2 TI2V.")
            }
            return
        }
        guard let quality else { return }
        switch quality {
        case .draft:
            guard !isLTX23FullModelRoot(modelRoot),
                  !isLTX23AudioToVideoModelRoot(modelRoot),
                  !isLTX25FullModelRoot(modelRoot),
                  !isLTX25ModelRoot(modelRoot) else {
                throw ValidationError(
                    "--quality draft requires \(ModelResolver.ModelID.ltxVideo23AVMLX.rawValue), not a final-quality checkpoint."
                )
            }
        case .final:
            let supportsRequestedFinalPath = isLTX23AudioToVideoModelRoot(modelRoot)
                || isLTX25ModelRoot(modelRoot)
            guard supportsRequestedFinalPath else {
                throw ValidationError(
                    "--quality final requires \(ModelResolver.ModelID.ltxVideo23FullMLX.rawValue) or an official LTX 2.5 root."
                )
            }
        }
    }

    private func isWan2ModelRoot(_ rootURL: URL) -> Bool {
        let resources = Wan2Resources(rootURL: rootURL)
        return resources.validate().isEmpty && (try? resources.loadConfiguration()) != nil
    }

    private func isMiniMaxH3ModelRoot(_ rootURL: URL) -> Bool {
        let resources = MiniMaxH3Resources(rootURL: rootURL)
        return resources.validate().isEmpty && (try? resources.loadConfiguration()) != nil
    }

    private func parseMiniMaxH3References() throws -> [MiniMaxH3ReferenceInput] {
        try references.map { raw in
            guard let separator = raw.firstIndex(of: ":") else {
                throw ValidationError("--reference must be image:path, video:path, or audio:path (got \(raw)).")
            }
            let rawKind = String(raw[..<separator]).lowercased()
            let rawPath = String(raw[raw.index(after: separator)...])
            guard let kind = MiniMaxH3ReferenceKind(rawValue: rawKind), !rawPath.isEmpty else {
                throw ValidationError("--reference must be image:path, video:path, or audio:path (got \(raw)).")
            }
            let url = URL(fileURLWithPath: rawPath).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Reference file not found: \(url.path)")
            }
            return MiniMaxH3ReferenceInput(kind: kind, url: url)
        }
    }

    private func parseLTXImageConditionings() throws -> [LTXVideoConditioningInput] {
        try imageConditioningArguments.map { raw in
            let parts = raw.split(separator: ":", maxSplits: 3, omittingEmptySubsequences: false)
            guard parts.count >= 2,
                  let frame = Int(parts[0]),
                  frame >= 0 else {
                throw ValidationError(
                    "--image-conditioning must be PIXEL_FRAME:PATH[:STRENGTH[:CRF]] (got \(raw))."
                )
            }
            let path = String(parts[1])
            guard !path.isEmpty else {
                throw ValidationError(
                    "--image-conditioning must include an image path (got \(raw))."
                )
            }
            let strength: Float
            if parts.count == 3 {
                guard let parsed = Float(parts[2]), (0...1).contains(parsed) else {
                    throw ValidationError(
                        "--image-conditioning strength must be in [0, 1] (got \(parts[2]))."
                    )
                }
                strength = parsed
            } else {
                strength = 1
            }
            let crf: Int?
            if parts.count == 4 {
                guard let parsed = Int(parts[3]), (0...51).contains(parsed) else {
                    throw ValidationError("--image-conditioning CRF must be in 0...51 (got \(parts[3])).")
                }
                crf = parsed
            } else {
                crf = nil
            }
            let url = URL(fileURLWithPath: path).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Image conditioning file not found: \(url.path)")
            }
            return LTXVideoConditioningInput(
                imageURL: url,
                pixelFrameIndex: frame,
                strength: strength,
                crf: crf
            )
        }
    }

    private func parseLTXLoRAConfigurations(
        _ arguments: [String],
        optionName: String,
        baseModelID: String
    ) throws -> [LTXLoRAConfiguration] {
        try arguments.map { raw in
            let separator = raw.lastIndex(of: "=")
            let path: String
            let strength: Float
            if let separator {
                path = String(raw[..<separator])
                let rawStrength = String(raw[raw.index(after: separator)...])
                guard let parsed = Float(rawStrength), parsed.isFinite else {
                    throw ValidationError(
                        "\(optionName) strength must be finite (got \(rawStrength))."
                    )
                }
                strength = parsed
            } else {
                path = raw
                strength = 1
            }
            guard !path.isEmpty else {
                throw ValidationError("\(optionName) must be PATH[=STRENGTH].")
            }
            let resolvedPath = try ManagedAdapterArgumentResolver.resolve(
                path,
                baseModelID: baseModelID
            ) ?? path
            let url = URL(fileURLWithPath: resolvedPath).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("LTX LoRA file not found: \(url.path)")
            }
            return LTXLoRAConfiguration(url: url, strength: strength)
        }
    }

    private func parseLTXReferenceVideoConditionings(
        downscaleFactor: Int,
        temporalScaleFactor: Int
    ) throws -> [LTXReferenceVideoConditioningInput] {
        let attentionMaskURL = conditioningAttentionMask.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        }
        if let attentionMaskURL,
           !FileManager.default.fileExists(atPath: attentionMaskURL.path) {
            throw ValidationError("IC-LoRA attention mask video not found: \(attentionMaskURL.path)")
        }
        return try videoConditioningArguments.map { raw in
            let separator = raw.lastIndex(of: "=")
            let path: String
            let strength: Float
            if let separator {
                path = String(raw[..<separator])
                let rawStrength = String(raw[raw.index(after: separator)...])
                guard let parsed = Float(rawStrength), (0...1).contains(parsed) else {
                    throw ValidationError(
                        "--video-conditioning strength must be in [0, 1] (got \(rawStrength))."
                    )
                }
                strength = parsed
            } else {
                path = raw
                strength = 1
            }
            guard !path.isEmpty else {
                throw ValidationError("--video-conditioning must be PATH[=STRENGTH].")
            }
            let url = URL(fileURLWithPath: path).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("IC-LoRA reference video not found: \(url.path)")
            }
            return LTXReferenceVideoConditioningInput(
                videoURL: url,
                strength: strength,
                attentionStrength: conditioningAttentionStrength == 1
                    ? nil
                    : conditioningAttentionStrength,
                attentionMaskVideoURL: attentionMaskURL,
                downscaleFactor: downscaleFactor,
                temporalScaleFactor: temporalScaleFactor
            )
        }
    }

    private func parseMiniMaxH3FrameInputs() throws -> [MiniMaxH3FrameInput] {
        try h3FrameArguments.map { raw in
            guard let separator = raw.firstIndex(of: ":") else {
                throw ValidationError("--h3-frame must be zero-based FRAME:PATH (got \(raw)).")
            }
            let rawIndex = String(raw[..<separator])
            let rawPath = String(raw[raw.index(after: separator)...])
            guard let frameIndex = Int(rawIndex), frameIndex >= 0, !rawPath.isEmpty else {
                throw ValidationError("--h3-frame must be zero-based FRAME:PATH (got \(raw)).")
            }
            let url = URL(fileURLWithPath: rawPath).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("H3 frame image not found: \(url.path)")
            }
            return MiniMaxH3FrameInput(frameIndex: frameIndex, url: url)
        }
    }

    private func runNativeWanGenerate(
        prompt: String,
        negativePrompt: String,
        width: Int,
        height: Int,
        numFrames: Int,
        steps: Int,
        guidanceScale: Float,
        shift: Float,
        fps: Int,
        seed: Int,
        sourceImageURL: URL,
        modelRoot: URL,
        outputURL: URL
    ) async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let options = try Wan2GenerationOptions(
            prompt: prompt,
            negativePrompt: negativePrompt,
            sourceImageURL: sourceImageURL,
            outputURL: outputURL,
            width: width,
            height: height,
            numFrames: numFrames,
            steps: steps,
            guidanceScale: guidanceScale,
            shift: shift,
            seed: UInt64(bitPattern: Int64(seed)),
            fps: fps
        )
        if !quiet {
            CLIStderr.write("Engine: native Wan2.2 TI2V\n")
            CLIStderr.write("Model root: \(modelRoot.path)\n")
            CLIStderr.write("Mode: image-to-video\n")
        }
        let reportsProgress = !quiet
        let generator = Wan2TI2VGenerator()
        let result = try await generator.generate(
            options: options,
            resources: Wan2Resources(rootURL: modelRoot),
            progressHandler: { progress in
                guard reportsProgress else { return }
                if progress.stage == .denoising {
                    CLIStderr.write("Denoising \(progress.stepIndex + 1)/\(progress.totalSteps)\n")
                } else {
                    CLIStderr.write("\(progress.stage.rawValue)\n")
                }
            }
        )
        if !quiet {
            CLIStderr.write("Decoded frames shape: \(shapeString(result.frames.shape))\n")
            CLIStderr.write("Writing MP4...\n")
        }
        try LTXVideoMP4Writer.writeMP4(frames: result.frames, fps: fps, to: outputURL)
        if !quiet {
            CLIStderr.write("Saved: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    private func runNativeAudioToVideoGenerate(
        prompt: String,
        negativePrompt: String?,
        audioURL: URL,
        audioStartTime: Double,
        audioMaxDuration: Double?,
        width: Int,
        height: Int,
        numFrames: Int,
        fps: Double,
        seed: Int,
        inferenceSteps: Int,
        a2vGuidanceScale: Float,
        videoCFGGuidanceScale: Float,
        sourceImageURL: URL?,
        imageStrength: Float,
        endImageURL: URL?,
        endImageStrength: Float,
        imageConditionings: [LTXVideoConditioningInput],
        generatedKeyframeCount: Int,
        generatedKeyframeIndices: [Int],
        hdrColorSpace: LTXHDRColorSpace?,
        hdrTransfer: LTXHDRTransfer,
        videoDecoder: LTXVideoDecoderKind,
        modelRoot: URL,
        outputURL: URL
    ) async throws {
        let endToEndStart = videoMonotonicSeconds()
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        if !quiet {
            let version = isLTX25FullModelRoot(modelRoot) ? "LTX 2.5" : "LTX 2.3"
            CLIStderr.write("Engine: native \(version) A2Vid\n")
            CLIStderr.write("Model root: \(modelRoot.path)\n")
            CLIStderr.write(
                "Mode: \(sourceImageURL == nil ? "audio-to-video" : "audio-and-image-to-video")\n"
            )
            CLIStderr.write("Source audio: \(audioURL.path) at \(audioStartTime)s\n")
            CLIStderr.write("Loading \(version) dev + distilled-LoRA model...\n")
        }

        let generator = LTXUnifiedAVGenerator()
        do {
            let loadTimings: LTXLoadTimings
            if isLTX23FullModelRoot(modelRoot) || isLTX25FullModelRoot(modelRoot) {
                loadTimings = try await generator.loadFull(
                    modelRoot: modelRoot,
                    videoDecoder: videoDecoder,
                    videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                )
            } else {
                loadTimings = try await generator.loadAudioToVideo(
                    modelRoot: modelRoot,
                    videoDecoder: videoDecoder,
                    videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                )
            }
            if !quiet {
                CLIStderr.write("Running guided stage 1 and distilled-LoRA stage 2...\n")
            }
            let result = try await generator.generateAudioToVideo(
                options: LTXAudioToVideoGenerationOptions(
                    prompt: prompt,
                    negativePrompt: negativePrompt ?? LTXAudioToVideoGenerationOptions.defaultNegativePrompt,
                    audioURL: audioURL,
                    audioStartTime: audioStartTime,
                    audioMaxDuration: audioMaxDuration,
                    width: width,
                    height: height,
                    numFrames: numFrames,
                    fps: fps,
                    seed: seed,
                    inferenceSteps: inferenceSteps,
                    guidance: LTXAudioToVideoGuidance(
                        classifierFreeScale: videoCFGGuidanceScale,
                        audioToVideoScale: a2vGuidanceScale
                    ),
                    sourceImageURL: sourceImageURL,
                    imageStrength: imageStrength,
                    endImageURL: endImageURL,
                    endImageStrength: endImageStrength,
                    imageConditionings: imageConditionings,
                    generatedKeyframeCount: generatedKeyframeCount,
                    generatedKeyframeIndices: generatedKeyframeIndices,
                    hdrColorSpace: hdrColorSpace,
                    hdrTransfer: hdrTransfer
                )
            )
            let unloadStart = videoMonotonicSeconds()
            await generator.unload()
            let unloadSeconds = videoMonotonicSeconds() - unloadStart

            if !quiet {
                CLIStderr.write("Decoded frames shape: \(shapeString(result.frames.shape))\n")
                CLIStderr.write("Writing MP4 with the original source-audio segment...\n")
            }
            let writeStart = videoMonotonicSeconds()
            if let hdrOutput = result.hdrOutput, let hdrColorSpace {
                try LTXHDRVideoWriter.write(
                    hdrOutput,
                    colorSpace: hdrColorSpace,
                    fps: fps,
                    to: outputURL,
                    sourceAudio: result.sourceAudio
                )
            } else {
                try LTXVideoMP4Writer.writeMP4(
                    frames: result.frames,
                    fps: fps,
                    to: outputURL,
                    sourceAudio: result.sourceAudio
                )
            }
            guard await mediaHasAudioTrack(at: outputURL) else {
                throw ValidationError("A2Vid output has no audio track at \(outputURL.path)")
            }
            let mp4WriteSeconds = videoMonotonicSeconds() - writeStart
            try emitLTXVideoTimingReport(
                LTXVideoTimingReport(
                    mode: "audio-to-video",
                    modelRoot: modelRoot.path,
                    residentModelReused: false,
                    load: loadTimings,
                    generation: result.timings,
                    unloadSeconds: unloadSeconds,
                    mp4WriteSeconds: mp4WriteSeconds,
                    totalSeconds: videoMonotonicSeconds() - endToEndStart
                ),
                printToStandardError: timings,
                outputPath: timingsOutput
            )
        } catch {
            await generator.unload()
            throw error
        }

        if !quiet {
            CLIStderr.write("Saved: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    private func runNativeGenerate(
        prompt: String,
        width: Int,
        height: Int,
        numFrames: Int,
        fps: Double,
        seed: Int,
        outputMode: LTXVideoOutputMode,
        autoDurationRange: LTX25AutoDuration?,
        negativePrompt: String?,
        inferenceSteps: Int,
        videoGuidance: LTXMultiModalGuidance,
        audioGuidance: LTXMultiModalGuidance,
        sigmas: [Float]?,
        stage2Sigmas: [Float]?,
        sampler: LTXSamplerConfiguration,
        pipeline: LTXGenerationPipeline,
        distilledLoRAStrengthStage1: Float,
        distilledLoRAStrengthStage2: Float,
        sourceImageURL: URL?,
        imageStrength: Float,
        endImageURL: URL?,
        endImageStrength: Float,
        imageConditionings: [LTXVideoConditioningInput],
        generatedKeyframeCount: Int,
        generatedKeyframeIndices: [Int],
        dfrOptions: LTX25DFROptions?,
        referenceVideos: [LTXReferenceVideoConditioningInput],
        loras: [LTXLoRAConfiguration],
        skipStage2: Bool,
        precomputedTextEmbeddingsURL: URL?,
        hdrColorSpace: LTXHDRColorSpace?,
        hdrTransfer: LTXHDRTransfer,
        hdrICLoRA: LTXHDRICLoRAOptions?,
        vaeSpatialTileSize: Int?,
        vaeSpatialTileOverlap: Int,
        skipHDRMP4: Bool,
        videoDecoder: LTXVideoDecoderKind,
        modelRoot: String,
        outputURL: URL
    ) async throws {
        let rootURL = URL(fileURLWithPath: modelRoot).standardizedFileURL
        let savedArtifactURL = skipHDRMP4
            ? outputURL.deletingLastPathComponent().appendingPathComponent(
                "\(outputURL.deletingPathExtension().lastPathComponent)_exr",
                isDirectory: true
            )
            : outputURL
        let route = resolveLTXVideoGenerationRoute(outputMode: outputMode, modelRoot: rootURL)
        if (timings || timingsOutput != nil), !route.supportsPhaseTimings {
            throw ValidationError(
                "--timings and --timings-output require a split LTX model, --quality final, --output-mode audio-video, or --audio."
            )
        }
        if route == .unifiedAV,
           isLTX23AudioToVideoModelRoot(rootURL),
           !isLTX23FullModelRoot(rootURL) {
            throw ValidationError(
                "This legacy A2Vid root has no vocoder for generated audio. Pull \(ModelResolver.ModelID.ltxVideo23FullMLX.rawValue)."
            )
        }
        try validateNativeModelRoot(rootURL)
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        if !quiet {
            CLIStderr.write("Engine: native\n")
            let isFinalQuality = isLTX23AudioToVideoModelRoot(rootURL) || isLTX25ModelRoot(rootURL)
            CLIStderr.write("Quality: \(isFinalQuality ? LTXVideoQuality.final.rawValue : LTXVideoQuality.draft.rawValue)\n")
            CLIStderr.write("Output mode: \(outputMode.rawValue)\n")
            CLIStderr.write("Runtime lane: \(route.rawValue)\n")
            CLIStderr.write("Model root: \(rootURL.path)\n")
            if isLTX25ModelRoot(rootURL) {
                CLIStderr.write("Video decoder: \(videoDecoder.rawValue)\n")
            }
            CLIStderr.write("Mode: \(sourceImageURL == nil ? "text-to-video" : "image-to-video")\n")
        }

        let makeUnifiedOptions: (Int) -> LTXUnifiedAVGenerationOptions = { resolvedFrames in
            LTXUnifiedAVGenerationOptions(
                prompt: prompt,
                negativePrompt: negativePrompt ?? LTXUnifiedAVGenerationOptions.defaultNegativePrompt,
                width: width,
                height: height,
                numFrames: resolvedFrames,
                fps: fps,
                seed: seed,
                inferenceSteps: inferenceSteps,
                videoGuidance: videoGuidance,
                audioGuidance: audioGuidance,
                sourceImageURL: sourceImageURL,
                imageStrength: imageStrength,
                endImageURL: endImageURL,
                endImageStrength: endImageStrength,
                imageConditionings: imageConditionings,
                generatedKeyframeCount: generatedKeyframeCount,
                generatedKeyframeIndices: generatedKeyframeIndices,
                referenceVideos: referenceVideos,
                loras: loras,
                dfr: dfrOptions,
                sigmas: sigmas,
                stage2Sigmas: stage2Sigmas,
                sampler: sampler,
                pipeline: pipeline,
                distilledLoRAStrengthStage1: distilledLoRAStrengthStage1,
                distilledLoRAStrengthStage2: distilledLoRAStrengthStage2,
                hdrColorSpace: hdrColorSpace,
                hdrTransfer: hdrTransfer,
                hdrICLoRA: hdrICLoRA,
                vaeSpatialTileSize: vaeSpatialTileSize,
                vaeSpatialTileOverlap: vaeSpatialTileOverlap,
                skipStage2: skipStage2,
                precomputedTextEmbeddingsURL: precomputedTextEmbeddingsURL
            )
        }

        switch route {
        case .legacyDistilledVideo:
            if !quiet {
                CLIStderr.write("Loading native distilled model...\n")
            }
            let generator = LTXDistilledLatentGenerator()
            do {
                try await generator.load(modelRoot: rootURL)
                if !quiet {
                    CLIStderr.write("Running native denoising + decode...\n")
                }
                let result = try await generator.generateVideo(
                    options: LTXDistilledLatentGenerationOptions(
                        prompt: prompt,
                        width: width,
                        height: height,
                        numFrames: numFrames,
                        fps: fps,
                        seed: seed,
                        sourceImageURL: sourceImageURL,
                        imageStrength: imageStrength,
                        endImageURL: endImageURL,
                        endImageStrength: endImageStrength
                    )
                )
                await generator.unload()

                if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
                    let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
                    let parent = base.deletingLastPathComponent()
                    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
                    let stem = base.lastPathComponent
                    try MLX.save(array: result.frames, url: parent.appendingPathComponent("\(stem)_frames.npy"))
                    try MLX.save(array: result.latents, url: parent.appendingPathComponent("\(stem)_latents.npy"))
                }

                if !quiet {
                    CLIStderr.write("Decoded frames shape: \(shapeString(result.frames.shape))\n")
                    CLIStderr.write("Writing MP4...\n")
                }
                try LTXVideoMP4Writer.writeMP4(
                    frames: result.frames,
                    fps: fps,
                    to: outputURL
                )
            } catch {
                await generator.unload()
                throw error
            }

        case .splitDistilledVideo:
            let endToEndStart = videoMonotonicSeconds()
            if !quiet {
                let version = isLTX25ModelRoot(rootURL) ? "LTX 2.5" : "LTX 2.3"
                CLIStderr.write("Loading native \(version) distilled model for video-only output...\n")
            }
            let generator = LTXUnifiedAVGenerator()
            do {
                let loadTimings = try await generator.loadVideoOnly(
                    modelRoot: rootURL,
                    videoDecoder: videoDecoder,
                    videoDecoderDType: hdrColorSpace == nil ? nil : .float32,
                    loadTextEncoder: precomputedTextEmbeddingsURL == nil
                )
                if !quiet {
                    CLIStderr.write(
                        "Running standalone distilled joint denoising + video decode (audio output disabled)...\n"
                    )
                }
                let resolvedFrames = try await resolveAutoDuration(
                    autoDurationRange,
                    prompt: prompt,
                    fps: fps,
                    generator: generator,
                    fallback: numFrames
                )
                let result = try await generator.generateVideoOnly(
                    options: makeUnifiedOptions(resolvedFrames)
                )
                let unloadStart = videoMonotonicSeconds()
                await generator.unload()
                let unloadSeconds = videoMonotonicSeconds() - unloadStart

                if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
                    let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
                    let parent = base.deletingLastPathComponent()
                    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
                    let stem = base.lastPathComponent
                    try MLX.save(array: result.frames, url: parent.appendingPathComponent("\(stem)_frames.npy"))
                    try MLX.save(array: result.videoLatents, url: parent.appendingPathComponent("\(stem)_latents.npy"))
                }

                if !quiet {
                    CLIStderr.write("Decoded frames shape: \(shapeString(result.frames.shape))\n")
                    CLIStderr.write("Writing video-only MP4...\n")
                }
                let writeStart = videoMonotonicSeconds()
                if let hdrOutput = result.hdrOutput, let hdrColorSpace {
                    try LTXHDRVideoWriter.write(
                        hdrOutput,
                        colorSpace: hdrColorSpace,
                        fps: result.playbackFPS,
                        to: outputURL
                    )
                } else {
                    try LTXVideoMP4Writer.writeMP4(
                        frames: result.frames,
                        fps: result.playbackFPS,
                        to: outputURL
                    )
                }
                let mp4WriteSeconds = videoMonotonicSeconds() - writeStart
                try emitLTXVideoTimingReport(
                    LTXVideoTimingReport(
                        mode: "standalone-distilled-video-only",
                        modelRoot: rootURL.path,
                        residentModelReused: false,
                        load: loadTimings,
                        generation: result.timings,
                        unloadSeconds: unloadSeconds,
                        mp4WriteSeconds: mp4WriteSeconds,
                        totalSeconds: videoMonotonicSeconds() - endToEndStart
                    ),
                    printToStandardError: timings,
                    outputPath: timingsOutput
                )
            } catch {
                await generator.unload()
                throw error
            }

        case .fullQualityVideo:
            let endToEndStart = videoMonotonicSeconds()
            if !quiet {
                let version = isLTX25FullModelRoot(rootURL) ? "LTX 2.5" : "LTX 2.3"
                CLIStderr.write("Loading native \(version) full-quality model for video-only output...\n")
            }
            let generator = LTXUnifiedAVGenerator()
            do {
                let loadTimings: LTXLoadTimings
                if hdrICLoRA != nil {
                    loadTimings = try await generator.loadVideoOnly(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: .float32,
                        loadTextEncoder: precomputedTextEmbeddingsURL == nil
                    )
                } else if dfrOptions != nil {
                    loadTimings = try await generator.loadDFR(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                } else if distilledLoRAStrengthStage1 != 0 {
                    loadTimings = try await generator.loadFullReusable(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                } else {
                    loadTimings = try await generator.loadFullVideoOnly(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                }
                if !quiet {
                    CLIStderr.write(
                        dfrOptions != nil
                            ? "Running LTX 2.5 DFR spatial detail and temporal refinement (audio output disabled)...\n"
                            : (pipeline == .devOneStage
                                ? "Running target-resolution dev one-stage generation (audio output disabled)...\n"
                                : "Running guided dev stage 1 + distilled-LoRA stage 2 (audio output disabled)...\n")
                    )
                }
                let resolvedFrames = try await resolveAutoDuration(
                    autoDurationRange,
                    prompt: prompt,
                    fps: fps,
                    generator: generator,
                    fallback: numFrames
                )
                let result = try await generator.generateVideoOnly(
                    options: makeUnifiedOptions(resolvedFrames)
                )
                let unloadStart = videoMonotonicSeconds()
                await generator.unload()
                let unloadSeconds = videoMonotonicSeconds() - unloadStart

                if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
                    let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
                    let parent = base.deletingLastPathComponent()
                    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
                    let stem = base.lastPathComponent
                    try MLX.save(array: result.frames, url: parent.appendingPathComponent("\(stem)_frames.npy"))
                    try MLX.save(array: result.videoLatents, url: parent.appendingPathComponent("\(stem)_latents.npy"))
                }

                if !quiet {
                    CLIStderr.write("Decoded frames shape: \(shapeString(result.frames.shape))\n")
                    CLIStderr.write(
                        skipHDRMP4
                            ? "Writing full-quality half-float EXR sequence...\n"
                            : "Writing full-quality video-only MP4...\n"
                    )
                }
                let writeStart = videoMonotonicSeconds()
                if let hdrOutput = result.hdrOutput, let hdrColorSpace {
                    try LTXHDRVideoWriter.write(
                        hdrOutput,
                        colorSpace: hdrColorSpace,
                        fps: result.playbackFPS,
                        to: outputURL,
                        writeHLGMaster: !skipHDRMP4
                    )
                } else {
                    try LTXVideoMP4Writer.writeMP4(
                        frames: result.frames,
                        fps: result.playbackFPS,
                        to: outputURL
                    )
                }
                let mp4WriteSeconds = videoMonotonicSeconds() - writeStart
                try emitLTXVideoTimingReport(
                    LTXVideoTimingReport(
                        mode: dfrOptions != nil ? "ltx25-dfr-video-only" : "full-video-only",
                        modelRoot: rootURL.path,
                        residentModelReused: false,
                        load: loadTimings,
                        generation: result.timings,
                        unloadSeconds: unloadSeconds,
                        mp4WriteSeconds: mp4WriteSeconds,
                        totalSeconds: videoMonotonicSeconds() - endToEndStart
                    ),
                    printToStandardError: timings,
                    outputPath: timingsOutput
                )
            } catch {
                await generator.unload()
                throw error
            }

        case .unifiedAV:
            let endToEndStart = videoMonotonicSeconds()
            if !quiet {
                CLIStderr.write("Loading native unified AV model...\n")
            }
            let generator = LTXUnifiedAVGenerator()
            do {
                let loadTimings: LTXLoadTimings
                if dfrOptions != nil {
                    loadTimings = try await generator.loadDFR(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                } else if distilledLoRAStrengthStage1 != 0 {
                    loadTimings = try await generator.loadFullReusable(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                } else if isLTX23FullModelRoot(rootURL) || isLTX25FullModelRoot(rootURL) {
                    loadTimings = try await generator.loadFull(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                } else {
                    loadTimings = try await generator.load(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                }
                if !quiet {
                    let lane = dfrOptions != nil
                        ? "LTX 2.5 DFR spatial detail and temporal refinement"
                        : (isLTX23FullModelRoot(rootURL) || isLTX25FullModelRoot(rootURL)
                            ? (pipeline == .devOneStage
                                ? "target-resolution dev one-stage"
                                : "guided dev stage 1 + distilled-LoRA stage 2")
                            : "standalone distilled two-stage")
                    CLIStderr.write("Running \(lane) unified AV denoising + decode...\n")
                }
                let resolvedFrames = try await resolveAutoDuration(
                    autoDurationRange,
                    prompt: prompt,
                    fps: fps,
                    generator: generator,
                    fallback: numFrames
                )
                let result = try await generator.generate(
                    options: makeUnifiedOptions(resolvedFrames)
                )
                let unloadStart = videoMonotonicSeconds()
                await generator.unload()
                let unloadSeconds = videoMonotonicSeconds() - unloadStart

                if !quiet {
                    CLIStderr.write("Decoded frames shape: \(shapeString(result.frames.shape))\n")
                    CLIStderr.write("Audio waveform shape: \(shapeString(result.audioWaveform.shape))\n")
                    CLIStderr.write("Writing MP4 with audio...\n")
                }
                let writeStart = videoMonotonicSeconds()
                if let hdrOutput = result.hdrOutput, let hdrColorSpace {
                    try LTXHDRVideoWriter.write(
                        hdrOutput,
                        colorSpace: hdrColorSpace,
                        fps: result.playbackFPS,
                        to: outputURL,
                        audioWaveform: result.audioWaveform,
                        audioSampleRate: result.audioSampleRate
                    )
                } else {
                    try LTXVideoMP4Writer.writeMP4(
                        frames: result.frames,
                        fps: result.playbackFPS,
                        to: outputURL,
                        audioWaveform: result.audioWaveform,
                        audioSampleRate: result.audioSampleRate
                    )
                }

                guard await mediaHasAudioTrack(at: outputURL) else {
                    throw ValidationError("Unified AV output has no audio track at \(outputURL.path)")
                }
                let mp4WriteSeconds = videoMonotonicSeconds() - writeStart
                try emitLTXVideoTimingReport(
                    LTXVideoTimingReport(
                        mode: dfrOptions != nil
                            ? "ltx25-dfr-unified-av"
                            : (isLTX23FullModelRoot(rootURL) || isLTX25FullModelRoot(rootURL)
                                ? "full-unified-av"
                                : "standalone-distilled-unified-av"),
                        modelRoot: rootURL.path,
                        residentModelReused: false,
                        load: loadTimings,
                        generation: result.timings,
                        unloadSeconds: unloadSeconds,
                        mp4WriteSeconds: mp4WriteSeconds,
                        totalSeconds: videoMonotonicSeconds() - endToEndStart
                    ),
                    printToStandardError: timings,
                    outputPath: timingsOutput
                )
            } catch {
                await generator.unload()
                throw error
            }
        }

        if !quiet {
            CLIStderr.write("Saved: \(savedArtifactURL.path)\n")
        }
        print(savedArtifactURL.path)
    }

    private func resolveAutoDuration(
        _ range: LTX25AutoDuration?,
        prompt: String,
        fps: Double,
        generator: LTXUnifiedAVGenerator,
        fallback: Int
    ) async throws -> Int {
        guard let range else { return fallback }
        let frameCount = try await generator.predictFrameCount(
            prompt: prompt,
            frameRate: fps,
            range: range,
            conditioning: .audioVideo
        )
        if !quiet {
            let seconds = Double(frameCount) / fps
            CLIStderr.write(
                "DurationHead selected \(frameCount) frames at \(fps) fps "
                    + "(~\(String(format: "%.2f", seconds))s)\n"
            )
        }
        return frameCount
    }

    private func mediaHasAudioTrack(at url: URL) async -> Bool {
        MediaVideoIO.hasAudioTrack(url)
    }

    func makePreflightEnvelope(
        outputURL: URL,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) -> VideoGenerationPreflightEnvelope {
        let input = VideoGenerationPreflightInput(
            prompt: prompt,
            outputURL: outputURL,
            model: resolvedRequestedModel,
            variant: variant,
            quality: quality,
            outputMode: outputMode,
            legacyVariant: legacyVariant,
            productSelectionValidationMessage: productSelectionValidationMessage,
            modelRoot: modelRoot,
            width: resolvedOutputWidth,
            height: resolvedOutputHeight,
            numFrames: numFrames ?? 65,
            numFramesSpecified: numFrames != nil,
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
            generationArgv: generationActionArguments(outputURL: outputURL),
            cwd: fileManager.currentDirectoryPath
        )
        return VideoGenerationPreflightAnalyzer(
            input: input,
            fileManager: fileManager,
            now: now
        ).envelope()
    }

    private func runPreflight(outputURL: URL) throws {
        let envelope = makePreflightEnvelope(outputURL: outputURL)
        if json {
            print(try StructuredRunOutput.encode(envelope))
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

func validateNativeModelRoot(_ rootURL: URL) throws {
    let fm = FileManager.default
    let rootURL = rootURL.resolvingSymlinksInPath()
    var isDirectory: ObjCBool = false
    guard fm.fileExists(atPath: rootURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw ValidationError("Model root directory not found: \(rootURL.path)")
    }

    let wanResources = Wan2Resources(rootURL: rootURL)
    if wanResources.validate().isEmpty, (try? wanResources.loadConfiguration()) != nil {
        return
    }

    if isLTX23AudioToVideoModelRoot(rootURL)
        || isLTX23SplitModelRoot(rootURL)
        || isLTX25FullModelRoot(rootURL)
        || isLTX25ModelRoot(rootURL) {
        return
    }

    let required = [
        rootURL.appendingPathComponent("text_encoder/config.json", isDirectory: false),
        rootURL.appendingPathComponent("text_encoder/model.safetensors.index.json", isDirectory: false),
    ]
    for file in required where !fm.fileExists(atPath: file.path) {
        throw ValidationError("Missing required LTX file: \(file.path)")
    }

    var tokenizerIsDir: ObjCBool = false
    let tokenizer = rootURL.appendingPathComponent("tokenizer", isDirectory: true)
    guard fm.fileExists(atPath: tokenizer.path, isDirectory: &tokenizerIsDir), tokenizerIsDir.boolValue else {
        throw ValidationError("Missing tokenizer directory: \(tokenizer.path)")
    }

    let entries = (try? fm.contentsOfDirectoryResolvingSymlinks(
        at: rootURL,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
    )) ?? []
    let hasTransformer = entries.contains { entry in
        let name = entry.lastPathComponent
        return name.hasPrefix("ltx-2-19") && name.hasSuffix(".safetensors")
    }
    let hasUpsampler = entries.contains { entry in
        let name = entry.lastPathComponent
        return name.hasPrefix("ltx-2-spatial-upscaler") && name.hasSuffix(".safetensors")
    }

    guard hasTransformer else {
        throw ValidationError("Missing LTX transformer weights under \(rootURL.path)")
    }
    guard hasUpsampler else {
        throw ValidationError("Missing LTX upsampler weights under \(rootURL.path)")
    }
}

func validateNativeAudioToVideoModelRoot(
    _ rootURL: URL,
    fileManager: FileManager = .default
) throws {
    let root = rootURL.resolvingSymlinksInPath()
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: root.path, isDirectory: &isDirectory),
          isDirectory.boolValue else {
        throw ValidationError("A2Vid model root directory not found: \(root.path)")
    }

    if isLTX25FullModelRoot(root, fileManager: fileManager) {
        return
    }

    let required = [
        "split_model.json",
        "config.json",
        "connector.safetensors",
        "transformer-dev.safetensors",
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
        "audio_vae.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors",
    ]
    for relativePath in required {
        let file = root.appendingPathComponent(relativePath, isDirectory: false)
        guard fileManager.fileExists(atPath: file.path) else {
            throw ValidationError(
                "Missing required LTX 2.3 A2Vid file: \(file.path). Pull \(ModelResolver.ModelID.ltxVideo23FullMLX.rawValue)."
            )
        }
    }
}

private func shapeString(_ shape: [Int]) -> String {
    "[" + shape.map(String.init).joined(separator: ", ") + "]"
}

func resolveVideoModelRoot(
    explicitModelRoot: String?,
    requestedModel: String,
    variant: LTXVideoVariant,
    allowAutoDownload: Bool = true
) async throws -> URL {
    if let explicitModelRoot, !explicitModelRoot.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        return URL(fileURLWithPath: explicitModelRoot).standardizedFileURL
    }

    let trimmedModel = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
    if !trimmedModel.isEmpty {
        let explicitModelURL = URL(fileURLWithPath: trimmedModel).standardizedFileURL
        if FileManager.default.fileExists(atPath: explicitModelURL.path)
            || trimmedModel.lowercased() != ModelResolver.ModelID.ltxVideoAV.rawValue
        {
            if let modelID = ModelResolver.ModelID(rawValue: trimmedModel.lowercased()),
               let installed = ModelResolver().resolveIfPresent(modelID) {
                return installed.rootURL
            }
            do {
                let resolved = try await ManagedModelResolver.resolveForRuntime(
                    requestedModel: trimmedModel,
                    defaultModelID: ModelResolver.ModelID.ltxVideoAV.rawValue,
                    allowAutoDownload: allowAutoDownload
                )
                return resolved.url
            } catch let error as ManagedModelResolver.ResolverError {
                throw ValidationError(error.localizedDescription)
            }
        }
    }

    if let suggested = suggestedVideoModelRoot(for: variant) {
        return URL(fileURLWithPath: suggested).standardizedFileURL
    }

    do {
        let resolved = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: ModelResolver.ModelID.ltxVideoAV.rawValue,
            defaultModelID: ModelResolver.ModelID.ltxVideoAV.rawValue,
            allowAutoDownload: allowAutoDownload
        )
        return resolved.url
    } catch let error as ManagedModelResolver.ResolverError {
        throw ValidationError(error.localizedDescription)
    }
}

private func suggestedVideoModelRoot(for variant: LTXVideoVariant) -> String? {
    if let envPath = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_MODEL_ROOT"], !envPath.isEmpty,
       isNativeVideoModelRootAvailable(at: envPath) {
        return envPath
    }

    let fm = FileManager.default
    let home = fm.homeDirectoryForCurrentUser
    let zeroModels = MereRunModelPaths.modelsDir
    let candidates: [String] = {
        switch variant {
        case .distilled:
            return [
                zeroModels.appendingPathComponent("LTX-2-distilled-bf16", isDirectory: true).path,
                zeroModels.appendingPathComponent("ltx-video-distilled", isDirectory: true).path,
                home.appendingPathComponent("models/LTX-2-distilled-bf16", isDirectory: true).path,
                home.appendingPathComponent("Models/LTX-2-distilled-bf16", isDirectory: true).path,
            ]
        case .unifiedAV:
            return [
                zeroModels.appendingPathComponent("video-ltx23-full-mlx", isDirectory: true).path,
                zeroModels.appendingPathComponent("video-ltx23-a2vid-mlx", isDirectory: true).path,
                zeroModels.appendingPathComponent("video-ltx-av", isDirectory: true).path,
                zeroModels.appendingPathComponent("video-ltx23-av-mlx", isDirectory: true).path,
                zeroModels.appendingPathComponent("LTX-2-mlx-av", isDirectory: true).path,
                home.appendingPathComponent("models/video-ltx-av", isDirectory: true).path,
                home.appendingPathComponent("models/video-ltx23-full-mlx", isDirectory: true).path,
                home.appendingPathComponent("models/video-ltx23-a2vid-mlx", isDirectory: true).path,
                home.appendingPathComponent("models/video-ltx23-av-mlx", isDirectory: true).path,
                home.appendingPathComponent("models/LTX-2-mlx-av", isDirectory: true).path,
                home.appendingPathComponent("Models/LTX-2-mlx-av", isDirectory: true).path,
            ]
        }
    }()

    for candidate in candidates where isNativeVideoModelRootAvailable(at: candidate) {
        return candidate
    }
    return nil
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

private func isNativeVideoModelRootAvailable(at path: String) -> Bool {
    let rootURL = URL(fileURLWithPath: path).standardizedFileURL
    do {
        try validateNativeModelRoot(rootURL)
        return true
    } catch {
        return false
    }
}
