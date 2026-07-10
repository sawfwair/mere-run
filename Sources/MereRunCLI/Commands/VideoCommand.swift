import ArgumentParser
import Foundation
import MediaIO
import MLX
import MereRunCore

enum LTXVideoVariant: String, CaseIterable, ExpressibleByArgument {
    case unifiedAV = "unified-av"
    case distilled = "distilled"
    case lingbot = "lingbot"
}

struct Video: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "video",
        abstract: "Generate videos with native Swift/MLX LTX pipelines.",
        subcommands: [
            VideoExportLatents.self,
            VideoGenerate.self
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
        abstract: "Generate MP4 video with native Swift/MLX video pipelines.",
        discussion: """
        Prints the output MP4 path to stdout.
        Progress and diagnostics are printed to stderr.

        Use the default distilled variant for faster video-only drafts. Use
        --variant unified-av for synchronized audio/video, and prefer
        --model video-ltx23-av-mlx for LTX 2.3 quality renders.

        Examples:
          swift run mere.run video generate "a cinematic drone flythrough over snowy mountains" --num-frames 65
          swift run mere.run video generate "woman walking in neon rain" --image frame.png
          swift run mere.run video generate "a car drives from dawn into sunset" --image start.png --end-image end.png
          swift run mere.run video generate "dialogue with clean background music" --variant unified-av --model video-ltx23-av-mlx --duration 15 --fps 24
        """
    )

    @Argument(help: "Prompt for video generation. Optional when --prompt-json is provided.")
    var prompt: String?

    @Option(name: [.customLong("prompt-json")], help: "LingBot structured prompt JSON; its caption and optional duration follow the released runner contract.")
    var promptJSON: String?

    @Option(name: [.customShort("o"), .long], help: "Output MP4 path (default: ./mererun-video-<timestamp>.mp4).")
    var output: String?

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local path to the LTX model root.")
    var model: String = ModelResolver.ModelID.ltxVideo23AVMLX.rawValue

    @Option(name: [.customLong("variant")], help: "Native pipeline: distilled, unified-av, or lingbot. LingBot model ids select lingbot automatically.")
    var variant: LTXVideoVariant = .distilled

    @Option(name: [.customLong("model-root")], help: "Local LTX model root. Takes precedence over --model.")
    var modelRoot: String?

    @Option(name: [.long], help: "Output width (must be divisible by 64; auto-snapped down).")
    var width: Int?

    @Option(name: [.long], help: "Output height (must be divisible by 64; auto-snapped down).")
    var height: Int?

    @Option(name: [.customLong("num-frames")], help: "Frame count (must be 8n+1; auto-adjusted).")
    var numFrames: Int?

    @Option(name: [.long], help: "Target output duration in seconds. Overrides --num-frames by choosing the nearest 8n+1 frame count for --fps.")
    var duration: Double?

    @Option(name: [.long], help: "Frames per second.")
    var fps: Int = 24

    @Option(name: [.long], help: "Seed value.")
    var seed: Int?

    @Option(name: [.long], help: "LingBot denoising step count.")
    var steps: Int = 40

    @Option(name: [.customLong("guidance-scale")], help: "LingBot classifier-free guidance scale.")
    var guidanceScale: Float = 3

    @Option(name: [.long], help: "LingBot Flow-UniPC timestep shift.")
    var shift: Float = 3

    @Flag(
        name: [.customLong("batch-cfg")],
        help: "Run LingBot positive and negative CFG branches as one masked MLX batch."
    )
    var batchCFG: Bool = false

    @Option(name: [.customLong("negative-prompt")], help: "Optional LingBot negative prompt; defaults to the upstream universal negative JSON.")
    var negativePrompt: String?

    @Option(name: [.customLong("negative-prompt-json")], help: "LingBot auto-negative JSON file, compacted with source key order preserved.")
    var negativePromptJSON: String?

    @Flag(name: [.customLong("temporal-probe")], help: "Decode an early LingBot denoised estimate at full duration, then stop.")
    var temporalProbe: Bool = false

    @Option(name: [.customLong("temporal-probe-step")], help: "LingBot denoiser step to decode for --temporal-probe.")
    var temporalProbeStep: Int = 4

    @Flag(name: [.customLong("refiner")], help: "Run the LingBot MoE second-stage refiner and write the refined MP4.")
    var refiner: Bool = false

    @Option(name: [.customLong("refiner-width")], help: "LingBot refiner output width (default: 1920).")
    var refinerWidth: Int?

    @Option(name: [.customLong("refiner-height")], help: "LingBot refiner output height (default: 1088).")
    var refinerHeight: Int?

    @Option(name: [.customLong("refiner-steps")], help: "LingBot refiner requested step count.")
    var refinerSteps: Int = 8

    @Option(name: [.customLong("refiner-guidance-scale")], help: "LingBot refiner classifier-free guidance scale.")
    var refinerGuidanceScale: Float = 3

    @Option(name: [.customLong("refiner-shift")], help: "LingBot refiner Flow-UniPC timestep shift.")
    var refinerShift: Float = 3

    @Option(name: [.customLong("refiner-threshold")], help: "LingBot refiner re-noising threshold.")
    var refinerThreshold: Float = 0.85

    @Option(name: [.customLong("refiner-sigma-tail-steps")], help: "LingBot refiner low-noise tail sigma count.")
    var refinerSigmaTailSteps: Int = 2

    @Flag(
        name: [.customLong("refiner-batch-cfg")],
        help: "Run LingBot refiner CFG branches as one masked MLX batch."
    )
    var refinerBatchCFG: Bool = false

    @Option(name: [.long], help: "Optional source image path (enables image-to-video).")
    var image: String?

    @Option(name: [.customLong("image-strength")], help: "Image conditioning strength in [0, 1].")
    var imageStrength: Float = 1.0

    @Option(name: [.customLong("end-image")], help: "Optional end keyframe path; conditions the last frame so the clip interpolates a directed start->end motion. Requires --image.")
    var endImage: String?

    @Option(name: [.customLong("end-image-strength")], help: "End keyframe conditioning strength in [0, 1].")
    var endImageStrength: Float = 1.0

    @Flag(name: [.customLong("preflight")], help: "Inspect the video generation request without running generation.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "With --preflight, emit a structured JSON report.")
    var json: Bool = false

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    func run() async throws {
        if json && !preflight {
            throw ValidationError("--json is only supported with --preflight for video generate.")
        }

        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-video", defaultExtension: "mp4")
        if preflight {
            try runPreflight(outputURL: outputURL)
            return
        }

        let effectiveVariant = effectiveVariant
        if promptJSON != nil, effectiveVariant != .lingbot {
            throw ValidationError("--prompt-json is supported only by the native LingBot pipeline.")
        }
        if negativePrompt != nil, negativePromptJSON != nil {
            throw ValidationError("Use either --negative-prompt or --negative-prompt-json, not both.")
        }
        if negativePromptJSON != nil, effectiveVariant != .lingbot {
            throw ValidationError("--negative-prompt-json is supported only by the native LingBot pipeline.")
        }
        let resolvedPrompt = try resolvePrompt()
        let trimmedPrompt = resolvedPrompt.caption.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else {
            throw ValidationError("Provide a non-empty prompt or --prompt-json.")
        }

        guard fps > 0 else {
            throw ValidationError("--fps must be >= 1")
        }
        let effectiveDuration = duration ?? resolvedPrompt.duration
        if let effectiveDuration {
            guard effectiveDuration > 0 else {
                throw ValidationError("--duration must be > 0")
            }
        }
        let dimensionMultiple = effectiveVariant == .lingbot ? 16 : 64
        let minimumFrames = effectiveVariant == .lingbot ? 5 : 9
        let requestedWidth = width ?? (effectiveVariant == .lingbot ? 320 : 768)
        let requestedHeight = height ?? (effectiveVariant == .lingbot ? 192 : 512)
        let requestedFrameOption = numFrames ?? (effectiveVariant == .lingbot ? 9 : 65)
        guard requestedWidth >= dimensionMultiple else {
            throw ValidationError("--width must be >= \(dimensionMultiple)")
        }
        guard requestedHeight >= dimensionMultiple else {
            throw ValidationError("--height must be >= \(dimensionMultiple)")
        }
        guard requestedFrameOption >= minimumFrames else {
            throw ValidationError("--num-frames must be >= \(minimumFrames)")
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
        if effectiveVariant == .lingbot, image != nil || endImage != nil {
            throw ValidationError("The native LingBot pipeline currently supports text-to-video only.")
        }
        if effectiveVariant == .lingbot {
            guard steps >= 1 else {
                throw ValidationError("--steps must be >= 1")
            }
            guard guidanceScale > 0 else {
                throw ValidationError("--guidance-scale must be > 0")
            }
            guard shift > 0 else {
                throw ValidationError("--shift must be > 0")
            }
        }
        if refiner, effectiveVariant != .lingbot {
            throw ValidationError("--refiner is supported only by the native LingBot pipeline.")
        }
        if batchCFG, effectiveVariant != .lingbot {
            throw ValidationError("--batch-cfg is supported only by the native LingBot pipeline.")
        }
        if refinerBatchCFG, effectiveVariant != .lingbot {
            throw ValidationError("--refiner-batch-cfg is supported only by the native LingBot pipeline.")
        }
        if refinerBatchCFG, !refiner {
            throw ValidationError("--refiner-batch-cfg requires --refiner.")
        }
        if temporalProbe, effectiveVariant != .lingbot {
            throw ValidationError("--temporal-probe is supported only by the native LingBot pipeline.")
        }
        if temporalProbe, !(1...steps).contains(temporalProbeStep) {
            throw ValidationError("--temporal-probe-step must be between 1 and --steps")
        }
        if (refinerWidth == nil) != (refinerHeight == nil) {
            throw ValidationError("--refiner-width and --refiner-height must be provided together.")
        }
        if refiner {
            if let refinerWidth, refinerWidth < 16 {
                throw ValidationError("--refiner-width must be >= 16")
            }
            if let refinerHeight, refinerHeight < 16 {
                throw ValidationError("--refiner-height must be >= 16")
            }
            guard refinerSteps >= 1 else {
                throw ValidationError("--refiner-steps must be >= 1")
            }
            guard refinerGuidanceScale > 0 else {
                throw ValidationError("--refiner-guidance-scale must be > 0")
            }
            guard refinerShift > 0 else {
                throw ValidationError("--refiner-shift must be > 0")
            }
            guard refinerThreshold > 0, refinerThreshold <= 1 else {
                throw ValidationError("--refiner-threshold must be in (0, 1]")
            }
            guard refinerSigmaTailSteps >= 0 else {
                throw ValidationError("--refiner-sigma-tail-steps must be >= 0")
            }
        }

        let resolvedWidth = max(dimensionMultiple, (requestedWidth / dimensionMultiple) * dimensionMultiple)
        let resolvedHeight = max(dimensionMultiple, (requestedHeight / dimensionMultiple) * dimensionMultiple)
        let resolvedRefinerWidth = refiner
            ? max(16, ((refinerWidth ?? 1_920) / 16) * 16)
            : nil
        let resolvedRefinerHeight = refiner
            ? max(16, ((refinerHeight ?? 1_088) / 16) * 16)
            : nil
        let requestedNumFrames = effectiveDuration.map {
            effectiveVariant == .lingbot
                ? nearestLingBotFrameCount(duration: $0, fps: fps)
                : nearestLTXFrameCount(duration: $0, fps: fps)
        } ?? requestedFrameOption
        let frameStride = effectiveVariant == .lingbot ? 4 : 8
        let resolvedNumFrames = max(minimumFrames, ((requestedNumFrames - 1) / frameStride) * frameStride + 1)
        if !quiet {
            if resolvedWidth != requestedWidth || resolvedHeight != requestedHeight {
                CLIStderr.write("Adjusted size to \(resolvedWidth)x\(resolvedHeight) (must be divisible by \(dimensionMultiple))\n")
            }
            if refiner, let resolvedRefinerWidth, let resolvedRefinerHeight {
                CLIStderr.write("Refiner output size: \(resolvedRefinerWidth)x\(resolvedRefinerHeight)\n")
            }
            if temporalProbe {
                CLIStderr.write(
                    "Temporal probe: full-duration base estimate at step \(temporalProbeStep); refiner will not run.\n"
                )
            }
            if let effectiveDuration {
                let resolvedSeconds = Double(resolvedNumFrames) / Double(fps)
                CLIStderr.write(
                    "Resolved duration \(String(format: "%.2f", effectiveDuration))s to \(resolvedNumFrames) frames at \(fps) fps (~\(String(format: "%.2f", resolvedSeconds))s; must satisfy \(frameStride)n+1)\n"
                )
            } else if resolvedNumFrames != requestedFrameOption {
                CLIStderr.write("Adjusted frame count to \(resolvedNumFrames) (must satisfy \(frameStride)n+1)\n")
            }
            if effectiveVariant == .unifiedAV && fps != 24 {
                CLIStderr.write(
                    "Warning: LTX unified AV is trained for 24 fps; --fps \(fps) can make generated motion look time-stretched relative to audio.\n"
                )
            }
        }

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

        let resolvedModelRoot = try await resolveVideoModelRoot(
            explicitModelRoot: modelRoot,
            requestedModel: model,
            variant: effectiveVariant
        ).path

        try await runNativeGenerate(
            prompt: trimmedPrompt,
            width: resolvedWidth,
            height: resolvedHeight,
            numFrames: resolvedNumFrames,
            fps: fps,
            seed: seed ?? 42,
            variant: effectiveVariant,
            sourceImageURL: sourceImageURL,
            imageStrength: imageStrength,
            endImageURL: endImageURL,
            endImageStrength: endImageStrength,
            refinerWidth: resolvedRefinerWidth,
            refinerHeight: resolvedRefinerHeight,
            modelRoot: resolvedModelRoot,
            outputURL: outputURL
        )
    }

    private func runNativeGenerate(
        prompt: String,
        width: Int,
        height: Int,
        numFrames: Int,
        fps: Int,
        seed: Int,
        variant: LTXVideoVariant,
        sourceImageURL: URL?,
        imageStrength: Float,
        endImageURL: URL?,
        endImageStrength: Float,
        refinerWidth: Int?,
        refinerHeight: Int?,
        modelRoot: String,
        outputURL: URL
    ) async throws {
        let rootURL = URL(fileURLWithPath: modelRoot).standardizedFileURL
        if variant == .lingbot {
            _ = try LingBotVideoResources(rootURL: rootURL)
        } else {
            try validateNativeModelRoot(rootURL)
        }
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        if !quiet {
            CLIStderr.write("Engine: native\n")
            CLIStderr.write("Variant: \(variant.rawValue)\n")
            CLIStderr.write("Model root: \(rootURL.path)\n")
            CLIStderr.write("Mode: \(sourceImageURL == nil ? "text-to-video" : "image-to-video")\n")
        }

        switch variant {
        case .distilled:
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
                try LTXVideoMP4Writer.writeMP4(frames: result.frames, fps: fps, to: outputURL)
            } catch {
                await generator.unload()
                throw error
            }

        case .unifiedAV:
            if !quiet {
                CLIStderr.write("Loading native unified AV model...\n")
            }
            let generator = LTXUnifiedAVGenerator()
            do {
                try await generator.load(modelRoot: rootURL)
                if !quiet {
                    CLIStderr.write("Running native unified AV denoising + decode...\n")
                }
                let result = try await generator.generate(
                    options: LTXUnifiedAVGenerationOptions(
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

                if !quiet {
                    CLIStderr.write("Decoded frames shape: \(shapeString(result.frames.shape))\n")
                    CLIStderr.write("Audio waveform shape: \(shapeString(result.audioWaveform.shape))\n")
                    CLIStderr.write("Writing MP4 with audio...\n")
                }
                try LTXVideoMP4Writer.writeMP4(
                    frames: result.frames,
                    fps: fps,
                    to: outputURL,
                    audioWaveform: result.audioWaveform,
                    audioSampleRate: result.audioSampleRate
                )

                guard await mediaHasAudioTrack(at: outputURL) else {
                    throw ValidationError("Unified AV output has no audio track at \(outputURL.path)")
                }
            } catch {
                await generator.unload()
                throw error
            }
        case .lingbot:
            if !quiet {
                let resources = try LingBotVideoResources(rootURL: rootURL)
                let architecture = resources.transformerConfig.numExperts > 0 ? "MoE" : "Dense"
                CLIStderr.write("Loading native LingBot-Video \(architecture) pipeline...\n")
            }
            let pipeline = LingBotVideoPipeline()
            let shouldReportProgress = !quiet
            let progressHandler: (@Sendable (LingBotVideoGenerationProgress) -> Void)?
            if shouldReportProgress {
                progressHandler = { progress in
                    if let branch = progress.branch, progress.blockIndex > 0 {
                        let phase = progress.stage == .refining ? "refining" : "denoising"
                        CLIStderr.write(
                            "LingBot \(phase) \(progress.stepIndex)/\(progress.totalSteps) "
                                + "\(branch.rawValue) block \(progress.blockIndex)/\(progress.totalBlocks)\n"
                        )
                    } else if progress.stage == .denoising {
                        CLIStderr.write("LingBot denoising \(progress.stepIndex)/\(progress.totalSteps)\n")
                    } else if progress.stage == .refining {
                        CLIStderr.write("LingBot refining \(progress.stepIndex)/\(progress.totalSteps)\n")
                    } else {
                        CLIStderr.write("LingBot stage: \(progress.stage.rawValue)\n")
                    }
                }
            } else {
                progressHandler = nil
            }
            let result = try await pipeline.generate(
                modelRoot: rootURL,
                options: LingBotVideoGenerationOptions(
                    prompt: prompt,
                    negativePrompt: try resolveNegativePrompt()
                        ?? LingBotVideoPipeline.defaultNegativePrompt,
                    width: width,
                    height: height,
                    numFrames: numFrames,
                    fps: fps,
                    steps: steps,
                    guidanceScale: guidanceScale,
                    shift: shift,
                    batchCFG: batchCFG,
                    seed: seed,
                    temporalProbe: temporalProbe,
                    temporalProbeStep: temporalProbeStep,
                    runRefiner: refiner,
                    refinerWidth: refinerWidth,
                    refinerHeight: refinerHeight,
                    refinerSteps: refinerSteps,
                    refinerGuidanceScale: refinerGuidanceScale,
                    refinerShift: refinerShift,
                    refinerThreshold: refinerThreshold,
                    refinerSigmaTailSteps: refinerSigmaTailSteps,
                    refinerBatchCFG: refinerBatchCFG
                ),
                progressHandler: progressHandler
            )
            if !quiet {
                CLIStderr.write("Decoded frames shape: \(shapeString(result.frames.shape))\n")
                if result.temporalProbe {
                    CLIStderr.write("LingBot temporal probe complete at denoiser step \(temporalProbeStep).\n")
                }
                if result.refined {
                    CLIStderr.write("LingBot refiner pass complete.\n")
                }
                CLIStderr.write("Writing MP4...\n")
            }
            try LTXVideoMP4Writer.writeMP4(frames: result.frames, fps: fps, to: outputURL)
            let temporalMetrics = LingBotVideoTemporalMetrics.analyze(frames: result.frames)
            if !quiet {
                CLIStderr.write(
                    String(
                        format: "Temporal luma: delta_mean=%.2f delta_peak=%.2f spatial_std=%.2f\n",
                        temporalMetrics.meanLumaDelta,
                        temporalMetrics.peakLumaDelta,
                        temporalMetrics.lumaStandardDeviation
                    )
                )
                if !temporalMetrics.isInformative {
                    CLIStderr.write(
                        "Temporal verdict: inconclusive low-contrast estimate; increase --temporal-probe-step.\n"
                    )
                } else if temporalMetrics.isLikelyUnstable {
                    CLIStderr.write("Temporal verdict: unstable; do not start the full run.\n")
                } else {
                    CLIStderr.write("Temporal verdict: stable enough to inspect and continue.\n")
                }
            }
        }

        if !quiet {
            CLIStderr.write("Saved: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    private func mediaHasAudioTrack(at url: URL) async -> Bool {
        MediaVideoIO.hasAudioTrack(url)
    }

    func makePreflightEnvelope(
        outputURL: URL,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) -> VideoGenerationPreflightEnvelope {
        let promptResolution: ResolvedVideoPrompt?
        let promptResolutionError: String?
        do {
            promptResolution = try resolvePrompt()
            promptResolutionError = nil
        } catch {
            promptResolution = nil
            promptResolutionError = error.localizedDescription
        }
        let resolvedInputPrompt = promptResolution?.caption ?? prompt ?? ""
        let negativePromptResolution: String?
        let negativePromptResolutionError: String?
        do {
            negativePromptResolution = try resolveNegativePrompt()
            negativePromptResolutionError = nil
        } catch {
            negativePromptResolution = nil
            negativePromptResolutionError = error.localizedDescription
        }
        let resolvedInputDuration = duration ?? promptResolution?.duration
        let requestedWidth = width ?? (effectiveVariant == .lingbot ? 320 : 768)
        let requestedHeight = height ?? (effectiveVariant == .lingbot ? 192 : 512)
        let requestedFrames = numFrames ?? (effectiveVariant == .lingbot ? 9 : 65)
        let input = VideoGenerationPreflightInput(
            prompt: resolvedInputPrompt,
            promptJSON: promptJSON,
            promptJSONError: promptResolutionError,
            negativePromptJSON: negativePromptJSON,
            negativePromptJSONError: negativePromptResolutionError,
            outputURL: outputURL,
            model: model,
            variant: effectiveVariant,
            modelRoot: modelRoot,
            width: requestedWidth,
            height: requestedHeight,
            numFrames: requestedFrames,
            duration: resolvedInputDuration,
            fps: fps,
            seed: seed,
            image: image,
            imageStrength: imageStrength,
            endImage: endImage,
            endImageStrength: endImageStrength,
            steps: steps,
            guidanceScale: guidanceScale,
            shift: shift,
            batchCFG: batchCFG,
            negativePrompt: negativePromptResolution,
            temporalProbe: temporalProbe,
            temporalProbeStep: temporalProbeStep,
            refiner: refiner,
            refinerWidth: refinerWidth,
            refinerHeight: refinerHeight,
            refinerSteps: refinerSteps,
            refinerGuidanceScale: refinerGuidanceScale,
            refinerShift: refinerShift,
            refinerThreshold: refinerThreshold,
            refinerSigmaTailSteps: refinerSigmaTailSteps,
            refinerBatchCFG: refinerBatchCFG,
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
        ]
        if let promptJSON {
            args += ["--prompt-json", promptJSON]
        } else if let prompt {
            args.append(prompt)
        }
        let requestedWidth = width ?? (effectiveVariant == .lingbot ? 320 : 768)
        let requestedHeight = height ?? (effectiveVariant == .lingbot ? 192 : 512)
        let requestedFrames = numFrames ?? (effectiveVariant == .lingbot ? 9 : 65)
        args += [
            "--output",
            outputURL.path,
            "--model",
            model,
            "--variant",
            effectiveVariant.rawValue,
            "--width",
            String(requestedWidth),
            "--height",
            String(requestedHeight),
            "--num-frames",
            String(requestedFrames),
            "--fps",
            String(fps),
            "--steps",
            String(steps),
            "--guidance-scale",
            String(guidanceScale),
            "--shift",
            String(shift),
        ]
        if let duration {
            args += ["--duration", String(duration)]
        }
        if let seed {
            args += ["--seed", String(seed)]
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
        if let negativePrompt {
            args += ["--negative-prompt", negativePrompt]
        } else if let negativePromptJSON {
            args += ["--negative-prompt-json", negativePromptJSON]
        }
        if temporalProbe {
            args += ["--temporal-probe", "--temporal-probe-step", String(temporalProbeStep)]
        }
        if batchCFG {
            args.append("--batch-cfg")
        }
        if refiner {
            args += [
                "--refiner",
                "--refiner-steps", String(refinerSteps),
                "--refiner-guidance-scale", String(refinerGuidanceScale),
                "--refiner-shift", String(refinerShift),
                "--refiner-threshold", String(refinerThreshold),
                "--refiner-sigma-tail-steps", String(refinerSigmaTailSteps),
            ]
            if let refinerWidth, let refinerHeight {
                args += ["--refiner-width", String(refinerWidth), "--refiner-height", String(refinerHeight)]
            }
            if refinerBatchCFG {
                args.append("--refiner-batch-cfg")
            }
        }
        if quiet {
            args.append("--quiet")
        }
        return args
    }

    private func resolvePrompt() throws -> ResolvedVideoPrompt {
        if let promptJSON = promptJSON?.trimmingCharacters(in: .whitespacesAndNewlines),
           !promptJSON.isEmpty {
            let url = URL(fileURLWithPath: promptJSON).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("LingBot prompt JSON not found: \(url.path)")
            }
            let sample = try LingBotVideoPromptSample.load(from: url)
            return ResolvedVideoPrompt(caption: sample.caption, duration: sample.duration)
        }
        return ResolvedVideoPrompt(caption: prompt ?? "", duration: nil)
    }

    private func resolveNegativePrompt() throws -> String? {
        if negativePrompt != nil, negativePromptJSON != nil {
            throw ValidationError("Use either --negative-prompt or --negative-prompt-json, not both.")
        }
        guard let negativePromptJSON = negativePromptJSON?.trimmingCharacters(in: .whitespacesAndNewlines),
              !negativePromptJSON.isEmpty
        else {
            return negativePrompt
        }
        let url = URL(fileURLWithPath: negativePromptJSON).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("LingBot negative prompt JSON not found: \(url.path)")
        }
        return try LingBotVideoPromptSample.compactJSONDocument(Data(contentsOf: url))
    }

    var effectiveVariant: LTXVideoVariant {
        if variant == .lingbot || Self.isLingBotModel(model) {
            return .lingbot
        }
        return variant
    }

    private static func isLingBotModel(_ model: String) -> Bool {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == ModelResolver.ModelID.lingBotVideoDense13B.rawValue
            || normalized == ModelResolver.ModelID.lingBotVideoMoE30BA3B.rawValue
            || normalized == LingBotVideoMoEQuantizer.defaultOutputModelID
            || normalized == "robbyant/lingbot-video-dense-1.3b"
            || normalized == "robbyant/lingbot-video-moe-30b-a3b"
    }
}

func validateNativeModelRoot(_ rootURL: URL) throws {
    let fm = FileManager.default
    let rootURL = rootURL.resolvingSymlinksInPath()
    var isDirectory: ObjCBool = false
    guard fm.fileExists(atPath: rootURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw ValidationError("Model root directory not found: \(rootURL.path)")
    }

    if isLTX23SplitModelRoot(rootURL) {
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

    let entries = (try? fm.contentsOfDirectory(
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

private func shapeString(_ shape: [Int]) -> String {
    "[" + shape.map(String.init).joined(separator: ", ") + "]"
}

private func resolveVideoModelRoot(
    explicitModelRoot: String?,
    requestedModel: String,
    variant: LTXVideoVariant,
    allowAutoDownload: Bool = true
) async throws -> URL {
    if let explicitModelRoot, !explicitModelRoot.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        return URL(fileURLWithPath: explicitModelRoot).standardizedFileURL
    }

    var trimmedModel = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
    if variant == .lingbot,
       trimmedModel.isEmpty || trimmedModel == ModelResolver.ModelID.ltxVideo23AVMLX.rawValue {
        trimmedModel = ModelResolver.ModelID.lingBotVideoDense13B.rawValue
    }
    if !trimmedModel.isEmpty {
        if trimmedModel == LingBotVideoMoEQuantizer.defaultOutputModelID {
            let convertedRoot = MereRunModelPaths.modelDir(trimmedModel)
            if FileManager.default.fileExists(atPath: convertedRoot.path) {
                return convertedRoot
            }
        }
        let explicitModelURL = URL(fileURLWithPath: trimmedModel).standardizedFileURL
        if FileManager.default.fileExists(atPath: explicitModelURL.path)
            || trimmedModel.lowercased() != ModelResolver.ModelID.ltxVideoAV.rawValue
        {
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
    if variant == .lingbot,
       let envPath = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LINGBOT_MODEL_ROOT"],
       !envPath.isEmpty,
       isLingBotVideoModelRootAvailable(at: envPath) {
        return envPath
    }
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
                zeroModels.appendingPathComponent("video-ltx-av", isDirectory: true).path,
                zeroModels.appendingPathComponent("video-ltx23-av-mlx", isDirectory: true).path,
                zeroModels.appendingPathComponent("LTX-2-mlx-av", isDirectory: true).path,
                home.appendingPathComponent("models/video-ltx-av", isDirectory: true).path,
                home.appendingPathComponent("models/video-ltx23-av-mlx", isDirectory: true).path,
                home.appendingPathComponent("models/LTX-2-mlx-av", isDirectory: true).path,
                home.appendingPathComponent("Models/LTX-2-mlx-av", isDirectory: true).path,
            ]
        case .lingbot:
            return [
                zeroModels.appendingPathComponent(LingBotVideoMoEQuantizer.defaultOutputModelID, isDirectory: true).path,
                zeroModels.appendingPathComponent(ModelResolver.ModelID.lingBotVideoDense13B.rawValue, isDirectory: true).path,
                home.appendingPathComponent("models/\(ModelResolver.ModelID.lingBotVideoDense13B.rawValue)", isDirectory: true).path,
                home.appendingPathComponent("Models/\(ModelResolver.ModelID.lingBotVideoDense13B.rawValue)", isDirectory: true).path,
            ]
        }
    }()

    for candidate in candidates {
        if variant == .lingbot {
            if isLingBotVideoModelRootAvailable(at: candidate) {
                return candidate
            }
        } else if isNativeVideoModelRootAvailable(at: candidate) {
            return candidate
        }
    }
    return nil
}

func nearestLTXFrameCount(duration: Double, fps: Int) -> Int {
    let targetFrames = max(9.0, duration * Double(max(1, fps)))
    let chunks = max(1, Int(((targetFrames - 1.0) / 8.0).rounded()))
    return chunks * 8 + 1
}

func nearestLingBotFrameCount(duration: Double, fps: Int) -> Int {
    let frameCount = max(1, Int(duration * Double(max(1, fps))))
    return max(5, ((frameCount - 1) / 4 + 1) * 4 + 1)
}

private struct ResolvedVideoPrompt {
    let caption: String
    let duration: Double?
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

private func isLingBotVideoModelRootAvailable(at path: String) -> Bool {
    let rootURL = URL(fileURLWithPath: path).standardizedFileURL
    return (try? LingBotVideoResources(rootURL: rootURL)) != nil
}
