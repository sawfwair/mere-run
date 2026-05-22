import ArgumentParser
import Foundation
import MediaIO
import MLX
import MereRunCore

enum LTXVideoVariant: String, CaseIterable, ExpressibleByArgument {
    case unifiedAV = "unified-av"
    case distilled = "distilled"
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
        abstract: "Generate a video with native Swift/MLX LTX.",
        discussion: """
        Prints the output MP4 path to stdout.
        Progress and diagnostics are printed to stderr.

        Examples:
          swift run mere.run video generate "a cinematic drone flythrough over snowy mountains"
          swift run mere.run video generate "woman walking in neon rain" --image frame.png
          swift run mere.run video generate "city time-lapse" --variant unified-av --model video-ltx-av
        """
    )

    @Argument(help: "Prompt for video generation.")
    var prompt: String

    @Option(name: [.customShort("o"), .long], help: "Output MP4 path (default: ./mererun-video-<timestamp>.mp4).")
    var output: String?

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local path to the LTX model root.")
    var model: String = ModelResolver.ModelID.ltxVideoAV.rawValue

    @Option(name: [.customLong("variant")], help: "Native model variant to run.")
    var variant: LTXVideoVariant = .distilled

    @Option(name: [.customLong("model-root")], help: "Local LTX model root. Takes precedence over --model.")
    var modelRoot: String?

    @Option(name: [.long], help: "Output width (must be divisible by 64; auto-snapped down).")
    var width: Int = 768

    @Option(name: [.long], help: "Output height (must be divisible by 64; auto-snapped down).")
    var height: Int = 512

    @Option(name: [.customLong("num-frames")], help: "Frame count (must be 8n+1; auto-adjusted).")
    var numFrames: Int = 65

    @Option(name: [.long], help: "Frames per second.")
    var fps: Int = 24

    @Option(name: [.long], help: "Seed value.")
    var seed: Int?

    @Option(name: [.long], help: "Optional source image path (enables image-to-video).")
    var image: String?

    @Option(name: [.customLong("image-strength")], help: "Image conditioning strength in [0, 1].")
    var imageStrength: Float = 1.0

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    func run() async throws {
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else {
            throw ValidationError("Prompt cannot be empty.")
        }

        guard fps > 0 else {
            throw ValidationError("--fps must be >= 1")
        }
        guard width >= 64 else {
            throw ValidationError("--width must be >= 64")
        }
        guard height >= 64 else {
            throw ValidationError("--height must be >= 64")
        }
        guard numFrames >= 9 else {
            throw ValidationError("--num-frames must be >= 9")
        }
        guard (0...1).contains(imageStrength) else {
            throw ValidationError("--image-strength must be between 0 and 1")
        }

        let resolvedWidth = max(64, (width / 64) * 64)
        let resolvedHeight = max(64, (height / 64) * 64)
        let resolvedNumFrames = max(9, ((numFrames - 1) / 8) * 8 + 1)
        if !quiet {
            if resolvedWidth != width || resolvedHeight != height {
                CLIStderr.write("Adjusted size to \(resolvedWidth)x\(resolvedHeight) (must be divisible by 64)\n")
            }
            if resolvedNumFrames != numFrames {
                CLIStderr.write("Adjusted frame count to \(resolvedNumFrames) (must satisfy 8n+1)\n")
            }
        }

        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-video", defaultExtension: "mp4")
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

        let resolvedModelRoot = try await resolveVideoModelRoot(
            explicitModelRoot: modelRoot,
            requestedModel: model,
            variant: variant
        ).path

        try await runNativeGenerate(
            prompt: trimmedPrompt,
            width: resolvedWidth,
            height: resolvedHeight,
            numFrames: resolvedNumFrames,
            fps: fps,
            seed: seed ?? 42,
            variant: variant,
            sourceImageURL: sourceImageURL,
            imageStrength: imageStrength,
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
        modelRoot: String,
        outputURL: URL
    ) async throws {
        let rootURL = URL(fileURLWithPath: modelRoot).standardizedFileURL
        try validateNativeModelRoot(rootURL)
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
                        imageStrength: imageStrength
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
                        imageStrength: imageStrength
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
        }

        if !quiet {
            CLIStderr.write("Saved: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    private func mediaHasAudioTrack(at url: URL) async -> Bool {
        MediaVideoIO.hasAudioTrack(url)
    }
}

func validateNativeModelRoot(_ rootURL: URL) throws {
    let fm = FileManager.default
    let rootURL = rootURL.resolvingSymlinksInPath()
    var isDirectory: ObjCBool = false
    guard fm.fileExists(atPath: rootURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw ValidationError("Model root directory not found: \(rootURL.path)")
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

    let trimmedModel = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
    if !trimmedModel.isEmpty {
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
                zeroModels.appendingPathComponent("LTX-2-mlx-av", isDirectory: true).path,
                home.appendingPathComponent("models/video-ltx-av", isDirectory: true).path,
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

private func isNativeVideoModelRootAvailable(at path: String) -> Bool {
    let rootURL = URL(fileURLWithPath: path).standardizedFileURL
    do {
        try validateNativeModelRoot(rootURL)
        return true
    } catch {
        return false
    }
}
