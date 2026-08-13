import ArgumentParser
import Foundation
import MediaIO
import MereRunCore

struct VideoDubIt: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "dub-it",
        abstract: "Generate synchronized video and audio identity from one LTX 2.5 IC-LoRA reference.",
        discussion: """
        Runs the official two-stage distilled Dub-It recipe natively in Swift/MLX.
        Frame count and frame rate come from the reference video. Its video is
        appended as IC-LoRA conditioning at both stages, while its audio is
        appended as clean negative-time reference tokens.

        Example:
          mere.run video dub-it "The speaker performs on a rain-lit street" \
            --reference-video speaker.mp4 --ic-lora dub-it.safetensors
        """
    )

    @Argument(help: "Prompt describing the new scene and performance.")
    var prompt: String

    @Option(name: [.customLong("reference-video")], help: "Reference container with both video and audio tracks.")
    var referenceVideo: String

    @Option(name: [.customLong("ic-lora")], help: "Exactly one Dub-It/IC-LoRA safetensors adapter.")
    var icLoRA: String

    @Option(name: [.customLong("ic-lora-strength")], help: "IC-LoRA runtime multiplier.")
    var icLoRAStrength: Float = 1

    @Option(name: [.customLong("reference-strength")], help: "Video reference conditioning strength in [0, 1].")
    var referenceStrength: Float = 1

    @Option(name: [.customShort("m"), .long], help: "Official LTX 2.5 model id or local root.")
    var model: String = ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue

    @Option(name: [.customLong("model-root")], help: "Explicit official LTX 2.5 checkpoint root.")
    var modelRoot: String?

    @Option(name: [.customShort("o"), .long], help: "Output MP4 path.")
    var output: String?

    @Option(name: [.long], help: "Output width; must be divisible by 64.")
    var width: Int = 1536

    @Option(name: [.long], help: "Output height; must be divisible by 64.")
    var height: Int = 1024

    @Option(name: [.long], help: "Deterministic generation seed.")
    var seed: Int = 10

    @Option(
        name: [.customLong("image-conditioning")],
        help: "Optional timed image guide as PIXEL_FRAME:PATH[:STRENGTH[:CRF]]. Repeatable."
    )
    var imageConditioningArguments: [String] = []

    @Option(
        name: [.customLong("stage-1-sigmas")],
        parsing: .upToNextOption,
        help: "Explicit descending stage-one sigma schedule ending at zero."
    )
    var stage1Sigmas: [Float] = []

    @Option(
        name: [.customLong("stage-2-sigmas")],
        parsing: .upToNextOption,
        help: "Explicit descending stage-two sigma schedule ending at zero."
    )
    var stage2Sigmas: [Float] = []

    @Flag(
        name: [.customLong("enhance-prompt")],
        help: "Expand the request with the native Gemma-4 LTX 2.5 prompt enhancer."
    )
    var enhancePrompt = false

    @Option(
        name: [.customLong("prompt-enhancer-model")],
        help: "Managed generative Gemma-4 instruct model id for prompt enhancement."
    )
    var promptEnhancerModel: String?

    @Option(
        name: [.customLong("prompt-enhancer-model-root")],
        help: "Explicit local generative Gemma-4 instruct checkpoint root."
    )
    var promptEnhancerModelRoot: String?

    @Option(
        name: [.customLong("video-decoder")],
        help: "LTX 2.5 decoder: diffusion or convolutional."
    )
    var videoDecoder: LTXVideoDecoderKind = .diffusion

    @Flag(name: [.short, .long], help: "Suppress progress diagnostics.")
    var quiet = false

    mutating func validate() throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("Prompt cannot be empty.")
        }
        guard width > 0, height > 0, width.isMultiple(of: 64), height.isMultiple(of: 64) else {
            throw ValidationError("--width and --height must be positive multiples of 64.")
        }
        guard icLoRAStrength.isFinite else {
            throw ValidationError("--ic-lora-strength must be finite.")
        }
        guard referenceStrength.isFinite, (0...1).contains(referenceStrength) else {
            throw ValidationError("--reference-strength must be in [0, 1].")
        }
        if !stage1Sigmas.isEmpty { _ = try validatedLTXSigmaSchedule(stage1Sigmas) }
        if !stage2Sigmas.isEmpty { _ = try validatedLTXSigmaSchedule(stage2Sigmas) }
    }

    mutating func run() async throws {
        let referenceURL = URL(fileURLWithPath: referenceVideo).standardizedFileURL
        guard FileManager.default.fileExists(atPath: referenceURL.path) else {
            throw ValidationError("Reference video not found: \(referenceURL.path)")
        }
        guard MediaVideoIO.hasAudioTrack(referenceURL) else {
            throw ValidationError("Dub-It reference video must contain an audio track.")
        }
        let resolvedICLoRA = try ManagedAdapterArgumentResolver.resolve(
            icLoRA,
            baseModelID: ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
        ) ?? icLoRA
        let icLoRAURL = URL(fileURLWithPath: resolvedICLoRA).standardizedFileURL
        guard FileManager.default.fileExists(atPath: icLoRAURL.path) else {
            throw ValidationError("IC-LoRA not found: \(icLoRAURL.path)")
        }

        let inspectionDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-dub-it-inspect-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: inspectionDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: inspectionDirectory) }
        let sequence = try MediaVideoIO.extractFrames(from: referenceURL, into: inspectionDirectory)
        guard sequence.frameURLs.count >= 9 else {
            throw ValidationError("Dub-It reference video must contain at least 9 decodable frames.")
        }
        let numFrames = ((sequence.frameURLs.count - 1) / 8) * 8 + 1
        let fps = max(1, sequence.fps)

        let imageConditionings = try parseImageConditionings(frameCount: numFrames)
        let generationPrompt: String
        if enhancePrompt {
            let referenceImage = imageConditionings.first?.imageURL
            if !quiet { CLIStderr.write("Enhancing Dub-It prompt with native Gemma-4\n") }
            generationPrompt = try await LTXPromptEnhancer.enhance(
                prompt: prompt,
                modelID: promptEnhancerModel,
                modelRoot: promptEnhancerModelRoot.map {
                    URL(fileURLWithPath: $0).standardizedFileURL
                },
                referenceImage: referenceImage
            )
        } else {
            generationPrompt = prompt
        }

        let root = try await resolveVideoModelRoot(
            explicitModelRoot: modelRoot,
            requestedModel: model,
            variant: .unifiedAV
        )
        guard isLTX25ModelRoot(root) else {
            throw ValidationError("video dub-it requires an official LTX 2.5 checkpoint.")
        }
        let outputURL = CLIOutput.resolveOutputURL(
            output,
            defaultPrefix: "mererun-ltx25-dub-it",
            defaultExtension: "mp4"
        )

        if !quiet {
            CLIStderr.write("Engine: native LTX 2.5 Dub-It\n")
            CLIStderr.write("Reference: \(referenceURL.path)\n")
            CLIStderr.write("Reference timeline: \(numFrames) frames @ \(fps) fps\n")
            CLIStderr.write("Output: \(width)x\(height)\n")
        }

        let generator = LTXUnifiedAVGenerator()
        do {
            // Dub-It is an official distilled two-stage recipe; loading the
            // distilled transformer directly avoids the full dev/CFG lane.
            try await generator.load(modelRoot: root, videoDecoder: videoDecoder)
            let result = try await generator.generate(
                options: LTXUnifiedAVGenerationOptions(
                    prompt: generationPrompt,
                    width: width,
                    height: height,
                    numFrames: numFrames,
                    fps: fps,
                    seed: seed,
                    imageConditionings: imageConditionings,
                    loras: [LTXLoRAConfiguration(url: icLoRAURL, strength: icLoRAStrength)],
                    sigmas: stage1Sigmas.isEmpty ? nil : stage1Sigmas,
                    stage2Sigmas: stage2Sigmas.isEmpty ? nil : stage2Sigmas,
                    dubIt: LTXDubItOptions(
                        referenceVideoURL: referenceURL,
                        referenceStrength: referenceStrength
                    )
                )
            )
            await generator.unload()
            try LTXVideoMP4Writer.writeMP4(
                frames: result.frames,
                fps: result.playbackFPS,
                to: outputURL,
                audioWaveform: result.audioWaveform,
                audioSampleRate: result.audioSampleRate
            )
        } catch {
            await generator.unload()
            throw error
        }
        if !quiet { CLIStderr.write("Saved: \(outputURL.path)\n") }
        print(outputURL.path)
    }

    private func parseImageConditionings(frameCount: Int) throws -> [LTXVideoConditioningInput] {
        try imageConditioningArguments.map { raw in
            let pieces = raw.split(separator: ":", maxSplits: 3, omittingEmptySubsequences: false)
            guard (2...4).contains(pieces.count),
                  let frame = Int(pieces[0]),
                  (0..<frameCount).contains(frame) else {
                throw ValidationError(
                    "--image-conditioning must be PIXEL_FRAME:PATH[:STRENGTH[:CRF]] inside 0..<\(frameCount)."
                )
            }
            let strength: Float
            if pieces.count == 3 {
                guard let parsed = Float(pieces[2]), parsed.isFinite, (0...1).contains(parsed) else {
                    throw ValidationError("Image conditioning strength must be in [0, 1].")
                }
                strength = parsed
            } else {
                strength = 1
            }
            let crf: Int?
            if pieces.count == 4 {
                guard let parsed = Int(pieces[3]), (0...51).contains(parsed) else {
                    throw ValidationError("Image conditioning CRF must be in 0...51.")
                }
                crf = parsed
            } else {
                crf = nil
            }
            let url = URL(fileURLWithPath: String(pieces[1])).standardizedFileURL
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
}
