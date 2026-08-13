import ArgumentParser
import Foundation
import MediaIO
import MereRunCore

struct AudioGenerate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate audio from text with the native LTX-2.5 audio-only model.",
        discussion: """
        Runs the LTX-2.5 text-to-audio pipeline without constructing or denoising
        the video modality. The full gated LTX-2.5 checkpoint is required.

        Examples:
          mere.run audio generate "A quiet forest at dawn with distant birds"
          mere.run audio generate "Ocean surf" --duration 8 --output surf.wav
          mere.run audio generate "A cinematic impact" --steps 40 --audio-cfg-guidance-scale 8
        """
    )

    @Argument(help: "Audio description, including speech, ambience, music, and timing cues.")
    var prompt: String

    @Option(name: [.customShort("o"), .long], help: "Output WAV path.")
    var output: String = "ltx25-audio.wav"

    @Option(name: [.customShort("m"), .long], help: "Managed LTX-2.5 full model id or local path.")
    var model: String = ModelResolver.ModelID.ltxVideo25FullBF16.rawValue

    @Option(name: [.customLong("model-root")], help: "Explicit local LTX-2.5 full checkpoint root.")
    var modelRoot: String?

    @Option(name: [.customLong("negative-prompt")], help: "Negative audio prompt.")
    var negativePrompt: String = LTXUnifiedAVGenerationOptions.defaultNegativePrompt

    @Flag(
        name: [.customLong("enhance-prompt")],
        help: "Expand the request with the native Gemma-4 LTX-2.5 caption enhancer."
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

    @Option(name: [.customLong("duration")], help: "Requested duration in seconds, snapped to the LTX frame grid.")
    var duration: Double?

    @Option(
        name: [.customLong("auto-duration")],
        parsing: .upToNextOption,
        help: "Use DurationHead with MIN_SECONDS MAX_SECONDS."
    )
    var autoDuration: [Double] = []

    @Option(
        name: [.customLong("num-frames")],
        help: "Duration as 8n+1 frames. Omit to predict within the official 1...20s range."
    )
    var numFrames: Int?

    @Option(
        name: [.customLong("fps")],
        help: "Video-clock frame rate used to define audio duration; fractional rates are supported."
    )
    var fps: Double = 24

    @Option(name: [.customLong("steps")], help: "Diffusion steps.")
    var steps: Int = 30

    @Option(name: [.customLong("seed")], help: "Deterministic generation seed.")
    var seed: Int = 10

    @Option(name: [.customLong("audio-cfg-guidance-scale")], help: "Audio classifier-free guidance scale.")
    var classifierFreeGuidanceScale: Float = 7

    @Option(name: [.customLong("audio-stg-guidance-scale")], help: "Audio spatiotemporal guidance scale.")
    var spatioTemporalGuidanceScale: Float = 1

    @Option(name: [.customLong("audio-rescale")], help: "Audio guidance rescale in [0, 1].")
    var guidanceRescale: Float = 0.7

    @Option(name: [.customLong("audio-stg-block")], help: "Transformer block skipped for audio STG. Repeatable.")
    var spatioTemporalBlocks: [Int] = [28]

    @Option(name: [.customLong("audio-skip-step")], help: "Reuse the prior denoised estimate for interleaved steps.")
    var skipStep: Int = 0

    @Option(
        name: [.customLong("sigmas")],
        help: "Custom comma-separated descending sigma schedule ending in 0."
    )
    var sigmaList: String?

    @Option(name: [.customLong("lora")], help: "LTX LoRA as PATH[=STRENGTH]. Repeat to stack adapters.")
    var loraArguments: [String] = []

    @Flag(name: [.customLong("quiet")], help: "Suppress progress diagnostics.")
    var quiet = false

    mutating func validate() throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("Prompt cannot be empty.")
        }
        guard fps.isFinite, fps >= 1 else {
            throw ValidationError("--fps must be finite and >= 1")
        }
        guard steps > 0 else { throw ValidationError("--steps must be positive") }
        if let numFrames, numFrames < 9 {
            throw ValidationError("--num-frames must be >= 9")
        }
        if let duration, !duration.isFinite || duration <= 0 {
            throw ValidationError("--duration must be finite and positive")
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
            if numFrames != nil, !quiet {
                CLIStderr.write("Warning: --auto-duration is ignored because --num-frames was supplied.\n")
            }
        }
        guard guidanceRescale >= 0, guidanceRescale <= 1 else {
            throw ValidationError("--audio-rescale must be in [0, 1]")
        }
        guard spatioTemporalBlocks.allSatisfy({ (0..<48).contains($0) }) else {
            throw ValidationError("--audio-stg-block must be between 0 and 47")
        }
        guard skipStep >= 0 else { throw ValidationError("--audio-skip-step must be >= 0") }
    }

    mutating func run() async throws {
        let requestedFrames = duration.map { nearestLTXFrameCount(duration: $0, fps: fps) }
            ?? numFrames.map { max(9, (($0 - 1) / 8) * 8 + 1) }
            ?? 121
        let root = try await resolveVideoModelRoot(
            explicitModelRoot: modelRoot,
            requestedModel: model,
            variant: .unifiedAV
        )
        guard isLTX25FullModelRoot(root) else {
            throw ValidationError(
                "audio generate requires \(ModelResolver.ModelID.ltxVideo25FullBF16.rawValue)."
            )
        }
        let sigmas: [Float]?
        if let sigmaList {
            let values = sigmaList.split(separator: ",", omittingEmptySubsequences: false)
            guard values.count >= 2 else {
                throw ValidationError("--sigmas requires at least two comma-separated values")
            }
            sigmas = try values.map { raw in
                guard let value = Float(raw.trimmingCharacters(in: .whitespaces)), value.isFinite else {
                    throw ValidationError("Invalid sigma value: \(raw)")
                }
                return value
            }
            _ = try validatedLTXSigmaSchedule(sigmas!)
        } else {
            sigmas = nil
        }
        let loras = try parseLoRAs()
        let generationPrompt: String
        if enhancePrompt {
            if !quiet { CLIStderr.write("Enhancing audio prompt with native Gemma-4\n") }
            generationPrompt = try await LTXPromptEnhancer.enhance(
                prompt: prompt,
                modelID: promptEnhancerModel,
                modelRoot: promptEnhancerModelRoot.map {
                    URL(fileURLWithPath: $0).standardizedFileURL
                }
            )
            if !quiet { CLIStderr.write("Enhanced prompt: \(generationPrompt)\n") }
        } else {
            generationPrompt = prompt
        }
        let outputURL = URL(fileURLWithPath: output).standardizedFileURL
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let generator = LTXUnifiedAVGenerator()
        if !quiet { CLIStderr.write("Loading native LTX-2.5 audio-only pipeline\n") }
        do {
            _ = try await generator.loadTextToAudio(modelRoot: root)
            let resolvedFrames: Int
            if numFrames == nil, duration == nil {
                let range = autoDuration.count == 2
                    ? LTX25AutoDuration(
                        minimumSeconds: autoDuration[0],
                        maximumSeconds: autoDuration[1]
                    )
                    : LTX25AutoDuration(minimumSeconds: 1, maximumSeconds: 20)
                resolvedFrames = try await generator.predictFrameCount(
                    prompt: generationPrompt,
                    frameRate: fps,
                    range: range,
                    conditioning: .audioOnly
                )
                if !quiet {
                    CLIStderr.write(
                        "DurationHead selected \(resolvedFrames) frames "
                            + "(~\(String(format: "%.2f", Double(resolvedFrames) / fps))s)\n"
                    )
                }
            } else {
                resolvedFrames = requestedFrames
            }
            if !quiet { CLIStderr.write("Generating \(Double(resolvedFrames) / fps)s of audio\n") }
            let result = try await generator.generateTextToAudio(
                options: LTXTextToAudioGenerationOptions(
                    prompt: generationPrompt,
                    negativePrompt: negativePrompt,
                    numFrames: resolvedFrames,
                    fps: fps,
                    seed: seed,
                    inferenceSteps: steps,
                    guidance: LTXTextToAudioGuidance(
                        classifierFreeScale: classifierFreeGuidanceScale,
                        spatioTemporalScale: spatioTemporalGuidanceScale,
                        rescale: guidanceRescale,
                        spatioTemporalBlocks: Set(spatioTemporalBlocks),
                        skipStep: skipStep
                    ),
                    sigmas: sigmas,
                    loras: loras
                )
            )
            let prepared = try LTXVideoMP4Writer.prepareAudio(result.audioWaveform)
            try MediaAudioIO.writeFloatWAV(
                samples: prepared.interleaved,
                sampleRate: result.audioSampleRate,
                channels: prepared.channels,
                to: outputURL
            )
            await generator.unload()
            if !quiet { CLIStderr.write("Saved: \(outputURL.path)\n") }
            print(outputURL.path)
        } catch {
            await generator.unload()
            throw error
        }
    }

    private func parseLoRAs() throws -> [LTXLoRAConfiguration] {
        try loraArguments.map { raw in
            let separator = raw.lastIndex(of: "=")
            let path = separator.map { String(raw[..<$0]) } ?? raw
            let strength: Float
            if let separator {
                let rawStrength = String(raw[raw.index(after: separator)...])
                guard let value = Float(rawStrength), value.isFinite else {
                    throw ValidationError("--lora strength must be finite")
                }
                strength = value
            } else {
                strength = 1
            }
            guard !path.isEmpty else { throw ValidationError("--lora must be PATH[=STRENGTH]") }
            let resolvedPath = try ManagedAdapterArgumentResolver.resolve(
                path,
                baseModelID: ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
            ) ?? path
            let url = URL(fileURLWithPath: resolvedPath).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("LTX LoRA file not found: \(url.path)")
            }
            return LTXLoRAConfiguration(url: url, strength: strength)
        }
    }
}
