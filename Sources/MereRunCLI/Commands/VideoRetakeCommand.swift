import ArgumentParser
import Foundation
import MediaIO
import MereRunCore

struct VideoRetake: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "retake",
        abstract: "Regenerate a timed video/audio region with native LTX 2.5."
    )

    @Argument(help: "Prompt describing the replacement region.")
    var prompt: String

    @Option(name: [.customLong("source")], help: "Source MP4/MOV or EXR-frame directory whose timeline is retained.")
    var source: String

    @Option(name: [.customLong("frame-rate")], help: "Required frame rate for an EXR-frame source directory.")
    var frameRate: Double?

    @Option(name: [.customLong("start-time")], help: "Inclusive retake start time in seconds.")
    var startTime: Double

    @Option(name: [.customLong("end-time")], help: "Exclusive retake end time in seconds.")
    var endTime: Double

    @Option(name: [.customShort("m"), .long], help: "Official LTX 2.5 model id or local root.")
    var model: String = ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue

    @Option(name: [.customLong("model-root")], help: "Explicit official LTX 2.5 checkpoint root.")
    var modelRoot: String?

    @Option(name: [.customShort("o"), .long], help: "Output MP4 path.")
    var output: String?

    @Option(name: [.long], help: "Seed value.")
    var seed: Int = 10

    @Option(name: [.customLong("negative-prompt")], help: "Negative prompt for optional full-model guidance.")
    var negativePrompt: String = LTXUnifiedAVGenerationOptions.defaultNegativePrompt

    @Flag(name: [.customLong("enhance-prompt")], help: "Expand the prompt with native Gemma-4.")
    var enhancePrompt = false

    @Option(name: [.customLong("prompt-enhancer-model")], help: "Managed Gemma-4 prompt enhancer id.")
    var promptEnhancerModel: String?

    @Option(name: [.customLong("prompt-enhancer-model-root")], help: "Explicit prompt enhancer root.")
    var promptEnhancerModelRoot: String?

    @Option(name: [.long], help: "Stage-one denoising steps for an optional full checkpoint.")
    var steps: Int = 30

    @Option(name: [.customLong("sigmas")], parsing: .upToNextOption, help: "Explicit descending sigma schedule ending at zero.")
    var sigmas: [Float] = []

    @Option(name: [.customLong("lora")], help: "LTX LoRA as PATH[=STRENGTH]. Repeatable.")
    var loraArguments: [String] = []

    @Option(name: [.customLong("video-cfg-guidance-scale")], help: "Video classifier-free guidance scale.")
    var videoCFGScale: Float = 3

    @Option(name: [.customLong("video-stg-scale")], help: "Video spatiotemporal guidance scale.")
    var videoSTGScale: Float = 1

    @Option(name: [.customLong("video-guidance-rescale")], help: "Video guidance rescale in [0, 1].")
    var videoRescale: Float = 0.7

    @Option(name: [.customLong("video-modality-scale")], help: "Video cross-modal guidance scale.")
    var videoModalityScale: Float = 3

    @Option(name: [.customLong("video-stg-block")], help: "Video STG block. Repeatable.")
    var videoSTGBlocks: [Int] = [28]

    @Option(name: [.customLong("video-guidance-skip-step")], help: "Video guidance skip interval.")
    var videoSkipStep: Int = 0

    @Option(name: [.customLong("audio-cfg-guidance-scale")], help: "Audio classifier-free guidance scale.")
    var audioCFGScale: Float = 7

    @Option(name: [.customLong("audio-stg-scale")], help: "Audio spatiotemporal guidance scale.")
    var audioSTGScale: Float = 1

    @Option(name: [.customLong("audio-guidance-rescale")], help: "Audio guidance rescale in [0, 1].")
    var audioRescale: Float = 0.7

    @Option(name: [.customLong("audio-modality-scale")], help: "Audio cross-modal guidance scale.")
    var audioModalityScale: Float = 3

    @Option(name: [.customLong("audio-stg-block")], help: "Audio STG block. Repeatable.")
    var audioSTGBlocks: [Int] = [28]

    @Option(name: [.customLong("audio-guidance-skip-step")], help: "Audio guidance skip interval.")
    var audioSkipStep: Int = 0

    @Option(name: [.customLong("video-decoder")], help: "LTX 2.5 decoder: diffusion or convolutional.")
    var videoDecoder: LTXVideoDecoderKind = .diffusion

    @Option(name: [.customLong("hdr")], help: "HDR source/output space: srgb-linear, acescg, or acescct.")
    var hdrColorSpace: LTXHDRColorSpace?

    @Option(name: [.customLong("hdr-transfer")], help: "HDR VAE working-space transfer: acescct or logc3.")
    var hdrTransfer: LTXHDRTransfer = .acesCCT

    @Flag(name: [.customLong("preserve-video")], help: "Keep the source video unchanged while retaking audio.")
    var preserveVideo = false

    @Flag(name: [.customLong("preserve-audio")], help: "Keep the source audio unchanged while retaking video.")
    var preserveAudio = false

    @Flag(name: [.short, .long], help: "Suppress progress diagnostics.")
    var quiet = false

    mutating func validate() throws {
        guard startTime.isFinite, endTime.isFinite, startTime >= 0, startTime < endTime else {
            throw ValidationError("Retake requires 0 <= --start-time < --end-time.")
        }
        guard steps > 0 else { throw ValidationError("--steps must be >= 1") }
        if !sigmas.isEmpty { _ = try validatedLTXSigmaSchedule(sigmas) }
        guard (0...1).contains(videoRescale), (0...1).contains(audioRescale) else {
            throw ValidationError("Retake guidance rescale values must be in [0, 1].")
        }
        guard videoSkipStep >= 0, audioSkipStep >= 0 else {
            throw ValidationError("Retake guidance skip intervals must be nonnegative.")
        }
        guard !(preserveVideo && preserveAudio) else {
            throw ValidationError("Retake must regenerate video, audio, or both.")
        }
        if let frameRate, !frameRate.isFinite || frameRate < 1 {
            throw ValidationError("--frame-rate must be finite and >= 1.")
        }
    }

    mutating func run() async throws {
        let sourceURL = URL(fileURLWithPath: source).standardizedFileURL
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw ValidationError("Source video not found: \(sourceURL.path)")
        }
        let isEXRSource = MediaHDRImageIO.isEXRDirectory(sourceURL)
        if isEXRSource, hdrColorSpace == nil {
            throw ValidationError("An EXR-frame source directory requires --hdr.")
        }
        if preserveAudio, (isEXRSource || !MediaVideoIO.hasAudioTrack(sourceURL)) {
            throw ValidationError("--preserve-audio requires a source audio track.")
        }

        let frameCount: Int
        let fps: Double
        let sourceWidth: Int
        let sourceHeight: Int
        if isEXRSource {
            guard let frameRate else {
                throw ValidationError("An EXR-frame source directory requires --frame-rate.")
            }
            let urls = try MediaHDRImageIO.exrFrameURLs(in: sourceURL)
            guard let first = urls.first else {
                throw ValidationError("The EXR-frame source directory is empty.")
            }
            let image = try MediaHDRImageIO.decodeEXR(first)
            frameCount = urls.count
            fps = frameRate
            sourceWidth = image.width
            sourceHeight = image.height
        } else {
            if frameRate != nil {
                throw ValidationError("--frame-rate is only valid for an EXR-frame source directory.")
            }
            let inspectionDirectory = FileManager.default.temporaryDirectory
                .appendingPathComponent("mererun-retake-inspect-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(
                at: inspectionDirectory,
                withIntermediateDirectories: true
            )
            defer { try? FileManager.default.removeItem(at: inspectionDirectory) }
            let sequence = try MediaVideoIO.extractFrames(from: sourceURL, into: inspectionDirectory)
            frameCount = sequence.frameURLs.count
            fps = max(1, sequence.fps)
            sourceWidth = sequence.frameWidth
            sourceHeight = sequence.frameHeight
        }
        guard frameCount >= 9, frameCount % 8 == 1 else {
            throw ValidationError("Retake source frame count must satisfy 8n+1 (got \(frameCount)).")
        }
        guard sourceWidth.isMultiple(of: 32), sourceHeight.isMultiple(of: 32) else {
            throw ValidationError(
                "Retake source dimensions must be divisible by 32 (got \(sourceWidth)x\(sourceHeight))."
            )
        }
        let duration = Double(frameCount) / fps
        guard endTime <= duration else {
            throw ValidationError("--end-time \(endTime) exceeds the decoded source duration \(duration).")
        }

        let root = try await resolveVideoModelRoot(
            explicitModelRoot: modelRoot,
            requestedModel: model,
            variant: .unifiedAV
        )
        guard isLTX25ModelRoot(root) else {
            throw ValidationError("video retake requires an official LTX 2.5 checkpoint.")
        }
        let loras = try parseLoRAs(
            baseModelID: isLTX25FullModelRoot(root)
                ? ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
                : ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
        )
        let generationPrompt: String
        if enhancePrompt {
            generationPrompt = try await LTXPromptEnhancer.enhance(
                prompt: prompt,
                modelID: promptEnhancerModel,
                modelRoot: promptEnhancerModelRoot.map {
                    URL(fileURLWithPath: $0).standardizedFileURL
                }
            )
        } else {
            generationPrompt = prompt
        }
        let outputURL = CLIOutput.resolveOutputURL(
            output,
            defaultPrefix: "mererun-ltx25-retake",
            defaultExtension: "mp4"
        )

        if !quiet {
            CLIStderr.write("Engine: native LTX 2.5 Retake\n")
            CLIStderr.write("Region: \(startTime)s...\(endTime)s\n")
            CLIStderr.write("Source: \(sourceURL.path)\n")
        }
        let generator = LTXUnifiedAVGenerator()
        do {
            if isLTX25FullModelRoot(root) {
                try await generator.loadFull(
                    modelRoot: root,
                    videoDecoder: videoDecoder,
                    videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                )
            } else {
                try await generator.load(
                    modelRoot: root,
                    videoDecoder: videoDecoder,
                    videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                )
            }
            let result = try await generator.generate(
                options: LTXUnifiedAVGenerationOptions(
                    prompt: generationPrompt,
                    negativePrompt: negativePrompt,
                    width: sourceWidth,
                    height: sourceHeight,
                    numFrames: frameCount,
                    fps: fps,
                    seed: seed,
                    inferenceSteps: steps,
                    videoGuidance: LTXMultiModalGuidance(
                        classifierFreeScale: videoCFGScale,
                        spatioTemporalScale: videoSTGScale,
                        rescale: videoRescale,
                        modalityScale: videoModalityScale,
                        spatioTemporalBlocks: Set(videoSTGBlocks),
                        skipStep: videoSkipStep
                    ),
                    audioGuidance: LTXMultiModalGuidance(
                        classifierFreeScale: audioCFGScale,
                        spatioTemporalScale: audioSTGScale,
                        rescale: audioRescale,
                        modalityScale: audioModalityScale,
                        spatioTemporalBlocks: Set(audioSTGBlocks),
                        skipStep: audioSkipStep
                    ),
                    loras: loras,
                    sigmas: sigmas.isEmpty ? nil : sigmas,
                    hdrColorSpace: hdrColorSpace,
                    hdrTransfer: hdrTransfer,
                    retake: LTXRetakeOptions(
                        sourceVideoURL: sourceURL,
                        startTime: startTime,
                        endTime: endTime,
                        regenerateVideo: !preserveVideo,
                        regenerateAudio: !preserveAudio
                    )
                )
            )
            await generator.unload()
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
        } catch {
            await generator.unload()
            throw error
        }
        if !quiet { CLIStderr.write("Saved: \(outputURL.path)\n") }
        print(outputURL.path)
    }

    private func parseLoRAs(baseModelID: String) throws -> [LTXLoRAConfiguration] {
        try loraArguments.map { raw in
            let separator = raw.lastIndex(of: "=")
            let path = separator.map { String(raw[..<$0]) } ?? raw
            let strength: Float
            if let separator {
                let rawStrength = String(raw[raw.index(after: separator)...])
                guard let parsed = Float(rawStrength), parsed.isFinite else {
                    throw ValidationError("--lora strength must be finite.")
                }
                strength = parsed
            } else {
                strength = 1
            }
            let resolvedPath = try ManagedAdapterArgumentResolver.resolve(
                path,
                baseModelID: baseModelID
            ) ?? path
            let url = URL(fileURLWithPath: resolvedPath).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("LTX LoRA not found: \(url.path)")
            }
            return LTXLoRAConfiguration(url: url, strength: strength)
        }
    }
}
