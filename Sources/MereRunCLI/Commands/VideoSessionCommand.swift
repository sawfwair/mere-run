import ArgumentParser
import Foundation
import MediaIO
import MLX
import MereRunCore

enum LTXVideoSessionResponseStatus: String, Codable, Hashable, Sendable {
    case result
    case error
}

enum LTXVideoSessionPipeline: String, Codable, Hashable, Sendable {
    case twoStage = "two-stage"
    case keyframeInterpolation = "keyframe-interpolation"
    case devOneStage = "dev-one-stage"

    var generationPipeline: LTXGenerationPipeline {
        switch self {
        case .twoStage: .twoStage
        case .keyframeInterpolation: .keyframeInterpolation
        case .devOneStage: .devOneStage
        }
    }
}

enum LTXVideoSessionSampler: String, Codable, Hashable, Sendable {
    case euler
    case res2s
}

struct LTXVideoSessionRequest: Codable, Hashable, Sendable {
    let id: String?
    let prompt: String
    let output: String
    let width: Int?
    let height: Int?
    let numFrames: Int?
    let fps: Int?
    let seed: Int?
    let image: String?
    let imageStrength: Float?
    let endImage: String?
    let endImageStrength: Float?
    let transformerExecution: LTXTransformerExecution?
    let guidanceProjectionCache: LTXGuidanceProjectionCacheMode?
    let teaCache: Bool?
    let teaCacheThreshold: Float?
    let teaCacheCalibrationOutput: String?
    let inferenceSteps: Int?
    let pipeline: LTXVideoSessionPipeline?
    let sampler: LTXVideoSessionSampler?
    let videoCFGGuidanceScale: Float?
    let audioCFGGuidanceScale: Float?
    let videoSTGScale: Float?
    let audioSTGScale: Float?
    let audioToVideoScale: Float?
    let videoToAudioScale: Float?
    let distilledLoRAStrengthStage1: Float?
    let distilledLoRAStrengthStage2: Float?

    enum CodingKeys: String, CodingKey {
        case id
        case prompt
        case output
        case width
        case height
        case numFrames = "num_frames"
        case fps
        case seed
        case image
        case imageStrength = "image_strength"
        case endImage = "end_image"
        case endImageStrength = "end_image_strength"
        case transformerExecution = "transformer_execution"
        case guidanceProjectionCache = "guidance_projection_cache"
        case teaCache = "tea_cache"
        case teaCacheThreshold = "tea_cache_threshold"
        case teaCacheCalibrationOutput = "tea_cache_calibration_output"
        case inferenceSteps = "inference_steps"
        case pipeline
        case sampler
        case videoCFGGuidanceScale = "video_cfg_guidance_scale"
        case audioCFGGuidanceScale = "audio_cfg_guidance_scale"
        case videoSTGScale = "video_stg_scale"
        case audioSTGScale = "audio_stg_scale"
        case audioToVideoScale = "audio_to_video_scale"
        case videoToAudioScale = "video_to_audio_scale"
        case distilledLoRAStrengthStage1 = "distilled_lora_strength_stage_1"
        case distilledLoRAStrengthStage2 = "distilled_lora_strength_stage_2"
    }
}

struct LTXVideoSessionResponse: Codable, Hashable, Sendable {
    let status: LTXVideoSessionResponseStatus
    let id: String?
    let output: String?
    let seed: Int?
    let timings: LTXVideoTimingReport?
    let promptCache: LTXPromptCacheStatistics?
    let error: String?

    static func result(
        id: String?,
        output: String,
        seed: Int,
        timings: LTXVideoTimingReport,
        promptCache: LTXPromptCacheStatistics
    ) -> Self {
        Self(
            status: .result,
            id: id,
            output: output,
            seed: seed,
            timings: timings,
            promptCache: promptCache,
            error: nil
        )
    }

    static func failure(id: String?, error: Error) -> Self {
        let message = (error as? LocalizedError)?.errorDescription
            ?? String(describing: error)
        return Self(
            status: .error,
            id: id,
            output: nil,
            seed: nil,
            timings: nil,
            promptCache: nil,
            error: message
        )
    }
}

struct VideoSession: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "session",
        abstract: "Keep an LTX 2.3 or LTX 2.5 runtime resident for JSONL generation requests.",
        discussion: """
        Loads a standalone distilled or full LTX 2.3/LTX 2.5 bundle once, then
        reads one JSON request per line from stdin and writes one JSON result per
        line to stdout. Full-model lanes keep dev weights unchanged and activate
        the distilled adapter only for Stage 2. A bounded prompt cache reuses
        materialized Gemma connector outputs across requests. Diagnostics go to
        stderr. Requests are processed serially.

        Request keys use snake_case. Required keys are prompt and output. Optional
        keys are id, width, height, num_frames, fps, seed, image,
        image_strength, end_image, end_image_strength, and
        transformer_execution. Full-model requests may also set
        guidance_projection_cache, tea_cache, tea_cache_threshold,
        tea_cache_calibration_output,
        inference_steps, pipeline, sampler, video_cfg_guidance_scale,
        audio_cfg_guidance_scale, video_stg_scale, audio_stg_scale,
        audio_to_video_scale, video_to_audio_scale, and stage-specific
        distilled_lora_strength values.
        """
    )

    @Option(name: [.customShort("m"), .long], help: "Managed standalone or full LTX 2.3/LTX 2.5 model id, or local model root.")
    var model: String = ModelResolver.ModelID.ltxVideo23AVMLX.rawValue

    @Option(name: [.customLong("model-root")], help: "Local standalone or full LTX 2.3/LTX 2.5 model root. Takes precedence over --model.")
    var modelRoot: String?

    @Option(
        name: [.customLong("video-decoder")],
        help: "LTX 2.5 VAE decoder: diffusion for maximum fidelity or convolutional for faster decode."
    )
    var videoDecoder: LTXVideoDecoderKind?

    @Option(
        name: [.customLong("ltx-transformer-execution")],
        help: "LTX 2.5 transformer blocks: eager or opt-in shared-graph compiled execution; fusion can change floating-point results slightly."
    )
    var transformerExecution: LTXTransformerExecution = .eager

    @Option(
        name: [.customLong("ltx-guidance-projection-cache")],
        help: "Reuse positive-prompt attention projections across full-model guidance passes: automatic, disabled, or enabled."
    )
    var guidanceProjectionCache: LTXGuidanceProjectionCacheMode = .disabled

    @Flag(
        name: [.customLong("ltx-teacache")],
        help: "Enable calibrated TeaCache block-residual reuse by default for full LTX 2.5 session requests."
    )
    var teaCache = false

    @Option(
        name: [.customLong("ltx-teacache-threshold")],
        help: "Default TeaCache threshold override for this session."
    )
    var teaCacheThreshold: Float?

    @Option(
        name: [.customLong("prompt-cache-capacity")],
        help: "Number of materialized prompt embeddings retained in the resident session; 0 disables caching."
    )
    var promptCacheCapacity: Int = 8

    @Flag(name: [.short, .long], help: "Suppress session diagnostics on stderr.")
    var quiet: Bool = false

    func run() async throws {
        guard promptCacheCapacity >= 0 else {
            throw ValidationError("--prompt-cache-capacity must be non-negative")
        }
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let rootURL = try await resolveVideoModelRoot(
            explicitModelRoot: modelRoot,
            requestedModel: model,
            variant: .unifiedAV,
            allowAutoDownload: true
        )
        try validateNativeModelRoot(rootURL)
        let usesLTX25 = isLTX25ModelRoot(rootURL)
        let usesFullTwoStage = isLTX23FullModelRoot(rootURL) || isLTX25FullModelRoot(rootURL)
        guard usesFullTwoStage || isLTX23SplitModelRoot(rootURL) || usesLTX25 else {
            throw ValidationError(
                "video session requires \(ModelResolver.ModelID.ltxVideo23AVMLX.rawValue) "
                    + ", \(ModelResolver.ModelID.ltxVideo23FullMLX.rawValue), "
                    + "\(ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue), or "
                    + "\(ModelResolver.ModelID.ltxVideo25FullBF16.rawValue)."
            )
        }

        let generator = LTXUnifiedAVGenerator()
        let resolvedVideoDecoder = videoDecoder
            ?? (isLTX25FullModelRoot(rootURL) ? .diffusion : .convolutional)
        let loadTimings = usesFullTwoStage
            ? try await generator.loadFullReusable(
                modelRoot: rootURL,
                videoDecoder: resolvedVideoDecoder
            )
            : try await generator.load(
                modelRoot: rootURL,
                videoDecoder: resolvedVideoDecoder
            )
        await generator.configurePromptCache(capacity: promptCacheCapacity)
        if !quiet {
            CLIStderr.write("LTX video session ready: \(rootURL.path)\n")
            CLIStderr.write(String(format: "Model load: %.3fs\n", loadTimings.totalSeconds))
        }

        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.sortedKeys]
        var completedRequestCount = 0

        while let line = readLine(strippingNewline: true) {
            let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedLine.isEmpty else { continue }

            let request: LTXVideoSessionRequest
            do {
                request = try decoder.decode(LTXVideoSessionRequest.self, from: Data(trimmedLine.utf8))
            } catch {
                try writeSessionResponse(.failure(id: nil, error: error), encoder: encoder)
                continue
            }

            do {
                let response = try await process(
                    request: request,
                    generator: generator,
                    modelRoot: rootURL,
                    usesFullTwoStage: usesFullTwoStage,
                    usesLTX25: usesLTX25,
                    loadTimings: completedRequestCount == 0 ? loadTimings : LTXLoadTimings(),
                    residentModelReused: completedRequestCount > 0
                )
                try writeSessionResponse(response, encoder: encoder)
                completedRequestCount += 1
            } catch {
                try writeSessionResponse(.failure(id: request.id, error: error), encoder: encoder)
            }
            Memory.clearCache()
        }

        await generator.unload()
        if !quiet {
            CLIStderr.write("LTX video session closed after \(completedRequestCount) generation(s).\n")
        }
    }

    private func process(
        request: LTXVideoSessionRequest,
        generator: LTXUnifiedAVGenerator,
        modelRoot: URL,
        usesFullTwoStage: Bool,
        usesLTX25: Bool,
        loadTimings: LTXLoadTimings,
        residentModelReused: Bool
    ) async throws -> LTXVideoSessionResponse {
        let requestStart = videoMonotonicSeconds()
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw ValidationError("prompt cannot be empty")
        }

        let width = request.width ?? 768
        let height = request.height ?? 512
        let numFrames = request.numFrames ?? 65
        let fps = request.fps ?? 24
        let seed = request.seed ?? (usesLTX25 ? 10 : 42)
        let imageStrength = request.imageStrength ?? 1
        let endImageStrength = request.endImageStrength ?? 1

        guard width >= 64, height >= 64, width % 64 == 0, height % 64 == 0 else {
            throw ValidationError("width and height must be at least 64 and divisible by 64")
        }
        guard numFrames >= 9, numFrames % 8 == 1 else {
            throw ValidationError("num_frames must be at least 9 and satisfy 8n+1")
        }
        guard fps > 0 else {
            throw ValidationError("fps must be positive")
        }
        guard (0...1).contains(imageStrength), (0...1).contains(endImageStrength) else {
            throw ValidationError("image strengths must be between 0 and 1")
        }
        guard request.endImage == nil || request.image != nil else {
            throw ValidationError("end_image requires image")
        }

        let sourceImageURL = try existingFileURL(request.image, field: "image")
        let endImageURL = try existingFileURL(request.endImage, field: "end_image")
        let outputURL = URL(fileURLWithPath: request.output).standardizedFileURL
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let pipeline = request.pipeline?.generationPipeline ?? .twoStage
        let sampler: LTXSamplerConfiguration = switch request.sampler {
        case .res2s: .hq
        case .euler, nil: LTXSamplerConfiguration()
        }
        let result = try await generator.generate(
            options: LTXUnifiedAVGenerationOptions(
                prompt: prompt,
                width: width,
                height: height,
                numFrames: numFrames,
                fps: Double(fps),
                seed: seed,
                inferenceSteps: request.inferenceSteps ?? 30,
                videoGuidance: LTXMultiModalGuidance(
                    classifierFreeScale: request.videoCFGGuidanceScale ?? 3,
                    spatioTemporalScale: request.videoSTGScale ?? 1,
                    rescale: 0.7,
                    modalityScale: request.audioToVideoScale ?? 3
                ),
                audioGuidance: LTXMultiModalGuidance(
                    classifierFreeScale: request.audioCFGGuidanceScale ?? 7,
                    spatioTemporalScale: request.audioSTGScale ?? 1,
                    rescale: 0.7,
                    modalityScale: request.videoToAudioScale ?? 3
                ),
                sourceImageURL: sourceImageURL,
                imageStrength: imageStrength,
                endImageURL: endImageURL,
                endImageStrength: endImageStrength,
                sampler: sampler,
                pipeline: pipeline,
                distilledLoRAStrengthStage1: request.distilledLoRAStrengthStage1 ?? 0,
                distilledLoRAStrengthStage2: request.distilledLoRAStrengthStage2
                    ?? (pipeline == .devOneStage ? 0 : 1),
                transformerExecution: request.transformerExecution ?? transformerExecution,
                guidanceProjectionCache: request.guidanceProjectionCache ?? guidanceProjectionCache,
                teaCache: (request.teaCache ?? teaCache) || request.teaCacheCalibrationOutput != nil
                    ? LTXTeaCacheConfiguration(
                        threshold: request.teaCacheThreshold ?? teaCacheThreshold,
                        calibrationOutputURL: request.teaCacheCalibrationOutput.map {
                            URL(fileURLWithPath: $0).standardizedFileURL
                        }
                    )
                    : nil
            )
        )

        let writeStart = videoMonotonicSeconds()
        try LTXVideoMP4Writer.writeMP4(
            frames: result.frames,
            fps: fps,
            to: outputURL,
            audioWaveform: result.audioWaveform,
            audioSampleRate: result.audioSampleRate
        )
        guard MediaVideoIO.hasAudioTrack(outputURL) else {
            throw ValidationError("session output has no audio track at \(outputURL.path)")
        }
        let writeSeconds = videoMonotonicSeconds() - writeStart
        let timings = LTXVideoTimingReport(
            mode: residentSessionMode(
                usesFullTwoStage: usesFullTwoStage,
                usesLTX25: usesLTX25,
                transformerExecution: request.transformerExecution ?? transformerExecution
            ),
            modelRoot: modelRoot.path,
            residentModelReused: residentModelReused,
            load: loadTimings,
            generation: result.timings,
            unloadSeconds: 0,
            mp4WriteSeconds: writeSeconds,
            totalSeconds: loadTimings.totalSeconds + videoMonotonicSeconds() - requestStart
        )
        return .result(
            id: request.id,
            output: outputURL.path,
            seed: seed,
            timings: timings,
            promptCache: await generator.promptCacheStatistics()
        )
    }

    private func residentSessionMode(
        usesFullTwoStage: Bool,
        usesLTX25: Bool,
        transformerExecution: LTXTransformerExecution
    ) -> String {
        let version = usesLTX25 ? "ltx25" : "ltx23"
        let pipeline = usesFullTwoStage ? "full-dev-distilled-lora" : "standalone-distilled"
        return "resident-\(version)-\(pipeline)-unified-av-\(transformerExecution.rawValue)"
    }

    private func existingFileURL(_ path: String?, field: String) throws -> URL? {
        guard let path else { return nil }
        let url = URL(fileURLWithPath: path).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("\(field) file not found: \(url.path)")
        }
        return url
    }

    private func writeSessionResponse(
        _ response: LTXVideoSessionResponse,
        encoder: JSONEncoder
    ) throws {
        var data = try encoder.encode(response)
        data.append(0x0A)
        FileHandle.standardOutput.write(data)
    }
}
