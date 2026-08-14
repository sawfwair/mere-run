import ArgumentParser
import Foundation
import Hummingbird
import MereRunCore
import NIOCore

struct MusicServe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start an ACE-Step or MiniMax Music 3 generation API."
    )

    @Option(name: [.long], help: "Host to bind to.")
    var host: String = "127.0.0.1"

    @Option(name: [.short, .long], help: "Port to listen on.")
    var port: Int = 8081

    @Option(name: [.customShort("m"), .long], help: "Managed ACE-Step/MiniMax model id or local checkpoint root.")
    var model: String = ModelResolver.ModelID.aceStep.rawValue

    @Option(
        name: [.customLong("memory-mode")],
        help: "MiniMax Music 3 loading: resident keeps all weights warm; staged lowers peak residency by reloading stages per request."
    )
    var miniMaxLoadingStrategy: MiniMaxMusic3LoadingStrategy?

    @Option(
        name: [.customLong("performance-mode")],
        help: "MiniMax Music 3 execution: reference, optimized (default), q8, or q4."
    )
    var miniMaxPerformanceMode: MiniMaxMusic3PerformanceMode?

    @Option(name: [.customLong("checkpoints-root")], help: "Root containing ACE-Step checkpoint directories.")
    var checkpointsRoot: String?

    @Option(name: [.customLong("decoder-subdirectory")], help: "ACE-Step decoder subdirectory.")
    var decoderSubdirectory: String = "acestep-v15-turbo"

    @Option(name: [.customLong("vae-subdirectory")], help: "ACE-Step VAE subdirectory.")
    var vaeSubdirectory: String = "vae"

    @Option(name: [.customLong("lm-subdirectory")], help: "Optional 5Hz LM subdirectory under the checkpoint root.")
    var lmSubdirectory: String?

    @Option(
        name: [.customLong("lm-model")],
        help: "Managed 5Hz LM model id or local planner root. Defaults to music-acestep-lm-1.7b when the selected checkpoint has no LM."
    )
    var lmModel: String?

    @Option(name: [.customLong("text-subdirectory")], help: "Optional text encoder subdirectory.")
    var textSubdirectory: String?

    @Option(
        name: [.customLong("adapter")],
        parsing: .upToNextOption,
        help: "PEFT LoRA or LyCORIS LoKr safetensors file(s) to keep resident."
    )
    var adapters: [String] = []

    @Option(name: [.customLong("adapter-kind")], help: "Adapter format: auto, lora, or lokr.")
    var adapterKind: ACEStepAdapterKind = .auto

    @Option(
        name: [.customLong("adapter-scale")],
        parsing: .upToNextOption,
        help: "One adapter scale for all paths, or one scale per adapter."
    )
    var adapterScales: [Float] = []

    @Option(name: [.customLong("api-key")], help: "Bearer token; also read from MERERUN_API_KEY.")
    var apiKey: String?

    func run() async throws {
        guard (1...65_535).contains(port) else {
            throw ValidationError("--port must be between 1 and 65535")
        }
        let resolvedAPIKey = apiKey
            ?? ProcessInfo.processInfo.environment[APIServe.apiKeyEnvironmentKey]
        if !Self.isLoopback(host), resolvedAPIKey?.isEmpty != false {
            throw ValidationError(
                "Non-loopback music API binds require --api-key or MERERUN_API_KEY."
            )
        }

        if isMiniMaxMusic3Request {
            try await runMiniMaxMusic3(apiKey: resolvedAPIKey)
            return
        }
        if miniMaxLoadingStrategy != nil || miniMaxPerformanceMode != nil {
            throw ValidationError("--memory-mode and --performance-mode require MiniMax Music 3.")
        }

        let root = try await ACEStepCLIHelper.resolveCheckpointsRoot(
            model: model,
            checkpointsRoot: checkpointsRoot,
            turboSubdirectory: decoderSubdirectory,
            vaeSubdirectory: vaeSubdirectory,
            lmSubdirectory: nil,
            textSubdirectory: textSubdirectory
        )
        let decoder = try ACEStepCLIHelper.resolveTurboSubdirectory(
            at: root,
            explicit: decoderSubdirectory
        )
        guard let text = try ACEStepCLIHelper.resolveTextSubdirectory(
            at: root,
            explicit: textSubdirectory
        ) else {
            throw ValidationError("ACE-Step text encoder not found.")
        }
        let lm = try await ACEStepCLIHelper.resolveLMResources(
            checkpointsRoot: root,
            lmModel: lmModel,
            lmSubdirectory: lmSubdirectory
        )
        let variant = try ACEStepCheckpointVariant.load(
            modelRootURL: root.appendingPathComponent(decoder, isDirectory: true)
        )
        let container = ACEStepModelContainer(
            decoderRootURL: root.appendingPathComponent(decoder, isDirectory: true),
            vaeRootURL: root.appendingPathComponent(vaeSubdirectory, isDirectory: true),
            lmRootURL: lm.rootURL,
            textEncoderRootURL: root.appendingPathComponent(text, isDirectory: true)
        )
        CLIStderr.write(
            "Loading resident ACE-Step \(variant.rawValue) session with planner \(lm.source)\n"
        )
        let resources = try await container.resources()
        let pipeline = try ACEStepPipeline(
            decoderResources: resources.decoderResources,
            vaeResources: resources.vaeResources,
            lmResources: resources.lmResources,
            textEncoderResources: resources.textEncoderResources
        )
        let loadedAdapters = try loadAdapters(into: pipeline)
        let server = MusicAPIServer(
            session: ACEStepGenerationSession(pipeline: pipeline),
            pipeline: pipeline,
            variant: variant,
            modelID: model,
            languageModelAvailable: true,
            languageModelSource: lm.source,
            adapters: loadedAdapters,
            apiKey: resolvedAPIKey
        )
        try await server.run(host: host, port: port)
    }

    private var isMiniMaxMusic3Request: Bool {
        if model == ModelResolver.ModelID.miniMaxMusic3.rawValue
            || model == MiniMaxMusic3Resources.repository
        {
            return true
        }
        return MiniMaxMusic3Resources.looksLikeRoot(
            ACEStepCLIHelper.resolveUserPath(model)
        )
    }

    private func runMiniMaxMusic3(apiKey: String?) async throws {
        if checkpointsRoot != nil
            || decoderSubdirectory != "acestep-v15-turbo"
            || vaeSubdirectory != "vae"
            || lmSubdirectory != nil
            || lmModel != nil
            || textSubdirectory != nil
            || !adapters.isEmpty
            || adapterKind != .auto
            || !adapterScales.isEmpty
        {
            throw ValidationError(
                "ACE-Step component, planner, and adapter options do not apply to MiniMax Music 3."
            )
        }
        let rootURL: URL
        if model == ModelResolver.ModelID.miniMaxMusic3.rawValue
            || model == MiniMaxMusic3Resources.repository
        {
            do {
                rootURL = try ModelResolver().resolve(.miniMaxMusic3).rootURL
            } catch {
                throw ValidationError(
                    "MiniMax Music 3 is not installed. Review its license, then run "
                        + "`mere.run model pull \(MiniMaxMusic3Resources.modelID) --accept-model-license`."
                )
            }
        } else {
            rootURL = ACEStepCLIHelper.resolveUserPath(model)
        }
        let resources = MiniMaxMusic3Resources(rootURL: rootURL)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw ValidationError(
                "Incomplete MiniMax Music 3 root at \(rootURL.path): "
                    + missing.map(\.lastPathComponent).joined(separator: ", ")
            )
        }
        let loadingStrategy = miniMaxLoadingStrategy ?? .resident
        let performanceMode = miniMaxPerformanceMode ?? .optimized
        CLIStderr.write(
            "Loading MiniMax Music 3 server in \(loadingStrategy.rawValue) memory mode "
                + "with \(performanceMode.rawValue) execution\n"
        )
        let server = try MiniMaxMusic3APIServer(
            resources: resources,
            modelID: ModelResolver.ModelID.miniMaxMusic3.rawValue,
            loadingStrategy: loadingStrategy,
            performanceMode: performanceMode,
            apiKey: apiKey
        )
        try await server.run(host: host, port: port)
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
                "--adapter-scale accepts one value for all adapters or one per adapter."
            )
        }
        return try adapters.enumerated().map { index, path in
            let scale = adapterScales.isEmpty
                ? 1
                : adapterScales.count == 1
                    ? adapterScales[0]
                    : adapterScales[index]
            let report = try pipeline.loadAdapter(
                from: ACEStepCLIHelper.resolveUserPath(path),
                kind: adapterKind,
                scale: scale
            )
            CLIStderr.write(
                "Loaded resident \(report.kind.rawValue) adapter "
                    + "\(report.filename) on \(report.matchedLayerCount) layers\n"
            )
            return report
        }
    }

    static func isLoopback(_ host: String) -> Bool {
        let normalized = host.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return normalized == "localhost"
            || normalized == "127.0.0.1"
            || normalized == "::1"
    }

    static func apiErrorMessage(_ error: Error) -> String {
        if let validationError = error as? ValidationError {
            return validationError.message
        }
        if let localizedError = error as? LocalizedError,
           let description = localizedError.errorDescription,
           !description.isEmpty
        {
            return description
        }
        return error.localizedDescription
    }
}

struct MusicAPIGenerationRequest: Codable, Sendable {
    var model: String?
    var prompt: String
    var lyrics: String?
    var instrumental: Bool?
    var instruction: String?
    var durationSeconds: Float?
    var quality: ACEStepQualityPreset?
    var task: ACEStepTask?
    var seed: UInt64?
    var retakeSeed: UInt64?
    var retakeVariance: Float?
    var candidates: Int?
    var steps: Int?
    var shift: Float?
    var inferMethod: ACEStepInferenceMethod?
    var guidanceScale: Float?
    var guidanceMode: ACEStepGuidanceMode?
    var cfgIntervalStart: Float?
    var cfgIntervalEnd: Float?
    var velocityNormThreshold: Float?
    var velocityEMAFactor: Float?
    var sampler: ACEStepSamplerMode?
    var useLanguageModel: Bool?
    var lmTopK: Int?
    var lmTopP: Float?
    var lmTemperature: Float?
    var lmRepetitionPenalty: Float?
    var lmCFGScale: Float?
    var lmNegativePrompt: String?
    var bpm: Int?
    var keyscale: String?
    var metadataLanguage: String?
    var timeSignature: String?
    var vocalLanguage: String?
    var sourceAudioPath: String?
    var referenceAudioPaths: [String]?
    var audioCoverStrength: Float?
    var coverNoiseStrength: Float?
    var sourceCaption: String?
    var sourceLyrics: String?
    var flowEditNMin: Float?
    var flowEditNMax: Float?
    var flowEditNAverage: Int?
    var trackName: String?
    var completeTrackClasses: [String]?
    var repaintStartSeconds: Float?
    var repaintEndSeconds: Float?
    var chunkMaskMode: ACEStepChunkMaskMode?
    var repaintMode: ACEStepRepaintMode?
    var repaintStrength: Float?
    var useTiledVAEDecode: Bool?
    var vaeChunkSize: Int?
    var vaeOverlap: Int?
    var responseFormat: String?

    enum CodingKeys: String, CodingKey {
        case model
        case prompt
        case lyrics
        case instrumental
        case instruction
        case durationSeconds = "duration_seconds"
        case quality
        case task
        case seed
        case retakeSeed = "retake_seed"
        case retakeVariance = "retake_variance"
        case candidates
        case steps
        case shift
        case inferMethod = "infer_method"
        case guidanceScale = "guidance_scale"
        case guidanceMode = "guidance_mode"
        case cfgIntervalStart = "cfg_interval_start"
        case cfgIntervalEnd = "cfg_interval_end"
        case velocityNormThreshold = "velocity_norm_threshold"
        case velocityEMAFactor = "velocity_ema_factor"
        case sampler
        case useLanguageModel = "use_lm"
        case lmTopK = "lm_top_k"
        case lmTopP = "lm_top_p"
        case lmTemperature = "lm_temperature"
        case lmRepetitionPenalty = "lm_repetition_penalty"
        case lmCFGScale = "lm_cfg_scale"
        case lmNegativePrompt = "lm_negative_prompt"
        case bpm
        case keyscale
        case metadataLanguage = "metadata_language"
        case timeSignature = "time_signature"
        case vocalLanguage = "vocal_language"
        case sourceAudioPath = "source_audio_path"
        case referenceAudioPaths = "reference_audio_paths"
        case audioCoverStrength = "audio_cover_strength"
        case coverNoiseStrength = "cover_noise_strength"
        case sourceCaption = "source_caption"
        case sourceLyrics = "source_lyrics"
        case flowEditNMin = "flow_edit_n_min"
        case flowEditNMax = "flow_edit_n_max"
        case flowEditNAverage = "flow_edit_n_average"
        case trackName = "track_name"
        case completeTrackClasses = "complete_track_classes"
        case repaintStartSeconds = "repaint_start_seconds"
        case repaintEndSeconds = "repaint_end_seconds"
        case chunkMaskMode = "chunk_mask_mode"
        case repaintMode = "repaint_mode"
        case repaintStrength = "repaint_strength"
        case useTiledVAEDecode = "use_tiled_vae_decode"
        case vaeChunkSize = "vae_chunk_size"
        case vaeOverlap = "vae_overlap"
        case responseFormat = "response_format"
    }
}

struct MusicAPIBatchRequest: Codable, Sendable {
    var requests: [MusicAPIGenerationRequest]
}

private struct MusicAPIHealthResponse: Codable {
    var status: String
    var model: String
    var checkpointVariant: ACEStepCheckpointVariant
    var resident: Bool
    var languageModelAvailable: Bool
    var languageModelSource: String
    var adapters: [ACEStepAdapterDescriptor]

    enum CodingKeys: String, CodingKey {
        case status
        case model
        case checkpointVariant = "checkpoint_variant"
        case resident
        case languageModelAvailable = "language_model_available"
        case languageModelSource = "language_model_source"
        case adapters
    }
}

private struct MusicAPICandidateResponse: Codable {
    var rank: Int
    var seed: UInt64
    var score: Float
    var metrics: ACEStepCandidateMetrics
    var audioBase64: String
    var format: String
    var sampleRate: Int
    var selected: Bool

    enum CodingKeys: String, CodingKey {
        case rank
        case seed
        case score
        case metrics
        case audioBase64 = "audio_base64"
        case format
        case sampleRate = "sample_rate"
        case selected
    }
}

private struct MusicAPIGenerationResponse: Codable {
    var model: String
    var quality: ACEStepQualityPreset
    var task: ACEStepTask
    var conditioningMetadata: ACEStepRecipeConditioningMetadata
    var candidates: [MusicAPICandidateResponse]

    enum CodingKeys: String, CodingKey {
        case model
        case quality
        case task
        case conditioningMetadata = "conditioning_metadata"
        case candidates
    }
}

private struct MusicAPIBatchResponse: Codable {
    var data: [MusicAPIGenerationResponse]
}

private struct PreparedMusicAPIRequest {
    var request: ACEStepSessionRequest
    var quality: ACEStepQualityPreset
    var task: ACEStepTask
    var candidateCount: Int
    var conditioningMetadata: ACEStepRecipeConditioningMetadata
}

private final class MusicAPIServer: @unchecked Sendable {
    let session: ACEStepGenerationSession
    let pipeline: ACEStepPipeline
    let variant: ACEStepCheckpointVariant
    let modelID: String
    let languageModelAvailable: Bool
    let languageModelSource: String
    let adapters: [ACEStepAdapterDescriptor]
    let apiKey: String?

    init(
        session: ACEStepGenerationSession,
        pipeline: ACEStepPipeline,
        variant: ACEStepCheckpointVariant,
        modelID: String,
        languageModelAvailable: Bool,
        languageModelSource: String,
        adapters: [ACEStepAdapterDescriptor],
        apiKey: String?
    ) {
        self.session = session
        self.pipeline = pipeline
        self.variant = variant
        self.modelID = modelID
        self.languageModelAvailable = languageModelAvailable
        self.languageModelSource = languageModelSource
        self.adapters = adapters
        self.apiKey = apiKey
    }

    func run(host: String, port: Int) async throws {
        let app = Application(
            router: buildRouter(),
            configuration: .init(address: .hostname(host, port: port))
        )
        CLIStderr.write("Resident music API: http://\(host):\(port)/v1/audio/music\n")
        CLIStderr.write("Batch endpoint: http://\(host):\(port)/v1/audio/music/batches\n")
        try await app.runService()
    }

    func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()
        router.get("/health") { [self] request, _ in
            guard isAuthorized(request) else {
                return errorResponse(status: .unauthorized, message: "Invalid API key.")
            }
            return try jsonResponse(
                MusicAPIHealthResponse(
                    status: "ok",
                    model: modelID,
                    checkpointVariant: variant,
                    resident: true,
                    languageModelAvailable: languageModelAvailable,
                    languageModelSource: languageModelSource,
                    adapters: adapters
                )
            )
        }
        router.post("/v1/audio/music") { [self] request, _ in
            await handleGeneration(request)
        }
        router.post("/v1/audio/music/batches") { [self] request, _ in
            await handleBatch(request)
        }
        return router
    }

    private func handleGeneration(_ request: Request) async -> Response {
        guard isAuthorized(request) else {
            return errorResponse(status: .unauthorized, message: "Invalid API key.")
        }
        do {
            let payload = try await decode(
                MusicAPIGenerationRequest.self,
                from: request
            )
            let prepared = try prepare(payload)
            let ranked = try session.generateBest(
                prepared.request,
                candidateCount: prepared.candidateCount
            )
            if payload.responseFormat?.lowercased() == "wav" {
                let data = try ACEStepWAVWriter.wavData(
                    ranked.best.audio,
                    sampleRate: 48_000
                )
                return Response(
                    status: .ok,
                    headers: [.contentType: "audio/wav"],
                    body: .init(byteBuffer: ByteBuffer(bytes: data))
                )
            }
            return try jsonResponse(
                try response(for: ranked, prepared: prepared)
            )
        } catch {
            return errorResponse(
                status: .badRequest,
                message: MusicServe.apiErrorMessage(error)
            )
        }
    }

    private func handleBatch(_ request: Request) async -> Response {
        guard isAuthorized(request) else {
            return errorResponse(status: .unauthorized, message: "Invalid API key.")
        }
        do {
            let payload = try await decode(MusicAPIBatchRequest.self, from: request)
            guard (1...32).contains(payload.requests.count) else {
                throw ValidationError("Batch requests must contain 1...32 items.")
            }
            guard !payload.requests.contains(where: {
                $0.responseFormat?.lowercased() == "wav"
            }) else {
                throw ValidationError(
                    "Batch requests support response_format=json only."
                )
            }
            let prepared = try payload.requests.map(prepare)
            let ranked = try session.generateBatch(
                prepared.map(\.request),
                candidateCounts: prepared.map(\.candidateCount)
            )
            let responses = try zip(ranked, prepared).map {
                try response(for: $0.0, prepared: $0.1)
            }
            return try jsonResponse(MusicAPIBatchResponse(data: responses))
        } catch {
            return errorResponse(
                status: .badRequest,
                message: MusicServe.apiErrorMessage(error)
            )
        }
    }

    private func prepare(
        _ payload: MusicAPIGenerationRequest
    ) throws -> PreparedMusicAPIRequest {
        let prompt = payload.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw ValidationError("prompt must not be empty.")
        }
        if let requestedModel = payload.model,
           requestedModel != modelID
        {
            throw ValidationError(
                "Resident music API loaded '\(modelID)', not "
                    + "'\(requestedModel)'."
            )
        }
        let task = payload.task ?? .textToMusic
        try variant.validate(task)
        let quality = payload.quality ?? .song
        let defaults = quality.defaults(for: variant, task: task)
        let candidates = payload.candidates ?? defaults.candidateCount
        guard (1...16).contains(candidates) else {
            throw ValidationError("candidates must be between 1 and 16.")
        }
        let steps = payload.steps ?? defaults.inferenceSteps
        guard steps >= 1 else {
            throw ValidationError("steps must be at least 1.")
        }
        let shift = payload.shift ?? defaults.shift
        guard shift > 0 else {
            throw ValidationError("shift must be greater than 0.")
        }
        let guidanceScale = payload.guidanceScale ?? defaults.guidanceScale
        guard guidanceScale >= 1 else {
            throw ValidationError("guidance_scale must be at least 1.")
        }
        let cfgIntervalStart = payload.cfgIntervalStart ?? 0
        let cfgIntervalEnd = payload.cfgIntervalEnd ?? 1
        guard (0...1).contains(cfgIntervalStart),
              (0...1).contains(cfgIntervalEnd),
              cfgIntervalStart <= cfgIntervalEnd
        else {
            throw ValidationError(
                "CFG intervals require 0 <= cfg_interval_start "
                    + "<= cfg_interval_end <= 1."
            )
        }
        let velocityNormThreshold = payload.velocityNormThreshold
            ?? defaults.velocityNormThreshold
        guard velocityNormThreshold >= 0 else {
            throw ValidationError(
                "velocity_norm_threshold must be nonnegative."
            )
        }
        let velocityEMAFactor = payload.velocityEMAFactor
            ?? defaults.velocityEMAFactor
        guard (0..<1).contains(velocityEMAFactor) else {
            throw ValidationError(
                "velocity_ema_factor must be in [0, 1)."
            )
        }
        let audioCoverStrength = payload.audioCoverStrength ?? 1
        guard (0...1).contains(audioCoverStrength) else {
            throw ValidationError(
                "audio_cover_strength must be between 0 and 1."
            )
        }
        let coverNoiseStrength = payload.coverNoiseStrength ?? 0
        guard (0...1).contains(coverNoiseStrength) else {
            throw ValidationError(
                "cover_noise_strength must be between 0 and 1."
            )
        }
        let retakeVariance = payload.retakeVariance ?? 0
        guard (0...1).contains(retakeVariance) else {
            throw ValidationError(
                "retake_variance must be between 0 and 1."
            )
        }
        let repaintStart = payload.repaintStartSeconds ?? 0
        let repaintEnd = payload.repaintEndSeconds ?? -1
        guard repaintStart >= 0,
              repaintEnd == -1 || repaintEnd > repaintStart
        else {
            throw ValidationError(
                "repaint range requires a nonnegative start and an end "
                    + "greater than the start, or -1."
            )
        }
        let repaintStrength = payload.repaintStrength ?? 0.5
        guard (0...1).contains(repaintStrength) else {
            throw ValidationError(
                "repaint_strength must be between 0 and 1."
            )
        }
        let lmTopK = payload.lmTopK ?? 0
        let lmTopP = payload.lmTopP ?? 0.9
        let lmTemperature = payload.lmTemperature ?? 0.85
        let lmRepetitionPenalty = payload.lmRepetitionPenalty ?? 1.0
        let lmCFGScale = payload.lmCFGScale ?? 2.0
        let lmNegativePrompt = payload.lmNegativePrompt ?? "NO USER INPUT"
        guard lmTopK >= 0,
              (0...1).contains(lmTopP),
              (0...2).contains(lmTemperature),
              lmRepetitionPenalty > 0,
              lmCFGScale >= 1,
              lmCFGScale.isFinite
        else {
            throw ValidationError(
                "LM sampling requires lm_top_k >= 0, lm_top_p in [0, 1], "
                    + "lm_temperature in [0, 2], lm_repetition_penalty > 0, "
                    + "and lm_cfg_scale >= 1."
            )
        }
        let effectiveLyrics: String
        if payload.instrumental == true {
            guard Self.nonEmpty(payload.lyrics) == nil else {
                throw ValidationError(
                    "instrumental cannot be combined with lyrics."
                )
            }
            effectiveLyrics = "[Instrumental]"
        } else {
            effectiveLyrics = payload.lyrics ?? ""
        }
        let effectiveLMRepetitionPenalty = lmRepetitionPenalty == 1
            ? nil
            : lmRepetitionPenalty
        let vaeChunkSize = payload.vaeChunkSize ?? 512
        let vaeOverlap = payload.vaeOverlap ?? 64
        guard vaeChunkSize > 0, vaeOverlap >= 0 else {
            throw ValidationError(
                "VAE tiling requires vae_chunk_size > 0 and vae_overlap >= 0."
            )
        }
        let responseFormat = payload.responseFormat?.lowercased() ?? "json"
        guard responseFormat == "json" || responseFormat == "wav" else {
            throw ValidationError("response_format must be json or wav.")
        }
        let sourceAudio = try payload.sourceAudioPath.map {
            try ACEStepCLIHelper.loadAudio48kHz($0, label: "Source audio")
        }
        let referenceAudio = try (payload.referenceAudioPaths ?? []).map {
            try ACEStepCLIHelper.loadAudio48kHz(
                $0,
                label: "Reference audio"
            )
        }
        let isFlowEdit = payload.sourceCaption != nil
        if task.requiresSourceAudio, sourceAudio == nil {
            throw ValidationError("source_audio_path is required for \(task.rawValue).")
        }
        if isFlowEdit, sourceAudio == nil {
            throw ValidationError("source_audio_path is required with source_caption.")
        }
        var duration = task.locksDurationToSource || isFlowEdit
            ? sourceAudio.map {
                ACEStepCLIHelper.durationSeconds(
                    of: $0,
                    fallback: defaults.fallbackDurationSeconds
                )
            } ?? defaults.fallbackDurationSeconds
            : payload.durationSeconds ?? defaults.fallbackDurationSeconds
        guard (1...600).contains(duration) else {
            throw ValidationError("duration_seconds must be between 1 and 600.")
        }
        let useLM = !isFlowEdit
            && !task.skipsLanguageModel
            && languageModelAvailable
            && (payload.useLanguageModel ?? defaults.usesLanguageModel)
        let instruction = Self.nonEmpty(payload.instruction)
            ?? task.instruction(
                trackName: payload.trackName,
                completeTrackClasses: payload.completeTrackClasses ?? []
            )
        let shouldPlanDuration = useLM
            && payload.durationSeconds == nil
            && defaults.automaticDuration
            && !task.locksDurationToSource
            && !isFlowEdit
        var effectivePrompt = prompt
        var lmCodeGenerationContext: ACEStepLMCodeGenerationContext?
        let effectiveLanguage = ACEStepPlanningPolicy.effectiveLanguage(
            vocalLanguage: payload.vocalLanguage,
            metadataLanguage: payload.metadataLanguage
        )
        var metadata = ACEStep5HzLMConstrainedSampler.UserMetadata(
            bpm: payload.bpm.map(String.init),
            caption: prompt,
            duration: shouldPlanDuration
                ? nil
                : String(max(1, Int(duration.rounded()))),
            keyscale: Self.nonEmpty(payload.keyscale),
            language: effectiveLanguage,
            timesignature: Self.nonEmpty(payload.timeSignature)
        )
        if useLM {
            let plan = try session.planMusic(
                caption: prompt,
                lyrics: effectiveLyrics,
                instruction: instruction,
                userMetadata: .init(
                    bpm: metadata.bpm,
                    duration: shouldPlanDuration
                        ? nil
                        : metadata.duration,
                    keyscale: metadata.keyscale,
                    language: metadata.language,
                    timesignature: metadata.timesignature
                ),
                lmConfig: .init(
                    maxNewTokens: 1_024,
                    temperature: lmTemperature,
                    topK: lmTopK,
                    topP: lmTopP,
                    repetitionPenalty: effectiveLMRepetitionPenalty
                )
            )
            lmCodeGenerationContext = plan.codeGenerationContext
            effectivePrompt = Self.nonEmpty(plan.metadata.caption) ?? prompt
            if shouldPlanDuration,
               let plannedDuration = plan.metadata.durationSeconds
            {
                let upperBound: Float = quality == .song ? 240 : 600
                duration = min(max(plannedDuration, 10), upperBound)
            }
            metadata = ACEStepPlanningPolicy.merge(
                userMetadata: metadata,
                plan: plan.metadata,
                caption: effectivePrompt,
                durationSeconds: duration
            )
            lmCodeGenerationContext = lmCodeGenerationContext?.applying(
                userMetadata: metadata
            )
        }
        let config = ACEStepInferenceConfig(
            durationSeconds: duration,
            fixNFE: steps,
            shift: shift,
            coverNoiseStrength: coverNoiseStrength,
            retakeSeed: payload.retakeSeed,
            retakeVariance: retakeVariance,
            inferMethod: payload.inferMethod ?? .ode,
            samplerMode: payload.sampler ?? defaults.samplerMode,
            guidanceScale: guidanceScale,
            guidanceMode: payload.guidanceMode ?? .apg,
            cfgIntervalStart: cfgIntervalStart,
            cfgIntervalEnd: cfgIntervalEnd,
            velocityNormThreshold: velocityNormThreshold,
            velocityEMAFactor: velocityEMAFactor,
            useTiledVaeDecode: payload.useTiledVAEDecode ?? true,
            vaeChunkSize: vaeChunkSize,
            vaeOverlap: vaeOverlap,
            seed: payload.seed
        )
        let repaint = task == .repaint || task == .lego
            ? ACEStepRepaintConfiguration(
                startSeconds: repaintStart,
                endSeconds: repaintEnd,
                chunkMaskMode: payload.chunkMaskMode ?? .auto,
                mode: payload.repaintMode ?? .balanced,
                strength: repaintStrength
            )
            : nil
        let flowEdit = payload.sourceCaption.map {
            ACEStepFlowEditConfiguration(
                sourceCaption: $0,
                sourceLyrics: payload.sourceLyrics ?? "",
                nMin: payload.flowEditNMin ?? 0,
                nMax: payload.flowEditNMax ?? 1,
                nAverage: payload.flowEditNAverage ?? 1,
                retakeSeed: payload.retakeSeed
            )
        }
        try flowEdit?.validate()
        return PreparedMusicAPIRequest(
            request: ACEStepSessionRequest(
                caption: effectivePrompt,
                lyrics: effectiveLyrics,
                config: config,
                lmConfig: .init(
                    maxNewTokens: 4_096,
                    temperature: lmTemperature,
                    topK: lmTopK,
                    topP: lmTopP,
                    repetitionPenalty: effectiveLMRepetitionPenalty,
                    cfgScale: lmCFGScale,
                    negativePrompt: lmNegativePrompt
                ),
                lmUserMetadata: metadata,
                lmCodeGenerationContext: lmCodeGenerationContext,
                sourceAudio48kHz: sourceAudio,
                referenceTimbreAudio48kHz:
                    referenceAudio.isEmpty ? nil : referenceAudio,
                audioCoverStrength: audioCoverStrength,
                vocalLanguage: effectiveLanguage,
                instruction: instruction,
                task: task,
                repaintConfiguration: repaint,
                flowEditConfiguration: flowEdit,
                useLanguageModel: useLM
            ),
            quality: quality,
            task: task,
            candidateCount: candidates,
            conditioningMetadata: ACEStepRecipeConditioningMetadata(
                metadata
            )
        )
    }

    private static func nonEmpty(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(
            in: .whitespacesAndNewlines
        ) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }

    private func response(
        for ranked: ACEStepRankedGeneration,
        prepared: PreparedMusicAPIRequest
    ) throws -> MusicAPIGenerationResponse {
        MusicAPIGenerationResponse(
            model: modelID,
            quality: prepared.quality,
            task: prepared.task,
            conditioningMetadata: prepared.conditioningMetadata,
            candidates: try ranked.candidates.enumerated().map { rank, candidate in
                MusicAPICandidateResponse(
                    rank: rank + 1,
                    seed: candidate.seed,
                    score: candidate.score,
                    metrics: candidate.metrics,
                    audioBase64: try ACEStepWAVWriter.wavData(
                        candidate.audio,
                        sampleRate: 48_000
                    ).base64EncodedString(),
                    format: "wav",
                    sampleRate: 48_000,
                    selected: candidate.index == ranked.best.index
                )
            }
        )
    }

    private func decode<T: Decodable>(
        _ type: T.Type,
        from request: Request
    ) async throws -> T {
        guard request.headers[.contentType]?.lowercased().contains("application/json") == true else {
            throw ValidationError("Content-Type must be application/json.")
        }
        let body = try await request.body.collect(upTo: 4 * 1_024 * 1_024)
        return try JSONDecoder().decode(T.self, from: Data(body.readableBytesView))
    }

    private func isAuthorized(_ request: Request) -> Bool {
        guard let apiKey, !apiKey.isEmpty else {
            return true
        }
        return request.headers[.authorization] == "Bearer \(apiKey)"
    }

    private func jsonResponse<T: Encodable>(_ value: T) throws -> Response {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(value)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private func errorResponse(
        status: HTTPResponse.Status,
        message: String
    ) -> Response {
        let data = (try? JSONEncoder().encode(["error": message]))
            ?? Data("{\"error\":\"music API error\"}".utf8)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }
}
