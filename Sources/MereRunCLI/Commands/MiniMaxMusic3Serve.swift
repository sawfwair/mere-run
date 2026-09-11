import AudioCore
import ArgumentParser
import Foundation
import Hummingbird
import MereRunCore
import NIOCore

struct MiniMaxMusic3SpeechRequest: Codable, Sendable {
    var model: String?
    var input: String
    var instructions: String
    var responseFormat: String?
    var seed: UInt64?
    var maxNewTokens: Int?
    var stream: Bool?
    var audioDuration: Float?
    var minimumAudioDuration: Float?
    var minNewTokens: Int?
    var samplingTier: MiniMaxMusic3SamplingTier?
    var flowStrategy: MiniMaxMusic3FlowStrategy?
    var flowSolver: MiniMaxMusic3FlowSolver?
    var autoregressiveGuidanceFrames: Int?
    var flowGuidanceEnd: Float?
    var seedStrategy: MiniMaxMusic3SeedStrategy?
    var lyricPreflight: MiniMaxMusic3LyricPreflightPolicy?
    var numInferenceSteps: Int?
    var guidanceScale: Float?
    var sampleRate: Int?
    var export: AudioExportOverrides?

    func generationSettings() -> MiniMaxMusic3GenerationSettings {
        var settings = MiniMaxMusic3GenerationSettings(caption: instructions, lyrics: input)
        settings.seed = seed
        settings.maxNewTokens = maxNewTokens
        settings.audioDuration = audioDuration
        settings.minimumAudioDuration = minimumAudioDuration
        settings.minNewTokens = minNewTokens
        settings.samplingTier = samplingTier
        settings.flowStrategy = flowStrategy
        settings.flowSolver = flowSolver
        settings.autoregressiveGuidanceFrames = autoregressiveGuidanceFrames
        settings.flowGuidanceEnd = flowGuidanceEnd
        settings.seedStrategy = seedStrategy
        settings.lyricPreflight = lyricPreflight
        settings.numInferenceSteps = numInferenceSteps
        settings.guidanceScale = guidanceScale
        settings.sampleRate = sampleRate
        return settings
    }

    enum CodingKeys: String, CodingKey {
        case model
        case input
        case instructions
        case responseFormat = "response_format"
        case seed
        case maxNewTokens = "max_new_tokens"
        case stream
        case audioDuration = "audio_duration"
        case minimumAudioDuration = "minimum_audio_duration"
        case minNewTokens = "min_new_tokens"
        case samplingTier = "sampling_tier"
        case flowStrategy = "flow_strategy"
        case flowSolver = "flow_solver"
        case autoregressiveGuidanceFrames = "autoregressive_guidance_frames"
        case flowGuidanceEnd = "flow_guidance_end"
        case seedStrategy = "seed_strategy"
        case lyricPreflight = "lyric_preflight"
        case numInferenceSteps = "num_inference_steps"
        case guidanceScale = "guidance_scale"
        case sampleRate = "sample_rate"
        case export
    }
}

private struct MiniMaxMusic3HealthResponse: Codable {
    var status: String
    var model: String
    var resident: Bool
    var memoryMode: MiniMaxMusic3LoadingStrategy
    var performanceMode: MiniMaxMusic3PerformanceMode
    var languageModelPrecision: MiniMaxMusic3WeightPrecision
    var depthDecoderPrecision: MiniMaxMusic3WeightPrecision
    var nativeSampleRate: Int
    var speechSampleRate: Int

    enum CodingKeys: String, CodingKey {
        case status
        case model
        case resident
        case memoryMode = "memory_mode"
        case performanceMode = "performance_mode"
        case languageModelPrecision = "language_model_precision"
        case depthDecoderPrecision = "depth_decoder_precision"
        case nativeSampleRate = "native_sample_rate"
        case speechSampleRate = "speech_sample_rate"
    }
}

final class MiniMaxMusic3APIServer: @unchecked Sendable {
    private let session: MiniMaxMusic3GenerationOperation
    private let modelID: String
    private let loadingStrategy: MiniMaxMusic3LoadingStrategy
    private let performanceMode: MiniMaxMusic3PerformanceMode
    private let apiKey: String?

    init(
        resources: MiniMaxMusic3Resources,
        modelID: String,
        loadingStrategy: MiniMaxMusic3LoadingStrategy,
        performanceMode: MiniMaxMusic3PerformanceMode,
        apiKey: String?
    ) throws {
        self.session = MiniMaxMusic3GenerationOperation(
            resources: resources,
            loadingStrategy: loadingStrategy,
            performanceMode: performanceMode
        )
        self.modelID = modelID
        self.loadingStrategy = loadingStrategy
        self.performanceMode = performanceMode
        self.apiKey = apiKey
    }

    func run(host: String, port: Int) async throws {
        try await session.load()
        let app = Application(
            router: buildRouter(),
            configuration: .init(address: .hostname(host, port: port))
        )
        CLIStderr.write(
            "MiniMax Music 3 speech API (\(loadingStrategy.rawValue)): "
                + "http://\(host):\(port)/v1/audio/speech\n"
        )
        try await app.runService()
    }

    func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()
        router.get("/health") { [self] request, _ in
            guard isAuthorized(request) else {
                return errorResponse(status: .unauthorized, message: "Invalid API key.")
            }
            return try jsonResponse(
                MiniMaxMusic3HealthResponse(
                    status: "ok",
                    model: modelID,
                    resident: loadingStrategy == .resident,
                    memoryMode: loadingStrategy,
                    performanceMode: performanceMode,
                    languageModelPrecision: performanceMode.languageModelPrecision,
                    depthDecoderPrecision: performanceMode.depthDecoderPrecision,
                    nativeSampleRate: 44_100,
                    speechSampleRate: 32_000
                )
            )
        }
        router.post("/v1/audio/speech") { [self] request, _ in
            await handleSpeech(request)
        }
        return router
    }

    private func handleSpeech(_ request: Request) async -> Response {
        guard isAuthorized(request) else {
            return errorResponse(status: .unauthorized, message: "Invalid API key.")
        }
        do {
            guard request.headers[.contentType]?.lowercased()
                .contains("application/json") == true
            else {
                throw ValidationError("Content-Type must be application/json.")
            }
            let body = try await request.body.collect(upTo: 4 * 1_024 * 1_024)
            let payload = try JSONDecoder().decode(
                MiniMaxMusic3SpeechRequest.self,
                from: Data(body.readableBytesView)
            )
            if let requestedModel = payload.model,
               requestedModel != modelID, requestedModel != MiniMaxMusic3Resources.repository {
                throw ValidationError("MiniMax Music 3 server loaded '\(modelID)', not '\(requestedModel)'.")
            }
            guard payload.responseFormat?.lowercased() ?? "wav" == "wav" else {
                throw ValidationError("response_format must be wav.")
            }
            guard payload.stream != true else {
                throw ValidationError("MiniMax Music 3 supports stream=false only.")
            }
            let plan = try payload.generationSettings().resolve(
                exportPlan: (payload.export ?? .init()).resolve(defaults: .referencePCM16), defaultSampleRate: 32_000
            )
            let result = try await session.generate(plan)
            let wav = try AudioExportService.data(result.waveform, plan: plan.export).data
            return Response(
                status: .ok,
                headers: [.contentType: "audio/wav"],
                body: .init(byteBuffer: ByteBuffer(bytes: wav))
            )
        } catch {
            return errorResponse(
                status: .badRequest,
                message: MusicServe.apiErrorMessage(error)
            )
        }
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
