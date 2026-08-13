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
    var numInferenceSteps: Int?
    var guidanceScale: Float?
    var sampleRate: Int?

    enum CodingKeys: String, CodingKey {
        case model
        case input
        case instructions
        case responseFormat = "response_format"
        case seed
        case maxNewTokens = "max_new_tokens"
        case stream
        case audioDuration = "audio_duration"
        case numInferenceSteps = "num_inference_steps"
        case guidanceScale = "guidance_scale"
        case sampleRate = "sample_rate"
    }
}

private struct MiniMaxMusic3HealthResponse: Codable {
    var status: String
    var model: String
    var resident: Bool
    var memoryMode: MiniMaxMusic3LoadingStrategy
    var nativeSampleRate: Int
    var speechSampleRate: Int

    enum CodingKeys: String, CodingKey {
        case status
        case model
        case resident
        case memoryMode = "memory_mode"
        case nativeSampleRate = "native_sample_rate"
        case speechSampleRate = "speech_sample_rate"
    }
}

private actor MiniMaxMusic3ServerSession {
    private let pipeline: MiniMaxMusic3Pipeline

    init(
        resources: MiniMaxMusic3Resources,
        loadingStrategy: MiniMaxMusic3LoadingStrategy
    ) throws {
        self.pipeline = try MiniMaxMusic3Pipeline(
            resources: resources,
            loadingStrategy: loadingStrategy
        )
    }

    func generate(
        _ request: MiniMaxMusic3SpeechRequest,
        modelID: String
    ) throws -> Data {
        let options = try Self.options(from: request, modelID: modelID)
        let result = try pipeline.generate(options: options.generation)
        let waveform = try ACEStepWAVWriter.resample(
            result.waveform.transposed(0, 2, 1),
            from: result.sampleRate,
            to: options.sampleRate
        )
        return try ACEStepWAVWriter.wavData(
            waveform,
            sampleRate: options.sampleRate,
            options: .init(
                format: .pcm16,
                normalization: .none,
                targetPeakDB: 0,
                fadeInMilliseconds: 0,
                fadeOutMilliseconds: 0,
                dither: false
            )
        )
    }

    private static func options(
        from request: MiniMaxMusic3SpeechRequest,
        modelID: String
    ) throws -> (generation: MiniMaxMusic3GenerationOptions, sampleRate: Int) {
        if let requestedModel = request.model,
           requestedModel != modelID,
           requestedModel != MiniMaxMusic3Resources.repository
        {
            throw ValidationError(
                "MiniMax Music 3 server loaded '\(modelID)', not '\(requestedModel)'."
            )
        }
        let caption = request.instructions.trimmingCharacters(in: .whitespacesAndNewlines)
        let lyrics = request.input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !caption.isEmpty else {
            throw ValidationError("instructions must contain a music description.")
        }
        guard !lyrics.isEmpty else {
            throw ValidationError("input must contain lyrics or [Instrumental].")
        }
        guard request.responseFormat?.lowercased() ?? "wav" == "wav" else {
            throw ValidationError("response_format must be wav.")
        }
        guard request.stream != true else {
            throw ValidationError("MiniMax Music 3 supports stream=false only.")
        }
        let duration = request.audioDuration ?? 60
        guard duration > 0, duration <= 360 else {
            throw ValidationError("audio_duration must be greater than 0 and at most 360 seconds.")
        }
        let durationFrames = Int(duration * Float(MiniMaxMusic3Prompt.frameRate))
        let maximumFrames = request.maxNewTokens ?? durationFrames
        guard (1...MiniMaxMusic3Prompt.maxAudioFrames).contains(maximumFrames) else {
            throw ValidationError("max_new_tokens must be between 1 and 9000.")
        }
        if request.audioDuration != nil,
           request.maxNewTokens != nil,
           durationFrames != maximumFrames
        {
            throw ValidationError(
                "audio_duration and max_new_tokens must describe the same 25 Hz frame limit."
            )
        }
        let steps = request.numInferenceSteps ?? 30
        guard steps > 0 else {
            throw ValidationError("num_inference_steps must be positive.")
        }
        let guidanceScale = request.guidanceScale ?? 1.7
        guard guidanceScale >= 1, guidanceScale.isFinite else {
            throw ValidationError("guidance_scale must be finite and at least 1.")
        }
        let sampleRate = request.sampleRate ?? 32_000
        guard sampleRate == 32_000 || sampleRate == 44_100 else {
            throw ValidationError("sample_rate must be 32000 or 44100.")
        }
        return (
            MiniMaxMusic3GenerationOptions(
                caption: caption,
                lyrics: lyrics,
                durationSeconds: duration,
                maximumFrames: maximumFrames,
                inferenceSteps: steps,
                seed: request.seed ?? 0,
                guidanceScale: guidanceScale
            ),
            sampleRate
        )
    }
}

final class MiniMaxMusic3APIServer: @unchecked Sendable {
    private let session: MiniMaxMusic3ServerSession
    private let modelID: String
    private let loadingStrategy: MiniMaxMusic3LoadingStrategy
    private let apiKey: String?

    init(
        resources: MiniMaxMusic3Resources,
        modelID: String,
        loadingStrategy: MiniMaxMusic3LoadingStrategy,
        apiKey: String?
    ) throws {
        self.session = try MiniMaxMusic3ServerSession(
            resources: resources,
            loadingStrategy: loadingStrategy
        )
        self.modelID = modelID
        self.loadingStrategy = loadingStrategy
        self.apiKey = apiKey
    }

    func run(host: String, port: Int) async throws {
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
            let wav = try await session.generate(payload, modelID: modelID)
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
