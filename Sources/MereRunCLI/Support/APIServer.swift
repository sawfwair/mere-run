import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Hummingbird
import HTTPTypes
import NIOCore
import AudioCore
import AudioSTT
import AudioTTS
import MediaIO
import MereRunCore

extension APIServerContract {
    static let videoGenerationRouterPath = RouterPath(videoGenerationRoutePath)
    static let geometryRouterPath = RouterPath(geometryRoutePath)
    static let multiViewGeometryRouterPath = RouterPath(multiViewGeometryRoutePath)
    static let imageTo3DRouterPath = RouterPath(imageTo3DRoutePath)
    static let depthVideoRouterPath = RouterPath(depthVideoRoutePath)
}

struct APIServerRequestContext: RequestContext, RemoteAddressRequestContext {
    var coreContext: CoreRequestContextStorage
    let remoteAddress: SocketAddress?

    init(source: Source) {
        self.coreContext = CoreRequestContextStorage(source: source)
        self.remoteAddress = source.channel.remoteAddress
    }
}

enum APIVFXArtifactRoutePolicy {
    /// Successful responses expose server-local `file:` URLs for this long.
    static let outputTTLSeconds: UInt64 = 60 * 60
    static let denialMessage = "VFX artifact routes are loopback-only because responses contain server-local file URLs; authenticated remote clients cannot use this route."

    static var routePaths: Set<String> {
        [
            APIServerContract.geometryRoutePath,
            APIServerContract.videoGenerationRoutePath,
            APIServerContract.multiViewGeometryRoutePath,
            APIServerContract.imageTo3DRoutePath,
            APIServerContract.instantMeshRoutePath,
            APIServerContract.depthVideoRoutePath,
        ]
    }

    static var modelIDs: Set<String> {
        [
            APIServerContract.defaultGeometryModelID,
            APIServerContract.defaultMultiViewGeometryModelID,
            APIServerContract.defaultImageTo3DModelID,
            APIServerContract.defaultInstantMeshModelID,
            ModelResolver.ModelID.visionDepthVDASmall.rawValue,
            ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue,
            ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue,
            ModelResolver.ModelID.ltxVideo25FullBF16.rawValue,
        ]
    }

    /// Trust only the connected peer socket. Proxy headers are intentionally
    /// ignored because this server has no trusted-proxy configuration.
    static func allows(remoteAddress: SocketAddress?) -> Bool {
        guard let remoteAddress else { return false }
        switch remoteAddress {
        case .v4:
            guard let address = remoteAddress.ipAddress else { return false }
            return address.split(separator: ".", omittingEmptySubsequences: false).first == "127"
        case .v6:
            guard let address = remoteAddress.ipAddress?.lowercased() else { return false }
            return address == "::1" || address.hasPrefix("::ffff:127.")
        case .unixDomainSocket:
            return true
        }
    }
}

enum APIVFXClientErrorPolicy {
    static func status(for error: Error) -> HTTPResponse.Status? {
        switch error {
        case VideoDepthAnythingGeneratorError.inputVideoNotFound:
            return .badRequest
        case is InstantMeshGeneratorError,
             is TripoSRGeneratorError,
             is MoGe2TokenGridError,
             is MoGe2GenerationError,
             is VideoDepthAnythingLimitError,
             is VideoGenerationError,
             is VideoGenerationIssue,
             is DepthAnything3LimitError,
             is MediaIOError,
             is VFXImageInputValidationError,
             is DepthAnything3CameraValidationError,
             is DepthAnything3CameraConditioningError,
             is DepthAnything3PreprocessingError,
             is VideoDepthAnythingPreprocessingError,
             is VideoDepthAnythingWindowingError,
             is MultiViewGeometryExportConfigurationError,
             is TripoSRPreprocessingError,
             is InstantMeshPreprocessingError:
            return .badRequest
        default:
            return nil
        }
    }
}

struct APIArtifactDirectoryCleanupScheduler: Sendable {
    static let productionDelayNanoseconds = APIVFXArtifactRoutePolicy.outputTTLSeconds * 1_000_000_000

    let delayNanoseconds: UInt64
    private let sleeper: @Sendable (UInt64) async throws -> Void
    private let removeDirectory: @Sendable (URL) -> Void

    init(
        delayNanoseconds: UInt64 = productionDelayNanoseconds,
        sleeper: @escaping @Sendable (UInt64) async throws -> Void = {
            try await Task.sleep(nanoseconds: $0)
        },
        removeDirectory: @escaping @Sendable (URL) -> Void = {
            try? FileManager.default.removeItem(at: $0)
        }
    ) {
        self.delayNanoseconds = delayNanoseconds
        self.sleeper = sleeper
        self.removeDirectory = removeDirectory
    }

    func scheduleCleanup(of directory: URL) {
        let delayNanoseconds = delayNanoseconds
        let sleeper = sleeper
        let removeDirectory = removeDirectory
        Task.detached(priority: .utility) {
            do {
                try await sleeper(delayNanoseconds)
            } catch {
                return
            }
            removeDirectory(directory)
        }
    }
}

func withVFXRequestAdmission<T: Sendable>(
    using admission: RuntimeRequestAdmission,
    isolation _: isolated (any Actor)? = #isolation,
    operation: () async throws -> T
) async throws -> T {
    try await withRuntimeRequestAdmission(using: admission, operation: operation)
}

// MARK: - Server Implementation

actor CodeGenServer {
    private let apiKey: String?
    private let fallbackLoraPath: String?
    private let defaultModelID: String
    private let contextSize: Int
    private let requestLimiter: APIRateLimiter
    private let services: RuntimeServingServices
    private var requestAdmission: RuntimeRequestAdmission { services.admission }
    private var pool: RuntimeModelPool { services.models }
    private var sidecarPool: APISidecarModelPool { services.media }
    private let artifactCleanupScheduler: APIArtifactDirectoryCleanupScheduler
    private let imageRunRecords: URL?
    private let transcriptionRunRecords: URL?
    private var processTelemetrySampler = RuntimeProcessTelemetrySampler()

    init(
        defaultModelID: String,
        modelPath: String?,
        fallbackLoraPath: String?,
        apiKey: String?,
        rateLimitPerMinute: Int,
        maxActiveRequests: Int = 1,
        engine: APIEngine,
        contextSize: Int = 32768,
        gemma4KVCacheQuantization: Gemma4KVCacheQuantization = Gemma4KVCacheQuantization(),
        memoryPressurePolicy: RuntimeMemoryPressurePolicy = .default,
        artifactCleanupScheduler: APIArtifactDirectoryCleanupScheduler = APIArtifactDirectoryCleanupScheduler(),
        imageRunRecords: URL? = nil,
        transcriptionRunRecords: URL? = nil,
        warmupDefaultModel: Bool = true
    ) async throws {
        self.imageRunRecords = imageRunRecords
        self.transcriptionRunRecords = transcriptionRunRecords
        self.apiKey = apiKey
        self.contextSize = contextSize
        self.fallbackLoraPath = fallbackLoraPath
        self.defaultModelID = defaultModelID
        self.requestLimiter = APIRateLimiter(limitPerMinute: rateLimitPerMinute)
        self.artifactCleanupScheduler = artifactCleanupScheduler
        self.services = RuntimeServingServices(
            defaultModelID: defaultModelID, modelPath: modelPath, engine: engine.runtimeServingEngine,
            maxActiveRequests: maxActiveRequests, gemma4KVCacheQuantization: gemma4KVCacheQuantization,
            memoryPressurePolicy: memoryPressurePolicy
        )
        try await services.prepare(warmupDefaultModel: warmupDefaultModel)
    }

    func run(host: String, port: Int) async throws {
        let router = buildRouter()
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )

        print("Starting server at http://\(host):\(port)")
        print("OpenAI-compatible base URL: http://\(host):\(port)/v1")
        print("Chat endpoint: http://\(host):\(port)/v1/chat/completions")
        print("Embeddings endpoint: http://\(host):\(port)/v1/embeddings")
        print("Images endpoint: http://\(host):\(port)/v1/images/generations")
        print("Image edits endpoint: http://\(host):\(port)/v1/images/edits")
        print("Video endpoint: http://\(host):\(port)\(APIServerContract.videoGenerationRoutePath)")
        print("VFX artifact endpoints are loopback-only; local file URLs expire after one hour.")
        print("Geometry endpoint: http://\(host):\(port)/v1/vision/geometry")
        print("Multi-view geometry endpoint: http://\(host):\(port)\(APIServerContract.multiViewGeometryRoutePath)")
        print("Image-to-3D endpoint: http://\(host):\(port)\(APIServerContract.imageTo3DRoutePath)")
        print("Multi-view image-to-3D endpoint: http://\(host):\(port)\(APIServerContract.instantMeshRoutePath)")
        print("Video depth endpoint: http://\(host):\(port)\(APIServerContract.depthVideoRoutePath)")
        print("Speech endpoint: http://\(host):\(port)/v1/audio/speech")
        print("Transcriptions endpoint: http://\(host):\(port)/v1/audio/transcriptions")
        print("Press Ctrl+C to stop.")

        try await app.runService()
    }

    nonisolated func buildRouter() -> Router<APIServerRequestContext> {
        let router = Router(context: APIServerRequestContext.self)

        // Health check
        router.get("/health") { _, _ in
            let data = try JSONEncoder().encode(APIServerContract.healthStatus())
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(bytes: data))
            )
        }

        // List models
        router.get("/v1/models") { [self] request, context in
            return try await self.handleModels(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        // Chat completions
        router.post("/v1/chat/completions") { [self] request, _ in
            return try await self.handleChatCompletions(request)
        }

        // Embeddings
        router.post("/v1/embeddings") { [self] request, _ in
            return try await self.handleEmbeddings(request)
        }

        router.post("/v1/images/generations") { [self] request, _ in
            return try await self.handleImageGenerations(request)
        }

        router.post("/v1/images/edits") { [self] request, _ in
            return try await self.handleImageEdits(request)
        }

        router.post(APIServerContract.videoGenerationRouterPath) { [self] request, context in
            return try await self.handleVideoGenerations(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        router.post(APIServerContract.geometryRouterPath) { [self] request, context in
            return try await self.handleVisionGeometry(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        router.post(APIServerContract.multiViewGeometryRouterPath) { [self] request, context in
            return try await self.handleVisionMultiViewGeometry(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        router.post(APIServerContract.imageTo3DRouterPath) { [self] request, context in
            return try await self.handleVisionImageTo3D(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        router.post(APIServerContract.instantMeshRouterPath) { [self] request, context in
            return try await self.handleVisionInstantMesh(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        router.post(APIServerContract.depthVideoRouterPath) { [self] request, context in
            return try await self.handleVisionDepthVideo(
                request,
                remoteAddress: context.remoteAddress
            )
        }

        router.post("/v1/audio/speech") { [self] request, _ in
            return try await self.handleAudioSpeech(request)
        }

        router.post("/v1/audio/transcriptions") { [self] request, _ in
            return try await self.handleAudioTranscriptions(request)
        }

        router.get("/runtime/status") { [self] request, _ in
            return try await self.handleRuntimeStatus(request)
        }

        router.post("/runtime/models/:id/load") { [self] request, context in
            guard let id = context.parameters.get("id", as: String.self) else {
                return self.makeErrorResponse(
                    status: .badRequest,
                    message: "Missing model id.",
                    type: "invalid_request_error"
                )
            }
            return try await self.handleRuntimeLoad(request, id: id)
        }

        router.post("/runtime/models/:id/unload") { [self] request, context in
            guard let id = context.parameters.get("id", as: String.self) else {
                return self.makeErrorResponse(
                    status: .badRequest,
                    message: "Missing model id.",
                    type: "invalid_request_error"
                )
            }
            return try await self.handleRuntimeUnload(request, id: id)
        }

        router.get("/runtime/models/:id/settings") { [self] request, context in
            guard let id = context.parameters.get("id", as: String.self) else {
                return self.makeErrorResponse(
                    status: .badRequest,
                    message: "Missing model id.",
                    type: "invalid_request_error"
                )
            }
            return try await self.handleRuntimeSettings(request, id: id)
        }

        router.patch("/runtime/models/:id/settings") { [self] request, context in
            guard let id = context.parameters.get("id", as: String.self) else {
                return self.makeErrorResponse(
                    status: .badRequest,
                    message: "Missing model id.",
                    type: "invalid_request_error"
                )
            }
            return try await self.handleRuntimeSettingsPatch(request, id: id)
        }

        return router
    }

    private func handleModels(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let includeLoopbackArtifactModels = APIVFXArtifactRoutePolicy.allows(
            remoteAddress: remoteAddress
        )
        let inventory = ModelInventory.snapshot(mode: .fast)
        var installedModelIDs = inventory.installedModelIDs
        if QwenImageEditRepository.resolveInstalledModelRoot() != nil {
            installedModelIDs.insert(QwenImageEditRepository.modelId)
        }
        var models = try await pool.modelsResponse(
            installedModelIDs: installedModelIDs,
            serverContextSize: contextSize
        )
        if !includeLoopbackArtifactModels {
            models.data.removeAll { APIVFXArtifactRoutePolicy.modelIDs.contains($0.id) }
        }
        for modelID in APIServerContract.companionModelIDs(
            installedModelIDs: installedModelIDs,
            includeLoopbackArtifactModels: includeLoopbackArtifactModels
        )
            where !models.data.contains(where: { $0.id == modelID }) {
            guard let profile = ManagedModelCatalog.apiProfile(for: modelID) else {
                continue
            }
            models.data.append(
                APIServerContract.companionModel(
                    id: modelID,
                    profile: profile
                )
            )
        }
        models.data.sort { $0.id < $1.id }

        let data = try JSONEncoder().encode(models)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private func handleChatCompletions(_ incomingRequest: Request) async throws -> Response {
        var request = incomingRequest
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        // Decode request body
        let body: ByteBuffer
        do {
            body = try await request.collectBody(upTo: 10 * 1024 * 1024) // 10MB limit
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        let openaiRequest: OpenAIChatRequest
        do {
            openaiRequest = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(body.readableBytesView))
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request payload.", type: "invalid_request_error")
        }

        let session: RuntimeChatSession
        do {
            session = try await services.startChat(
                openaiRequest, fallbackLoraPath: fallbackLoraPath, contextSize: contextSize
            )
        } catch {
            return runtimeErrorResponse(error)
        }

        do {
            if session.engine == .textChatDeepseekV4Flash {
                return try await proxyDeepseekV4FlashChatCompletions(
                    body: body, contentType: request.headers[.contentType], session: session
                )
            }
            if openaiRequest.stream == true {
                return try await handleStreamingChat(session)
            }
            return handleNonStreamingChat(session)
        } catch let error as APIRequestValidationError {
            await session.finish(cancelled: Task.isCancelled)
            return makeErrorResponse(
                status: .badRequest, message: error.localizedDescription, type: "invalid_request_error"
            )
        } catch {
            await session.finish(cancelled: Task.isCancelled || error is CancellationError)
            return makeErrorResponse(status: .internalServerError, message: "Request failed.", type: "server_error")
        }
    }

    private func handleEmbeddings(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        let openaiRequest: OpenAIEmbeddingRequest
        do {
            openaiRequest = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: Data(body.readableBytesView))
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request payload.", type: "invalid_request_error")
        }

        do {
            return try await withRuntimeRequestAdmission(using: requestAdmission) {
                let texts = try APIServerContract.embeddingTexts(from: openaiRequest)
                let resolved = try await embeddingModel(for: openaiRequest.model)
                let result = try await sidecarPool.embed(
                    modelID: resolved.modelID,
                    modelPath: resolved.modelPath,
                    texts: texts
                )
                let response = APIServerContract.embeddingResponse(
                    modelId: resolved.modelID,
                    embeddings: result.embeddings,
                    tokenCounts: result.tokenCounts
                )
                let encoded = try jsonResponse(response)
                return encoded
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleImageGenerations(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        do {
            let openaiRequest = try APIServerContract.decodeImageGenerationRequest(
                from: Data(body.readableBytesView)
            )
            let plan = try APIServerContract.imageGenerationPlan(from: openaiRequest)
            return try await withRuntimeRequestAdmission(using: requestAdmission) {
                let outputURL = try await generateImage(plan)
                let response = try APIServerContract.imageResponse(outputURL: outputURL, plan: plan)
                let encoded = try jsonResponse(response)
                return encoded
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleVideoGenerations(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let denied = vfxArtifactAccessResponseIfNeeded(
            for: request,
            remoteAddress: remoteAddress
        ) {
            return denied
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 10 * 1_024 * 1_024)
        } catch {
            return makeErrorResponse(
                status: .badRequest,
                message: "Invalid request body.",
                type: "invalid_request_error"
            )
        }

        do {
            let request = try APIServerContract.decodeVideoGenerationRequest(
                from: Data(body.readableBytesView)
            )
            let plan = try APIServerContract.videoGenerationPlan(from: request)
            return try await withVFXRequestAdmission(using: requestAdmission) {
                let outputDirectory = try temporaryOutputDirectory(
                    directoryName: "mere-run-api-video"
                )
                let outputURL = outputDirectory.appendingPathComponent("output.mp4")
                do {
                    _ = try await APIVideoGeneration.generate(plan, outputURL: outputURL)
                    return try retainedArtifactJSONResponse(
                        APIServerContract.videoGenerationResponse(
                            outputURL: outputURL,
                            plan: plan
                        ),
                        outputDirectory: outputDirectory
                    )
                } catch {
                    try? FileManager.default.removeItem(at: outputDirectory)
                    throw error
                }
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleImageEdits(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 100 * 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        do {
            let form = try MultipartFormData.parse(
                body: Data(body.readableBytesView),
                boundary: boundary
            )
            let imageFiles = (form.files(named: "image") + form.files(named: "image[]"))
                .filter { !$0.body.isEmpty }
            guard !imageFiles.isEmpty else {
                throw APIRequestValidationError.invalidField("image", "image file is required")
            }
            let inputImageURLs = try imageFiles.map {
                try writeMultipartFile($0, directoryName: "mere-run-api-image-edits")
            }
            let maskImageURL = try form.file(named: "mask").flatMap { mask -> URL? in
                guard !mask.body.isEmpty else { return nil }
                return try writeMultipartFile(mask, directoryName: "mere-run-api-image-edits")
            }
            let plan = try APIServerContract.imageEditPlan(
                from: form,
                inputImageURLs: inputImageURLs,
                maskImageURL: maskImageURL
            )
            return try await withRuntimeRequestAdmission(using: requestAdmission) {
                let outputURL = try await generateImage(plan)
                let response = try APIServerContract.imageResponse(outputURL: outputURL, plan: plan)
                let encoded = try jsonResponse(response)
                return encoded
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleVisionGeometry(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let rejection = vfxArtifactAccessResponseIfNeeded(
            for: request,
            remoteAddress: remoteAddress
        ) {
            return rejection
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        do {
            return try await withVFXRequestAdmission(using: requestAdmission) {
                let body: ByteBuffer
                do {
                    body = try await request.body.collect(upTo: 100 * 1024 * 1024)
                } catch {
                    return makeErrorResponse(
                        status: .badRequest,
                        message: "Invalid request body.",
                        type: "invalid_request_error"
                    )
                }
                let form = try MultipartFormData.parse(
                    body: Data(body.readableBytesView),
                    boundary: boundary
                )
                guard let image = form.file(named: "image"), !image.body.isEmpty else {
                    throw APIRequestValidationError.invalidField("image", "image file is required")
                }
                if let contentType = image.contentType?.lowercased(),
                    !contentType.hasPrefix("image/") && contentType != "application/octet-stream"
                {
                    throw APIRequestValidationError.invalidField("image", "uploaded part must be an image")
                }
                let plan = try APIServerContract.geometryPlan(from: form)
                let inputURL = try writeMultipartFile(
                    image,
                    directoryName: "mere-run-api-geometry-inputs",
                    defaultExtension: "png"
                )
                defer { try? FileManager.default.removeItem(at: inputURL) }
                let outputDirectory = try temporaryOutputDirectory(directoryName: "mere-run-api-geometry")
                do {
                    let result = try await MoGe2GenerationOperation.execute(
                        plan.request(imageURL: inputURL, outputDirectory: outputDirectory),
                        prepareRuntime: { try MLXBundleSupport.ensureAvailable(quiet: true) }
                    )
                    return try retainedArtifactJSONResponse(
                        APIServerContract.geometryResponse(from: result),
                        outputDirectory: outputDirectory
                    )
                } catch {
                    try? FileManager.default.removeItem(at: outputDirectory)
                    throw error
                }
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleVisionMultiViewGeometry(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let rejection = vfxArtifactAccessResponseIfNeeded(
            for: request,
            remoteAddress: remoteAddress
        ) {
            return rejection
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        do {
            return try await withVFXRequestAdmission(using: requestAdmission) {
                let body: ByteBuffer
                do {
                    body = try await request.body.collect(
                        upTo: APIServerContract.maximumMultiViewGeometryUploadByteCount
                    )
                } catch {
                    return makeErrorResponse(
                        status: .badRequest,
                        message: "Invalid request body or multi-view upload exceeds 512 MiB.",
                        type: "invalid_request_error"
                    )
                }
                let form = try MultipartFormData.parse(
                    body: Data(body.readableBytesView),
                    boundary: boundary
                )
                let plan = try APIServerContract.multiViewGeometryPlan(from: form)
                let imageParts = form.parts.filter {
                    $0.filename != nil && ($0.name == "image" || $0.name == "image[]")
                }
                let inputURLs = try imageParts.map {
                    try writeMultipartFile(
                        $0,
                        directoryName: "mere-run-api-geometry-multiview-inputs",
                        defaultExtension: "png"
                    )
                }
                defer {
                    for url in inputURLs { try? FileManager.default.removeItem(at: url) }
                }
                let outputDirectory = try temporaryOutputDirectory(
                    directoryName: "mere-run-api-geometry-multiview"
                )
                try MLXBundleSupport.ensureAvailable(quiet: true)
                let generator = DepthAnything3Generator()
                do {
                    let result = try await generator.generate(
                        imageURLs: inputURLs,
                        model: plan.modelID,
                        knownCameras: plan.knownCameras,
                        referenceViewStrategy: plan.referenceViewStrategy,
                        processResolution: plan.processResolution,
                        progress: nil
                    )
                    let exportStart = Date()
                    let export = try MultiViewGeometryExporter.export(
                        run: result,
                        outputDirectory: outputDirectory,
                        configuration: try MultiViewGeometryExportConfiguration(
                            confidencePercentile: plan.confidencePercentile,
                            maximumPointCount: plan.maximumPointCount
                        )
                    )
                    let exportSeconds = Date().timeIntervalSince(exportStart)
                    await generator.unload()
                    return try retainedArtifactJSONResponse(
                        APIServerContract.multiViewGeometryResponse(
                            from: result,
                            export: export,
                            exportSeconds: exportSeconds
                        ),
                        outputDirectory: outputDirectory
                    )
                } catch {
                    await generator.unload()
                    try? FileManager.default.removeItem(at: outputDirectory)
                    throw error
                }
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleVisionImageTo3D(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let rejection = vfxArtifactAccessResponseIfNeeded(
            for: request,
            remoteAddress: remoteAddress
        ) {
            return rejection
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        do {
            return try await withVFXRequestAdmission(using: requestAdmission) {
                let body: ByteBuffer
                do {
                    body = try await request.body.collect(
                        upTo: APIServerContract.maximumImageTo3DUploadByteCount
                    )
                } catch {
                    return makeErrorResponse(
                        status: .badRequest,
                        message: "Invalid request body or image upload exceeds 100 MiB.",
                        type: "invalid_request_error"
                    )
                }
                let form = try MultipartFormData.parse(
                    body: Data(body.readableBytesView),
                    boundary: boundary
                )
                let plan = try APIServerContract.imageTo3DPlan(from: form)
                guard let image = form.file(named: "image") else {
                    throw APIRequestValidationError.invalidField(
                        "image",
                        "exactly one non-empty uploaded image file is required"
                    )
                }
                let inputURL = try writeMultipartFile(
                    image,
                    directoryName: "mere-run-api-image-to-3d-inputs",
                    defaultExtension: "png"
                )
                defer { try? FileManager.default.removeItem(at: inputURL) }
                let outputDirectory = try temporaryOutputDirectory(
                    directoryName: "mere-run-api-image-to-3d"
                )
                do {
                    let result = try await TripoSRGenerationOperation.execute(
                        plan.request(imageURL: inputURL, outputDirectory: outputDirectory),
                        prepareRuntime: { try MLXBundleSupport.ensureAvailable(quiet: true) }
                    )
                    return try retainedArtifactJSONResponse(
                        APIServerContract.imageTo3DResponse(from: result),
                        outputDirectory: outputDirectory
                    )
                } catch {
                    try? FileManager.default.removeItem(at: outputDirectory)
                    throw error
                }
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleVisionInstantMesh(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let rejection = vfxArtifactAccessResponseIfNeeded(
            for: request,
            remoteAddress: remoteAddress
        ) {
            return rejection
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        do {
            return try await withVFXRequestAdmission(using: requestAdmission) {
                let body: ByteBuffer
                do {
                    body = try await request.body.collect(
                        upTo: APIServerContract.maximumInstantMeshUploadByteCount
                    )
                } catch {
                    return makeErrorResponse(
                        status: .badRequest,
                        message: "Invalid request body or multi-view upload exceeds 512 MiB.",
                        type: "invalid_request_error"
                    )
                }
                let form = try MultipartFormData.parse(
                    body: Data(body.readableBytesView),
                    boundary: boundary
                )
                let plan = try APIServerContract.instantMeshPlan(from: form)
                let uploads = form.parts
                    .filter { $0.filename != nil && ($0.name == "image" || $0.name == "image[]") }
                    .filter { !$0.body.isEmpty }
                let inputURLs = try uploads.map {
                    try writeMultipartFile(
                        $0,
                        directoryName: "mere-run-api-instantmesh-inputs",
                        defaultExtension: "png"
                    )
                }
                defer {
                    for inputURL in inputURLs {
                        try? FileManager.default.removeItem(at: inputURL)
                    }
                }
                let outputDirectory = try temporaryOutputDirectory(
                    directoryName: "mere-run-api-instantmesh"
                )
                do {
                    let result = try await InstantMeshGenerationOperation.execute(
                        plan.request(viewURLs: inputURLs, outputDirectory: outputDirectory),
                        prepareRuntime: { try MLXBundleSupport.ensureAvailable(quiet: true) }
                    )
                    return try retainedArtifactJSONResponse(
                        APIServerContract.instantMeshResponse(from: result),
                        outputDirectory: outputDirectory
                    )
                } catch {
                    try? FileManager.default.removeItem(at: outputDirectory)
                    throw error
                }
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleVisionDepthVideo(
        _ request: Request,
        remoteAddress: SocketAddress?
    ) async throws -> Response {
        if let rejection = vfxArtifactAccessResponseIfNeeded(
            for: request,
            remoteAddress: remoteAddress
        ) {
            return rejection
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        do {
            return try await withVFXRequestAdmission(using: requestAdmission) {
                let body: ByteBuffer
                do {
                    body = try await request.body.collect(
                        upTo: APIServerContract.maximumDepthVideoUploadByteCount
                    )
                } catch {
                    return makeErrorResponse(
                        status: .badRequest,
                        message: "Invalid request body or video upload exceeds 512 MiB.",
                        type: "invalid_request_error"
                    )
                }
                let form = try MultipartFormData.parse(
                    body: Data(body.readableBytesView),
                    boundary: boundary
                )
                let plan = try APIServerContract.depthVideoPlan(from: form)
                guard let video = form.file(named: "video") else {
                    throw APIRequestValidationError.invalidField(
                        "video",
                        "exactly one non-empty uploaded video file is required"
                    )
                }
                let inputURL = try writeMultipartFile(
                    video,
                    directoryName: "mere-run-api-depth-video-inputs",
                    defaultExtension: "mp4"
                )
                defer { try? FileManager.default.removeItem(at: inputURL) }
                let outputDirectory = try temporaryOutputDirectory(
                    directoryName: "mere-run-api-depth-video"
                )
                do {
                    let result = try await VideoDepthAnythingGenerationOperation.execute(
                        plan.request(videoURL: inputURL, outputDirectory: outputDirectory),
                        prepareRuntime: { try MLXBundleSupport.ensureAvailable(quiet: true) }
                    )
                    return try retainedArtifactJSONResponse(
                        APIServerContract.depthVideoResponse(from: result),
                        outputDirectory: outputDirectory
                    )
                } catch {
                    try? FileManager.default.removeItem(at: outputDirectory)
                    throw error
                }
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleAudioSpeech(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 10 * 1024 * 1024)
        } catch {
            return makeErrorResponse(
                status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        do {
            let openaiRequest = try APIServerContract.decodeSpeechRequest(
                from: Data(body.readableBytesView)
            )
            let plan = try APIServerContract.speechPlan(from: openaiRequest)
            return try await withRuntimeRequestAdmission(using: requestAdmission) {
                let outputURL = try await synthesizeSpeech(plan)
                defer { try? FileManager.default.removeItem(at: outputURL) }
                let responseURL = try speechResponseURL(outputURL, responseFormat: plan.responseFormat)
                defer {
                    if responseURL != outputURL { try? FileManager.default.removeItem(at: responseURL) }
                }
                let data = try Data(contentsOf: responseURL)
                let response = binaryResponse(
                    data,
                    contentType: APIServerContract.speechContentType(for: plan.responseFormat)
                )
                return response
            }
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleAudioTranscriptions(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType])
        guard boundary != nil else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data.",
                type: "invalid_request_error"
            )
        }
        guard await requestLimiter.allowRequest() else {
            return makeErrorResponse(
                status: .tooManyRequests,
                message: "Rate limit exceeded.",
                type: "rate_limit_error"
            )
        }

        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 100 * 1024 * 1024)
        } catch {
            return makeErrorResponse(
                status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }

        do {
            let form = try MultipartFormData.parse(body: Data(body.readableBytesView), boundary: boundary)
            let plan = try APIServerContract.transcriptionPlan(from: form)
            guard let file = form.file(named: "file"), !file.body.isEmpty else {
                throw APIRequestValidationError.invalidField("file", "audio file is required")
            }
            let audioURL = try writeMultipartFile(file, directoryName: "mere-run-api-audio")
            defer { try? FileManager.default.removeItem(at: audioURL) }
            let result = try await transcribeAudio(audioURL: audioURL, plan: plan)
            let response: Response
            switch plan.responseFormat {
            case "text":
                response = binaryResponse(Data(result.text.utf8), contentType: "text/plain; charset=utf-8")
            case "srt", "vtt":
                let subtitle = APIServerContract.transcriptionSubtitle(from: result, format: plan.responseFormat)
                let contentType = plan.responseFormat == "srt"
                    ? "application/x-subrip; charset=utf-8"
                    : "text/vtt; charset=utf-8"
                response = binaryResponse(Data(subtitle.utf8), contentType: contentType)
            case "verbose_json":
                response = try jsonResponse(
                    APIServerContract.transcriptionResponse(from: result, verbose: true)
                )
            default:
                response = try jsonResponse(
                    APIServerContract.transcriptionResponse(from: result, verbose: false)
                )
            }
            return response
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleNonStreamingChat(_ session: RuntimeChatSession) -> Response {
        let request = session.request
        let modelID = session.modelID
        let (stream, continuation) = AsyncStream<NonStreamingChatEvent>.makeStream()
        let heartbeatTask = Task<Void, Never> {
            do {
                while !Task.isCancelled {
                    try await Task.sleep(for: .milliseconds(250))
                    guard !Task.isCancelled else { break }
                    continuation.yield(.heartbeat)
                }
            } catch {
                // Cancellation ends the heartbeat loop.
            }
        }
        let generationTask = Task {
            do {
                let result = try await session.chat()
                let responseToolCalls = openAIToolCalls(
                    from: result.toolCalls,
                    tools: request.tools,
                    parallelToolCalls: request.parallelToolCalls
                )
                let hasToolCalls = responseToolCalls?.isEmpty == false
                let response = OpenAIChatResponse(
                    id: "chatcmpl-\(UUID().uuidString.prefix(8))",
                    object: "chat.completion",
                    created: Int(Date().timeIntervalSince1970),
                    model: modelID,
                    choices: [
                        OpenAIChatChoice(
                            index: 0,
                            message: OpenAIChatMessage(
                                role: "assistant",
                                content: Self.openAIMessageContent(
                                    for: result,
                                    hasToolCalls: hasToolCalls
                                ),
                                reasoning_content: result.reasoningContent,
                                tool_calls: responseToolCalls
                            ),
                            finish_reason: Self.openAIFinishReason(
                                for: result,
                                hasToolCalls: hasToolCalls
                            ),
                            logprobs: OpenAIChatLogprobs(result.logprobs)
                        )
                    ],
                    usage: Self.openAIUsage(for: result),
                    mere_diffusion: result.diffusion
                )
                let data = try JSONEncoder().encode(response)
                var trailers = Self.openAITimingHeaders(for: result)
                trailers["x-mere-runtime-status"] = "200"
                await session.finish()
                heartbeatTask.cancel()
                continuation.yield(.completion(NonStreamingChatPayload(
                    data: data,
                    trailers: trailers
                )))
                continuation.finish()
            } catch {
                await session.finish(
                    cancelled: Task.isCancelled || error is CancellationError
                )
                heartbeatTask.cancel()
                if !Task.isCancelled,
                   let data = try? JSONEncoder().encode(OpenAIErrorResponse(
                       error: OpenAIError(message: "Request failed.", type: "server_error")
                   )) {
                    continuation.yield(.completion(NonStreamingChatPayload(
                        data: data,
                        trailers: ["x-mere-runtime-status": "500"]
                    )))
                }
                continuation.finish()
            }
        }
        continuation.onTermination = { termination in
            if case .cancelled = termination {
                session.observeClientDisconnect()
                heartbeatTask.cancel()
                generationTask.cancel()
            }
        }

        var headers: HTTPFields = [
            .contentType: "application/json",
            .trailer: Self.openAITimingTrailerNames.joined(separator: ", "),
        ]
        if let requestIDName = HTTPField.Name("x-mere-request-id") {
            headers[requestIDName] = session.requestID.uuidString
        }

        return Response(
            status: .ok,
            headers: headers,
            body: .init { writer in
                do {
                    // Leading JSON whitespace doubles as a bounded outbound
                    // disconnect probe without exposing partial model output.
                    try await writer.write(ByteBuffer(string: "\n"))
                    var iterator = stream.makeAsyncIterator()
                    while let event = await iterator.next() {
                        switch event {
                        case .heartbeat:
                            try await writer.write(ByteBuffer(string: " "))
                        case .completion(let payload):
                            try await writer.write(ByteBuffer(bytes: payload.data))
                            try await writer.finish(Self.httpFields(payload.trailers))
                            return
                        }
                    }
                    try await writer.finish(nil)
                } catch {
                    session.observeClientDisconnect()
                    heartbeatTask.cancel()
                    generationTask.cancel()
                    throw error
                }
            }
        )
    }

    private nonisolated static func httpFields(_ values: [String: String]) -> HTTPFields {
        var fields: HTTPFields = [:]
        for (name, value) in values {
            if let fieldName = HTTPField.Name(name) {
                fields[fieldName] = value
            }
        }
        return fields
    }

    private struct NonStreamingChatPayload: Sendable {
        let data: Data
        let trailers: [String: String]
    }

    private enum NonStreamingChatEvent: Sendable {
        case heartbeat
        case completion(NonStreamingChatPayload)
    }

    private nonisolated static let openAITimingTrailerNames = [
        "Server-Timing",
        "x-mere-prompt-tokens",
        "x-mere-generated-tokens",
        "x-mere-prefill-tokens-per-second",
        "x-mere-decode-tokens-per-second",
        "x-mere-kv-cache",
        "x-mere-runtime-status",
    ]

    private func embeddingModel(for requestedModel: String) async throws -> (modelID: String, modelPath: String) {
        let normalized = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if let spec = ManagedModelCatalog.spec(for: normalized),
           spec.id != Qwen3EmbeddingCatalog.modelId {
            throw APIRequestValidationError.invalidField(
                "model",
                "use \(Qwen3EmbeddingCatalog.modelId) or a local Qwen3 embedding model path"
            )
        }

        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: normalized,
            defaultModelID: Qwen3EmbeddingCatalog.modelId,
            progress: nil
        )
        let modelID = resolution.source == .explicitPath ? normalized : resolution.spec.id
        return (modelID, resolution.url.standardizedFileURL.path)
    }

    private func generateImage(_ plan: APIServerContract.ImageGenerationPlan) async throws -> URL {
        let modelID: String
        let modelRoot: URL
        let manifest: MereRunModelManifest
        let qwenEdit = plan.inputImage != nil && QwenImageEditRepository.canonicalModelId(for: plan.modelID) != nil
        if qwenEdit {
            guard let canonicalID = QwenImageEditRepository.canonicalModelId(for: plan.modelID),
                  let root = QwenImageEditRepository.resolveInstalledModelRoot(modelSpec: canonicalID) else {
                throw APIRequestValidationError.invalidField(
                    "model", "\(plan.modelID) is not installed; pull it before serving image edits"
                )
            }
            modelID = canonicalID
            modelRoot = root
            // Older managed Qwen edit installs did not require a manifest.
            manifest = try MereRunModelManifest.loadIfPresent(from: root)
                ?? MereRunModelManifest(id: canonicalID, engine: .qwenImageEdit, family: .qwen)
        } else {
            let resolved = try resolveImageModel(plan.modelID)
            modelID = resolved.modelID
            modelRoot = resolved.rootURL
            manifest = resolved.manifest
        }
        let outputURL = try temporaryOutputURL(directoryName: "mere-run-api-images", extension: "png")
        let operation = try plan.operationPlan(
            modelRoot: modelRoot, outputURL: outputURL, manifest: manifest, qwenEditDefaults: qwenEdit
        )
        let recording = try imageRunRecords.map { root in
            var requested = operation.options
            requested.mask = plan.maskImage
            return try ImageRunSession(directory: root.appendingPathComponent("image-\(UUID().uuidString.lowercased())"),
                                requested: requested, modelSelector: plan.modelID)
        }
        do {
            try MLXBundleSupport.ensureAvailable(quiet: true)
        } catch {
            try recording?.fail(error)
            throw error
        }
        let pool = sidecarPool
        let outcome = try await ImageGenerationOperation.execute(operation, recording: recording, executor: { kind, request, _ in
            try await pool.generateImage(kind: kind, modelID: modelID, modelPath: modelRoot.path, request: request)
        })
        return outcome.result.outputURL
    }

    private func synthesizeSpeech(_ plan: APIServerContract.SpeechPlan) async throws -> URL {
        let selection = try plan.modelSelection()
        try MLXBundleSupport.ensureAvailable(quiet: true)
        let outputURL = try temporaryOutputURL(directoryName: "mere-run-api-speech", extension: "wav")
        _ = try await sidecarPool.synthesizeSpeech(
            selection: selection,
            plan: plan.synthesisPlan(outputURL: outputURL)
        )
        return outputURL
    }

    private func speechResponseURL(_ wavURL: URL, responseFormat: String) throws -> URL {
        guard responseFormat != "wav" else {
            return wavURL
        }
        let outputURL = try temporaryOutputURL(
            directoryName: "mere-run-api-speech",
            extension: responseFormat
        )
        do {
            try MediaAudioIO.transcode(wavURL, to: outputURL, format: responseFormat)
        } catch {
            try? FileManager.default.removeItem(at: outputURL)
            throw error
        }
        return outputURL
    }

    private func transcribeAudio(
        audioURL: URL,
        plan: APIServerContract.TranscriptionPlan
    ) async throws -> ASRResult {
        try await APITranscription.execute(
            audioURL: audioURL, options: plan, recordingRoot: transcriptionRunRecords, services: services
        ).result
    }

    private func resolveImageModel(
        _ requestedModel: String
    ) throws -> (modelID: String, rootURL: URL, manifest: MereRunModelManifest) {
        let selection = ImageGenerationModelSelection(requestedModel.trimmingCharacters(in: .whitespacesAndNewlines))
        if case .unknown = selection {
            throw APIRequestValidationError.invalidField("model", "use a mere.run image model id or a local model path")
        }
        let root = try selection.resolveRoot()
        let manifest = try MereRunModelManifest.loadRequired(from: root)
        if case .managed(let id) = selection { return (id.rawValue, root, manifest) }
        return (manifest.id, root, manifest)
    }

    private nonisolated func temporaryOutputURL(
        directoryName: String,
        extension pathExtension: String
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("\(UUID().uuidString).\(pathExtension)")
    }

    private nonisolated func temporaryOutputDirectory(directoryName: String) throws -> URL {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent(directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let directory = parent.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        return directory
    }

    private nonisolated func writeMultipartFile(
        _ file: MultipartFormData.Part,
        directoryName: String,
        defaultExtension: String = "wav"
    ) throws -> URL {
        let pathExtension = sanitizedPathExtension(from: file.filename) ?? defaultExtension
        let outputURL = try temporaryOutputURL(directoryName: directoryName, extension: pathExtension)
        try file.body.write(to: outputURL)
        return outputURL
    }

    private nonisolated func sanitizedPathExtension(from filename: String?) -> String? {
        guard let rawExtension = filename.flatMap({ URL(fileURLWithPath: $0).pathExtension.lowercased() }),
              !rawExtension.isEmpty,
              rawExtension.allSatisfy({ $0.isLetter || $0.isNumber }) else {
            return nil
        }
        return rawExtension
    }

    private func handleRuntimeStatus(_ request: Request) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        var status = await services.status()
        status.process = processTelemetrySampler.snapshot()
        let data = try JSONEncoder().encode(status)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private func handleRuntimeLoad(_ request: Request, id: String) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        do {
            let snapshot = try await services.loadModel(id: id)
            return try jsonResponse(snapshot)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleRuntimeUnload(_ request: Request, id: String) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        do {
            let snapshot = try await services.unloadModel(id: id)
            return try jsonResponse(snapshot)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleRuntimeSettings(_ request: Request, id: String) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        do {
            let settings = try await pool.settings(idOrAlias: id)
            return try jsonResponse(settings)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func handleRuntimeSettingsPatch(_ request: Request, id: String) async throws -> Response {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIServerContract.acceptsJSONContentType(request.headers[.contentType]) else {
            return makeErrorResponse(
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json.",
                type: "invalid_request_error"
            )
        }
        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: 1024 * 1024)
        } catch {
            return makeErrorResponse(status: .badRequest, message: "Invalid request body.", type: "invalid_request_error")
        }
        do {
            let settings = try JSONDecoder().decode(RuntimeModelSettings.self, from: Data(body.readableBytesView))
            let updated = try await pool.updateSettings(idOrAlias: id, settings: settings)
            return try jsonResponse(updated)
        } catch {
            return runtimeErrorResponse(error)
        }
    }

    private func proxyDeepseekV4FlashChatCompletions(
        body: ByteBuffer,
        contentType: String?,
        session: RuntimeChatSession
    ) async throws -> Response {
        let upstreamURL = try await session.deepseekChatCompletionsURL()
        let data = Data(body.readableBytesView)

        if !DeepseekV4FlashClient.requestWantsStreamingResponse(data) {
            let upstreamResponse: DeepseekV4FlashClientResponse
            do {
                upstreamResponse = try await DeepseekV4FlashClient.normalizedChatCompletionData(
                    url: upstreamURL,
                    requestBody: data,
                    contentType: contentType
                )
            } catch {
                await session.finish()
                throw error
            }
            await session.finish()
            var headers: HTTPFields = [:]
            headers[.contentType] = upstreamResponse.contentType
            return Response(
                status: .init(code: upstreamResponse.statusCode),
                headers: headers,
                body: .init(byteBuffer: ByteBuffer(bytes: upstreamResponse.body))
            )
        }

        let upstreamRequest = DeepseekV4FlashClient.makeChatCompletionsRequest(
            url: upstreamURL,
            requestBody: data,
            contentType: contentType
        )
#if os(Linux)
        let (upstreamData, response) = try await URLSession.shared.data(for: upstreamRequest)
        let http = response as? HTTPURLResponse
        var headers: HTTPFields = [:]
        headers[.contentType] = http?.value(forHTTPHeaderField: "Content-Type") ?? "application/json"
        if DeepseekV4FlashClient.isEventStreamContentType(headers[.contentType]) {
            headers[.init("Cache-Control")!] = "no-cache"
            headers[.connection] = "keep-alive"
            await session.finish()
            let stream = AsyncStream<ByteBuffer> { continuation in
                if !upstreamData.isEmpty {
                    continuation.yield(ByteBuffer(bytes: upstreamData))
                }
                continuation.finish()
            }
            return Response(
                status: .init(code: http?.statusCode ?? 502),
                headers: headers,
                body: .init(asyncSequence: stream)
            )
        }
        let repairedData = DeepseekV4FlashClient.normalizedChatCompletionBody(
            upstreamData,
            contentType: headers[.contentType]
        )
        await session.finish()
        return Response(
            status: .init(code: http?.statusCode ?? 502),
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: repairedData))
        )
#else
        let (upstreamBytes, response) = try await URLSession.shared.bytes(for: upstreamRequest)
        let http = response as? HTTPURLResponse
        var headers: HTTPFields = [:]
        headers[.contentType] = http?.value(forHTTPHeaderField: "Content-Type") ?? "application/json"
        if DeepseekV4FlashClient.isEventStreamContentType(headers[.contentType]) {
            headers[.init("Cache-Control")!] = "no-cache"
            headers[.connection] = "keep-alive"
            let stream = RuntimeChatProxyStream.make(from: upstreamBytes, session: session)
            return Response(
                status: .init(code: http?.statusCode ?? 502),
                headers: headers,
                body: .init(asyncSequence: stream)
            )
        }

        var upstreamData = Data()
        do {
            for try await byte in upstreamBytes {
                upstreamData.append(byte)
            }
        } catch {
            await session.finish()
            throw error
        }
        await session.finish()
        let repairedData = DeepseekV4FlashClient.normalizedChatCompletionBody(
            upstreamData,
            contentType: headers[.contentType]
        )
        return Response(
            status: .init(code: http?.statusCode ?? 502),
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: repairedData))
        )
#endif
    }

    private func handleStreamingChat(_ session: RuntimeChatSession) async throws -> Response {
        let request = session.request
        let modelID = session.modelID
        let includeUsage = session.includeUsage
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        let encoder = JSONEncoder()

        // Create async stream for SSE
        let (stream, continuation) = AsyncStream<ByteBuffer>.makeStream()
        let heartbeatTask = Self.startStreamingKeepalive(continuation: continuation)

        let generationTask = Task {
            defer { heartbeatTask.cancel() }
            do {
                let streamedContent = StreamingContentTracker()
                let shouldBufferForToolCalls = request.tools?.isEmpty == false
                let result = try await session.chat { progress in
                    guard !shouldBufferForToolCalls else { return }
                    if let diffusion = progress.diffusion {
                        let chunk = OpenAIChatResponse(
                            id: id,
                            object: "chat.completion.chunk",
                            created: Int(Date().timeIntervalSince1970),
                            model: modelID,
                            choices: [
                                OpenAIChatChoice(
                                    index: 0,
                                    delta: OpenAIChatDelta(
                                        mere_diffusion_draft: OpenAIDiffusionDraft(diffusion)
                                    ),
                                    finish_reason: nil
                                )
                            ]
                        )
                        if let data = try? encoder.encode(chunk),
                           let json = String(data: data, encoding: .utf8) {
                            continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                        }
                        return
                    }
                    if progress.stage == .generating,
                       let token = progress.message,
                       !token.isEmpty,
                       !APIServerContract.isStreamingStatusMessage(token) {
                        streamedContent.markStreamed()
                        let chunk = OpenAIChatResponse(
                            id: id,
                            object: "chat.completion.chunk",
                            created: Int(Date().timeIntervalSince1970),
                            model: modelID,
                            choices: [
                                OpenAIChatChoice(
                                    index: 0,
                                    delta: OpenAIChatDelta(content: token),
                                    finish_reason: nil
                                )
                            ]
                        )
                        if let data = try? encoder.encode(chunk),
                           let json = String(data: data, encoding: .utf8) {
                            let line = "data: \(json)\n\n"
                            continuation.yield(ByteBuffer(string: line))
                        }
                    }
                }
                let responseToolCalls = openAIToolCalls(
                    from: result.toolCalls,
                    tools: request.tools,
                    parallelToolCalls: request.parallelToolCalls
                )
                if let toolCalls = responseToolCalls, !toolCalls.isEmpty {
                    let chunk = OpenAIChatResponse(
                        id: id,
                        object: "chat.completion.chunk",
                        created: Int(Date().timeIntervalSince1970),
                        model: modelID,
                        choices: [
                            OpenAIChatChoice(
                                index: 0,
                                delta: Self.openAIBufferedDelta(for: result, toolCalls: toolCalls),
                                finish_reason: nil
                            )
                        ]
                    )
                    if let data = try? encoder.encode(chunk),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                    }
                } else if !streamedContent.didStream, !result.response.isEmpty {
                    let chunk = OpenAIChatResponse(
                        id: id,
                        object: "chat.completion.chunk",
                        created: Int(Date().timeIntervalSince1970),
                        model: modelID,
                        choices: [
                            OpenAIChatChoice(
                                index: 0,
                                delta: Self.openAIBufferedDelta(for: result, toolCalls: []),
                                finish_reason: nil
                            )
                        ]
                    )
                    if let data = try? encoder.encode(chunk),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                    }
                }

                // Final chunk with finish_reason
                let finalChunk = OpenAIChatResponse(
                    id: id,
                    object: "chat.completion.chunk",
                    created: Int(Date().timeIntervalSince1970),
                    model: modelID,
                    choices: [
                        OpenAIChatChoice(
                            index: 0,
                            delta: OpenAIChatDelta(),
                            finish_reason: Self.openAIFinishReason(
                                for: result,
                                hasToolCalls: responseToolCalls?.isEmpty == false
                            )
                        )
                    ],
                    mere_diffusion: result.diffusion
                )
                if let data = try? encoder.encode(finalChunk),
                   let json = String(data: data, encoding: .utf8) {
                    continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                }

                if includeUsage {
                    let usageChunk = OpenAIChatResponse(
                        id: id,
                        object: "chat.completion.chunk",
                        created: Int(Date().timeIntervalSince1970),
                        model: modelID,
                        choices: [],
                        usage: Self.openAIUsage(for: result)
                    )
                    if let data = try? encoder.encode(usageChunk),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                    }
                }

                continuation.yield(ByteBuffer(string: "data: [DONE]\n\n"))
                await session.finish()
                continuation.finish()
            } catch {
                if !Task.isCancelled {
                    let errorResponse = OpenAIErrorResponse(
                        error: OpenAIError(message: "Request failed.", type: "server_error")
                    )
                    if let data = try? encoder.encode(errorResponse),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield(ByteBuffer(string: "data: \(json)\n\n"))
                    }
                }
                await session.finish(
                    cancelled: Task.isCancelled || error is CancellationError
                )
                continuation.finish()
            }
        }
        continuation.onTermination = { termination in
            heartbeatTask.cancel()
            if case .cancelled = termination {
                session.observeClientDisconnect()
                generationTask.cancel()
            }
        }

        return Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .init("Cache-Control")!: "no-cache",
                .connection: "keep-alive"
            ],
            body: .init(asyncSequence: stream)
        )
    }

    nonisolated static func startStreamingKeepalive(
        continuation: AsyncStream<ByteBuffer>.Continuation,
        interval: Duration = .seconds(15)
    ) -> Task<Void, Never> {
        Task {
            do {
                while !Task.isCancelled {
                    try await Task.sleep(for: interval)
                    guard !Task.isCancelled else { return }
                    // SSE comments keep buffered tool responses alive without
                    // exposing unfinished tool arguments or reasoning as text.
                    if case .terminated = continuation.yield(ByteBuffer(string: ": keep-alive\n\n")) {
                        return
                    }
                }
            } catch {
                // Cancellation ends the keepalive loop.
            }
        }
    }

    private nonisolated func openAIToolCalls(
        from toolCalls: [ToolCall]?,
        tools: [ToolDefinition]?,
        parallelToolCalls: Bool = true
    ) -> [OpenAIChatToolCall]? {
        guard let toolCalls, !toolCalls.isEmpty, let tools, !tools.isEmpty else { return nil }
        let validated = ToolCallPolicy.validatedCalls(
            toolCalls,
            tools: tools,
            parallelToolCalls: parallelToolCalls
        )
        guard !validated.isEmpty else { return nil }
        return validated.enumerated().map { index, call in
            let parameterTypes = tools
                .first(where: { $0.name == call.name })?
                .parameters
                .mapValues(\.type) ?? [:]
            return OpenAIChatToolCall(
                id: "call_\(index)_\(UUID().uuidString.prefix(8))",
                function: OpenAIChatToolCallFunction(
                    name: call.name,
                    arguments: APIServerContract.openAIToolArgumentsJSON(
                        call.arguments,
                        parameterTypes: parameterTypes
                    )
                )
            )
        }
    }

    nonisolated static func openAIFinishReason(
        for result: ChatResponse,
        hasToolCalls: Bool? = nil
    ) -> String {
        if hasToolCalls ?? (result.toolCalls?.isEmpty == false) {
            return "tool_calls"
        }
        return result.finishReason == .length ? "length" : "stop"
    }

    nonisolated static func openAIMessageContent(
        for result: ChatResponse,
        hasToolCalls: Bool
    ) -> String {
        guard !hasToolCalls else { return "" }
        guard result.reasoningContent != nil else { return result.response }
        return ChatReasoningMarkup.splitThinkBlocks(in: result.response).visibleContent
    }

    nonisolated static func openAIBufferedDelta(
        for result: ChatResponse,
        toolCalls: [OpenAIChatToolCall]
    ) -> OpenAIChatDelta {
        OpenAIChatDelta(
            role: toolCalls.isEmpty ? nil : "assistant",
            content: toolCalls.isEmpty ? openAIMessageContent(for: result, hasToolCalls: false) : nil,
            reasoning_content: result.reasoningContent,
            tool_calls: toolCalls.isEmpty
                ? nil
                : toolCalls.enumerated().map(OpenAIChatToolCallDelta.init(indexAndToolCall:))
        )
    }

    /// Generators that don't report a prompt token count fall back to zero
    /// rather than omitting the usage object, matching OpenAI's schema.
    nonisolated static func openAIUsage(for result: ChatResponse) -> OpenAIUsage {
        let promptTokens = result.promptTokens ?? 0
        return OpenAIUsage(
            prompt_tokens: promptTokens,
            completion_tokens: result.tokensGenerated,
            total_tokens: promptTokens + result.tokensGenerated
        )
    }

    nonisolated static func openAITimingHeaders(for result: ChatResponse) -> [String: String] {
        var headers = [
            "x-mere-prompt-tokens": String(result.promptTokens ?? 0),
            "x-mere-generated-tokens": String(result.tokensGenerated),
        ]
        guard let timing = result.timing else { return headers }

        let timeToFirstToken = timing.loadSeconds
            + timing.prefillSeconds
            + (timing.cacheConversionSeconds ?? 0)
            + (timing.firstTokenSeconds ?? 0)
        var serverTiming = [
            "model_load;dur=\(milliseconds(timing.loadSeconds))",
            "prefill;dur=\(milliseconds(timing.prefillSeconds))",
            "decode;dur=\(milliseconds(timing.decodeSeconds))",
            "ttft;dur=\(milliseconds(timeToFirstToken))",
        ]
        if let cacheConversionSeconds = timing.cacheConversionSeconds {
            serverTiming.insert(
                "kv_pack;dur=\(milliseconds(cacheConversionSeconds))",
                at: 2
            )
        }
        headers["server-timing"] = serverTiming.joined(separator: ", ")
        if let throughput = timing.prefillTokensPerSecond {
            headers["x-mere-prefill-tokens-per-second"] = fixedPrecision(throughput)
        }
        if let throughput = timing.decodeTokensPerSecond {
            headers["x-mere-decode-tokens-per-second"] = fixedPrecision(throughput)
        }
        if let kvCache = timing.decodeKVCache {
            headers["x-mere-kv-cache"] = kvCache
        }
        if let diffusion = result.diffusion {
            headers["x-mere-diffusion-seed"] = String(diffusion.seed)
            headers["x-mere-diffusion-canvas-tokens"] = String(diffusion.canvasTokens)
            headers["x-mere-diffusion-denoising-steps"] = String(diffusion.denoisingSteps)
            headers["x-mere-diffusion-work-tokens"] = String(diffusion.workTokens)
            headers["x-mere-diffusion-work-tokens-per-second"] = fixedPrecision(
                diffusion.workTokensPerSecond
            )
            if let firstDraftSeconds = diffusion.firstDraftSeconds {
                headers["x-mere-diffusion-first-draft-seconds"] = fixedPrecision(firstDraftSeconds)
            }
        }
        return headers
    }

    private nonisolated static func milliseconds(_ seconds: Double) -> String {
        fixedPrecision(seconds * 1_000)
    }

    private nonisolated static func fixedPrecision(_ value: Double) -> String {
        String(format: "%.3f", value)
    }

    private func unauthorizedResponseIfNeeded(for request: Request) -> Response? {
        guard let apiKey, !apiKey.isEmpty else { return nil }
        guard request.headers[.authorization] == "Bearer \(apiKey)" else {
            return makeErrorResponse(
                status: .unauthorized,
                message: "Unauthorized.",
                type: "authentication_error",
                extraHeaders: [.init("WWW-Authenticate")!: "Bearer"]
            )
        }
        return nil
    }

    /// Authentication is evaluated first so a remote caller never learns
    /// route details without a valid bearer token on non-loopback binds.
    private func vfxArtifactAccessResponseIfNeeded(
        for request: Request,
        remoteAddress: SocketAddress?
    ) -> Response? {
        if let unauthorized = unauthorizedResponseIfNeeded(for: request) {
            return unauthorized
        }
        guard APIVFXArtifactRoutePolicy.allows(remoteAddress: remoteAddress) else {
            return makeErrorResponse(
                status: .forbidden,
                message: APIVFXArtifactRoutePolicy.denialMessage,
                type: "permission_error"
            )
        }
        return nil
    }

    private func retainedArtifactJSONResponse<T: Encodable>(
        _ payload: T,
        outputDirectory: URL
    ) throws -> Response {
        let response = try jsonResponse(payload)
        artifactCleanupScheduler.scheduleCleanup(of: outputDirectory)
        return response
    }

    private nonisolated func jsonResponse<T: Encodable>(
        _ payload: T,
        status: HTTPResponse.Status = .ok
    ) throws -> Response {
        let data = try JSONEncoder().encode(payload)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private nonisolated func binaryResponse(_ data: Data, contentType: String) -> Response {
        Response(
            status: .ok,
            headers: [.contentType: contentType],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private nonisolated func runtimeErrorResponse(_ error: Error) -> Response {
        if let status = APIVFXClientErrorPolicy.status(for: error) {
            return makeErrorResponse(
                status: status,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        }
        switch error {
        case let error as RuntimeModelPoolError:
            switch error {
            case .unknownModel, .unsupportedModel, .modelNotInstalled, .incompatibleEngine, .invalidSettings:
                return makeErrorResponse(
                    status: .badRequest,
                    message: error.localizedDescription,
                    type: "invalid_request_error"
                )
            case .unloadConflict:
                return makeErrorResponse(
                    status: .conflict,
                    message: error.localizedDescription,
                    type: "conflict_error"
                )
            case .rawProxyUnavailable:
                return makeErrorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    type: "server_error"
                )
            }
        case let error as APISidecarModelPoolError:
            return makeErrorResponse(
                status: .serviceUnavailable,
                message: error.localizedDescription,
                type: "memory_pressure_error"
            )
        case let error as SpeechTranscriptionIssue:
            return makeErrorResponse(
                status: .badRequest, message: error.localizedDescription, type: "invalid_request_error"
            )
        case let error as ImageGenerationIssue:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        case let error as APIRequestValidationError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        case let error as MultipartFormData.ParseError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        case let error as ManagedModelResolver.ResolverError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        case let error as InstantMeshResourceError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        case let error as Qwen3EmbeddingModel.EmbeddingError:
            return makeErrorResponse(
                status: .badRequest,
                message: error.localizedDescription,
                type: "invalid_request_error"
            )
        default:
            return makeErrorResponse(
                status: .internalServerError,
                message: "Request failed.",
                type: "server_error"
            )
        }
    }

    private nonisolated func makeErrorResponse(
        status: HTTPResponse.Status,
        message: String,
        type: String,
        extraHeaders: HTTPFields = [:]
    ) -> Response {
        let payload = OpenAIErrorResponse(error: OpenAIError(message: message, type: type))
        let data = (try? JSONEncoder().encode(payload)) ?? Data("{\"error\":{\"message\":\"\(message)\",\"type\":\"\(type)\"}}".utf8)
        var headers: HTTPFields = [.contentType: "application/json"]
        for field in extraHeaders {
            headers.append(field)
        }
        return Response(
            status: status,
            headers: headers,
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }
}

actor APIRateLimiter {
    private let limitPerMinute: Int
    private var requestTimes: [Date] = []

    init(limitPerMinute: Int) {
        self.limitPerMinute = limitPerMinute
    }

    func allowRequest(now: Date = Date()) -> Bool {
        let cutoff = now.addingTimeInterval(-60)
        requestTimes.removeAll { $0 < cutoff }
        guard requestTimes.count < limitPerMinute else {
            return false
        }
        requestTimes.append(now)
        return true
    }
}

final class StreamingContentTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var streamed = false

    var didStream: Bool {
        lock.lock()
        defer { lock.unlock() }
        return streamed
    }

    func markStreamed() {
        lock.lock()
        streamed = true
        lock.unlock()
    }
}
