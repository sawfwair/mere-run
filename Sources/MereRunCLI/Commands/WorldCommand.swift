import ArgumentParser
import Foundation
import Hummingbird
import MereRunCore
import NIOCore
import NIOPosix

enum WorldBackend: String, CaseIterable, ExpressibleByArgument {
    case dreamx
    case cosmos3
}

struct World: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "world",
        abstract: "Run persistent local conditioned-video world sessions.",
        subcommands: [WorldServe.self]
    )
}

struct WorldServe: AsyncParsableCommand {
    private static let apiKeyEnvironmentKey = "MERERUN_API_KEY"

    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Serve one warm native world-model session over HTTP."
    )

    @Option(name: [.long], help: "Host to bind to.")
    var host = "127.0.0.1"

    @Option(name: [.long], help: "Port to listen on.")
    var port = 8791

    @Option(name: [.long], help: "Bearer token required by world endpoints. Also read from MERERUN_API_KEY.")
    var apiKey: String?

    @Option(name: [.long], help: "World runtime backend: dreamx or cosmos3.")
    var backend: WorldBackend = .dreamx

    @Option(name: [.long], help: "Wan2.2 TI2V base resource model id or local root.")
    var baseModel = Wan2Resources.modelID

    @Option(name: [.long], help: "Converted DreamX-World-5B AR model id or local root.")
    var model = Wan2DreamXCausalResources.modelID

    @Option(name: [.long], help: "Directory for transition videos and terminal state frames.")
    var stateDirectory: String?

    @Flag(name: [.long], help: "Load and warm all models before accepting requests.")
    var prepare = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: false)
        let resolvedKey = resolvedAPIKey()
        if !Self.isLoopback(host), resolvedKey?.isEmpty != false {
            throw ValidationError("Non-loopback world servers require --api-key or MERERUN_API_KEY.")
        }
        let stateURL = stateDirectory.map { URL(fileURLWithPath: $0).standardizedFileURL }
            ?? Self.defaultStateDirectory()
        let sessionBackend: WorldSessionBackend
        switch backend {
        case .dreamx:
            let baseRoot = try resolveRoot(baseModel)
            let causalRoot = try resolveRoot(model)
            let baseResources = Wan2Resources(rootURL: baseRoot)
            let missingBase = baseResources.validate()
            guard missingBase.isEmpty else {
                throw ValidationError(
                    "Wan2.2 base resources are incomplete: "
                        + missingBase.map(\.path).joined(separator: ", ")
                )
            }
            let causalResources = Wan2DreamXCausalResources(rootURL: causalRoot)
            let missingCausal = causalResources.validate()
            guard missingCausal.isEmpty else {
                throw ValidationError(
                    "DreamX causal resources are incomplete: "
                        + missingCausal.map(\.path).joined(separator: ", ")
                )
            }
            sessionBackend = .dreamx(Wan2WorldSession(
                resources: baseResources,
                stateDirectory: stateURL,
                causalWeightsURL: causalResources.weightsURL
            ))
        case .cosmos3:
            let requestedModel = model == Wan2DreamXCausalResources.modelID
                ? Cosmos3Resources.modelID
                : model
            let root = try resolveRoot(requestedModel)
            let resources = Cosmos3Resources(rootURL: root)
            let missing = resources.validate()
            guard missing.isEmpty else {
                throw ValidationError(
                    "Cosmos3-Edge resources are incomplete: "
                        + missing.map(\.path).joined(separator: ", ")
                )
            }
            sessionBackend = .cosmos3(Cosmos3WorldSession(
                resources: resources,
                stateDirectory: stateURL
            ))
        }
        let runtime = WorldHTTPRuntime(session: sessionBackend)
        if prepare {
            try await runtime.prepare()
        }
        let server = WorldHTTPServer(runtime: runtime, apiKey: resolvedKey)
        try await server.run(host: host, port: port)
    }

    func resolvedAPIKey(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        if let apiKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines),
           !apiKey.isEmpty {
            return apiKey
        }
        if let apiKey = environment[Self.apiKeyEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !apiKey.isEmpty {
            return apiKey
        }
        return nil
    }

    private func resolveRoot(_ value: String) throws -> URL {
        let expanded = NSString(string: value).expandingTildeInPath
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory), isDirectory.boolValue {
            return URL(fileURLWithPath: expanded).standardizedFileURL
        }
        guard let modelID = ModelResolver.ModelID(rawValue: value) else {
            throw ValidationError("World model is neither an installed model id nor a local directory: \(value)")
        }
        return try ModelResolver().resolve(modelID).rootURL
    }

    private static func isLoopback(_ host: String) -> Bool {
        let normalized = host.lowercased()
        return normalized == "localhost" || normalized == "127.0.0.1" || normalized == "::1"
    }

    private static func defaultStateDirectory() -> URL {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return support
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("world-sessions", isDirectory: true)
            .appendingPathComponent("default", isDirectory: true)
    }
}

private struct WorldRequestContext: RequestContext, RemoteAddressRequestContext {
    var coreContext: CoreRequestContextStorage
    let remoteAddress: SocketAddress?

    init(source: Source) {
        coreContext = CoreRequestContextStorage(source: source)
        remoteAddress = source.channel.remoteAddress
    }
}

struct WorldTransitionPayload: Codable, Sendable {
    let prompt: String
    let camera: Wan2WorldCameraControl
    let sourceImage: String?
    let output: String?
    let width: Int?
    let height: Int?
    let numFrames: Int?
    let steps: Int?
    let guidanceScale: Float?
    let shift: Float?
    let seed: UInt64?
    let fps: Int?
    let modelSpaceActions: [[Float]]?

    fileprivate func request(defaults: WorldTransitionDefaults) -> WorldTransitionRuntimeRequest {
        return WorldTransitionRuntimeRequest(
            base: Wan2WorldTransitionRequest(
                prompt: prompt,
                camera: camera,
                sourceImageURL: sourceImage.map { URL(fileURLWithPath: $0).standardizedFileURL },
                outputURL: output.map { URL(fileURLWithPath: $0).standardizedFileURL },
                width: width ?? defaults.width,
                height: height ?? defaults.height,
                numFrames: numFrames ?? defaults.numFrames,
                steps: steps ?? defaults.steps,
                guidanceScale: guidanceScale ?? defaults.guidanceScale,
                shift: shift ?? defaults.shift,
                seed: seed ?? defaults.seed,
                fps: fps ?? defaults.fps
            ),
            modelSpaceActions: modelSpaceActions
        )
    }

}

private struct WorldTransitionRuntimeRequest: Sendable {
    let base: Wan2WorldTransitionRequest
    let modelSpaceActions: [[Float]]?
}

private struct WorldTransitionDefaults: Sendable {
    let width: Int
    let height: Int
    let numFrames: Int
    let steps: Int
    let guidanceScale: Float
    let shift: Float
    let seed: UInt64
    let fps: Int
}

private struct WorldResetPayload: Codable, Sendable {
    let sourceImage: String?
}

private enum WorldJobStatus: String, Codable, Sendable {
    case queued
    case generating
    case cancelling
    case completed
    case cancelled
    case failed

    var isActive: Bool {
        self == .queued || self == .generating || self == .cancelling
    }
}

private struct WorldJobProgress: Codable, Sendable {
    let stage: String
    let stepIndex: Int
    let totalSteps: Int
    let updatedAt: Date

    enum CodingKeys: String, CodingKey {
        case stage
        case stepIndex = "step_index"
        case totalSteps = "total_steps"
        case updatedAt = "updated_at"
    }
}

private struct WorldTransitionReceiptPayload: Codable, Sendable {
    let requestID: UUID
    let previousStateID: UUID?
    let stateID: UUID
    let transitionIndex: Int
    let outputURL: URL
    let terminalFrameURL: URL
    let camera: Wan2WorldCameraControl
    let conditioningMode: String
    let actionDomain: Cosmos3ActionDomain?
    let actionSpace: String?
    let modelSpaceActions: [[Float]]?
    let rawActions: [[Float]]?
    let seed: UInt64

    enum CodingKeys: String, CodingKey {
        case camera, seed
        case requestID = "request_id"
        case previousStateID = "previous_state_id"
        case stateID = "state_id"
        case transitionIndex = "transition_index"
        case outputURL = "output_url"
        case terminalFrameURL = "terminal_frame_url"
        case conditioningMode = "conditioning_mode"
        case actionDomain = "action_domain"
        case actionSpace = "action_space"
        case modelSpaceActions = "model_space_actions"
        case rawActions = "raw_actions"
    }
}

private struct WorldJobSnapshot: Codable, Sendable {
    let jobID: UUID
    let status: WorldJobStatus
    let createdAt: Date
    let startedAt: Date?
    let completedAt: Date?
    let progress: WorldJobProgress?
    let progressEvents: [WorldJobProgress]
    let receipt: WorldTransitionReceiptPayload?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case status, progress, receipt, error
        case progressEvents = "progress_events"
        case jobID = "job_id"
        case createdAt = "created_at"
        case startedAt = "started_at"
        case completedAt = "completed_at"
    }
}

private struct WorldSessionSnapshotPayload: Codable, Sendable {
    let sessionID: UUID
    let phase: String
    let transitionCount: Int
    let currentStateID: UUID?
    let currentFrameURL: URL?
    let conditioningMode: String
    let keepsModelsWarm: Bool
    let keepsTerminalLatent: Bool

    enum CodingKeys: String, CodingKey {
        case phase
        case sessionID = "session_id"
        case transitionCount = "transition_count"
        case currentStateID = "current_state_id"
        case currentFrameURL = "current_frame_url"
        case conditioningMode = "conditioning_mode"
        case keepsModelsWarm = "keeps_models_warm"
        case keepsTerminalLatent = "keeps_terminal_latent"
    }
}

private struct WorldRuntimeSnapshot: Codable, Sendable {
    let object = "world.session"
    let backend: String
    let session: WorldSessionSnapshotPayload
    let activeJob: WorldJobSnapshot?

    enum CodingKeys: String, CodingKey {
        case object, backend, session
        case activeJob = "active_job"
    }
}

private struct WorldHealth: Encodable, Sendable {
    let status = "ok"
    let object = "world.health"
}

private struct WorldErrorPayload: Codable, Sendable {
    let error: String
}

private enum WorldHTTPRuntimeError: LocalizedError {
    case busy
    case jobNotFound

    var errorDescription: String? {
        switch self {
        case .busy: "A world transition is already active."
        case .jobNotFound: "World transition job not found."
        }
    }
}

private enum WorldSessionBackend: Sendable {
    case dreamx(Wan2WorldSession)
    case cosmos3(Cosmos3WorldSession)

    var identifier: String {
        switch self {
        case .dreamx: "dreamx"
        case .cosmos3: "cosmos3"
        }
    }

    var transitionDefaults: WorldTransitionDefaults {
        switch self {
        case .dreamx:
            return WorldTransitionDefaults(
                width: 512,
                height: 320,
                numFrames: 17,
                steps: 40,
                guidanceScale: 5,
                shift: 5,
                seed: 42,
                fps: 24
            )
        case .cosmos3:
            return WorldTransitionDefaults(
                width: 320,
                height: 176,
                numFrames: 17,
                steps: 30,
                guidanceScale: 1,
                shift: 3,
                seed: 0,
                fps: 30
            )
        }
    }

    func prepare() async throws {
        switch self {
        case .dreamx(let session):
            try await session.prepare()
        case .cosmos3(let session):
            try await session.prepare()
        }
    }

    func transition(
        _ runtimeRequest: WorldTransitionRuntimeRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> WorldTransitionReceiptPayload {
        let request = runtimeRequest.base
        switch self {
        case .dreamx(let session):
            let receipt = try await session.transition(
                request,
                progressHandler: progressHandler
            )
            return WorldTransitionReceiptPayload(
                requestID: receipt.requestID,
                previousStateID: receipt.previousStateID,
                stateID: receipt.stateID,
                transitionIndex: receipt.transitionIndex,
                outputURL: receipt.outputURL,
                terminalFrameURL: receipt.terminalFrameURL,
                camera: receipt.camera,
                conditioningMode: receipt.conditioningMode.rawValue,
                actionDomain: nil,
                actionSpace: nil,
                modelSpaceActions: nil,
                rawActions: nil,
                seed: receipt.seed
            )
        case .cosmos3(let session):
            let receipt = try await session.transition(
                Cosmos3WorldTransitionRequest(
                    requestID: request.requestID,
                    prompt: request.prompt,
                    camera: request.camera,
                    sourceImageURL: request.sourceImageURL,
                    outputURL: request.outputURL,
                    width: request.width,
                    height: request.height,
                    numFrames: request.numFrames,
                    steps: request.steps,
                    guidanceScale: request.guidanceScale,
                    shift: request.shift,
                    seed: request.seed,
                    fps: request.fps,
                    modelSpaceActions: runtimeRequest.modelSpaceActions
                ),
                progressHandler: progressHandler
            )
            return WorldTransitionReceiptPayload(
                requestID: receipt.requestID,
                previousStateID: receipt.previousStateID,
                stateID: receipt.stateID,
                transitionIndex: receipt.transitionIndex,
                outputURL: receipt.outputURL,
                terminalFrameURL: receipt.terminalFrameURL,
                camera: receipt.camera,
                conditioningMode: receipt.conditioningMode,
                actionDomain: receipt.actionDomain,
                actionSpace: receipt.actionSpace,
                modelSpaceActions: receipt.modelSpaceActions,
                rawActions: receipt.modelSpaceActions,
                seed: receipt.seed
            )
        }
    }

    func reset(sourceImageURL: URL?) async throws {
        switch self {
        case .dreamx(let session):
            try await session.reset(sourceImageURL: sourceImageURL)
        case .cosmos3(let session):
            try await session.reset(sourceImageURL: sourceImageURL)
        }
    }

    func unload() async throws {
        switch self {
        case .dreamx(let session):
            try await session.unload()
        case .cosmos3(let session):
            try await session.unload()
        }
    }

    func snapshot() async -> WorldSessionSnapshotPayload {
        switch self {
        case .dreamx(let session):
            let snapshot = await session.snapshot()
            return WorldSessionSnapshotPayload(
                sessionID: snapshot.sessionID,
                phase: snapshot.phase.rawValue,
                transitionCount: snapshot.transitionCount,
                currentStateID: snapshot.currentStateID,
                currentFrameURL: snapshot.currentFrameURL,
                conditioningMode: snapshot.conditioningMode.rawValue,
                keepsModelsWarm: snapshot.keepsModelsWarm,
                keepsTerminalLatent: snapshot.keepsTerminalLatent
            )
        case .cosmos3(let session):
            let snapshot = await session.snapshot()
            return WorldSessionSnapshotPayload(
                sessionID: snapshot.sessionID,
                phase: snapshot.phase.rawValue,
                transitionCount: snapshot.transitionCount,
                currentStateID: snapshot.currentStateID,
                currentFrameURL: snapshot.currentFrameURL,
                conditioningMode: snapshot.conditioningMode,
                keepsModelsWarm: snapshot.keepsModelsWarm,
                keepsTerminalLatent: snapshot.keepsTerminalLatent
            )
        }
    }
}

private actor WorldHTTPRuntime {
    private let session: WorldSessionBackend
    private var jobs: [UUID: WorldJobSnapshot] = [:]
    private var tasks: [UUID: Task<Void, Never>] = [:]

    init(session: WorldSessionBackend) {
        self.session = session
    }

    func prepare() async throws {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        try await session.prepare()
    }

    func snapshot() async -> WorldRuntimeSnapshot {
        WorldRuntimeSnapshot(
            backend: session.identifier,
            session: await session.snapshot(),
            activeJob: activeJob
        )
    }

    func startTransition(_ payload: WorldTransitionPayload) throws -> WorldJobSnapshot {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        let id = UUID()
        let job = WorldJobSnapshot(
            jobID: id,
            status: .queued,
            createdAt: Date(),
            startedAt: nil,
            completedAt: nil,
            progress: nil,
            progressEvents: [],
            receipt: nil,
            error: nil
        )
        jobs[id] = job
        let defaults = session.transitionDefaults
        tasks[id] = Task { [weak self] in
            await self?.runTransition(id: id, request: payload.request(defaults: defaults))
        }
        return job
    }

    func job(id: UUID) throws -> WorldJobSnapshot {
        guard let job = jobs[id] else { throw WorldHTTPRuntimeError.jobNotFound }
        return job
    }

    func cancel(id: UUID) throws -> WorldJobSnapshot {
        guard let job = jobs[id] else { throw WorldHTTPRuntimeError.jobNotFound }
        guard job.status.isActive else { return job }
        tasks[id]?.cancel()
        let updated = WorldJobSnapshot(
            jobID: job.jobID,
            status: .cancelling,
            createdAt: job.createdAt,
            startedAt: job.startedAt,
            completedAt: nil,
            progress: job.progress,
            progressEvents: job.progressEvents,
            receipt: nil,
            error: nil
        )
        jobs[id] = updated
        return updated
    }

    func reset(_ payload: WorldResetPayload?) async throws -> WorldRuntimeSnapshot {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        try await session.reset(sourceImageURL: payload?.sourceImage.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        })
        return await snapshot()
    }

    func unload() async throws -> WorldRuntimeSnapshot {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        try await session.unload()
        return await snapshot()
    }

    private var activeJob: WorldJobSnapshot? {
        jobs.values
            .filter { $0.status.isActive }
            .sorted { $0.createdAt < $1.createdAt }
            .first
    }

    private func runTransition(id: UUID, request: WorldTransitionRuntimeRequest) async {
        guard let job = jobs[id] else { return }
        jobs[id] = WorldJobSnapshot(
            jobID: id,
            status: .generating,
            createdAt: job.createdAt,
            startedAt: Date(),
            completedAt: nil,
            progress: nil,
            progressEvents: [],
            receipt: nil,
            error: nil
        )
        do {
            let receipt = try await session.transition(request) { [weak self] progress in
                Task { await self?.recordProgress(id: id, progress: progress) }
            }
            let current = jobs[id]!
            jobs[id] = WorldJobSnapshot(
                jobID: id,
                status: .completed,
                createdAt: current.createdAt,
                startedAt: current.startedAt,
                completedAt: Date(),
                progress: current.progress,
                progressEvents: current.progressEvents,
                receipt: receipt,
                error: nil
            )
        } catch is CancellationError {
            finish(id: id, status: .cancelled, error: nil)
        } catch {
            finish(id: id, status: .failed, error: error.localizedDescription)
        }
        tasks[id] = nil
    }

    private func recordProgress(id: UUID, progress: GenerationProgress) {
        guard let job = jobs[id], job.status.isActive else { return }
        let event = WorldJobProgress(
            stage: progress.stage.rawValue,
            stepIndex: progress.stepIndex,
            totalSteps: progress.totalSteps,
            updatedAt: Date()
        )
        jobs[id] = WorldJobSnapshot(
            jobID: id,
            status: job.status == .cancelling ? .cancelling : .generating,
            createdAt: job.createdAt,
            startedAt: job.startedAt,
            completedAt: nil,
            progress: event,
            progressEvents: job.progressEvents + [event],
            receipt: nil,
            error: nil
        )
    }

    private func finish(id: UUID, status: WorldJobStatus, error: String?) {
        guard let job = jobs[id] else { return }
        jobs[id] = WorldJobSnapshot(
            jobID: id,
            status: status,
            createdAt: job.createdAt,
            startedAt: job.startedAt,
            completedAt: Date(),
            progress: job.progress,
            progressEvents: job.progressEvents,
            receipt: nil,
            error: error
        )
    }
}

private struct WorldHTTPServer: Sendable {
    let runtime: WorldHTTPRuntime
    let apiKey: String?

    func run(host: String, port: Int) async throws {
        let app = Application(
            router: router(),
            configuration: .init(address: .hostname(host, port: port))
        )
        print("Native world server: http://\(host):\(port)/v1/world/session")
        print("Press Ctrl+C to stop.")
        try await app.runService()
    }

    private func router() -> Router<WorldRequestContext> {
        let router = Router(context: WorldRequestContext.self)
        router.get("/health") { _, _ in
            try response(WorldHealth())
        }
        router.get("/v1/world/session") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            return try response(await runtime.snapshot())
        }
        router.post("/v1/world/session/prepare") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                try await runtime.prepare()
                return try response(await runtime.snapshot())
            } catch {
                return try errorResponse(error)
            }
        }
        router.post("/v1/world/session/transitions") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                let payload: WorldTransitionPayload = try await decode(request)
                return try response(try await runtime.startTransition(payload), status: .accepted)
            } catch {
                return try errorResponse(error)
            }
        }
        router.get("/v1/world/jobs/:id") { request, context in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                guard let raw = context.parameters.get("id", as: String.self), let id = UUID(uuidString: raw) else {
                    return try response(WorldErrorPayload(error: "Invalid job id."), status: .badRequest)
                }
                return try response(try await runtime.job(id: id))
            } catch {
                return try errorResponse(error)
            }
        }
        router.delete("/v1/world/jobs/:id") { request, context in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                guard let raw = context.parameters.get("id", as: String.self), let id = UUID(uuidString: raw) else {
                    return try response(WorldErrorPayload(error: "Invalid job id."), status: .badRequest)
                }
                return try response(try await runtime.cancel(id: id), status: .accepted)
            } catch {
                return try errorResponse(error)
            }
        }
        router.post("/v1/world/session/reset") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                let payload: WorldResetPayload? = try await decodeOptional(request)
                return try response(try await runtime.reset(payload))
            } catch {
                return try errorResponse(error)
            }
        }
        router.post("/v1/world/session/unload") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                return try response(try await runtime.unload())
            } catch {
                return try errorResponse(error)
            }
        }
        return router
    }
}

private func decode<T: Decodable>(_ request: Request, as _: T.Type = T.self) async throws -> T {
    let body = try await request.body.collect(upTo: 1024 * 1024)
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(T.self, from: Data(body.readableBytesView))
}

private func decodeOptional<T: Decodable>(_ request: Request, as _: T.Type = T.self) async throws -> T? {
    let body = try await request.body.collect(upTo: 1024 * 1024)
    guard body.readableBytes > 0 else { return nil }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(T.self, from: Data(body.readableBytesView))
}

private func response<T: Encodable>(
    _ payload: T,
    status: HTTPResponse.Status = .ok
) throws -> Response {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    encoder.dateEncodingStrategy = .iso8601
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(payload)
    return Response(
        status: status,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(bytes: data))
    )
}

private func errorResponse(_ error: Error) throws -> Response {
    let status: HTTPResponse.Status
    switch error {
    case WorldHTTPRuntimeError.busy, Wan2WorldSessionError.busy:
        status = .conflict
    case WorldHTTPRuntimeError.jobNotFound:
        status = .notFound
    default:
        status = .badRequest
    }
    return try response(WorldErrorPayload(error: error.localizedDescription), status: status)
}

private func unauthorized(_ request: Request, apiKey: String?) -> Response? {
    guard let apiKey, !apiKey.isEmpty else { return nil }
    guard request.headers[.authorization] == "Bearer \(apiKey)" else {
        return try? response(WorldErrorPayload(error: "Unauthorized."), status: .unauthorized)
    }
    return nil
}
