import ArgumentParser
import AudioCore
import Foundation
import Hummingbird
import MereRunCore
import NIOCore

actor YuE2JobQueue {
    struct Job: Codable, Sendable {
        enum Status: String, Codable, Sendable { case queued, running, succeeded, truncated, failed, cancelled }
        let id: UUID
        var status: Status
        var error: String?
    }
    typealias Execute = @Sendable (YuE2GenerationPlan, URL) async throws -> Bool
    private let root: URL
    private let maximumPending: Int
    private let execute: Execute
    private var jobs: [UUID: Job] = [:]
    private var tasks: [UUID: Task<Void, Never>] = [:]
    private var tail: Task<Void, Never>?

    init(root: URL, maximumPending: Int, execute: @escaping Execute) {
        self.root = root
        self.maximumPending = maximumPending
        self.execute = execute
    }

    func submit(_ plan: YuE2GenerationPlan) throws -> Job {
        try plan.validate()
        guard tasks.count < maximumPending else {
            throw HTTPError(.tooManyRequests, message: "YuE2 generation queue is full.")
        }
        let id = UUID()
        let job = Job(id: id, status: .queued)
        jobs[id] = job
        let previous = tail
        let task = Task {
            await previous?.value
            do {
                try Task.checkCancellation()
                setStatus(id, .running)
                let directory = root.appendingPathComponent(id.uuidString)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
                let truncated = try await execute(plan, directory)
                try Task.checkCancellation()
                setStatus(id, truncated ? .truncated : .succeeded)
            } catch is CancellationError { setStatus(id, .cancelled) }
            catch { setStatus(id, .failed, error: error.localizedDescription) }
            tasks[id] = nil
        }
        tasks[id] = task
        tail = task
        return job
    }

    func get(_ id: UUID) throws -> Job {
        guard let job = jobs[id] else { throw HTTPError(.notFound) }
        return job
    }

    func cancel(_ id: UUID) throws -> Job {
        let job = try get(id)
        if job.status == .queued || job.status == .running {
            tasks[id]?.cancel()
            setStatus(id, .cancelled)
        }
        return try get(id)
    }

    func wait(_ id: UUID) async { await tasks[id]?.value }

    func artifact(_ id: UUID, name: String) throws -> Data {
        let job = try get(id)
        guard job.status == .succeeded || job.status == .truncated else { throw HTTPError(.conflict, message: "Job has no completed output.") }
        let paths = ["audio": "audio.wav", "score": "stages/score/score.abc", "latents": "stages/acoustic/latent.npy"]
        guard let path = paths[name] else { throw HTTPError(.notFound) }
        let url = root.appendingPathComponent(id.uuidString).appendingPathComponent(path)
        guard FileManager.default.fileExists(atPath: url.path) else { throw HTTPError(.notFound) }
        return try Data(contentsOf: url)
    }

    private func setStatus(_ id: UUID, _ status: Job.Status, error: String? = nil) {
        jobs[id]?.status = status
        jobs[id]?.error = error
    }
}

struct YuE2Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "serve", abstract: "Serve queued native YuE2 jobs with reusable artifacts.")
    @Option(name: .long, help: "Host to bind to (default: 127.0.0.1).") var host = "127.0.0.1"
    @Option(name: .long, help: "HTTP port (default: 8082).") var port = 8082
    @Option(name: [.short, .long], help: "Managed YuE2 ID or local model directory.") var model = "music-yue2"
    @Option(name: [.short, .long], help: "Directory for immutable per-job artifacts.") var output: String
    @Option(name: .customLong("api-key"), help: "Bearer token; also read from MERERUN_API_KEY.") var apiKey: String?
    @Option(name: .customLong("max-pending"), help: "Maximum queued and running jobs (default: 16).") var maximumPending = 16
    @Option(name: .customLong("memory-mode"), help: "Weight loading: resident (default), staged, or low-memory.") var memoryMode: YuE2MemoryMode = .resident

    func validate() throws {
        guard (1...65535).contains(port), (1...128).contains(maximumPending) else { throw ValidationError("Invalid port or queue capacity.") }
        if !MusicServe.isLoopback(host), (apiKey ?? ProcessInfo.processInfo.environment[APIServe.apiKeyEnvironmentKey])?.isEmpty != false {
            throw ValidationError("Non-loopback music API binds require --api-key or MERERUN_API_KEY.")
        }
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: false)
        let root = model == "music-yue2" ? try ModelResolver().resolve(.yue2).rootURL : YuE2Transcribe.url(model)
        let resources = YuE2Resources(rootURL: root)
        guard resources.validate().isEmpty else { throw ValidationError("YuE2 model files are incomplete.") }
        let operation = YuE2GenerationOperation(resources: resources, memoryMode: memoryMode)
        let destination = YuE2Transcribe.url(output)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let queue = YuE2JobQueue(root: destination, maximumPending: maximumPending) { plan, directory in
            let result = try await operation.generate(plan, stageOutput: directory.appendingPathComponent("stages"))
            _ = try AudioExportService.write(result.waveform, plan: AudioExportPlan(options: .music), to: directory.appendingPathComponent("audio.wav"))
            return result.scoreTruncated || result.musicTruncated
        }
        let key = apiKey ?? ProcessInfo.processInfo.environment[APIServe.apiKeyEnvironmentKey]
        let router = Self.router(queue: queue, apiKey: key)
        let application = Application(router: router, configuration: .init(address: .hostname(host, port: port)))
        CLIStderr.write("Native YuE2 job API: http://\(host):\(port)/v1/jobs\n")
        try await application.runService()
    }

    static func router(queue: YuE2JobQueue, apiKey: String?) -> Router<BasicRequestContext> {
        let router = Router()
        @Sendable func authorize(_ request: Request) throws {
            if let apiKey, !apiKey.isEmpty, request.headers[.authorization] != "Bearer \(apiKey)" { throw HTTPError(.unauthorized) }
        }
        router.get("/health") { request, _ in
            try authorize(request)
            return try json(["status": "ok", "runtime": "native-swift-mlx"])
        }
        router.post("/v1/jobs") { request, _ in
            try authorize(request)
            guard request.headers[.contentType]?.lowercased().contains("application/json") == true else {
                throw HTTPError(.badRequest, message: "Content-Type must be application/json.")
            }
            let bytes = try await request.body.collect(upTo: 1_048_576)
            let plan: YuE2GenerationPlan
            do { plan = try JSONDecoder().decode(YuE2GenerationPlan.self, from: Data(bytes.readableBytesView)) }
            catch { throw HTTPError(.badRequest, message: error.localizedDescription) }
            return try await json(queue.submit(plan), status: .accepted)
        }
        router.get("/v1/jobs/:id") { request, context in
            try authorize(request)
            return try await json(queue.get(identifier(context)))
        }
        router.post("/v1/jobs/:id/cancel") { request, context in
            try authorize(request)
            return try await json(queue.cancel(identifier(context)))
        }
        router.get("/v1/jobs/:id/:artifact") { request, context in
            try authorize(request)
            let name = context.parameters.get("artifact", as: String.self) ?? ""
            let data = try await queue.artifact(identifier(context), name: name)
            return Response(status: .ok, headers: [.contentType: name == "audio" ? "audio/wav" : "application/octet-stream"],
                            body: .init(byteBuffer: ByteBuffer(bytes: data)))
        }
        return router
    }

    private static func identifier(_ context: BasicRequestContext) throws -> UUID {
        guard let text = context.parameters.get("id", as: String.self), let id = UUID(uuidString: text) else { throw HTTPError(.notFound) }
        return id
    }
    private static func json(_ value: some Encodable, status: HTTPResponse.Status = .ok) throws -> Response {
        Response(status: status, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: try JSONEncoder().encode(value))))
    }
}
