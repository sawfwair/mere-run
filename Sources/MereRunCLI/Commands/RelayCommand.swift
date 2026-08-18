import ArgumentParser
import Foundation
import Hummingbird
import MereRunCore
import MereRunRelayKit
import NIOCore

struct Relay: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "relay",
        abstract: "Host the relay API surface directly on this machine.",
        subcommands: [RelayServe.self]
    )
}

/// The direct lane: serves the same graph-job HTTP surface as the hosted
/// relay, executes jobs on this machine, and pairs clients with a short
/// code shown in the terminal — no broker, no cloud hop. Reachable over the
/// LAN or a tailnet; prompts and outputs never leave your machines.
struct RelayServe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Serve the relay graph-job API from this machine and run submitted jobs locally."
    )

    @Option(name: [.long], help: "Interface to bind.")
    var host = "0.0.0.0"

    @Option(name: [.long], help: "Port to listen on.")
    var port = 6373

    @Option(name: [.long], help: "Display name for this relay (defaults to the machine's hostname).")
    var name: String?

    @Option(name: [.customLong("pairing-window")], help: "Minutes after startup during which new devices can pair.")
    var pairingWindowMinutes = 15

    @Option(name: [.customLong("state-dir")], help: ArgumentHelp("Override the spool directory.", visibility: .hidden))
    var stateDirectory: String?

    func run() async throws {
        let relayName = name ?? ProcessInfo.processInfo.hostName
        let root = stateDirectory.map { URL(fileURLWithPath: $0).standardizedFileURL }
            ?? MereRunModelPaths.applicationSupportBase.appendingPathComponent("local-relay", isDirectory: true)
        var generator = SystemRandomNumberGenerator()
        let pairingCode = String(format: "%06d", Int.random(in: 0...999_999, using: &generator))
        let state = try await LocalRelayState(
            rootDirectory: root,
            relayName: relayName,
            pairingCode: pairingCode,
            pairingWindowMinutes: pairingWindowMinutes
        )

        let router = Self.buildRouter(state: state)
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )

        let pairedCount = await state.pairedDeviceCount
        let displayCode = "\(pairingCode.prefix(3))-\(pairingCode.suffix(3))"
        print("mere.run direct relay \"\(relayName)\"")
        print("Listening on http://\(host):\(port)")
        print("Reach it from your phone at http://\(ProcessInfo.processInfo.hostName):\(port)")
        print("or over Tailscale at this machine's tailnet address (use `tailscale serve` for HTTPS).")
        print("")
        print("Pairing code: \(displayCode)  (valid for \(pairingWindowMinutes) minutes; restart to reopen pairing)")
        if pairedCount > 0 {
            print("\(pairedCount) device\(pairedCount == 1 ? "" : "s") already paired.")
        }
        print("Press Ctrl+C to stop.")
        fflush(stdout)

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try await app.runService()
            }
            group.addTask {
                let queue = await state.makeQueue()
                for await jobID in queue {
                    await Self.runJob(jobID: jobID, state: state)
                }
            }
            try await group.next()
            group.cancelAll()
        }
    }

    // MARK: - Job execution

    private static let executionQueue = DispatchQueue(label: "run.mere.local-relay.execute", qos: .userInitiated)

    private static func runJob(jobID: String, state: LocalRelayState) async {
        guard await state.beginRun(jobID: jobID) else { return }
        let bundleDir = await state.bundleDirectory(jobID: jobID)
        let runDir = await state.runDirectory(jobID: jobID)
        do {
            // WorkflowRunner owns run-directory creation and refuses an
            // existing one without resume; retry already cleared it.
            let outcome: WorkflowRunOutcome = try await withCheckedThrowingContinuation { continuation in
                executionQueue.async {
                    do {
                        let runner = try WorkflowRunner(
                            bundleDirectory: bundleDir,
                            runDirectory: runDir,
                            resume: false,
                            executor: .init(kind: "worker", profile: nil, jobReference: nil),
                            eventHandler: nil
                        )
                        continuation.resume(returning: try runner.execute())
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
            let manifestError = (try? WorkflowBundleCodec.decoder().decode(
                GraphRunManifest.self,
                from: Data(contentsOf: runDir.appendingPathComponent(GraphRunManifest.filename))
            ))?.error
            await state.finishRun(jobID: jobID, state: outcome.state, error: manifestError)
        } catch let error as RelayClientError {
            await state.finishRun(jobID: jobID, state: .failed, error: error.message)
        } catch let error as ValidationError {
            await state.finishRun(jobID: jobID, state: .failed, error: error.message)
        } catch {
            await state.finishRun(jobID: jobID, state: .failed, error: String(describing: error))
        }
    }

    // MARK: - Router

    static func buildRouter(state: LocalRelayState) -> Router<BasicRequestContext> {
        let router = Router(context: BasicRequestContext.self)

        router.get("/.well-known/mere-run-relay") { _, _ in
            try jsonResponse(LocalRelayDiscoveryDocument(
                schemaVersion: 1,
                kind: LocalRelayDiscoveryDocument.kind,
                authMode: LocalRelayDiscoveryDocument.authMode,
                relayName: state.relayName,
                contractVersions: [WorkflowJobManifest.contractVersion]
            ))
        }

        router.post("/api/pair") { request, _ in
            let body = try await request.body.collect(upTo: 1 << 16)
            guard let pairRequest = try? WorkflowBundleCodec.decoder().decode(
                LocalRelayPairRequest.self,
                from: Data(body.readableBytesView)
            ) else {
                return errorResponse(.badRequest, "Pairing needs a JSON body with `code` and `device_name`.")
            }
            switch try await state.pair(code: pairRequest.code, deviceName: pairRequest.deviceName) {
            case .paired(let token):
                return try jsonResponse(LocalRelayPairResponse(token: token, relayName: state.relayName))
            case .rejected:
                return errorResponse(.unauthorized, "The pairing code did not match.")
            case .closed:
                return errorResponse(.forbidden, "Pairing is closed. Restart `mere.run relay serve` to reopen it.")
            }
        }

        router.get("/api/graph-jobs/capabilities") { request, _ in
            try await requireAuth(request, state: state)
            return try jsonResponse(await state.probe())
        }

        router.get("/api/fleet") { request, _ in
            try await requireAuth(request, state: state)
            return try jsonResponse(await state.fleetSnapshot())
        }

        router.post("/api/fleet/nodes/{id}/refresh") { request, context in
            try await requireAuth(request, state: state)
            let deviceID = context.parameters.get("id", as: String.self) ?? "local"
            return try jsonResponse(RelayFleetRefreshResult(deviceID: deviceID, requested: true))
        }

        router.post("/api/graph-jobs") { request, _ in
            try await requireAuth(request, state: state)
            return try await handle {
                let body = try await request.body.collect(upTo: 64 << 20)
                let createRequest = try WorkflowBundleCodec.decoder().decode(
                    RelayGraphCreateRequest.self,
                    from: Data(body.readableBytesView)
                )
                return try jsonResponse(await state.create(request: createRequest))
            }
        }

        router.put("/api/graph-jobs/{id}/assets/{digest}") { request, context in
            try await requireAuth(request, state: state)
            let jobID = try requireParameter(context, "id")
            let digest = try requireParameter(context, "digest")
            let body = try await request.body.collect(upTo: 256 << 20)
            return try await handle {
                try await state.storeAsset(jobID: jobID, digest: digest, data: Data(body.readableBytesView))
                return try jsonResponse(["stored": digest])
            }
        }

        router.post("/api/graph-jobs/{id}/commit") { request, context in
            try await requireAuth(request, state: state)
            let jobID = try requireParameter(context, "id")
            return try await handle {
                let record = try await state.commit(jobID: jobID)
                return try jsonResponse(record.response(artifacts: []))
            }
        }

        router.get("/api/graph-jobs") { request, _ in
            try await requireAuth(request, state: state)
            let limit = request.uri.queryParameters["limit"].flatMap { Int($0) } ?? 20
            let records = await state.list(limit: limit)
            var responses: [RelayGraphJobResponse] = []
            for record in records {
                responses.append(record.response(artifacts: await state.artifacts(jobID: record.jobID)))
            }
            return try jsonResponse(RelayGraphJobListResponse(jobs: responses))
        }

        router.get("/api/graph-jobs/{id}") { request, context in
            try await requireAuth(request, state: state)
            let jobID = try requireParameter(context, "id")
            guard let record = await state.record(jobID: jobID) else {
                return errorResponse(.notFound, "Unknown job \(jobID).")
            }
            return try jsonResponse(record.response(artifacts: await state.artifacts(jobID: jobID)))
        }

        router.delete("/api/graph-jobs/{id}") { request, context in
            try await requireAuth(request, state: state)
            let jobID = try requireParameter(context, "id")
            return try await handle {
                let record = try await state.cancel(jobID: jobID)
                return try jsonResponse(record.response(artifacts: await state.artifacts(jobID: jobID)))
            }
        }

        router.post("/api/graph-jobs/{id}/retry") { request, context in
            try await requireAuth(request, state: state)
            let jobID = try requireParameter(context, "id")
            return try await handle {
                let record = try await state.retry(jobID: jobID)
                return try jsonResponse(record.response(artifacts: []))
            }
        }

        router.get("/api/graph-jobs/{id}/run-manifest") { request, context in
            try await requireAuth(request, state: state)
            let jobID = try requireParameter(context, "id")
            guard let data = await state.manifestData(jobID: jobID) else {
                return errorResponse(.notFound, "Job \(jobID) has no run manifest yet.")
            }
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(bytes: data))
            )
        }

        router.get("/api/graph-jobs/{id}/events") { request, context in
            try await requireAuth(request, state: state)
            let jobID = try requireParameter(context, "id")
            let text = await state.eventsText(jobID: jobID)
            return Response(
                status: .ok,
                headers: [.contentType: "text/plain; charset=utf-8"],
                body: .init(byteBuffer: ByteBuffer(string: text))
            )
        }

        router.get("/api/graph-jobs/{id}/artifacts/{name}") { request, context in
            try await requireAuth(request, state: state)
            let jobID = try requireParameter(context, "id")
            let name = try requireParameter(context, "name")
            return try await handle {
                let (url, artifact) = try await state.artifactURL(jobID: jobID, name: name)
                guard let fileHandle = FileHandle(forReadingAtPath: url.path) else {
                    return errorResponse(.notFound, "Artifact \(name) is missing from the run directory.")
                }
                var headers = HTTPFields()
                headers[.contentType] = artifact.contentType
                headers[.contentLength] = String(artifact.sizeBytes)
                return Response(
                    status: .ok,
                    headers: headers,
                    body: .init(asyncSequence: FileChunkSequence(handle: fileHandle))
                )
            }
        }

        return router
    }

    // MARK: - Helpers

    /// Client-meaningful failures travel as 400s with the message in the
    /// body, the way the hosted relay reports them; everything else stays a
    /// 500 through Hummingbird's default handling.
    private static func handle(_ body: () async throws -> Response) async throws -> Response {
        do {
            return try await body()
        } catch let error as RelayClientError {
            return errorResponse(.badRequest, error.message)
        }
    }

    private static func requireAuth(_ request: Request, state: LocalRelayState) async throws {
        let header = request.headers[.authorization] ?? ""
        let token = header.hasPrefix("Bearer ") ? String(header.dropFirst("Bearer ".count)) : ""
        guard await state.authorized(bearerToken: token) else {
            throw HTTPError(.unauthorized, message: "Pair with this relay before calling its API.")
        }
    }

    private static func requireParameter(_ context: BasicRequestContext, _ name: String) throws -> String {
        guard let value = context.parameters.get(name, as: String.self), !value.isEmpty else {
            throw HTTPError(.badRequest, message: "Missing path parameter \(name).")
        }
        return value
    }

    private static func jsonResponse(_ value: some Encodable) throws -> Response {
        let data = try WorkflowBundleCodec.encoder().encode(value)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    private static func errorResponse(_ status: HTTPResponse.Status, _ message: String) -> Response {
        let payload = (try? JSONEncoder().encode(["error": message])) ?? Data()
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: payload))
        )
    }
}

/// Pull-based file reader so large artifacts stream with backpressure
/// instead of buffering in memory. FileHandle is class-bound but this
/// sequence is consumed by exactly one response writer.
struct FileChunkSequence: AsyncSequence, @unchecked Sendable {
    typealias Element = ByteBuffer

    let handle: FileHandle

    struct AsyncIterator: AsyncIteratorProtocol {
        let handle: FileHandle

        mutating func next() async throws -> ByteBuffer? {
            let chunk = try handle.read(upToCount: 1 << 20)
            guard let chunk, !chunk.isEmpty else {
                try? handle.close()
                return nil
            }
            return ByteBuffer(bytes: chunk)
        }
    }

    func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(handle: handle)
    }
}
