import Foundation
import MereRunRelayKit

/// State machine behind `mere.run relay serve`: paired-device tokens, the
/// pairing window, the job spool, and a serial execution queue. The HTTP
/// layer in `RelayCommand` is a thin translation onto this actor; every
/// response it sends uses the same wire types the hosted relay serves, so
/// `RelayWorkflowExecutor` clients cannot tell the two apart.
actor LocalRelayState {
    struct JobRecord {
        let jobID: String
        var state: GraphRunState
        let createdAt: Date
        var updatedAt: Date
        var error: String?
        var graphName: String
        var expectedAssetDigests: Set<String>
    }

    struct DeviceRecord: Codable, Equatable {
        let name: String
        let tokenSHA256: String
        let createdAt: Date

        enum CodingKeys: String, CodingKey {
            case name
            case tokenSHA256 = "token_sha256"
            case createdAt = "created_at"
        }
    }

    struct DeviceFile: Codable {
        var schemaVersion: Int
        var devices: [DeviceRecord]

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case devices
        }
    }

    enum PairingOutcome {
        case paired(token: String)
        case rejected
        case closed
    }

    static let pairingAttemptLimit = 10

    let rootDirectory: URL
    let relayName: String
    private let pairingCode: String
    private let pairingDeadline: Date
    private var pairingAttempts = 0
    private var devices: [DeviceRecord]
    private var jobs: [String: JobRecord] = [:]
    private var queueContinuation: AsyncStream<String>.Continuation?
    private var runningJobID: String?
    private var cachedProbe: (probe: WorkflowExecutorProbe, at: Date)?
    private let probeProvider: @Sendable () -> WorkflowExecutorProbe

    init(
        rootDirectory: URL,
        relayName: String,
        pairingCode: String,
        pairingWindowMinutes: Int,
        probeProvider: @escaping @Sendable () -> WorkflowExecutorProbe = { WorkflowExecutorProbe.local() }
    ) throws {
        self.rootDirectory = rootDirectory
        self.relayName = relayName
        self.pairingCode = pairingCode
        self.probeProvider = probeProvider
        self.pairingDeadline = Date().addingTimeInterval(TimeInterval(pairingWindowMinutes) * 60)
        try FileManager.default.createDirectory(
            at: rootDirectory.appendingPathComponent("jobs", isDirectory: true),
            withIntermediateDirectories: true
        )
        let deviceURL = rootDirectory.appendingPathComponent("devices.json")
        if let data = try? Data(contentsOf: deviceURL),
           let file = try? WorkflowBundleCodec.decoder().decode(DeviceFile.self, from: data) {
            self.devices = file.devices
        } else {
            self.devices = []
        }
        self.jobs = Self.recoverSpooledJobs(rootDirectory: rootDirectory)
    }

    // MARK: - Pairing and authentication

    func pair(code: String, deviceName: String) throws -> PairingOutcome {
        guard Date() < pairingDeadline, pairingAttempts < Self.pairingAttemptLimit else {
            return .closed
        }
        let normalized = code.replacingOccurrences(of: "-", with: "").trimmingCharacters(in: .whitespaces)
        guard normalized == pairingCode else {
            pairingAttempts += 1
            return .rejected
        }
        var tokenBytes = [UInt8](repeating: 0, count: 32)
        var generator = SystemRandomNumberGenerator()
        for index in tokenBytes.indices {
            tokenBytes[index] = UInt8.random(in: .min ... .max, using: &generator)
        }
        let token = Data(tokenBytes).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
        let record = DeviceRecord(
            name: deviceName.isEmpty ? "device" : deviceName,
            tokenSHA256: ModelArtifactPinDigest.sha256(Data(token.utf8)),
            createdAt: Date()
        )
        devices.append(record)
        try persistDevices()
        return .paired(token: token)
    }

    func authorized(bearerToken: String?) -> Bool {
        guard let bearerToken, !bearerToken.isEmpty else { return false }
        let digest = ModelArtifactPinDigest.sha256(Data(bearerToken.utf8))
        return devices.contains { $0.tokenSHA256 == digest }
    }

    var pairedDeviceCount: Int { devices.count }

    private func persistDevices() throws {
        let file = DeviceFile(schemaVersion: 1, devices: devices)
        try WorkflowBundleCodec.write(file, to: rootDirectory.appendingPathComponent("devices.json"))
    }

    // MARK: - Probe

    func probe() -> WorkflowExecutorProbe {
        if let cachedProbe, Date().timeIntervalSince(cachedProbe.at) < 10 {
            return cachedProbe.probe
        }
        let fresh = probeProvider()
        cachedProbe = (fresh, Date())
        return fresh
    }

    // MARK: - Job lifecycle

    func bundleDirectory(jobID: String) -> URL {
        rootDirectory
            .appendingPathComponent("jobs", isDirectory: true)
            .appendingPathComponent(jobID, isDirectory: true)
            .appendingPathComponent("bundle", isDirectory: true)
    }

    func runDirectory(jobID: String) -> URL {
        rootDirectory
            .appendingPathComponent("jobs", isDirectory: true)
            .appendingPathComponent(jobID, isDirectory: true)
            .appendingPathComponent("run", isDirectory: true)
    }

    func create(request: RelayGraphCreateRequest) throws -> RelayGraphCreateResponse {
        let jobID = request.job.jobID
        guard jobs[jobID] == nil else {
            throw RelayClientError("A job with ID \(jobID) already exists on this relay.")
        }
        let bundleDir = bundleDirectory(jobID: jobID)
        try FileManager.default.createDirectory(at: bundleDir, withIntermediateDirectories: true)
        for (name, contents) in request.bundleDocuments {
            guard !name.contains("/"), !name.contains("..") else {
                throw RelayClientError("Bundle document names must be plain filenames: \(name)")
            }
            try contents.write(to: bundleDir.appendingPathComponent(name), options: .atomic)
        }
        let digests = Set(request.assets.groups.flatMap(\.entries).map(\.digest))
        jobs[jobID] = JobRecord(
            jobID: jobID,
            state: .planned,
            createdAt: Date(),
            updatedAt: Date(),
            error: nil,
            graphName: request.graph.name,
            expectedAssetDigests: digests
        )
        return RelayGraphCreateResponse(jobID: jobID, state: .planned, missingAssetDigests: digests.sorted())
    }

    func storeAsset(jobID: String, digest: String, data: Data) throws {
        guard let record = jobs[jobID], record.state == .planned else {
            throw RelayClientError("Job \(jobID) is not accepting assets.")
        }
        guard record.expectedAssetDigests.contains(digest) else {
            throw RelayClientError("Job \(jobID) does not expect asset \(digest).")
        }
        guard ModelArtifactPinDigest.sha256(data) == digest else {
            throw RelayClientError("Uploaded asset does not match digest \(digest).")
        }
        let assetDir = bundleDirectory(jobID: jobID)
            .appendingPathComponent("assets", isDirectory: true)
            .appendingPathComponent("sha256", isDirectory: true)
        try FileManager.default.createDirectory(at: assetDir, withIntermediateDirectories: true)
        try data.write(to: assetDir.appendingPathComponent(digest), options: .atomic)
    }

    func commit(jobID: String) throws -> JobRecord {
        guard var record = jobs[jobID] else {
            throw RelayClientError("Unknown job \(jobID).")
        }
        guard record.state == .planned else {
            throw RelayClientError("Job \(jobID) was already committed.")
        }
        let assetDir = bundleDirectory(jobID: jobID)
            .appendingPathComponent("assets", isDirectory: true)
            .appendingPathComponent("sha256", isDirectory: true)
        for digest in record.expectedAssetDigests {
            guard FileManager.default.fileExists(atPath: assetDir.appendingPathComponent(digest).path) else {
                throw RelayClientError("Job \(jobID) is missing asset \(digest); upload it before committing.")
            }
        }
        let job = try WorkflowBundleCodec.decoder().decode(
            WorkflowJobManifest.self,
            from: Data(contentsOf: bundleDirectory(jobID: jobID).appendingPathComponent(WorkflowJobManifest.filename))
        )
        try validateWorker(probe(), for: job, executor: "relay:\(relayName)", allowMixedBackend: true)
        record.state = .queued
        record.updatedAt = Date()
        jobs[jobID] = record
        queueContinuation?.yield(jobID)
        return record
    }

    func record(jobID: String) -> JobRecord? {
        jobs[jobID]
    }

    func list(limit: Int) -> [JobRecord] {
        Array(jobs.values.sorted { $0.createdAt > $1.createdAt }.prefix(max(1, limit)))
    }

    func cancel(jobID: String) throws -> JobRecord {
        guard var record = jobs[jobID] else {
            throw RelayClientError("Unknown job \(jobID).")
        }
        switch record.state {
        case .planned, .queued, .preflighting:
            record.state = .cancelled
            record.updatedAt = Date()
            jobs[jobID] = record
        case .running, .assigned:
            let runDir = runDirectory(jobID: jobID)
            try? Data(Date().ISO8601Format().utf8).write(
                to: runDir.appendingPathComponent("cancel.request"),
                options: .atomic
            )
            WorkflowChildProcessRegistry.terminateAll(in: runDir)
        case .finished, .failed, .cancelled:
            break
        }
        return jobs[jobID] ?? record
    }

    func retry(jobID: String) throws -> JobRecord {
        guard var record = jobs[jobID] else {
            throw RelayClientError("Unknown job \(jobID).")
        }
        guard record.state == .failed || record.state == .cancelled else {
            throw RelayClientError("Job \(jobID) is \(record.state.rawValue); only failed or cancelled jobs retry.")
        }
        try? FileManager.default.removeItem(at: runDirectory(jobID: jobID))
        record.state = .queued
        record.error = nil
        record.updatedAt = Date()
        jobs[jobID] = record
        queueContinuation?.yield(jobID)
        return record
    }

    /// The manifest served to clients, when the run has produced one.
    func manifestData(jobID: String) -> Data? {
        try? Data(contentsOf: runDirectory(jobID: jobID).appendingPathComponent(GraphRunManifest.filename))
    }

    func eventsText(jobID: String) -> String {
        guard let data = try? Data(contentsOf: runDirectory(jobID: jobID).appendingPathComponent("events.jsonl")) else {
            return ""
        }
        return String(decoding: data, as: UTF8.self)
    }

    func artifactURL(jobID: String, name: String) throws -> (url: URL, artifact: GraphRunArtifact) {
        guard let record = jobs[jobID], record.state == .finished else {
            throw RelayClientError("Job \(jobID) is not finished.")
        }
        guard let data = manifestData(jobID: jobID),
              let manifest = try? WorkflowBundleCodec.decoder().decode(GraphRunManifest.self, from: data),
              let artifact = manifest.outputs.first(where: { $0.name == name }) else {
            throw RelayClientError("Job \(jobID) has no artifact named \(name).")
        }
        let root = runDirectory(jobID: jobID).standardizedFileURL
        let candidate = root.appendingPathComponent(artifact.path).standardizedFileURL
        guard candidate.path.hasPrefix(root.path + "/") else {
            throw RelayClientError("Artifact path escapes the run directory.")
        }
        return (candidate, artifact)
    }

    func artifacts(jobID: String) -> [GraphRunArtifact] {
        guard let record = jobs[jobID], record.state == .finished,
              let data = manifestData(jobID: jobID),
              let manifest = try? WorkflowBundleCodec.decoder().decode(GraphRunManifest.self, from: data) else {
            return []
        }
        return manifest.outputs
    }

    // MARK: - Fleet

    func fleetSnapshot() -> RelayFleetSnapshot {
        let probe = probe()
        let busy = runningJobID != nil
        let queueDepth = jobs.values.filter { $0.state == .queued }.count
        let hostName = ProcessInfo.processInfo.hostName
        let node = RelayFleetNode(
            agentID: "local",
            deviceID: "local",
            deviceName: hostName,
            reportedName: hostName,
            version: probe.workerVersion,
            status: busy ? "busy" : "online",
            currentJobID: runningJobID,
            lastSeen: Date().ISO8601Format(),
            capabilities: RelayFleetCapabilities(
                models: probe.installedModelIDs,
                graphWorker: RelayFleetGraphWorker(
                    workerVersion: probe.workerVersion,
                    contractVersions: probe.contractVersions,
                    acceleratorBackend: probe.acceleratorBackend,
                    memoryBytes: probe.memoryBytes,
                    availableDiskBytes: probe.availableDiskBytes,
                    nodeKinds: probe.nodeKinds,
                    installedModelIDs: probe.installedModelIDs
                )
            ),
            runtime: RelayFleetRuntime(
                mereRunVersion: probe.workerVersion,
                installedModels: probe.installedModelIDs
            ),
            policy: RelayFleetPolicy(
                enabled: true,
                draining: false,
                revoked: false,
                priority: 0,
                preferredModels: [],
                displayName: relayName
            )
        )
        let activity = list(limit: 20).map { record in
            RelayFleetActivity(
                id: record.jobID,
                kind: "graph",
                status: record.state.rawValue,
                agentID: "local",
                model: nil,
                label: record.graphName,
                createdAt: record.createdAt.ISO8601Format(),
                durationMilliseconds: record.state == .finished || record.state == .failed
                    ? Int(record.updatedAt.timeIntervalSince(record.createdAt) * 1000)
                    : nil,
                error: record.error
            )
        }
        return RelayFleetSnapshot(
            generatedAt: Date().ISO8601Format(),
            summary: RelayFleetSummary(
                totalNodes: 1,
                onlineNodes: 1,
                busyNodes: busy ? 1 : 0,
                availableNodes: busy ? 0 : 1,
                queueDepth: queueDepth,
                installedModels: probe.installedModelIDs.count,
                routableModels: probe.installedModelIDs.count
            ),
            nodes: [node],
            activity: activity
        )
    }

    // MARK: - Execution queue

    /// One stream of committed job IDs; the serve command's run loop consumes
    /// it and executes strictly serially — the local lane models one machine.
    func makeQueue() -> AsyncStream<String> {
        AsyncStream { continuation in
            self.queueContinuation = continuation
            for record in jobs.values.sorted(by: { $0.createdAt < $1.createdAt }) where record.state == .queued {
                continuation.yield(record.jobID)
            }
        }
    }

    func beginRun(jobID: String) -> Bool {
        guard var record = jobs[jobID], record.state == .queued else { return false }
        record.state = .running
        record.updatedAt = Date()
        jobs[jobID] = record
        runningJobID = jobID
        return true
    }

    func finishRun(jobID: String, state: GraphRunState, error: String?) {
        guard var record = jobs[jobID] else { return }
        record.state = state
        record.error = error
        record.updatedAt = Date()
        jobs[jobID] = record
        runningJobID = nil
    }

    // MARK: - Startup recovery

    /// Jobs already on disk come back with the state their run manifest
    /// recorded; anything the previous process left mid-flight is failed
    /// honestly rather than silently re-run.
    private static func recoverSpooledJobs(rootDirectory: URL) -> [String: JobRecord] {
        var jobs: [String: JobRecord] = [:]
        let jobsDir = rootDirectory.appendingPathComponent("jobs", isDirectory: true)
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: jobsDir,
            includingPropertiesForKeys: [.creationDateKey]
        ) else { return jobs }
        for entry in entries {
            let jobID = entry.lastPathComponent
            let manifestURL = entry
                .appendingPathComponent("run", isDirectory: true)
                .appendingPathComponent(GraphRunManifest.filename)
            let bundleJobURL = entry
                .appendingPathComponent("bundle", isDirectory: true)
                .appendingPathComponent(WorkflowJobManifest.filename)
            var graphName = jobID
            if let graph = try? WorkflowGraphDocument.load(
                from: entry.appendingPathComponent("bundle", isDirectory: true).appendingPathComponent("graph.json")
            ) {
                graphName = graph.name
            }
            let createdAt = (try? entry.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date()
            if let data = try? Data(contentsOf: manifestURL),
               let manifest = try? WorkflowBundleCodec.decoder().decode(GraphRunManifest.self, from: data) {
                let interrupted = !(manifest.state == .finished || manifest.state == .failed || manifest.state == .cancelled)
                jobs[jobID] = JobRecord(
                    jobID: jobID,
                    state: interrupted ? .failed : manifest.state,
                    createdAt: manifest.createdAt,
                    updatedAt: manifest.updatedAt,
                    error: interrupted
                        ? "The local relay stopped before this job completed. Retry to run it again."
                        : manifest.error,
                    graphName: manifest.graphName,
                    expectedAssetDigests: []
                )
            } else if FileManager.default.fileExists(atPath: bundleJobURL.path) {
                jobs[jobID] = JobRecord(
                    jobID: jobID,
                    state: .failed,
                    createdAt: createdAt,
                    updatedAt: createdAt,
                    error: "The local relay stopped before this job started. Retry to run it again.",
                    graphName: graphName,
                    expectedAssetDigests: []
                )
            }
        }
        return jobs
    }
}

extension LocalRelayState.JobRecord {
    func response(artifacts: [GraphRunArtifact]) -> RelayGraphJobResponse {
        RelayGraphJobResponse(
            jobID: jobID,
            state: state,
            createdAt: createdAt,
            updatedAt: updatedAt,
            artifacts: artifacts,
            error: error,
            placement: nil,
            metrics: nil
        )
    }
}
