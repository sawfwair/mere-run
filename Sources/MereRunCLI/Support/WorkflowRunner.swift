import ArgumentParser
import MereRunRelayKit
import Foundation
import MereRunCore
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

struct WorkflowRunOutcome: Codable, Equatable {
    let runDirectory: String
    let jobID: String
    let state: GraphRunState
    let outputs: [GraphRunArtifact]

    enum CodingKeys: String, CodingKey {
        case runDirectory = "run_directory"
        case jobID = "job_id"
        case state
        case outputs
    }
}

private struct WorkflowPreparedParallelNode: @unchecked Sendable {
    let node: WorkflowNode
    let index: Int
    let directory: URL
    let invocation: WorkflowNodeInvocation
    let fingerprint: String
    let provider: WorkflowNodeProviderIdentity
    let models: [WorkflowModelProvenance]
    let maxAttempts: Int
}

private enum WorkflowParallelBufferedEvent {
    case provider(WorkflowPluginNodeEvent)
    case retrying(attempt: Int, message: String)
    case started(attempt: Int)
}

private struct WorkflowParallelNodeOutcome {
    let verified: WorkflowVerifiedNodeOutputs?
    let attempt: Int
    let exitStatus: Int32?
    let events: [WorkflowParallelBufferedEvent]
    let error: String?
    let cancelled: Bool
}

private final class WorkflowParallelOutcomeBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: WorkflowParallelNodeOutcome?

    func store(_ outcome: WorkflowParallelNodeOutcome) {
        lock.lock()
        stored = outcome
        lock.unlock()
    }

    func load() -> WorkflowParallelNodeOutcome? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}

struct WorkflowRunner: @unchecked Sendable {
    let bundleDirectory: URL
    let runDirectory: URL
    let resume: Bool
    let executor: GraphRunExecutorRecord
    let fileManager: FileManager
    let processRunner: any WorkflowProcessRunning
    let cacheDirectory: URL?
    let now: () -> Date
    let eventHandler: ((GraphRunEvent) -> Void)?
    let artifactStore: WorkflowArtifactStore
    let runStore: WorkflowRunStore

    init(
        bundleDirectory: URL,
        runDirectory: URL,
        resume: Bool = false,
        executor: GraphRunExecutorRecord = .init(kind: "local", profile: nil, jobReference: nil),
        fileManager: FileManager = .default,
        processRunner: any WorkflowProcessRunning = WorkflowProcessRunner(),
        cacheDirectory: URL? = nil,
        now: @escaping () -> Date = Date.init,
        eventHandler: ((GraphRunEvent) -> Void)? = nil
    ) {
        self.bundleDirectory = bundleDirectory.standardizedFileURL
        self.runDirectory = runDirectory.standardizedFileURL
        self.resume = resume
        self.executor = executor
        self.fileManager = fileManager
        self.processRunner = processRunner
        self.cacheDirectory = cacheDirectory?.standardizedFileURL
            ?? (processRunner is WorkflowProcessRunner
                ? MereRunModelPaths.applicationSupportBase
                    .appendingPathComponent("graph-cache/v1/nodes", isDirectory: true)
                : nil)
        self.now = now
        self.eventHandler = eventHandler
        self.artifactStore = WorkflowArtifactStore(
            bundleDirectory: self.bundleDirectory, runDirectory: self.runDirectory,
            cacheDirectory: self.cacheDirectory, resume: resume, fileManager: fileManager
        )
        self.runStore = WorkflowRunStore(
            bundleDirectory: self.bundleDirectory, runDirectory: self.runDirectory,
            resume: resume, executor: executor, fileManager: fileManager, now: now, eventHandler: eventHandler
        )
    }

    func execute() throws -> WorkflowRunOutcome {
        let graph = try WorkflowGraphDocument.load(from: bundleDirectory.appendingPathComponent("graph.json"))
        let inputs = try WorkflowInputsDocument.load(from: bundleDirectory.appendingPathComponent("inputs.json"))
        let assets = try WorkflowBundleCodec.decoder().decode(
            WorkflowAssetManifest.self,
            from: Data(contentsOf: bundleDirectory.appendingPathComponent(WorkflowAssetManifest.filename))
        )
        let job = try WorkflowBundleCodec.decoder().decode(
            WorkflowJobManifest.self,
            from: Data(contentsOf: bundleDirectory.appendingPathComponent(WorkflowJobManifest.filename))
        )
        guard job.contractVersion == WorkflowJobManifest.contractVersion else {
            throw ValidationError("Unsupported workflow job contract '\(job.contractVersion)'.")
        }
        guard assets.schemaVersion == 1 else {
            throw ValidationError("Unsupported workflow asset manifest version '\(assets.schemaVersion)'.")
        }
        guard try WorkflowBundleCodec.hash(graph) == job.graphFingerprint else {
            throw ValidationError("Workflow graph fingerprint does not match job.json.")
        }
        guard try WorkflowBundleCodec.hash(WorkflowPortableInputFingerprint(inputs: inputs, assets: assets)) == job.inputFingerprint else {
            throw ValidationError("Workflow input fingerprint does not match job.json.")
        }
        guard workflowVersion(MereRunCLIVersion.current, satisfiesMinimum: job.requirements.minimumMereRunVersion) else {
            throw ValidationError(
                "Workflow requires mere.run \(job.requirements.minimumMereRunVersion) or newer; this worker is \(MereRunCLIVersion.current)."
            )
        }
        let availableProviders = WorkflowGraphProviderRegistry.discoveredCatalog().providers.map(\.requirement)
        let missingProviders = job.requirements.providers.filter { !availableProviders.contains($0) }
        guard missingProviders.isEmpty else {
            throw ValidationError(
                "Worker is missing exact graph providers: \(missingProviders.map { "\($0.id)@\($0.version)" }.joined(separator: ", "))."
            )
        }
        let environment = ProcessInfo.processInfo.environment
        let missingSecrets = job.requirements.secretNames.filter {
            environment[workflowSecretEnvironmentKey($0)]?.isEmpty != false
        }
        guard missingSecrets.isEmpty else {
            throw ValidationError(
                "Worker is missing configured secrets: \(missingSecrets.joined(separator: ", "))."
            )
        }

        let validation = WorkflowGraphValidator.validate(graph: graph, inputs: inputs)
        guard validation.status != .blocked else {
            throw ValidationError(validation.diagnostics.map(\.message).joined(separator: " "))
        }

        let session = try runStore.prepare(graph: graph, job: job, order: validation.order)
        defer { session.lease.release() }
        let localizedInputs = try artifactStore.localizeInputs(graph: graph, inputs: inputs, assets: assets)
        var manifest = session.manifest
        var sequence = session.eventCount
        try runStore.persist(manifest)
        try runStore.record(
            GraphRunEvent(
                sequence: sequence,
                createdAt: now(),
                type: "run_started",
                state: .running,
                nodeID: nil,
                message: nil
            )
        )
        sequence += 1
        manifest.state = .running
        manifest.updatedAt = now()
        try runStore.persist(manifest)

        var nodeOutputs: [String: [String: WorkflowValue]] = [:]
        do {
            if graph.execution?.resolvedMaxParallelNodes ?? 1 > 1 {
                try executeParallelNodes(
                    graph: graph,
                    job: job,
                    validation: validation,
                    localizedInputs: localizedInputs,
                    manifest: &manifest,
                    sequence: &sequence,
                    nodeOutputs: &nodeOutputs
                )
            } else {
            for nodeID in validation.order {
                guard let node = graph.nodes.first(where: { $0.id == nodeID }),
                      let nodeIndex = manifest.nodes.firstIndex(where: { $0.id == nodeID }) else {
                    throw ValidationError("Workflow execution order referenced missing node '\(nodeID)'.")
                }
                if Task.isCancelled || fileManager.fileExists(atPath: runDirectory.appendingPathComponent("cancel.request").path) {
                    throw WorkflowCancellationError()
                }

                let nodeDirectory = runDirectory
                    .appendingPathComponent("nodes", isDirectory: true)
                    .appendingPathComponent(String(format: "%03d-%@", nodeIndex, node.id), isDirectory: true)
                try fileManager.createDirectory(at: nodeDirectory, withIntermediateDirectories: true)
                let resolvedArguments = try resolve(
                    node.arguments,
                    inputs: localizedInputs.values,
                    nodeOutputs: nodeOutputs
                )
                let referencedNodeIDs = Set(node.arguments.values.flatMap(\.references).compactMap { reference -> String? in
                    guard let parsed = try? WorkflowReference(reference),
                          case .nodeOutput(let sourceNodeID, _) = parsed.source else { return nil }
                    return sourceNodeID
                })
                let upstreamOutputs = manifest.nodes
                    .filter { referencedNodeIDs.contains($0.id) }
                    .flatMap(\.outputs)
                    .map(WorkflowNodeOutputFingerprint.init)
                    .sorted { ($0.sourceName, $0.sha256 ?? "") < ($1.sourceName, $1.sha256 ?? "") }
                guard let providerIdentity = WorkflowNodeRegistry.provider(for: node) else {
                    throw ValidationError("Workflow provider '\(node.resolvedProviderID)' is unavailable.")
                }
                let nodeModels = modelProvenance(for: node, arguments: resolvedArguments, job: job)
                let fingerprint = try WorkflowBundleCodec.hash(WorkflowNodeFingerprint(
                    kind: node.kind,
                    provider: providerIdentity,
                    arguments: resolvedArguments,
                    models: nodeModels,
                    upstreamOutputs: upstreamOutputs
                ))
                if workflowNodeAllowsResumeReuse(node), try artifactStore.shouldResume(
                    manifest.nodes[nodeIndex],
                    expectedFingerprint: fingerprint,
                    nodeOutputs: &nodeOutputs
                ) {
                    try runStore.record(.init(
                        sequence: sequence,
                        createdAt: now(),
                        type: "node_resumed",
                        state: .finished,
                        nodeID: nodeID,
                        message: "Reused verified node outputs."
                    ))
                    sequence += 1
                    continue
                }
                let invocation = try WorkflowNodeCommandBuilder.invocation(
                    node: node,
                    arguments: resolvedArguments,
                    nodeDirectory: nodeDirectory,
                    jobID: job.jobID
                )
                manifest.nodes[nodeIndex].fingerprint = fingerprint
                manifest.nodes[nodeIndex].provider = providerIdentity
                manifest.nodes[nodeIndex].models = nodeModels
                manifest.nodes[nodeIndex].maxAttempts = node.execution?.resolvedMaxAttempts ?? 1
                if (node.execution?.resolvedCache ?? .automatic) == .automatic,
                   let cached = try artifactStore.restoreCachedOutputs(
                    fingerprint: fingerprint,
                    invocation: invocation,
                    node: node,
                    nodeDirectory: nodeDirectory
                   ) {
                    nodeOutputs[nodeID] = cached.values
                    manifest.nodes[nodeIndex].attempt = 0
                    manifest.nodes[nodeIndex].artifacts = cached.artifacts
                    manifest.nodes[nodeIndex].outputs = cached.outputs
                    manifest.nodes[nodeIndex].state = .finished
                    manifest.nodes[nodeIndex].startedAt = now()
                    manifest.nodes[nodeIndex].completedAt = now()
                    manifest.updatedAt = now()
                    try runStore.persist(manifest)
                    try runStore.record(.init(
                        sequence: sequence,
                        createdAt: now(),
                        type: "node_cache_hit",
                        state: .finished,
                        nodeID: nodeID,
                        message: "Restored verified outputs for node fingerprint \(fingerprint)."
                    ))
                    sequence += 1
                    continue
                }
                manifest.nodes[nodeIndex].state = .preflighting
                manifest.nodes[nodeIndex].startedAt = now()
                manifest.updatedAt = now()
                try runStore.persist(manifest)
                try runStore.record(.init(
                    sequence: sequence,
                    createdAt: now(),
                    type: "node_preflight_started",
                    state: .preflighting,
                    nodeID: nodeID,
                    message: nil
                ))
                sequence += 1

                let preflight = try preflightInvocation(
                    invocation,
                    currentDirectory: nodeDirectory,
                    nodeID: nodeID
                )
                try throwIfCancellationRequested()
                try Data(preflight.stdout.utf8).write(
                    to: nodeDirectory.appendingPathComponent("preflight.json"),
                    options: .atomic
                )
                guard preflight.status == 0 else {
                    throw ValidationError("Node '\(nodeID)' preflight \(preflight.failureSummary).")
                }
                if invocation.streamsEvents {
                    let report = try WorkflowBundleCodec.decoder().decode(
                        WorkflowPluginNodePreflight.self,
                        from: Data(preflight.stdout.utf8)
                    )
                    guard report.contractVersion == WorkflowPluginNodePreflight.contractVersion,
                          report.status != "blocked" else {
                        throw ValidationError(
                            report.diagnostics.map(\.message).joined(separator: " ")
                        )
                    }
                }

                let maxAttempts = node.execution?.resolvedMaxAttempts ?? 1
                manifest.nodes[nodeIndex].attempt = 0
                manifest.nodes[nodeIndex].maxAttempts = maxAttempts
                var verifiedOutputs: WorkflowVerifiedNodeOutputs?
                while verifiedOutputs == nil {
                    manifest.nodes[nodeIndex].attempt += 1
                    let attempt = manifest.nodes[nodeIndex].attempt
                    manifest.nodes[nodeIndex].state = .running
                    manifest.nodes[nodeIndex].exitStatus = nil
                    manifest.nodes[nodeIndex].error = nil
                    manifest.updatedAt = now()
                    try runStore.persist(manifest)
                    try runStore.record(.init(
                        sequence: sequence,
                        createdAt: now(),
                        type: "node_started",
                        state: .running,
                        nodeID: nodeID,
                        message: "\(invocation.command.joined(separator: " ")) (attempt \(attempt)/\(maxAttempts))"
                    ))
                    sequence += 1

                    do {
                        var providerOutputs: [String: WorkflowValue]?
                        var providerSequence = -1
                        var deltaBytes = Data()
                        var lastDeltaAt = Date.distantPast
                        let execution = try executeInvocation(
                            invocation,
                            currentDirectory: nodeDirectory,
                            timeoutSeconds: node.execution?.timeoutSeconds,
                            stdoutLineHandler: invocation.streamsEvents ? { line in
                                let event = try WorkflowBundleCodec.decoder().decode(
                                    WorkflowPluginNodeEvent.self,
                                    from: Data(line.utf8)
                                )
                                guard event.contractVersion == WorkflowPluginNodeEvent.contractVersion,
                                      event.sequence == providerSequence + 1 else {
                                    throw ValidationError("Graph provider emitted an invalid event sequence for node '\(nodeID)'.")
                                }
                                providerSequence = event.sequence
                                if event.type == "node_result" {
                                    providerOutputs = event.outputs
                                } else {
                                    let runArtifact: GraphRunEventArtifact?
                                    if let eventArtifact = event.artifact {
                                        let artifactURL = try artifactStore.invocationOutputURL(
                                            eventArtifact.path,
                                            nodeDirectory: nodeDirectory
                                        )
                                        runArtifact = GraphRunEventArtifact(
                                            name: eventArtifact.name,
                                            path: try artifactStore.portableArtifactPath(for: artifactURL),
                                            contentType: eventArtifact.contentType
                                        )
                                    } else {
                                        runArtifact = nil
                                    }
                                    try runStore.record(event.runEvent(
                                        sequence: sequence,
                                        nodeID: nodeID,
                                        artifact: runArtifact
                                    ))
                                    sequence += 1
                                }
                            } : nil,
                            stdoutChunkHandler: invocation.streamsStdoutDeltas ? { chunk in
                                // Each delta carries the accumulated text so far, so
                                // downstream retention can coalesce to the latest
                                // event without losing content.
                                deltaBytes.append(chunk)
                                let stamp = now()
                                guard stamp.timeIntervalSince(lastDeltaAt) >= 0.3 else { return }
                                lastDeltaAt = stamp
                                try runStore.record(.init(
                                    sequence: sequence,
                                    createdAt: stamp,
                                    type: "node_output_delta",
                                    state: .running,
                                    nodeID: nodeID,
                                    message: String(decoding: deltaBytes, as: UTF8.self)
                                ))
                                sequence += 1
                            } : nil
                        )
                        let result = execution.result
                        if let intrinsicOutputs = execution.outputs {
                            providerOutputs = intrinsicOutputs
                        }
                        try throwIfCancellationRequested()
                        try Data(result.stdout.utf8).write(
                            to: nodeDirectory.appendingPathComponent("stdout.txt"),
                            options: .atomic
                        )
                        manifest.nodes[nodeIndex].exitStatus = result.status
                        guard result.status == 0 else {
                            throw ValidationError("Node '\(nodeID)' \(result.failureSummary).")
                        }
                        verifiedOutputs = try artifactStore.verifyOutputs(
                            invocation.outputs,
                            providerValues: providerOutputs,
                            node: node,
                            nodeDirectory: nodeDirectory
                        )
                    } catch is WorkflowCancellationError {
                        throw WorkflowCancellationError()
                    } catch {
                        let message = (error as? ValidationError)?.message ?? error.localizedDescription
                        manifest.nodes[nodeIndex].error = message
                        manifest.updatedAt = now()
                        try runStore.persist(manifest)
                        guard attempt < maxAttempts else { throw error }
                        try runStore.record(.init(
                            sequence: sequence,
                            createdAt: now(),
                            type: "node_retrying",
                            state: .running,
                            nodeID: nodeID,
                            message: "Attempt \(attempt) failed: \(message)"
                        ))
                        sequence += 1
                        try artifactStore.clearAttemptOutputs(invocation.outputs, nodeDirectory: nodeDirectory)
                    }
                }

                guard let verified = verifiedOutputs else {
                    throw ValidationError("Node '\(nodeID)' finished without verified outputs.")
                }
                nodeOutputs[nodeID] = verified.values
                manifest.nodes[nodeIndex].artifacts = verified.artifacts
                manifest.nodes[nodeIndex].outputs = verified.outputs
                if node.execution?.resolvedCache != .never {
                    do {
                        try artifactStore.storeCachedOutputs(
                            verified,
                            fingerprint: fingerprint,
                            policy: node.execution?.resolvedCache ?? .automatic,
                            nodeDirectory: nodeDirectory
                        )
                        if cacheDirectory != nil {
                            try runStore.record(.init(
                                sequence: sequence,
                                createdAt: now(),
                                type: "node_cache_stored",
                                state: .running,
                                nodeID: nodeID,
                                message: "Stored verified outputs for node fingerprint \(fingerprint)."
                            ))
                            sequence += 1
                        }
                    } catch {
                        try runStore.record(.init(
                            sequence: sequence,
                            createdAt: now(),
                            type: "node_cache_store_failed",
                            state: .running,
                            nodeID: nodeID,
                            message: error.localizedDescription
                        ))
                        sequence += 1
                    }
                }
                manifest.nodes[nodeIndex].state = .finished
                manifest.nodes[nodeIndex].completedAt = now()
                manifest.updatedAt = now()
                try runStore.persist(manifest)
                try runStore.record(.init(
                    sequence: sequence,
                    createdAt: now(),
                    type: "node_finished",
                    state: .finished,
                    nodeID: nodeID,
                    message: nil
                ))
                sequence += 1
            }
            }

            manifest.outputs = try artifactStore.materializeGraphOutputs(graph: graph, nodeOutputs: nodeOutputs)
            manifest.state = .finished
            manifest.updatedAt = now()
            try runStore.persist(manifest)
            try runStore.record(.init(
                sequence: sequence,
                createdAt: now(),
                type: "run_finished",
                state: .finished,
                nodeID: nil,
                message: nil
            ))
        } catch is WorkflowCancellationError {
            manifest.state = .cancelled
            manifest.error = "Workflow cancellation requested."
            manifest.updatedAt = now()
            try runStore.persist(manifest)
            try runStore.record(.init(
                sequence: sequence,
                createdAt: now(),
                type: "run_cancelled",
                state: .cancelled,
                nodeID: nil,
                message: manifest.error
            ))
        } catch {
            let message = (error as? ValidationError)?.message ?? error.localizedDescription
            if let running = manifest.nodes.firstIndex(where: { $0.state == .running || $0.state == .preflighting }) {
                manifest.nodes[running].state = .failed
                manifest.nodes[running].completedAt = now()
                manifest.nodes[running].error = message
            }
            manifest.state = .failed
            manifest.error = message
            manifest.updatedAt = now()
            try runStore.persist(manifest)
            try runStore.record(.init(
                sequence: sequence,
                createdAt: now(),
                type: "run_failed",
                state: .failed,
                nodeID: nil,
                message: message
            ))
        }

        return WorkflowRunOutcome(
            runDirectory: runDirectory.path,
            jobID: manifest.jobID,
            state: manifest.state,
            outputs: manifest.outputs
        )
    }

    private func executeParallelNodes(
        graph: WorkflowGraphDocument,
        job: WorkflowJobManifest,
        validation: WorkflowGraphValidation,
        localizedInputs: WorkflowInputsDocument,
        manifest: inout GraphRunManifest,
        sequence: inout Int,
        nodeOutputs: inout [String: [String: WorkflowValue]]
    ) throws {
        let maximumParallelNodes = graph.execution?.resolvedMaxParallelNodes ?? 1
        var pending = validation.order
        var completed = Set<String>()

        while !pending.isEmpty {
            try throwIfCancellationRequested()
            let ready = pending.filter { nodeID in
                validation.dependencies[nodeID, default: []].isSubset(of: completed)
            }
            guard !ready.isEmpty else {
                throw ValidationError("Workflow scheduler could not find a ready node.")
            }

            var prepared: [WorkflowPreparedParallelNode] = []
            for nodeID in ready.prefix(maximumParallelNodes) {
                guard let node = graph.nodes.first(where: { $0.id == nodeID }),
                      let nodeIndex = manifest.nodes.firstIndex(where: { $0.id == nodeID }) else {
                    throw ValidationError("Workflow execution order referenced missing node '\(nodeID)'.")
                }
                let nodeDirectory = runDirectory
                    .appendingPathComponent("nodes", isDirectory: true)
                    .appendingPathComponent(String(format: "%03d-%@", nodeIndex, node.id), isDirectory: true)
                try fileManager.createDirectory(at: nodeDirectory, withIntermediateDirectories: true)
                let resolvedArguments = try resolve(
                    node.arguments,
                    inputs: localizedInputs.values,
                    nodeOutputs: nodeOutputs
                )
                let referencedNodeIDs = Set(
                    node.arguments.values.flatMap(\.references).compactMap { reference -> String? in
                        guard let parsed = try? WorkflowReference(reference),
                              case .nodeOutput(let sourceNodeID, _) = parsed.source else { return nil }
                        return sourceNodeID
                    }
                )
                let upstreamOutputs = manifest.nodes
                    .filter { referencedNodeIDs.contains($0.id) }
                    .flatMap(\.outputs)
                    .map(WorkflowNodeOutputFingerprint.init)
                    .sorted { ($0.sourceName, $0.sha256 ?? "") < ($1.sourceName, $1.sha256 ?? "") }
                guard let provider = WorkflowNodeRegistry.provider(for: node) else {
                    throw ValidationError("Workflow provider '\(node.resolvedProviderID)' is unavailable.")
                }
                let models = modelProvenance(for: node, arguments: resolvedArguments, job: job)
                let fingerprint = try WorkflowBundleCodec.hash(WorkflowNodeFingerprint(
                    kind: node.kind,
                    provider: provider,
                    arguments: resolvedArguments,
                    models: models,
                    upstreamOutputs: upstreamOutputs
                ))
                if workflowNodeAllowsResumeReuse(node), try artifactStore.shouldResume(
                    manifest.nodes[nodeIndex],
                    expectedFingerprint: fingerprint,
                    nodeOutputs: &nodeOutputs
                ) {
                    try runStore.record(.init(
                        sequence: sequence,
                        createdAt: now(),
                        type: "node_resumed",
                        state: .finished,
                        nodeID: nodeID,
                        message: "Reused verified node outputs."
                    ))
                    sequence += 1
                    completed.insert(nodeID)
                    pending.removeAll { $0 == nodeID }
                    continue
                }

                let invocation = try WorkflowNodeCommandBuilder.invocation(
                    node: node,
                    arguments: resolvedArguments,
                    nodeDirectory: nodeDirectory,
                    jobID: job.jobID
                )
                manifest.nodes[nodeIndex].fingerprint = fingerprint
                manifest.nodes[nodeIndex].provider = provider
                manifest.nodes[nodeIndex].models = models
                manifest.nodes[nodeIndex].maxAttempts = node.execution?.resolvedMaxAttempts ?? 1
                if (node.execution?.resolvedCache ?? .automatic) == .automatic,
                   let cached = try artifactStore.restoreCachedOutputs(
                    fingerprint: fingerprint,
                    invocation: invocation,
                    node: node,
                    nodeDirectory: nodeDirectory
                   ) {
                    nodeOutputs[nodeID] = cached.values
                    manifest.nodes[nodeIndex].attempt = 0
                    manifest.nodes[nodeIndex].artifacts = cached.artifacts
                    manifest.nodes[nodeIndex].outputs = cached.outputs
                    manifest.nodes[nodeIndex].state = .finished
                    manifest.nodes[nodeIndex].startedAt = now()
                    manifest.nodes[nodeIndex].completedAt = now()
                    manifest.updatedAt = now()
                    try runStore.persist(manifest)
                    try runStore.record(.init(
                        sequence: sequence,
                        createdAt: now(),
                        type: "node_cache_hit",
                        state: .finished,
                        nodeID: nodeID,
                        message: "Restored verified outputs for node fingerprint \(fingerprint)."
                    ))
                    sequence += 1
                    completed.insert(nodeID)
                    pending.removeAll { $0 == nodeID }
                    continue
                }

                manifest.nodes[nodeIndex].state = .preflighting
                manifest.nodes[nodeIndex].startedAt = now()
                manifest.updatedAt = now()
                try runStore.persist(manifest)
                try runStore.record(.init(
                    sequence: sequence,
                    createdAt: now(),
                    type: "node_preflight_started",
                    state: .preflighting,
                    nodeID: nodeID,
                    message: nil
                ))
                sequence += 1
                let preflight = try preflightInvocation(
                    invocation,
                    currentDirectory: nodeDirectory,
                    nodeID: nodeID
                )
                try throwIfCancellationRequested()
                try Data(preflight.stdout.utf8).write(
                    to: nodeDirectory.appendingPathComponent("preflight.json"),
                    options: .atomic
                )
                guard preflight.status == 0 else {
                    throw ValidationError("Node '\(nodeID)' preflight \(preflight.failureSummary).")
                }
                if invocation.streamsEvents {
                    let report = try WorkflowBundleCodec.decoder().decode(
                        WorkflowPluginNodePreflight.self,
                        from: Data(preflight.stdout.utf8)
                    )
                    guard report.contractVersion == WorkflowPluginNodePreflight.contractVersion,
                          report.status != "blocked" else {
                        throw ValidationError(report.diagnostics.map(\.message).joined(separator: " "))
                    }
                }
                let maxAttempts = node.execution?.resolvedMaxAttempts ?? 1
                manifest.nodes[nodeIndex].attempt = 1
                manifest.nodes[nodeIndex].maxAttempts = maxAttempts
                manifest.nodes[nodeIndex].state = .running
                manifest.nodes[nodeIndex].error = nil
                manifest.nodes[nodeIndex].exitStatus = nil
                manifest.updatedAt = now()
                try runStore.persist(manifest)
                try runStore.record(.init(
                    sequence: sequence,
                    createdAt: now(),
                    type: "node_started",
                    state: .running,
                    nodeID: nodeID,
                    message: "\(invocation.command.joined(separator: " ")) (attempt 1/\(maxAttempts))"
                ))
                sequence += 1
                prepared.append(.init(
                    node: node,
                    index: nodeIndex,
                    directory: nodeDirectory,
                    invocation: invocation,
                    fingerprint: fingerprint,
                    provider: provider,
                    models: models,
                    maxAttempts: maxAttempts
                ))
            }

            guard !prepared.isEmpty else { continue }
            let queue = OperationQueue()
            queue.maxConcurrentOperationCount = maximumParallelNodes
            let boxes = prepared.map { item -> WorkflowParallelOutcomeBox in
                let box = WorkflowParallelOutcomeBox()
                queue.addOperation {
                    box.store(runParallelNode(item))
                }
                return box
            }
            queue.waitUntilAllOperationsAreFinished()

            var firstFailure: String?
            var wasCancelled = false
            for (item, box) in zip(prepared, boxes) {
                guard let outcome = box.load() else {
                    throw ValidationError("Parallel node '\(item.node.id)' did not return an outcome.")
                }
                manifest.nodes[item.index].attempt = outcome.attempt
                manifest.nodes[item.index].exitStatus = outcome.exitStatus
                for event in outcome.events {
                    switch event {
                    case .provider(let providerEvent):
                        let artifact: GraphRunEventArtifact?
                        if let eventArtifact = providerEvent.artifact {
                            let artifactURL = try artifactStore.invocationOutputURL(
                                eventArtifact.path,
                                nodeDirectory: item.directory
                            )
                            artifact = GraphRunEventArtifact(
                                name: eventArtifact.name,
                                path: try artifactStore.portableArtifactPath(for: artifactURL),
                                contentType: eventArtifact.contentType
                            )
                        } else {
                            artifact = nil
                        }
                        try runStore.record(providerEvent.runEvent(
                            sequence: sequence,
                            nodeID: item.node.id,
                            artifact: artifact
                        ))
                    case .retrying(let attempt, let message):
                        try runStore.record(.init(
                            sequence: sequence,
                            createdAt: now(),
                            type: "node_retrying",
                            state: .running,
                            nodeID: item.node.id,
                            message: "Attempt \(attempt) failed: \(message)"
                        ))
                    case .started(let attempt):
                        try runStore.record(.init(
                            sequence: sequence,
                            createdAt: now(),
                            type: "node_started",
                            state: .running,
                            nodeID: item.node.id,
                            message: "\(item.invocation.command.joined(separator: " ")) (attempt \(attempt)/\(item.maxAttempts))"
                        ))
                    }
                    sequence += 1
                }

                guard let verified = outcome.verified else {
                    let message = outcome.error ?? "Node '\(item.node.id)' failed without a diagnostic."
                    manifest.nodes[item.index].state = outcome.cancelled ? .cancelled : .failed
                    manifest.nodes[item.index].error = message
                    manifest.nodes[item.index].completedAt = now()
                    firstFailure = firstFailure ?? message
                    wasCancelled = wasCancelled || outcome.cancelled
                    continue
                }
                nodeOutputs[item.node.id] = verified.values
                manifest.nodes[item.index].artifacts = verified.artifacts
                manifest.nodes[item.index].outputs = verified.outputs
                if item.node.execution?.resolvedCache != .never {
                    do {
                        try artifactStore.storeCachedOutputs(
                            verified,
                            fingerprint: item.fingerprint,
                            policy: item.node.execution?.resolvedCache ?? .automatic,
                            nodeDirectory: item.directory
                        )
                        if cacheDirectory != nil {
                            try runStore.record(.init(
                                sequence: sequence,
                                createdAt: now(),
                                type: "node_cache_stored",
                                state: .running,
                                nodeID: item.node.id,
                                message: "Stored verified outputs for node fingerprint \(item.fingerprint)."
                            ))
                            sequence += 1
                        }
                    } catch {
                        try runStore.record(.init(
                            sequence: sequence,
                            createdAt: now(),
                            type: "node_cache_store_failed",
                            state: .running,
                            nodeID: item.node.id,
                            message: error.localizedDescription
                        ))
                        sequence += 1
                    }
                }
                manifest.nodes[item.index].state = .finished
                manifest.nodes[item.index].completedAt = now()
                manifest.nodes[item.index].error = nil
                completed.insert(item.node.id)
                pending.removeAll { $0 == item.node.id }
                try runStore.record(.init(
                    sequence: sequence,
                    createdAt: now(),
                    type: "node_finished",
                    state: .finished,
                    nodeID: item.node.id,
                    message: nil
                ))
                sequence += 1
            }
            manifest.updatedAt = now()
            try runStore.persist(manifest)
            if wasCancelled { throw WorkflowCancellationError() }
            if let firstFailure { throw ValidationError(firstFailure) }
        }
    }

    private func runParallelNode(
        _ prepared: WorkflowPreparedParallelNode
    ) -> WorkflowParallelNodeOutcome {
        var attempt = 0
        var exitStatus: Int32?
        var events: [WorkflowParallelBufferedEvent] = []
        while attempt < prepared.maxAttempts {
            attempt += 1
            do {
                var providerOutputs: [String: WorkflowValue]?
                var providerSequence = -1
                let execution = try executeInvocation(
                    prepared.invocation,
                    currentDirectory: prepared.directory,
                    timeoutSeconds: prepared.node.execution?.timeoutSeconds,
                    stdoutLineHandler: prepared.invocation.streamsEvents ? { line in
                        let event = try WorkflowBundleCodec.decoder().decode(
                            WorkflowPluginNodeEvent.self,
                            from: Data(line.utf8)
                        )
                        guard event.contractVersion == WorkflowPluginNodeEvent.contractVersion,
                              event.sequence == providerSequence + 1 else {
                            throw ValidationError(
                                "Graph provider emitted an invalid event sequence for node '\(prepared.node.id)'."
                            )
                        }
                        providerSequence = event.sequence
                        if event.type == "node_result" {
                            providerOutputs = event.outputs
                        } else {
                            events.append(.provider(event))
                        }
                    } : nil
                )
                let result = execution.result
                if let intrinsicOutputs = execution.outputs {
                    providerOutputs = intrinsicOutputs
                }
                try throwIfCancellationRequested()
                try Data(result.stdout.utf8).write(
                    to: prepared.directory.appendingPathComponent("stdout.txt"),
                    options: .atomic
                )
                exitStatus = result.status
                guard result.status == 0 else {
                    throw ValidationError("Node '\(prepared.node.id)' \(result.failureSummary).")
                }
                let verified = try artifactStore.verifyOutputs(
                    prepared.invocation.outputs,
                    providerValues: providerOutputs,
                    node: prepared.node,
                    nodeDirectory: prepared.directory
                )
                return .init(
                    verified: verified,
                    attempt: attempt,
                    exitStatus: exitStatus,
                    events: events,
                    error: nil,
                    cancelled: false
                )
            } catch is WorkflowCancellationError {
                return .init(
                    verified: nil,
                    attempt: attempt,
                    exitStatus: exitStatus,
                    events: events,
                    error: "Workflow cancellation requested.",
                    cancelled: true
                )
            } catch {
                let message = (error as? ValidationError)?.message ?? error.localizedDescription
                guard attempt < prepared.maxAttempts else {
                    return .init(
                        verified: nil,
                        attempt: attempt,
                        exitStatus: exitStatus,
                        events: events,
                        error: message,
                        cancelled: false
                    )
                }
                events.append(.retrying(attempt: attempt, message: message))
                events.append(.started(attempt: attempt + 1))
                do {
                    try artifactStore.clearAttemptOutputs(
                        prepared.invocation.outputs,
                        nodeDirectory: prepared.directory
                    )
                } catch {
                    return .init(
                        verified: nil,
                        attempt: attempt,
                        exitStatus: exitStatus,
                        events: events,
                        error: error.localizedDescription,
                        cancelled: false
                    )
                }
            }
        }
        return .init(
            verified: nil,
            attempt: attempt,
            exitStatus: exitStatus,
            events: events,
            error: "Node '\(prepared.node.id)' exhausted its retry policy.",
            cancelled: false
        )
    }

    private func preflightInvocation(
        _ invocation: WorkflowNodeInvocation,
        currentDirectory: URL,
        nodeID: String
    ) throws -> WorkflowProcessResult {
        if invocation.intrinsic != nil {
            let report = ["node_id": nodeID, "status": "ready"]
            let stdout = String(
                decoding: try WorkflowBundleCodec.lineEncoder().encode(report),
                as: UTF8.self
            ) + "\n"
            return WorkflowProcessResult(status: 0, stdout: stdout)
        }
        return try runProcess(
            executable: invocation.executable,
            arguments: invocation.preflightArguments,
            currentDirectory: currentDirectory,
            timeoutSeconds: nil,
            stdoutLineHandler: nil
        )
    }

    private func executeInvocation(
        _ invocation: WorkflowNodeInvocation,
        currentDirectory: URL,
        timeoutSeconds: Int?,
        stdoutLineHandler: ((String) throws -> Void)?,
        stdoutChunkHandler: ((Data) throws -> Void)? = nil
    ) throws -> (result: WorkflowProcessResult, outputs: [String: WorkflowValue]?) {
        if let intrinsic = invocation.intrinsic {
            try throwIfCancellationRequested()
            let outputs = try intrinsic.evaluate()
            let stdout = String(
                decoding: try WorkflowBundleCodec.lineEncoder().encode(outputs),
                as: UTF8.self
            ) + "\n"
            return (WorkflowProcessResult(status: 0, stdout: stdout), outputs)
        }
        let result = try runProcess(
            executable: invocation.executable,
            arguments: invocation.runArguments,
            currentDirectory: currentDirectory,
            timeoutSeconds: timeoutSeconds,
            stdoutLineHandler: stdoutLineHandler,
            stdoutChunkHandler: stdoutChunkHandler
        )
        guard result.status == 0, let outputName = invocation.stdoutOutputName else {
            return (result, nil)
        }
        var scalar = result.stdout
        if invocation.streamsStdoutDeltas {
            // Streamed generation emits reasoning inline; the stored value is
            // the clean reply, matching the non-streamed path's contract.
            scalar = GeneratedTextFilters.strippingThinking(scalar)
        }
        if scalar.hasSuffix("\r\n") {
            scalar.removeLast(2)
        } else if scalar.hasSuffix("\n") {
            scalar.removeLast()
        }
        return (result, [outputName: .string(scalar)])
    }

    private func runProcess(
        executable: URL,
        arguments: [String],
        currentDirectory: URL,
        timeoutSeconds: Int?,
        stdoutLineHandler: ((String) throws -> Void)?,
        stdoutChunkHandler: ((Data) throws -> Void)? = nil
    ) throws -> WorkflowProcessResult {
        if let streamingRunner = processRunner as? any WorkflowStreamingProcessRunning {
            return try streamingRunner.run(
                executable: executable,
                arguments: arguments,
                currentDirectory: currentDirectory,
                timeoutSeconds: timeoutSeconds,
                stdoutLineHandler: stdoutLineHandler,
                stdoutChunkHandler: stdoutChunkHandler
            )
        }
        if timeoutSeconds != nil {
            throw ValidationError("Injected workflow process runner does not support execution timeouts.")
        }
        guard executable.standardizedFileURL == CurrentExecutable.url().standardizedFileURL else {
            throw ValidationError("Injected workflow process runner cannot execute companion providers.")
        }
        let result = try processRunner.run(arguments: arguments, currentDirectory: currentDirectory)
        if let stdoutLineHandler {
            for line in result.stdout.split(whereSeparator: \Character.isNewline) {
                try stdoutLineHandler(String(line))
            }
        }
        return result
    }

    private func modelProvenance(
        for node: WorkflowNode,
        arguments: [String: WorkflowValue],
        job: WorkflowJobManifest
    ) -> [WorkflowModelProvenance] {
        var modelIDs = Set(WorkflowNodeRegistry.entry(for: node)?.requirements.modelIDs ?? [])
        if let modelID = arguments["model"]?.stringValue {
            modelIDs.insert(modelID)
        } else {
            switch node.kind {
            case "image.train-lora": modelIDs.insert(ImageTrainLoRA.defaultManagedModelID.rawValue)
            case "image.generate": modelIDs.insert(ImageGenerate.defaultManagedModelID.rawValue)
            case "video.generate": modelIDs.insert(ModelResolver.ModelID.ltxVideo23AVMLX.rawValue)
            default: break
            }
        }
        return job.requirements.models.filter { modelIDs.contains($0.id) }.map { provenance in
            let modelRoot = ModelResolver.ModelID(rawValue: provenance.id)
                .flatMap { ModelResolver(fileManager: fileManager).resolveIfPresent($0)?.rootURL }
                ?? MereRunModelPaths.modelDir(provenance.id)
            let manifestURL = modelRoot
                .appendingPathComponent(MereRunModelManifest.filename)
            let manifestDigest = fileManager.fileExists(atPath: manifestURL.path)
                ? try? ModelArtifactPin.fileSHA256(manifestURL)
                : nil
            return WorkflowModelProvenance(
                id: provenance.id,
                repository: provenance.repository,
                revision: provenance.revision,
                catalogSHA256: provenance.catalogSHA256,
                installManifestSHA256: manifestDigest
            )
        }.sorted { $0.id < $1.id }
    }

    private func resolve(
        _ arguments: [String: WorkflowValue],
        inputs: [String: WorkflowValue],
        nodeOutputs: [String: [String: WorkflowValue]]
    ) throws -> [String: WorkflowValue] {
        var resolved: [String: WorkflowValue] = [:]
        for key in arguments.keys.sorted() {
            guard let value = arguments[key] else { continue }
            resolved[key] = try resolve(value, inputs: inputs, nodeOutputs: nodeOutputs)
        }
        return resolved
    }

    private func resolve(
        _ value: WorkflowValue,
        inputs: [String: WorkflowValue],
        nodeOutputs: [String: [String: WorkflowValue]]
    ) throws -> WorkflowValue {
        switch value {
        case .reference(let rawReference):
            let reference = try WorkflowReference(rawReference)
            switch reference.source {
            case .input(let name):
                guard let value = inputs[name] else {
                    throw ValidationError("Workflow input '\(name)' was not materialized.")
                }
                return value
            case .nodeOutput(let nodeID, let output):
                guard let value = nodeOutputs[nodeID]?[output] else {
                    throw ValidationError("Workflow node output '\(rawReference)' is not available.")
                }
                return value
            }
        case .array(let values):
            return .array(try values.map { try resolve($0, inputs: inputs, nodeOutputs: nodeOutputs) })
        case .object(let values):
            var resolved: [String: WorkflowValue] = [:]
            for key in values.keys.sorted() {
                guard let value = values[key] else { continue }
                resolved[key] = try resolve(value, inputs: inputs, nodeOutputs: nodeOutputs)
            }
            return .object(resolved)
        case .string(let rawValue) where rawValue.hasPrefix("asset://"):
            let name = String(rawValue.dropFirst("asset://".count))
            guard let localized = inputs[name] else {
                throw ValidationError("Workflow asset '\(name)' was not materialized.")
            }
            return localized
        default:
            return value
        }
    }

    private func throwIfCancellationRequested() throws {
        if Task.isCancelled || fileManager.fileExists(atPath: runDirectory.appendingPathComponent("cancel.request").path) {
            throw WorkflowCancellationError()
        }
    }


}

private struct WorkflowNodeFingerprint: Codable {
    let kind: String
    let provider: WorkflowNodeProviderIdentity
    let arguments: [String: WorkflowValue]
    let models: [WorkflowModelProvenance]
    let upstreamOutputs: [WorkflowNodeOutputFingerprint]

    enum CodingKeys: String, CodingKey {
        case kind
        case provider
        case arguments
        case models
        case upstreamOutputs = "upstream_outputs"
    }
}

private struct WorkflowNodeOutputFingerprint: Codable {
    let sourceName: String
    let type: WorkflowPortType
    let value: WorkflowValue?
    let sha256: String?

    init(_ output: GraphRunNodeOutput) {
        sourceName = output.name
        type = output.type
        value = output.value
        sha256 = output.sha256
    }

    enum CodingKeys: String, CodingKey {
        case sourceName = "source_name"
        case type
        case value
        case sha256
    }
}

private struct WorkflowPluginNodePreflight: Codable {
    static let contractVersion = "mere.run/plugin-graph-preflight.v1"

    let contractVersion: String
    let status: String
    let diagnostics: [GraphRunEventDiagnostic]

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case status
        case diagnostics
    }
}

private struct WorkflowPluginNodeEvent: Codable {
    static let contractVersion = "mere.run/plugin-graph-event.v1"

    let contractVersion: String
    let sequence: Int
    let createdAt: String
    let type: String
    let message: String?
    let progress: GraphRunProgress?
    let artifact: GraphRunEventArtifact?
    let diagnostic: GraphRunEventDiagnostic?
    let metric: GraphRunMetric?
    let outputs: [String: WorkflowValue]?

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case sequence
        case createdAt = "created_at"
        case type
        case message
        case progress
        case artifact
        case diagnostic
        case metric
        case outputs
    }

    func runEvent(
        sequence runSequence: Int,
        nodeID: String,
        artifact runArtifact: GraphRunEventArtifact?
    ) -> GraphRunEvent {
        GraphRunEvent(
            sequence: runSequence,
            createdAt: Date(),
            type: runEventType,
            state: .running,
            nodeID: nodeID,
            message: message,
            progress: progress,
            artifact: runArtifact,
            diagnostic: diagnostic,
            metric: metric
        )
    }

    private var runEventType: String {
        switch type {
        case "progress": "node_progress"
        case "diagnostic": "node_diagnostic"
        case "metric": "node_metric"
        case "heartbeat": "node_heartbeat"
        default: type
        }
    }
}

func workflowNodeAllowsResumeReuse(_ node: WorkflowNode) -> Bool {
    node.arguments.values.allSatisfy { $0.secretNames.isEmpty }
}

struct WorkflowCancellationError: Error {}
