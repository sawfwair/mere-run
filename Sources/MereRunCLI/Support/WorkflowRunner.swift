import ArgumentParser
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

struct WorkflowProcessResult: Equatable {
    let status: Int32
    let stdout: String
    let stderr: String
    let terminationReason: WorkflowProcessTerminationReason

    init(
        status: Int32,
        stdout: String,
        stderr: String = "",
        terminationReason: WorkflowProcessTerminationReason = .exit
    ) {
        self.status = status
        self.stdout = stdout
        self.stderr = stderr
        self.terminationReason = terminationReason
    }

    var failureSummary: String {
        var summary = terminationReason == .uncaughtSignal
            ? "terminated by signal \(status)"
            : "exited with status \(status)"
        if !stderr.isEmpty {
            summary += ". stderr: \(stderr)"
        } else if !stdout.isEmpty {
            summary += ". stdout: \(stdout.suffix(16 * 1_024))"
        }
        return summary
    }
}

enum WorkflowProcessTerminationReason: String, Equatable {
    case exit
    case uncaughtSignal = "uncaught_signal"

    init(_ reason: Process.TerminationReason) {
        switch reason {
        case .exit:
            self = .exit
        case .uncaughtSignal:
            self = .uncaughtSignal
        @unknown default:
            self = .exit
        }
    }
}

private final class WorkflowProcessStderrTail: @unchecked Sendable {
    private static let maximumBytes = 16 * 1_024
    private let lock = NSLock()
    private var data = Data()

    func append(_ newData: Data) {
        lock.lock()
        defer { lock.unlock() }
        if newData.count >= Self.maximumBytes {
            data = Data(newData.suffix(Self.maximumBytes))
            return
        }
        data.append(newData)
        if data.count > Self.maximumBytes {
            data.removeFirst(data.count - Self.maximumBytes)
        }
    }

    var string: String {
        lock.lock()
        defer { lock.unlock() }
        return String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

protocol WorkflowProcessRunning {
    func run(arguments: [String], currentDirectory: URL) throws -> WorkflowProcessResult
}

protocol WorkflowStreamingProcessRunning {
    func run(
        executable: URL,
        arguments: [String],
        currentDirectory: URL,
        stdoutLineHandler: ((String) throws -> Void)?
    ) throws -> WorkflowProcessResult
}

struct WorkflowProcessRunner: WorkflowProcessRunning, WorkflowStreamingProcessRunning {
    static func stdoutCaptureURL(in directory: URL) -> URL {
        directory.appendingPathComponent(".workflow-stdout-\(UUID().uuidString)")
    }

    func run(arguments: [String], currentDirectory: URL) throws -> WorkflowProcessResult {
        try run(
            executable: CurrentExecutable.url(),
            arguments: arguments,
            currentDirectory: currentDirectory,
            stdoutLineHandler: nil
        )
    }

    func run(
        executable: URL,
        arguments: [String],
        currentDirectory: URL,
        stdoutLineHandler: ((String) throws -> Void)?
    ) throws -> WorkflowProcessResult {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectory
        let stdout = Pipe()
        let stderr = Pipe()
        let stderrTail = WorkflowProcessStderrTail()
        let runDirectory = currentDirectory.deletingLastPathComponent().deletingLastPathComponent()
        let processIDURL = runDirectory.appendingPathComponent("worker-child.pid")
        defer {
            try? FileManager.default.removeItem(at: processIDURL)
        }
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = stdout
        process.standardError = stderr
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            stderrTail.append(data)
            try? FileHandle.standardError.write(contentsOf: data)
        }
        try process.run()
        try Data(String(process.processIdentifier).utf8).write(to: processIDURL, options: .atomic)
        var captured = Data()
        var pending = Data()
        while true {
            let data = stdout.fileHandleForReading.availableData
            if data.isEmpty { break }
            captured.append(data)
            guard stdoutLineHandler != nil else { continue }
            pending.append(data)
            while let newline = pending.firstIndex(of: 0x0A) {
                let lineData = pending.prefix(upTo: newline)
                pending.removeSubrange(...newline)
                let line = String(decoding: lineData, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !line.isEmpty { try stdoutLineHandler?(line) }
            }
        }
        process.waitUntilExit()
        stderr.fileHandleForReading.readabilityHandler = nil
        let remainingStderr = stderr.fileHandleForReading.readDataToEndOfFile()
        if !remainingStderr.isEmpty {
            stderrTail.append(remainingStderr)
            try? FileHandle.standardError.write(contentsOf: remainingStderr)
        }
        if !pending.isEmpty {
            let line = String(decoding: pending, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !line.isEmpty { try stdoutLineHandler?(line) }
        }
        if FileManager.default.fileExists(atPath: runDirectory.appendingPathComponent("cancel.request").path) {
            throw WorkflowCancellationError()
        }
        return WorkflowProcessResult(
            status: process.terminationStatus,
            stdout: String(decoding: captured, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines),
            stderr: stderrTail.string,
            terminationReason: WorkflowProcessTerminationReason(process.terminationReason)
        )
    }
}

struct WorkflowRunner {
    let bundleDirectory: URL
    let runDirectory: URL
    let resume: Bool
    let executor: GraphRunExecutorRecord
    let fileManager: FileManager
    let processRunner: any WorkflowProcessRunning
    let now: () -> Date
    let eventHandler: ((GraphRunEvent) -> Void)?

    init(
        bundleDirectory: URL,
        runDirectory: URL,
        resume: Bool = false,
        executor: GraphRunExecutorRecord = .init(kind: "local", profile: nil, jobReference: nil),
        fileManager: FileManager = .default,
        processRunner: any WorkflowProcessRunning = WorkflowProcessRunner(),
        now: @escaping () -> Date = Date.init,
        eventHandler: ((GraphRunEvent) -> Void)? = nil
    ) {
        self.bundleDirectory = bundleDirectory.standardizedFileURL
        self.runDirectory = runDirectory.standardizedFileURL
        self.resume = resume
        self.executor = executor
        self.fileManager = fileManager
        self.processRunner = processRunner
        self.now = now
        self.eventHandler = eventHandler
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

        let validation = WorkflowGraphValidator.validate(graph: graph, inputs: inputs)
        guard validation.status != .blocked else {
            throw ValidationError(validation.diagnostics.map(\.message).joined(separator: " "))
        }

        try prepareRunDirectory()
        let localizedInputs = try localizeInputs(graph: graph, inputs: inputs, assets: assets)
        var manifest = try initialManifest(graph: graph, job: job, order: validation.order)
        var sequence = try existingEventCount()
        try persist(manifest)
        try record(
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
        try persist(manifest)

        var nodeOutputs: [String: [String: WorkflowValue]] = [:]
        do {
            for nodeID in validation.order {
                guard let node = graph.nodes.first(where: { $0.id == nodeID }),
                      let nodeIndex = manifest.nodes.firstIndex(where: { $0.id == nodeID }) else {
                    throw ValidationError("Workflow execution order referenced missing node '\(nodeID)'.")
                }
                if fileManager.fileExists(atPath: runDirectory.appendingPathComponent("cancel.request").path) {
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
                if try shouldResume(
                    manifest.nodes[nodeIndex],
                    expectedFingerprint: fingerprint,
                    nodeOutputs: &nodeOutputs
                ) {
                    try record(.init(
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
                manifest.nodes[nodeIndex].state = .preflighting
                manifest.nodes[nodeIndex].startedAt = now()
                manifest.updatedAt = now()
                try persist(manifest)
                try record(.init(
                    sequence: sequence,
                    createdAt: now(),
                    type: "node_preflight_started",
                    state: .preflighting,
                    nodeID: nodeID,
                    message: nil
                ))
                sequence += 1

                let preflight = try runProcess(
                    executable: invocation.executable,
                    arguments: invocation.preflightArguments,
                    currentDirectory: nodeDirectory,
                    stdoutLineHandler: nil
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

                manifest.nodes[nodeIndex].state = .running
                manifest.updatedAt = now()
                try persist(manifest)
                try record(.init(
                    sequence: sequence,
                    createdAt: now(),
                    type: "node_started",
                    state: .running,
                    nodeID: nodeID,
                    message: invocation.command.joined(separator: " ")
                ))
                sequence += 1

                var providerOutputs: [String: WorkflowValue]?
                var providerSequence = -1
                let result = try runProcess(
                    executable: invocation.executable,
                    arguments: invocation.runArguments,
                    currentDirectory: nodeDirectory,
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
                                let artifactURL = try invocationOutputURL(
                                    eventArtifact.path,
                                    nodeDirectory: nodeDirectory
                                )
                                runArtifact = GraphRunEventArtifact(
                                    name: eventArtifact.name,
                                    path: try portableArtifactPath(for: artifactURL),
                                    contentType: eventArtifact.contentType
                                )
                            } else {
                                runArtifact = nil
                            }
                            try record(event.runEvent(
                                sequence: sequence,
                                nodeID: nodeID,
                                artifact: runArtifact
                            ))
                            sequence += 1
                        }
                    } : nil
                )
                try throwIfCancellationRequested()
                try Data(result.stdout.utf8).write(
                    to: nodeDirectory.appendingPathComponent("stdout.txt"),
                    options: .atomic
                )
                manifest.nodes[nodeIndex].exitStatus = result.status
                guard result.status == 0 else {
                    throw ValidationError("Node '\(nodeID)' failed with status \(result.status).")
                }

                let verified = try verifyOutputs(
                    invocation.outputs,
                    providerValues: providerOutputs,
                    node: node,
                    nodeDirectory: nodeDirectory
                )
                nodeOutputs[nodeID] = verified.values
                manifest.nodes[nodeIndex].artifacts = verified.artifacts
                manifest.nodes[nodeIndex].outputs = verified.outputs
                manifest.nodes[nodeIndex].state = .finished
                manifest.nodes[nodeIndex].completedAt = now()
                manifest.updatedAt = now()
                try persist(manifest)
                try record(.init(
                    sequence: sequence,
                    createdAt: now(),
                    type: "node_finished",
                    state: .finished,
                    nodeID: nodeID,
                    message: nil
                ))
                sequence += 1
            }

            manifest.outputs = try materializeGraphOutputs(graph: graph, nodeOutputs: nodeOutputs)
            manifest.state = .finished
            manifest.updatedAt = now()
            try persist(manifest)
            try record(.init(
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
            try persist(manifest)
            try record(.init(
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
            try persist(manifest)
            try record(.init(
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

    private func runProcess(
        executable: URL,
        arguments: [String],
        currentDirectory: URL,
        stdoutLineHandler: ((String) throws -> Void)?
    ) throws -> WorkflowProcessResult {
        if let streamingRunner = processRunner as? any WorkflowStreamingProcessRunning {
            return try streamingRunner.run(
                executable: executable,
                arguments: arguments,
                currentDirectory: currentDirectory,
                stdoutLineHandler: stdoutLineHandler
            )
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

    private func verifyOutputs(
        _ descriptors: [String: WorkflowInvocationOutput],
        providerValues: [String: WorkflowValue]?,
        node: WorkflowNode,
        nodeDirectory: URL
    ) throws -> WorkflowVerifiedNodeOutputs {
        var artifacts: [GraphRunArtifact] = []
        var records: [GraphRunNodeOutput] = []
        var values: [String: WorkflowValue] = [:]
        for name in descriptors.keys.sorted() {
            guard let descriptor = descriptors[name] else { continue }
            let providerValue = providerValues?[name]
            switch descriptor.type {
            case .asset, .assetCollection, .assetArray:
                guard let path = descriptor.path else {
                    throw ValidationError("Node '\(node.id)' output '\(name)' has no declared path.")
                }
                let url = try invocationOutputURL(path, nodeDirectory: nodeDirectory)
                if providerValue == .null || !fileManager.fileExists(atPath: url.path) {
                    if descriptor.optional { continue }
                    throw ValidationError("Node '\(node.id)' did not produce declared output '\(name)'.")
                }
                if let providerPath = providerValue?.stringValue {
                    let reported = try invocationOutputURL(providerPath, nodeDirectory: nodeDirectory)
                    guard reported.standardizedFileURL == url.standardizedFileURL else {
                        throw ValidationError("Node '\(node.id)' reported an unexpected path for output '\(name)'.")
                    }
                }
                let outputArtifact = try artifact(
                    name: name,
                    nodeKind: node.kind,
                    url: url,
                    contentType: descriptor.contentTypes.first
                )
                artifacts.append(outputArtifact)
                records.append(.init(
                    name: name,
                    type: descriptor.type,
                    value: nil,
                    path: outputArtifact.path,
                    contentType: outputArtifact.contentType,
                    sizeBytes: outputArtifact.sizeBytes,
                    sha256: outputArtifact.sha256
                ))
                values[name] = .string(url.path)
            case .assetDirectory:
                guard let path = descriptor.path else {
                    throw ValidationError("Node '\(node.id)' directory output '\(name)' has no declared path.")
                }
                let url = try invocationOutputURL(path, nodeDirectory: nodeDirectory)
                var isDirectory: ObjCBool = false
                if providerValue == .null || !fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) {
                    if descriptor.optional { continue }
                    throw ValidationError("Node '\(node.id)' did not produce declared directory output '\(name)'.")
                }
                guard isDirectory.boolValue else {
                    throw ValidationError("Node '\(node.id)' output '\(name)' is not a directory.")
                }
                if let providerPath = providerValue?.stringValue {
                    let reported = try invocationOutputURL(providerPath, nodeDirectory: nodeDirectory)
                    guard reported.standardizedFileURL == url.standardizedFileURL else {
                        throw ValidationError("Node '\(node.id)' reported an unexpected path for output '\(name)'.")
                    }
                }
                let directory = try directoryIdentity(url)
                let manifestURL = nodeDirectory
                    .appendingPathComponent("artifacts", isDirectory: true)
                    .appendingPathComponent("\(name).manifest.json")
                try WorkflowBundleCodec.write(directory.manifest, to: manifestURL)
                let manifestArtifact = try artifact(
                    name: "\(name)_manifest",
                    nodeKind: node.kind,
                    url: manifestURL,
                    contentType: "application/json"
                )
                artifacts.append(manifestArtifact)
                records.append(.init(
                    name: name,
                    type: .assetDirectory,
                    value: nil,
                    path: try portableArtifactPath(for: url),
                    contentType: descriptor.contentTypes.first,
                    sizeBytes: directory.sizeBytes,
                    sha256: directory.sha256
                ))
                values[name] = .string(url.path)
            case .string, .integer, .number, .boolean, .enumeration, .json:
                guard let providerValue, providerValue != .null else {
                    if descriptor.optional { continue }
                    throw ValidationError("Node '\(node.id)' did not report declared value output '\(name)'.")
                }
                guard workflowValue(providerValue, matches: descriptor.type) else {
                    throw ValidationError("Node '\(node.id)' output '\(name)' has the wrong value type.")
                }
                records.append(.init(
                    name: name,
                    type: descriptor.type,
                    value: providerValue,
                    path: nil,
                    contentType: nil,
                    sizeBytes: nil,
                    sha256: try WorkflowBundleCodec.hash(providerValue)
                ))
                values[name] = providerValue
            }
        }
        if let providerValues {
            let undeclared = Set(providerValues.keys).subtracting(descriptors.keys)
            guard undeclared.isEmpty else {
                throw ValidationError(
                    "Node '\(node.id)' reported undeclared outputs: \(undeclared.sorted().joined(separator: ", "))."
                )
            }
        }
        return WorkflowVerifiedNodeOutputs(artifacts: artifacts, outputs: records, values: values)
    }

    private func invocationOutputURL(_ path: String, nodeDirectory: URL) throws -> URL {
        let candidate: URL
        if path.hasPrefix("/") {
            candidate = URL(fileURLWithPath: path).standardizedFileURL
        } else {
            guard isConfinedRelativeWorkflowPath(path) else {
                throw ValidationError("Workflow provider output path is not confined: \(path)")
            }
            candidate = nodeDirectory.appendingPathComponent(path).standardizedFileURL
        }
        let root = nodeDirectory.standardizedFileURL.path
        guard candidate.path.hasPrefix(root + "/") else {
            throw ValidationError("Workflow provider output path escapes the node directory: \(path)")
        }
        return candidate
    }

    private func directoryIdentity(_ directory: URL) throws -> WorkflowDirectoryIdentity {
        let root = directory.resolvingSymlinksInPath()
        var entries: [WorkflowAssetEntry] = []
        for path in try fileManager.subpathsOfDirectory(atPath: root.path).sorted() {
            let url = root.appendingPathComponent(path)
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            if values.isSymbolicLink == true {
                throw ValidationError("Workflow output directories cannot contain symbolic links: \(url.path)")
            }
            guard values.isRegularFile == true else { continue }
            entries.append(WorkflowAssetEntry(
                path: path,
                digest: try ModelArtifactPin.fileSHA256(url),
                sizeBytes: try ModelArtifactPin.fileByteCount(url),
                contentType: contentType(for: url)
            ))
        }
        let manifest = WorkflowOutputDirectoryManifest(contractVersion: "mere.run/output-directory.v1", entries: entries)
        return WorkflowDirectoryIdentity(
            manifest: manifest,
            sizeBytes: entries.reduce(0) { $0 + $1.sizeBytes },
            sha256: try WorkflowBundleCodec.hash(manifest)
        )
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
            let manifestURL = MereRunModelPaths.modelDir(provenance.id)
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

    private func prepareRunDirectory() throws {
        if fileManager.fileExists(atPath: runDirectory.path) {
            guard resume || bundleDirectory == runDirectory else {
                throw ValidationError("Run directory already exists. Pass --resume to reuse it: \(runDirectory.path)")
            }
        } else {
            try fileManager.createDirectory(at: runDirectory, withIntermediateDirectories: true)
        }
        try fileManager.createDirectory(
            at: runDirectory.appendingPathComponent("nodes", isDirectory: true),
            withIntermediateDirectories: true
        )
        let cancellationURL = runDirectory.appendingPathComponent("cancel.request")
        if fileManager.fileExists(atPath: cancellationURL.path) {
            try fileManager.removeItem(at: cancellationURL)
        }
        let processIDURL = runDirectory.appendingPathComponent("worker-child.pid")
        if fileManager.fileExists(atPath: processIDURL.path) {
            try fileManager.removeItem(at: processIDURL)
        }
        try fileManager.createDirectory(
            at: runDirectory.appendingPathComponent("outputs", isDirectory: true),
            withIntermediateDirectories: true
        )
        for filename in ["graph.json", "inputs.json", WorkflowAssetManifest.filename, WorkflowJobManifest.filename] {
            let source = bundleDirectory.appendingPathComponent(filename)
            let destination = runDirectory.appendingPathComponent(filename)
            if source != destination, !fileManager.fileExists(atPath: destination.path) {
                try fileManager.copyItem(at: source, to: destination)
            }
        }
        let actionsURL = runDirectory.appendingPathComponent("actions.json")
        if !fileManager.fileExists(atPath: actionsURL.path) {
            try WorkflowBundleCodec.write([DeclarativeAction](), to: actionsURL)
        }
    }

    private func localizeInputs(
        graph: WorkflowGraphDocument,
        inputs: WorkflowInputsDocument,
        assets: WorkflowAssetManifest
    ) throws -> WorkflowInputsDocument {
        var localized = inputs.values
        let root = runDirectory.appendingPathComponent("inputs", isDirectory: true)
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        var groupNames = Set<String>()
        for group in assets.groups {
            guard groupNames.insert(group.name).inserted,
                  group.name.range(of: "^[a-z][a-z0-9-]{0,63}$", options: .regularExpression) != nil else {
                throw ValidationError("Asset manifest contains an invalid or duplicate group '\(group.name)'.")
            }
            let groupRoot = root.appendingPathComponent(group.name, isDirectory: true)
            if group.kind == .assetDirectory {
                try fileManager.createDirectory(at: groupRoot, withIntermediateDirectories: true)
            }
            for entry in group.entries {
                guard isConfinedRelativeWorkflowPath(entry.path),
                      entry.digest.count == 64,
                      entry.digest.allSatisfy({ $0.isHexDigit && !$0.isUppercase }) else {
                    throw ValidationError("Workflow asset manifest contains an invalid path or digest.")
                }
                let source = bundleDirectory
                    .appendingPathComponent("assets", isDirectory: true)
                    .appendingPathComponent("sha256", isDirectory: true)
                    .appendingPathComponent(entry.digest)
                guard fileManager.fileExists(atPath: source.path),
                      try ModelArtifactPin.fileByteCount(source) == entry.sizeBytes,
                      try ModelArtifactPin.fileSHA256(source) == entry.digest else {
                    throw ValidationError("Workflow asset digest verification failed for '\(group.name)/\(entry.path)'.")
                }
                let destination = group.kind == .asset
                    ? root.appendingPathComponent("\(group.name)-\(entry.path)")
                    : groupRoot.appendingPathComponent(entry.path)
                try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                if !fileManager.fileExists(atPath: destination.path) {
                    do {
                        try fileManager.linkItem(at: source, to: destination)
                    } catch {
                        try fileManager.copyItem(at: source, to: destination)
                    }
                }
            }
            if group.kind == .asset, let entry = group.entries.first {
                guard group.entries.count == 1 else {
                    throw ValidationError("Asset group '\(group.name)' must contain exactly one file.")
                }
                localized[group.name] = .string(root.appendingPathComponent("\(group.name)-\(entry.path)").path)
            } else {
                localized[group.name] = .string(groupRoot.path)
            }
        }
        return WorkflowInputsDocument(values: localized)
    }

    private func initialManifest(
        graph: WorkflowGraphDocument,
        job: WorkflowJobManifest,
        order: [String]
    ) throws -> GraphRunManifest {
        let manifestURL = runDirectory.appendingPathComponent(GraphRunManifest.filename)
        if resume, fileManager.fileExists(atPath: manifestURL.path) {
            var existing = try WorkflowBundleCodec.decoder().decode(
                GraphRunManifest.self,
                from: Data(contentsOf: manifestURL)
            )
            guard existing.graphFingerprint == job.graphFingerprint else {
                throw ValidationError("Cannot resume: workflow graph fingerprint changed.")
            }
            existing.attempt += 1
            existing.state = .planned
            existing.error = nil
            existing.executor = executor
            existing.updatedAt = now()
            return existing
        }
        let nodesByID = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.id, $0) })
        return GraphRunManifest(
            contractVersion: GraphRunManifest.contractVersion,
            jobID: job.jobID,
            graphName: graph.name,
            graphFingerprint: job.graphFingerprint,
            state: .planned,
            createdAt: now(),
            updatedAt: now(),
            attempt: 1,
            executor: executor,
            nodes: order.compactMap { id in
                nodesByID[id].map {
                    GraphRunNodeRecord(
                        id: id,
                        kind: $0.kind,
                        state: .planned,
                        startedAt: nil,
                        completedAt: nil,
                        exitStatus: nil,
                        fingerprint: "",
                        artifacts: [],
                        error: nil
                    )
                }
            },
            outputs: [],
            error: nil
        )
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

    private func shouldResume(
        _ node: GraphRunNodeRecord,
        expectedFingerprint: String,
        nodeOutputs: inout [String: [String: WorkflowValue]]
    ) throws -> Bool {
        guard resume,
              node.state == .finished,
              node.fingerprint == expectedFingerprint,
              !node.outputs.isEmpty || !node.artifacts.isEmpty else { return false }
        var outputs: [String: WorkflowValue] = [:]
        if !node.outputs.isEmpty {
            for output in node.outputs {
                if let value = output.value {
                    guard try WorkflowBundleCodec.hash(value) == output.sha256 else { return false }
                    outputs[output.name] = value
                    continue
                }
                guard let path = output.path else { return false }
                let url = try artifactURL(for: path)
                if output.type == .assetDirectory {
                    var isDirectory: ObjCBool = false
                    guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory),
                          isDirectory.boolValue,
                          try directoryIdentity(url).sha256 == output.sha256 else { return false }
                } else {
                    guard fileManager.fileExists(atPath: url.path),
                          try ModelArtifactPin.fileByteCount(url) == output.sizeBytes,
                          try ModelArtifactPin.fileSHA256(url) == output.sha256 else { return false }
                }
                outputs[output.name] = .string(url.path)
            }
        } else {
            for artifact in node.artifacts {
                let url = try artifactURL(for: artifact.path)
                guard fileManager.fileExists(atPath: url.path),
                      try ModelArtifactPin.fileByteCount(url) == artifact.sizeBytes,
                      try ModelArtifactPin.fileSHA256(url) == artifact.sha256 else {
                    return false
                }
                outputs[artifact.name] = .string(url.path)
            }
        }
        nodeOutputs[node.id] = outputs
        return true
    }

    private func materializeGraphOutputs(
        graph: WorkflowGraphDocument,
        nodeOutputs: [String: [String: WorkflowValue]]
    ) throws -> [GraphRunArtifact] {
        var artifacts: [GraphRunArtifact] = []
        let outputsRoot = runDirectory.appendingPathComponent("outputs", isDirectory: true)
        for name in graph.outputs.keys.sorted() {
            guard case .reference(let rawReference)? = graph.outputs[name] else { continue }
            let reference = try WorkflowReference(rawReference)
            guard case .nodeOutput(let nodeID, let output) = reference.source,
                  let sourceValue = nodeOutputs[nodeID]?[output],
                  let node = graph.nodes.first(where: { $0.id == nodeID }),
                  let outputContract = WorkflowNodeRegistry.output(node: node, name: output) else {
                throw ValidationError("Workflow output '\(name)' was not produced.")
            }
            if outputContract.type != .asset && outputContract.type != .assetCollection && outputContract.type != .assetArray {
                let destination = outputsRoot.appendingPathComponent("\(name).json")
                try WorkflowBundleCodec.encoder().encode(sourceValue).write(to: destination, options: .atomic)
                artifacts.append(try artifact(
                    name: name,
                    nodeKind: "graph.output",
                    url: destination,
                    contentType: "application/json"
                ))
                continue
            }
            guard let sourcePath = sourceValue.stringValue else {
                throw ValidationError("Workflow output '\(name)' did not resolve to an artifact path.")
            }
            let source = URL(fileURLWithPath: sourcePath)
            let suffix = source.pathExtension.isEmpty ? "" : ".\(source.pathExtension)"
            let destination = outputsRoot.appendingPathComponent("\(name)\(suffix)")
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            do {
                try fileManager.linkItem(at: source, to: destination)
            } catch {
                try fileManager.copyItem(at: source, to: destination)
            }
            artifacts.append(try artifact(name: name, nodeKind: "graph.output", url: destination))
        }
        return artifacts
    }

    private func artifact(
        name: String,
        nodeKind: String,
        url: URL,
        contentType explicitContentType: String? = nil
    ) throws -> GraphRunArtifact {
        GraphRunArtifact(
            name: name,
            kind: nodeKind,
            path: try portableArtifactPath(for: url),
            contentType: explicitContentType ?? contentType(for: url),
            sizeBytes: try ModelArtifactPin.fileByteCount(url),
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }

    private func throwIfCancellationRequested() throws {
        if fileManager.fileExists(atPath: runDirectory.appendingPathComponent("cancel.request").path) {
            throw WorkflowCancellationError()
        }
    }

    private func artifactURL(for path: String) throws -> URL {
        let candidate = path.hasPrefix("/")
            ? URL(fileURLWithPath: path).standardizedFileURL
            : runDirectory.appendingPathComponent(path).standardizedFileURL
        let root = runDirectory.standardizedFileURL.path
        guard candidate.path == root || candidate.path.hasPrefix(root + "/") else {
            throw ValidationError("Workflow artifact path escapes the run directory: \(path)")
        }
        return candidate
    }

    private func portableArtifactPath(for url: URL) throws -> String {
        let candidate = url.standardizedFileURL.path
        let root = runDirectory.standardizedFileURL.path
        guard candidate.hasPrefix(root + "/") else {
            throw ValidationError("Workflow artifact path escapes the run directory: \(candidate)")
        }
        return String(candidate.dropFirst(root.count + 1))
    }

    private func contentType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "png": "image/png"
        case "jpg", "jpeg": "image/jpeg"
        case "webp": "image/webp"
        case "mp4": "video/mp4"
        case "wav": "audio/wav"
        case "json": "application/json"
        case "txt": "text/plain"
        case "safetensors": "application/x-safetensors"
        default: "application/octet-stream"
        }
    }

    private func persist(_ manifest: GraphRunManifest) throws {
        try WorkflowBundleCodec.write(manifest, to: runDirectory.appendingPathComponent(GraphRunManifest.filename))
    }

    private func record(_ event: GraphRunEvent) throws {
        let data = try WorkflowBundleCodec.encoder().encode(event)
        let url = runDirectory.appendingPathComponent("events.jsonl")
        if !fileManager.fileExists(atPath: url.path) {
            try Data().write(to: url)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: data)
        try handle.write(contentsOf: Data("\n".utf8))
        eventHandler?(event)
    }

    private func existingEventCount() throws -> Int {
        let url = runDirectory.appendingPathComponent("events.jsonl")
        guard fileManager.fileExists(atPath: url.path) else { return 0 }
        return try String(contentsOf: url, encoding: .utf8).split(separator: "\n").count
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

private struct WorkflowVerifiedNodeOutputs {
    let artifacts: [GraphRunArtifact]
    let outputs: [GraphRunNodeOutput]
    let values: [String: WorkflowValue]
}

private struct WorkflowOutputDirectoryManifest: Codable {
    let contractVersion: String
    let entries: [WorkflowAssetEntry]

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case entries
    }
}

private struct WorkflowDirectoryIdentity {
    let manifest: WorkflowOutputDirectoryManifest
    let sizeBytes: Int64
    let sha256: String
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

private func workflowValue(_ value: WorkflowValue, matches type: WorkflowPortType) -> Bool {
    switch type {
    case .string, .enumeration, .asset, .assetDirectory:
        value.stringValue != nil
    case .integer:
        value.integerValue != nil
    case .number:
        value.numberValue != nil
    case .boolean:
        value.booleanValue != nil
    case .json:
        true
    case .assetCollection, .assetArray:
        if case .array = value { true } else { false }
    }
}

private func isConfinedRelativeWorkflowPath(_ path: String) -> Bool {
    !path.isEmpty
        && !path.hasPrefix("/")
        && path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { component in
            !component.isEmpty && component != "." && component != ".."
        }
}

private func workflowVersion(_ current: String, satisfiesMinimum minimum: String) -> Bool {
    func components(_ value: String) -> [Int] {
        value
            .split(separator: ".")
            .prefix(3)
            .map { component in
                Int(component.prefix(while: \Character.isNumber)) ?? 0
            }
    }
    let currentParts = components(current)
    let minimumParts = components(minimum)
    for index in 0..<max(currentParts.count, minimumParts.count) {
        let lhs = index < currentParts.count ? currentParts[index] : 0
        let rhs = index < minimumParts.count ? minimumParts[index] : 0
        if lhs != rhs { return lhs > rhs }
    }
    return true
}

private struct WorkflowCancellationError: Error {}
