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
}

protocol WorkflowProcessRunning {
    func run(arguments: [String], currentDirectory: URL) throws -> WorkflowProcessResult
}

struct WorkflowProcessRunner: WorkflowProcessRunning {
    func run(arguments: [String], currentDirectory: URL) throws -> WorkflowProcessResult {
        let process = Process()
        process.executableURL = CurrentExecutable.url()
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectory
        let stdoutURL = currentDirectory.appendingPathComponent(".workflow-stdout-(UUID().uuidString)")
        guard FileManager.default.createFile(atPath: stdoutURL.path, contents: nil) else {
            throw ValidationError("Could not create workflow child stdout capture.")
        }
        let stdout = try FileHandle(forWritingTo: stdoutURL)
        let runDirectory = currentDirectory.deletingLastPathComponent().deletingLastPathComponent()
        let processIDURL = runDirectory.appendingPathComponent("worker-child.pid")
        defer {
            try? stdout.close()
            try? FileManager.default.removeItem(at: stdoutURL)
            try? FileManager.default.removeItem(at: processIDURL)
        }
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = stdout
        process.standardError = FileHandle.standardError
        try process.run()
        try Data(String(process.processIdentifier).utf8).write(to: processIDURL, options: .atomic)
        process.waitUntilExit()
        try stdout.close()
        let data = try Data(contentsOf: stdoutURL)
        if FileManager.default.fileExists(atPath: runDirectory.appendingPathComponent("cancel.request").path) {
            throw WorkflowCancellationError()
        }
        return WorkflowProcessResult(
            status: process.terminationStatus,
            stdout: String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
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
                let fingerprint = try WorkflowBundleCodec.hash(WorkflowNodeFingerprint(
                    nodeID: node.id,
                    kind: node.kind,
                    arguments: resolvedArguments,
                    inputFingerprint: job.inputFingerprint,
                    upstreamArtifacts: manifest.nodes.prefix(nodeIndex).flatMap(\.artifacts)
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
                    nodeDirectory: nodeDirectory
                )
                manifest.nodes[nodeIndex].fingerprint = fingerprint
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

                let preflight = try processRunner.run(
                    arguments: invocation.preflightArguments,
                    currentDirectory: nodeDirectory
                )
                try throwIfCancellationRequested()
                try Data(preflight.stdout.utf8).write(
                    to: nodeDirectory.appendingPathComponent("preflight.json"),
                    options: .atomic
                )
                guard preflight.status == 0 else {
                    throw ValidationError("Node '\(nodeID)' preflight failed with status \(preflight.status).")
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

                let result = try processRunner.run(
                    arguments: invocation.runArguments,
                    currentDirectory: nodeDirectory
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

                var artifacts: [GraphRunArtifact] = []
                var outputs: [String: WorkflowValue] = [:]
                for name in invocation.outputs.keys.sorted() {
                    guard let url = invocation.outputs[name], fileManager.fileExists(atPath: url.path) else {
                        throw ValidationError("Node '\(nodeID)' did not produce declared output '\(name)'.")
                    }
                    let artifact = try artifact(name: name, nodeKind: node.kind, url: url)
                    artifacts.append(artifact)
                    outputs[name] = .string(url.path)
                }
                nodeOutputs[nodeID] = outputs
                manifest.nodes[nodeIndex].artifacts = artifacts
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
              !node.artifacts.isEmpty else { return false }
        var outputs: [String: WorkflowValue] = [:]
        for artifact in node.artifacts {
            let url = try artifactURL(for: artifact.path)
            guard fileManager.fileExists(atPath: url.path),
                  try ModelArtifactPin.fileByteCount(url) == artifact.sizeBytes,
                  try ModelArtifactPin.fileSHA256(url) == artifact.sha256 else {
                return false
            }
            outputs[artifact.name] = .string(url.path)
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
                  let sourcePath = nodeOutputs[nodeID]?[output]?.stringValue else {
                throw ValidationError("Workflow output '\(name)' was not produced.")
            }
            let source = URL(fileURLWithPath: sourcePath)
            let destination = outputsRoot.appendingPathComponent("\(name).\(source.pathExtension)")
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

    private func artifact(name: String, nodeKind: String, url: URL) throws -> GraphRunArtifact {
        GraphRunArtifact(
            name: name,
            kind: nodeKind,
            path: try portableArtifactPath(for: url),
            contentType: contentType(for: url),
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
        case "mp4": "video/mp4"
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
    let nodeID: String
    let kind: String
    let arguments: [String: WorkflowValue]
    let inputFingerprint: String
    let upstreamArtifacts: [GraphRunArtifact]

    enum CodingKeys: String, CodingKey {
        case nodeID = "node_id"
        case kind
        case arguments
        case inputFingerprint = "input_fingerprint"
        case upstreamArtifacts = "upstream_artifacts"
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
