import ArgumentParser
import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

struct Run: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Inspect durable mere.run workflow reports and run directories.",
        subcommands: [
            RunList.self,
            RunInspect.self,
            RunWatch.self,
            RunFetch.self,
            RunCancel.self,
            RunRetry.self,
        ]
    )
}

struct RunList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "Find durable run directories, structured reports, and run plans under a root."
    )

    @Option(name: [.customLong("root")], help: "Root directory or file to scan.")
    var root: String?

    @Option(name: [.customLong("executor")], help: "Relay executor profile to list, such as relay:fleet.")
    var executor: String?

    @Option(name: [.customLong("limit")], help: "Maximum remote jobs to return.")
    var limit: Int = 50

    @Option(name: [.customLong("max-depth")], help: "Maximum child-directory depth to scan from --root.")
    var maxDepth: Int = 4

    @Flag(name: [.customLong("json")], help: "Emit a structured JSON run-list report.")
    var json: Bool = false

    func run() async throws {
        if let executor {
            guard root == nil else { throw ValidationError("--root and --executor are mutually exclusive.") }
            guard (1...500).contains(limit) else { throw ValidationError("--limit must be between 1 and 500.") }
            let jobs = try await WorkflowRemoteJobController.list(executor: executor, limit: limit)
            let result = RemoteRunListResult(executor: executor, jobs: jobs)
            if json {
                print(try StructuredRunOutput.encode(result))
            } else {
                for job in jobs { print("[\(job.state.rawValue)] \(job.jobReference)") }
            }
            return
        }
        let envelope = try makeListEnvelope()
        if json {
            print(try StructuredRunOutput.encode(envelope))
        } else {
            print(envelope.summary)
            for entry in envelope.result.entries {
                let state = entry.state.map { " state=\($0)" } ?? ""
                let detail = entry.format.map { " format=\($0)" } ?? ""
                print("[\(entry.status.rawValue)] \(entry.relativePath) kind=\(entry.kind)\(state)\(detail)")
            }
            for diagnostic in envelope.diagnostics {
                stderr("[\(diagnostic.severity.rawValue)] \(diagnostic.title): \(diagnostic.message)")
            }
        }
        if envelope.status == .blocked {
            throw ExitCode.failure
        }
    }

    func makeListEnvelope(
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) throws -> RunListEnvelope {
        guard maxDepth >= 0 else {
            throw ValidationError("--max-depth must be >= 0")
        }
        guard let root else {
            throw ValidationError("Provide --root for local runs or --executor for relay jobs.")
        }
        return RunListAnalyzer(
            root: root,
            maxDepth: maxDepth,
            fileManager: fileManager,
            now: now
        ).envelope()
    }

    private func stderr(_ message: String) {
        FileHandle.standardError.write(Data("\(message)\n".utf8))
    }
}

struct RunInspect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "inspect",
        abstract: "Inspect a run directory, structured report, or run plan."
    )

    @Argument(help: "Run directory, structured report JSON, or run plan JSON path.")
    var path: String

    @Flag(name: [.customLong("json")], help: "Emit a structured JSON inspection report.")
    var json: Bool = false

    func run() async throws {
        if path.hasPrefix("ssh://") || path.hasPrefix("relay://") {
            let job = try await WorkflowRemoteJobController.inspect(WorkflowRemoteReference(path))
            if json {
                print(try StructuredRunOutput.encode(job))
            } else {
                print("[\(job.state.rawValue)] \(job.jobReference)")
                if let error = job.error { stderr(error) }
            }
            return
        }
        if let graphManifest = try localGraphManifest(at: path) {
            if json {
                print(try StructuredRunOutput.encode(graphManifest))
            } else {
                print("[\(graphManifest.state.rawValue)] \(graphManifest.jobID) \(graphManifest.graphName)")
                for node in graphManifest.nodes {
                    print("[\(node.state.rawValue)] \(node.id) kind=\(node.kind) artifacts=\(node.artifacts.count)")
                }
                if let error = graphManifest.error { stderr(error) }
            }
            if graphManifest.state == .failed { throw ExitCode.failure }
            return
        }
        let envelope = makeInspectionEnvelope()
        if json {
            print(try StructuredRunOutput.encode(envelope))
        } else {
            print(envelope.summary)
            if let runDirectory = envelope.result.runDirectory {
                print("Status: \(runDirectory.status)")
                if let manifest = runDirectory.manifest {
                    print("Format: \(manifest.format)")
                    print("Model: \(manifest.model)")
                    print("Progress: \(manifest.step)/\(manifest.totalSteps)")
                }
                print("Events: \(runDirectory.events.count)")
                print("Actions: \(runDirectory.actions.count)")
                print("Artifacts: \(runDirectory.artifacts.count)")
            } else if let report = envelope.result.report {
                print("Report: \(report.command.joined(separator: " ")) \(report.mode.rawValue) \(report.status.rawValue)")
                print("Diagnostics: \(report.diagnosticCount)")
                print("Actions: \(report.actionCount)")
            } else if let plan = envelope.result.plan {
                print("Plan: \(plan.kind)")
                print("Command: \(plan.command.joined(separator: " "))")
            }
            for diagnostic in envelope.diagnostics {
                stderr("[\(diagnostic.severity.rawValue)] \(diagnostic.title): \(diagnostic.message)")
            }
        }
        if envelope.status == .blocked {
            throw ExitCode.failure
        }
    }

    func makeInspectionEnvelope(
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) -> RunInspectionEnvelope {
        RunInspectionAnalyzer(
            path: path,
            fileManager: fileManager,
            now: now
        ).envelope()
    }

    private func stderr(_ message: String) {
        FileHandle.standardError.write(Data("\(message)\n".utf8))
    }
}

struct RunWatch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "watch",
        abstract: "Watch the worker event stream for an SSH or relay graph job."
    )

    @Argument(help: "ssh:// or relay:// job reference.") var reference: String
    @Option(name: [.customLong("poll-interval")], help: "Polling interval in seconds.") var pollInterval = 2.0
    @Flag(name: [.customLong("json-stream")], help: "Emit worker NDJSON events to stdout.") var jsonStream = false
    @Flag(name: [.customLong("json")], help: "Emit one final job object to stdout.") var json = false

    func run() async throws {
        guard pollInterval >= 0.25 else { throw ValidationError("--poll-interval must be at least 0.25 seconds.") }
        guard !(json && jsonStream) else { throw ValidationError("--json and --json-stream are mutually exclusive.") }
        let parsed = try WorkflowRemoteReference(reference)
        var emittedLineCount = 0
        var previousState: GraphRunState?
        while true {
            let lines = normalizedEventLines(try await WorkflowRemoteJobController.events(parsed))
            if emittedLineCount > lines.count { emittedLineCount = 0 }
            for line in lines.dropFirst(emittedLineCount) {
                if jsonStream { stdout(line) } else { stderr(line) }
            }
            emittedLineCount = lines.count

            let job = try await WorkflowRemoteJobController.inspect(parsed)
            if job.state != previousState {
                if !jsonStream { stderr("[\(job.state.rawValue)] \(job.jobReference)") }
                previousState = job.state
            }
            if job.state.isTerminal {
                if json { print(try StructuredRunOutput.encode(job)) }
                if !json && !jsonStream { print("[\(job.state.rawValue)] \(job.jobReference)") }
                if job.state == .failed { throw ExitCode.failure }
                return
            }
            try await Task.sleep(nanoseconds: UInt64(pollInterval * 1_000_000_000))
        }
    }
}

struct RunFetch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "fetch",
        abstract: "Fetch a remote graph run into the standard local run-directory format."
    )

    @Argument(help: "ssh:// or relay:// job reference.") var reference: String
    @Option(name: [.customLong("into")], help: "Destination run directory.") var into: String
    @Flag(name: [.customLong("all-artifacts")], help: "Fetch intermediate artifacts as well as final outputs and reports.")
    var allArtifacts = false
    @Option(
        name: [.customLong("artifact")],
        parsing: .unconditionalSingleValue,
        help: "Fetch a named artifact. Repeat to fetch multiple artifacts."
    )
    var artifacts: [String] = []
    @Flag(name: [.customLong("json")], help: "Emit the fetched job as JSON.") var json = false

    func run() async throws {
        if allArtifacts && !artifacts.isEmpty {
            throw ValidationError("--artifact and --all-artifacts are mutually exclusive.")
        }
        let destination = URL(fileURLWithPath: into).standardizedFileURL
        let job = try await WorkflowRemoteJobController.fetch(
            WorkflowRemoteReference(reference),
            into: destination,
            allArtifacts: allArtifacts,
            artifactNames: Set(artifacts)
        )
        if json { print(try StructuredRunOutput.encode(job)) } else { print(job.runDirectory ?? destination.path) }
    }
}

struct RunCancel: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "cancel", abstract: "Cancel a graph run.")
    @Argument(help: "Local run directory or ssh:// or relay:// job reference.") var reference: String
    @Flag(name: [.customLong("json")], help: "Emit the cancelled job as JSON.") var json = false

    func run() async throws {
        if reference.hasPrefix("ssh://") || reference.hasPrefix("relay://") {
            let job = try await WorkflowRemoteJobController.cancel(WorkflowRemoteReference(reference))
            if json { print(try StructuredRunOutput.encode(job)) } else { print("[\(job.state.rawValue)] \(job.jobReference)") }
            return
        }
        let runDirectory = URL(fileURLWithPath: reference).standardizedFileURL
        guard FileManager.default.fileExists(atPath: runDirectory.appendingPathComponent(GraphRunManifest.filename).path) else {
            throw ValidationError("Graph run directory not found: \(runDirectory.path)")
        }
        let marker = runDirectory.appendingPathComponent("cancel.request")
        try Data().write(to: marker, options: .atomic)
        WorkflowChildProcessRegistry.terminateAll(in: runDirectory)
        let result = LocalCancellationResult(runDirectory: runDirectory.path, cancellationRequested: true)
        if json { print(try StructuredRunOutput.encode(result)) } else { print("Cancellation requested: \(runDirectory.path)") }
    }
}

struct RunRetry: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "retry", abstract: "Retry an immutable relay graph job.")
    @Argument(help: "relay:// job reference.") var reference: String
    @Flag(name: [.customLong("json")], help: "Emit the retry job as JSON.") var json = false

    func run() async throws {
        let job = try await WorkflowRemoteJobController.retry(WorkflowRemoteReference(reference))
        if json { print(try StructuredRunOutput.encode(job)) } else { print("[\(job.state.rawValue)] \(job.jobReference)") }
    }
}

private struct RemoteRunListResult: Codable {
    let executor: String
    let jobs: [WorkflowRemoteJob]
}

private struct LocalCancellationResult: Codable {
    let runDirectory: String
    let cancellationRequested: Bool

    enum CodingKeys: String, CodingKey {
        case runDirectory = "run_directory"
        case cancellationRequested = "cancellation_requested"
    }
}

private extension GraphRunState {
    var isTerminal: Bool {
        self == .finished || self == .failed || self == .cancelled
    }
}

private func normalizedEventLines(_ raw: String) -> [String] {
    raw.split(whereSeparator: \Character.isNewline).compactMap { substring in
        let line = substring.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty, !line.hasPrefix("event:"), !line.hasPrefix("id:"), line != "data: [DONE]" else {
            return nil
        }
        if line.hasPrefix("data:") {
            return line.dropFirst(5).trimmingCharacters(in: .whitespaces)
        }
        return line
    }
}

private func stderr(_ message: String) {
    FileHandle.standardError.write(Data("\(message)\n".utf8))
}

private func stdout(_ message: String) {
    FileHandle.standardOutput.write(Data("\(message)\n".utf8))
}

private func localGraphManifest(at path: String) throws -> GraphRunManifest? {
    let target = URL(fileURLWithPath: path).standardizedFileURL
    let manifestURL: URL
    var isDirectory: ObjCBool = false
    if FileManager.default.fileExists(atPath: target.path, isDirectory: &isDirectory), isDirectory.boolValue {
        manifestURL = target.appendingPathComponent(GraphRunManifest.filename)
    } else {
        manifestURL = target
    }
    guard FileManager.default.fileExists(atPath: manifestURL.path) else { return nil }
    let data = try Data(contentsOf: manifestURL)
    return try? WorkflowBundleCodec.decoder().decode(GraphRunManifest.self, from: data)
}
