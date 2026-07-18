import ArgumentParser
import Foundation
import MereRunCore
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

struct Graph: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "graph",
        abstract: "Validate, materialize, run, and submit portable workflow graphs.",
        subcommands: [
            GraphCatalog.self,
            GraphDataset.self,
            GraphValidate.self,
            GraphPreflight.self,
            GraphMaterialize.self,
            GraphExportJob.self,
            GraphRun.self,
            GraphRunJob.self,
            GraphSubmit.self,
            GraphSubmitJob.self,
            GraphWorker.self,
        ]
    )
}

struct GraphCatalogResult: Codable, Equatable {
    let graphSchemaVersion: Int
    let graphKind: String
    let jobContractVersion: String
    let providers: [WorkflowGraphProviderRequirement]
    let nodes: [WorkflowNodeCatalogEntry]

    enum CodingKeys: String, CodingKey {
        case graphSchemaVersion = "graph_schema_version"
        case graphKind = "graph_kind"
        case jobContractVersion = "job_contract_version"
        case providers
        case nodes
    }
}

struct GraphCatalog: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "catalog",
        abstract: "List registered workflow node contracts."
    )

    @Flag(name: [.long], help: "Emit the graph catalog as JSON.")
    var json = false

    func run() throws {
        let pluginCatalog = WorkflowGraphProviderRegistry.discoveredCatalog()
        let pluginProviders = pluginCatalog.providers.map(\.requirement)
        let builtIn = WorkflowGraphProviderRequirement(
            id: WorkflowNodeProviderIdentity.builtInID,
            version: WorkflowNodeRegistry.builtInProvider.version,
            catalogSHA256: WorkflowNodeRegistry.builtInProvider.catalogSHA256,
            nodeKinds: WorkflowNodeRegistry.entries.map(\.kind).sorted()
        )
        let result = GraphCatalogResult(
            graphSchemaVersion: WorkflowGraphDocument.schemaVersion,
            graphKind: WorkflowGraphDocument.kind,
            jobContractVersion: WorkflowJobManifest.contractVersion,
            providers: [builtIn] + pluginProviders,
            nodes: WorkflowNodeRegistry.catalogEntries(pluginNodes: pluginCatalog.nodes)
        )
        if json {
            print(try StructuredRunOutput.encode(result))
        } else {
            for node in result.nodes {
                print("\(node.kind): \(node.title)")
            }
        }
    }
}

struct GraphDocumentRequest: Codable, Equatable {
    let graphPath: String
    let inputsPath: String?
    let executor: String?

    enum CodingKeys: String, CodingKey {
        case graphPath = "graph_path"
        case inputsPath = "inputs_path"
        case executor
    }
}

struct GraphValidationResult: Codable, Equatable {
    let graphName: String
    let nodeCount: Int
    let inputCount: Int
    let outputCount: Int
    let executionOrder: [String]

    enum CodingKeys: String, CodingKey {
        case graphName = "graph_name"
        case nodeCount = "node_count"
        case inputCount = "input_count"
        case outputCount = "output_count"
        case executionOrder = "execution_order"
    }
}

typealias GraphValidationEnvelope = StructuredRunEnvelope<GraphDocumentRequest, GraphValidationResult>

struct GraphValidate: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "validate",
        abstract: "Validate a typed workflow graph without executing it."
    )

    @Argument(help: "Workflow graph JSON file.")
    var file: String

    @Option(name: [.customLong("inputs-json")], help: "Workflow input values JSON file.")
    var inputsJSON: String?

    @Flag(name: [.long], help: "Emit a structured validation report.")
    var json = false

    func run() throws {
        let envelope = try makeEnvelope()
        if json {
            print(try StructuredRunOutput.encode(envelope))
        } else {
            print(envelope.summary)
            emitDiagnostics(envelope.diagnostics)
        }
        if envelope.status == .blocked { throw ExitCode.failure }
    }

    func makeEnvelope(now: @escaping () -> Date = Date.init) throws -> GraphValidationEnvelope {
        let loaded = try loadGraphRequest(file: file, inputsJSON: inputsJSON)
        let validation = WorkflowGraphValidator.validate(graph: loaded.graph, inputs: loaded.inputs)
        return GraphValidationEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["graph", "validate"],
            mode: .preflight,
            status: validation.status,
            createdAt: now(),
            cwd: FileManager.default.currentDirectoryPath,
            summary: validation.status == .blocked
                ? "Workflow graph is invalid."
                : "Workflow graph is valid with \(loaded.graph.nodes.count) node(s).",
            request: .init(graphPath: loaded.graphURL.path, inputsPath: loaded.inputsURL?.path, executor: nil),
            result: .init(
                graphName: loaded.graph.name,
                nodeCount: loaded.graph.nodes.count,
                inputCount: loaded.graph.inputs.count,
                outputCount: loaded.graph.outputs.count,
                executionOrder: validation.order
            ),
            diagnostics: validation.diagnostics,
            actions: []
        )
    }
}

struct GraphPreflightResult: Codable, Equatable {
    let graphName: String
    let executionOrder: [String]
    let requiredNodeKinds: [String]
    let requiredModelIDs: [String]
    let installedModelIDs: [String]
    let requiredProviders: [WorkflowGraphProviderRequirement]
    let availableProviders: [WorkflowGraphProviderRequirement]
    let requiredSecretNames: [String]
    let availableSecretNames: [String]
    let minimumAcceleratorMemoryBytes: Int64?
    let acceleratorMemoryBytes: UInt64
    let minimumSystemMemoryBytes: Int64?
    let systemMemoryBytes: UInt64
    let minimumDiskBytes: Int64?
    let availableDiskBytes: Int64?
    let minimumCPUCores: Int?
    let logicalCPUCores: Int
    let networkAccessRequired: Bool
    let networkAccessAvailable: Bool
    let executor: String

    enum CodingKeys: String, CodingKey {
        case graphName = "graph_name"
        case executionOrder = "execution_order"
        case requiredNodeKinds = "required_node_kinds"
        case requiredModelIDs = "required_model_ids"
        case installedModelIDs = "installed_model_ids"
        case requiredProviders = "required_providers"
        case availableProviders = "available_providers"
        case requiredSecretNames = "required_secret_names"
        case availableSecretNames = "available_secret_names"
        case minimumAcceleratorMemoryBytes = "minimum_accelerator_memory_bytes"
        case acceleratorMemoryBytes = "accelerator_memory_bytes"
        case minimumSystemMemoryBytes = "minimum_system_memory_bytes"
        case systemMemoryBytes = "system_memory_bytes"
        case minimumDiskBytes = "minimum_disk_bytes"
        case availableDiskBytes = "available_disk_bytes"
        case minimumCPUCores = "minimum_cpu_cores"
        case logicalCPUCores = "logical_cpu_cores"
        case networkAccessRequired = "network_access_required"
        case networkAccessAvailable = "network_access_available"
        case executor
    }
}

typealias GraphPreflightEnvelope = StructuredRunEnvelope<GraphDocumentRequest, GraphPreflightResult>

struct GraphPreflight: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "preflight",
        abstract: "Preflight a workflow graph against an executor."
    )

    @Argument(help: "Workflow graph JSON file.")
    var file: String

    @Option(name: [.customLong("inputs-json")], help: "Workflow input values JSON file.")
    var inputsJSON: String?

    @Option(name: [.long], help: "Executor reference: local, ssh:<profile>, or relay:<profile>.")
    var executor = "local"

    @Flag(name: [.long], help: "Emit a structured preflight report.")
    var json = false

    func run() async throws {
        let envelope = try await makeEnvelope()
        if json {
            print(try StructuredRunOutput.encode(envelope))
        } else {
            print(envelope.summary)
            emitDiagnostics(envelope.diagnostics)
        }
        if envelope.status == .blocked { throw ExitCode.failure }
    }

    func makeEnvelope(now: @escaping () -> Date = Date.init) async throws -> GraphPreflightEnvelope {
        let loaded = try loadGraphRequest(file: file, inputsJSON: inputsJSON)
        let validation = WorkflowGraphValidator.validate(graph: loaded.graph, inputs: loaded.inputs)
        let requirements = WorkflowGraphRequirements.resolve(graph: loaded.graph, inputs: loaded.inputs)
        let probe = try await WorkflowExecutorController.probe(reference: executor)
        var diagnostics = validation.diagnostics
        if !probe.contractVersions.contains(WorkflowJobManifest.contractVersion) {
            diagnostics.append(.init(
                id: "executor_contract_unsupported",
                severity: .blocker,
                title: "Executor contract is incompatible",
                message: "Executor '\(executor)' does not support \(WorkflowJobManifest.contractVersion)."
            ))
        }
        if !workflowVersion(probe.workerVersion, satisfiesMinimum: MereRunCLIVersion.current) {
            diagnostics.append(.init(
                id: "executor_worker_version_too_old",
                severity: .blocker,
                title: "Executor worker is too old",
                message: "Executor '\(executor)' reports mere.run \(probe.workerVersion); this graph requires \(MereRunCLIVersion.current) or newer."
            ))
        }
        if !requirements.acceleratorBackends.contains(probe.acceleratorBackend), probe.acceleratorBackend != "mixed" {
            diagnostics.append(.init(
                id: "executor_accelerator_unsupported",
                severity: .blocker,
                title: "Executor accelerator is unsupported",
                message: "Executor '\(executor)' reports accelerator backend '\(probe.acceleratorBackend)'; this graph accepts \(requirements.acceleratorBackends.joined(separator: ", "))."
            ))
        }
        for kind in requirements.nodeKinds where !probe.nodeKinds.contains(kind) {
            diagnostics.append(.init(
                id: "executor_node_kind_missing_\(kind)",
                severity: .blocker,
                title: "Executor does not support workflow node",
                message: "Executor '\(executor)' does not support node kind '\(kind)'."
            ))
        }
        for provider in requirements.providers {
            guard probe.providers.contains(provider) else {
                diagnostics.append(.init(
                    id: "executor_provider_missing_\(provider.id)",
                    severity: .blocker,
                    title: "Executor graph provider is missing or incompatible",
                    message: "Executor '\(executor)' requires provider '\(provider.id)' at version \(provider.version) with catalog \(provider.catalogSHA256)."
                ))
                continue
            }
        }
        for modelID in requirements.modelIDs where !probe.installedModelIDs.contains(modelID) {
            diagnostics.append(.init(
                id: "executor_model_missing_\(modelID)",
                severity: .blocker,
                title: "Executor model is missing",
                message: "Executor '\(executor)' does not have model '\(modelID)' installed.",
                suggestedActionIDs: ["pull-model-\(modelID)"]
            ))
        }
        for secretName in requirements.secretNames where !probe.availableSecretNames.contains(secretName) {
            diagnostics.append(.init(
                id: "executor_secret_missing_\(secretName)",
                severity: .blocker,
                title: "Executor secret is missing",
                message: "Executor '\(executor)' does not expose configured secret '\(secretName)'. Configure \(workflowSecretEnvironmentKey(secretName)) on that worker."
            ))
        }
        if let minimum = requirements.minimumAcceleratorMemoryBytes,
           probe.memoryBytes < UInt64(minimum) {
            diagnostics.append(.init(
                id: "executor_accelerator_memory_insufficient",
                severity: .blocker,
                title: "Executor accelerator memory is insufficient",
                message: "Executor '\(executor)' reports \(probe.memoryBytes) accelerator-memory bytes; this graph requires at least \(minimum)."
            ))
        }
        if let minimum = requirements.minimumSystemMemoryBytes,
           probe.systemMemoryBytes < UInt64(minimum) {
            diagnostics.append(.init(
                id: "executor_system_memory_insufficient",
                severity: .blocker,
                title: "Executor system memory is insufficient",
                message: "Executor '\(executor)' reports \(probe.systemMemoryBytes) system-memory bytes; this graph requires at least \(minimum)."
            ))
        }
        if let minimum = requirements.minimumDiskBytes,
           probe.availableDiskBytes.map({ $0 < minimum }) != false {
            diagnostics.append(.init(
                id: "executor_disk_insufficient",
                severity: .blocker,
                title: "Executor disk space is insufficient",
                message: "Executor '\(executor)' does not report at least \(minimum) free disk bytes."
            ))
        }
        if let minimum = requirements.minimumCPUCores,
           probe.logicalCPUCores < minimum {
            diagnostics.append(.init(
                id: "executor_cpu_insufficient",
                severity: .blocker,
                title: "Executor CPU capacity is insufficient",
                message: "Executor '\(executor)' reports \(probe.logicalCPUCores) logical CPU cores; this graph requires at least \(minimum)."
            ))
        }
        if requirements.networkAccess, !probe.networkAccess {
            diagnostics.append(.init(
                id: "executor_network_unavailable",
                severity: .blocker,
                title: "Executor network access is unavailable",
                message: "Executor '\(executor)' does not allow network access required by this graph."
            ))
        }
        diagnostics.append(contentsOf: assetDiagnostics(graph: loaded.graph, inputs: loaded.inputs))
        let missingModelIDs = requirements.modelIDs.filter { !probe.installedModelIDs.contains($0) }
        let actions = missingModelIDs.map { modelID in
            DeclarativeAction(
                id: "pull-model-\(modelID)",
                label: "Pull \(modelID)",
                kind: .command,
                enabled: executor == "local",
                disabledReason: executor == "local" ? nil : "Pull the model on the selected executor.",
                command: executor == "local" ? DeclarativeCommand(
                    argv: ["mere.run", "model", "pull", modelID],
                    cwd: FileManager.default.currentDirectoryPath,
                    commandPath: ["model", "pull"]
                ) : nil
            )
        }
        let status = StructuredRunOutput.status(for: diagnostics)
        return GraphPreflightEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["graph", "preflight"],
            mode: .preflight,
            status: status,
            createdAt: now(),
            cwd: FileManager.default.currentDirectoryPath,
            summary: status == .blocked
                ? "Workflow is blocked on executor \(executor)."
                : "Workflow is ready for executor \(executor).",
            request: .init(graphPath: loaded.graphURL.path, inputsPath: loaded.inputsURL?.path, executor: executor),
            result: .init(
                graphName: loaded.graph.name,
                executionOrder: validation.order,
                requiredNodeKinds: requirements.nodeKinds,
                requiredModelIDs: requirements.modelIDs,
                installedModelIDs: probe.installedModelIDs,
                requiredProviders: requirements.providers,
                availableProviders: probe.providers,
                requiredSecretNames: requirements.secretNames,
                availableSecretNames: probe.availableSecretNames,
                minimumAcceleratorMemoryBytes: requirements.minimumAcceleratorMemoryBytes,
                acceleratorMemoryBytes: probe.memoryBytes,
                minimumSystemMemoryBytes: requirements.minimumSystemMemoryBytes,
                systemMemoryBytes: probe.systemMemoryBytes,
                minimumDiskBytes: requirements.minimumDiskBytes,
                availableDiskBytes: probe.availableDiskBytes,
                minimumCPUCores: requirements.minimumCPUCores,
                logicalCPUCores: probe.logicalCPUCores,
                networkAccessRequired: requirements.networkAccess,
                networkAccessAvailable: probe.networkAccess,
                executor: executor
            ),
            diagnostics: diagnostics,
            actions: actions
        )
    }
}

struct GraphMaterializationRequest: Codable, Equatable {
    let graphPath: String
    let inputsPath: String?
    let destination: String

    enum CodingKeys: String, CodingKey {
        case graphPath = "graph_path"
        case inputsPath = "inputs_path"
        case destination
    }
}

struct GraphMaterializationResult: Codable, Equatable {
    let jobID: String
    let bundleDirectory: String
    let graphFingerprint: String
    let inputFingerprint: String
    let assetCount: Int

    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case bundleDirectory = "bundle_directory"
        case graphFingerprint = "graph_fingerprint"
        case inputFingerprint = "input_fingerprint"
        case assetCount = "asset_count"
    }
}

typealias GraphMaterializationEnvelope = StructuredRunEnvelope<GraphMaterializationRequest, GraphMaterializationResult>

struct GraphMaterialize: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "materialize",
        abstract: "Create an immutable workflow job bundle in a run directory."
    )

    @Argument(help: "Workflow graph JSON file.")
    var file: String

    @Option(name: [.customLong("inputs-json")], help: "Workflow input values JSON file.")
    var inputsJSON: String?

    @Option(name: [.customLong("run-dir")], help: "Destination run directory.")
    var runDirectory: String

    @Flag(name: [.long], help: "Emit a structured materialization report.")
    var json = false

    func run() throws {
        let envelope = try makeEnvelope()
        if json { print(try StructuredRunOutput.encode(envelope)) } else { print(envelope.summary) }
    }

    func makeEnvelope(now: @escaping () -> Date = Date.init) throws -> GraphMaterializationEnvelope {
        try materializationEnvelope(
            file: file,
            inputsJSON: inputsJSON,
            destination: runDirectory,
            command: ["graph", "materialize"],
            now: now
        )
    }
}

struct GraphExportJob: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "export-job",
        abstract: "Export an immutable workflow job bundle for any executor."
    )

    @Argument(help: "Workflow graph JSON file.")
    var file: String

    @Option(name: [.customLong("inputs-json")], help: "Workflow input values JSON file.")
    var inputsJSON: String?

    @Option(name: [.long], help: "Destination job bundle directory.")
    var output: String

    @Flag(name: [.long], help: "Emit a structured export report.")
    var json = false

    func run() throws {
        let envelope = try materializationEnvelope(
            file: file,
            inputsJSON: inputsJSON,
            destination: output,
            command: ["graph", "export-job"]
        )
        if json { print(try StructuredRunOutput.encode(envelope)) } else { print(envelope.summary) }
    }
}

struct GraphRunRequest: Codable, Equatable {
    let graphPath: String
    let inputsPath: String?
    let runDirectory: String
    let resume: Bool

    enum CodingKeys: String, CodingKey {
        case graphPath = "graph_path"
        case inputsPath = "inputs_path"
        case runDirectory = "run_directory"
        case resume
    }
}

typealias GraphRunEnvelope = StructuredRunEnvelope<GraphRunRequest, WorkflowRunOutcome>

struct GraphRun: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Run a workflow graph locally."
    )

    @Argument(help: "Workflow graph JSON file.")
    var file: String

    @Option(name: [.customLong("inputs-json")], help: "Workflow input values JSON file.")
    var inputsJSON: String?

    @Option(name: [.customLong("run-dir")], help: "Durable run directory.")
    var runDirectory: String

    @Flag(name: [.long], help: "Reuse verified outputs from a prior interrupted run.")
    var resume = false

    @Flag(name: [.long], help: "Emit one final structured run report.")
    var json = false

    @Flag(name: [.customLong("json-stream")], help: "Emit run events as newline-delimited JSON.")
    var jsonStream = false

    func run() async throws {
        guard !(json && jsonStream) else {
            throw ValidationError("--json and --json-stream are mutually exclusive.")
        }
        let runURL = URL(fileURLWithPath: runDirectory).standardizedFileURL
        if !resume {
            let loaded = try loadGraphRequest(file: file, inputsJSON: inputsJSON)
            _ = try WorkflowBundleMaterializer(
                graph: loaded.graph,
                suppliedInputs: loaded.inputs,
                destination: runURL
            ).materialize()
        }
        let outcome = try WorkflowRunner(
            bundleDirectory: runURL,
            runDirectory: runURL,
            resume: resume,
            eventHandler: jsonStream ? { event in
                emitGraphStreamEvent(event)
            } : nil
        ).execute()
        let envelope = GraphRunEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["graph", "run"],
            mode: .run,
            status: structuredStatus(outcome.state),
            createdAt: Date(),
            cwd: FileManager.default.currentDirectoryPath,
            summary: "Workflow \(outcome.state.rawValue): \(outcome.jobID)",
            request: .init(
                graphPath: URL(fileURLWithPath: file).standardizedFileURL.path,
                inputsPath: inputsJSON.map { URL(fileURLWithPath: $0).standardizedFileURL.path },
                runDirectory: runURL.path,
                resume: resume
            ),
            result: outcome,
            diagnostics: outcome.state == .failed ? [.init(
                id: "workflow_run_failed",
                severity: .blocker,
                title: "Workflow run failed",
                message: "Inspect \(runURL.appendingPathComponent(GraphRunManifest.filename).path) for node details."
            )] : [],
            actions: []
        )
        if json { print(try StructuredRunOutput.encode(envelope)) }
        if !json && !jsonStream { print(envelope.summary) }
        if outcome.state == .failed || outcome.state == .cancelled { throw ExitCode.failure }
    }
}

struct GraphSubmitRequest: Codable, Equatable {
    let graphPath: String
    let inputsPath: String?
    let runDirectory: String
    let executor: String

    enum CodingKeys: String, CodingKey {
        case graphPath = "graph_path"
        case inputsPath = "inputs_path"
        case runDirectory = "run_directory"
        case executor
    }
}

struct GraphRunJobRequest: Codable, Equatable {
    let bundlePath: String
    let runDirectory: String
    let resume: Bool

    enum CodingKeys: String, CodingKey {
        case bundlePath = "bundle_path"
        case runDirectory = "run_directory"
        case resume
    }
}

typealias GraphRunJobEnvelope = StructuredRunEnvelope<GraphRunJobRequest, WorkflowRunOutcome>

struct GraphRunJob: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run-job",
        abstract: "Run an existing immutable workflow job bundle locally."
    )

    @Option(name: [.long], help: "Portable job bundle directory.")
    var bundle: String

    @Option(name: [.customLong("run-dir")], help: "Durable run directory, separate from the immutable bundle.")
    var runDirectory: String

    @Flag(name: [.long], help: "Reuse verified outputs from a prior interrupted run.")
    var resume = false

    @Flag(name: [.long], help: "Emit one final structured run report.")
    var json = false

    @Flag(name: [.customLong("json-stream")], help: "Emit run events as newline-delimited JSON.")
    var jsonStream = false

    func run() async throws {
        guard !(json && jsonStream) else {
            throw ValidationError("--json and --json-stream are mutually exclusive.")
        }
        let eventHandler: ((GraphRunEvent) -> Void)? = jsonStream ? { event in
            emitGraphStreamEvent(event)
        } : nil
        let envelope: GraphRunJobEnvelope = try makeEnvelope(eventHandler: eventHandler)
        if json { print(try StructuredRunOutput.encode(envelope)) }
        if !json && !jsonStream { print(envelope.summary) }
        if envelope.result.state == .failed || envelope.result.state == .cancelled { throw ExitCode.failure }
    }

    func makeEnvelope(
        now: @escaping () -> Date = Date.init,
        eventHandler: ((GraphRunEvent) -> Void)? = nil,
        processRunner: WorkflowProcessRunning = WorkflowProcessRunner()
    ) throws -> GraphRunJobEnvelope {
        let bundleURL = URL(fileURLWithPath: bundle).standardizedFileURL
        let runURL = URL(fileURLWithPath: runDirectory).standardizedFileURL
        try requireSeparateWorkflowDirectories(bundle: bundleURL, run: runURL)
        _ = try verifiedPortableWorkflowBundle(at: bundleURL)
        let outcome = try WorkflowRunner(
            bundleDirectory: bundleURL,
            runDirectory: runURL,
            resume: resume,
            processRunner: processRunner,
            eventHandler: eventHandler
        ).execute()
        return GraphRunJobEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["graph", "run-job"],
            mode: .run,
            status: structuredStatus(outcome.state),
            createdAt: now(),
            cwd: FileManager.default.currentDirectoryPath,
            summary: "Workflow \(outcome.state.rawValue): \(outcome.jobID)",
            request: .init(
                bundlePath: bundleURL.path,
                runDirectory: runURL.path,
                resume: resume
            ),
            result: outcome,
            diagnostics: outcome.state == .failed ? [.init(
                id: "workflow_run_failed",
                severity: .blocker,
                title: "Workflow run failed",
                message: "Inspect \(runURL.appendingPathComponent(GraphRunManifest.filename).path) for node details."
            )] : [],
            actions: []
        )
    }
}

struct GraphSubmitJobRequest: Codable, Equatable {
    let bundlePath: String
    let runDirectory: String
    let executor: String

    enum CodingKeys: String, CodingKey {
        case bundlePath = "bundle_path"
        case runDirectory = "run_directory"
        case executor
    }
}

typealias GraphSubmitJobEnvelope = StructuredRunEnvelope<GraphSubmitJobRequest, WorkflowRemoteJob>

struct GraphSubmitJob: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "submit-job",
        abstract: "Submit an existing immutable workflow job bundle to SSH or Relay."
    )

    @Option(name: [.long], help: "Portable job bundle directory.")
    var bundle: String

    @Option(name: [.long], help: "Remote executor reference: ssh:<profile> or relay:<profile>.")
    var executor: String

    @Option(name: [.customLong("run-dir")], help: "Local durable run directory, separate from the immutable bundle.")
    var runDirectory: String

    @Flag(name: [.long], help: "Emit a structured submission report.")
    var json = false

    func run() async throws {
        let envelope = try await makeEnvelope()
        if json { print(try StructuredRunOutput.encode(envelope)) } else { print(envelope.result.jobReference) }
    }

    func makeEnvelope(
        now: @escaping () -> Date = Date.init,
        submitter: (String, URL, URL) async throws -> WorkflowRemoteJob = { reference, bundle, run in
            try await WorkflowExecutorController.submit(
                reference: reference,
                bundleDirectory: bundle,
                localRunDirectory: run
            )
        }
    ) async throws -> GraphSubmitJobEnvelope {
        let bundleURL = URL(fileURLWithPath: bundle).standardizedFileURL
        let runURL = URL(fileURLWithPath: runDirectory).standardizedFileURL
        try requireSeparateWorkflowDirectories(bundle: bundleURL, run: runURL)
        let verified = try verifiedPortableWorkflowBundle(at: bundleURL)
        try copyPortableWorkflowBundle(verified, to: runURL)
        let remoteJob = try await submitter(executor, runURL, runURL)
        return GraphSubmitJobEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["graph", "submit-job"],
            mode: .run,
            status: structuredStatus(remoteJob.state),
            createdAt: now(),
            cwd: FileManager.default.currentDirectoryPath,
            summary: "Submitted workflow job \(remoteJob.jobID) to \(executor).",
            request: .init(
                bundlePath: bundleURL.path,
                runDirectory: runURL.path,
                executor: executor
            ),
            result: remoteJob,
            diagnostics: [],
            actions: []
        )
    }
}

typealias GraphSubmitEnvelope = StructuredRunEnvelope<GraphSubmitRequest, WorkflowRemoteJob>

struct GraphSubmit: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "submit",
        abstract: "Submit a portable workflow job to an SSH or relay executor."
    )

    @Argument(help: "Workflow graph JSON file.")
    var file: String

    @Option(name: [.customLong("inputs-json")], help: "Workflow input values JSON file.")
    var inputsJSON: String?

    @Option(name: [.long], help: "Remote executor reference: ssh:<profile> or relay:<profile>.")
    var executor: String

    @Option(name: [.customLong("run-dir")], help: "Local durable job record and bundle directory.")
    var runDirectory: String

    @Flag(name: [.long], help: "Emit a structured submission report.")
    var json = false

    func run() async throws {
        let loaded = try loadGraphRequest(file: file, inputsJSON: inputsJSON)
        let runURL = URL(fileURLWithPath: runDirectory).standardizedFileURL
        let bundle = try WorkflowBundleMaterializer(
            graph: loaded.graph,
            suppliedInputs: loaded.inputs,
            destination: runURL
        ).materialize()
        let remoteJob = try await WorkflowExecutorController.submit(
            reference: executor,
            bundleDirectory: bundle.directory,
            localRunDirectory: runURL
        )
        let envelope = GraphSubmitEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["graph", "submit"],
            mode: .run,
            status: structuredStatus(remoteJob.state),
            createdAt: Date(),
            cwd: FileManager.default.currentDirectoryPath,
            summary: "Submitted workflow job \(remoteJob.jobID) to \(executor).",
            request: .init(
                graphPath: loaded.graphURL.path,
                inputsPath: loaded.inputsURL?.path,
                runDirectory: runURL.path,
                executor: executor
            ),
            result: remoteJob,
            diagnostics: [],
            actions: []
        )
        if json { print(try StructuredRunOutput.encode(envelope)) } else { print(remoteJob.jobReference) }
    }
}

struct GraphWorker: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "worker",
        abstract: "Run the portable workflow worker protocol.",
        subcommands: [GraphWorkerProbe.self, GraphWorkerExecute.self, GraphWorkerInspect.self, GraphWorkerCancel.self]
    )
}

struct GraphWorkerProbe: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "probe", abstract: "Report worker capabilities.")

    @Flag(name: [.long], help: "Emit a structured worker capability report.")
    var json = false

    func run() throws {
        let probe = WorkflowExecutorProbe.local()
        if json { print(try StructuredRunOutput.encode(probe)) } else { print(probe.summary) }
    }
}

struct GraphWorkerExecute: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "execute", abstract: "Execute a materialized job bundle.")

    @Option(name: [.long], help: "Portable job bundle directory.")
    var bundle: String

    @Option(name: [.customLong("run-dir")], help: "Durable worker run directory.")
    var runDirectory: String

    @Flag(name: [.customLong("json-stream")], help: "Emit newline-delimited run events.")
    var jsonStream = false

    @Flag(name: [.long], help: "Resume verified completed nodes.")
    var resume = false

    func run() throws {
        let bundleURL = URL(fileURLWithPath: bundle).standardizedFileURL
        let job = try WorkflowBundleCodec.decoder().decode(
            WorkflowJobManifest.self,
            from: Data(contentsOf: bundleURL.appendingPathComponent(WorkflowJobManifest.filename))
        )
        try validateWorker(.local(), for: job, executor: "worker")
        let outcome = try WorkflowRunner(
            bundleDirectory: bundleURL,
            runDirectory: URL(fileURLWithPath: runDirectory),
            resume: resume,
            executor: .init(kind: "worker", profile: nil, jobReference: nil),
            eventHandler: jsonStream ? { event in
                emitGraphStreamEvent(event)
            } : nil
        ).execute()
        if !jsonStream { print(try StructuredRunOutput.encode(outcome)) }
        if outcome.state == .failed || outcome.state == .cancelled { throw ExitCode.failure }
    }
}

struct GraphWorkerInspect: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "inspect", abstract: "Inspect a worker run manifest.")

    @Option(name: [.customLong("run-dir")], help: "Worker run directory.")
    var runDirectory: String

    @Flag(name: [.long], help: "Emit the run manifest as JSON.")
    var json = false

    func run() throws {
        let url = URL(fileURLWithPath: runDirectory).appendingPathComponent(GraphRunManifest.filename)
        let manifest = try WorkflowBundleCodec.decoder().decode(GraphRunManifest.self, from: Data(contentsOf: url))
        if json { print(try StructuredRunOutput.encode(manifest)) } else { print("\(manifest.jobID) \(manifest.state.rawValue)") }
    }
}

struct GraphWorkerCancel: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "cancel", abstract: "Request cooperative worker cancellation.")

    @Option(name: [.customLong("run-dir")], help: "Worker run directory.")
    var runDirectory: String

    @Flag(name: [.long], help: "Emit a structured cancellation response.")
    var json = false

    func run() throws {
        let url = URL(fileURLWithPath: runDirectory).appendingPathComponent("cancel.request")
        try Data(Date().ISO8601Format().utf8).write(to: url, options: .atomic)
        WorkflowChildProcessRegistry.terminateAll(in: url.deletingLastPathComponent())
        let result = GraphCancellationResult(cancelled: true, runDirectory: url.deletingLastPathComponent().path)
        if json { print(try StructuredRunOutput.encode(result)) } else { print("Cancellation requested.") }
    }
}

struct GraphCancellationResult: Codable, Equatable {
    let cancelled: Bool
    let runDirectory: String

    enum CodingKeys: String, CodingKey {
        case cancelled
        case runDirectory = "run_directory"
    }
}

struct WorkflowGraphRequirements: Equatable {
    let nodeKinds: [String]
    let modelIDs: [String]
    let providers: [WorkflowGraphProviderRequirement]
    let secretNames: [String]
    let acceleratorBackends: [String]
    let minimumAcceleratorMemoryBytes: Int64?
    let minimumSystemMemoryBytes: Int64?
    let minimumDiskBytes: Int64?
    let minimumCPUCores: Int?
    let networkAccess: Bool

    static func resolve(graph: WorkflowGraphDocument, inputs: WorkflowInputsDocument) -> WorkflowGraphRequirements {
        var modelIDs = graph.nodes.compactMap { node -> String? in
            if let model = node.arguments["model"] {
                if case .string(let value) = model { return value }
                if case .reference(let rawReference) = model,
                   let reference = try? WorkflowReference(rawReference),
                   case .input(let name) = reference.source {
                    return (inputs.values[name] ?? graph.inputs[name]?.defaultValue)?.stringValue
                }
            }
            switch node.kind {
            case "image.train-lora": return ImageTrainLoRA.defaultManagedModelID.rawValue
            case "image.generate": return ImageGenerate.defaultManagedModelID.rawValue
            case "video.generate": return ModelResolver.ModelID.ltxVideo23AVMLX.rawValue
            default: return nil
            }
        }
        for node in graph.nodes {
            modelIDs.append(contentsOf: WorkflowNodeRegistry.entry(for: node)?.requirements.modelIDs ?? [])
        }
        var acceleratorBackends = Set(["cpu", "metal", "cuda", "rocm"])
        var minimumAcceleratorMemoryBytes: Int64?
        var minimumSystemMemoryBytes: Int64?
        var minimumDiskBytes: Int64?
        var minimumCPUCores: Int?
        var networkAccess = false
        for node in graph.nodes {
            guard let nodeRequirements = WorkflowNodeRegistry.entry(for: node)?.requirements else { continue }
            let accepted = nodeRequirements.acceleratorBackends
            if !accepted.isEmpty { acceleratorBackends.formIntersection(accepted) }
            if let minimum = nodeRequirements.minimumAcceleratorMemoryBytes {
                minimumAcceleratorMemoryBytes = max(minimumAcceleratorMemoryBytes ?? 0, minimum)
            }
            if let minimum = nodeRequirements.minimumSystemMemoryBytes {
                minimumSystemMemoryBytes = max(minimumSystemMemoryBytes ?? 0, minimum)
            }
            if let minimum = nodeRequirements.minimumDiskBytes {
                minimumDiskBytes = max(minimumDiskBytes ?? 0, minimum)
            }
            if let minimum = nodeRequirements.minimumCPUCores {
                minimumCPUCores = max(minimumCPUCores ?? 0, minimum)
            }
            networkAccess = networkAccess || nodeRequirements.networkAccess == true
        }
        return .init(
            nodeKinds: Array(Set(graph.nodes.map(\.kind))).sorted(),
            modelIDs: Array(Set(modelIDs)).sorted(),
            providers: Array(Set(graph.nodes.compactMap { node -> WorkflowGraphProviderRequirement? in
                guard node.resolvedProviderID != WorkflowNodeProviderIdentity.builtInID else { return nil }
                return WorkflowGraphProviderRegistry.discoveredCatalog().provider(id: node.resolvedProviderID)?.requirement
            })).sorted { $0.id < $1.id },
            secretNames: Array(Set(graph.nodes.flatMap { node in
                node.arguments.values.flatMap(\.secretNames)
            })).sorted(),
            acceleratorBackends: acceleratorBackends.sorted(),
            minimumAcceleratorMemoryBytes: minimumAcceleratorMemoryBytes,
            minimumSystemMemoryBytes: minimumSystemMemoryBytes,
            minimumDiskBytes: minimumDiskBytes,
            minimumCPUCores: minimumCPUCores,
            networkAccess: networkAccess
        )
    }
}

private struct LoadedGraphRequest {
    let graphURL: URL
    let inputsURL: URL?
    let graph: WorkflowGraphDocument
    let inputs: WorkflowInputsDocument
}

private func loadGraphRequest(file: String, inputsJSON: String?) throws -> LoadedGraphRequest {
    let graphURL = URL(fileURLWithPath: file).standardizedFileURL
    guard FileManager.default.fileExists(atPath: graphURL.path) else {
        throw ValidationError("Workflow graph not found: \(graphURL.path)")
    }
    let inputsURL = inputsJSON.map { URL(fileURLWithPath: $0).standardizedFileURL }
    if let inputsURL, !FileManager.default.fileExists(atPath: inputsURL.path) {
        throw ValidationError("Workflow inputs not found: \(inputsURL.path)")
    }
    return LoadedGraphRequest(
        graphURL: graphURL,
        inputsURL: inputsURL,
        graph: try WorkflowGraphDocument.load(from: graphURL),
        inputs: try WorkflowInputsDocument.load(from: inputsURL)
    )
}

private func materializationEnvelope(
    file: String,
    inputsJSON: String?,
    destination: String,
    command: [String],
    now: @escaping () -> Date = Date.init
) throws -> GraphMaterializationEnvelope {
    let loaded = try loadGraphRequest(file: file, inputsJSON: inputsJSON)
    let bundle = try WorkflowBundleMaterializer(
        graph: loaded.graph,
        suppliedInputs: loaded.inputs,
        destination: URL(fileURLWithPath: destination),
        now: now
    ).materialize()
    return GraphMaterializationEnvelope(
        schemaVersion: 1,
        mereRunVersion: MereRunCLIVersion.current,
        command: command,
        mode: .materialize,
        status: .planned,
        createdAt: now(),
        cwd: FileManager.default.currentDirectoryPath,
        summary: "Materialized workflow job \(bundle.job.jobID) at \(bundle.directory.path).",
        request: .init(
            graphPath: loaded.graphURL.path,
            inputsPath: loaded.inputsURL?.path,
            destination: bundle.directory.path
        ),
        result: .init(
            jobID: bundle.job.jobID,
            bundleDirectory: bundle.directory.path,
            graphFingerprint: bundle.job.graphFingerprint,
            inputFingerprint: bundle.job.inputFingerprint,
            assetCount: bundle.assets.groups.reduce(0) { $0 + $1.entries.count }
        ),
        diagnostics: [],
        actions: []
    )
}

private func assetDiagnostics(
    graph: WorkflowGraphDocument,
    inputs: WorkflowInputsDocument
) -> [PreflightDiagnostic] {
    var diagnostics: [PreflightDiagnostic] = []
    func inspect(path: String, expectsDirectory: Bool, id: String, label: String) {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else {
            diagnostics.append(.init(
                id: "workflow_asset_missing_\(id)",
                severity: .blocker,
                title: "Workflow asset is missing",
                message: "\(label) was not found at \(path).",
                locations: [.init(kind: expectsDirectory ? "directory" : "file", path: path)]
            ))
            return
        }
        if expectsDirectory != isDirectory.boolValue {
            diagnostics.append(.init(
                id: "workflow_asset_type_invalid_\(id)",
                severity: .blocker,
                title: "Workflow asset type is invalid",
                message: "\(label) must point to a \(expectsDirectory ? "directory" : "regular file").",
                locations: [.init(kind: expectsDirectory ? "directory" : "file", path: path)]
            ))
        }
        if (try? URL(fileURLWithPath: path).resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true {
            diagnostics.append(.init(
                id: "workflow_asset_symlink_\(id)",
                severity: .blocker,
                title: "Workflow asset is a symbolic link",
                message: "Portable workflow assets cannot be symbolic links: \(path).",
                locations: [.init(kind: expectsDirectory ? "directory" : "file", path: path)]
            ))
        }
    }
    for (name, definition) in graph.inputs where definition.type == .asset || definition.type == .assetDirectory {
        guard let path = (inputs.values[name] ?? definition.defaultValue)?.stringValue else { continue }
        inspect(
            path: path,
            expectsDirectory: definition.type == .assetDirectory,
            id: name,
            label: "Input '\(name)'"
        )
    }
    for node in graph.nodes {
        guard let catalog = WorkflowNodeRegistry.entry(for: node) else { continue }
        for field in catalog.inputs {
            guard let value = node.arguments[field.name] else { continue }
            let values: [WorkflowValue]
            if field.type == .assetArray || field.type == .assetCollection, case .array(let nested) = value {
                values = nested
            } else if field.type == .asset || field.type == .assetDirectory {
                values = [value]
            } else {
                continue
            }
            for (index, nested) in values.enumerated() {
                guard case .string(let path) = nested, !path.hasPrefix("asset://") else { continue }
                inspect(
                    path: path,
                    expectsDirectory: field.type == .assetDirectory,
                    id: "\(node.id)-\(field.name)-\(index)".replacingOccurrences(of: "_", with: "-"),
                    label: "Node '\(node.id)' argument '\(field.name)'"
                )
            }
        }
    }
    return diagnostics
}

private func emitDiagnostics(_ diagnostics: [PreflightDiagnostic]) {
    for diagnostic in diagnostics {
        FileHandle.standardError.write(Data("[\(diagnostic.severity.rawValue)] \(diagnostic.message)\n".utf8))
    }
}

func emitGraphStreamEvent(_ event: GraphRunEvent, to handle: FileHandle = .standardOutput) {
    guard let encoded = try? WorkflowBundleCodec.lineEncoder().encode(event) else { return }
    handle.write(encoded)
    handle.write(Data("\n".utf8))
}

private func structuredStatus(_ state: GraphRunState) -> StructuredRunStatus {
    switch state {
    case .planned: .planned
    case .preflighting, .running: .running
    case .queued: .queued
    case .assigned: .assigned
    case .finished: .finished
    case .failed: .failed
    case .cancelled: .cancelled
    }
}
