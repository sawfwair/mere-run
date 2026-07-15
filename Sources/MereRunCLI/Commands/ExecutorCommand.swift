import ArgumentParser
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import MereRunCore

enum WorkflowExecutorKind: String, Codable, Equatable, Sendable {
    case ssh
    case relay
}

struct WorkflowExecutorProfile: Codable, Equatable, Sendable {
    let name: String
    let kind: WorkflowExecutorKind
    let destination: String?
    let remoteRoot: String?
    let port: Int?
    let identityFile: String?
    let mereRunPath: String?
    let url: String?
    let tokenFile: String?

    enum CodingKeys: String, CodingKey {
        case name
        case kind
        case destination
        case remoteRoot = "remote_root"
        case port
        case identityFile = "identity_file"
        case mereRunPath = "mere_run_path"
        case url
        case tokenFile = "token_file"
    }
}

struct WorkflowExecutorProfiles: Codable, Equatable {
    let schemaVersion: Int
    var profiles: [WorkflowExecutorProfile]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case profiles
    }
}

enum WorkflowExecutorProfileStore {
    static var fileURL: URL {
        MereRunModelPaths.applicationSupportBase.appendingPathComponent("executors.json")
    }

    static func load(
        from url: URL = fileURL,
        fileManager: FileManager = .default
    ) throws -> WorkflowExecutorProfiles {
        guard fileManager.fileExists(atPath: url.path) else {
            return WorkflowExecutorProfiles(schemaVersion: 1, profiles: [])
        }
        let profiles = try JSONDecoder().decode(WorkflowExecutorProfiles.self, from: Data(contentsOf: url))
        guard profiles.schemaVersion == 1 else {
            throw ValidationError("Unsupported executor profile schema_version \(profiles.schemaVersion).")
        }
        return profiles
    }

    static func save(
        _ value: WorkflowExecutorProfiles,
        to url: URL = fileURL,
        fileManager: FileManager = .default
    ) throws {
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try WorkflowBundleCodec.encoder().encode(value).write(to: url, options: .atomic)
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    static func require(_ reference: String) throws -> WorkflowExecutorProfile {
        let parsed = try WorkflowExecutorReference(reference)
        guard parsed.kind != nil else {
            throw ValidationError("Executor profile references must include ssh: or relay:.")
        }
        guard let profile = try load().profiles.first(where: { $0.name == parsed.name }) else {
            throw ValidationError("Executor profile not found: \(parsed.name)")
        }
        guard profile.kind == parsed.kind else {
            throw ValidationError("Executor profile '\(parsed.name)' is not a \(parsed.kind!.rawValue) profile.")
        }
        return profile
    }
}

struct WorkflowExecutorReference: Equatable {
    let kind: WorkflowExecutorKind?
    let name: String

    init(_ value: String) throws {
        if value == "local" {
            kind = nil
            name = "local"
            return
        }
        let parts = value.split(separator: ":", maxSplits: 1).map(String.init)
        guard parts.count == 2,
              let parsedKind = WorkflowExecutorKind(rawValue: parts[0]),
              isValidExecutorName(parts[1]) else {
            throw ValidationError("Executor must be local, ssh:<profile>, or relay:<profile>.")
        }
        kind = parsedKind
        name = parts[1]
    }
}

struct WorkflowRemoteReference: Equatable {
    let kind: WorkflowExecutorKind
    let profile: String
    let jobID: String

    init(_ value: String) throws {
        guard let url = URL(string: value),
              let scheme = url.scheme,
              let kind = WorkflowExecutorKind(rawValue: scheme),
              let profile = url.host,
              isValidExecutorName(profile),
              url.query == nil,
              url.fragment == nil else {
            throw ValidationError("Run reference must be ssh://<profile>/<job-id> or relay://<profile>/<job-id>.")
        }
        let jobID = url.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard !jobID.isEmpty,
              !jobID.contains("/"),
              jobID.range(of: "^[A-Za-z0-9][A-Za-z0-9-]{0,127}$", options: .regularExpression) != nil else {
            throw ValidationError("Remote job id is invalid.")
        }
        self.kind = kind
        self.profile = profile
        self.jobID = jobID
    }

    var executorReference: String { "\(kind.rawValue):\(profile)" }
    var rawValue: String { "\(kind.rawValue)://\(profile)/\(jobID)" }
}

struct WorkflowExecutorProbe: Codable, Equatable, Sendable {
    let schemaVersion: Int
    let workerVersion: String
    let contractVersions: [String]
    let platform: String
    let architecture: String
    let acceleratorBackend: String
    let memoryBytes: UInt64
    let availableDiskBytes: Int64?
    let nodeKinds: [String]
    let installedModelIDs: [String]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case workerVersion = "worker_version"
        case contractVersions = "contract_versions"
        case platform
        case architecture
        case acceleratorBackend = "accelerator_backend"
        case memoryBytes = "memory_bytes"
        case availableDiskBytes = "available_disk_bytes"
        case nodeKinds = "node_kinds"
        case installedModelIDs = "installed_model_ids"
    }

    var summary: String {
        "\(platform)/\(architecture) \(acceleratorBackend), \(installedModelIDs.count) installed model(s)"
    }

    static func local(fileManager: FileManager = .default) -> WorkflowExecutorProbe {
        #if os(Linux)
        let platform = "linux"
        let backend = ProcessInfo.processInfo.environment["MERERUN_LINUX_ACCEL"] ?? "cpu"
        let memoryBytes = backend == "cuda"
            ? linuxNVIDIAMemoryBytes() ?? ProcessInfo.processInfo.physicalMemory
            : ProcessInfo.processInfo.physicalMemory
        #else
        let platform = "macos"
        let backend = "metal"
        let memoryBytes = ProcessInfo.processInfo.physicalMemory
        #endif
        #if arch(arm64)
        let architecture = "arm64"
        #elseif arch(x86_64)
        let architecture = "x86_64"
        #else
        let architecture = "unknown"
        #endif
        let fileSystemAttributes = try? FileManager.default.attributesOfFileSystem(
            forPath: MereRunModelPaths.applicationSupportBase.path
        )
        let disk = (fileSystemAttributes?[.systemFreeSize] as? NSNumber)?.int64Value
        return WorkflowExecutorProbe(
            schemaVersion: 1,
            workerVersion: MereRunCLIVersion.current,
            contractVersions: [WorkflowJobManifest.contractVersion],
            platform: platform,
            architecture: architecture,
            acceleratorBackend: backend,
            memoryBytes: memoryBytes,
            availableDiskBytes: disk ?? nil,
            nodeKinds: WorkflowNodeRegistry.entries.map(\.kind).sorted(),
            installedModelIDs: ModelInventory.rows(fileManager: fileManager).filter(\.isInstalled).map(\.id).sorted()
        )
    }

    #if os(Linux)
    private static func linuxNVIDIAMemoryBytes() -> UInt64? {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ]
        process.standardOutput = output
        process.standardError = Pipe()
        guard (try? process.run()) != nil else { return nil }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        return parseNVIDIAMemoryBytes(String(decoding: data, as: UTF8.self))
    }
    #endif

    static func parseNVIDIAMemoryBytes(_ output: String) -> UInt64? {
        output.split(whereSeparator: \Character.isNewline)
            .compactMap { line in
                line.split(whereSeparator: \Character.isWhitespace).first.flatMap { UInt64($0) }
            }
            .max()
            .map { $0 * 1_024 * 1_024 }
    }
}

struct WorkflowRemoteJob: Codable, Equatable, Sendable {
    let jobID: String
    let jobReference: String
    let state: GraphRunState
    let executor: String
    let runDirectory: String?
    let createdAt: Date?
    let updatedAt: Date?
    let artifacts: [GraphRunArtifact]
    let error: String?
    let placement: WorkflowGraphPlacement?
    let metrics: WorkflowGraphExecutionMetrics?

    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case jobReference = "job_reference"
        case state
        case executor
        case runDirectory = "run_directory"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case artifacts
        case error
        case placement
        case metrics
    }
}

struct WorkflowGraphExecutionMetrics: Codable, Equatable, Sendable {
    let bundleBytesDownloaded: UInt64
    let downloadMilliseconds: UInt64
    let executionMilliseconds: UInt64
    let uploadMilliseconds: UInt64
    let totalMilliseconds: UInt64
    let artifactBytesUploaded: UInt64
    let artifactPartsUploaded: UInt64
    let artifactBytesReused: UInt64
    let artifactPartsReused: UInt64

    enum CodingKeys: String, CodingKey {
        case bundleBytesDownloaded = "bundle_bytes_downloaded"
        case downloadMilliseconds = "download_ms"
        case executionMilliseconds = "execution_ms"
        case uploadMilliseconds = "upload_ms"
        case totalMilliseconds = "total_ms"
        case artifactBytesUploaded = "artifact_bytes_uploaded"
        case artifactPartsUploaded = "artifact_parts_uploaded"
        case artifactBytesReused = "artifact_bytes_reused"
        case artifactPartsReused = "artifact_parts_reused"
    }
}

struct WorkflowGraphPlacementBlocker: Codable, Equatable, Sendable {
    let code: String
    let message: String
}

struct WorkflowGraphPlacementNode: Codable, Equatable, Sendable {
    let agentID: String
    let deviceID: String
    let deviceName: String
    let status: String
    let eligible: Bool
    let blockers: [WorkflowGraphPlacementBlocker]

    enum CodingKeys: String, CodingKey {
        case agentID = "agent_id"
        case deviceID = "device_id"
        case deviceName = "device_name"
        case status
        case eligible
        case blockers
    }
}

struct WorkflowGraphPlacement: Codable, Equatable, Sendable {
    let connectedNodes: Int
    let graphWorkerNodes: Int
    let eligibleNodes: Int
    let diagnostic: String?
    let nodes: [WorkflowGraphPlacementNode]

    enum CodingKeys: String, CodingKey {
        case connectedNodes = "connected_nodes"
        case graphWorkerNodes = "graph_worker_nodes"
        case eligibleNodes = "eligible_nodes"
        case diagnostic
        case nodes
    }
}

struct RelayFleetGraphWorker: Codable, Equatable, Sendable {
    let workerVersion: String
    let contractVersions: [String]
    let acceleratorBackend: String
    let memoryBytes: UInt64
    let availableDiskBytes: Int64?
    let nodeKinds: [String]
    let installedModelIDs: [String]

    enum CodingKeys: String, CodingKey {
        case workerVersion = "worker_version"
        case contractVersions = "contract_versions"
        case acceleratorBackend = "accelerator_backend"
        case memoryBytes = "memory_bytes"
        case availableDiskBytes = "available_disk_bytes"
        case nodeKinds = "node_kinds"
        case installedModelIDs = "installed_model_ids"
    }
}

struct RelayFleetCapabilities: Codable, Equatable, Sendable {
    let models: [String]
    let graphWorker: RelayFleetGraphWorker?

    enum CodingKeys: String, CodingKey {
        case models
        case graphWorker = "graph_worker"
    }
}

struct RelayFleetRuntime: Codable, Equatable, Sendable {
    let mereRunVersion: String?
    let installedModels: [String]

    enum CodingKeys: String, CodingKey {
        case mereRunVersion = "mere_run_version"
        case installedModels = "installed_models"
    }
}

struct RelayFleetPolicy: Codable, Equatable, Sendable {
    let enabled: Bool
    let draining: Bool
    let revoked: Bool
    let priority: Int
    let preferredModels: [String]
    let displayName: String?

    enum CodingKeys: String, CodingKey {
        case enabled
        case draining
        case revoked
        case priority
        case preferredModels = "preferred_models"
        case displayName = "display_name"
    }
}

struct RelayFleetNode: Codable, Equatable, Sendable {
    let agentID: String
    let deviceID: String
    let deviceName: String
    let reportedName: String
    let version: String
    let status: String
    let currentJobID: String?
    let lastSeen: String
    let capabilities: RelayFleetCapabilities
    let runtime: RelayFleetRuntime?
    let policy: RelayFleetPolicy

    enum CodingKeys: String, CodingKey {
        case agentID = "agent_id"
        case deviceID = "device_id"
        case deviceName = "device_name"
        case reportedName = "reported_name"
        case version
        case status
        case currentJobID = "current_job_id"
        case lastSeen = "last_seen"
        case capabilities
        case runtime
        case policy
    }
}

struct RelayFleetActivity: Codable, Equatable, Sendable {
    let id: String
    let kind: String
    let status: String
    let agentID: String?
    let model: String?
    let label: String
    let createdAt: String
    let durationMilliseconds: Int?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case status
        case agentID = "agent_id"
        case model
        case label
        case createdAt = "created_at"
        case durationMilliseconds = "duration_ms"
        case error
    }
}

struct RelayFleetSummary: Codable, Equatable, Sendable {
    let totalNodes: Int
    let onlineNodes: Int
    let busyNodes: Int
    let availableNodes: Int
    let queueDepth: Int
    let installedModels: Int
    let routableModels: Int

    enum CodingKeys: String, CodingKey {
        case totalNodes = "total_nodes"
        case onlineNodes = "online_nodes"
        case busyNodes = "busy_nodes"
        case availableNodes = "available_nodes"
        case queueDepth = "queue_depth"
        case installedModels = "installed_models"
        case routableModels = "routable_models"
    }
}

struct RelayFleetSnapshot: Codable, Equatable, Sendable {
    let generatedAt: String
    let summary: RelayFleetSummary
    let nodes: [RelayFleetNode]
    let activity: [RelayFleetActivity]

    enum CodingKeys: String, CodingKey {
        case generatedAt = "generated_at"
        case summary
        case nodes
        case activity
    }
}

struct RelayFleetRefreshResult: Codable, Equatable, Sendable {
    let deviceID: String
    let requested: Bool

    enum CodingKeys: String, CodingKey {
        case deviceID = "device_id"
        case requested
    }
}

struct RelayFleetNodePolicyPatch: Codable, Equatable, Sendable {
    let enabled: Bool?
    let draining: Bool?
    let priority: Int?
    let displayName: String?

    enum CodingKeys: String, CodingKey {
        case enabled
        case draining
        case priority
        case displayName = "display_name"
    }
}

struct WorkflowExecutorListResult: Codable, Equatable {
    let profiles: [WorkflowExecutorProfile]
}

struct Executor: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "executor",
        abstract: "Manage local, SSH, and relay workflow executors.",
        subcommands: [
            ExecutorAdd.self,
            ExecutorList.self,
            ExecutorInspect.self,
            ExecutorProbe.self,
            ExecutorLogin.self,
            ExecutorAuthStatus.self,
            ExecutorLogout.self,
            ExecutorFleet.self,
            ExecutorNodeRefresh.self,
            ExecutorNodeConfigure.self,
            ExecutorRemove.self,
        ]
    )
}

struct ExecutorAdd: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "add",
        abstract: "Add an executor profile.",
        subcommands: [ExecutorAddSSH.self, ExecutorAddRelay.self]
    )
}

struct ExecutorAddSSH: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "ssh", abstract: "Add an SSH executor.")

    @Argument(help: "Profile name.") var name: String
    @Option(name: [.long], help: "SSH destination, such as user@host.") var destination: String
    @Option(name: [.customLong("remote-root")], help: "Remote cache and job root.") var remoteRoot: String
    @Option(name: [.long], help: "SSH port.") var port: Int?
    @Option(name: [.customLong("identity-file")], help: "SSH identity file.") var identityFile: String?
    @Option(name: [.customLong("mere-run-path")], help: "Remote mere.run executable path.") var mereRunPath = "mere.run"
    @Flag(name: [.long], help: "Emit the saved profile as JSON.") var json = false

    func run() throws {
        try validateExecutorName(name)
        guard !destination.isEmpty, !remoteRoot.isEmpty,
              !destination.hasPrefix("-"),
              !destination.contains(where: { $0.isNewline || $0 == "\0" }),
              !remoteRoot.contains(where: { $0.isNewline || $0 == "\0" }) else {
            throw ValidationError("--destination and --remote-root must not be empty.")
        }
        if let port, !(1...65_535).contains(port) {
            throw ValidationError("--port must be between 1 and 65535.")
        }
        let profile = WorkflowExecutorProfile(
            name: name,
            kind: .ssh,
            destination: destination,
            remoteRoot: remoteRoot,
            port: port,
            identityFile: identityFile.map(expandPath),
            mereRunPath: mereRunPath,
            url: nil,
            tokenFile: nil
        )
        try saveProfile(profile)
        if json { print(try StructuredRunOutput.encode(profile)) } else { print("Saved ssh:\(name)") }
    }
}

struct ExecutorAddRelay: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "relay", abstract: "Add a relay executor.")

    @Argument(help: "Profile name.") var name: String
    @Option(name: [.long], help: "Relay base URL. No default is assumed.") var url: String
    @Option(name: [.customLong("token-file")], help: "File containing a relay bearer token.") var tokenFile: String?
    @Flag(name: [.long], help: "Emit the saved profile as JSON.") var json = false

    func run() throws {
        try validateExecutorName(name)
        guard let parsed = URL(string: url),
              parsed.user == nil,
              parsed.password == nil,
              parsed.query == nil,
              parsed.fragment == nil,
              parsed.scheme == "https" || isLoopbackHTTP(parsed) else {
            throw ValidationError("Relay URL must use HTTPS, except for loopback development URLs.")
        }
        let profile = WorkflowExecutorProfile(
            name: name,
            kind: .relay,
            destination: nil,
            remoteRoot: nil,
            port: nil,
            identityFile: nil,
            mereRunPath: nil,
            url: parsed.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/")),
            tokenFile: tokenFile.map(expandPath)
        )
        try saveProfile(profile)
        if json { print(try StructuredRunOutput.encode(profile)) } else { print("Saved relay:\(name)") }
    }
}

struct ExecutorList: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "list", abstract: "List executor profiles.")
    @Flag(name: [.long], help: "Emit executor profiles as JSON.") var json = false

    func run() throws {
        let profiles = try WorkflowExecutorProfileStore.load().profiles.sorted { $0.name < $1.name }
        if json {
            print(try StructuredRunOutput.encode(WorkflowExecutorListResult(profiles: profiles)))
        } else {
            for profile in profiles { print("\(profile.kind.rawValue):\(profile.name)") }
        }
    }
}

struct ExecutorInspect: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "inspect", abstract: "Inspect an executor profile.")
    @Argument(help: "Executor reference.") var reference: String
    @Flag(name: [.long], help: "Emit the profile as JSON.") var json = false

    func run() throws {
        let profile = try WorkflowExecutorProfileStore.require(reference)
        if json { print(try StructuredRunOutput.encode(profile)) } else { print("\(profile.kind.rawValue):\(profile.name)") }
    }
}

struct ExecutorProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "probe", abstract: "Probe executor capabilities.")
    @Argument(help: "Executor reference.") var reference: String
    @Flag(name: [.long], help: "Emit capabilities as JSON.") var json = false

    func run() async throws {
        let probe = try await WorkflowExecutorController.probe(reference: reference)
        if json { print(try StructuredRunOutput.encode(probe)) } else { print(probe.summary) }
    }
}

struct ExecutorLogin: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "login",
        abstract: "Sign in to a relay executor through its advertised device authorization flow."
    )

    @Argument(help: "Relay executor reference.") var reference: String
    @Flag(name: [.long], help: "Emit the saved authentication result as JSON.") var json = false

    func run() async throws {
        let profile = try requireRelayProfile(reference)
        let tokenFile = profile.tokenFile.map { URL(fileURLWithPath: $0) }
            ?? RelayAuthentication.defaultTokenFile(profileName: profile.name)
        let result = try await RelayAuthentication.login(
            profile: profile,
            tokenFile: tokenFile,
            progress: { message in
                FileHandle.standardError.write(Data("\(message)\n".utf8))
            }
        )
        if profile.tokenFile == nil {
            try saveProfile(profile.withTokenFile(tokenFile.path))
        }
        if json {
            print(try StructuredRunOutput.encode(result))
        } else {
            print("Signed in to relay:\(profile.name)")
        }
    }
}

struct ExecutorAuthStatus: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "auth-status",
        abstract: "Inspect relay authentication without printing credentials."
    )

    @Argument(help: "Relay executor reference.") var reference: String
    @Flag(name: [.long], help: "Emit authentication status as JSON.") var json = false

    func run() throws {
        let profile = try requireRelayProfile(reference)
        let result = RelayAuthentication.status(profile: profile)
        if json {
            print(try StructuredRunOutput.encode(result))
        } else {
            print(result.authenticated ? "authenticated" : "not authenticated")
        }
    }
}

struct ExecutorLogout: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "logout",
        abstract: "Remove the saved credential for a relay executor."
    )

    @Argument(help: "Relay executor reference.") var reference: String
    @Flag(name: [.long], help: "Emit a structured result as JSON.") var json = false

    func run() throws {
        let profile = try requireRelayProfile(reference)
        var removed = false
        if let tokenFile = profile.tokenFile, FileManager.default.fileExists(atPath: tokenFile) {
            try FileManager.default.removeItem(atPath: tokenFile)
            removed = true
        }
        let result = RelayLogoutResult(executor: "relay:\(profile.name)", removed: removed)
        if json {
            print(try StructuredRunOutput.encode(result))
        } else {
            print(removed ? "Signed out of relay:\(profile.name)" : "No saved relay credential")
        }
    }
}

struct ExecutorFleet: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "fleet",
        abstract: "Inspect relay node identity, eligibility, inventory, and recent work."
    )

    @Argument(help: "Relay executor reference.") var reference: String
    @Flag(name: [.long], help: "Emit the fleet snapshot as JSON.") var json = false

    func run() async throws {
        let profile = try requireRelayProfile(reference)
        let fleet = try await RelayWorkflowExecutor(profile: profile).fleet()
        if json {
            print(try StructuredRunOutput.encode(fleet))
            return
        }
        print("\(fleet.summary.availableNodes)/\(fleet.summary.totalNodes) nodes available; "
            + "\(fleet.summary.queueDepth) queued")
        for node in fleet.nodes {
            let models = node.capabilities.graphWorker?.installedModelIDs
                ?? node.runtime?.installedModels
                ?? node.capabilities.models
            print("\(node.deviceID)\t\(node.status)\t\(node.deviceName)\t\(models.joined(separator: ","))")
        }
        for activity in fleet.activity.prefix(10) {
            print("\(activity.kind)\t\(activity.status)\t\(activity.id)\t\(activity.label)")
        }
    }
}

struct ExecutorNodeRefresh: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "node-refresh",
        abstract: "Ask a connected relay node to rescan its capabilities and installed models."
    )

    @Argument(help: "Relay executor reference.") var reference: String
    @Argument(help: "Stable fleet device id.") var deviceID: String
    @Flag(name: [.long], help: "Emit the request result as JSON.") var json = false

    func run() async throws {
        let profile = try requireRelayProfile(reference)
        let result = try await RelayWorkflowExecutor(profile: profile).refreshNode(deviceID: deviceID)
        if json {
            print(try StructuredRunOutput.encode(result))
        } else {
            print("Inventory refresh requested for \(result.deviceID)")
        }
    }
}

struct ExecutorNodeConfigure: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "node-configure",
        abstract: "Update relay scheduling policy for a fleet node."
    )

    @Argument(help: "Relay executor reference.") var reference: String
    @Argument(help: "Stable fleet device id.") var deviceID: String
    @Flag(name: [.customLong("enable")], help: "Allow the node to accept work.") var enable = false
    @Flag(name: [.customLong("disable")], help: "Prevent the node from accepting work.") var disable = false
    @Flag(name: [.customLong("drain")], help: "Finish current work without accepting new work.") var drain = false
    @Flag(name: [.customLong("undrain")], help: "Allow new work after draining.") var undrain = false
    @Option(name: [.long], help: "Scheduler priority from 0 through 100.") var priority: Int?
    @Option(name: [.customLong("display-name")], help: "Operator-facing node name.") var displayName: String?
    @Flag(name: [.long], help: "Emit the updated node as JSON.") var json = false

    func run() async throws {
        guard !(enable && disable), !(drain && undrain) else {
            throw ValidationError("Choose only one of --enable/--disable and --drain/--undrain.")
        }
        if let priority, !(0...100).contains(priority) {
            throw ValidationError("--priority must be between 0 and 100.")
        }
        let patch = RelayFleetNodePolicyPatch(
            enabled: enable ? true : disable ? false : nil,
            draining: drain ? true : undrain ? false : nil,
            priority: priority,
            displayName: displayName
        )
        guard patch.enabled != nil || patch.draining != nil || patch.priority != nil || patch.displayName != nil else {
            throw ValidationError("Specify at least one node policy change.")
        }
        let profile = try requireRelayProfile(reference)
        let node = try await RelayWorkflowExecutor(profile: profile).configureNode(
            deviceID: deviceID,
            patch: patch
        )
        if json {
            print(try StructuredRunOutput.encode(node))
        } else {
            print("Updated \(node.deviceID): \(node.status), priority \(node.policy.priority)")
        }
    }
}

struct ExecutorRemove: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "remove", abstract: "Remove an executor profile.")
    @Argument(help: "Profile name or executor reference.") var reference: String
    @Flag(name: [.long], help: "Emit a structured result.") var json = false

    func run() throws {
        let name = (try? WorkflowExecutorReference(reference).name) ?? reference
        var profiles = try WorkflowExecutorProfileStore.load()
        guard profiles.profiles.contains(where: { $0.name == name }) else {
            throw ValidationError("Executor profile not found: \(name)")
        }
        profiles.profiles.removeAll { $0.name == name }
        try WorkflowExecutorProfileStore.save(profiles)
        let result = ExecutorRemovalResult(name: name, removed: true)
        if json { print(try StructuredRunOutput.encode(result)) } else { print("Removed \(name)") }
    }
}

struct ExecutorRemovalResult: Codable, Equatable {
    let name: String
    let removed: Bool
}

enum WorkflowExecutorController {
    static func probe(reference: String) async throws -> WorkflowExecutorProbe {
        let parsed = try WorkflowExecutorReference(reference)
        guard let kind = parsed.kind else { return .local() }
        let profile = try WorkflowExecutorProfileStore.require(reference)
        switch kind {
        case .ssh:
            return try SSHWorkflowExecutor(profile: profile).probe()
        case .relay:
            return try await RelayWorkflowExecutor(profile: profile).probe()
        }
    }

    static func submit(
        reference: String,
        bundleDirectory: URL,
        localRunDirectory: URL
    ) async throws -> WorkflowRemoteJob {
        let parsed = try WorkflowExecutorReference(reference)
        guard let kind = parsed.kind else {
            throw ValidationError("Use `graph run` for local execution.")
        }
        let profile = try WorkflowExecutorProfileStore.require(reference)
        switch kind {
        case .ssh:
            return try SSHWorkflowExecutor(profile: profile).submit(
                bundleDirectory: bundleDirectory,
                localRunDirectory: localRunDirectory
            )
        case .relay:
            return try await RelayWorkflowExecutor(profile: profile).submit(
                bundleDirectory: bundleDirectory,
                localRunDirectory: localRunDirectory
            )
        }
    }
}

enum WorkflowRemoteJobController {
    static func inspect(_ reference: WorkflowRemoteReference) async throws -> WorkflowRemoteJob {
        let profile = try WorkflowExecutorProfileStore.require(reference.executorReference)
        switch reference.kind {
        case .ssh:
            return try SSHWorkflowExecutor(profile: profile).inspect(jobID: reference.jobID)
        case .relay:
            return try await RelayWorkflowExecutor(profile: profile).inspect(jobID: reference.jobID)
        }
    }

    static func events(_ reference: WorkflowRemoteReference) async throws -> String {
        let profile = try WorkflowExecutorProfileStore.require(reference.executorReference)
        switch reference.kind {
        case .ssh:
            return try SSHWorkflowExecutor(profile: profile).events(jobID: reference.jobID)
        case .relay:
            return try await RelayWorkflowExecutor(profile: profile).events(jobID: reference.jobID)
        }
    }

    static func cancel(_ reference: WorkflowRemoteReference) async throws -> WorkflowRemoteJob {
        let profile = try WorkflowExecutorProfileStore.require(reference.executorReference)
        switch reference.kind {
        case .ssh:
            return try SSHWorkflowExecutor(profile: profile).cancel(jobID: reference.jobID)
        case .relay:
            return try await RelayWorkflowExecutor(profile: profile).cancel(jobID: reference.jobID)
        }
    }

    static func retry(_ reference: WorkflowRemoteReference) async throws -> WorkflowRemoteJob {
        guard reference.kind == .relay else {
            throw ValidationError("Remote retry is supported for relay jobs; resubmit the immutable bundle for SSH.")
        }
        let profile = try WorkflowExecutorProfileStore.require(reference.executorReference)
        return try await RelayWorkflowExecutor(profile: profile).retry(jobID: reference.jobID)
    }

    static func fetch(
        _ reference: WorkflowRemoteReference,
        into destination: URL,
        allArtifacts: Bool
    ) async throws -> WorkflowRemoteJob {
        let profile = try WorkflowExecutorProfileStore.require(reference.executorReference)
        switch reference.kind {
        case .ssh:
            return try SSHWorkflowExecutor(profile: profile).fetch(
                jobID: reference.jobID,
                into: destination,
                allArtifacts: allArtifacts
            )
        case .relay:
            return try await RelayWorkflowExecutor(profile: profile).fetch(
                jobID: reference.jobID,
                into: destination,
                allArtifacts: allArtifacts
            )
        }
    }

    static func list(executor reference: String, limit: Int) async throws -> [WorkflowRemoteJob] {
        let parsed = try WorkflowExecutorReference(reference)
        guard parsed.kind == .relay else {
            throw ValidationError("Remote run listing is supported for relay executor profiles.")
        }
        let profile = try WorkflowExecutorProfileStore.require(reference)
        return try await RelayWorkflowExecutor(profile: profile).list(limit: limit)
    }
}

private func saveProfile(_ profile: WorkflowExecutorProfile) throws {
    var profiles = try WorkflowExecutorProfileStore.load()
    profiles.profiles.removeAll { $0.name == profile.name }
    profiles.profiles.append(profile)
    profiles.profiles.sort { $0.name < $1.name }
    try WorkflowExecutorProfileStore.save(profiles)
}

private func requireRelayProfile(_ reference: String) throws -> WorkflowExecutorProfile {
    let profile = try WorkflowExecutorProfileStore.require(reference)
    guard profile.kind == .relay else {
        throw ValidationError("Relay authentication commands require relay:<profile>.")
    }
    return profile
}

private extension WorkflowExecutorProfile {
    func withTokenFile(_ path: String) -> WorkflowExecutorProfile {
        WorkflowExecutorProfile(
            name: name,
            kind: kind,
            destination: destination,
            remoteRoot: remoteRoot,
            port: port,
            identityFile: identityFile,
            mereRunPath: mereRunPath,
            url: url,
            tokenFile: path
        )
    }
}

private func isValidExecutorName(_ name: String) -> Bool {
    name.range(of: "^[a-z][a-z0-9-]{0,63}$", options: .regularExpression) != nil
}

private func validateExecutorName(_ name: String) throws {
    guard isValidExecutorName(name) else {
        throw ValidationError("Executor profile names must match [a-z][a-z0-9-]{0,63}.")
    }
}

private func expandPath(_ path: String) -> String {
    NSString(string: path).expandingTildeInPath
}

private func isLoopbackHTTP(_ url: URL) -> Bool {
    guard url.scheme == "http" else { return false }
    return url.host == "127.0.0.1" || url.host == "localhost" || url.host == "::1"
}
