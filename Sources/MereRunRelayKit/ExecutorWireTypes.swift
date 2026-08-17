import Foundation

public struct WorkflowExecutorProbe: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let workerVersion: String
    public let contractVersions: [String]
    public let platform: String
    public let architecture: String
    public let acceleratorBackend: String
    public let memoryBytes: UInt64
    public let systemMemoryBytes: UInt64
    public let logicalCPUCores: Int
    public let availableDiskBytes: Int64?
    public let networkAccess: Bool
    public let nodeKinds: [String]
    public let installedModelIDs: [String]
    public let availableSecretNames: [String]
    public let providers: [WorkflowGraphProviderRequirement]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case workerVersion = "worker_version"
        case contractVersions = "contract_versions"
        case platform
        case architecture
        case acceleratorBackend = "accelerator_backend"
        case memoryBytes = "memory_bytes"
        case systemMemoryBytes = "system_memory_bytes"
        case logicalCPUCores = "logical_cpu_cores"
        case availableDiskBytes = "available_disk_bytes"
        case networkAccess = "network_access"
        case nodeKinds = "node_kinds"
        case installedModelIDs = "installed_model_ids"
        case availableSecretNames = "available_secret_names"
        case providers
    }

    public init(
        schemaVersion: Int,
        workerVersion: String,
        contractVersions: [String],
        platform: String,
        architecture: String,
        acceleratorBackend: String,
        memoryBytes: UInt64,
        systemMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        logicalCPUCores: Int = ProcessInfo.processInfo.processorCount,
        availableDiskBytes: Int64?,
        networkAccess: Bool = true,
        nodeKinds: [String],
        installedModelIDs: [String],
        availableSecretNames: [String] = [],
        providers: [WorkflowGraphProviderRequirement] = []
    ) {
        self.schemaVersion = schemaVersion
        self.workerVersion = workerVersion
        self.contractVersions = contractVersions
        self.platform = platform
        self.architecture = architecture
        self.acceleratorBackend = acceleratorBackend
        self.memoryBytes = memoryBytes
        self.systemMemoryBytes = systemMemoryBytes
        self.logicalCPUCores = logicalCPUCores
        self.availableDiskBytes = availableDiskBytes
        self.networkAccess = networkAccess
        self.nodeKinds = nodeKinds
        self.installedModelIDs = installedModelIDs
        self.availableSecretNames = availableSecretNames
        self.providers = providers
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
        workerVersion = try container.decode(String.self, forKey: .workerVersion)
        contractVersions = try container.decode([String].self, forKey: .contractVersions)
        platform = try container.decode(String.self, forKey: .platform)
        architecture = try container.decode(String.self, forKey: .architecture)
        acceleratorBackend = try container.decode(String.self, forKey: .acceleratorBackend)
        memoryBytes = try container.decode(UInt64.self, forKey: .memoryBytes)
        systemMemoryBytes = try container.decodeIfPresent(UInt64.self, forKey: .systemMemoryBytes)
            ?? 0
        logicalCPUCores = try container.decodeIfPresent(Int.self, forKey: .logicalCPUCores)
            ?? 0
        availableDiskBytes = try container.decodeIfPresent(Int64.self, forKey: .availableDiskBytes)
        networkAccess = try container.decodeIfPresent(Bool.self, forKey: .networkAccess) ?? false
        nodeKinds = try container.decode([String].self, forKey: .nodeKinds)
        installedModelIDs = try container.decode([String].self, forKey: .installedModelIDs)
        availableSecretNames = try container.decodeIfPresent([String].self, forKey: .availableSecretNames) ?? []
        providers = try container.decodeIfPresent(
            [WorkflowGraphProviderRequirement].self,
            forKey: .providers
        ) ?? []
    }

    public var summary: String {
        "\(platform)/\(architecture) \(acceleratorBackend), \(installedModelIDs.count) installed model(s)"
    }
}

public struct WorkflowRemoteJob: Codable, Equatable, Sendable {
    public let jobID: String
    public let jobReference: String
    public let state: GraphRunState
    public let executor: String
    public let runDirectory: String?
    public let createdAt: Date?
    public let updatedAt: Date?
    public let artifacts: [GraphRunArtifact]
    public let error: String?
    public let placement: WorkflowGraphPlacement?
    public let metrics: WorkflowGraphExecutionMetrics?

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

    public init(
        jobID: String,
        jobReference: String,
        state: GraphRunState,
        executor: String,
        runDirectory: String?,
        createdAt: Date?,
        updatedAt: Date?,
        artifacts: [GraphRunArtifact],
        error: String?,
        placement: WorkflowGraphPlacement?,
        metrics: WorkflowGraphExecutionMetrics?
    ) {
        self.jobID = jobID
        self.jobReference = jobReference
        self.state = state
        self.executor = executor
        self.runDirectory = runDirectory
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.artifacts = artifacts
        self.error = error
        self.placement = placement
        self.metrics = metrics
    }
}

public struct WorkflowGraphExecutionMetrics: Codable, Equatable, Sendable {
    public let bundleBytesDownloaded: UInt64
    public let downloadMilliseconds: UInt64
    public let executionMilliseconds: UInt64
    public let uploadMilliseconds: UInt64
    public let totalMilliseconds: UInt64
    public let artifactBytesUploaded: UInt64
    public let artifactPartsUploaded: UInt64
    public let artifactBytesReused: UInt64
    public let artifactPartsReused: UInt64

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

    public init(
        bundleBytesDownloaded: UInt64,
        downloadMilliseconds: UInt64,
        executionMilliseconds: UInt64,
        uploadMilliseconds: UInt64,
        totalMilliseconds: UInt64,
        artifactBytesUploaded: UInt64,
        artifactPartsUploaded: UInt64,
        artifactBytesReused: UInt64,
        artifactPartsReused: UInt64
    ) {
        self.bundleBytesDownloaded = bundleBytesDownloaded
        self.downloadMilliseconds = downloadMilliseconds
        self.executionMilliseconds = executionMilliseconds
        self.uploadMilliseconds = uploadMilliseconds
        self.totalMilliseconds = totalMilliseconds
        self.artifactBytesUploaded = artifactBytesUploaded
        self.artifactPartsUploaded = artifactPartsUploaded
        self.artifactBytesReused = artifactBytesReused
        self.artifactPartsReused = artifactPartsReused
    }
}

public struct WorkflowGraphPlacementBlocker: Codable, Equatable, Sendable {
    public let code: String
    public let message: String

    public init(code: String, message: String) {
        self.code = code
        self.message = message
    }
}

public struct WorkflowGraphPlacementNode: Codable, Equatable, Sendable {
    public let agentID: String
    public let deviceID: String
    public let deviceName: String
    public let status: String
    public let eligible: Bool
    public let blockers: [WorkflowGraphPlacementBlocker]

    enum CodingKeys: String, CodingKey {
        case agentID = "agent_id"
        case deviceID = "device_id"
        case deviceName = "device_name"
        case status
        case eligible
        case blockers
    }

    public init(
        agentID: String,
        deviceID: String,
        deviceName: String,
        status: String,
        eligible: Bool,
        blockers: [WorkflowGraphPlacementBlocker]
    ) {
        self.agentID = agentID
        self.deviceID = deviceID
        self.deviceName = deviceName
        self.status = status
        self.eligible = eligible
        self.blockers = blockers
    }
}

public struct WorkflowGraphPlacement: Codable, Equatable, Sendable {
    public let connectedNodes: Int
    public let graphWorkerNodes: Int
    public let eligibleNodes: Int
    public let diagnostic: String?
    public let nodes: [WorkflowGraphPlacementNode]

    enum CodingKeys: String, CodingKey {
        case connectedNodes = "connected_nodes"
        case graphWorkerNodes = "graph_worker_nodes"
        case eligibleNodes = "eligible_nodes"
        case diagnostic
        case nodes
    }

    public init(
        connectedNodes: Int,
        graphWorkerNodes: Int,
        eligibleNodes: Int,
        diagnostic: String?,
        nodes: [WorkflowGraphPlacementNode]
    ) {
        self.connectedNodes = connectedNodes
        self.graphWorkerNodes = graphWorkerNodes
        self.eligibleNodes = eligibleNodes
        self.diagnostic = diagnostic
        self.nodes = nodes
    }
}

public struct RelayFleetGraphWorker: Codable, Equatable, Sendable {
    public let workerVersion: String
    public let contractVersions: [String]
    public let acceleratorBackend: String
    public let memoryBytes: UInt64
    public let availableDiskBytes: Int64?
    public let nodeKinds: [String]
    public let installedModelIDs: [String]

    enum CodingKeys: String, CodingKey {
        case workerVersion = "worker_version"
        case contractVersions = "contract_versions"
        case acceleratorBackend = "accelerator_backend"
        case memoryBytes = "memory_bytes"
        case availableDiskBytes = "available_disk_bytes"
        case nodeKinds = "node_kinds"
        case installedModelIDs = "installed_model_ids"
    }

    public init(
        workerVersion: String,
        contractVersions: [String],
        acceleratorBackend: String,
        memoryBytes: UInt64,
        availableDiskBytes: Int64?,
        nodeKinds: [String],
        installedModelIDs: [String]
    ) {
        self.workerVersion = workerVersion
        self.contractVersions = contractVersions
        self.acceleratorBackend = acceleratorBackend
        self.memoryBytes = memoryBytes
        self.availableDiskBytes = availableDiskBytes
        self.nodeKinds = nodeKinds
        self.installedModelIDs = installedModelIDs
    }
}

public struct RelayFleetCapabilities: Codable, Equatable, Sendable {
    public let models: [String]
    public let graphWorker: RelayFleetGraphWorker?

    enum CodingKeys: String, CodingKey {
        case models
        case graphWorker = "graph_worker"
    }

    public init(models: [String], graphWorker: RelayFleetGraphWorker?) {
        self.models = models
        self.graphWorker = graphWorker
    }
}

public struct RelayFleetRuntime: Codable, Equatable, Sendable {
    public let mereRunVersion: String?
    public let installedModels: [String]

    enum CodingKeys: String, CodingKey {
        case mereRunVersion = "mere_run_version"
        case installedModels = "installed_models"
    }

    public init(mereRunVersion: String?, installedModels: [String]) {
        self.mereRunVersion = mereRunVersion
        self.installedModels = installedModels
    }
}

public struct RelayFleetPolicy: Codable, Equatable, Sendable {
    public let enabled: Bool
    public let draining: Bool
    public let revoked: Bool
    public let priority: Int
    public let preferredModels: [String]
    public let displayName: String?

    enum CodingKeys: String, CodingKey {
        case enabled
        case draining
        case revoked
        case priority
        case preferredModels = "preferred_models"
        case displayName = "display_name"
    }

    public init(
        enabled: Bool,
        draining: Bool,
        revoked: Bool,
        priority: Int,
        preferredModels: [String],
        displayName: String?
    ) {
        self.enabled = enabled
        self.draining = draining
        self.revoked = revoked
        self.priority = priority
        self.preferredModels = preferredModels
        self.displayName = displayName
    }
}

public struct RelayFleetNode: Codable, Equatable, Sendable {
    public let agentID: String
    public let deviceID: String
    public let deviceName: String
    public let reportedName: String
    public let version: String
    public let status: String
    public let currentJobID: String?
    public let lastSeen: String
    public let capabilities: RelayFleetCapabilities
    public let runtime: RelayFleetRuntime?
    public let policy: RelayFleetPolicy

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

    public init(
        agentID: String,
        deviceID: String,
        deviceName: String,
        reportedName: String,
        version: String,
        status: String,
        currentJobID: String?,
        lastSeen: String,
        capabilities: RelayFleetCapabilities,
        runtime: RelayFleetRuntime?,
        policy: RelayFleetPolicy
    ) {
        self.agentID = agentID
        self.deviceID = deviceID
        self.deviceName = deviceName
        self.reportedName = reportedName
        self.version = version
        self.status = status
        self.currentJobID = currentJobID
        self.lastSeen = lastSeen
        self.capabilities = capabilities
        self.runtime = runtime
        self.policy = policy
    }
}

public struct RelayFleetActivity: Codable, Equatable, Sendable {
    public let id: String
    public let kind: String
    public let status: String
    public let agentID: String?
    public let model: String?
    public let label: String
    public let createdAt: String
    public let durationMilliseconds: Int?
    public let error: String?

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

    public init(
        id: String,
        kind: String,
        status: String,
        agentID: String?,
        model: String?,
        label: String,
        createdAt: String,
        durationMilliseconds: Int?,
        error: String?
    ) {
        self.id = id
        self.kind = kind
        self.status = status
        self.agentID = agentID
        self.model = model
        self.label = label
        self.createdAt = createdAt
        self.durationMilliseconds = durationMilliseconds
        self.error = error
    }
}

public struct RelayFleetSummary: Codable, Equatable, Sendable {
    public let totalNodes: Int
    public let onlineNodes: Int
    public let busyNodes: Int
    public let availableNodes: Int
    public let queueDepth: Int
    public let installedModels: Int
    public let routableModels: Int

    enum CodingKeys: String, CodingKey {
        case totalNodes = "total_nodes"
        case onlineNodes = "online_nodes"
        case busyNodes = "busy_nodes"
        case availableNodes = "available_nodes"
        case queueDepth = "queue_depth"
        case installedModels = "installed_models"
        case routableModels = "routable_models"
    }

    public init(
        totalNodes: Int,
        onlineNodes: Int,
        busyNodes: Int,
        availableNodes: Int,
        queueDepth: Int,
        installedModels: Int,
        routableModels: Int
    ) {
        self.totalNodes = totalNodes
        self.onlineNodes = onlineNodes
        self.busyNodes = busyNodes
        self.availableNodes = availableNodes
        self.queueDepth = queueDepth
        self.installedModels = installedModels
        self.routableModels = routableModels
    }
}

public struct RelayFleetSnapshot: Codable, Equatable, Sendable {
    public let generatedAt: String
    public let summary: RelayFleetSummary
    public let nodes: [RelayFleetNode]
    public let activity: [RelayFleetActivity]

    enum CodingKeys: String, CodingKey {
        case generatedAt = "generated_at"
        case summary
        case nodes
        case activity
    }

    public init(
        generatedAt: String,
        summary: RelayFleetSummary,
        nodes: [RelayFleetNode],
        activity: [RelayFleetActivity]
    ) {
        self.generatedAt = generatedAt
        self.summary = summary
        self.nodes = nodes
        self.activity = activity
    }
}

public struct RelayFleetRefreshResult: Codable, Equatable, Sendable {
    public let deviceID: String
    public let requested: Bool

    enum CodingKeys: String, CodingKey {
        case deviceID = "device_id"
        case requested
    }

    public init(deviceID: String, requested: Bool) {
        self.deviceID = deviceID
        self.requested = requested
    }
}

public struct RelayFleetNodePolicyPatch: Codable, Equatable, Sendable {
    public let enabled: Bool?
    public let draining: Bool?
    public let priority: Int?
    public let displayName: String?

    enum CodingKeys: String, CodingKey {
        case enabled
        case draining
        case priority
        case displayName = "display_name"
    }

    public init(enabled: Bool?, draining: Bool?, priority: Int?, displayName: String?) {
        self.enabled = enabled
        self.draining = draining
        self.priority = priority
        self.displayName = displayName
    }
}
