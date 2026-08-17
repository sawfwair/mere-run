import Foundation

public enum GraphRunState: String, Codable, Equatable, Sendable {
    case planned
    case preflighting
    case queued
    case assigned
    case running
    case finished
    case failed
    case cancelled
}

public struct GraphRunArtifact: Codable, Equatable, Sendable {
    public let name: String
    public let kind: String
    public let path: String
    public let contentType: String
    public let sizeBytes: Int64
    public let sha256: String

    enum CodingKeys: String, CodingKey {
        case name
        case kind
        case path
        case contentType = "content_type"
        case sizeBytes = "size_bytes"
        case sha256
    }

    public init(name: String, kind: String, path: String, contentType: String, sizeBytes: Int64, sha256: String) {
        self.name = name
        self.kind = kind
        self.path = path
        self.contentType = contentType
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
    }
}

public struct GraphRunNodeOutput: Codable, Equatable, Sendable {
    public let name: String
    public let type: WorkflowPortType
    public let value: WorkflowValue?
    public let path: String?
    public let contentType: String?
    public let sizeBytes: Int64?
    public let sha256: String?

    enum CodingKeys: String, CodingKey {
        case name
        case type
        case value
        case path
        case contentType = "content_type"
        case sizeBytes = "size_bytes"
        case sha256
    }

    public init(
        name: String,
        type: WorkflowPortType,
        value: WorkflowValue?,
        path: String?,
        contentType: String?,
        sizeBytes: Int64?,
        sha256: String?
    ) {
        self.name = name
        self.type = type
        self.value = value
        self.path = path
        self.contentType = contentType
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
    }
}

public struct GraphRunNodeRecord: Codable, Equatable, Sendable {
    public let id: String
    public let kind: String
    public var state: GraphRunState
    public var startedAt: Date?
    public var completedAt: Date?
    public var exitStatus: Int32?
    public var attempt: Int
    public var maxAttempts: Int
    public var fingerprint: String
    public var provider: WorkflowNodeProviderIdentity?
    public var models: [WorkflowModelProvenance]
    public var artifacts: [GraphRunArtifact]
    public var outputs: [GraphRunNodeOutput]
    public var error: String?

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case state
        case startedAt = "started_at"
        case completedAt = "completed_at"
        case exitStatus = "exit_status"
        case attempt
        case maxAttempts = "max_attempts"
        case fingerprint
        case provider
        case models
        case artifacts
        case outputs
        case error
    }

    public init(
        id: String,
        kind: String,
        state: GraphRunState,
        startedAt: Date?,
        completedAt: Date?,
        exitStatus: Int32?,
        attempt: Int = 0,
        maxAttempts: Int = 1,
        fingerprint: String,
        provider: WorkflowNodeProviderIdentity? = nil,
        models: [WorkflowModelProvenance] = [],
        artifacts: [GraphRunArtifact],
        outputs: [GraphRunNodeOutput] = [],
        error: String?
    ) {
        self.id = id
        self.kind = kind
        self.state = state
        self.startedAt = startedAt
        self.completedAt = completedAt
        self.exitStatus = exitStatus
        self.attempt = attempt
        self.maxAttempts = maxAttempts
        self.fingerprint = fingerprint
        self.provider = provider
        self.models = models
        self.artifacts = artifacts
        self.outputs = outputs
        self.error = error
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        kind = try container.decode(String.self, forKey: .kind)
        state = try container.decode(GraphRunState.self, forKey: .state)
        startedAt = try container.decodeIfPresent(Date.self, forKey: .startedAt)
        completedAt = try container.decodeIfPresent(Date.self, forKey: .completedAt)
        exitStatus = try container.decodeIfPresent(Int32.self, forKey: .exitStatus)
        attempt = try container.decodeIfPresent(Int.self, forKey: .attempt) ?? 0
        maxAttempts = try container.decodeIfPresent(Int.self, forKey: .maxAttempts) ?? 1
        fingerprint = try container.decode(String.self, forKey: .fingerprint)
        provider = try container.decodeIfPresent(WorkflowNodeProviderIdentity.self, forKey: .provider)
        models = try container.decodeIfPresent([WorkflowModelProvenance].self, forKey: .models) ?? []
        artifacts = try container.decode([GraphRunArtifact].self, forKey: .artifacts)
        outputs = try container.decodeIfPresent([GraphRunNodeOutput].self, forKey: .outputs) ?? []
        error = try container.decodeIfPresent(String.self, forKey: .error)
    }
}

public struct GraphRunExecutorRecord: Codable, Equatable, Sendable {
    public let kind: String
    public let profile: String?
    public let jobReference: String?

    enum CodingKeys: String, CodingKey {
        case kind
        case profile
        case jobReference = "job_reference"
    }

    public init(kind: String, profile: String?, jobReference: String?) {
        self.kind = kind
        self.profile = profile
        self.jobReference = jobReference
    }
}

public struct GraphRunManifest: Codable, Equatable, Sendable {
    public static let contractVersion = "mere.run/graph-run.v1"
    public static let filename = "run.json"

    public let contractVersion: String
    public let jobID: String
    public let graphName: String
    public let graphFingerprint: String
    public let sourceGraphFingerprint: String?
    public let sourceInputFingerprint: String?
    public var state: GraphRunState
    public let createdAt: Date
    public var updatedAt: Date
    public var attempt: Int
    public var executor: GraphRunExecutorRecord
    public var nodes: [GraphRunNodeRecord]
    public var outputs: [GraphRunArtifact]
    public var error: String?

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case jobID = "job_id"
        case graphName = "graph_name"
        case graphFingerprint = "graph_fingerprint"
        case sourceGraphFingerprint = "source_graph_fingerprint"
        case sourceInputFingerprint = "source_input_fingerprint"
        case state
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case attempt
        case executor
        case nodes
        case outputs
        case error
    }

    public init(
        contractVersion: String,
        jobID: String,
        graphName: String,
        graphFingerprint: String,
        sourceGraphFingerprint: String? = nil,
        sourceInputFingerprint: String? = nil,
        state: GraphRunState,
        createdAt: Date,
        updatedAt: Date,
        attempt: Int,
        executor: GraphRunExecutorRecord,
        nodes: [GraphRunNodeRecord],
        outputs: [GraphRunArtifact],
        error: String?
    ) {
        self.contractVersion = contractVersion
        self.jobID = jobID
        self.graphName = graphName
        self.graphFingerprint = graphFingerprint
        self.sourceGraphFingerprint = sourceGraphFingerprint
        self.sourceInputFingerprint = sourceInputFingerprint
        self.state = state
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.attempt = attempt
        self.executor = executor
        self.nodes = nodes
        self.outputs = outputs
        self.error = error
    }
}

public struct GraphRunEvent: Codable, Equatable, Sendable {
    public let sequence: Int
    public let createdAt: Date
    public let type: String
    public let state: GraphRunState
    public let nodeID: String?
    public let message: String?
    public let progress: GraphRunProgress?
    public let artifact: GraphRunEventArtifact?
    public let diagnostic: GraphRunEventDiagnostic?
    public let metric: GraphRunMetric?

    enum CodingKeys: String, CodingKey {
        case sequence
        case createdAt = "created_at"
        case type
        case state
        case nodeID = "node_id"
        case message
        case progress
        case artifact
        case diagnostic
        case metric
    }

    public init(
        sequence: Int,
        createdAt: Date,
        type: String,
        state: GraphRunState,
        nodeID: String?,
        message: String?,
        progress: GraphRunProgress? = nil,
        artifact: GraphRunEventArtifact? = nil,
        diagnostic: GraphRunEventDiagnostic? = nil,
        metric: GraphRunMetric? = nil
    ) {
        self.sequence = sequence
        self.createdAt = createdAt
        self.type = type
        self.state = state
        self.nodeID = nodeID
        self.message = message
        self.progress = progress
        self.artifact = artifact
        self.diagnostic = diagnostic
        self.metric = metric
    }
}

public struct GraphRunProgress: Codable, Equatable, Sendable {
    public let phase: String?
    public let current: Double?
    public let total: Double?
    public let fraction: Double?
    public let unit: String?

    public init(phase: String?, current: Double?, total: Double?, fraction: Double?, unit: String?) {
        self.phase = phase
        self.current = current
        self.total = total
        self.fraction = fraction
        self.unit = unit
    }
}

public struct GraphRunEventArtifact: Codable, Equatable, Sendable {
    public let name: String
    public let path: String
    public let contentType: String

    enum CodingKeys: String, CodingKey {
        case name
        case path
        case contentType = "content_type"
    }

    public init(name: String, path: String, contentType: String) {
        self.name = name
        self.path = path
        self.contentType = contentType
    }
}

public struct GraphRunEventDiagnostic: Codable, Equatable, Sendable {
    public let id: String
    public let severity: String
    public let title: String
    public let message: String

    public init(id: String, severity: String, title: String, message: String) {
        self.id = id
        self.severity = severity
        self.title = title
        self.message = message
    }
}

public struct GraphRunMetric: Codable, Equatable, Sendable {
    public let name: String
    public let value: Double
    public let unit: String?

    public init(name: String, value: Double, unit: String?) {
        self.name = name
        self.value = value
        self.unit = unit
    }
}
