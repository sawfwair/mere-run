import Foundation

public struct WorkflowAssetEntry: Codable, Equatable, Sendable {
    public let path: String
    public let digest: String
    public let sizeBytes: Int64
    public let contentType: String

    enum CodingKeys: String, CodingKey {
        case path
        case digest
        case sizeBytes = "size_bytes"
        case contentType = "content_type"
    }

    public init(path: String, digest: String, sizeBytes: Int64, contentType: String) {
        self.path = path
        self.digest = digest
        self.sizeBytes = sizeBytes
        self.contentType = contentType
    }
}

public struct WorkflowAssetGroup: Codable, Equatable, Sendable {
    public let name: String
    public let kind: WorkflowPortType
    public let entries: [WorkflowAssetEntry]

    public init(name: String, kind: WorkflowPortType, entries: [WorkflowAssetEntry]) {
        self.name = name
        self.kind = kind
        self.entries = entries
    }
}

public struct WorkflowAssetManifest: Codable, Equatable, Sendable {
    public static let filename = "assets.json"

    public let schemaVersion: Int
    public let groups: [WorkflowAssetGroup]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case groups
    }

    public init(schemaVersion: Int, groups: [WorkflowAssetGroup]) {
        self.schemaVersion = schemaVersion
        self.groups = groups
    }
}

public struct WorkflowJobOutput: Codable, Equatable, Sendable {
    public let name: String
    public let reference: String

    public init(name: String, reference: String) {
        self.name = name
        self.reference = reference
    }
}

public struct WorkflowJobRequirements: Codable, Equatable, Sendable {
    public let minimumMereRunVersion: String
    public let nodeKinds: [String]
    public let modelIDs: [String]
    public let models: [WorkflowModelProvenance]
    public let providers: [WorkflowGraphProviderRequirement]
    public let secretNames: [String]
    public let acceleratorBackends: [String]
    public let minimumAcceleratorMemoryBytes: Int64?
    public let minimumSystemMemoryBytes: Int64?
    public let minimumDiskBytes: Int64?
    public let minimumCPUCores: Int?
    public let networkAccess: Bool

    enum CodingKeys: String, CodingKey {
        case minimumMereRunVersion = "minimum_mere_run_version"
        case nodeKinds = "node_kinds"
        case modelIDs = "model_ids"
        case models
        case providers
        case secretNames = "secret_names"
        case acceleratorBackends = "accelerator_backends"
        case minimumAcceleratorMemoryBytes = "minimum_accelerator_memory_bytes"
        case minimumSystemMemoryBytes = "minimum_system_memory_bytes"
        case minimumDiskBytes = "minimum_disk_bytes"
        case minimumCPUCores = "minimum_cpu_cores"
        case networkAccess = "network_access"
    }

    public init(
        minimumMereRunVersion: String,
        nodeKinds: [String],
        modelIDs: [String],
        models: [WorkflowModelProvenance] = [],
        providers: [WorkflowGraphProviderRequirement] = [],
        secretNames: [String] = [],
        acceleratorBackends: [String],
        minimumAcceleratorMemoryBytes: Int64?,
        minimumSystemMemoryBytes: Int64? = nil,
        minimumDiskBytes: Int64? = nil,
        minimumCPUCores: Int? = nil,
        networkAccess: Bool = false
    ) {
        self.minimumMereRunVersion = minimumMereRunVersion
        self.nodeKinds = nodeKinds
        self.modelIDs = modelIDs
        self.models = models
        self.providers = providers
        self.secretNames = secretNames
        self.acceleratorBackends = acceleratorBackends
        self.minimumAcceleratorMemoryBytes = minimumAcceleratorMemoryBytes
        self.minimumSystemMemoryBytes = minimumSystemMemoryBytes
        self.minimumDiskBytes = minimumDiskBytes
        self.minimumCPUCores = minimumCPUCores
        self.networkAccess = networkAccess
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        minimumMereRunVersion = try container.decode(String.self, forKey: .minimumMereRunVersion)
        nodeKinds = try container.decode([String].self, forKey: .nodeKinds)
        modelIDs = try container.decode([String].self, forKey: .modelIDs)
        models = try container.decodeIfPresent([WorkflowModelProvenance].self, forKey: .models) ?? []
        providers = try container.decodeIfPresent(
            [WorkflowGraphProviderRequirement].self,
            forKey: .providers
        ) ?? []
        secretNames = try container.decodeIfPresent([String].self, forKey: .secretNames) ?? []
        acceleratorBackends = try container.decode([String].self, forKey: .acceleratorBackends)
        minimumAcceleratorMemoryBytes = try container.decodeIfPresent(
            Int64.self,
            forKey: .minimumAcceleratorMemoryBytes
        )
        minimumSystemMemoryBytes = try container.decodeIfPresent(Int64.self, forKey: .minimumSystemMemoryBytes)
        minimumDiskBytes = try container.decodeIfPresent(Int64.self, forKey: .minimumDiskBytes)
        minimumCPUCores = try container.decodeIfPresent(Int.self, forKey: .minimumCPUCores)
        networkAccess = try container.decodeIfPresent(Bool.self, forKey: .networkAccess) ?? false
    }
}

public struct WorkflowModelProvenance: Codable, Equatable, Hashable, Sendable {
    public let id: String
    public let repository: String?
    public let revision: String?
    public let catalogSHA256: String
    public let installManifestSHA256: String?

    enum CodingKeys: String, CodingKey {
        case id
        case repository
        case revision
        case catalogSHA256 = "catalog_sha256"
        case installManifestSHA256 = "install_manifest_sha256"
    }

    public init(
        id: String,
        repository: String?,
        revision: String?,
        catalogSHA256: String,
        installManifestSHA256: String? = nil
    ) {
        self.id = id
        self.repository = repository
        self.revision = revision
        self.catalogSHA256 = catalogSHA256
        self.installManifestSHA256 = installManifestSHA256
    }
}

public struct WorkflowJobManifest: Codable, Equatable, Sendable {
    public static let contractVersion = "mere.run/job-bundle.v1"
    public static let filename = "job.json"

    public let contractVersion: String
    public let jobID: String
    public let createdAt: Date
    public let graphFingerprint: String
    public let inputFingerprint: String
    public let sourceGraphFingerprint: String?
    public let sourceInputFingerprint: String?
    public let requirements: WorkflowJobRequirements
    public let outputs: [WorkflowJobOutput]

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case jobID = "job_id"
        case createdAt = "created_at"
        case graphFingerprint = "graph_fingerprint"
        case inputFingerprint = "input_fingerprint"
        case sourceGraphFingerprint = "source_graph_fingerprint"
        case sourceInputFingerprint = "source_input_fingerprint"
        case requirements
        case outputs
    }

    public init(
        contractVersion: String,
        jobID: String,
        createdAt: Date,
        graphFingerprint: String,
        inputFingerprint: String,
        sourceGraphFingerprint: String? = nil,
        sourceInputFingerprint: String? = nil,
        requirements: WorkflowJobRequirements,
        outputs: [WorkflowJobOutput]
    ) {
        self.contractVersion = contractVersion
        self.jobID = jobID
        self.createdAt = createdAt
        self.graphFingerprint = graphFingerprint
        self.inputFingerprint = inputFingerprint
        self.sourceGraphFingerprint = sourceGraphFingerprint
        self.sourceInputFingerprint = sourceInputFingerprint
        self.requirements = requirements
        self.outputs = outputs
    }
}
