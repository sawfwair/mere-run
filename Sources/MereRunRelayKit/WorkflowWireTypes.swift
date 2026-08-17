import Foundation

/// The typed value form shared by workflow graphs, inputs, node outputs, and
/// run manifests. Serialization is position-free JSON: scalars encode
/// directly, `$ref` wraps node references, and `$secret` wraps secret names.
public indirect enum WorkflowValue: Codable, Equatable, Sendable {
    case null
    case string(String)
    case integer(Int64)
    case number(Double)
    case boolean(Bool)
    case array([WorkflowValue])
    case object([String: WorkflowValue])
    case reference(String)
    case secretReference(String)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .boolean(value)
        } else if let value = try? container.decode(Int64.self) {
            self = .integer(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([WorkflowValue].self) {
            self = .array(value)
        } else {
            let value = try container.decode([String: WorkflowValue].self)
            if value.count == 1, case .string(let reference)? = value["$ref"] {
                self = .reference(reference)
            } else if value.count == 1, case .string(let name)? = value["$secret"] {
                self = .secretReference(name)
            } else {
                self = .object(value)
            }
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .string(let value):
            try container.encode(value)
        case .integer(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .boolean(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .reference(let value):
            try container.encode(["$ref": WorkflowValue.string(value)])
        case .secretReference(let value):
            try container.encode(["$secret": WorkflowValue.string(value)])
        }
    }

    public var references: [String] {
        switch self {
        case .reference(let value):
            [value]
        case .array(let values):
            values.flatMap(\.references)
        case .object(let values):
            values.keys.sorted().flatMap { values[$0]?.references ?? [] }
        default:
            []
        }
    }

    public var secretNames: [String] {
        switch self {
        case .secretReference(let value):
            [value]
        case .array(let values):
            values.flatMap(\.secretNames)
        case .object(let values):
            values.keys.sorted().flatMap { values[$0]?.secretNames ?? [] }
        default:
            []
        }
    }

    public var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    public var integerValue: Int64? {
        guard case .integer(let value) = self else { return nil }
        return value
    }

    public var numberValue: Double? {
        switch self {
        case .number(let value): value
        case .integer(let value): Double(value)
        default: nil
        }
    }

    public var booleanValue: Bool? {
        guard case .boolean(let value) = self else { return nil }
        return value
    }
}

/// The portable port/input type vocabulary shared by graph documents, node
/// catalogs, and asset manifests.
public enum WorkflowPortType: String, Codable, Equatable, Sendable {
    case string
    case integer
    case number
    case boolean
    case enumeration = "enum"
    case json
    case asset
    case assetDirectory = "asset_directory"
    case assetCollection = "asset_collection"
    // Decodes catalogs produced before asset_collection became the portable name.
    case assetArray = "asset_array"
}

/// Identity of the provider that contributed a graph node kind.
public struct WorkflowNodeProviderIdentity: Codable, Equatable, Hashable, Sendable {
    public static let builtInID = "mere.run"

    public let id: String
    public let version: String
    public let catalogSHA256: String

    enum CodingKeys: String, CodingKey {
        case id
        case version
        case catalogSHA256 = "catalog_sha256"
    }

    public init(id: String, version: String, catalogSHA256: String) {
        self.id = id
        self.version = version
        self.catalogSHA256 = catalogSHA256
    }
}

/// Exact provider requirement recorded in job manifests and reported by
/// worker probes.
public struct WorkflowGraphProviderRequirement: Codable, Equatable, Hashable, Sendable {
    public let id: String
    public let version: String
    public let catalogSHA256: String
    public let nodeKinds: [String]

    enum CodingKeys: String, CodingKey {
        case id
        case version
        case catalogSHA256 = "catalog_sha256"
        case nodeKinds = "node_kinds"
    }

    public init(id: String, version: String, catalogSHA256: String, nodeKinds: [String]) {
        self.id = id
        self.version = version
        self.catalogSHA256 = catalogSHA256
        self.nodeKinds = nodeKinds
    }
}
