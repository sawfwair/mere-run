import ArgumentParser
import Foundation

enum WorkflowPortType: String, Codable, Equatable, Sendable {
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

typealias WorkflowInputType = WorkflowPortType
typealias WorkflowFieldType = WorkflowPortType

indirect enum WorkflowValue: Codable, Equatable, Sendable {
    case null
    case string(String)
    case integer(Int64)
    case number(Double)
    case boolean(Bool)
    case array([WorkflowValue])
    case object([String: WorkflowValue])
    case reference(String)

    init(from decoder: Decoder) throws {
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
            } else {
                self = .object(value)
            }
        }
    }

    func encode(to encoder: Encoder) throws {
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
        }
    }

    var references: [String] {
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

    var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    var integerValue: Int64? {
        guard case .integer(let value) = self else { return nil }
        return value
    }

    var numberValue: Double? {
        switch self {
        case .number(let value): value
        case .integer(let value): Double(value)
        default: nil
        }
    }

    var booleanValue: Bool? {
        guard case .boolean(let value) = self else { return nil }
        return value
    }
}

struct WorkflowInputDefinition: Codable, Equatable, Sendable {
    let type: WorkflowInputType
    let required: Bool?
    let defaultValue: WorkflowValue?
    let values: [String]?
    let contentTypes: [String]?

    enum CodingKeys: String, CodingKey {
        case type
        case required
        case defaultValue = "default"
        case values
        case contentTypes = "content_types"
    }
}

struct WorkflowNode: Codable, Equatable, Sendable {
    let id: String
    let kind: String
    let provider: String?
    let arguments: [String: WorkflowValue]
    let dependsOn: [String]?

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case provider
        case arguments
        case dependsOn = "depends_on"
    }

    init(
        id: String,
        kind: String,
        provider: String? = nil,
        arguments: [String: WorkflowValue],
        dependsOn: [String]?
    ) {
        self.id = id
        self.kind = kind
        self.provider = provider
        self.arguments = arguments
        self.dependsOn = dependsOn
    }

    var resolvedProviderID: String {
        provider ?? WorkflowNodeProviderIdentity.builtInID
    }
}

struct WorkflowGraphDocument: Codable, Equatable, Sendable {
    static let schemaVersion = 1
    static let kind = "mere.run/workflow-graph"

    let schemaVersion: Int
    let kind: String
    let name: String
    let inputs: [String: WorkflowInputDefinition]
    let nodes: [WorkflowNode]
    let outputs: [String: WorkflowValue]
    let metadata: [String: WorkflowValue]?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case kind
        case name
        case inputs
        case nodes
        case outputs
        case metadata
    }

    static func load(from url: URL) throws -> WorkflowGraphDocument {
        try JSONDecoder().decode(WorkflowGraphDocument.self, from: Data(contentsOf: url))
    }
}

struct WorkflowInputsDocument: Codable, Equatable, Sendable {
    let values: [String: WorkflowValue]

    init(values: [String: WorkflowValue]) {
        self.values = values
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        values = try container.decode([String: WorkflowValue].self)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(values)
    }

    static func load(from url: URL?) throws -> WorkflowInputsDocument {
        guard let url else { return WorkflowInputsDocument(values: [:]) }
        return try JSONDecoder().decode(WorkflowInputsDocument.self, from: Data(contentsOf: url))
    }
}

struct WorkflowNodeField: Codable, Equatable, Sendable {
    let name: String
    let type: WorkflowFieldType
    let required: Bool
    let description: String?
    let defaultValue: WorkflowValue?
    let values: [String]?
    let acceptedContentTypes: [String]?
    let minimum: Double?
    let maximum: Double?
    let step: Double?
    let multiline: Bool?
    let secret: Bool?
    let advanced: Bool?

    enum CodingKeys: String, CodingKey {
        case name
        case type
        case required
        case description
        case defaultValue = "default"
        case values
        case acceptedContentTypes = "content_types"
        case minimum
        case maximum
        case step
        case multiline
        case secret
        case advanced
    }

    init(
        name: String,
        type: WorkflowFieldType,
        required: Bool,
        description: String? = nil,
        defaultValue: WorkflowValue? = nil,
        values: [String]? = nil,
        acceptedContentTypes: [String]? = nil,
        minimum: Double? = nil,
        maximum: Double? = nil,
        step: Double? = nil,
        multiline: Bool? = nil,
        secret: Bool? = nil,
        advanced: Bool? = nil
    ) {
        self.name = name
        self.type = type
        self.required = required
        self.description = description
        self.defaultValue = defaultValue
        self.values = values
        self.acceptedContentTypes = acceptedContentTypes
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.multiline = multiline
        self.secret = secret
        self.advanced = advanced
    }
}

struct WorkflowNodeOutput: Codable, Equatable, Sendable {
    let name: String
    let type: WorkflowFieldType
    let optional: Bool
    let description: String?
    let contentTypes: [String]

    enum CodingKeys: String, CodingKey {
        case name
        case type
        case optional
        case description
        case contentTypes = "content_types"
    }

    init(
        name: String,
        type: WorkflowFieldType,
        optional: Bool = false,
        description: String? = nil,
        contentTypes: [String] = []
    ) {
        self.name = name
        self.type = type
        self.optional = optional
        self.description = description
        self.contentTypes = contentTypes
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        type = try container.decode(WorkflowFieldType.self, forKey: .type)
        optional = try container.decode(Bool.self, forKey: .optional)
        description = try container.decodeIfPresent(String.self, forKey: .description)
        contentTypes = try container.decodeIfPresent([String].self, forKey: .contentTypes) ?? []
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(type, forKey: .type)
        try container.encode(optional, forKey: .optional)
        try container.encodeIfPresent(description, forKey: .description)
        if !contentTypes.isEmpty {
            try container.encode(contentTypes, forKey: .contentTypes)
        }
    }
}

struct WorkflowNodeRequirements: Codable, Equatable, Sendable {
    let modelIDs: [String]
    let acceleratorBackends: [String]
    let minimumAcceleratorMemoryBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case modelIDs = "model_ids"
        case acceleratorBackends = "accelerator_backends"
        case minimumAcceleratorMemoryBytes = "minimum_accelerator_memory_bytes"
    }

    static let none = WorkflowNodeRequirements(
        modelIDs: [],
        acceleratorBackends: [],
        minimumAcceleratorMemoryBytes: nil
    )
}

struct WorkflowNodeTraits: Codable, Equatable, Sendable {
    let deterministic: Bool
    let cacheable: Bool
    let sideEffects: String
    let supportsProgress: Bool
    let supportsPreviews: Bool

    enum CodingKeys: String, CodingKey {
        case deterministic
        case cacheable
        case sideEffects = "side_effects"
        case supportsProgress = "supports_progress"
        case supportsPreviews = "supports_previews"
    }

    static let core = WorkflowNodeTraits(
        deterministic: false,
        cacheable: true,
        sideEffects: "local",
        supportsProgress: false,
        supportsPreviews: false
    )
}

struct WorkflowNodeProviderIdentity: Codable, Equatable, Hashable, Sendable {
    static let builtInID = "mere.run"

    let id: String
    let version: String
    let catalogSHA256: String

    enum CodingKeys: String, CodingKey {
        case id
        case version
        case catalogSHA256 = "catalog_sha256"
    }
}

struct WorkflowNodeCatalogEntry: Codable, Equatable, Sendable {
    let kind: String
    let title: String
    let description: String
    let category: String
    let inputs: [WorkflowNodeField]
    let outputs: [WorkflowNodeOutput]
    let requirements: WorkflowNodeRequirements
    let traits: WorkflowNodeTraits
    let provider: WorkflowNodeProviderIdentity?

    init(
        kind: String,
        title: String,
        description: String = "",
        category: String,
        inputs: [WorkflowNodeField],
        outputs: [WorkflowNodeOutput],
        requirements: WorkflowNodeRequirements = .none,
        traits: WorkflowNodeTraits = .core,
        provider: WorkflowNodeProviderIdentity? = nil
    ) {
        self.kind = kind
        self.title = title
        self.description = description
        self.category = category
        self.inputs = inputs
        self.outputs = outputs
        self.requirements = requirements
        self.traits = traits
        self.provider = provider
    }
}

enum WorkflowNodeRegistry {
    static let entries: [WorkflowNodeCatalogEntry] = [
        WorkflowNodeCatalogEntry(
            kind: "image.train-lora",
            title: "Train image LoRA",
            category: "image",
            inputs: [
                .init(name: "data", type: .assetDirectory, required: true),
                .init(name: "model", type: .string, required: false),
                .init(name: "recipe", type: .string, required: false),
                .init(name: "training_steps", type: .integer, required: false),
                .init(name: "width", type: .integer, required: false),
                .init(name: "height", type: .integer, required: false),
                .init(name: "max_text_length", type: .integer, required: false),
                .init(name: "seed", type: .integer, required: false),
                .init(name: "rank", type: .integer, required: false),
                .init(name: "lite", type: .boolean, required: false),
                .init(name: "base_quantization_bits", type: .integer, required: false),
                .init(name: "sample_prompt", type: .string, required: false),
            ],
            outputs: [.init(name: "adapter", type: .asset, contentTypes: ["application/x-safetensors"])],
            requirements: .init(
                modelIDs: [],
                acceleratorBackends: ["metal", "cuda"],
                minimumAcceleratorMemoryBytes: nil
            )
        ),
        WorkflowNodeCatalogEntry(
            kind: "image.generate",
            title: "Generate image",
            category: "image",
            inputs: [
                .init(name: "prompt", type: .string, required: true),
                .init(name: "negative_prompt", type: .string, required: false),
                .init(name: "model", type: .string, required: false),
                .init(name: "width", type: .integer, required: false),
                .init(name: "height", type: .integer, required: false),
                .init(name: "steps", type: .integer, required: false),
                .init(name: "seed", type: .integer, required: false),
                .init(name: "input", type: .asset, required: false, acceptedContentTypes: ["image/png", "image/jpeg", "image/webp"]),
                .init(name: "reference_images", type: .assetArray, required: false, acceptedContentTypes: ["image/png", "image/jpeg", "image/webp"]),
                .init(name: "lora", type: .asset, required: false, acceptedContentTypes: ["application/x-safetensors"]),
                .init(name: "lora_scale", type: .number, required: false),
                .init(name: "cfg_scale", type: .number, required: false),
                .init(name: "strength", type: .number, required: false),
                .init(name: "krea_base_quantization_bits", type: .integer, required: false),
            ],
            outputs: [.init(name: "image", type: .asset, contentTypes: ["image/png"])],
            requirements: .init(
                modelIDs: [],
                acceleratorBackends: ["metal", "cuda"],
                minimumAcceleratorMemoryBytes: nil
            )
        ),
        WorkflowNodeCatalogEntry(
            kind: "video.generate",
            title: "Generate video",
            category: "video",
            inputs: [
                .init(name: "prompt", type: .string, required: true),
                .init(name: "model", type: .string, required: false),
                .init(name: "width", type: .integer, required: false),
                .init(name: "height", type: .integer, required: false),
                .init(name: "num_frames", type: .integer, required: false),
                .init(name: "duration", type: .number, required: false),
                .init(name: "fps", type: .integer, required: false),
                .init(name: "seed", type: .integer, required: false),
                .init(name: "steps", type: .integer, required: false),
                .init(name: "image", type: .asset, required: false, acceptedContentTypes: ["image/png", "image/jpeg", "image/webp"]),
                .init(name: "end_image", type: .asset, required: false, acceptedContentTypes: ["image/png", "image/jpeg", "image/webp"]),
            ],
            outputs: [.init(name: "video", type: .asset, contentTypes: ["video/mp4"])],
            requirements: .init(
                modelIDs: [],
                acceleratorBackends: ["metal", "cuda"],
                minimumAcceleratorMemoryBytes: nil
            )
        ),
    ]

    static func entry(for kind: String) -> WorkflowNodeCatalogEntry? {
        entries.first { $0.kind == kind }
    }

    static var builtInProvider: WorkflowNodeProviderIdentity {
        WorkflowNodeProviderIdentity(
            id: WorkflowNodeProviderIdentity.builtInID,
            version: MereRunCLIVersion.current,
            catalogSHA256: (try? WorkflowBundleCodec.hash(entries)) ?? ""
        )
    }

    static var catalogEntries: [WorkflowNodeCatalogEntry] {
        catalogEntries(pluginNodes: WorkflowGraphProviderRegistry.discoveredCatalog().nodes)
    }

    static func catalogEntries(pluginNodes: [WorkflowNodeCatalogEntry]) -> [WorkflowNodeCatalogEntry] {
        let builtIn = entries.map { entry in
            WorkflowNodeCatalogEntry(
                kind: entry.kind,
                title: entry.title,
                description: entry.description,
                category: entry.category,
                inputs: entry.inputs,
                outputs: entry.outputs,
                requirements: entry.requirements,
                traits: entry.traits,
                provider: builtInProvider
            )
        }
        return (builtIn + pluginNodes).sorted {
            ($0.provider?.id ?? "", $0.kind) < ($1.provider?.id ?? "", $1.kind)
        }
    }

    static func entry(for node: WorkflowNode) -> WorkflowNodeCatalogEntry? {
        if node.resolvedProviderID == WorkflowNodeProviderIdentity.builtInID {
            return entry(for: node.kind)
        }
        return WorkflowGraphProviderRegistry.discoveredCatalog().nodes.first {
            $0.kind == node.kind && $0.provider?.id == node.resolvedProviderID
        }
    }

    static func output(node: WorkflowNode, name: String) -> WorkflowNodeOutput? {
        entry(for: node)?.outputs.first { $0.name == name }
    }

    static func provider(for node: WorkflowNode) -> WorkflowNodeProviderIdentity? {
        if node.resolvedProviderID == WorkflowNodeProviderIdentity.builtInID {
            return builtInProvider
        }
        return WorkflowGraphProviderRegistry.discoveredCatalog().provider(id: node.resolvedProviderID)?.identity
    }
}

struct WorkflowReference: Equatable, Sendable {
    enum Source: Equatable, Sendable {
        case input(String)
        case nodeOutput(nodeID: String, output: String)
    }

    let rawValue: String
    let source: Source

    init(_ rawValue: String) throws {
        self.rawValue = rawValue
        let parts = rawValue.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        if parts.count == 2, parts[0] == "inputs" {
            source = .input(parts[1])
            return
        }
        if parts.count == 4, parts[0] == "nodes", parts[2] == "outputs" {
            source = .nodeOutput(nodeID: parts[1], output: parts[3])
            return
        }
        throw ValidationError("Invalid workflow reference '\(rawValue)'.")
    }
}

struct WorkflowGraphValidation: Equatable {
    let diagnostics: [PreflightDiagnostic]
    let order: [String]
    let dependencies: [String: Set<String>]

    var status: StructuredRunStatus {
        StructuredRunOutput.status(for: diagnostics)
    }
}

enum WorkflowGraphValidator {
    static func validate(
        graph: WorkflowGraphDocument,
        inputs: WorkflowInputsDocument
    ) -> WorkflowGraphValidation {
        var diagnostics: [PreflightDiagnostic] = []
        validateHeader(graph, diagnostics: &diagnostics)
        validateIDs(graph, diagnostics: &diagnostics)
        validateInputs(graph, supplied: inputs, diagnostics: &diagnostics)

        let nodesByID = graph.nodes.reduce(into: [String: WorkflowNode]()) { nodes, node in
            nodes[node.id] = node
        }
        var dependencies: [String: Set<String>] = [:]
        for node in graph.nodes {
            validateNode(node, graph: graph, nodesByID: nodesByID, diagnostics: &diagnostics)
            var nodeDependencies = Set(node.dependsOn ?? [])
            for reference in node.arguments.values.flatMap(\.references) {
                guard let parsed = try? WorkflowReference(reference) else {
                    diagnostics.append(invalidReference(reference, nodeID: node.id))
                    continue
                }
                if case .nodeOutput(let dependency, _) = parsed.source {
                    nodeDependencies.insert(dependency)
                }
            }
            dependencies[node.id] = nodeDependencies
        }
        validateGraphOutputs(graph, nodesByID: nodesByID, diagnostics: &diagnostics)
        let order = topologicalOrder(graph.nodes.map(\.id), dependencies: dependencies)
        if order.count != graph.nodes.count {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "workflow_cycle",
                    severity: .blocker,
                    title: "Workflow contains a cycle",
                    message: "Node dependencies must form a directed acyclic graph."
                )
            )
        }
        return WorkflowGraphValidation(diagnostics: diagnostics, order: order, dependencies: dependencies)
    }

    private static func validateHeader(
        _ graph: WorkflowGraphDocument,
        diagnostics: inout [PreflightDiagnostic]
    ) {
        if graph.schemaVersion != WorkflowGraphDocument.schemaVersion {
            diagnostics.append(.init(
                id: "workflow_schema_unsupported",
                severity: .blocker,
                title: "Unsupported workflow schema",
                message: "Expected schema_version 1, found \(graph.schemaVersion)."
            ))
        }
        if graph.kind != WorkflowGraphDocument.kind {
            diagnostics.append(.init(
                id: "workflow_kind_unsupported",
                severity: .blocker,
                title: "Unsupported workflow kind",
                message: "Expected kind '\(WorkflowGraphDocument.kind)'."
            ))
        }
        if graph.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            diagnostics.append(.init(
                id: "workflow_name_empty",
                severity: .blocker,
                title: "Workflow name is empty",
                message: "Provide a non-empty workflow name."
            ))
        }
        if graph.nodes.isEmpty {
            diagnostics.append(.init(
                id: "workflow_nodes_empty",
                severity: .blocker,
                title: "Workflow has no nodes",
                message: "Add at least one registered workflow node."
            ))
        }
    }

    private static func validateIDs(
        _ graph: WorkflowGraphDocument,
        diagnostics: inout [PreflightDiagnostic]
    ) {
        let allIDs = Array(graph.inputs.keys) + graph.nodes.map(\.id) + Array(graph.outputs.keys)
        for id in allIDs where !isValidID(id) {
            diagnostics.append(.init(
                id: "workflow_id_invalid_\(id)",
                severity: .blocker,
                title: "Invalid workflow identifier",
                message: "'\(id)' must match [a-z][a-z0-9-]{0,63}."
            ))
        }
        let nodeIDs = graph.nodes.map(\.id)
        for id in Set(nodeIDs) where nodeIDs.filter({ $0 == id }).count > 1 {
            diagnostics.append(.init(
                id: "workflow_node_duplicate_\(id)",
                severity: .blocker,
                title: "Duplicate workflow node",
                message: "Node id '\(id)' appears more than once."
            ))
        }
    }

    private static func validateInputs(
        _ graph: WorkflowGraphDocument,
        supplied: WorkflowInputsDocument,
        diagnostics: inout [PreflightDiagnostic]
    ) {
        for name in supplied.values.keys where graph.inputs[name] == nil {
            diagnostics.append(.init(
                id: "workflow_input_unknown_\(name)",
                severity: .blocker,
                title: "Unknown workflow input",
                message: "Input '\(name)' is not declared by the workflow."
            ))
        }
        for (name, definition) in graph.inputs {
            guard let value = supplied.values[name] ?? definition.defaultValue else {
                if definition.required != false {
                    diagnostics.append(.init(
                        id: "workflow_input_missing_\(name)",
                        severity: .blocker,
                        title: "Required workflow input is missing",
                        message: "Provide a value for input '\(name)'."
                    ))
                }
                continue
            }
            if !matches(value, inputType: definition.type, enumValues: definition.values) {
                diagnostics.append(.init(
                    id: "workflow_input_type_\(name)",
                    severity: .blocker,
                    title: "Workflow input type mismatch",
                    message: "Input '\(name)' does not match declared type '\(definition.type.rawValue)'."
                ))
            }
        }
    }

    private static func validateNode(
        _ node: WorkflowNode,
        graph: WorkflowGraphDocument,
        nodesByID: [String: WorkflowNode],
        diagnostics: inout [PreflightDiagnostic]
    ) {
        guard let entry = WorkflowNodeRegistry.entry(for: node) else {
            diagnostics.append(.init(
                id: "workflow_node_kind_unsupported_\(node.id)",
                severity: .blocker,
                title: "Unsupported workflow node",
                message: "Node '\(node.id)' uses unregistered kind '\(node.kind)' from provider '\(node.resolvedProviderID)'."
            ))
            return
        }
        for name in node.arguments.keys where !entry.inputs.contains(where: { $0.name == name }) {
            diagnostics.append(.init(
                id: "workflow_node_argument_unknown_\(node.id)_\(name)",
                severity: .blocker,
                title: "Unknown node argument",
                message: "Node '\(node.id)' does not accept argument '\(name)'."
            ))
        }
        for field in entry.inputs {
            guard let value = node.arguments[field.name] else {
                if field.required {
                    diagnostics.append(.init(
                        id: "workflow_node_argument_missing_\(node.id)_\(field.name)",
                        severity: .blocker,
                        title: "Required node argument is missing",
                        message: "Node '\(node.id)' requires argument '\(field.name)'."
                    ))
                }
                continue
            }
            validateValue(
                value,
                expected: field.type,
                enumValues: field.values,
                acceptedContentTypes: field.acceptedContentTypes,
                nodeID: node.id,
                field: field.name,
                graph: graph,
                nodesByID: nodesByID,
                diagnostics: &diagnostics
            )
        }
        for dependency in node.dependsOn ?? [] where nodesByID[dependency] == nil {
            diagnostics.append(.init(
                id: "workflow_dependency_missing_\(node.id)_\(dependency)",
                severity: .blocker,
                title: "Workflow dependency is missing",
                message: "Node '\(node.id)' depends on unknown node '\(dependency)'."
            ))
        }
    }

    private static func validateValue(
        _ value: WorkflowValue,
        expected: WorkflowFieldType,
        enumValues: [String]?,
        acceptedContentTypes: [String]?,
        nodeID: String,
        field: String,
        graph: WorkflowGraphDocument,
        nodesByID: [String: WorkflowNode],
        diagnostics: inout [PreflightDiagnostic]
    ) {
        if case .reference(let rawReference) = value {
            guard let reference = try? WorkflowReference(rawReference) else {
                diagnostics.append(invalidReference(rawReference, nodeID: nodeID))
                return
            }
            switch reference.source {
            case .input(let input):
                guard let definition = graph.inputs[input] else {
                    diagnostics.append(invalidReference(rawReference, nodeID: nodeID))
                    return
                }
                if !compatible(inputType: definition.type, fieldType: expected) {
                    diagnostics.append(typeMismatch(nodeID: nodeID, field: field, reference: rawReference))
                } else if !contentTypesAreCompatible(
                    produced: definition.contentTypes,
                    accepted: acceptedContentTypes
                ) {
                    diagnostics.append(contentTypeMismatch(nodeID: nodeID, field: field, reference: rawReference))
                }
            case .nodeOutput(let sourceNodeID, let outputName):
                guard let sourceNode = nodesByID[sourceNodeID],
                      let output = WorkflowNodeRegistry.output(node: sourceNode, name: outputName) else {
                    diagnostics.append(invalidReference(rawReference, nodeID: nodeID))
                    return
                }
                if !compatible(outputType: output.type, fieldType: expected) {
                    diagnostics.append(typeMismatch(nodeID: nodeID, field: field, reference: rawReference))
                } else if !contentTypesAreCompatible(
                    produced: output.contentTypes,
                    accepted: acceptedContentTypes
                ) {
                    diagnostics.append(contentTypeMismatch(nodeID: nodeID, field: field, reference: rawReference))
                }
            }
            return
        }
        if expected == .assetArray || expected == .assetCollection, case .array(let values) = value {
            for nested in values {
                validateValue(
                    nested,
                    expected: .asset,
                    enumValues: nil,
                    acceptedContentTypes: acceptedContentTypes,
                    nodeID: nodeID,
                    field: field,
                    graph: graph,
                    nodesByID: nodesByID,
                    diagnostics: &diagnostics
                )
            }
            return
        }
        if expected == .json {
            switch value {
            case .array(let values):
                for nested in values {
                    validateValue(
                        nested,
                        expected: .json,
                        enumValues: nil,
                        acceptedContentTypes: nil,
                        nodeID: nodeID,
                        field: field,
                        graph: graph,
                        nodesByID: nodesByID,
                        diagnostics: &diagnostics
                    )
                }
            case .object(let values):
                for nested in values.values {
                    validateValue(
                        nested,
                        expected: .json,
                        enumValues: nil,
                        acceptedContentTypes: nil,
                        nodeID: nodeID,
                        field: field,
                        graph: graph,
                        nodesByID: nodesByID,
                        diagnostics: &diagnostics
                    )
                }
            default:
                break
            }
            return
        }
        if !matches(value, fieldType: expected, enumValues: enumValues) {
            diagnostics.append(.init(
                id: "workflow_node_argument_type_\(nodeID)_\(field)",
                severity: .blocker,
                title: "Node argument type mismatch",
                message: "Node '\(nodeID)' argument '\(field)' must be '\(expected.rawValue)'."
            ))
        }
    }

    private static func validateGraphOutputs(
        _ graph: WorkflowGraphDocument,
        nodesByID: [String: WorkflowNode],
        diagnostics: inout [PreflightDiagnostic]
    ) {
        for (name, value) in graph.outputs {
            guard case .reference(let rawReference) = value,
                  let reference = try? WorkflowReference(rawReference),
                  case .nodeOutput(let nodeID, let outputName) = reference.source,
                  let node = nodesByID[nodeID],
                  WorkflowNodeRegistry.output(node: node, name: outputName) != nil else {
                diagnostics.append(.init(
                    id: "workflow_output_invalid_\(name)",
                    severity: .blocker,
                    title: "Workflow output reference is invalid",
                    message: "Output '\(name)' must reference a registered node output."
                ))
                continue
            }
        }
    }

    private static func topologicalOrder(
        _ nodeIDs: [String],
        dependencies: [String: Set<String>]
    ) -> [String] {
        var remaining = nodeIDs.reduce(into: [String: Set<String>]()) { values, id in
            values[id] = dependencies[id] ?? []
        }
        var order: [String] = []
        while !remaining.isEmpty {
            guard let next = nodeIDs.first(where: { remaining[$0]?.isEmpty == true }) else { break }
            order.append(next)
            remaining.removeValue(forKey: next)
            for id in remaining.keys {
                remaining[id]?.remove(next)
            }
        }
        return order
    }

    private static func isValidID(_ id: String) -> Bool {
        id.range(of: "^[a-z][a-z0-9-]{0,63}$", options: .regularExpression) != nil
    }

    private static func matches(
        _ value: WorkflowValue,
        inputType: WorkflowInputType,
        enumValues: [String]?
    ) -> Bool {
        switch inputType {
        case .string, .asset, .assetDirectory:
            return value.stringValue != nil
        case .integer:
            return value.integerValue != nil
        case .number:
            return value.numberValue != nil
        case .boolean:
            return value.booleanValue != nil
        case .enumeration:
            guard let string = value.stringValue else { return false }
            return enumValues?.contains(string) ?? true
        case .json:
            return true
        case .assetCollection, .assetArray:
            if case .array = value { return true }
            return false
        }
    }

    private static func matches(
        _ value: WorkflowValue,
        fieldType: WorkflowFieldType,
        enumValues: [String]?
    ) -> Bool {
        return switch fieldType {
        case .string, .asset, .assetDirectory:
            value.stringValue != nil
        case .enumeration:
            if let string = value.stringValue {
                enumValues?.contains(string) ?? true
            } else {
                false
            }
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

    private static func compatible(inputType: WorkflowInputType, fieldType: WorkflowFieldType) -> Bool {
        switch (inputType, fieldType) {
        case (.string, .string), (.integer, .integer), (.number, .number), (.integer, .number),
             (.boolean, .boolean), (.enumeration, .string), (.enumeration, .enumeration),
             (.json, .json), (.asset, .asset), (.assetDirectory, .assetDirectory),
             (.assetCollection, .assetCollection), (.assetArray, .assetArray),
             (.assetCollection, .assetArray), (.assetArray, .assetCollection):
            true
        case (.string, .json), (.integer, .json), (.number, .json), (.boolean, .json),
             (.enumeration, .json):
            true
        default:
            false
        }
    }

    private static func compatible(outputType: WorkflowFieldType, fieldType: WorkflowFieldType) -> Bool {
        outputType == fieldType
            || (outputType == .integer && fieldType == .number)
            || (fieldType == .json && [.string, .integer, .number, .boolean, .enumeration, .json].contains(outputType))
    }

    private static func contentTypesAreCompatible(produced: [String]?, accepted: [String]?) -> Bool {
        guard let produced, let accepted else { return true }
        return !Set(produced).isDisjoint(with: accepted)
    }

    private static func invalidReference(_ reference: String, nodeID: String) -> PreflightDiagnostic {
        .init(
            id: "workflow_reference_invalid_\(nodeID)",
            severity: .blocker,
            title: "Workflow reference is invalid",
            message: "Node '\(nodeID)' references unknown value '\(reference)'."
        )
    }

    private static func typeMismatch(nodeID: String, field: String, reference: String) -> PreflightDiagnostic {
        .init(
            id: "workflow_reference_type_\(nodeID)_\(field)",
            severity: .blocker,
            title: "Workflow reference type mismatch",
            message: "Reference '\(reference)' is incompatible with node '\(nodeID)' argument '\(field)'."
        )
    }

    private static func contentTypeMismatch(nodeID: String, field: String, reference: String) -> PreflightDiagnostic {
        .init(
            id: "workflow_reference_content_type_\(nodeID)_\(field)",
            severity: .blocker,
            title: "Workflow artifact content type mismatch",
            message: "Artifact reference '\(reference)' is not accepted by node '\(nodeID)' argument '\(field)'."
        )
    }
}
