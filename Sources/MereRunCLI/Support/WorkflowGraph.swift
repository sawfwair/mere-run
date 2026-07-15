import ArgumentParser
import Foundation

enum WorkflowInputType: String, Codable, Equatable, Sendable {
    case string
    case integer
    case number
    case boolean
    case enumeration = "enum"
    case asset
    case assetDirectory = "asset_directory"
}

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
    let arguments: [String: WorkflowValue]
    let dependsOn: [String]?

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case arguments
        case dependsOn = "depends_on"
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

enum WorkflowFieldType: String, Codable, Equatable, Sendable {
    case string
    case integer
    case number
    case boolean
    case asset
    case assetDirectory = "asset_directory"
    case assetArray = "asset_array"
}

struct WorkflowNodeField: Codable, Equatable, Sendable {
    let name: String
    let type: WorkflowFieldType
    let required: Bool
    let acceptedContentTypes: [String]?

    enum CodingKeys: String, CodingKey {
        case name
        case type
        case required
        case acceptedContentTypes = "accepted_content_types"
    }

    init(
        name: String,
        type: WorkflowFieldType,
        required: Bool,
        acceptedContentTypes: [String]? = nil
    ) {
        self.name = name
        self.type = type
        self.required = required
        self.acceptedContentTypes = acceptedContentTypes
    }
}

struct WorkflowNodeOutput: Codable, Equatable, Sendable {
    let name: String
    let type: WorkflowFieldType
    let contentTypes: [String]

    enum CodingKeys: String, CodingKey {
        case name
        case type
        case contentTypes = "content_types"
    }
}

struct WorkflowNodeCatalogEntry: Codable, Equatable, Sendable {
    let kind: String
    let title: String
    let category: String
    let inputs: [WorkflowNodeField]
    let outputs: [WorkflowNodeOutput]
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
                .init(name: "seed", type: .integer, required: false),
                .init(name: "rank", type: .integer, required: false),
                .init(name: "sample_prompt", type: .string, required: false),
            ],
            outputs: [.init(name: "adapter", type: .asset, contentTypes: ["application/x-safetensors"])]
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
            ],
            outputs: [.init(name: "image", type: .asset, contentTypes: ["image/png"])]
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
            outputs: [.init(name: "video", type: .asset, contentTypes: ["video/mp4"])]
        ),
    ]

    static func entry(for kind: String) -> WorkflowNodeCatalogEntry? {
        entries.first { $0.kind == kind }
    }

    static func output(nodeKind: String, name: String) -> WorkflowNodeOutput? {
        entry(for: nodeKind)?.outputs.first { $0.name == name }
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
        guard let entry = WorkflowNodeRegistry.entry(for: node.kind) else {
            diagnostics.append(.init(
                id: "workflow_node_kind_unsupported_\(node.id)",
                severity: .blocker,
                title: "Unsupported workflow node",
                message: "Node '\(node.id)' uses unregistered kind '\(node.kind)'."
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
                      let output = WorkflowNodeRegistry.output(nodeKind: sourceNode.kind, name: outputName) else {
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
        if expected == .assetArray, case .array(let values) = value {
            for nested in values {
                validateValue(
                    nested,
                    expected: .asset,
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
        if !matches(value, fieldType: expected) {
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
                  WorkflowNodeRegistry.output(nodeKind: node.kind, name: outputName) != nil else {
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
        }
    }

    private static func matches(_ value: WorkflowValue, fieldType: WorkflowFieldType) -> Bool {
        switch fieldType {
        case .string, .asset, .assetDirectory:
            value.stringValue != nil
        case .integer:
            value.integerValue != nil
        case .number:
            value.numberValue != nil
        case .boolean:
            value.booleanValue != nil
        case .assetArray:
            if case .array = value { true } else { false }
        }
    }

    private static func compatible(inputType: WorkflowInputType, fieldType: WorkflowFieldType) -> Bool {
        switch (inputType, fieldType) {
        case (.string, .string), (.integer, .integer), (.number, .number), (.integer, .number),
             (.boolean, .boolean), (.enumeration, .string), (.asset, .asset),
             (.assetDirectory, .assetDirectory):
            true
        default:
            false
        }
    }

    private static func compatible(outputType: WorkflowFieldType, fieldType: WorkflowFieldType) -> Bool {
        outputType == fieldType || (outputType == .integer && fieldType == .number)
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
