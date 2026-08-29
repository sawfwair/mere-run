import Foundation

enum EvaluationSchemaValidation {
    static func validateManifestJSON(_ data: Data) throws {
        let root = try JSONDecoder().decode(EvaluationJSONNode.self, from: data)
        let manifest = try object(root, path: "manifest")
        try rejectUnknownFields(
            manifest,
            allowed: [
                "schema_version", "id", "version", "description", "case_files", "image_files",
                "prompt_sets", "arms", "sampling_profiles", "scorer", "gates", "defaults",
                "adapter_requirements",
            ],
            path: "manifest"
        )
        try validateObjects(
            manifest["prompt_sets"],
            path: "prompt_sets",
            allowed: ["id", "system_prompt_file"]
        )
        try validateObjects(
            manifest["arms"],
            path: "arms",
            allowed: [
                "id", "model_slot", "adapter_slot", "adapter_scale", "prompt_set", "profile_ids",
            ]
        )
        try validateObjects(
            manifest["sampling_profiles"],
            path: "sampling_profiles",
            allowed: [
                "id", "temperature", "top_p", "top_k", "min_p", "reasoning_effort", "show_thinking",
            ]
        )
        try validateObject(
            manifest["scorer"],
            path: "scorer",
            allowed: ["kind", "executable", "arguments", "timeout_seconds"]
        )
        try validateGateObjects(manifest["gates"])
        try validateObject(
            manifest["defaults"],
            path: "defaults",
            allowed: [
                "trials", "max_tokens", "context_size", "logprobs", "top_logprobs", "response_format",
            ]
        )
        try validateObject(
            manifest["adapter_requirements"],
            path: "adapter_requirements",
            allowed: [
                "require_training_manifest", "require_completed_training", "require_base_model_match",
            ]
        )
    }

    static func validateCaseJSON(_ data: Data) throws {
        let root = try JSONDecoder().decode(EvaluationJSONNode.self, from: data)
        let benchmarkCase = try object(root, path: "case")
        try rejectUnknownFields(
            benchmarkCase,
            allowed: [
                "id", "split", "capability_tags", "messages", "assertions", "max_tokens", "metadata",
            ],
            path: "case"
        )
        try validateObjects(
            benchmarkCase["messages"],
            path: "messages",
            allowed: ["role", "content", "image_file"]
        )
        try validateObjects(
            benchmarkCase["assertions"],
            path: "assertions",
            allowed: ["id", "kind", "value", "case_insensitive"]
        )
    }

    private static func validateGateObjects(_ node: EvaluationJSONNode?) throws {
        guard let node else { return }
        let gates = try array(node, path: "gates")
        for (index, gateNode) in gates.enumerated() {
            let path = "gates[\(index)]"
            let gate = try object(gateNode, path: path)
            try rejectUnknownFields(
                gate,
                allowed: [
                    "id", "aggregation", "metric_id", "comparator", "threshold", "required", "filter",
                ],
                path: path
            )
            try validateObject(
                gate["filter"],
                path: "\(path).filter",
                allowed: ["splits", "arms", "profiles", "capabilities"]
            )
        }
    }

    private static func validateObjects(
        _ node: EvaluationJSONNode?,
        path: String,
        allowed: Set<String>
    ) throws {
        guard let node else { return }
        let values = try array(node, path: path)
        for (index, value) in values.enumerated() {
            let itemPath = "\(path)[\(index)]"
            let item = try object(value, path: itemPath)
            try rejectUnknownFields(item, allowed: allowed, path: itemPath)
        }
    }

    private static func validateObject(
        _ node: EvaluationJSONNode?,
        path: String,
        allowed: Set<String>
    ) throws {
        guard let node else { return }
        try rejectUnknownFields(try object(node, path: path), allowed: allowed, path: path)
    }

    private static func rejectUnknownFields(
        _ object: [String: EvaluationJSONNode],
        allowed: Set<String>,
        path: String
    ) throws {
        let unknown = Set(object.keys).subtracting(allowed).sorted()
        guard unknown.isEmpty else {
            throw EvaluationSchemaError.unknownFields(path: path, fields: unknown)
        }
    }

    private static func object(
        _ node: EvaluationJSONNode,
        path: String
    ) throws -> [String: EvaluationJSONNode] {
        guard case .object(let value) = node else {
            throw EvaluationSchemaError.expected(path: path, type: "object")
        }
        return value
    }

    private static func array(
        _ node: EvaluationJSONNode,
        path: String
    ) throws -> [EvaluationJSONNode] {
        guard case .array(let value) = node else {
            throw EvaluationSchemaError.expected(path: path, type: "array")
        }
        return value
    }
}

private enum EvaluationSchemaError: LocalizedError {
    case expected(path: String, type: String)
    case unknownFields(path: String, fields: [String])

    var errorDescription: String? {
        switch self {
        case .expected(let path, let type):
            return "\(path) must be a JSON \(type)"
        case .unknownFields(let path, let fields):
            return "\(path) contains unsupported fields: \(fields.joined(separator: ", "))"
        }
    }
}

private enum EvaluationJSONNode: Decodable {
    case object([String: EvaluationJSONNode])
    case array([EvaluationJSONNode])
    case string(String)
    case number(Double)
    case boolean(Bool)
    case null

    init(from decoder: Decoder) throws {
        if let container = try? decoder.container(keyedBy: EvaluationDynamicCodingKey.self) {
            var values: [String: EvaluationJSONNode] = [:]
            for key in container.allKeys {
                values[key.stringValue] = try container.decode(EvaluationJSONNode.self, forKey: key)
            }
            self = .object(values)
            return
        }
        if var container = try? decoder.unkeyedContainer() {
            var values: [EvaluationJSONNode] = []
            while !container.isAtEnd {
                values.append(try container.decode(EvaluationJSONNode.self))
            }
            self = .array(values)
            return
        }
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .boolean(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else {
            self = .string(try container.decode(String.self))
        }
    }
}

private struct EvaluationDynamicCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init?(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }
}
