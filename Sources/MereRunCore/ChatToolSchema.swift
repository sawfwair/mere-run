import Foundation

extension ChatToolValidation {
    public static let schemaKeywords: Set<String> = [
        "$schema", "$id", "$ref", "$defs", "definitions", "title", "description", "default", "examples",
        "type", "properties", "required", "additionalProperties", "items", "enum", "const",
        "anyOf", "oneOf", "allOf", "not", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "minLength", "maxLength", "pattern", "format", "minItems", "maxItems", "uniqueItems"
    ]

    /// Reject unsupported schema semantics before sampling, rather than
    /// accepting calls whose declared constraints were never checked.
    public static func validateSchema(_ schema: [String: ChatJSONValue], root: [String: ChatJSONValue]? = nil, depth: Int = 0) throws {
        guard depth < 64 else { throw schemaError("schema nesting exceeds 64 levels") }
        let root = root ?? schema
        if let keyword = Set(schema.keys).subtracting(schemaKeywords).sorted().first { throw schemaError("keyword '\(keyword)' is unsupported; use the published tool_schema_subset") }
        if let type = schema["type"] {
            let types = type.arrayValue ?? [type]
            guard !types.isEmpty, types.allSatisfy({ ["object", "array", "string", "number", "integer", "boolean", "null"].contains($0.stringValue ?? "") }) else { throw schemaError("type must name supported JSON types") }
        }
        if let reference = schema["$ref"] {
            guard let reference = reference.stringValue, reference.hasPrefix("#/"), reference.count < 1024 else { throw schemaError("only local references are supported") }
            var target: ChatJSONValue = .object(root)
            for part in reference.dropFirst(2).split(separator: "/") {
                let key = part.replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")
                guard let next = target.objectValue?[key] else { throw schemaError("reference does not exist") }
                target = next
            }
            guard target.objectValue != nil else { throw schemaError("reference must name an object schema") }
        }
        for keyword in ["properties", "$defs", "definitions"] {
            if let value = schema[keyword] {
                guard let properties = value.objectValue else { throw schemaError("\(keyword) must be an object") }
                for value in properties.values { try child(value, root: root, depth: depth) }
            }
        }
        for keyword in ["items", "not"] { if let value = schema[keyword] { try child(value, root: root, depth: depth) } }
        if let value = schema["additionalProperties"], value.boolValue == nil { try child(value, root: root, depth: depth) }
        for keyword in ["anyOf", "oneOf", "allOf"] {
            if let value = schema[keyword] {
                guard let options = value.arrayValue, !options.isEmpty else { throw schemaError("\(keyword) must contain schemas") }
                for value in options { try child(value, root: root, depth: depth) }
            }
        }
        if let required = schema["required"], required.arrayValue?.allSatisfy({ $0.stringValue != nil }) != true { throw schemaError("required must be an array of names") }
        for keyword in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "minItems", "maxItems"] {
            if let value = schema[keyword] {
                guard let number = value.doubleValue, number.isFinite else { throw schemaError("\(keyword) must be numeric") }
                if keyword.hasSuffix("Length") || keyword.hasSuffix("Items"), number < 0 || number.rounded() != number { throw schemaError("\(keyword) must be a non-negative integer") }
            }
        }
        if let pattern = schema["pattern"] {
            guard let pattern = pattern.stringValue, pattern.count <= 1024, (try? NSRegularExpression(pattern: pattern)) != nil else { throw schemaError("pattern must be a valid regex of at most 1024 characters") }
        }
        if let format = schema["format"], format != .string("uri") { throw schemaError("only the uri string format is supported") }
        if let unique = schema["uniqueItems"], unique.boolValue == nil { throw schemaError("uniqueItems must be boolean") }
        if let values = schema["enum"], values.arrayValue?.isEmpty != false { throw schemaError("enum must be a non-empty array") }
    }

    private static func child(_ value: ChatJSONValue, root: [String: ChatJSONValue], depth: Int) throws {
        guard let object = value.objectValue else { throw schemaError("nested schemas must be objects") }
        try validateSchema(object, root: root, depth: depth + 1)
    }
    private static func schemaError(_ message: String) -> ChatRequestIssue { ChatRequestIssue("tools.input_schema", message) }
}

extension ChatJSONValue {
    var doubleValue: Double? { if case .number(let value) = self { return value }; return nil }
}
