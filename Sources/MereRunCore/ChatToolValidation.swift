import Foundation

public enum ChatToolValidation {
    static func typedArguments(_ arguments: [String: String], tool: ToolDefinition) throws -> [String: ChatJSONValue] {
        let schema = tool.parametersJSONSchema
        var typed: [String: ChatJSONValue] = [:]
        for (key, raw) in arguments {
            let property = schema["properties"]?.objectValue?[key]?.objectValue ?? [:]
            if try resolvedType(property, root: schema) == "string" { typed[key] = .string(raw) }
            else {
                do { typed[key] = try JSONDecoder().decode(ChatJSONValue.self, from: Data(raw.utf8)) }
                catch { throw ChatRequestIssue("tool_use.input", "parameter '\(key)' is not valid JSON for its schema") }
            }
        }
        try validate(.object(typed), schema: schema)
        return typed
    }
    /// A declared subset, shared by adapter output validation. Strict decoding
    /// requires constrained generation and is a separate capability.
    public static func validate(_ value: ChatJSONValue, schema: [String: ChatJSONValue],
                                root: [String: ChatJSONValue]? = nil, depth: Int = 0) throws {
        guard depth < 64 else { throw ChatRequestIssue("tool_use.validation_depth", "schema validation exceeds 64 levels") }
        let root = root ?? schema
        if let reference = schema["$ref"]?.stringValue {
            guard reference.hasPrefix("#/"), reference.count < 1024 else { throw ChatRequestIssue("tools", "only local schema references are supported") }
            var target: ChatJSONValue = .object(root)
            for part in reference.dropFirst(2).split(separator: "/") {
                let key = part.replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")
                guard let next = target.objectValue?[key] else { throw ChatRequestIssue("tools", "schema reference does not exist") }
                target = next
            }
            guard let referenced = target.objectValue else { throw ChatRequestIssue("tools", "schema reference must name an object") }
            try validate(value, schema: referenced, root: root, depth: depth + 1)
        }
        if let values = schema["enum"]?.arrayValue, !values.contains(value) { throw ChatRequestIssue("tool_use.input", "value is outside the schema enum") }
        if let constant = schema["const"], value != constant { throw ChatRequestIssue("tool_use.input", "value does not match const") }
        for keyword in ["anyOf", "oneOf", "allOf"] {
            if let alternatives = schema[keyword]?.arrayValue {
                let matches = try alternatives.filter { alternative in
                    guard let object = alternative.objectValue else { throw ChatRequestIssue("tools", "schema alternatives must be objects") }
                    do { try validate(value, schema: object, root: root, depth: depth + 1); return true }
                    catch let issue as ChatRequestIssue where issue.field == "tool_use.input" { return false }
                }.count
                guard keyword == "allOf" ? matches == alternatives.count : keyword == "oneOf" ? matches == 1 : matches > 0 else {
                    throw ChatRequestIssue("tool_use.input", "value does not satisfy \(keyword)")
                }
            }
        }
        if let excluded = schema["not"]?.objectValue {
            var matches = false
            do { try validate(value, schema: excluded, root: root, depth: depth + 1); matches = true }
            catch let issue as ChatRequestIssue where issue.field == "tool_use.input" { }
            if matches { throw ChatRequestIssue("tool_use.input", "value matches excluded schema") }
        }
        if let types = schema["type"]?.arrayValue {
            var alternatives = schema; alternatives.removeValue(forKey: "type")
            for type in types {
                alternatives["type"] = type
                do { try validate(value, schema: alternatives, root: root, depth: depth + 1); return }
                catch let issue as ChatRequestIssue where issue.field == "tool_use.input" { }
            }
            throw ChatRequestIssue("tool_use.input", "value does not match an allowed schema type")
        }
        let type = schema["type"]?.stringValue
        switch value {
        case .object(let object):
            guard type == nil || type == "object" else { throw mismatch(type) }
            let properties = schema["properties"]?.objectValue ?? [:]
            let required = schema["required"]?.arrayValue?.compactMap(\.stringValue) ?? []
            guard required.allSatisfy({ object[$0] != nil }) else { throw ChatRequestIssue("tool_use.input", "missing a required property") }
            for (key, child) in object {
                if let childSchema = properties[key]?.objectValue { try validate(child, schema: childSchema, root: root, depth: depth + 1) }
                else if schema["additionalProperties"] == .bool(false) { throw ChatRequestIssue("tool_use.input", "unknown property '\(key)'") }
                else if let additional = schema["additionalProperties"]?.objectValue { try validate(child, schema: additional, root: root, depth: depth + 1) }
            }
        case .array(let values):
            guard type == nil || type == "array" else { throw mismatch(type) }
            if let items = schema["items"]?.objectValue { for child in values { try validate(child, schema: items, root: root, depth: depth + 1) } }
            try bounds(Double(values.count), minimum: schema["minItems"], maximum: schema["maxItems"])
            if schema["uniqueItems"] == .bool(true), Set(values).count != values.count { throw ChatRequestIssue("tool_use.input", "array items must be unique") }
        case .string(let string):
            guard type == nil || type == "string" else { throw mismatch(type) }
            try bounds(Double(string.unicodeScalars.count), minimum: schema["minLength"], maximum: schema["maxLength"])
            if let pattern = schema["pattern"]?.stringValue, try !patternMatches(pattern, string: string) { throw ChatRequestIssue("tool_use.input", "string does not match the schema pattern") }
            if schema["format"] == .string("uri") {
                let uri = #"^[A-Za-z][A-Za-z0-9+.-]*:(?:[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=\-]|%[A-Fa-f0-9]{2})*$"#
                guard try patternMatches(uri, string: string), URLComponents(string: string)?.scheme != nil else {
                    throw ChatRequestIssue("tool_use.input", "string must be an absolute URI with valid percent escapes")
                }
            }
        case .number(let number):
            guard number.isFinite else { throw ChatRequestIssue("tool_use.input", "numbers must be finite JSON values") }
            guard type == nil || type == "number" || (type == "integer" && number.rounded() == number) else { throw mismatch(type) }
            try bounds(number, minimum: schema["minimum"], maximum: schema["maximum"])
            if let minimum = schema["exclusiveMinimum"]?.doubleValue, number <= minimum { throw mismatch("exclusiveMinimum") }
            if let maximum = schema["exclusiveMaximum"]?.doubleValue, number >= maximum { throw mismatch("exclusiveMaximum") }
        case .bool:
            guard type == nil || type == "boolean" else { throw mismatch(type) }
        case .null:
            guard type == nil || type == "null" else { throw mismatch(type) }
        }
    }

    public static func resolvedType(_ schema: [String: ChatJSONValue], root: [String: ChatJSONValue], depth: Int = 0) throws -> String? {
        guard depth < 64 else { throw ChatRequestIssue("tools", "schema reference nesting exceeds 64 levels") }
        if let type = schema["type"]?.stringValue { return type }
        guard let reference = schema["$ref"]?.stringValue else { return nil }
        guard reference.hasPrefix("#/"), reference.count < 1024 else { throw ChatRequestIssue("tools", "only local schema references are supported") }
        var target: ChatJSONValue = .object(root)
        for part in reference.dropFirst(2).split(separator: "/") {
            let key = part.replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")
            guard let next = target.objectValue?[key] else { throw ChatRequestIssue("tools", "schema reference does not exist") }
            target = next
        }
        guard let object = target.objectValue else { throw ChatRequestIssue("tools", "schema reference must name an object") }
        return try resolvedType(object, root: root, depth: depth + 1)
    }

    private static func mismatch(_ type: String?) -> ChatRequestIssue { ChatRequestIssue("tool_use.input", "expected schema type \(type ?? "unknown")") }

    private static func bounds(_ value: Double, minimum: ChatJSONValue?, maximum: ChatJSONValue?) throws {
        if let minimum = minimum?.doubleValue, value < minimum { throw ChatRequestIssue("tool_use.input", "value is below the schema minimum") }
        if let maximum = maximum?.doubleValue, value > maximum { throw ChatRequestIssue("tool_use.input", "value exceeds the schema maximum") }
    }

    private static func patternMatches(_ pattern: String, string: String) throws -> Bool {
        let regex = try NSRegularExpression(pattern: pattern)
        let deadline = Date().addingTimeInterval(0.25)
        var found = false
        var expired = false
        regex.enumerateMatches(in: string, options: [.reportProgress], range: NSRange(string.startIndex..., in: string)) { result, _, stop in
            if Date() > deadline { expired = true; stop.pointee = true }
            if result != nil { found = true; stop.pointee = true }
        }
        if expired { throw ChatRequestIssue("tool_use.validation_timeout", "schema pattern exceeded the matching deadline") }
        return found
    }
}
