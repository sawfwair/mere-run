import Foundation

/// Deterministic JSON with the spacing used by the upstream Jinja tojson filter.
enum LFM2PromptJSON {
    static func render(_ value: OpenAIJSONValue) throws -> String {
        switch value {
        case .object(let fields):
            let parts = try fields.keys.sorted().map { key in
                try quote(key) + ": " + render(fields[key]!)
            }
            return "{" + parts.joined(separator: ", ") + "}"
        case .array(let values):
            return "[" + (try values.map(render)).joined(separator: ", ") + "]"
        default:
            let encoder = JSONEncoder()
            encoder.outputFormatting = .withoutEscapingSlashes
            return String(decoding: try encoder.encode(value), as: UTF8.self)
        }
    }

    static func tool(_ definition: ToolDefinition) throws -> String {
        try render(.object([
            "type": .string("function"),
            "function": .object([
                "name": .string(definition.name),
                "description": .string(definition.description),
                "parameters": .object(definition.parametersJSONSchema),
            ]),
        ]))
    }

    private static func quote(_ text: String) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .withoutEscapingSlashes
        return String(decoding: try encoder.encode(text), as: UTF8.self)
    }
}
