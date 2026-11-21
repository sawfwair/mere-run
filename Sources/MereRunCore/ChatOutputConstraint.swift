import Foundation
import MLX

public struct ChatOutputConstraint: Hashable, Sendable {
    public enum Mode: Hashable, Sendable { case json, tools }
    public let schema: [String: ChatJSONValue]
    public let mode: Mode
    public init(schema: [String: ChatJSONValue], mode: Mode = .json) throws {
        try ChatToolValidation.validateSchema(schema)
        _ = try ChatSchemaGrammar(schema: schema)
        self.schema = schema; self.mode = mode
    }

    public static func tools(_ tools: [ToolDefinition], choice: ChatToolChoice, parallel: Bool) throws -> Self {
        var alternatives: [ChatJSONValue] = []
        var definitions: [String: ChatJSONValue] = [:]
        for (index, tool) in tools.enumerated() {
            if case .function(let name) = choice, name != tool.name { continue }
            let key = "tool_\(index)"
            definitions[key] = rebase(.object(tool.parametersJSONSchema), prefix: "#/$defs/" + key)
            alternatives.append(.object([
                "type": .string("object"), "additionalProperties": .bool(false), "description": .string(tool.description),
                "properties": .object(["name": .object(["const": .string(tool.name)]), "arguments": .object(["$ref": .string("#/$defs/" + key)])]),
                "required": .array([.string("name"), .string("arguments")])
            ]))
        }
        guard !alternatives.isEmpty else { throw ChatRequestIssue("tools", "strict generation requires an active tool") }
        let calls: [String: ChatJSONValue] = ["type": .string("object"), "additionalProperties": .bool(false),
            "properties": .object(["tool_calls": .object(["type": .string("array"), "minItems": .number(1),
                "maxItems": .number(parallel ? 16 : 1), "items": .object(["anyOf": .array(alternatives)])])]),
            "required": .array([.string("tool_calls")])]
        var root = calls
        if choice == .auto {
            root = ["anyOf": .array([.object(calls), .object(["type": .string("object"), "additionalProperties": .bool(false),
                "properties": .object(["text": .object(["type": .string("string")])]), "required": .array([.string("text")])])])]
        }
        root["$defs"] = .object(definitions)
        return try Self(schema: root, mode: .tools)
    }

    /// Validate native channel grammar and tool argument schemas before generation.
    public func validateNativeRecipient(_ request: ChatRequest) throws { _ = try MuseGlimmerConstraint(request: request) }

    private static func rebase(_ value: ChatJSONValue, prefix: String) -> ChatJSONValue {
        guard var schema = value.objectValue else { return value }
        if let reference = schema["$ref"]?.stringValue {
            schema["$ref"] = .string(prefix + reference.dropFirst(1))
        }
        for keyword in ["properties", "$defs", "definitions"] {
            if let children = schema[keyword]?.objectValue {
                schema[keyword] = .object(children.mapValues { rebase($0, prefix: prefix) })
            }
        }
        if let items = schema["items"] { schema["items"] = rebase(items, prefix: prefix) }
        if let alternatives = schema["anyOf"]?.arrayValue {
            schema["anyOf"] = .array(alternatives.map { rebase($0, prefix: prefix) })
        }
        // enum/const/default/examples are data, including any literal $ref keys.
        return .object(schema)
    }

    func visiblePrefix(_ raw: String) -> String {
        guard mode == .tools else { return raw }
        let prefix = "{\"text\":"
        guard raw.hasPrefix(prefix) else { return "" }
        let value = String(raw.dropFirst(prefix.count))
        guard value.first == "\"" else { return "" }
        var literal = "\""
        var escaped = false
        for scalar in value.unicodeScalars.dropFirst() {
            literal.unicodeScalars.append(scalar)
            if scalar == "\"", !escaped {
                return (try? JSONDecoder().decode(String.self, from: Data(literal.utf8))) ?? ""
            }
            if scalar == "\\" { escaped.toggle() } else { escaped = false }
        }
        // A trailing JSON escape or UTF-16 surrogate pair can span tokens.
        // Only those at-most-twelve scalars are withheld from the visible text.
        for _ in 0...12 {
            if let text = try? JSONDecoder().decode(String.self, from: Data((literal + "\"").utf8)) { return text }
            guard literal.unicodeScalars.count > 1 else { break }
            literal.unicodeScalars.removeLast()
        }
        return ""
    }

    func finish(_ response: ChatResponse, raw: String, tools: [ToolDefinition]) throws -> ChatResponse {
        let value: ChatJSONValue
        do { value = try JSONDecoder().decode(ChatJSONValue.self, from: Data(raw.utf8)) }
        catch { throw ChatRequestIssue("output_schema.generation", "output budget ended before a valid constrained response completed; increase max_tokens") }
        try ChatToolValidation.validate(value, schema: schema)
        var result = response
        result.reasoningContent = nil
        result.generationEvents = []
        switch mode {
        case .json:
            result.response = raw
            result.generationEvents = [.text(raw)]
        case .tools:
            if let text = value.objectValue?["text"]?.stringValue {
                result.response = text; result.generationEvents = [.text(text)]
            } else {
                let values = value.objectValue?["tool_calls"]?.arrayValue ?? []
                let calls = try values.map { value -> ToolCall in
                    guard let name = value.objectValue?["name"]?.stringValue,
                          let input = value.objectValue?["arguments"]?.objectValue,
                          let tool = tools.first(where: { $0.name == name }) else {
                        throw ChatRequestIssue("output_schema.generation", "constrained tool envelope is invalid")
                    }
                    try ChatToolValidation.validate(.object(input), schema: tool.parametersJSONSchema)
                    return ToolCall(name: name, jsonArguments: input)
                }
                result.response = ""; result.toolCalls = calls
            }
        }
        result.finishReason = .stop
        return result
    }
}

struct ChatSchemaDecodeResult {
    let tokens: [Int]
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
}

/// Constraints run on final-target decoding. Candidate rejection is bounded;
/// exhaustion fails explicitly instead of weakening the grammar or emitting a
/// malformed tool. Sampling controls remain active for every draw.
func schemaConstrainedDecode(
    constraint: ChatOutputConstraint, initialLogits: MLXArray, sampling: GenerationConfig,
    eos: Set<Int>, budget: Int, history: [Int], decode: ([Int]) -> String,
    forward: (Int) throws -> MLXArray,
    progress: (@Sendable (ChatProgress) -> Void)?
) throws -> ChatSchemaDecodeResult {
    let started = Date()
    let grammar = try ChatSchemaGrammar(schema: constraint.schema)
    var logits = initialLogits
    var generated: [Int] = []
    var first: Double?
    var emitted = ""
    var visible = ""
    var complete = false
    for _ in 0..<budget {
        try Task.checkCancellation()
        var config = sampling
        if !complete { config.bannedTokens += Array(eos) }
        var accepted: (Int, String, Bool)?
        for _ in 0..<256 {
            try Task.checkCancellation()
            let token = sampleToken(logits: logits[0, -1, 0...], config: config, previousTokens: history + generated)
            guard !config.bannedTokens.contains(token) else { break }
            if eos.contains(token), complete {
                return .init(tokens: generated, decodeSeconds: Date().timeIntervalSince(started), firstTokenSeconds: first)
            }
            var text = decode(generated + [token])
            let hadPartialScalar = text.last == "\u{FFFD}"
            while text.last == "\u{FFFD}" { text.removeLast() }
            var probe = grammar
            if (!text.isEmpty || hadPartialScalar), text.utf8.starts(with: emitted.utf8), probe.accept(text) {
                // Empty-decoding control tokens cannot advance the response.
                if text != decode(generated) || hadPartialScalar {
                    accepted = (token, text, probe.isComplete && !hadPartialScalar); break
                }
            }
            config.bannedTokens.append(token)
        }
        guard let (token, text, completed) = accepted else {
            throw ChatRequestIssue("output_schema.generation", "no admissible token found after \(generated.count) output tokens within the constrained sampling limit")
        }
        generated.append(token)
        if first == nil { first = Date().timeIntervalSince(started) }
        let nextVisible = constraint.visiblePrefix(text)
        if nextVisible.utf8.starts(with: visible.utf8) {
            let delta = String(decoding: nextVisible.utf8.dropFirst(visible.utf8.count), as: UTF8.self)
            if !delta.isEmpty { progress?(.init(stage: .generating, event: .text(delta))) }
            visible = nextVisible
        }
        emitted = text
        complete = completed
        if complete, text.last == "}" || text.last == "]" || text.last == "\"" || text == "true" || text == "false" || text == "null" { return .init(tokens: generated, decodeSeconds: Date().timeIntervalSince(started), firstTokenSeconds: first) }
        logits = try forward(token)
        MLX.eval(logits)
    }
    if complete { return .init(tokens: generated, decodeSeconds: Date().timeIntervalSince(started), firstTokenSeconds: first) }
    throw ChatRequestIssue("output_schema.generation", "output budget ended before the constrained response completed; increase max_tokens")
}
