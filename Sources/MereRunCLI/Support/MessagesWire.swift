import Foundation
import MereRunCore

// Anthropic wire contracts are confined to CLI Support.
enum MessagesVersion: String, CaseIterable, Sendable {
    case original = "2023-06-01"
    case october2026 = "2026-10-21"

    static func resolve(_ value: String?) throws -> Self {
        guard let value else { return .original }
        guard let version = Self(rawValue: value) else {
            throw MessagesFailure.invalid("anthropic-version", "supported versions are 2023-06-01 and 2026-10-21")
        }
        return version
    }
}

struct MessagesFailure: Error, LocalizedError, Sendable {
    let status: Int
    let type: String
    let message: String
    var errorDescription: String? { message }

    static func invalid(_ field: String, _ reason: String) -> Self {
        Self(status: 400, type: "invalid_request_error", message: "Invalid '\(field)': \(reason).")
    }
    static func output(_ field: String, _ reason: String) -> Self {
        Self(status: 500, type: "api_error", message: "Local adapter output error in '\(field)': \(reason).")
    }
    static func unsupported(_ field: String, model: String? = nil) -> Self {
        invalid(field, "\(model.map { "adapter for '\($0)'" } ?? "local inference") does not support this capability; select a compatible adapter or omit this option")
    }
}

struct MessagesCacheControl: Codable, Equatable, Sendable {
    let type: String
    let ttl: String?
    func validate() throws {
        guard type == "ephemeral", ttl == nil || ttl == "5m" || ttl == "1h" else {
            throw MessagesFailure.invalid("cache_control", "use ephemeral with ttl 5m or 1h")
        }
    }
}

enum MessagesContent: Codable, Sendable {
    case text(String)
    case blocks([MessagesBlock])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) { self = .text(text) }
        else { self = .blocks(try container.decode([MessagesBlock].self)) }
    }
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text): try container.encode(text)
        case .blocks(let blocks): try container.encode(blocks)
        }
    }
    var blocks: [MessagesBlock] {
        switch self {
        case .text(let text): [.text(.init(text: text))]
        case .blocks(let blocks): blocks
        }
    }
}

struct MessagesText: Codable, Sendable {
    let text: String
    var cache_control: MessagesCacheControl?
    var citations: [ChatJSONValue]?
}
struct MessagesMediaSource: Codable, Sendable {
    let type: String
    let media_type: String?
    let data: String?
    let url: String?
    let content: MessagesContent?
}
struct MessagesImage: Codable, Sendable {
    let source: MessagesMediaSource
    let cache_control: MessagesCacheControl?
}
struct MessagesDocument: Codable, Sendable {
    let source: MessagesMediaSource
    let title: String?
    let context: String?
    let citations: CitationConfiguration?
    let cache_control: MessagesCacheControl?
    struct CitationConfiguration: Codable, Sendable { let enabled: Bool }
}
struct MessagesThinking: Codable, Sendable { let thinking: String; let signature: String }
struct MessagesToolUse: Codable, Sendable {
    let id: String
    let name: String
    let input: [String: ChatJSONValue]
    var cache_control: MessagesCacheControl?
}
struct MessagesToolResult: Codable, Sendable {
    let tool_use_id: String
    let content: MessagesContent?
    let is_error: Bool?
    let cache_control: MessagesCacheControl?
}
struct MessagesToolReference: Codable, Sendable { let tool_name: String }

struct MessagesSearchResult: Codable, Sendable {
    let tool_use_id: String
    let content: ResultContent
    struct ResultContent: Codable, Sendable {
        let type: String
        var tool_references: [MessagesBlock]?
        var error_code: String?
        var error_message: String?
    }
}

indirect enum MessagesBlock: Codable, Sendable {
    case text(MessagesText)
    case image(MessagesImage)
    case document(MessagesDocument)
    case thinking(MessagesThinking)
    case toolUse(MessagesToolUse)
    case toolResult(MessagesToolResult)
    case toolReference(MessagesToolReference)
    case serverToolUse(MessagesToolUse)
    case toolSearchResult(MessagesSearchResult)

    enum Key: String, CodingKey { case type }
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Key.self)
        switch try container.decode(String.self, forKey: .type) {
        case "text": self = .text(try .init(from: decoder))
        case "image": self = .image(try .init(from: decoder))
        case "document": self = .document(try .init(from: decoder))
        case "thinking": self = .thinking(try .init(from: decoder))
        case "tool_use": self = .toolUse(try .init(from: decoder))
        case "tool_result": self = .toolResult(try .init(from: decoder))
        case "tool_reference": self = .toolReference(try .init(from: decoder))
        case "server_tool_use": self = .serverToolUse(try .init(from: decoder))
        case "tool_search_tool_result": self = .toolSearchResult(try .init(from: decoder))
        case let type: throw MessagesFailure.unsupported("content.\(type)")
        }
    }
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: Key.self)
        switch self {
        case .text(let block):
            try container.encode("text", forKey: .type); try block.encode(to: encoder)
        case .image(let block):
            try container.encode("image", forKey: .type); try block.encode(to: encoder)
        case .document(let block):
            try container.encode("document", forKey: .type); try block.encode(to: encoder)
        case .thinking(let block):
            try container.encode("thinking", forKey: .type); try block.encode(to: encoder)
        case .toolUse(let block):
            try container.encode("tool_use", forKey: .type); try block.encode(to: encoder)
        case .toolResult(let block):
            try container.encode("tool_result", forKey: .type); try block.encode(to: encoder)
        case .toolReference(let block):
            try container.encode("tool_reference", forKey: .type); try block.encode(to: encoder)
        case .serverToolUse(let block):
            try container.encode("server_tool_use", forKey: .type); try block.encode(to: encoder)
        case .toolSearchResult(let block):
            try container.encode("tool_search_tool_result", forKey: .type); try block.encode(to: encoder)
        }
    }
}

struct MessagesTurn: Codable, Sendable {
    let role: String
    let content: MessagesContent
}
struct MessagesTool: Codable, Sendable {
    let type: String?
    let name: String
    let description: String?
    let input_schema: [String: ChatJSONValue]?
    let strict: Bool?
    let defer_loading: Bool?
    let cache_control: MessagesCacheControl?
    let max_uses: Int?
    let allowed_callers: [String]?
    let input_examples: [[String: ChatJSONValue]]?
}
struct MessagesToolChoice: Codable, Sendable {
    let type: String
    let name: String?
    let disable_parallel_tool_use: Bool?
}
struct MessagesThinkingConfiguration: Codable, Sendable {
    let type: String
    let budget_tokens: Int?
    let display: String?
}
struct MessagesOutputConfiguration: Codable, Sendable {
    let effort: String?
    let format: ChatJSONValue?
}
struct MessagesContextManagement: Codable, Sendable {
    let edits: [Edit]
    struct Edit: Codable, Sendable {
        let type: String
        let trigger: Quantity?
        let keep: ChatJSONValue?
        let clear_at_least: Quantity?
        let exclude_tools: [String]?
        let clear_tool_inputs: Bool?
    }
    struct Quantity: Codable, Sendable { let type: String; let value: Int }
}

struct MessagesRequest: Decodable, Sendable {
    let model: String
    let messages: [MessagesTurn]
    let max_tokens: Int?
    let stream: Bool?
    let system: MessagesContent?
    let tools: [MessagesTool]?
    let tool_choice: MessagesToolChoice?
    let thinking: MessagesThinkingConfiguration?
    let output_config: MessagesOutputConfiguration?
    let stop_sequences: [String]?
    let temperature: Double?
    let top_p: Double?
    let top_k: Int?
    let cache_control: MessagesCacheControl?
    let context_management: MessagesContextManagement?
    let metadata: [String: ChatJSONValue]?
    let service_tier: String?
    let unsupportedFields: [String]
    let samplingFields: [String]

    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        func field<T: Decodable>(_ name: String, _ type: T.Type = T.self) throws -> T? {
            try c.decodeIfPresent(type, forKey: Key(stringValue: name)!)
        }
        model = try c.decode(String.self, forKey: Key(stringValue: "model")!)
        messages = try c.decode([MessagesTurn].self, forKey: Key(stringValue: "messages")!)
        max_tokens = try field("max_tokens"); stream = try field("stream"); system = try field("system")
        tools = try field("tools"); tool_choice = try field("tool_choice"); thinking = try field("thinking")
        output_config = try field("output_config"); stop_sequences = try field("stop_sequences")
        temperature = try field("temperature"); top_p = try field("top_p"); top_k = try field("top_k")
        cache_control = try field("cache_control"); context_management = try field("context_management")
        metadata = try field("metadata"); service_tier = try field("service_tier")
        let known: Set<String> = ["model", "messages", "max_tokens", "stream", "system", "tools", "tool_choice", "thinking",
                                  "output_config", "stop_sequences", "temperature", "top_p", "top_k", "cache_control",
                                  "context_management", "metadata", "service_tier"]
        unsupportedFields = c.allKeys.map(\.stringValue).filter { !known.contains($0) }.sorted()
        samplingFields = ["temperature", "top_p", "top_k"].filter { c.contains(Key(stringValue: $0)!) }
    }

    func validate(version: MessagesVersion, counting: Bool) throws {
        if let field = unsupportedFields.first { throw MessagesFailure.unsupported(field) }
        guard !model.isEmpty, !messages.isEmpty else { throw MessagesFailure.invalid("messages", "model and messages must be non-empty") }
        if !counting, max_tokens == nil { throw MessagesFailure.invalid("max_tokens", "is required") }
        if let max_tokens, max_tokens < 0 { throw MessagesFailure.invalid("max_tokens", "must be non-negative") }
        if version == .october2026, !samplingFields.isEmpty {
            throw MessagesFailure.invalid("sampling", "temperature, top_p and top_k are removed in API version 2026-10-21")
        }
        if let service_tier, service_tier != "auto" { throw MessagesFailure.unsupported("service_tier") }
        try cache_control?.validate()
        let controls = (tools ?? []).compactMap(\.cache_control) + Self.cacheControls(system?.blocks ?? [])
            + Self.cacheControls(messages.flatMap { $0.content.blocks }) + (cache_control.map { [$0] } ?? [])
        guard controls.count <= 4 else { throw MessagesFailure.invalid("cache_control", "at most four cache breakpoints are supported") }
        var hasShortTTL = false
        for control in controls {
            try control.validate()
            if control.ttl == "1h", hasShortTTL { throw MessagesFailure.invalid("cache_control.ttl", "one-hour cache breakpoints must precede five-minute breakpoints") }
            if control.ttl != "1h" { hasShortTTL = true }
        }
        for message in messages {
            guard ["user", "assistant", "system"].contains(message.role) else { throw MessagesFailure.invalid("messages.role", "expected user, assistant or system") }
            for block in message.content.blocks {
                if version == .october2026, message.role == "assistant", case .document = block {
                    throw MessagesFailure.invalid("messages.content.document", "send documents in user messages under API version 2026-10-21")
                }
                try Self.validate(block, role: message.role, version: version, nested: false)
            }
        }
    }

    private static func cacheControls(_ blocks: [MessagesBlock]) -> [MessagesCacheControl] {
        blocks.flatMap { block in
            let own = MessagesAdapter.cacheControl(block).map { [$0] } ?? []
            if case .toolResult(let result) = block { return own + cacheControls(result.content?.blocks ?? []) }
            return own
        }
    }

    private static func validate(_ block: MessagesBlock, role: String, version: MessagesVersion, nested: Bool) throws {
        switch block {
        case .text(let text):
            try text.cache_control?.validate()
            if version == .october2026, role == "user", text.citations?.isEmpty == false {
                throw MessagesFailure.invalid("citations", "user text cannot contain citations under API version 2026-10-21")
            }
        case .image(let image): try image.cache_control?.validate()
        case .document(let document): try document.cache_control?.validate()
        case .thinking:
            guard role == "assistant", !nested else { throw MessagesFailure.invalid("thinking", "must appear in assistant content") }
        case .toolUse(let call):
            guard role == "assistant", !nested, !call.id.isEmpty, !call.name.isEmpty else {
                throw MessagesFailure.invalid("tool_use", "requires assistant role and non-empty id and name")
            }
            try call.cache_control?.validate()
        case .toolResult(let result):
            guard role == "user", !nested else { throw MessagesFailure.invalid("tool_result", "must appear directly in user content") }
            try result.cache_control?.validate()
            for child in result.content?.blocks ?? [] { try validate(child, role: role, version: version, nested: true) }
        case .serverToolUse:
            guard role == "assistant", !nested else { throw MessagesFailure.invalid("server_tool_use", "requires assistant role") }
        case .toolSearchResult(let result):
            guard role == "assistant", !nested else { throw MessagesFailure.invalid("tool_search_tool_result", "requires assistant role") }
            guard ["tool_search_tool_search_result", "tool_search_tool_result_error"].contains(result.content.type),
                  (result.content.tool_references ?? []).allSatisfy({ if case .toolReference(let reference) = $0 { !reference.tool_name.isEmpty } else { false } }) else {
                throw MessagesFailure.invalid("tool_search_tool_result", "requires documented search references or a search error")
            }
        case .toolReference:
            guard nested else { throw MessagesFailure.invalid("tool_reference", "must appear in a tool result") }
        }
    }
}

struct MessagesUsage: Encodable, Sendable {
    var input_tokens: Int
    var output_tokens: Int
    var cache_creation_input_tokens = 0
    var cache_read_input_tokens = 0
}
struct MessagesResponse: Encodable, Sendable {
    let id: String
    let type = "message"
    let role = "assistant"
    let model: String
    let content: [MessagesBlock]
    let stop_reason: String?
    let stop_sequence: String?
    let usage: MessagesUsage
    var context_management: ChatJSONValue?

    enum CodingKeys: String, CodingKey { case id, type, role, model, content, stop_reason, stop_sequence, usage, context_management }
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id); try c.encode(type, forKey: .type); try c.encode(role, forKey: .role)
        try c.encode(model, forKey: .model); try c.encode(content, forKey: .content); try c.encode(usage, forKey: .usage)
        try c.encodeIfPresent(context_management, forKey: .context_management)
        try c.encode(stop_reason, forKey: .stop_reason); try c.encode(stop_sequence, forKey: .stop_sequence)
    }
}
struct MessagesErrorEnvelope: Encodable, Sendable {
    let type = "error"
    let error: Detail
    let request_id: String
    struct Detail: Encodable, Sendable { let type: String; let message: String }
}
