import Foundation

// MARK: - Request Types

public enum OpenAIJSONValue: Codable, Hashable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: OpenAIJSONValue])
    case array([OpenAIJSONValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([OpenAIJSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: OpenAIJSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }

    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    public var boolValue: Bool? {
        if case .bool(let value) = self { return value }
        return nil
    }

    public var objectValue: [String: OpenAIJSONValue]? {
        if case .object(let value) = self { return value }
        return nil
    }

    public var arrayValue: [OpenAIJSONValue]? {
        if case .array(let value) = self { return value }
        return nil
    }
}

private struct OpenAIDynamicCodingKey: CodingKey, Hashable {
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

public struct OpenAIChatRequest: Codable, Sendable {
    public var model: String
    public var messages: [OpenAIChatMessage]
    public var temperature: Double?
    public var top_p: Double?
    public var max_tokens: Int?
    public var max_completion_tokens: Int?
    public var stream: Bool?
    public var stop: OpenAIStopSequence?
    public var seed: Int?
    public var presence_penalty: Double?
    public var frequency_penalty: Double?
    public var logprobs: Bool?
    public var top_logprobs: Int?
    public var reasoning_effort: String?
    public var tools: [OpenAIChatTool]?
    public var tool_choice: OpenAIChatToolChoice?
    public var parallel_tool_calls: Bool?
    public var response_format: OpenAIResponseFormat?
    public var stream_options: OpenAIStreamOptions?
    public var n: Int?
    public var store: Bool?
    public var metadata: [String: String]?
    public var user: String?
    public var service_tier: String?
    public var modalities: [String]?
    public var audio: OpenAIJSONValue?
    public var prediction: OpenAIJSONValue?
    public var think: Bool?
    public var thinking: OpenAIJSONValue?
    public var lora: String?
    public var unknownFields: [String: OpenAIJSONValue]

    private enum CodingKeys: String, CodingKey, CaseIterable {
        case model
        case messages
        case temperature
        case top_p
        case max_tokens
        case max_completion_tokens
        case stream
        case stop
        case seed
        case presence_penalty
        case frequency_penalty
        case logprobs
        case top_logprobs
        case reasoning_effort
        case tools
        case tool_choice
        case parallel_tool_calls
        case response_format
        case stream_options
        case n
        case store
        case metadata
        case user
        case service_tier
        case modalities
        case audio
        case prediction
        case think
        case thinking
        case lora
    }

    public init(
        model: String,
        messages: [OpenAIChatMessage],
        temperature: Double? = nil,
        top_p: Double? = nil,
        max_tokens: Int? = nil,
        max_completion_tokens: Int? = nil,
        stream: Bool? = nil,
        stop: OpenAIStopSequence? = nil,
        seed: Int? = nil,
        presence_penalty: Double? = nil,
        frequency_penalty: Double? = nil,
        logprobs: Bool? = nil,
        top_logprobs: Int? = nil,
        reasoning_effort: String? = nil,
        tools: [OpenAIChatTool]? = nil,
        tool_choice: OpenAIChatToolChoice? = nil,
        parallel_tool_calls: Bool? = nil,
        response_format: OpenAIResponseFormat? = nil,
        stream_options: OpenAIStreamOptions? = nil,
        n: Int? = nil,
        store: Bool? = nil,
        metadata: [String: String]? = nil,
        user: String? = nil,
        service_tier: String? = nil,
        modalities: [String]? = nil,
        audio: OpenAIJSONValue? = nil,
        prediction: OpenAIJSONValue? = nil,
        think: Bool? = nil,
        thinking: OpenAIJSONValue? = nil,
        lora: String? = nil
    ) {
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.stream = stream
        self.stop = stop
        self.seed = seed
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.reasoning_effort = reasoning_effort
        self.tools = tools
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.response_format = response_format
        self.stream_options = stream_options
        self.n = n
        self.store = store
        self.metadata = metadata
        self.user = user
        self.service_tier = service_tier
        self.modalities = modalities
        self.audio = audio
        self.prediction = prediction
        self.think = think
        self.thinking = thinking
        self.lora = lora
        self.unknownFields = [:]
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        messages = try container.decode([OpenAIChatMessage].self, forKey: .messages)
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        top_p = try container.decodeIfPresent(Double.self, forKey: .top_p)
        max_tokens = try container.decodeIfPresent(Int.self, forKey: .max_tokens)
        max_completion_tokens = try container.decodeIfPresent(Int.self, forKey: .max_completion_tokens)
        stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        stop = try container.decodeIfPresent(OpenAIStopSequence.self, forKey: .stop)
        seed = try container.decodeIfPresent(Int.self, forKey: .seed)
        presence_penalty = try container.decodeIfPresent(Double.self, forKey: .presence_penalty)
        frequency_penalty = try container.decodeIfPresent(Double.self, forKey: .frequency_penalty)
        logprobs = try container.decodeIfPresent(Bool.self, forKey: .logprobs)
        top_logprobs = try container.decodeIfPresent(Int.self, forKey: .top_logprobs)
        reasoning_effort = try container.decodeIfPresent(String.self, forKey: .reasoning_effort)
        tools = try container.decodeIfPresent([OpenAIChatTool].self, forKey: .tools)
        tool_choice = try container.decodeIfPresent(OpenAIChatToolChoice.self, forKey: .tool_choice)
        parallel_tool_calls = try container.decodeIfPresent(Bool.self, forKey: .parallel_tool_calls)
        response_format = try container.decodeIfPresent(OpenAIResponseFormat.self, forKey: .response_format)
        stream_options = try container.decodeIfPresent(OpenAIStreamOptions.self, forKey: .stream_options)
        n = try container.decodeIfPresent(Int.self, forKey: .n)
        store = try container.decodeIfPresent(Bool.self, forKey: .store)
        metadata = try container.decodeIfPresent([String: String].self, forKey: .metadata)
        user = try container.decodeIfPresent(String.self, forKey: .user)
        service_tier = try container.decodeIfPresent(String.self, forKey: .service_tier)
        modalities = try container.decodeIfPresent([String].self, forKey: .modalities)
        audio = try container.decodeIfPresent(OpenAIJSONValue.self, forKey: .audio)
        prediction = try container.decodeIfPresent(OpenAIJSONValue.self, forKey: .prediction)
        think = try container.decodeIfPresent(Bool.self, forKey: .think)
        thinking = try container.decodeIfPresent(OpenAIJSONValue.self, forKey: .thinking)
        lora = try container.decodeIfPresent(String.self, forKey: .lora)

        let knownKeys = Set(CodingKeys.allCases.map(\.rawValue))
        let dynamic = try decoder.container(keyedBy: OpenAIDynamicCodingKey.self)
        var unknown: [String: OpenAIJSONValue] = [:]
        for key in dynamic.allKeys where !knownKeys.contains(key.stringValue) {
            unknown[key.stringValue] = try dynamic.decode(OpenAIJSONValue.self, forKey: key)
        }
        unknownFields = unknown
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(model, forKey: .model)
        try container.encode(messages, forKey: .messages)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(top_p, forKey: .top_p)
        try container.encodeIfPresent(max_tokens, forKey: .max_tokens)
        try container.encodeIfPresent(max_completion_tokens, forKey: .max_completion_tokens)
        try container.encodeIfPresent(stream, forKey: .stream)
        try container.encodeIfPresent(stop, forKey: .stop)
        try container.encodeIfPresent(seed, forKey: .seed)
        try container.encodeIfPresent(presence_penalty, forKey: .presence_penalty)
        try container.encodeIfPresent(frequency_penalty, forKey: .frequency_penalty)
        try container.encodeIfPresent(logprobs, forKey: .logprobs)
        try container.encodeIfPresent(top_logprobs, forKey: .top_logprobs)
        try container.encodeIfPresent(reasoning_effort, forKey: .reasoning_effort)
        try container.encodeIfPresent(tools, forKey: .tools)
        try container.encodeIfPresent(tool_choice, forKey: .tool_choice)
        try container.encodeIfPresent(parallel_tool_calls, forKey: .parallel_tool_calls)
        try container.encodeIfPresent(response_format, forKey: .response_format)
        try container.encodeIfPresent(stream_options, forKey: .stream_options)
        try container.encodeIfPresent(n, forKey: .n)
        try container.encodeIfPresent(store, forKey: .store)
        try container.encodeIfPresent(metadata, forKey: .metadata)
        try container.encodeIfPresent(user, forKey: .user)
        try container.encodeIfPresent(service_tier, forKey: .service_tier)
        try container.encodeIfPresent(modalities, forKey: .modalities)
        try container.encodeIfPresent(audio, forKey: .audio)
        try container.encodeIfPresent(prediction, forKey: .prediction)
        try container.encodeIfPresent(think, forKey: .think)
        try container.encodeIfPresent(thinking, forKey: .thinking)
        try container.encodeIfPresent(lora, forKey: .lora)

        var dynamic = encoder.container(keyedBy: OpenAIDynamicCodingKey.self)
        for (key, value) in unknownFields {
            guard let codingKey = OpenAIDynamicCodingKey(stringValue: key) else { continue }
            try dynamic.encode(value, forKey: codingKey)
        }
    }
}

public struct OpenAIChatMessage: Codable, Sendable {
    public var role: String
    public var content: String
    public var name: String?
    public var tool_call_id: String?
    public var tool_calls: [OpenAIChatToolCall]?
    public var imageURLs: [String]

    public init(
        role: String,
        content: String = "",
        name: String? = nil,
        tool_call_id: String? = nil,
        tool_calls: [OpenAIChatToolCall]? = nil,
        imageURLs: [String] = []
    ) {
        self.role = role
        self.content = content
        self.name = name
        self.tool_call_id = tool_call_id
        self.tool_calls = tool_calls
        self.imageURLs = imageURLs
    }

    private enum CodingKeys: String, CodingKey {
        case role
        case content
        case name
        case tool_call_id
        case tool_calls
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        let decoded = try container.decodeFlexibleContentIfPresent(forKey: .content)
        content = decoded.text
        imageURLs = decoded.imageURLs
        name = try container.decodeIfPresent(String.self, forKey: .name)
        tool_call_id = try container.decodeIfPresent(String.self, forKey: .tool_call_id)
        tool_calls = try container.decodeIfPresent([OpenAIChatToolCall].self, forKey: .tool_calls)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encode(content, forKey: .content)
        try container.encodeIfPresent(name, forKey: .name)
        try container.encodeIfPresent(tool_call_id, forKey: .tool_call_id)
        try container.encodeIfPresent(tool_calls, forKey: .tool_calls)
    }
}

public struct OpenAIChatToolCall: Codable, Hashable, Sendable {
    public var id: String
    public var type: String
    public var function: OpenAIChatToolCallFunction?

    public init(id: String, type: String = "function", function: OpenAIChatToolCallFunction? = nil) {
        self.id = id
        self.type = type
        self.function = function
    }
}

public struct OpenAIChatToolCallFunction: Codable, Hashable, Sendable {
    public var name: String
    public var arguments: String

    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }
}

public struct OpenAIChatTool: Codable, Hashable, Sendable {
    public var type: String
    public var function: OpenAIChatToolFunction?

    public init(type: String = "function", function: OpenAIChatToolFunction? = nil) {
        self.type = type
        self.function = function
    }
}

public struct OpenAIChatToolFunction: Codable, Hashable, Sendable {
    public var name: String
    public var description: String?
    public var parameters: OpenAIJSONValue?
    public var strict: Bool?

    public init(
        name: String,
        description: String? = nil,
        parameters: OpenAIJSONValue? = nil,
        strict: Bool? = nil
    ) {
        self.name = name
        self.description = description
        self.parameters = parameters
        self.strict = strict
    }
}

public enum OpenAIChatToolChoice: Codable, Hashable, Sendable {
    case mode(String)
    case function(name: String)
    case custom(OpenAIJSONValue)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .mode(value)
            return
        }
        let object = try container.decode([String: OpenAIJSONValue].self)
        if object["type"]?.stringValue == "function",
           let function = object["function"]?.objectValue,
           let name = function["name"]?.stringValue {
            self = .function(name: name)
        } else {
            self = .custom(.object(object))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .mode(let value):
            try container.encode(value)
        case .function(let name):
            try container.encode([
                "type": OpenAIJSONValue.string("function"),
                "function": .object(["name": .string(name)]),
            ])
        case .custom(let value):
            try container.encode(value)
        }
    }
}

public enum OpenAIStopSequence: Codable, Hashable, Sendable {
    case string(String)
    case array([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .string(value)
        } else {
            self = .array(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        }
    }

    public var values: [String] {
        switch self {
        case .string(let value):
            return value.isEmpty ? [] : [value]
        case .array(let value):
            return value.filter { !$0.isEmpty }
        }
    }
}

public struct OpenAIResponseFormat: Codable, Hashable, Sendable {
    public var type: String
    public var json_schema: OpenAIJSONValue?

    public init(type: String, json_schema: OpenAIJSONValue? = nil) {
        self.type = type
        self.json_schema = json_schema
    }
}

public struct OpenAIStreamOptions: Codable, Hashable, Sendable {
    public var include_usage: Bool?

    public init(include_usage: Bool? = nil) {
        self.include_usage = include_usage
    }
}

private struct OpenAIChatContent: Sendable {
    var text: String
    var imageURLs: [String]
}

public struct OpenAIImageURL: Codable, Hashable, Sendable {
    public var url: String
    public var detail: String?

    private enum CodingKeys: String, CodingKey {
        case url
        case detail
    }

    public init(url: String, detail: String? = nil) {
        self.url = url
        self.detail = detail
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self.url = value
            self.detail = nil
            return
        }
        let object = try decoder.container(keyedBy: CodingKeys.self)
        self.url = try object.decode(String.self, forKey: .url)
        self.detail = try object.decodeIfPresent(String.self, forKey: .detail)
    }
}

public struct OpenAIChatContentPart: Codable, Hashable, Sendable {
    public var type: String?
    public var text: String?
    public var image_url: OpenAIImageURL?
    public var input_image: OpenAIImageURL?

    public init(type: String? = nil, text: String? = nil, image_url: OpenAIImageURL? = nil, input_image: OpenAIImageURL? = nil) {
        self.type = type
        self.text = text
        self.image_url = image_url
        self.input_image = input_image
    }

    var extractedImageURL: String? {
        image_url?.url ?? input_image?.url
    }
}

private extension KeyedDecodingContainer {
    func decodeFlexibleContentIfPresent(forKey key: Key) throws -> OpenAIChatContent {
        guard contains(key) else {
            return OpenAIChatContent(text: "", imageURLs: [])
        }
        if try decodeNil(forKey: key) {
            return OpenAIChatContent(text: "", imageURLs: [])
        }
        if let content = try? decode(String.self, forKey: key) {
            return OpenAIChatContent(text: content, imageURLs: [])
        }
        if let parts = try? decode([OpenAIChatContentPart].self, forKey: key) {
            let text = parts
                .filter { $0.type == nil || $0.type == "text" || $0.type == "input_text" }
                .compactMap(\.text)
                .joined(separator: "\n")
            let imageURLs = parts.compactMap(\.extractedImageURL)
            return OpenAIChatContent(text: text, imageURLs: imageURLs)
        }
        throw DecodingError.typeMismatch(
            String.self,
            DecodingError.Context(
                codingPath: codingPath + [key],
                debugDescription: "Expected chat message content to be a string, null, or text content parts."
            )
        )
    }
}

// MARK: - Response Types

public struct OpenAIChatResponse: Codable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var model: String
    public var choices: [OpenAIChatChoice]
    public var usage: OpenAIUsage?

    public init(
        id: String,
        object: String,
        created: Int,
        model: String,
        choices: [OpenAIChatChoice],
        usage: OpenAIUsage? = nil
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct OpenAIChatChoice: Codable, Sendable {
    public var index: Int
    public var message: OpenAIChatMessage?
    public var delta: OpenAIChatDelta?
    public var finish_reason: String?

    public init(
        index: Int,
        message: OpenAIChatMessage? = nil,
        delta: OpenAIChatDelta? = nil,
        finish_reason: String? = nil
    ) {
        self.index = index
        self.message = message
        self.delta = delta
        self.finish_reason = finish_reason
    }
}

public struct OpenAIChatDelta: Codable, Sendable {
    public var role: String?
    public var content: String?
    public var tool_calls: [OpenAIChatToolCallDelta]?

    public init(role: String? = nil, content: String? = nil, tool_calls: [OpenAIChatToolCallDelta]? = nil) {
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
    }
}

public struct OpenAIChatToolCallDelta: Codable, Hashable, Sendable {
    public var index: Int
    public var id: String?
    public var type: String?
    public var function: OpenAIChatToolCallFunction?

    public init(index: Int, id: String? = nil, type: String? = nil, function: OpenAIChatToolCallFunction? = nil) {
        self.index = index
        self.id = id
        self.type = type
        self.function = function
    }

    public init(indexAndToolCall: (offset: Int, element: OpenAIChatToolCall)) {
        self.index = indexAndToolCall.offset
        self.id = indexAndToolCall.element.id
        self.type = indexAndToolCall.element.type
        self.function = indexAndToolCall.element.function
    }

    public init(_ toolCall: OpenAIChatToolCall) {
        self.index = 0
        self.id = toolCall.id
        self.type = toolCall.type
        self.function = toolCall.function
    }
}

public struct OpenAIUsage: Codable, Sendable {
    public var prompt_tokens: Int
    public var completion_tokens: Int
    public var total_tokens: Int

    public init(prompt_tokens: Int, completion_tokens: Int, total_tokens: Int) {
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
    }
}

// MARK: - Models Endpoint

public struct OpenAIModelsResponse: Codable, Sendable {
    public var object: String
    public var data: [OpenAIModel]

    public init(object: String, data: [OpenAIModel]) {
        self.object = object
        self.data = data
    }
}

public struct OpenAIModel: Codable, Sendable {
    public var id: String
    public var object: String
    public var created: Int
    public var owned_by: String

    public init(id: String, object: String, created: Int, owned_by: String) {
        self.id = id
        self.object = object
        self.created = created
        self.owned_by = owned_by
    }
}

// MARK: - Error Response

public struct OpenAIErrorResponse: Codable, Sendable {
    public var error: OpenAIError

    public init(error: OpenAIError) {
        self.error = error
    }
}

public struct OpenAIError: Codable, Sendable {
    public var message: String
    public var type: String
    public var code: String?

    public init(message: String, type: String, code: String? = nil) {
        self.message = message
        self.type = type
        self.code = code
    }
}
