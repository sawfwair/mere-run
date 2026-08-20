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
    public var min_p: Double?
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
        case min_p
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
        min_p: Double? = nil,
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
        self.min_p = min_p
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
        min_p = try container.decodeIfPresent(Double.self, forKey: .min_p)
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
        try container.encodeIfPresent(min_p, forKey: .min_p)
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

public enum OpenAIEmbeddingInput: Codable, Hashable, Sendable {
    case string(String)
    case array([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .string(value)
            return
        }
        self = .array(try container.decode([String].self))
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

    public var texts: [String] {
        switch self {
        case .string(let value):
            return [value]
        case .array(let value):
            return value
        }
    }
}

public struct OpenAIEmbeddingRequest: Codable, Sendable {
    public var model: String
    public var input: OpenAIEmbeddingInput
    public var encoding_format: String?
    public var dimensions: Int?
    public var user: String?
    public var unknownFields: [String: OpenAIJSONValue]

    private enum CodingKeys: String, CodingKey, CaseIterable {
        case model
        case input
        case encoding_format
        case dimensions
        case user
    }

    public init(
        model: String,
        input: OpenAIEmbeddingInput,
        encoding_format: String? = nil,
        dimensions: Int? = nil,
        user: String? = nil
    ) {
        self.model = model
        self.input = input
        self.encoding_format = encoding_format
        self.dimensions = dimensions
        self.user = user
        self.unknownFields = [:]
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        input = try container.decode(OpenAIEmbeddingInput.self, forKey: .input)
        encoding_format = try container.decodeIfPresent(String.self, forKey: .encoding_format)
        dimensions = try container.decodeIfPresent(Int.self, forKey: .dimensions)
        user = try container.decodeIfPresent(String.self, forKey: .user)

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
        try container.encode(input, forKey: .input)
        try container.encodeIfPresent(encoding_format, forKey: .encoding_format)
        try container.encodeIfPresent(dimensions, forKey: .dimensions)
        try container.encodeIfPresent(user, forKey: .user)

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
    public var reasoning_content: String?
    public var name: String?
    public var tool_call_id: String?
    public var tool_calls: [OpenAIChatToolCall]?
    public var imageURLs: [String]
    public var audioURLs: [String]
    public var videoURLs: [String]

    public init(
        role: String,
        content: String = "",
        reasoning_content: String? = nil,
        name: String? = nil,
        tool_call_id: String? = nil,
        tool_calls: [OpenAIChatToolCall]? = nil,
        imageURLs: [String] = [],
        audioURLs: [String] = [],
        videoURLs: [String] = []
    ) {
        self.role = role
        self.content = content
        self.reasoning_content = reasoning_content
        self.name = name
        self.tool_call_id = tool_call_id
        self.tool_calls = tool_calls
        self.imageURLs = imageURLs
        self.audioURLs = audioURLs
        self.videoURLs = videoURLs
    }

    private enum CodingKeys: String, CodingKey {
        case role
        case content
        case reasoning_content
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
        audioURLs = decoded.audioURLs
        videoURLs = decoded.videoURLs
        reasoning_content = try container.decodeIfPresent(String.self, forKey: .reasoning_content)
        name = try container.decodeIfPresent(String.self, forKey: .name)
        tool_call_id = try container.decodeIfPresent(String.self, forKey: .tool_call_id)
        tool_calls = try container.decodeIfPresent([OpenAIChatToolCall].self, forKey: .tool_calls)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        if imageURLs.isEmpty, audioURLs.isEmpty, videoURLs.isEmpty {
            try container.encode(content, forKey: .content)
        } else {
            var parts: [OpenAIChatContentPart] = []
            if !content.isEmpty {
                parts.append(OpenAIChatContentPart(type: "text", text: content))
            }
            parts += imageURLs.map {
                OpenAIChatContentPart(type: "image_url", image_url: OpenAIImageURL(url: $0))
            }
            parts += audioURLs.map {
                OpenAIChatContentPart(type: "audio_url", audio_url: OpenAIImageURL(url: $0))
            }
            parts += videoURLs.map {
                OpenAIChatContentPart(type: "video_url", video_url: OpenAIImageURL(url: $0))
            }
            try container.encode(parts, forKey: .content)
        }
        try container.encodeIfPresent(reasoning_content, forKey: .reasoning_content)
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
    var audioURLs: [String]
    var videoURLs: [String]
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
    public var audio_url: OpenAIImageURL?
    public var video_url: OpenAIImageURL?

    public init(
        type: String? = nil,
        text: String? = nil,
        image_url: OpenAIImageURL? = nil,
        input_image: OpenAIImageURL? = nil,
        audio_url: OpenAIImageURL? = nil,
        video_url: OpenAIImageURL? = nil
    ) {
        self.type = type
        self.text = text
        self.image_url = image_url
        self.input_image = input_image
        self.audio_url = audio_url
        self.video_url = video_url
    }

    var extractedImageURL: String? {
        image_url?.url ?? input_image?.url
    }

    var extractedAudioURL: String? { audio_url?.url }
    var extractedVideoURL: String? { video_url?.url }
}

private extension KeyedDecodingContainer {
    func decodeFlexibleContentIfPresent(forKey key: Key) throws -> OpenAIChatContent {
        guard contains(key) else {
            return OpenAIChatContent(text: "", imageURLs: [], audioURLs: [], videoURLs: [])
        }
        if try decodeNil(forKey: key) {
            return OpenAIChatContent(text: "", imageURLs: [], audioURLs: [], videoURLs: [])
        }
        if let content = try? decode(String.self, forKey: key) {
            return OpenAIChatContent(text: content, imageURLs: [], audioURLs: [], videoURLs: [])
        }
        if let parts = try? decode([OpenAIChatContentPart].self, forKey: key) {
            let text = parts
                .filter { $0.type == nil || $0.type == "text" || $0.type == "input_text" }
                .compactMap(\.text)
                .joined(separator: "\n")
            let imageURLs = parts.compactMap(\.extractedImageURL)
            let audioURLs = parts.compactMap(\.extractedAudioURL)
            let videoURLs = parts.compactMap(\.extractedVideoURL)
            return OpenAIChatContent(
                text: text,
                imageURLs: imageURLs,
                audioURLs: audioURLs,
                videoURLs: videoURLs
            )
        }
        throw DecodingError.typeMismatch(
            String.self,
            DecodingError.Context(
                codingPath: codingPath + [key],
                debugDescription: "Expected chat message content to be a string, null, or supported text/media content parts."
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
    public var logprobs: OpenAIChatLogprobs?

    public init(
        index: Int,
        message: OpenAIChatMessage? = nil,
        delta: OpenAIChatDelta? = nil,
        finish_reason: String? = nil,
        logprobs: OpenAIChatLogprobs? = nil
    ) {
        self.index = index
        self.message = message
        self.delta = delta
        self.finish_reason = finish_reason
        self.logprobs = logprobs
    }
}

public struct OpenAIChatLogprobs: Codable, Sendable {
    public var content: [OpenAIChatTokenLogprob]
    public var mere_summary: ChatLogprobSummary?
    public var mere_capture_seconds: Double?
    public var mere_source: String?

    public init?(_ diagnostics: ChatLogprobDiagnostics?) {
        guard let diagnostics else { return nil }
        self.content = (diagnostics.tokens ?? []).compactMap(OpenAIChatTokenLogprob.init)
        self.mere_summary = diagnostics.summary
        self.mere_capture_seconds = diagnostics.captureSeconds
        self.mere_source = diagnostics.source.rawValue
    }
}

public struct OpenAIChatTokenLogprob: Codable, Sendable {
    public var token: String
    public var bytes: [Int]
    /// OpenAI-compatible intrinsic model log probability.
    public var logprob: Double
    public var top_logprobs: [OpenAIChatTopLogprob]
    public var raw_logprob: Double
    public var policy_logprob: Double
    public var raw_entropy: Double
    public var policy_entropy: Double
    public var raw_top1_top2_margin: Double
    public var policy_top1_top2_margin: Double
    public var region: String

    public init?(_ measurement: ChatTokenLogprob) {
        guard let token = measurement.token else { return nil }
        self.token = token
        self.bytes = token.utf8.map(Int.init)
        self.logprob = measurement.rawLogprob
        self.top_logprobs = measurement.topLogprobs.compactMap(OpenAIChatTopLogprob.init)
        self.raw_logprob = measurement.rawLogprob
        self.policy_logprob = measurement.policyLogprob
        self.raw_entropy = measurement.rawEntropy
        self.policy_entropy = measurement.policyEntropy
        self.raw_top1_top2_margin = measurement.rawTop1Top2Margin
        self.policy_top1_top2_margin = measurement.policyTop1Top2Margin
        self.region = measurement.region.rawValue
    }
}

public struct OpenAIChatTopLogprob: Codable, Sendable {
    public var token: String
    public var bytes: [Int]
    public var logprob: Double
    public var raw_logprob: Double
    public var policy_logprob: Double

    public init?(_ candidate: ChatTopLogprob) {
        guard let token = candidate.token else { return nil }
        self.token = token
        self.bytes = token.utf8.map(Int.init)
        self.logprob = candidate.rawLogprob
        self.raw_logprob = candidate.rawLogprob
        self.policy_logprob = candidate.policyLogprob
    }
}

public struct OpenAIChatDelta: Codable, Sendable {
    public var role: String?
    public var content: String?
    public var reasoning_content: String?
    public var tool_calls: [OpenAIChatToolCallDelta]?

    public init(
        role: String? = nil,
        content: String? = nil,
        reasoning_content: String? = nil,
        tool_calls: [OpenAIChatToolCallDelta]? = nil
    ) {
        self.role = role
        self.content = content
        self.reasoning_content = reasoning_content
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

public struct OpenAIEmbeddingResponse: Codable, Sendable {
    public var object: String
    public var model: String
    public var data: [OpenAIEmbeddingDatum]
    public var usage: OpenAIEmbeddingUsage

    public init(
        model: String,
        data: [OpenAIEmbeddingDatum],
        usage: OpenAIEmbeddingUsage,
        object: String = "list"
    ) {
        self.object = object
        self.model = model
        self.data = data
        self.usage = usage
    }
}

public struct OpenAIEmbeddingDatum: Codable, Sendable {
    public var object: String
    public var index: Int
    public var embedding: [Float]

    public init(index: Int, embedding: [Float], object: String = "embedding") {
        self.object = object
        self.index = index
        self.embedding = embedding
    }
}

public struct OpenAIEmbeddingUsage: Codable, Sendable {
    public var prompt_tokens: Int
    public var total_tokens: Int

    public init(prompt_tokens: Int, total_tokens: Int) {
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens
    }
}

// MARK: - Images Endpoint

public struct OpenAIImageGenerationRequest: Codable, Sendable {
    public var prompt: String
    public var model: String?
    public var n: Int?
    public var size: String?
    public var response_format: String?
    public var quality: String?
    public var style: String?
    public var user: String?
    public var seed: UInt64?
    public var negative_prompt: String?
    public var steps: Int?
    public var guidance_scale: Double?

    public init(
        prompt: String,
        model: String? = nil,
        n: Int? = nil,
        size: String? = nil,
        response_format: String? = nil,
        quality: String? = nil,
        style: String? = nil,
        user: String? = nil,
        seed: UInt64? = nil,
        negative_prompt: String? = nil,
        steps: Int? = nil,
        guidance_scale: Double? = nil
    ) {
        self.prompt = prompt
        self.model = model
        self.n = n
        self.size = size
        self.response_format = response_format
        self.quality = quality
        self.style = style
        self.user = user
        self.seed = seed
        self.negative_prompt = negative_prompt
        self.steps = steps
        self.guidance_scale = guidance_scale
    }
}

public struct OpenAIImageGenerationResponse: Codable, Sendable {
    public var created: Int
    public var data: [OpenAIImageGenerationData]

    public init(created: Int, data: [OpenAIImageGenerationData]) {
        self.created = created
        self.data = data
    }
}

public struct OpenAIImageGenerationData: Codable, Sendable {
    public var url: String?
    public var b64_json: String?
    public var revised_prompt: String?

    public init(url: String? = nil, b64_json: String? = nil, revised_prompt: String? = nil) {
        self.url = url
        self.b64_json = b64_json
        self.revised_prompt = revised_prompt
    }
}

// MARK: - Videos Endpoint

/// Local-first video generation request. The typed fields cover the common
/// OpenAI-style surface; `options` carries additional `mere.run video generate`
/// flags so advanced native pipelines remain reachable without a second API.
public struct OpenAIVideoGenerationRequest: Codable, Sendable {
    public var prompt: String
    public var model: String?
    public var size: String?
    public var seconds: Double?
    public var num_frames: Int?
    public var fps: Int?
    public var seed: Int?
    public var quality: String?
    public var output_mode: String?
    public var options: [String]?

    public init(
        prompt: String,
        model: String? = nil,
        size: String? = nil,
        seconds: Double? = nil,
        num_frames: Int? = nil,
        fps: Int? = nil,
        seed: Int? = nil,
        quality: String? = nil,
        output_mode: String? = nil,
        options: [String]? = nil
    ) {
        self.prompt = prompt
        self.model = model
        self.size = size
        self.seconds = seconds
        self.num_frames = num_frames
        self.fps = fps
        self.seed = seed
        self.quality = quality
        self.output_mode = output_mode
        self.options = options
    }
}

public struct OpenAIVideoGenerationArtifact: Codable, Sendable {
    public var url: String
    public var media_type: String
    public var byte_count: Int64
    public var sha256: String

    public init(url: String, media_type: String, byte_count: Int64, sha256: String) {
        self.url = url
        self.media_type = media_type
        self.byte_count = byte_count
        self.sha256 = sha256
    }
}

public struct OpenAIVideoGenerationResponse: Codable, Sendable {
    public var created: Int
    public var object: String
    public var status: String
    public var model: String
    public var artifact: OpenAIVideoGenerationArtifact
    public var exr_directory_url: String?

    public init(
        created: Int,
        object: String = "video.generation",
        status: String = "completed",
        model: String,
        artifact: OpenAIVideoGenerationArtifact,
        exr_directory_url: String? = nil
    ) {
        self.created = created
        self.object = object
        self.status = status
        self.model = model
        self.artifact = artifact
        self.exr_directory_url = exr_directory_url
    }
}

// MARK: - Audio Endpoint

public struct OpenAIAudioSpeechRequest: Codable, Sendable {
    public var model: String?
    public var input: String
    public var voice: String?
    public var response_format: String?
    public var speed: Double?
    public var instructions: String?
    public var temperature: Float?

    public init(
        model: String? = nil,
        input: String,
        voice: String? = nil,
        response_format: String? = nil,
        speed: Double? = nil,
        instructions: String? = nil,
        temperature: Float? = nil
    ) {
        self.model = model
        self.input = input
        self.voice = voice
        self.response_format = response_format
        self.speed = speed
        self.instructions = instructions
        self.temperature = temperature
    }
}

public struct OpenAIAudioTranscriptionResponse: Codable, Sendable {
    public var text: String
    public var language: String?
    public var duration: Double?
    public var segments: [OpenAIAudioTranscriptionSegment]?

    public init(
        text: String,
        language: String? = nil,
        duration: Double? = nil,
        segments: [OpenAIAudioTranscriptionSegment]? = nil
    ) {
        self.text = text
        self.language = language
        self.duration = duration
        self.segments = segments
    }
}

public struct OpenAIAudioTranscriptionSegment: Codable, Sendable {
    public var id: Int
    public var start: Double
    public var end: Double
    public var text: String

    public init(id: Int, start: Double, end: Double, text: String) {
        self.id = id
        self.start = start
        self.end = end
        self.text = text
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
    public var name: String?
    public var task: String?
    public var reasoning: Bool?
    public var thinking_levels: [String]?
    public var tool_call: Bool?
    public var structured_output: Bool?
    public var modalities: OpenAIModelModalities?
    public var limit: OpenAIModelLimit?
    public var openai_compat: OpenAIModelCompatibility?

    public init(
        id: String,
        object: String,
        created: Int,
        owned_by: String,
        name: String? = nil,
        task: String? = nil,
        reasoning: Bool? = nil,
        thinking_levels: [String]? = nil,
        tool_call: Bool? = nil,
        structured_output: Bool? = nil,
        modalities: OpenAIModelModalities? = nil,
        limit: OpenAIModelLimit? = nil,
        openai_compat: OpenAIModelCompatibility? = nil
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.owned_by = owned_by
        self.name = name
        self.task = task
        self.reasoning = reasoning
        self.thinking_levels = thinking_levels
        self.tool_call = tool_call
        self.structured_output = structured_output
        self.modalities = modalities
        self.limit = limit
        self.openai_compat = openai_compat
    }
}

public struct OpenAIModelModalities: Codable, Equatable, Sendable {
    public var input: [String]
    public var output: [String]

    public init(input: [String], output: [String]) {
        self.input = input
        self.output = output
    }
}

public struct OpenAIModelLimit: Codable, Equatable, Sendable {
    public var context: Int
    public var output: Int

    public init(context: Int, output: Int) {
        self.context = context
        self.output = output
    }
}

public struct OpenAIModelCompatibility: Codable, Equatable, Sendable {
    public var supports_store: Bool
    public var supports_developer_role: Bool
    public var supports_reasoning_effort: Bool
    public var supports_usage_in_streaming: Bool
    public var supports_finish_reason: Bool
    public var max_tokens_field: String
    public var supports_strict_mode: Bool
    public var thinking_format: String?
    public var thinking_level_map: [String: String]?
    public var requires_reasoning_content_on_assistant_messages: Bool

    public init(
        supports_store: Bool,
        supports_developer_role: Bool,
        supports_reasoning_effort: Bool,
        supports_usage_in_streaming: Bool,
        supports_finish_reason: Bool,
        max_tokens_field: String,
        supports_strict_mode: Bool,
        thinking_format: String? = nil,
        thinking_level_map: [String: String]? = nil,
        requires_reasoning_content_on_assistant_messages: Bool = false
    ) {
        self.supports_store = supports_store
        self.supports_developer_role = supports_developer_role
        self.supports_reasoning_effort = supports_reasoning_effort
        self.supports_usage_in_streaming = supports_usage_in_streaming
        self.supports_finish_reason = supports_finish_reason
        self.max_tokens_field = max_tokens_field
        self.supports_strict_mode = supports_strict_mode
        self.thinking_format = thinking_format
        self.thinking_level_map = thinking_level_map
        self.requires_reasoning_content_on_assistant_messages = requires_reasoning_content_on_assistant_messages
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
