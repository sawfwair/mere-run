import Foundation

// MARK: - Request Types

public struct OpenAIChatRequest: Codable, Sendable {
    public var model: String
    public var messages: [OpenAIChatMessage]
    public var temperature: Double?
    public var top_p: Double?
    public var max_tokens: Int?
    public var stream: Bool?
    public var lora: String?

    public init(
        model: String,
        messages: [OpenAIChatMessage],
        temperature: Double? = nil,
        top_p: Double? = nil,
        max_tokens: Int? = nil,
        stream: Bool? = nil,
        lora: String? = nil
    ) {
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        self.lora = lora
    }
}

public struct OpenAIChatMessage: Codable, Sendable {
    public var role: String
    public var content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
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

    public init(role: String? = nil, content: String? = nil) {
        self.role = role
        self.content = content
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
