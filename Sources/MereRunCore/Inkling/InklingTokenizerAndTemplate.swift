import Foundation
@preconcurrency import Hub
@preconcurrency import Tokenizers

public final class InklingTokenizerAndTemplate: @unchecked Sendable {
    public let tokenizer: any Tokenizer
    public let maxLength: Int
    public let endSamplingTokenID: Int?

    public init(tokenizer: any Tokenizer, maxLength: Int, endSamplingTokenID: Int?) {
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.endSamplingTokenID = endSamplingTokenID
    }

    public static func load(
        from rootURL: URL,
        maxLengthOverride: Int? = nil,
        hubApi: HubApi = .shared
    ) async throws -> InklingTokenizerAndTemplate {
        let tokenizer = try await AutoTokenizer.from(modelFolder: rootURL, hubApi: hubApi)
        return InklingTokenizerAndTemplate(
            tokenizer: tokenizer,
            maxLength: maxLengthOverride ?? InklingResources.defaultContextLength,
            endSamplingTokenID: tokenizer.convertTokenToId("<|content_model_end_sampling|>")
        )
    }

    public func encodeForGeneration(
        messages: [ChatMessage],
        tools: [ToolDefinition]? = nil,
        addGenerationPrompt: Bool = true,
        reasoningEffort: Double = 0.9,
        maxLength: Int
    ) throws -> [Int] {
        var encoded = tokenizer.encode(
            text: try Self.renderPrompt(
                messages: messages,
                tools: tools,
                addGenerationPrompt: addGenerationPrompt,
                reasoningEffort: reasoningEffort
            ),
            addSpecialTokens: false
        )
        let targetLength = min(maxLength, self.maxLength)
        if encoded.count > targetLength {
            encoded = Array(encoded.suffix(targetLength))
        }
        return encoded
    }

    public func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    public func encodeRaw(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(token: Int) -> String {
        tokenizer.decode(tokens: [token])
    }

    public var stopTokenIDs: [Int] {
        [endSamplingTokenID].compactMap { $0 }
    }

    public static func renderPrompt(
        messages: [ChatMessage],
        tools: [ToolDefinition]? = nil,
        addGenerationPrompt: Bool = true,
        reasoningEffort: Double = 0.9
    ) throws -> String {
        guard (0...0.99).contains(reasoningEffort) else {
            throw InklingError.generationFailed("Inkling reasoning effort must be between 0 and 0.99.")
        }
        guard messages.allSatisfy({ $0.imageUrl == nil }) else {
            throw InklingError.generationFailed(
                "The initial Inkling-Small runtime is text-only; image and audio towers are not loaded."
            )
        }

        var prompt = ""
        if let tools, !tools.isEmpty {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let declarations = tools.map(InklingToolDeclaration.init)
            let json = String(decoding: try encoder.encode(declarations), as: UTF8.self)
            prompt += "<|message_system|>tool_declare<|content_xml|>\(json)<|end_message|>"
        }

        var effortEmitted = false
        var toolNamesByID: [String: String] = [:]
        for message in messages {
            if !effortEmitted, message.role != .system {
                prompt += effortMessage(reasoningEffort)
                effortEmitted = true
            }
            if let calls = message.toolCalls {
                for call in calls {
                    if let id = call.id {
                        toolNamesByID[id] = call.name
                    }
                }
            }
            prompt += try render(message, toolNamesByID: toolNamesByID)
        }
        if !effortEmitted {
            prompt += effortMessage(reasoningEffort)
        }
        if addGenerationPrompt {
            prompt += "<|message_model|>"
        }
        return prompt
    }

    private static func effortMessage(_ value: Double) -> String {
        let rendered = value == 0 ? "0" : String(value)
        return "<|message_system|><|content_text|>Thinking effort level: \(rendered)<|end_message|>"
    }

    private static func render(
        _ message: ChatMessage,
        toolNamesByID: [String: String]
    ) throws -> String {
        let roleToken: String
        switch message.role {
        case .system:
            roleToken = "<|message_system|>"
        case .user:
            roleToken = "<|message_user|>"
        case .assistant:
            roleToken = "<|message_model|>"
        case .tool:
            roleToken = "<|message_tool|>"
        }

        if message.role == .tool {
            let toolName = message.name
                ?? message.toolCallID.flatMap { toolNamesByID[$0] }
                ?? ""
            return roleToken
                + toolName
                + "<|content_text|>"
                + message.content
                + "<|end_message|>"
        }

        var result = ""
        if message.role == .assistant,
           let reasoning = message.reasoningContent,
           !reasoning.isEmpty {
            result += "<|message_model|><|content_thinking|>\(reasoning)<|end_message|>"
        }
        result += roleToken + "<|content_text|>" + message.content + "<|end_message|>"
        if message.role == .assistant, let calls = message.toolCalls {
            for call in calls {
                let invocation = InklingToolInvocation(name: call.name, args: call.arguments)
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.sortedKeys]
                let json = String(decoding: try encoder.encode(invocation), as: UTF8.self)
                result += "<|message_model|>\(call.name)<|content_invoke_tool_json|>\(json)<|end_message|>"
            }
        }
        if message.role == .assistant {
            result += "<|content_model_end_sampling|>"
        }
        return result
    }
}

private struct InklingToolDeclaration: Encodable {
    let description: String
    let name: String
    let parameters: InklingToolParameters
    let type = "function"

    init(_ definition: ToolDefinition) {
        description = definition.description
        name = definition.name
        parameters = InklingToolParameters(definition)
    }
}

private struct InklingToolParameters: Encodable {
    let properties: [String: ToolParameterProperty]
    let required: [String]
    let type = "object"

    init(_ definition: ToolDefinition) {
        properties = definition.parameters
        required = definition.required
    }
}

private struct InklingToolInvocation: Codable {
    let name: String
    let args: [String: OpenAIJSONValue]
}

struct InklingParsedOutput: Sendable, Hashable {
    let visible: String
    let reasoning: String?
    let toolCalls: [ToolCall]
}

enum InklingOutputParser {
    static func parse(_ text: String) -> InklingParsedOutput {
        let reasoning = channelContents("<|content_thinking|>", in: text)
        var visible = channelContents("<|content_text|>", in: text)
        if visible.isEmpty, !text.contains("<|content_thinking|>"), !text.contains("<|content_invoke_tool_json|>") {
            visible = [stripControlTokens(text)]
        }
        return InklingParsedOutput(
            visible: visible.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines),
            reasoning: reasoning.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty,
            toolCalls: parseToolCalls(text)
        )
    }

    private static func channelContents(_ marker: String, in text: String) -> [String] {
        var result: [String] = []
        var searchStart = text.startIndex
        while let markerRange = text.range(of: marker, range: searchStart..<text.endIndex) {
            let contentStart = markerRange.upperBound
            guard let end = text.range(of: "<|end_message|>", range: contentStart..<text.endIndex) else {
                result.append(String(text[contentStart...]))
                break
            }
            result.append(String(text[contentStart..<end.lowerBound]))
            searchStart = end.upperBound
        }
        return result
    }

    private static func stripControlTokens(_ text: String) -> String {
        text.replacingOccurrences(
            of: "<\\|[^>]+\\|>",
            with: "",
            options: .regularExpression
        )
    }

    private static func parseToolCalls(_ text: String) -> [ToolCall] {
        let marker = "<|content_invoke_tool_json|>"
        var result: [ToolCall] = []
        var searchStart = text.startIndex
        let decoder = JSONDecoder()
        while let markerRange = text.range(of: marker, range: searchStart..<text.endIndex) {
            let jsonStart = markerRange.upperBound
            guard let end = text.range(of: "<|end_message|>", range: jsonStart..<text.endIndex) else {
                break
            }
            let data = Data(text[jsonStart..<end.lowerBound].utf8)
            if let invocation = try? decoder.decode(InklingToolInvocation.self, from: data) {
                result.append(ToolCall(
                    name: invocation.name,
                    arguments: invocation.args.mapValues(stringValue)
                ))
            }
            searchStart = end.upperBound
        }
        return result
    }

    private static func stringValue(_ value: OpenAIJSONValue) -> String {
        switch value {
        case .string(let value):
            return value
        default:
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            return (try? String(decoding: encoder.encode(value), as: UTF8.self)) ?? "null"
        }
    }
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}
