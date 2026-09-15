import Foundation
import MereRunCore

extension APIServerContract {
    static let defaultMaxTokens = 2048

    static func chatRequest(
        from openaiRequest: OpenAIChatRequest,
        fallbackLoraPath: String?,
        contextSize: Int,
        capabilities: APIEngineCapabilities = .localText,
        servedModelID: String? = nil,
        apiProfile: ManagedModelAPIProfile? = nil
    ) throws -> ChatRequest {
        guard !openaiRequest.messages.isEmpty else {
            throw APIRequestValidationError.invalidField("messages", "must contain at least one message")
        }

        try validateTopLevelOptions(openaiRequest, capabilities: capabilities)

        if let requestLora = openaiRequest.lora?.trimmingCharacters(in: .whitespacesAndNewlines),
           !requestLora.isEmpty {
            throw APIRequestValidationError.invalidField(
                "lora",
                "per-request LoRA paths are not supported; start the server with --lora instead"
            )
        }

        let maxTokens = try resolveMaxTokens(
            maxTokens: openaiRequest.max_tokens,
            maxCompletionTokens: openaiRequest.max_completion_tokens,
            capabilities: capabilities
        )
        let tools = try toolDefinitions(from: openaiRequest, capabilities: capabilities)
        let toolChoice = try chatToolChoice(from: openaiRequest, capabilities: capabilities)
        let parallelToolCalls = openaiRequest.parallel_tool_calls ?? true
        let requiresJSON = try requiresJSONResponseFormat(
            openaiRequest.response_format,
            capabilities: capabilities
        )

        var messages = try openaiRequest.messages.map { msg in
            try chatMessage(from: msg, capabilities: capabilities)
        }
        if let instruction = toolChoiceInstruction(
            choice: toolChoice,
            parallelToolCalls: parallelToolCalls,
            tools: tools
        ) {
            if let systemIndex = messages.firstIndex(where: { $0.role == .system }) {
                messages[systemIndex].content = [messages[systemIndex].content, instruction]
                    .filter { !$0.isEmpty }
                    .joined(separator: "\n\n")
            } else {
                messages.insert(ChatMessage(role: .system, content: instruction), at: 0)
            }
        }

        let lora: LoRA?
        if let loraPath = fallbackLoraPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !loraPath.isEmpty {
            lora = LoRA.local(path: loraPath, scale: 1.0)
        } else {
            lora = nil
        }

        // Resolve each omitted sampling field independently.
        let laneModelID = servedModelID ?? ""
        let resolvedAPIProfile = apiProfile
            ?? servedModelID.flatMap { ManagedModelCatalog.apiProfile(for: $0) }
        let reasoningEffort = try reasoningEffort(
            from: openaiRequest.reasoning_effort,
            capabilities: capabilities,
            profile: resolvedAPIProfile
        )
        let logprobCapture: ChatLogprobCapture
        if openaiRequest.logprobs == true {
            if let topLogprobs = openaiRequest.top_logprobs, topLogprobs > 0 {
                logprobCapture = .top(topLogprobs)
            } else {
                logprobCapture = .tokens
            }
        } else {
            logprobCapture = .none
        }
        if openaiRequest.mere_show_unmasking == true {
            guard laneModelID == DiffusionGemmaResources.modelID else {
                throw APIRequestValidationError.invalidField(
                    "mere_show_unmasking",
                    "progressive unmasking is supported only by \(DiffusionGemmaResources.modelID)"
                )
            }
            guard openaiRequest.stream == true else {
                throw APIRequestValidationError.invalidField(
                    "mere_show_unmasking",
                    "requires stream=true"
                )
            }
        }

        let request = ChatRequest(
            messages: messages,
            maxTokens: maxTokens,
            presencePenalty: openaiRequest.presence_penalty ?? 0,
            frequencyPenalty: openaiRequest.frequency_penalty ?? 0,
            repetitionPenalty: openaiRequest.repetition_penalty ?? 1,
            seed: openaiRequest.seed.map(UInt64.init),
            reasoningEffort: reasoningEffort,
            lora: lora,
            requiresJSON: requiresJSON,
            tools: tools,
            toolChoice: toolChoice,
            parallelToolCalls: parallelToolCalls,
            stopSequences: openaiRequest.stop?.values ?? [],
            maxContextTokens: contextSize,
            logprobCapture: logprobCapture,
            showUnmasking: openaiRequest.mere_show_unmasking == true
        )
        do {
            let resolved = try ChatRequestResolver.resolve(
                request, modelID: laneModelID,
                sampling: ChatSamplingOptions(
                    temperature: openaiRequest.temperature, topP: openaiRequest.top_p,
                    topK: openaiRequest.top_k, minP: openaiRequest.min_p
                ), policy: .openAI, apiProfile: resolvedAPIProfile
            )
            try validateSamplingCapabilities(openaiRequest, capabilities: capabilities)
            return resolved
        } catch let issue as ChatRequestIssue {
            throw APIRequestValidationError.invalidField(issue.field, issue.message)
        }
    }


    static func includeUsageInStreaming(
        _ openaiRequest: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws -> Bool {
        guard openaiRequest.stream_options?.include_usage == true else {
            return false
        }
        guard capabilities.supportsUsageInStreaming else {
            throw APIRequestValidationError.invalidField(
                "stream_options.include_usage",
                "this engine cannot emit usage chunks while streaming"
            )
        }
        return true
    }

    private static func chatMessage(
        from msg: OpenAIChatMessage,
        capabilities: APIEngineCapabilities
    ) throws -> ChatMessage {
        let normalizedRole = msg.role.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let role: ChatMessage.Role
        switch normalizedRole {
        case "system":
            role = .system
        case "developer":
            guard capabilities.supportsDeveloperRole else {
                throw APIRequestValidationError.invalidField(
                    "messages.role",
                    "developer messages are not supported by this engine"
                )
            }
            role = .system
        case "user":
            role = .user
        case "assistant":
            role = .assistant
        case "tool":
            role = .tool
        default:
            throw APIRequestValidationError.invalidField(
                "messages.role",
                "unsupported role '\(msg.role)'"
            )
        }

        let imageURL = try firstImageURL(from: msg, capabilities: capabilities)
        let audioURL = try firstAudioURL(from: msg, capabilities: capabilities)
        let videoURL = try firstVideoURL(from: msg, capabilities: capabilities)
        let content = capabilities.usesNativeToolHistory ? msg.content : renderMessageContent(msg)
        let toolCalls = try chatMessageToolCalls(from: msg)
        return ChatMessage(
            role: role,
            content: content,
            imageUrl: imageURL,
            audioUrl: audioURL,
            videoUrl: videoURL,
            reasoningContent: msg.reasoning_content,
            name: msg.name,
            toolCallID: msg.tool_call_id,
            toolCalls: toolCalls
        )
    }

    private static func chatMessageToolCalls(
        from message: OpenAIChatMessage
    ) throws -> [ChatMessageToolCall]? {
        guard message.role.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "assistant",
              let openAIToolCalls = message.tool_calls,
              !openAIToolCalls.isEmpty else {
            return nil
        }

        return try openAIToolCalls.compactMap { toolCall in
            guard toolCall.type == "function", let function = toolCall.function else {
                return nil
            }
            guard let data = function.arguments.data(using: .utf8) else {
                throw APIRequestValidationError.invalidField(
                    "messages.tool_calls.function.arguments",
                    "must be a UTF-8 JSON object"
                )
            }
            let arguments: [String: OpenAIJSONValue]
            do {
                arguments = try JSONDecoder().decode([String: OpenAIJSONValue].self, from: data)
            } catch {
                throw APIRequestValidationError.invalidField(
                    "messages.tool_calls.function.arguments",
                    "must be a JSON object"
                )
            }
            return ChatMessageToolCall(
                id: toolCall.id,
                name: function.name,
                arguments: arguments
            )
        }
    }

    private static func firstImageURL(
        from msg: OpenAIChatMessage,
        capabilities: APIEngineCapabilities
    ) throws -> String? {
        guard !msg.imageURLs.isEmpty else { return nil }
        guard capabilities.supportsVisionContentParts else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "image content parts are not supported by this engine"
            )
        }
        guard msg.imageURLs.count == 1 else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "only one image content part is currently supported"
            )
        }
        return msg.imageURLs.first
    }

    private static func firstAudioURL(
        from msg: OpenAIChatMessage,
        capabilities: APIEngineCapabilities
    ) throws -> String? {
        guard !msg.audioURLs.isEmpty else { return nil }
        guard capabilities.supportsAudioContentParts else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "audio content parts are not supported by this engine"
            )
        }
        guard msg.audioURLs.count == 1 else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "only one audio content part per message is currently supported"
            )
        }
        return msg.audioURLs.first
    }

    private static func firstVideoURL(
        from msg: OpenAIChatMessage,
        capabilities: APIEngineCapabilities
    ) throws -> String? {
        guard !msg.videoURLs.isEmpty else { return nil }
        guard capabilities.supportsVideoContentParts else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "video content parts are not supported by this engine"
            )
        }
        guard msg.videoURLs.count == 1 else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "only one video content part per message is currently supported"
            )
        }
        return msg.videoURLs.first
    }

    private static func renderMessageContent(_ msg: OpenAIChatMessage) -> String {
        guard msg.role.lowercased() == "assistant",
              let toolCalls = msg.tool_calls,
              !toolCalls.isEmpty else {
            return msg.content
        }
        let renderedCalls = toolCalls.compactMap { call -> String? in
            guard call.type == "function", let function = call.function else { return nil }
            return "<|tool_call>call:\(function.name)\(function.arguments)<tool_call|>"
        }
        guard !renderedCalls.isEmpty else {
            return msg.content
        }
        return ([msg.content] + renderedCalls)
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    private static func validateTopLevelOptions(
        _ request: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws {
        if let n = request.n, n != 1 {
            throw APIRequestValidationError.invalidField("n", "only n=1 is supported")
        }
        if request.store == true {
            throw APIRequestValidationError.invalidField("store", "stored chat completions are not supported")
        }
        if let modalities = request.modalities,
           modalities.contains(where: { $0 != "text" }) {
            throw APIRequestValidationError.invalidField("modalities", "only text output is supported")
        }
        if request.audio != nil {
            throw APIRequestValidationError.invalidField("audio", "audio output is not supported by /v1/chat/completions")
        }
        if request.prediction != nil {
            throw APIRequestValidationError.invalidField("prediction", "predicted outputs are not supported")
        }
        if let stop = request.stop, !stop.values.isEmpty, !capabilities.supportsStopSequences {
            throw APIRequestValidationError.invalidField("stop", "stop sequences are not supported by this engine")
        }
        if request.seed != nil, !capabilities.supportsSeed {
            throw APIRequestValidationError.invalidField("seed", "deterministic seeds are not supported by this engine")
        }
        if let seed = request.seed, seed < 0 {
            throw APIRequestValidationError.invalidField("seed", "must be an unsigned integer")
        }
        if request.logprobs == true, !capabilities.supportsLogprobs {
            throw APIRequestValidationError.invalidField("logprobs", "token log probabilities are not supported by this engine")
        }
        if request.stream == true, request.logprobs == true {
            throw APIRequestValidationError.invalidField(
                "logprobs",
                "native token log probabilities currently require stream=false"
            )
        }
        if request.logprobs == true,
           let responseType = request.response_format?.type,
           responseType != "text" {
            throw APIRequestValidationError.invalidField(
                "logprobs",
                "native token log probabilities currently require an unconstrained text response"
            )
        }
        if request.top_logprobs != nil, !capabilities.supportsLogprobs {
            throw APIRequestValidationError.invalidField("top_logprobs", "token log probabilities are not supported by this engine")
        }
        if let topLogprobs = request.top_logprobs {
            guard request.logprobs == true else {
                throw APIRequestValidationError.invalidField(
                    "top_logprobs",
                    "requires logprobs=true"
                )
            }
            guard (0...20).contains(topLogprobs) else {
                throw APIRequestValidationError.invalidField(
                    "top_logprobs",
                    "must be between 0 and 20"
                )
            }
        }
        if request.reasoning_effort != nil, !capabilities.supportsReasoningEffort {
            throw APIRequestValidationError.invalidField("reasoning_effort", "reasoning effort is not supported by this engine")
        }
        if request.think != nil || request.thinking != nil {
            guard capabilities.supportsProviderThinkingControls else {
                throw APIRequestValidationError.invalidField(
                    "thinking",
                    "provider thinking controls are not supported by this engine"
                )
            }
        }
    }

    private static func toolDefinitions(
        from request: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws -> [ToolDefinition]? {
        guard let tools = request.tools, !tools.isEmpty else {
            switch request.tool_choice {
            case .mode("required")?, .function(_)?:
                throw APIRequestValidationError.invalidField(
                    "tool_choice",
                    "requires at least one function in tools"
                )
            default:
                return nil
            }
        }
        guard capabilities.supportsTools else {
            throw APIRequestValidationError.invalidField("tools", "tools are not supported by this engine")
        }

        let convertedTools = try tools.map { try toolDefinition(from: $0) }
        switch request.tool_choice {
        case nil, .mode("auto")?, .mode("required")?:
            return convertedTools
        case .mode("none")?:
            return nil
        case .function(let name)?:
            guard let selected = convertedTools.first(where: { $0.name == name }) else {
                throw APIRequestValidationError.invalidField(
                    "tool_choice",
                    "requested tool '\(name)' is not present in tools"
                )
            }
            return [selected]
        case .custom?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported object shape")
        case .mode(let value)?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported mode '\(value)'")
        }
    }

    private static func chatToolChoice(
        from request: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws -> ChatToolChoice {
        guard request.tool_choice != nil else { return .auto }
        guard capabilities.supportsToolChoice else {
            throw APIRequestValidationError.invalidField(
                "tool_choice",
                "tool choice is not supported by this engine"
            )
        }
        switch request.tool_choice {
        case nil, .mode("auto")?, .mode("none")?:
            return .auto
        case .mode("required")?:
            return .required
        case .function(let name)?:
            return .function(name)
        case .custom?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported object shape")
        case .mode(let value)?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported mode '\(value)'")
        }
    }

    private static func toolChoiceInstruction(
        choice: ChatToolChoice,
        parallelToolCalls: Bool,
        tools: [ToolDefinition]?
    ) -> String? {
        guard tools?.isEmpty == false else { return nil }
        switch choice {
        case .auto where !parallelToolCalls:
            return "If you call a function, call at most one provided function."
        case .required where parallelToolCalls:
            return "You must call one or more provided functions. Do not answer without a function call."
        case .required:
            return "You must call exactly one provided function. Do not answer without a function call."
        case .function(let name) where parallelToolCalls:
            return "You must call the provided function '\(name)' at least once. Do not call any other function."
        case .function(let name):
            return "You must call the provided function '\(name)' exactly once. Do not call any other function."
        case .auto:
            return nil
        }
    }

    private static func toolDefinition(from tool: OpenAIChatTool) throws -> ToolDefinition {
        guard tool.type == "function", let function = tool.function else {
            throw APIRequestValidationError.invalidField("tools", "only function tools are supported")
        }

        let schema: [String: OpenAIJSONValue]
        if let parameters = function.parameters, parameters != .null {
            guard let object = parameters.objectValue else {
                throw APIRequestValidationError.invalidField("tools", "function parameters must be a JSON object")
            }
            schema = object
        } else {
            schema = ["type": .string("object"), "properties": .object([:]), "required": .array([])]
        }

        return ToolDefinition(
            name: function.name,
            description: function.description ?? "",
            parameterSchema: schema
        )
    }

    static func openAIToolArgumentsJSON(
        _ arguments: [String: String],
        parameterTypes: [String: String]
    ) -> String {
        let normalized = Dictionary(uniqueKeysWithValues: arguments.map { key, rawValue in
            guard let parameterType = parameterTypes[key], parameterType != "string" else {
                return (key, OpenAIJSONValue.string(rawValue))
            }
            guard let data = rawValue.data(using: .utf8),
                  let decoded = try? JSONDecoder().decode(OpenAIJSONValue.self, from: data) else {
                return (key, OpenAIJSONValue.string(rawValue))
            }
            return (key, decoded)
        })
        let data = (try? JSONEncoder().encode(normalized)) ?? Data("{}".utf8)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    private static func requiresJSONResponseFormat(
        _ responseFormat: OpenAIResponseFormat?,
        capabilities: APIEngineCapabilities
    ) throws -> Bool {
        guard let responseFormat else { return false }
        switch responseFormat.type {
        case "text":
            return false
        case "json_object":
            guard capabilities.supportsStructuredOutputs else {
                throw APIRequestValidationError.invalidField(
                    "response_format",
                    "JSON mode is not supported by this engine"
                )
            }
            return true
        case "json_schema":
            guard capabilities.supportsStrictMode else {
                throw APIRequestValidationError.invalidField(
                    "response_format",
                    "strict JSON schema outputs are not supported by this engine"
                )
            }
            return true
        default:
            throw APIRequestValidationError.invalidField(
                "response_format",
                "unsupported response format '\(responseFormat.type)'"
            )
        }
    }

    private static func resolveMaxTokens(
        maxTokens: Int?,
        maxCompletionTokens: Int?,
        capabilities: APIEngineCapabilities
    ) throws -> Int {
        if maxCompletionTokens != nil, !capabilities.supportsMaxCompletionTokens {
            throw APIRequestValidationError.invalidField(
                "max_completion_tokens",
                "this engine does not support max_completion_tokens"
            )
        }
        if let maxTokens, let maxCompletionTokens, maxTokens != maxCompletionTokens {
            throw APIRequestValidationError.invalidField(
                "max_completion_tokens",
                "must match max_tokens when both are provided"
            )
        }
        return maxCompletionTokens ?? maxTokens ?? defaultMaxTokens
    }

    private static func reasoningEffort(
        from rawValue: String?,
        capabilities: APIEngineCapabilities,
        profile: ManagedModelAPIProfile?
    ) throws -> Double? {
        guard let rawValue else { return nil }
        guard !capabilities.supportsRawProxy else { return nil }
        let normalized = rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let level = ManagedModelThinkingLevel(rawValue: normalized),
              let strength = profile?.reasoningEffortStrengths[level] else {
            let supportedLevels = ManagedModelThinkingLevel.allCases
                .filter { profile?.reasoningEffortStrengths[$0] != nil }
                .map(\.rawValue)
                .joined(separator: ", ")
            throw APIRequestValidationError.invalidField(
                "reasoning_effort",
                "must be one of \(supportedLevels)"
            )
        }
        return strength
    }

    static func isStreamingStatusMessage(_ message: String) -> Bool {
        switch message {
        case "Generating...", "Generating response", "Retrying generation", "DS4 chat completion":
            return true
        default:
            return false
        }
    }

    private static func validateSamplingCapabilities(
        _ request: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws {
        if let topK = request.top_k, topK != 0, !capabilities.supportsTopK {
            throw APIRequestValidationError.invalidField("top_k", "top-k sampling is not supported by this engine")
        }
        if let penalty = request.presence_penalty, penalty != 0, !capabilities.supportsPenalties {
            throw APIRequestValidationError.invalidField("presence_penalty", "presence penalties are not supported by this engine")
        }
        if let penalty = request.frequency_penalty, penalty != 0, !capabilities.supportsPenalties {
            throw APIRequestValidationError.invalidField("frequency_penalty", "frequency penalties are not supported by this engine")
        }
        if let penalty = request.repetition_penalty, penalty != 1, !capabilities.supportsRepetitionPenalty {
            throw APIRequestValidationError.invalidField(
                "repetition_penalty", "repetition penalties are not supported by this engine"
            )
        }
    }
}
