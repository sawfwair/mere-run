import Foundation
import Crypto
import MediaIO
import MereRunCore

struct MessagesCapabilities: Codable, Equatable, Sendable {
    let messages: Bool
    let streaming: Bool
    let tools: Bool
    let token_counting: Bool
    let ordered_images: Bool
    let thinking: [String]
    let thinking_display: [String]?
    let effort: [String]
    let exact_thinking_budget: Bool
    let strict_tools: Bool
    let json_schema: Bool
    let documents: Bool
    let citations: Bool
    let prompt_cache: Bool
    let prompt_cache_max_tokens: Int?
    let tool_search: Bool
    let context_editing: [String]
    let tool_schema_subset: [String]?

    static func adapter(_ engine: RuntimeServingEngine, profile: ManagedModelAPIProfile) -> Self {
        let native = engine != .textChatDeepseekV4Flash
        let glimmer = engine == .textChatMuseGlimmer
        let textCounting = glimmer || [.textChatQ35, .textChatQ36, .textChatGemma4].contains(engine)
        let nativeEffort = [.textChatQ35, .textChatQ36].contains(engine) && profile.compatibility.supportsReasoningEffort
        let fixedHigh = [.textChatQ35, .textChatQ36].contains(engine) && profile.thinkingLevels == [.high]
        let switchableThinking = [.textChatQ35, .textChatQ36, .textChatGemma4].contains(engine)
            && (profile.thinkingLevels.isEmpty || profile.thinkingLevels.contains(.off))
        return Self(messages: native, streaming: textCounting, tools: textCounting, token_counting: textCounting,
                    ordered_images: glimmer, thinking: glimmer ? ["adaptive"] : textCounting ? (switchableThinking ? ["adaptive", "disabled"] : ["adaptive"]) : [],
                    thinking_display: textCounting ? ["summarized"] : [],
                    effort: glimmer || nativeEffort ? ["low", "medium", "high", "xhigh"] : fixedHigh ? ["high"] : [],
                    exact_thinking_budget: false, strict_tools: false, json_schema: false,
                    documents: native, citations: false, prompt_cache: glimmer,
                    prompt_cache_max_tokens: glimmer ? MuseGlimmerGenerator.maximumCachedPrefixTokens : 0, tool_search: textCounting,
                    context_editing: textCounting ? ["clear_thinking_20251015", "clear_tool_uses_20250919"] : [],
                    tool_schema_subset: textCounting ? ChatToolValidation.schemaKeywords.sorted() : [])
    }
}

extension RuntimeChatRequest {
    static func messages(_ wire: MessagesRequest, version: MessagesVersion, counting: Bool,
                         continuity: MessagesContinuity, authenticatedScope: String = "loopback") throws -> Self {
        try wire.validate(version: version, counting: counting)
        return Self(model: wire.model, streaming: wire.stream == true) { configuration in
            let caps = MessagesCapabilities.adapter(configuration.engine, profile: configuration.profile)
            guard caps.messages else { throw MessagesFailure.unsupported("messages", model: configuration.modelID) }
            if counting, !caps.token_counting { throw MessagesFailure.unsupported("count_tokens", model: configuration.modelID) }
            if wire.stream == true, !caps.streaming { throw MessagesFailure.unsupported("stream", model: configuration.modelID) }
            if wire.output_config?.format != nil { throw MessagesFailure.unsupported("output_config.format", model: configuration.modelID) }
            let runtimeCapabilities = APIEngineCapabilities.catalog(configuration.profile)
            if wire.top_k != nil, !runtimeCapabilities.supportsTopK { throw MessagesFailure.unsupported("top_k", model: configuration.modelID) }
            if wire.stop_sequences?.isEmpty == false, !runtimeCapabilities.supportsStopSequences { throw MessagesFailure.unsupported("stop_sequences", model: configuration.modelID) }
            let limit = min(configuration.contextSize, configuration.profile.contextWindow ?? configuration.contextSize)
            let outputLimit = min(limit, configuration.settings.maxTokens ?? configuration.profile.maximumOutputTokens ?? limit,
                                  configuration.profile.maximumOutputTokens ?? limit)
            let maxTokens = wire.max_tokens ?? 1
            guard maxTokens <= outputLimit else { throw MessagesFailure.invalid("max_tokens", "exceeds the effective output limit of \(outputLimit)") }
            let allTools = try MessagesAdapter.tools(wire.tools ?? [], capabilities: caps)
            let references = MessagesAdapter.references(wire.messages.flatMap { $0.content.blocks })
            let names = Set(allTools.map(\.name))
            guard references.isSubset(of: names) else { throw MessagesFailure.invalid("tool_reference", "reference not found in the supplied catalog") }
            let tools = allTools.filter { definition in
                guard wire.tool_choice?.type != "none" else { return false }
                return wire.tools?.first(where: { $0.name == definition.name })?.defer_loading != true || references.contains(definition.name)
            }
            let choice = try MessagesAdapter.choice(wire.tool_choice, tools: tools)
            let thinking = wire.thinking
            if let thinking {
                guard caps.thinking.contains(thinking.type) else {
                    throw MessagesFailure.unsupported("thinking.\(thinking.type)", model: configuration.modelID)
                }
                if thinking.budget_tokens != nil { throw MessagesFailure.unsupported("thinking.budget_tokens", model: configuration.modelID) }
                if let display = thinking.display, caps.thinking_display?.contains(display) != true { throw MessagesFailure.unsupported("thinking.display.\(display)") }
            }
            var effort: Double?
            if let label = wire.output_config?.effort {
                guard caps.effort.contains(label) else { throw MessagesFailure.unsupported("output_config.effort.\(label)", model: configuration.modelID) }
                effort = ManagedModelThinkingLevel(rawValue: label).flatMap { configuration.profile.reasoningEffortStrengths[$0] }
                    ?? ["low": 0.1, "medium": 0.5, "high": 0.8, "xhigh": 0.95][label]
            }
            var messages: [ChatMessage] = []
            var cacheBoundaries: [ChatCacheBoundary] = []
            if let system = wire.system {
                guard system.blocks.allSatisfy({ if case .text = $0 { true } else { false } }) else {
                    throw MessagesFailure.invalid("system", "only text blocks are supported")
                }
                for (index, block) in system.blocks.enumerated() {
                    if let cache = MessagesAdapter.cacheControl(block) { cacheBoundaries.append(.init(messageIndex: 0, blockCount: index + 1, ttlSeconds: cache.ttl == "1h" ? 3600 : 300)) }
                }
                messages.append(ChatMessage(role: .system, blocks: try MessagesAdapter.blocks(system.blocks, configuration: configuration, continuity: continuity)))
            }
            var pending: Set<String> = []
            var used: Set<String> = []
            var conversationTurn = 0
            for turn in MessagesAdapter.normalizedTurns(wire.messages) {
                if turn.role == "user", turn.content.blocks.contains(where: { if case .toolResult = $0 { false } else { true } }) { conversationTurn += 1 }
                var resultIDs: Set<String> = []
                for block in turn.content.blocks {
                    if case .toolResult(let result) = block {
                        guard pending.contains(result.tool_use_id), resultIDs.insert(result.tool_use_id).inserted else {
                            throw MessagesFailure.invalid("tool_result.tool_use_id", "must match an unresolved tool use exactly once")
                        }
                    }
                }
                if !pending.isEmpty, (turn.role != "user" || resultIDs != pending) {
                    throw MessagesFailure.invalid("tool_result", "the next user message must resolve all outstanding tool IDs")
                }
                if !pending.isEmpty {
                    let prefix = turn.content.blocks.prefix { if case .toolResult = $0 { true } else { false } }
                    guard prefix.count == resultIDs.count else { throw MessagesFailure.invalid("tool_result", "results must precede other content in the combined user turn") }
                }
                var ordinary: [ChatContentBlock] = []
                func flush() {
                    if !ordinary.isEmpty {
                        var message = ChatMessage(role: ChatMessage.Role(rawValue: turn.role)!, blocks: ordinary)
                        message.conversationTurn = conversationTurn
                        messages.append(message)
                        ordinary = []
                    }
                }
                let converted = try MessagesAdapter.blocks(turn.content.blocks, configuration: configuration, continuity: continuity)
                for (wireBlock, block) in zip(turn.content.blocks, converted) {
                    switch block {
                    case .toolUse(let call):
                        if case .serverToolUse = wireBlock {
                            guard wire.tools?.contains(where: { $0.name == call.name && ["tool_search_tool_regex_20251119", "tool_search_tool_bm25_20251119"].contains($0.type ?? "") }) == true else {
                                throw MessagesFailure.unsupported("server_tool_use.\(call.name)")
                            }
                        }
                        guard let id = call.id, used.insert(id).inserted else { throw MessagesFailure.invalid("tool_use.id", "must be unique across the conversation") }
                        pending.insert(id); ordinary.append(block)
                    case .toolResult(let id, let content, let isError):
                        guard pending.remove(id) != nil else { throw MessagesFailure.invalid("tool_result.tool_use_id", "must match an unresolved tool use exactly once") }
                        flush()
                        var message = ChatMessage(role: .tool, blocks: content)
                        message.toolCallID = id
                        message.conversationTurn = conversationTurn
                        message.toolResultIsError = isError
                        message.name = wire.messages.flatMap { $0.content.blocks }.compactMap {
                            switch $0 {
                            case .toolUse(let use), .serverToolUse(let use): return use.id == id ? use.name : nil
                            default: return nil
                            }
                        }.first
                        if isError {
                            message.content = "Tool error:\n" + message.content
                            message.contentBlocks?.insert(.text("Tool error:"), at: 0)
                        }
                        messages.append(message)
                    default: ordinary.append(block)
                    }
                    if let marker = MessagesAdapter.cacheControl(wireBlock) {
                        let isResult: Bool
                        if case .toolResult = block { isResult = true } else { isResult = false }
                        let messageIndex = isResult ? messages.count - 1 : messages.count
                        let blockCount = isResult ? (messages.last?.contentBlocks?.count ?? 1) : ordinary.count
                        cacheBoundaries.append(.init(messageIndex: messageIndex, blockCount: blockCount, ttlSeconds: marker.ttl == "1h" ? 3600 : 300))
                    }
                }
                flush()
            }
            guard pending.isEmpty else { throw MessagesFailure.invalid("messages", "unresolved tool uses require results before another generation") }

            var request = ChatRequest(messages: messages, maxTokens: max(1, maxTokens), reasoningEffort: effort,
                                      showThinking: thinking != nil && thinking?.type != "disabled", tools: tools.isEmpty ? nil : tools,
                                      toolChoice: choice, parallelToolCalls: !(wire.tool_choice?.disable_parallel_tool_use ?? false),
                                      stopSequences: wire.stop_sequences ?? [], maxContextTokens: limit)
            request = try ChatRequestResolver.resolve(request, modelID: configuration.modelID,
                sampling: ChatSamplingOptions(temperature: wire.temperature ?? configuration.settings.temperature,
                                              topP: wire.top_p ?? configuration.settings.topP,
                                              topK: wire.top_k, thinking: thinking != nil && thinking?.type != "disabled"),
                policy: .openAI, apiProfile: configuration.profile)
            let fullTTL = wire.cache_control
            let toolTTL = wire.tools?.last?.cache_control
            if !cacheBoundaries.isEmpty || fullTTL != nil || toolTTL != nil {
                request.prefixCachePolicy = .init(scope: authenticatedScope, boundaries: cacheBoundaries,
                    wholePromptTTL: fullTTL.map { $0.ttl == "1h" ? 3600 : 300 },
                    toolPrefixTTL: toolTTL.map { $0.ttl == "1h" ? 3600 : 300 })
            }
            request.cacheWarmOnly = !counting && wire.max_tokens == 0
            request.reserveOutputContext = !counting
            if request.cacheWarmOnly, !caps.prompt_cache || request.prefixCachePolicy == nil { throw MessagesFailure.unsupported("zero-output cache warming") }
            request.contextEdits = try MessagesAdapter.contextEdits(wire.context_management, capabilities: caps)
            let searches = Dictionary(uniqueKeysWithValues: (wire.tools ?? []).compactMap { tool -> (String, ChatToolSearch.Mode)? in
                switch tool.type {
                case "tool_search_tool_regex_20251119": return (tool.name, .regex)
                case "tool_search_tool_bm25_20251119": return (tool.name, .bm25)
                default: return nil
                }
            })
            if !searches.isEmpty, wire.tool_choice?.type != "none" {
                request.toolSearch = ChatToolSearch(catalog: allTools.filter { searches[$0.name] == nil }, searchTools: searches)
            }
            if let lora = configuration.fallbackLoraPath { request.lora = .local(path: lora, scale: 1) }
            return RuntimePreparedChat(request: request, includeUsage: true, scope: APIModelScope(warnings: []))
        }
    }
}

enum MessagesAdapter {
    static func normalizedTurns(_ turns: [MessagesTurn]) -> [MessagesTurn] {
        var result: [MessagesTurn] = []
        for turn in turns {
            if let last = result.last, last.role == turn.role, ["user", "assistant"].contains(turn.role) {
                result[result.count - 1] = MessagesTurn(role: turn.role, content: .blocks(last.content.blocks + turn.content.blocks))
            } else { result.append(turn) }
        }
        return result
    }

    static func tools(_ catalog: [MessagesTool], capabilities: MessagesCapabilities) throws -> [ToolDefinition] {
        if !catalog.isEmpty, !capabilities.tools { throw MessagesFailure.unsupported("tools") }
        guard catalog.filter({ $0.defer_loading == true }).count <= 10000 else { throw MessagesFailure.invalid("tools", "at most 10000 tools can be deferred") }
        guard catalog.count <= 10002 else { throw MessagesFailure.invalid("tools", "at most 10000 deferred tools are supported") }
        if !catalog.isEmpty, catalog.allSatisfy({ $0.defer_loading == true }) { throw MessagesFailure.invalid("tools", "at least one tool must be non-deferred") }
        var names: Set<String> = []
        return try catalog.map { tool in
            let search = ["tool_search_tool_regex_20251119", "tool_search_tool_bm25_20251119"].contains(tool.type ?? "")
            if let type = tool.type, type != "custom", !search { throw MessagesFailure.unsupported("tools.type.\(type)") }
            guard !tool.name.isEmpty, names.insert(tool.name).inserted else { throw MessagesFailure.invalid("tools.name", "must be non-empty and unique") }
            if tool.strict == true { throw MessagesFailure.unsupported("tools.strict") }
            if tool.defer_loading == true, tool.cache_control != nil { throw MessagesFailure.invalid("tools.cache_control", "deferred tools cannot carry cache breakpoints") }
            if search {
                guard tool.defer_loading != true, tool.strict != true, tool.max_uses == nil else { throw MessagesFailure.invalid("tool_search", "search must be non-deferred and use the documented options") }
                let query = tool.type == "tool_search_tool_regex_20251119" ? "pattern" : "query"
                return ToolDefinition(name: tool.name, description: "Search the local tool catalog and load matching tool definitions.", parameterSchema: [
                    "type": .string("object"), "properties": .object([query: .object(["type": .string("string")]),
                    "limit": .object(["type": .string("integer")])]), "required": .array([.string(query)])])
            }
            if tool.allowed_callers?.contains(where: { $0 != "direct" }) == true { throw MessagesFailure.unsupported("tools.allowed_callers") }
            if tool.max_uses != nil { throw MessagesFailure.unsupported("tools.max_uses") }
            if tool.input_examples != nil { throw MessagesFailure.unsupported("tools.input_examples") }
            try tool.cache_control?.validate()
            guard let schema = tool.input_schema, schema["type"] == .string("object") else {
                throw MessagesFailure.invalid("tools.input_schema", "must be an object schema")
            }
            try ChatToolValidation.validateSchema(schema)
            return ToolDefinition(name: tool.name, description: tool.description ?? "", parameterSchema: schema)
        }
    }

    static func choice(_ value: MessagesToolChoice?, tools: [ToolDefinition]) throws -> ChatToolChoice {
        guard let value else { return .auto }
        if value.name != nil, value.type != "tool" { throw MessagesFailure.invalid("tool_choice.name", "is only valid with type tool") }
        switch value.type {
        case "auto": return .auto
        case "any":
            guard !tools.isEmpty else { throw MessagesFailure.invalid("tool_choice", "any requires tools") }
            return .required
        case "tool":
            guard let name = value.name, tools.contains(where: { $0.name == name }) else { throw MessagesFailure.invalid("tool_choice.name", "must name a supplied tool") }
            return .function(name)
        case "none": return .auto
        default: throw MessagesFailure.invalid("tool_choice.type", "expected auto, any, tool or none")
        }
    }

    static func blocks(_ blocks: [MessagesBlock], configuration: RuntimeChatConfiguration,
                       continuity: MessagesContinuity) throws -> [ChatContentBlock] {
        try blocks.map { block in
            switch block {
            case .text(let text):
                if text.citations?.isEmpty == false { throw MessagesFailure.unsupported("text.citations") }
                return .text(text.text)
            case .thinking(let thinking):
                try continuity.verify(signature: thinking.signature, thinking: thinking.thinking, model: configuration.modelID)
                return .reasoning(text: thinking.thinking, continuity: thinking.signature)
            case .toolUse(let call): return .toolUse(.init(id: call.id, name: call.name, arguments: call.input))
            case .toolResult(let result):
                return .toolResult(id: result.tool_use_id,
                    content: try self.blocks(result.content?.blocks ?? [], configuration: configuration, continuity: continuity),
                    isError: result.is_error ?? false)
            case .toolReference(let reference): return .toolReference(reference.tool_name)
            case .serverToolUse(let call):
                guard [.textChatMuseGlimmer, .textChatQ35, .textChatQ36, .textChatGemma4].contains(configuration.engine) else { throw MessagesFailure.unsupported("server_tool_use") }
                return .toolUse(.init(id: call.id, name: call.name, arguments: call.input))
            case .toolSearchResult(let result):
                var content = try self.blocks(result.content.tool_references ?? [], configuration: configuration, continuity: continuity)
                if let code = result.content.error_code {
                    content.append(.text("Tool search error (\(code)): " + (result.content.error_message ?? "")))
                }
                return .toolResult(id: result.tool_use_id, content: content, isError: result.content.type == "tool_search_tool_result_error")
            case .image(let image):
                guard configuration.engine == .textChatMuseGlimmer else { throw MessagesFailure.unsupported("image", model: configuration.modelID) }
                guard image.source.type == "base64", let media = image.source.media_type,
                      ["image/png", "image/jpeg", "image/gif", "image/webp"].contains(media),
                      let data = image.source.data, Data(base64Encoded: data) != nil else {
                    throw MessagesFailure.invalid("image.source", "supply local base64 image data with a supported media_type")
                }
                return .image(reference: "data:\(media);base64,\(data)", identity: digest(Data(data.utf8)))
            case .document(let document):
                if document.citations?.enabled == true { throw MessagesFailure.unsupported("document.citations") }
                let source = document.source
                let bytes: Data
                let decoded: MediaDocument
                if source.type == "text", let text = source.data {
                    bytes = Data(text.utf8); decoded = try MediaDocumentIO.text(bytes)
                } else if source.type == "base64", let base64 = source.data, let data = Data(base64Encoded: base64) {
                    bytes = data
                    switch source.media_type {
                    case "application/pdf": decoded = try MediaDocumentIO.pdf(data)
                    case "text/plain": decoded = try MediaDocumentIO.text(data)
                    default: throw MessagesFailure.invalid("document.source.media_type", "use application/pdf or text/plain")
                    }
                } else { throw MessagesFailure.unsupported("document.source.\(source.type)") }
                return .document(.init(identity: digest(bytes), title: document.title, context: document.context, pages: decoded.pages))
            }
        }
    }

    static func cacheControl(_ block: MessagesBlock) -> MessagesCacheControl? {
        switch block {
        case .text(let block): block.cache_control
        case .image(let block): block.cache_control
        case .document(let block): block.cache_control
        case .toolUse(let block): block.cache_control
        case .toolResult(let block): block.cache_control
        default: nil
        }
    }

    static func references(_ blocks: [MessagesBlock]) -> Set<String> {
        var result: Set<String> = []
        for block in blocks {
            switch block {
            case .toolReference(let reference): result.insert(reference.tool_name)
            case .toolResult(let tool): result.formUnion(references(tool.content?.blocks ?? []))
            case .toolSearchResult(let tool): result.formUnion(references(tool.content.tool_references ?? []))
            default: break
            }
        }
        return result
    }

    static func contextEdits(_ management: MessagesContextManagement?, capabilities: MessagesCapabilities) throws -> [ChatContextEdit] {
        try (management?.edits ?? []).map { edit in
            guard capabilities.context_editing.contains(edit.type) else { throw MessagesFailure.unsupported("context_management.\(edit.type)") }
            if edit.type == "clear_thinking_20251015" {
                guard edit.trigger == nil, edit.clear_at_least == nil, edit.exclude_tools == nil, edit.clear_tool_inputs == nil else {
                    throw MessagesFailure.invalid("clear_thinking", "only keep is supported by this strategy")
                }
                if edit.keep == .string("all") { return .reasoning(keepTurns: nil) }
                return .reasoning(keepTurns: try quantity(edit.keep, kind: "thinking_turns", default: 1, minimum: 1))
            }
            let trigger = edit.trigger ?? .init(type: "input_tokens", value: 100000)
            let minimum = edit.clear_at_least ?? .init(type: "input_tokens", value: 0)
            guard ["input_tokens", "tool_uses"].contains(trigger.type), trigger.value >= 0, minimum.type == "input_tokens", minimum.value >= 0 else {
                throw MessagesFailure.invalid("context_management", "trigger must be input_tokens or tool_uses; clear_at_least must be input_tokens; values must be non-negative")
            }
            return .toolResults(triggerTokens: trigger.value, triggerToolUses: trigger.type == "tool_uses" ? trigger.value : nil,
                keepUses: try quantity(edit.keep, kind: "tool_uses", default: 3, minimum: 0),
                minimumTokens: minimum.value, excludeTools: Set(edit.exclude_tools ?? []), clearInputs: edit.clear_tool_inputs ?? false)
        }
    }
    private static func quantity(_ value: ChatJSONValue?, kind: String, default fallback: Int, minimum: Int) throws -> Int {
        guard let value else { return fallback }
        guard let object = value.objectValue, object["type"] == .string(kind), case .number(let number) = object["value"],
              number.isFinite, number.rounded() == number, number >= Double(minimum), number < Double(Int.max) else {
            throw MessagesFailure.invalid("context_management.keep", "expected a valid \(kind) quantity")
        }
        return Int(number)
    }

    static func digest(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
}
