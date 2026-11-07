import Foundation
import MereRunCore
import NIOCore

/// Serializes callback delivery and guarantees one ordered block lifecycle.
final class MessagesStreamWriter: @unchecked Sendable {
    private let lock = NSLock()
    private let yield: @Sendable (ByteBuffer) -> Void
    private let continuity: MessagesContinuity
    private let model: String
    private let requestID: String
    private let showThinking: Bool
    private var index = -1
    private var channel: String?
    private var reasoning = ""
    private var terminated = false
    private var streamedText = false
    private var streamedThinking = false
    private var started = false
    private var startFields: [String: ChatJSONValue]

    init(id: String, inputTokens: Int, model: String, requestID: String, showThinking: Bool, continuity: MessagesContinuity,
         yield: @escaping @Sendable (ByteBuffer) -> Void) throws {
        self.model = model; self.requestID = requestID; self.showThinking = showThinking
        self.continuity = continuity; self.yield = yield
        let message = MessagesResponse(id: id, model: model, content: [], stop_reason: nil, stop_sequence: nil,
                                       usage: .init(input_tokens: inputTokens, output_tokens: 0))
        let encoded = try JSONEncoder().encode(message)
        startFields = try JSONDecoder().decode([String: ChatJSONValue].self, from: encoded)
    }
    func ping() { lock.withLock { if !terminated { send("ping", [:]) } } }
    func observe(_ event: ChatGenerationEvent) {
        lock.withLock {
            guard !terminated else { return }
            if case .usage(let input, _, let read, let write) = event {
                begin(usage: .init(input_tokens: max(0, input - read - write), output_tokens: 0,
                                  cache_creation_input_tokens: write, cache_read_input_tokens: read))
                return
            }
            begin()
            switch event {
            case .text(let text):
                guard !text.isEmpty else { return }
                streamedText = true; open("text")
                send("content_block_delta", ["index": .number(Double(index)), "delta": .object(["type": .string("text_delta"), "text": .string(text)])])
            case .reasoning(let text):
                guard showThinking, !text.isEmpty else { return }
                streamedThinking = true; open("thinking"); reasoning += text
                send("content_block_delta", ["index": .number(Double(index)), "delta": .object(["type": .string("thinking_delta"), "thinking": .string(text)])])
            case .catalogSearchStart(let call):
                close(); index += 1
                send("content_block_start", ["index": .number(Double(index)), "content_block": .object([
                    "type": .string("server_tool_use"), "id": .string(call.id ?? ""), "name": .string(call.name), "input": .object([:])])])
                send("content_block_delta", ["index": .number(Double(index)), "delta": .object([
                    "type": .string("input_json_delta"), "partial_json": .string(ChatJSONValue.object(call.arguments).jsonString)])])
                send("content_block_stop", ["index": .number(Double(index))])
            case .catalogSearchResult(let id, let names, let error):
                close(); index += 1
                let block = MessagesAdapter.searchResult(id: id, names: names, error: error)
                if let bytes = try? JSONEncoder().encode(block), let value = try? JSONDecoder().decode(ChatJSONValue.self, from: bytes) {
                    send("content_block_start", ["index": .number(Double(index)), "content_block": value])
                    send("content_block_stop", ["index": .number(Double(index))])
                }
            default: break
            }
        }
    }
    func finish(_ response: MessagesResponse) throws {
        try lock.withLock {
            guard !terminated else { return }
            begin(usage: .init(input_tokens: response.usage.input_tokens, output_tokens: 0,
                              cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
                              cache_read_input_tokens: response.usage.cache_read_input_tokens))
            // Adapters with typed streaming must emit all text and reasoning.
            // Completed calls are released only after validation of the entire result.
            for block in response.content {
                switch block {
                case .text(let text) where !streamedText:
                    open("text")
                    send("content_block_delta", ["index": .number(Double(index)), "delta": .object(["type": .string("text_delta"), "text": .string(text.text)])])
                case .thinking(let thinking) where !streamedThinking:
                    open("thinking"); reasoning = thinking.thinking
                    send("content_block_delta", ["index": .number(Double(index)), "delta": .object(["type": .string("thinking_delta"), "thinking": .string(thinking.thinking)])])
                case .toolUse(let call):
                    close(); index += 1
                    send("content_block_start", ["index": .number(Double(index)), "content_block": .object([
                        "type": .string("tool_use"), "id": .string(call.id), "name": .string(call.name), "input": .object([:])])])
                    let data = try JSONEncoder().encode(call.input)
                    send("content_block_delta", ["index": .number(Double(index)), "delta": .object([
                        "type": .string("input_json_delta"), "partial_json": .string(String(decoding: data, as: UTF8.self))])])
                    send("content_block_stop", ["index": .number(Double(index))])
                default: break
                }
            }
            close()
            var terminal: [String: ChatJSONValue] = ["delta": .object(["stop_reason": response.stop_reason.map(ChatJSONValue.string) ?? .null,
                                                      "stop_sequence": response.stop_sequence.map(ChatJSONValue.string) ?? .null]),
                                   "usage": .object(["input_tokens": .number(Double(response.usage.input_tokens)),
                                                     "output_tokens": .number(Double(response.usage.output_tokens)),
                                                     "cache_creation_input_tokens": .number(Double(response.usage.cache_creation_input_tokens)),
                                                     "cache_read_input_tokens": .number(Double(response.usage.cache_read_input_tokens))])]
            if let management = response.context_management { terminal["context_management"] = management }
            send("message_delta", terminal)
            send("message_stop", [:]); terminated = true
        }
    }
    func fail(_ failure: MessagesFailure) {
        lock.withLock {
            guard !terminated else { return }
            send("error", ["error": .object(["type": .string(failure.type), "message": .string(failure.message)]),
                           "request_id": .string(requestID)])
            terminated = true
        }
    }
    private func open(_ type: String) {
        guard channel != type else { return }
        close(); index += 1; channel = type
        var content: [String: ChatJSONValue] = ["type": .string(type)]
        if type == "thinking" { content["thinking"] = .string(""); content["signature"] = .string(""); reasoning = "" }
        else { content["text"] = .string("") }
        send("content_block_start", ["index": .number(Double(index)), "content_block": .object(content)])
    }
    private func begin(usage: MessagesUsage? = nil) {
        guard !started else { return }
        if let usage {
            startFields["usage"] = .object([
                "input_tokens": .number(Double(usage.input_tokens)), "output_tokens": .number(0),
                "cache_creation_input_tokens": .number(Double(usage.cache_creation_input_tokens)),
                "cache_read_input_tokens": .number(Double(usage.cache_read_input_tokens))])
        }
        send("message_start", ["message": .object(startFields)]); started = true
    }
    private func close() {
        guard let channel else { return }
        if channel == "thinking" {
            send("content_block_delta", ["index": .number(Double(index)), "delta": .object([
                "type": .string("signature_delta"), "signature": .string(continuity.sign(thinking: reasoning, model: model))])])
        }
        send("content_block_stop", ["index": .number(Double(index))]); self.channel = nil
    }
    private func send(_ type: String, _ fields: [String: ChatJSONValue]) {
        var fields = fields; fields["type"] = .string(type)
        // The encoder cannot fail for these finite, internally constructed values.
        guard let data = try? JSONEncoder().encode(fields) else { return }
        yield(ByteBuffer(string: "event: \(type)\ndata: \(String(decoding: data, as: UTF8.self))\n\n"))
    }
}

extension MessagesAdapter {
    static func response(_ result: ChatResponse, session: RuntimeChatSession, id: String,
                         continuity: MessagesContinuity, inputTokens: Int) throws -> MessagesResponse {
        var blocks: [MessagesBlock] = result.generationEvents.map {
            eventBlocks($0, model: session.modelID, showThinking: session.request.showThinking, continuity: continuity)
        } ?? []
        if result.generationEvents == nil, session.request.showThinking, let reasoning = result.reasoningContent, !reasoning.isEmpty {
            blocks.append(.thinking(.init(thinking: reasoning, signature: continuity.sign(thinking: reasoning, model: session.modelID))))
        }
        let visible = ChatReasoningMarkup.splitThinkBlocks(in: result.response).visibleContent
        if result.generationEvents == nil, !visible.isEmpty { blocks.append(.text(.init(text: visible))) }
        let calls = result.toolCalls ?? []
        guard session.request.parallelToolCalls || calls.count <= 1 else { throw MessagesFailure.output("tool_use", "model generated disallowed parallel calls") }
        for call in calls {
            let definition = session.request.tools?.first(where: { $0.name == call.name })
                ?? session.request.toolSearch?.catalog.first(where: { $0.name == call.name })
            guard let definition, let input = call.jsonArguments else {
                throw MessagesFailure.output("tool_use", "adapter did not return validated typed arguments for a declared tool")
            }
            do { try ChatToolValidation.validate(.object(input), schema: definition.parametersJSONSchema) }
            catch { throw MessagesFailure.output("tool_use.input", error.localizedDescription) }
            blocks.append(.toolUse(.init(id: "toolu_" + UUID().uuidString.replacingOccurrences(of: "-", with: ""), name: call.name, input: input)))
        }
        let searchedNames = result.generationEvents?.compactMap { event -> String? in
            if case .catalogSearchStart(let call) = event { return call.name }; return nil
        } ?? []
        switch session.request.cacheWarmOnly ? .auto : session.request.toolChoice {
        case .required where calls.isEmpty && searchedNames.isEmpty: throw MessagesFailure.output("tool_choice", "model did not produce a required tool call")
        case .function(let name) where !searchedNames.contains(name) && (calls.isEmpty || calls.contains(where: { $0.name != name })):
            throw MessagesFailure.output("tool_choice", "model did not produce the selected tool")
        default: break
        }
        let reason = !calls.isEmpty ? "tool_use" : result.finishReason == .length ? "max_tokens" : result.finishReason == .stopSequence ? "stop_sequence" : "end_turn"
        return MessagesResponse(id: id, model: session.modelID, content: blocks, stop_reason: reason, stop_sequence: result.matchedStopSequence,
                                usage: .init(input_tokens: max(0, (result.promptTokens ?? inputTokens) - result.cacheReadTokens - result.cacheCreationTokens),
                                             output_tokens: result.tokensGenerated, cache_creation_input_tokens: result.cacheCreationTokens,
                                             cache_read_input_tokens: result.cacheReadTokens),
                                context_management: session.appliedContextEdits.isEmpty ? nil : .object(["applied_edits": .array(session.appliedContextEdits.map { edit in
                                    var fields: [String: ChatJSONValue] = ["type": .string(edit.reasoning ? "clear_thinking_20251015" : "clear_tool_uses_20250919"),
                                        "cleared_input_tokens": .number(Double(edit.clearedTokens))]
                                    fields[edit.reasoning ? "cleared_thinking_turns" : "cleared_tool_uses"] = .number(Double(edit.reasoning ? edit.clearedThinkingTurns : edit.clearedToolUses))
                                    return .object(fields)
                                })]))
    }
}

extension MessagesAdapter {
    static func searchResult(id: String, names: [String], error: ChatToolSearchFailure?) -> MessagesBlock {
        .toolSearchResult(.init(tool_use_id: id, content: .init(
            type: error == nil ? "tool_search_tool_search_result" : "tool_search_tool_result_error",
            tool_references: error == nil ? names.map { .toolReference(.init(tool_name: $0)) } : nil,
            error_code: error?.code, error_message: error?.message)))
    }
    static func eventBlocks(_ events: [ChatGenerationEvent], model: String, showThinking: Bool,
                            continuity: MessagesContinuity) -> [MessagesBlock] {
        var blocks: [MessagesBlock] = []
        for event in events {
            switch event {
            case .text(let text):
                if case .text(let last) = blocks.last {
                    blocks[blocks.count - 1] = .text(.init(text: last.text + text))
                } else { blocks.append(.text(.init(text: text))) }
            case .reasoning(let text) where showThinking:
                let combined: String
                if case .thinking(let last) = blocks.last { combined = last.thinking + text; blocks.removeLast() }
                else { combined = text }
                blocks.append(.thinking(.init(thinking: combined, signature: continuity.sign(thinking: combined, model: model))))
            case .catalogSearchStart(let call):
                blocks.append(.serverToolUse(.init(id: call.id ?? "", name: call.name, input: call.arguments)))
            case .catalogSearchResult(let id, let names, let error): blocks.append(searchResult(id: id, names: names, error: error))
            default: break
            }
        }
        return blocks
    }
}
