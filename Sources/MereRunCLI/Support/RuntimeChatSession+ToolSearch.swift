import Foundation
import MereRunCore

/// Captures ordered typed events from synchronous runtime callbacks.
final class RuntimeChatEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [ChatGenerationEvent] = []
    func append(_ event: ChatGenerationEvent) { lock.withLock { storage.append(event) } }
    var events: [ChatGenerationEvent] { lock.withLock { storage } }
}

extension RuntimeChatSession {
    func chatWithToolSearch(_ search: ChatToolSearch, generate: (ChatRequest, (@Sendable (ChatProgress) -> Void)?) async throws -> ChatResponse,
                            progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws -> ChatResponse {
        var input = request
        var allEvents: [ChatGenerationEvent] = []
        var outputTokens = 0
        var inputTokens = 0
        var cacheReads = 0
        var cacheWrites = 0
        for _ in 0..<64 {
            try Task.checkCancellation()
            let observed = RuntimeChatEvents()
            var result = try await generate(input) { progress in
                if let event = progress.event { observed.append(event) }
                progressHandler?(progress)
            }
            guard result.tokensGenerated >= 0, result.tokensGenerated <= input.maxTokens else {
                throw MessagesFailure.output("output_tokens", "adapter exceeded the shared output budget")
            }
            outputTokens += result.tokensGenerated
            inputTokens += result.promptTokens ?? 0
            cacheReads += result.cacheReadTokens
            cacheWrites += result.cacheCreationTokens
            allEvents += observed.events
            let calls = result.toolCalls ?? []
            guard input.parallelToolCalls || calls.count <= 1 else { throw MessagesFailure.output("tool_use", "adapter generated parallel calls when disabled") }
            guard calls.allSatisfy({ call in input.tools?.contains(where: { $0.name == call.name }) == true }) else {
                throw MessagesFailure.output("tool_use", "adapter requested a tool that was not loaded in the prepared prompt")
            }
            let searches = calls.filter { search.searchTools[$0.name] != nil }
            if searches.isEmpty {
                result.tokensGenerated = outputTokens
                result.promptTokens = inputTokens
                result.cacheReadTokens = cacheReads; result.cacheCreationTokens = cacheWrites
                result.generationEvents = allEvents
                return result
            }
            guard result.tokensGenerated > 0 else { throw MessagesFailure.output("tool_search", "adapter returned a search round without accounting for output tokens") }
            var uses: [ChatMessageToolCall] = []
            var results: [ChatMessage] = []
            for call in searches {
                guard let arguments = call.jsonArguments else { throw MessagesFailure.invalid("tool_search", "search input must be validated typed JSON") }
                let id = "srvtoolu_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
                let use = ChatMessageToolCall(id: id, name: call.name, arguments: arguments)
                uses.append(use)
                let start = ChatGenerationEvent.catalogSearchStart(use)
                allEvents.append(start); progressHandler?(.init(stage: .generating, event: start))
                let found: [ToolDefinition]
                let failure: ChatToolSearchFailure?
                do { found = try search.search(name: call.name, input: arguments); failure = nil }
                catch is CancellationError { throw CancellationError() }
                catch {
                    found = []
                    failure = (error as? ChatToolSearchFailure) ?? .init(code: "invalid_tool_input", message: error.localizedDescription)
                }
                let event = ChatGenerationEvent.catalogSearchResult(id: id, names: found.map(\.name), error: failure)
                allEvents.append(event); progressHandler?(.init(stage: .generating, event: event))
                var tools = input.tools ?? []
                for tool in found where !tools.contains(where: { $0.name == tool.name }) { tools.append(tool) }
                input.tools = tools
                results.append(ChatMessage(role: .tool, content: failure?.message ?? "Discovered tools: " + found.map(\.name).joined(separator: ", "), name: call.name, toolCallID: id))
            }
            let clientCalls = calls.filter { search.searchTools[$0.name] == nil }
            if !clientCalls.isEmpty {
                result.toolCalls = clientCalls
                result.tokensGenerated = outputTokens; result.promptTokens = inputTokens
                result.cacheReadTokens = cacheReads; result.cacheCreationTokens = cacheWrites
                result.generationEvents = allEvents
                return result
            }
            let content = observed.events.compactMap { event -> ChatContentBlock? in
                switch event {
                case .text(let text): return .text(text)
                case .reasoning(let text): return .reasoning(text: text, continuity: nil)
                default: return nil
                }
            }
            input.messages.append(ChatMessage(role: .assistant, blocks: content + uses.map(ChatContentBlock.toolUse)))
            input.messages += results
            input.maxTokens = request.maxTokens - outputTokens
            input.toolChoice = .auto
            input.preparedPrompt = nil
            if input.maxTokens <= 0 {
                result.toolCalls = nil; result.tokensGenerated = outputTokens; result.finishReason = .length
                result.promptTokens = inputTokens
                result.cacheReadTokens = cacheReads; result.cacheCreationTokens = cacheWrites
                result.generationEvents = allEvents
                return result
            }
            input = try await prepareContinuation(input)
        }
        throw MessagesFailure.output("tool_search", "local search exceeded the 64-round request limit")
    }
}
