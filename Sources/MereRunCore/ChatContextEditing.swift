import Foundation

public enum ChatContextEdit: Hashable, Sendable {
    case reasoning(keepTurns: Int?)
    case toolResults(triggerTokens: Int, triggerToolUses: Int? = nil, keepUses: Int, minimumTokens: Int, excludeTools: Set<String>, clearInputs: Bool)
}
public struct ChatAppliedContextEdit: Hashable, Sendable {
    public let reasoning: Bool
    public let clearedTokens: Int
    public let clearedToolUses: Int
    public var clearedThinkingTurns: Int = 0
}
public struct ChatPromptPreparation: Sendable {
    public let request: ChatRequest
    public let prompt: ChatPreparedPrompt
    public let applied: [ChatAppliedContextEdit]
    public let originalInputTokens: Int

    public static func prepare(_ original: ChatRequest,
                               tokenize: (ChatRequest) async throws -> ChatPreparedPrompt) async throws -> Self {
        var request = original
        var prompt = try await tokenize(request)
        let originalCount = prompt.tokens.count
        var applied: [ChatAppliedContextEdit] = []
        guard request.contextEdits.count <= 16 else { throw ChatRequestIssue("context_management", "at most 16 edits are supported") }
        for edit in request.contextEdits {
            try Task.checkCancellation()
            switch edit {
            case .reasoning(let keep):
                guard let keep else { continue }
                var turn = 0
                var turns: [Int] = []
                for message in request.messages {
                    if let specified = message.conversationTurn { turn = specified }
                    else if message.role == .user { turn += 1 }
                    turns.append(turn)
                }
                let cutoff = turn - keep + 1
                let clearedIndices = request.messages.indices.filter { index in
                    turns[index] < cutoff && (request.messages[index].reasoningContent?.isEmpty == false ||
                        request.messages[index].contentBlocks?.contains(where: { if case .reasoning = $0 { true } else { false } }) == true)
                }
                guard !clearedIndices.isEmpty else { continue }
                let clearedTurns = Set(clearedIndices.map { turns[$0] }).count
                for index in clearedIndices {
                    let blocks = request.messages[index].contentBlocks ?? []
                    let removed = blocks.indices.filter { if case .reasoning = blocks[$0] { true } else { false } }
                    request.remapCacheBoundaries(messageIndex: index) { count in count - removed.filter { $0 < count }.count }
                    request.messages[index].reasoningContent = nil
                    request.messages[index].contentBlocks?.removeAll { if case .reasoning = $0 { true } else { false } }
                }
                let edited = try await tokenize(request)
                applied.append(.init(reasoning: true, clearedTokens: max(0, prompt.tokens.count - edited.tokens.count), clearedToolUses: 0, clearedThinkingTurns: clearedTurns))
                prompt = edited
            case .toolResults(let trigger, let toolTrigger, let keep, let minimum, let excluded, let clearInputs):
                let toolCount = request.messages.filter { $0.role == .tool && $0.toolResultCleared != true }.count
                guard toolTrigger.map({ toolCount > $0 }) ?? (prompt.tokens.count > trigger) else { continue }
                var candidate = request
                let results = candidate.messages.indices.filter { candidate.messages[$0].role == .tool && candidate.messages[$0].toolResultCleared != true }
                var removed = 0
                var edited = prompt
                for index in results.dropLast(min(keep, results.count)) {
                    let message = candidate.messages[index]
                    guard !excluded.contains(message.name ?? ""), let id = message.toolCallID else { continue }
                    candidate.messages[index].content = "[Tool result cleared]"
                    candidate.messages[index].contentBlocks = [.text("[Tool result cleared]")]
                    candidate.messages[index].toolResultCleared = true
                    candidate.remapCacheBoundaries(messageIndex: index) { _ in 1 }
                    if clearInputs {
                        for useIndex in candidate.messages.indices {
                            if let calls = candidate.messages[useIndex].toolCalls {
                                candidate.messages[useIndex].toolCalls = calls.map { call in
                                    call.id == id ? ChatMessageToolCall(id: id, name: call.name, arguments: [:]) : call
                                }
                            }
                            candidate.messages[useIndex].contentBlocks = candidate.messages[useIndex].contentBlocks?.map { block in
                                if case .toolUse(let call) = block, call.id == id { return .toolUse(.init(id: id, name: call.name, arguments: [:])) }
                                return block
                            }
                        }
                    }
                    removed += 1
                    edited = try await tokenize(candidate)
                    if prompt.tokens.count - edited.tokens.count >= minimum, minimum > 0 { break }
                }
                let cleared = max(0, prompt.tokens.count - edited.tokens.count)
                if removed > 0, cleared >= minimum {
                    request = candidate; prompt = edited
                    applied.append(.init(reasoning: false, clearedTokens: cleared, clearedToolUses: removed))
                }
            }
        }
        guard prompt.tokens.count <= prompt.contextLimit else { throw ChatRequestIssue("messages", "prompt is too long: prepared input exceeds the model context limit") }
        if request.reserveOutputContext, !request.cacheWarmOnly, request.maxTokens > prompt.contextLimit - prompt.tokens.count {
            throw ChatRequestIssue("max_tokens", "leaves insufficient context for the prepared input; reduce max_tokens or clear conversation content")
        }
        request.preparedPrompt = prompt
        return Self(request: request, prompt: prompt, applied: applied, originalInputTokens: originalCount)
    }
}

private extension ChatRequest {
    mutating func remapCacheBoundaries(messageIndex: Int, blockCount: (Int) -> Int) {
        guard let policy = prefixCachePolicy else { return }
        let boundaries = policy.boundaries.compactMap { boundary -> ChatCacheBoundary? in
            guard boundary.messageIndex == messageIndex else { return boundary }
            let count = blockCount(boundary.blockCount)
            return count > 0 ? .init(messageIndex: messageIndex, blockCount: count, ttlSeconds: boundary.ttlSeconds) : nil
        }
        prefixCachePolicy = .init(scope: policy.scope, boundaries: boundaries, wholePromptTTL: policy.wholePromptTTL,
                                  toolPrefixTTL: policy.toolPrefixTTL, toolBoundaries: policy.toolBoundaries)
    }
}
