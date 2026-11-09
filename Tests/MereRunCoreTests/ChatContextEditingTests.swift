import Foundation
import XCTest
@testable import MereRunCore

final class ChatContextEditingTests: XCTestCase {
    private func tokenize(_ request: ChatRequest) throws -> ChatPreparedPrompt {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return try ChatPreparedPrompt(tokens: encoder.encode(request.messages).map(Int.init), contextLimit: 10000)
    }

    func testThinkingClearingPreservesCacheMarkerContentBoundaries() async throws {
        var request = ChatRequest(messages: [.init(role: .user, content: "first"),
            .init(role: .assistant, blocks: [.reasoning(text: "old reasoning", continuity: nil), .text("cached"), .text("after marker")]),
            .init(role: .user, content: "next")])
        request.contextEdits = [.reasoning(keepTurns: 1)]
        request.tools = [ToolDefinition(name: "read", description: "", parameters: [:])]
        request.prefixCachePolicy = .init(scope: "scope", boundaries: [
            .init(messageIndex: 1, blockCount: 1, ttlSeconds: 3600),
            .init(messageIndex: 1, blockCount: 2, ttlSeconds: 300),
            .init(messageIndex: 1, blockCount: 3, ttlSeconds: 300)], wholePromptTTL: 300,
            toolBoundaries: [.init(throughTool: "read", ttlSeconds: 3600)])
        let prepared = try await ChatPromptPreparation.prepare(request, tokenize: tokenize)
        XCTAssertEqual(prepared.request.prefixCachePolicy?.boundaries.map(\.blockCount), [1, 2])
        XCTAssertEqual(prepared.request.messages[1].contentBlocks, [.text("cached"), .text("after marker")])
        XCTAssertEqual(request.prefixCachePolicy?.boundaries.map(\.blockCount), [1, 2, 3])
        XCTAssertEqual(prepared.request.prefixCachePolicy?.toolBoundaries, request.prefixCachePolicy?.toolBoundaries)
        XCTAssertEqual(prepared.request.prefixCachePolicy?.wholePromptTTL, 300)
    }

    func testContinuationDoesNotClearOrReportTheSameContentTwice() async throws {
        var request = ChatRequest(messages: [.init(role: .user, content: "first"),
            .init(role: .assistant, blocks: [.reasoning(text: "old reasoning", continuity: nil),
                .toolUse(.init(id: "a", name: "read", arguments: ["path": .string("old")]))]),
            .init(role: .tool, blocks: [.text(String(repeating: "old result ", count: 20)), .text("tail")]),
            .init(role: .user, content: "next")])
        request.messages[2].name = "read"; request.messages[2].toolCallID = "a"
        request.prefixCachePolicy = .init(scope: "scope", boundaries: [.init(messageIndex: 2, blockCount: 2, ttlSeconds: 300)], wholePromptTTL: nil)
        request.contextEdits = [.reasoning(keepTurns: 1), .toolResults(triggerTokens: 0, keepUses: 0, minimumTokens: 0, excludeTools: [], clearInputs: true)]
        let first = try await ChatPromptPreparation.prepare(request, tokenize: tokenize)
        let next = try await ChatPromptPreparation.prepare(first.request, tokenize: tokenize)
        XCTAssertEqual(first.applied.count, 2)
        XCTAssertTrue(next.applied.isEmpty)
        XCTAssertEqual(first.prompt, next.prompt)
        XCTAssertEqual(first.request.prefixCachePolicy?.boundaries.first?.blockCount, 1)
        XCTAssertEqual(first.request.messages[1].toolCalls?.first?.id, "a")
        XCTAssertEqual(first.request.messages[2].toolCallID, "a")
        let restored = try JSONDecoder().decode([ChatMessage].self, from: JSONEncoder().encode(first.request.messages))
        XCTAssertEqual(restored[2].toolResultCleared, true)
        let legacy = try JSONDecoder().decode(ChatMessage.self, from: Data(#"{"role":"tool","content":"legacy","toolCallID":"a"}"#.utf8))
        XCTAssertNil(legacy.toolResultCleared)
    }

    func testClearingKeepsIDsRecentResultsAndExclusions() async throws {
        let history: [ChatMessage] = [
            .init(role: .user, content: "start"),
            .init(role: .assistant, content: "", reasoningContent: "old thinking", toolCalls: [.init(id: "a", name: "read", arguments: ["path": .string("old")])]),
            .init(role: .tool, content: String(repeating: "old result ", count: 20), name: "read", toolCallID: "a"),
            .init(role: .assistant, content: "", toolCalls: [.init(id: "b", name: "memory", arguments: [:])]),
            .init(role: .tool, content: "keep memory", name: "memory", toolCallID: "b"),
            .init(role: .assistant, content: "", toolCalls: [.init(id: "c", name: "read", arguments: [:])]),
            .init(role: .tool, content: "recent result", name: "read", toolCallID: "c"),
            .init(role: .user, content: "next")]
        var request = ChatRequest(messages: history)
        request.contextEdits = [.reasoning(keepTurns: 1), .toolResults(triggerTokens: 0, keepUses: 1, minimumTokens: 1, excludeTools: ["memory"], clearInputs: true)]
        let result = try await ChatPromptPreparation.prepare(request) { request in
            let text = request.messages.map { $0.content + ($0.reasoningContent ?? "") }.joined()
            return try ChatPreparedPrompt(tokens: text.utf8.map(Int.init) + [0], contextLimit: 10000)
        }
        XCTAssertNil(result.request.messages[1].reasoningContent)
        XCTAssertEqual(result.request.messages[2].toolCallID, "a")
        XCTAssertEqual(result.request.messages[2].content, "[Tool result cleared]")
        XCTAssertEqual(result.request.messages[4].content, "keep memory")
        XCTAssertEqual(result.request.messages[6].content, "recent result")
        XCTAssertEqual(result.request.messages[1].toolCalls?.first?.arguments, [:])
        XCTAssertLessThan(result.prompt.tokens.count, result.originalInputTokens)
        XCTAssertEqual(result.applied.count, 2)
        XCTAssertEqual(request.messages, history)
    }
    func testMinimumClearingAndTriggerPreventUnwantedEdits() async throws {
        var request = ChatRequest(messages: [.init(role: .tool, content: "small", name: "x", toolCallID: "a")])
        request.contextEdits = [.toolResults(triggerTokens: 0, keepUses: 0, minimumTokens: 1000, excludeTools: [], clearInputs: false)]
        let result = try await ChatPromptPreparation.prepare(request) { request in
            try ChatPreparedPrompt(tokens: Array(repeating: 1, count: request.messages[0].content.count), contextLimit: 10000)
        }
        XCTAssertTrue(result.applied.isEmpty)
        XCTAssertEqual(result.request.messages, request.messages)
    }
}
