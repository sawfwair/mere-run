import Foundation
import XCTest
@testable import MereRunCore

final class ChatContextEditingTests: XCTestCase {
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
