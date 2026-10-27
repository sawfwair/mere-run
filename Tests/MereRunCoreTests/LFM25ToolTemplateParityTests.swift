import Foundation
import XCTest
@testable import MereRunCore

struct LFM25ToolTemplateFixture: Decodable {
    enum ID: String, Decodable { case declaration, toolResult = "tool-result", laterUser = "later-user" }
    let id: ID
    let renderedPrompt: String
    let tokens: [Int]
    enum CodingKeys: String, CodingKey { case id, tokens; case renderedPrompt = "rendered_prompt" }

    static func load() throws -> [Self] {
        let fixtures = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        return try JSONDecoder().decode(
            [Self].self, from: Data(contentsOf: fixtures.appending(path: "LFM25A2B24B/tool-template-parity.json"))
        )
    }

    var messages: [ChatMessage] {
        let user = ChatMessage(role: .user, content: "Compute 17 plus 25.")
        let active = [
            user,
            ChatMessage(
                role: .assistant, content: "", reasoningContent: "I will use calculate.",
                toolCalls: [.init(name: "calculate", arguments: ["expression": .string("17 + 25")])]
            ),
            ChatMessage(role: .tool, content: "42", name: "calculate", toolCallID: "call_0"),
        ]
        switch id {
        case .declaration: return [user]
        case .toolResult: return active
        case .laterUser: return active + [ChatMessage(role: .user, content: "Give the result.")]
        }
    }

    static let tools = [ToolDefinition(
        name: "calculate", description: "Evaluate an arithmetic expression.",
        parameters: ["expression": .init(type: "string", description: "Expression to evaluate.")],
        required: ["expression"]
    )]
}

final class LFM25ToolTemplateParityTests: MereRunCoreTestCase {
    func testToolDeclarationAndRoundHistoryMatchPinnedUpstreamTemplate() throws {
        for fixture in try LFM25ToolTemplateFixture.load() {
            for visibility in [true, false] {
                let prompt = try LFM2TokenizerAndTemplate.renderPrompt(
                    messages: fixture.messages, tools: LFM25ToolTemplateFixture.tools,
                    includeThinking: visibility, generationPromptSuffix: "<think>"
                )
                XCTAssertEqual(prompt, fixture.renderedPrompt, fixture.id.rawValue)
            }
        }
    }

    func testPythonScalarArgumentsAndNestedJSONRemainDistinctFromQuotedStrings() throws {
        let rendered = try LFM2ToolParser.renderToolCalls([.init(name: "record", arguments: [
            "enabled": .bool(true), "missing": .null, "text": .string("True"),
            "values": .array([.number(1), .bool(false)]),
        ])])
        XCTAssertEqual(rendered,
            "<|tool_call_start|>[record(enabled=True, missing=None, text='True', values=[1, false])]<|tool_call_end|>")
        let parsed = try XCTUnwrap(LFM2ToolParser.parseToolCalls(rendered).first)
        XCTAssertEqual(parsed.arguments["enabled"], "true")
        XCTAssertEqual(parsed.arguments["missing"], "null")
        XCTAssertEqual(parsed.arguments["text"], "True")
        XCTAssertEqual(parsed.arguments["values"], "[1, false]")
    }
}
