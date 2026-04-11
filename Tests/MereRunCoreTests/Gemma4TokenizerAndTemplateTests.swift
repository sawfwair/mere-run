import XCTest
@testable import MereRunCore

final class Gemma4TokenizerAndTemplateTests: MereRunCoreTestCase {
    func testRenderMessagesEncodesToolResponsesForToolRole() {
        let messages = [
            ChatMessage(role: .user, content: "Write the file."),
            ChatMessage(role: .tool, content: "Wrote 4 bytes to /tmp/note.txt"),
        ]

        let rendered = Gemma4TokenizerAndTemplate.renderMessages(messages)
        let toolMessage = tryCastDictionary(rendered[1])
        let responses = tryCastArray(toolMessage["tool_responses"])
        let firstResponse = tryCastDictionary(responses[0])

        XCTAssertEqual(toolMessage["role"] as? String, "tool")
        XCTAssertNil(toolMessage["content"])
        XCTAssertEqual(firstResponse["name"] as? String, "tool")
        XCTAssertEqual(firstResponse["response"] as? String, "Wrote 4 bytes to /tmp/note.txt")
    }

    func testRenderMessagesExtractsAssistantToolCalls() {
        let messages = [
            ChatMessage(
                role: .assistant,
                content: "Working...\n<|tool_call>call:write_file{path:<|\"|>note.txt<|\"|>,content:<|\"|>BLUE<|\"|>}<tool_call|>"
            ),
        ]

        let rendered = Gemma4TokenizerAndTemplate.renderMessages(messages)
        let assistantMessage = tryCastDictionary(rendered[0])
        let toolCalls = tryCastArray(assistantMessage["tool_calls"])
        let firstCall = tryCastDictionary(toolCalls[0])
        let function = tryCastDictionary(firstCall["function"])
        let arguments = tryCastDictionary(function["arguments"])

        XCTAssertEqual(assistantMessage["role"] as? String, "assistant")
        XCTAssertEqual(assistantMessage["content"] as? String, "Working...")
        XCTAssertEqual(function["name"] as? String, "write_file")
        XCTAssertEqual(arguments["path"] as? String, "note.txt")
        XCTAssertEqual(arguments["content"] as? String, "BLUE")
    }

    private func tryCastDictionary(_ value: Any?) -> [String: Any] {
        guard let dict = value as? [String: Any] else {
            XCTFail("Expected dictionary, got \(String(describing: value))")
            return [:]
        }
        return dict
    }

    private func tryCastArray(_ value: Any?) -> [[String: Any]] {
        guard let array = value as? [[String: Any]] else {
            XCTFail("Expected array of dictionaries, got \(String(describing: value))")
            return []
        }
        return array
    }
}
