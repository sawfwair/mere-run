import XCTest
@testable import MereRunCore

final class LFM2ToolParserTests: MereRunCoreTestCase {
    func testParsesCheckpointToolSyntaxWithoutClosingStopToken() {
        let calls = LFM2ToolParser.parseToolCalls("<|tool_call_start|>[desktop_observe()]")
        XCTAssertEqual(calls, [ToolCall(name: "desktop_observe", arguments: [:])])
    }

    func testParsesMultipleCallsAndEscapedArguments() {
        let text = "<|tool_call_start|>[desktop_type(elementToken='token', text='line\\nnext'), desktop_key(key='s', modifiers=[\"cmd\"])]<|tool_call_end|>"
        XCTAssertEqual(LFM2ToolParser.parseToolCalls(text), [
            ToolCall(name: "desktop_type", arguments: ["elementToken": "token", "text": "line\nnext"]),
            ToolCall(name: "desktop_key", arguments: ["key": "s", "modifiers": "[\"cmd\"]"]),
        ])
    }

    func testRejectsMalformedOrUnmarkedText() {
        XCTAssertTrue(LFM2ToolParser.parseToolCalls("[desktop_observe()]").isEmpty)
        XCTAssertTrue(LFM2ToolParser.parseToolCalls("<|tool_call_start|>[desktop_type(text='unfinished)]").isEmpty)
    }

    func testRendersNativeToolHistory() throws {
        let rendered = try LFM2ToolParser.renderToolCalls([
            ChatMessageToolCall(name: "desktop_type", arguments: [
                "text": .string("line\nnext"), "x": .number(49),
            ]),
        ])
        XCTAssertEqual(
            rendered,
            "<|tool_call_start|>[desktop_type(text='line\\nnext', x=49)]<|tool_call_end|>"
        )
        XCTAssertEqual(
            LFM2ToolParser.parseToolCalls(rendered),
            [ToolCall(name: "desktop_type", arguments: ["text": "line\nnext", "x": "49"])]
        )
    }
}
