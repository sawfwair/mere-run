import XCTest
@testable import MereRunCore

final class Q35ToolParserTests: XCTestCase {
    func testParsesCheckpointToolCallProtocolWithoutArguments() {
        let text = """
        <tool_call>
        <function=film_status>
        </function>
        </tool_call>
        """

        XCTAssertEqual(
            Q35ToolParser.parseToolCalls(text),
            [ToolCall(name: "film_status", arguments: [:])]
        )
    }

    func testParsesMultilineAndTypedArgumentPayloadsWithoutFlatteningThem() {
        let text = """
        I will record the confirmed requirements.
        <tool_call>
        <function=film_update_brief>
        <parameter=audience>
        local-AI filmmakers
        </parameter>
        <parameter=references>
        ["tactile paper craft","stop motion"]
        </parameter>
        <parameter=generateScore>
        false
        </parameter>
        </function>
        </tool_call>
        """

        XCTAssertEqual(
            Q35ToolParser.parseToolCalls(text),
            [
                ToolCall(
                    name: "film_update_brief",
                    arguments: [
                        "audience": "local-AI filmmakers",
                        "references": "[\"tactile paper craft\",\"stop motion\"]",
                        "generateScore": "false",
                    ]
                ),
            ]
        )
    }

    func testParsesMultipleToolCallsAndIgnoresMalformedBlocks() {
        let text = """
        <tool_call><function=film_status></function></tool_call>
        <tool_call><function=></function></tool_call>
        <tool_call><function=film_recover></function></tool_call>
        """

        XCTAssertEqual(
            Q35ToolParser.parseToolCalls(text).map(\.name),
            ["film_status", "film_recover"]
        )
    }

    func testCompletionDetectionRequiresClosingCheckpointTag() {
        XCTAssertFalse(Q35ToolParser.containsCompletedToolCall("<tool_call><function=film_status>"))
        XCTAssertTrue(Q35ToolParser.containsCompletedToolCall("</function></tool_call>"))
    }
}
