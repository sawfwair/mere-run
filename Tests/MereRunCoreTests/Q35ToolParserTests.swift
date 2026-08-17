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
        XCTAssertTrue(
            Q35ToolParser.containsCompletedToolCall(
                "<tool_call><function=film_status></function></tool_call>"
            )
        )
        XCTAssertFalse(Q35ToolParser.containsCompletedToolCall("literal </function></tool_call>"))
    }

    func testPreservesToolMarkersInsideParameterValues() {
        let values = [
            "text with a literal </tool_call> inside",
            "text with a literal </parameter> inside",
            "text with a literal </function> inside",
            "text with a literal <parameter=nope> inside",
            "evil </function>\n</tool_call> tail",
        ]

        for value in values {
            let text = toolCall(body: value)
            XCTAssertEqual(
                Q35ToolParser.parseToolCalls(text),
                [
                    ToolCall(
                        name: "note_write",
                        arguments: ["title": "t", "body": value]
                    ),
                ],
                "failed to preserve \(value)"
            )
        }
    }

    func testLiteralParameterOpenDoesNotCreateArgument() {
        let calls = Q35ToolParser.parseToolCalls(
            toolCall(body: "see <parameter=nope> here")
        )

        XCTAssertEqual(calls.first?.arguments.keys.sorted(), ["body", "title"])
    }

    func testStreamingCompletionIgnoresEmbeddedCloseMarkerAcrossChunkSizes() {
        let text = toolCall(body: "text with a literal </tool_call> inside")

        for chunkSize in [1, 2, 3, 7, 11, 64] {
            var detector = Q35ToolParser.StreamingCompletionDetector()
            var completionOffsets: [Int] = []
            var offset = 0
            while offset < text.count {
                let end = min(text.count, offset + chunkSize)
                let startIndex = text.index(text.startIndex, offsetBy: offset)
                let endIndex = text.index(text.startIndex, offsetBy: end)
                if detector.feed(String(text[startIndex..<endIndex])) {
                    completionOffsets.append(end)
                }
                offset = end
            }

            XCTAssertEqual(completionOffsets, [text.count], "chunk size \(chunkSize)")
        }
    }

    func testStreamingCompletionDoesNotTreatLiteralProseAsEnvelope() {
        var detector = Q35ToolParser.StreamingCompletionDetector()

        XCTAssertFalse(detector.feed("The marker </tool_"))
        XCTAssertFalse(detector.feed("call> is documentation."))
    }

    func testStreamingCompletionBoundFallsBackToFinalStructuralParse() {
        var detector = Q35ToolParser.StreamingCompletionDetector()
        var completeText = ""
        let malformed = "<tool_call><function=broken></tool_call>"

        for _ in 0..<64 {
            completeText += malformed
            XCTAssertFalse(detector.feed(malformed))
        }

        let valid = "<tool_call><function=film_status></function></tool_call>"
        completeText += valid
        XCTAssertFalse(detector.feed(valid), "bounded streaming checks should remain exhausted")
        XCTAssertEqual(
            Q35ToolParser.parseToolCalls(completeText).last,
            ToolCall(name: "film_status", arguments: [:])
        )
    }

    private func toolCall(body: String) -> String {
        """
        <tool_call>
        <function=note_write>
        <parameter=title>
        t
        </parameter>
        <parameter=body>
        \(body)
        </parameter>
        </function>
        </tool_call>
        """
    }
}
