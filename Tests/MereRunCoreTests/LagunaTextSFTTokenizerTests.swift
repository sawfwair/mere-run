import XCTest
@testable import MereRunCore

final class LagunaTextSFTTokenizerTests: XCTestCase {
    func testRendererPreservesTypedToolCallsAndResults() throws {
        let rendered = LagunaTokenizerAndTemplate.renderMessages([
            ChatMessage(
                role: .assistant,
                content: "",
                toolCalls: [ChatMessageToolCall(
                    id: "call-1",
                    name: "create_status_summary",
                    arguments: ["project": .string("Example App")]
                )]
            ),
            ChatMessage(
                role: .tool,
                content: #"{"state":"ready"}"#,
                name: "create_status_summary",
                toolCallID: "call-1"
            ),
        ])

        let calls = try XCTUnwrap(rendered[0]["tool_calls"] as? [[String: any Sendable]])
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(rendered[1]["name"] as? String, "create_status_summary")
        XCTAssertEqual(rendered[1]["tool_call_id"] as? String, "call-1")
    }

    func testAssistantTargetTextTrimsBoundaryWhitespace() {
        XCTAssertEqual(
            LagunaTextSFTTokenizer.assistantTargetText("  A concise answer.\n"),
            "A concise answer."
        )
    }

    func testOfficialTokenizerBuildsAssistantOnlyTargetsWhenAvailable() async throws {
        guard let path = ProcessInfo.processInfo.environment[
            "MERERUN_LAGUNA_TOKENIZER_PATH"
        ] else {
            throw XCTSkip(
                "Set MERERUN_LAGUNA_TOKENIZER_PATH to run the official Laguna SFT tokenizer test."
            )
        }
        let tokenizer = try await LagunaTokenizerAndTemplate.load(
            from: URL(fileURLWithPath: path),
            maxLength: 256
        )
        let examples = try LagunaTextSFTTokenizer.tokenize(
            [
                TextSFTExample(
                    id: "official-tokenizer",
                    sources: ["test"],
                    messages: [
                        ChatMessage(role: .system, content: "You answer concisely."),
                        ChatMessage(role: .user, content: "Say ready now."),
                        ChatMessage(role: .assistant, content: "  ready\n"),
                    ]
                ),
            ],
            tokenizerAndTemplate: tokenizer,
            maxSequenceLength: 256
        )

        let example = try XCTUnwrap(examples.first)
        XCTAssertEqual(example.inputTokenIds.count, example.labelTokenIds.count)
        XCTAssertEqual(example.inputTokenIds.count, example.lossMask.count)
        XCTAssertTrue(example.lossMask.contains(0))
        XCTAssertTrue(example.lossMask.contains(1))
        XCTAssertEqual(
            example.labelTokenIds.last,
            tokenizer.assistantEndTokenID ?? tokenizer.eosTokenID
        )
    }
}
