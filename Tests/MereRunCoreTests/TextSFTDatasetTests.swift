import Foundation
import XCTest
@testable import MereRunCore

final class TextSFTDatasetTests: XCTestCase {
    func testLoadsChatJSONLDatasetAndSummarizes() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let datasetURL = directory.appendingPathComponent("pairs.jsonl")
        try """
        {"id":"local-assistant-0001","sources":["docs:local"],"messages":[{"role":"system","content":"You know Mere."},{"role":"user","content":"What is local mode?"},{"role":"assistant","content":"Local mode is explicit and project-specific."}]}
        {"id":"local-assistant-0002","sources":["docs:run"],"messages":[{"role":"system","content":"You know Mere."},{"role":"user","content":"Which model?"},{"role":"assistant","content":"Use text-chat-gemma4-12b-4bit for the local assistant lane."}]}
        """.write(to: datasetURL, atomically: true, encoding: .utf8)

        let examples = try TextSFTDataset.load(from: datasetURL)
        let summary = TextSFTDataset.summarize(examples)

        XCTAssertEqual(examples.count, 2)
        XCTAssertEqual(summary.exampleCount, 2)
        XCTAssertEqual(summary.messageCount, 6)
        XCTAssertEqual(summary.sourceCount, 2)
        XCTAssertEqual(summary.fingerprint.count, 64)
    }

    func testRejectsDuplicatePrompts() throws {
        let examples = [
            TextSFTExample(
                id: "one",
                sources: ["docs"],
                messages: [
                    ChatMessage(role: .system, content: "System"),
                    ChatMessage(role: .user, content: "Same prompt"),
                    ChatMessage(role: .assistant, content: "First answer"),
                ]
            ),
            TextSFTExample(
                id: "two",
                sources: ["docs"],
                messages: [
                    ChatMessage(role: .system, content: "System"),
                    ChatMessage(role: .user, content: " same PROMPT "),
                    ChatMessage(role: .assistant, content: "Second answer"),
                ]
            ),
        ]

        XCTAssertThrowsError(try TextSFTDataset.validate(examples)) { error in
            XCTAssertTrue(String(describing: error).contains("duplicatePrompt"))
        }
    }

    func testAllowsSameQuestionAtDifferentToolLoopStates() throws {
        let tool = Self.summaryTool()
        let initial = TextSFTExample(
            id: "initial",
            sources: ["test"],
            messages: [
                ChatMessage(role: .system, content: "Use available tools."),
                ChatMessage(role: .user, content: "Summarize the project status."),
                ChatMessage(
                    role: .assistant,
                    content: "",
                    toolCalls: [Self.summaryCall(id: "call-1")]
                ),
            ],
            tools: [tool]
        )
        let continued = TextSFTExample(
            id: "continued",
            sources: ["test"],
            messages: [
                ChatMessage(role: .system, content: "Use available tools."),
                ChatMessage(role: .user, content: "Summarize the project status."),
                ChatMessage(
                    role: .assistant,
                    content: "",
                    toolCalls: [Self.summaryCall(id: "call-1")]
                ),
                ChatMessage(
                    role: .tool,
                    content: #"{"state":"ready"}"#,
                    name: "create_status_summary",
                    toolCallID: "call-1"
                ),
                ChatMessage(role: .assistant, content: "The project is ready."),
            ],
            tools: [tool]
        )

        XCTAssertNoThrow(try TextSFTDataset.validate([initial, continued]))
    }

    func testRejectsDuplicateCompleteControllerState() throws {
        let tool = Self.summaryTool()
        let messages = [
            ChatMessage(role: .system, content: "Use available tools."),
            ChatMessage(role: .user, content: "Summarize the project status."),
            ChatMessage(
                role: .assistant,
                content: "",
                toolCalls: [Self.summaryCall(id: "call-1")]
            ),
        ]
        let examples = [
            TextSFTExample(id: "one", sources: ["test"], messages: messages, tools: [tool]),
            TextSFTExample(id: "two", sources: ["test"], messages: messages, tools: [tool]),
        ]

        XCTAssertThrowsError(try TextSFTDataset.validate(examples)) { error in
            XCTAssertTrue(String(describing: error).contains("duplicatePrompt"))
        }
    }

    func testFingerprintIncludesTypedCallsAndCompleteToolSchema() {
        let base = TextSFTExample(
            id: "one",
            sources: ["test"],
            messages: [
                ChatMessage(role: .system, content: "Use available tools."),
                ChatMessage(role: .user, content: "Summarize the project status."),
                ChatMessage(
                    role: .assistant,
                    content: "",
                    toolCalls: [Self.summaryCall(id: "call-1")]
                ),
            ],
            tools: [Self.summaryTool()]
        )
        let changedCall = TextSFTExample(
            id: base.id,
            sources: base.sources,
            messages: [
                ChatMessage(role: .system, content: "Use available tools."),
                ChatMessage(role: .user, content: "Summarize the project status."),
                ChatMessage(
                    role: .assistant,
                    content: "",
                    toolCalls: [ChatMessageToolCall(
                        id: "call-1",
                        name: "create_status_summary",
                        arguments: ["project": .string("Example API")]
                    )]
                ),
            ],
            tools: base.tools
        )
        let changedSchema = TextSFTExample(
            id: base.id,
            sources: base.sources,
            messages: base.messages,
            tools: [ToolDefinition(
                name: "create_status_summary",
                description: "Create a concise status report.",
                parameters: [
                    "project": ToolParameterProperty(
                        type: "string",
                        description: "Project identifier"
                    ),
                    "includeRisks": ToolParameterProperty(
                        type: "boolean",
                        description: "Include known risks"
                    ),
                ],
                required: ["project", "includeRisks"]
            )]
        )

        let fingerprint = TextSFTDataset.fingerprint([base])
        XCTAssertNotEqual(fingerprint, TextSFTDataset.fingerprint([changedCall]))
        XCTAssertNotEqual(fingerprint, TextSFTDataset.fingerprint([changedSchema]))
    }

    func testFingerprintPreservesRenderedToolOrderWhileDuplicatesCanonicalizeIt() throws {
        let firstTool = Self.summaryTool()
        let secondTool = ToolDefinition(
            name: "record_status_note",
            description: "Record a status note.",
            parameters: [
                "note": ToolParameterProperty(type: "string", description: "Status note"),
            ],
            required: ["note"]
        )
        let messages = [
            ChatMessage(role: .user, content: "Summarize the project status."),
            ChatMessage(role: .assistant, content: "The project is ready."),
        ]
        let forward = TextSFTExample(
            id: "forward",
            sources: ["test"],
            messages: messages,
            tools: [firstTool, secondTool]
        )
        let reversed = TextSFTExample(
            id: "reversed",
            sources: ["test"],
            messages: messages,
            tools: [secondTool, firstTool]
        )

        XCTAssertNotEqual(
            TextSFTDataset.fingerprint([forward]),
            TextSFTDataset.fingerprint([reversed])
        )
        XCTAssertThrowsError(try TextSFTDataset.validate([forward, reversed]))
    }

    func testDecodesLegacyRowsWithoutTools() throws {
        let row = #"{"id":"legacy","sources":["test"],"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Say ready now."},{"role":"assistant","content":"Ready now."}]}"#
        let example = try JSONDecoder().decode(TextSFTExample.self, from: Data(row.utf8))

        XCTAssertNil(example.tools)
    }

    private static func summaryTool() -> ToolDefinition {
        ToolDefinition(
            name: "create_status_summary",
            description: "Create a status summary.",
            parameters: [
                "project": ToolParameterProperty(
                    type: "string",
                    description: "Project identifier"
                ),
            ],
            required: ["project"]
        )
    }

    private static func summaryCall(id: String) -> ChatMessageToolCall {
        ChatMessageToolCall(
            id: id,
            name: "create_status_summary",
            arguments: ["project": .string("Example App")]
        )
    }
}
