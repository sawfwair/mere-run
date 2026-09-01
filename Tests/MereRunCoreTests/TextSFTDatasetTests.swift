import Foundation
import MediaIO
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

    func testPreparesDatasetRelativeVLMImagesWithContentProvenance() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let imageDirectory = directory.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: imageDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let imageURL = imageDirectory.appendingPathComponent("frame.png")
        try MediaImageIO.writePNG(
            try MediaImage(width: 2, height: 2, rgba8: Array(repeating: 64, count: 16)),
            to: imageURL
        )
        let datasetURL = directory.appendingPathComponent("pairs.jsonl")
        try writeVLMExample(imageReference: "images/frame.png", to: datasetURL)

        let first = try TextSFTDataset.loadForTraining(
            from: datasetURL,
            mediaPolicy: .requireSingleLocalImage
        )

        XCTAssertEqual(first.summary.imageReferenceCount, 1)
        XCTAssertEqual(first.summary.uniqueImageCount, 1)
        XCTAssertEqual(first.summary.imageFingerprint?.count, 64)
        XCTAssertEqual(first.examples[0].messages[1].imageUrl, imageURL.path)

        try MediaImageIO.writePNG(
            try MediaImage(width: 2, height: 2, rgba8: Array(repeating: 192, count: 16)),
            to: imageURL
        )
        let second = try TextSFTDataset.loadForTraining(
            from: datasetURL,
            mediaPolicy: .requireSingleLocalImage
        )

        XCTAssertNotEqual(first.summary.imageFingerprint, second.summary.imageFingerprint)
        XCTAssertNotEqual(first.summary.fingerprint, second.summary.fingerprint)
    }

    func testVLMTrainingRejectsRemoteAbsoluteAndEscapingImages() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let datasetURL = directory.appendingPathComponent("pairs.jsonl")

        for reference in ["https://example.com/frame.png", "/tmp/frame.png", "../frame.png"] {
            try writeVLMExample(imageReference: reference, to: datasetURL)
            XCTAssertThrowsError(try TextSFTDataset.loadForTraining(
                from: datasetURL,
                mediaPolicy: .requireSingleLocalImage
            ))
        }
    }

    func testVLMTrainingRejectsImageSymlinks() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let imageDirectory = directory.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: imageDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let sourceURL = directory.appendingPathComponent("source.png")
        try MediaImageIO.writePNG(
            try MediaImage(width: 1, height: 1, rgba8: [255, 0, 0, 255]),
            to: sourceURL
        )
        try FileManager.default.createSymbolicLink(
            at: imageDirectory.appendingPathComponent("frame.png"),
            withDestinationURL: sourceURL
        )
        let datasetURL = directory.appendingPathComponent("pairs.jsonl")
        try writeVLMExample(imageReference: "images/frame.png", to: datasetURL)

        XCTAssertThrowsError(try TextSFTDataset.loadForTraining(
            from: datasetURL,
            mediaPolicy: .requireSingleLocalImage
        )) { error in
            XCTAssertTrue(String(describing: error).contains("imageSymlinkNotAllowed"))
        }
    }

    private func writeVLMExample(imageReference: String, to url: URL) throws {
        let example = TextSFTExample(
            id: "vlm-1",
            sources: ["test"],
            messages: [
                ChatMessage(role: .system, content: "Describe images precisely."),
                ChatMessage(
                    role: .user,
                    content: "What is visible in this image?",
                    imageUrl: imageReference
                ),
                ChatMessage(role: .assistant, content: "A simple test image is visible."),
            ]
        )
        let data = try JSONEncoder().encode(example)
        try Data(data + Data("\n".utf8)).write(to: url, options: .atomic)
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
