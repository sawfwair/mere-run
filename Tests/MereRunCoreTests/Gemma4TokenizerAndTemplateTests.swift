import XCTest
@testable import MereRunCore

final class Gemma4TokenizerAndTemplateTests: MereRunCoreTestCase {
    private static let fixtureTemplateSHA256 =
        "e5173c5f6afb445d9ff38060e6a2b15c3323fdbe20bb72552dcdfbf410c74a11"

    func testChatMessageDecodesPayloadWithoutAgentMetadata() throws {
        let data = Data(#"{"role":"user","content":"hello"}"#.utf8)
        let message = try JSONDecoder().decode(ChatMessage.self, from: data)

        XCTAssertEqual(message, ChatMessage(role: .user, content: "hello"))
    }

    func testUpdatedUpstreamTemplateRendersToolResponse() async throws {
        guard let modelRoot = ProcessInfo.processInfo.environment["MERERUN_GEMMA4_TEMPLATE_TEST_ROOT"] else {
            throw XCTSkip("Set MERERUN_GEMMA4_TEMPLATE_TEST_ROOT to a tokenizer-only Gemma 4 model root.")
        }

        let tokenizer = try await Gemma4TokenizerAndTemplate.load(
            from: URL(fileURLWithPath: modelRoot, isDirectory: true)
        )
        let messages = [
            ChatMessage(role: .user, content: "Write the file."),
            ChatMessage(
                role: .assistant,
                content: "Working...",
                reasoningContent: "I should write the requested file.",
                toolCalls: [
                    ChatMessageToolCall(
                        id: "call_123",
                        name: "write_file",
                        arguments: [
                            "path": .string("note.txt"),
                            "overwrite": .bool(false),
                            "metadata": .object([
                                "source": .null,
                            ]),
                            "tags": .array([
                                .string("draft"),
                            ]),
                        ]
                    ),
                ]
            ),
            ChatMessage(
                role: .tool,
                content: "Wrote 4 bytes to note.txt",
                name: "write_file",
                toolCallID: "call_123"
            ),
        ]

        let tokens = try tokenizer.encodeForGeneration(
            messages: messages,
            addGenerationPrompt: true,
            includeThinking: true,
            maxLength: 4_096
        )
        let rendered = tokenizer.decode(tokens: tokens)

        XCTAssertTrue(rendered.contains("call:write_file"), rendered)
        XCTAssertTrue(rendered.contains("note.txt"), rendered)
        XCTAssertTrue(rendered.contains("overwrite"), rendered)
        XCTAssertTrue(rendered.contains("false"), rendered)
        XCTAssertTrue(rendered.contains("source"), rendered)
        XCTAssertTrue(rendered.contains("null"), rendered)
        XCTAssertTrue(rendered.contains("draft"), rendered)
        XCTAssertTrue(rendered.contains("response:write_file"), rendered)
        XCTAssertTrue(rendered.contains("Wrote 4 bytes to note.txt"), rendered)
        XCTAssertFalse(rendered.contains("response:unknown{value:null}"), rendered)
    }

    func testCanonicalTemplateOverlaySelectsStandardVariantFor12B() throws {
        let modelRoot = try makeTemplateModelRoot(hiddenSize: 3_840, numHiddenLayers: 48)
        defer { try? FileManager.default.removeItem(at: modelRoot) }

        let selection = try Gemma4CanonicalChatTemplate.override(
            for: modelRoot,
            recognizedStaleTemplateSHA256s: [Self.fixtureTemplateSHA256]
        )

        XCTAssertEqual(selection?.variant, .standard)
        XCTAssertTrue(selection?.template.contains("Google Gemma 4 Canonical Chat Template") == true)
        XCTAssertTrue(selection?.template.contains("tool_calls[].function.arguments") == true)
        XCTAssertTrue(selection?.template.contains("JSON object (mapping)") == true)
    }

    func testCanonicalTemplateOverlayKeepsE4BGenerationPrimerSeparate() throws {
        let modelRoot = try makeTemplateModelRoot(hiddenSize: 2_560, numHiddenLayers: 42)
        defer { try? FileManager.default.removeItem(at: modelRoot) }

        let selection = try Gemma4CanonicalChatTemplate.override(
            for: modelRoot,
            recognizedStaleTemplateSHA256s: [Self.fixtureTemplateSHA256]
        )

        XCTAssertEqual(selection?.variant, .e4b)
        XCTAssertFalse(selection?.template.contains("<|channel>thought\n<channel|>") == true)
    }

    func testCanonicalTemplateOverlayPreservesCustomTemplate() throws {
        let modelRoot = try makeTemplateModelRoot(hiddenSize: 3_840, numHiddenLayers: 48)
        defer { try? FileManager.default.removeItem(at: modelRoot) }

        let selection = try Gemma4CanonicalChatTemplate.override(for: modelRoot)

        XCTAssertNil(selection)
    }

    func testCanonicalTemplateOverlayRejectsUnknownModelProfile() throws {
        let modelRoot = try makeTemplateModelRoot(hiddenSize: 1_024, numHiddenLayers: 12)
        defer { try? FileManager.default.removeItem(at: modelRoot) }

        let selection = try Gemma4CanonicalChatTemplate.override(
            for: modelRoot,
            recognizedStaleTemplateSHA256s: [Self.fixtureTemplateSHA256]
        )

        XCTAssertNil(selection)
    }

    func testRenderMessagesPreservesOpenAIToolResponseFields() {
        let messages = [
            ChatMessage(role: .user, content: "Write the file."),
            ChatMessage(
                role: .tool,
                content: "Wrote 4 bytes to /tmp/note.txt",
                name: "write_file",
                toolCallID: "call_123"
            ),
        ]

        let rendered = Gemma4TokenizerAndTemplate.renderMessages(messages)
        let toolMessage = tryCastDictionary(rendered[1])

        XCTAssertEqual(toolMessage["role"] as? String, "tool")
        XCTAssertEqual(toolMessage["content"] as? String, "Wrote 4 bytes to /tmp/note.txt")
        XCTAssertEqual(toolMessage["name"] as? String, "write_file")
        XCTAssertEqual(toolMessage["tool_call_id"] as? String, "call_123")
    }

    func testRenderMessagesExtractsAssistantToolCalls() {
        let messages = [
            ChatMessage(
                role: .assistant,
                content: "Working...\n<|tool_call>call:write_file{path:<|\"|>note.txt<|\"|>,content:<|\"|>BLUE<|\"|>}<tool_call|>",
                reasoningContent: "I should write the requested file.",
                toolCalls: [
                    ChatMessageToolCall(
                        id: "call_123",
                        name: "write_file",
                        arguments: [
                            "path": .string("note.txt"),
                            "content": .string("BLUE"),
                        ]
                    ),
                ]
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
        XCTAssertEqual(assistantMessage["reasoning_content"] as? String, "I should write the requested file.")
        XCTAssertEqual(firstCall["id"] as? String, "call_123")
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

    private func makeTemplateModelRoot(hiddenSize: Int, numHiddenLayers: Int) throws -> URL {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-template-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let config = """
        {
          "text_config": {
            "hidden_size": \(hiddenSize),
            "num_hidden_layers": \(numHiddenLayers)
          }
        }
        """
        try Data(config.utf8).write(to: rootURL.appendingPathComponent("config.json"))
        try Data("stale fixture\n".utf8).write(to: rootURL.appendingPathComponent("chat_template.jinja"))
        return rootURL
    }
}
