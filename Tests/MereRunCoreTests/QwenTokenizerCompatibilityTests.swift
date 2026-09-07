import Foundation
import XCTest
@testable import MereRunCore

final class QwenTokenizerCompatibilityTests: MereRunCoreTestCase {

    private func normalizedTokenizerClass(from object: [String: Any]) throws -> String? {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        let config = try QwenTokenizer.normalizedTokenizerConfig(
            data: data,
            url: URL(fileURLWithPath: "/tmp/tokenizer_config.json"),
            overrideTokenizerClass: "Qwen2Tokenizer"
        )
        return config["tokenizer_class"].string()
    }

    func testTokenizersBackendIsRemappedToSupportedTokenizer() throws {
        let tokenizerClass = try normalizedTokenizerClass(from: [
            "tokenizer_class": "TokenizersBackend",
            "model_max_length": 262_144,
        ])
        XCTAssertEqual(tokenizerClass, "Qwen2Tokenizer")
    }

    func testSupportedTokenizerClassIsPreserved() throws {
        let tokenizerClass = try normalizedTokenizerClass(from: [
            "tokenizer_class": "Qwen2Tokenizer",
            "model_max_length": 262_144,
        ])
        XCTAssertEqual(tokenizerClass, "Qwen2Tokenizer")
    }

    func testMissingTokenizerClassDoesNotCrashNormalization() throws {
        let tokenizerClass = try normalizedTokenizerClass(from: [
            "model_max_length": 262_144,
        ])
        XCTAssertNil(tokenizerClass)
    }

    func testSiblingChatTemplateIsLoadedWhenConfigOmitsTemplate() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let configURL = temp.appendingPathComponent("tokenizer_config.json")
        let templateURL = temp.appendingPathComponent("chat_template.jinja")
        try TestFileSystem.writeFile(templateURL, contents: Data("hello {{ messages }}".utf8))

        let data = try JSONSerialization.data(withJSONObject: [
            "tokenizer_class": "TokenizersBackend",
            "model_max_length": 262_144,
        ], options: [])
        let config = try QwenTokenizer.normalizedTokenizerConfig(
            data: data,
            url: configURL,
            overrideTokenizerClass: "Qwen2Tokenizer"
        )

        XCTAssertEqual(config["chat_template"].string(), "hello {{ messages }}")
    }

    func testInstalledOrnithTemplatePreservesToolGroupsAndCompleteSchemaWhenAvailable() throws {
        guard let root = ProcessInfo.processInfo.environment["MERERUN_ORNITH_TOKENIZER_ROOT"] else {
            throw XCTSkip("Set MERERUN_ORNITH_TOKENIZER_ROOT to check the installed Ornith tokenizer without model weights.")
        }
        let schema: [String: OpenAIJSONValue] = [
            "type": .string("object"),
            "properties": .object([
                "action": .object(["type": .string("string"), "enum": .array([.string("navigate"), .string("screenshot")])]),
                "values": .object(["type": .string("array"), "items": .object(["type": .string("string")])]),
            ]),
            "required": .array([.string("action")]), "additionalProperties": .bool(false),
        ]
        let tool = ToolDefinition(name: "browser", description: "Inspect a page.", parameterSchema: schema)
        let template = try Q35TokenizerAndTemplate.load(from: URL(fileURLWithPath: root), maxLengthOverride: 8_192)
        let messages = [
            ChatMessage(role: .user, content: "Inspect the site."),
            ChatMessage(role: .assistant, content: "", reasoningContent: "Open both pages.", toolCalls: [
                ChatMessageToolCall(id: "first", name: "browser", arguments: ["action": .string("navigate")]),
                ChatMessageToolCall(id: "second", name: "browser", arguments: ["action": .string("navigate")]),
            ]),
            ChatMessage(role: .tool, content: "First page opened.", toolCallID: "first"),
            ChatMessage(role: .tool, content: "Second page opened.", toolCallID: "second"),
            ChatMessage(role: .assistant, content: "", reasoningContent: "Inspect the image.", toolCalls: [
                ChatMessageToolCall(id: "third", name: "browser", arguments: ["action": .string("screenshot")]),
            ]),
            ChatMessage(role: .tool, content: "Screenshot available.", toolCallID: "third"),
        ]
        let tokens = try template.encodeForGeneration(messages: messages, tools: [tool], maxLength: 8_192)
        let prompt = template.decode(tokens: tokens)
        XCTAssertEqual(prompt.components(separatedBy: "<|im_start|>user").count - 1, 3)
        XCTAssertTrue(prompt.contains("""
        <|im_start|>user
        <tool_response>
        First page opened.
        </tool_response>
        <tool_response>
        Second page opened.
        </tool_response><|im_end|>
        """))
        XCTAssertTrue(prompt.contains("<|im_start|>user\n<tool_response>\nScreenshot available."))
        XCTAssertTrue(prompt.contains("<think>\nInspect the image.\n</think>"))
        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n<think>\n"))
        let start = try XCTUnwrap(prompt.range(of: "<tools>\n")?.upperBound)
        let end = try XCTUnwrap(prompt.range(of: "\n</tools>", range: start..<prompt.endIndex)?.lowerBound)
        let declaration = try JSONDecoder().decode(OpenAIJSONValue.self, from: Data(prompt[start..<end].utf8))
        XCTAssertEqual(declaration.objectValue?["function"]?.objectValue?["parameters"], .object(schema))
    }
}
