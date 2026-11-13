import Foundation
import Jinja
import XCTest
@testable import MereRunCore

final class Inkling2IntegrationTests: XCTestCase {
    func testManagedCheckpointAndNativeServingRouting() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Inkling2Resources.modelID))
        XCTAssertEqual(spec.validationKind, .inkling2)
        XCTAssertEqual(spec.upstreamRepoId, "thinkingmachines/Inkling-2-Small")
        XCTAssertEqual(spec.hubFallback?.revision, Inkling2Resources.revision)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatInkling2)
        XCTAssertEqual(spec.apiProfile?.inputModalities, [.text, .image])
        XCTAssertEqual(spec.apiProfile?.reasoningEffortStrengths[.off], 0)
        XCTAssertFalse(spec.apiProfile?.structuredOutput ?? true)
        for name in [Inkling2Resources.modelID, Inkling2Resources.repository] {
            XCTAssertEqual(NativeChatRuntime.commandFamily(modelID: name), .inkling2)
            guard case .textChatInkling2 = try NativeChatRuntime.command(modelID: name, modelPath: nil) else {
                return XCTFail("Expected native Inkling 2 runtime")
            }
        }
        let manifest = MereRunModelManifest.template(for: .inkling2Small)
        XCTAssertEqual(manifest.engine, .inkling2)
        XCTAssertEqual(manifest.sources?.first?.revision, Inkling2Resources.revision)
        XCTAssertTrue(manifest.supports?.contains(.visionChat) == true)
    }

    private func render(_ messages: [ChatMessage], effort: Double = 0.9, tools: [ToolDefinition]? = nil) throws -> String {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Inkling2/chat_template.jinja")
        return try Inkling2ChatTemplate(source: String(contentsOf: url, encoding: .utf8))
            .render(messages: messages, tools: tools, effort: effort)
    }

    func testReleasedTemplateHasNoAssistantSuffixAndPreservesEffort() throws {
        XCTAssertEqual(try render([ChatMessage(role: .user, content: "Hello")], effort: 0.2),
            "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"
            + "<|message_user|><|content_text|>Hello<|end_message|>")
    }

    func testReleasedTemplatePreservesTypedToolArgumentsAndImageOrder() throws {
        let call = ChatMessageToolCall(id: "c1", name: "read", arguments: ["path": .string("a.txt")])
        let rendered = try render([
            ChatMessage(role: .assistant, content: "", reasoningContent: "Inspect first.", toolCalls: [call]),
            ChatMessage(role: .tool, content: "data", toolCallID: "c1"),
            ChatMessage(role: .user, content: "Describe", imageUrl: "/image.png"),
        ])
        XCTAssertTrue(rendered.contains("<|message_model|><|content_thinking|>Inspect first.<|end_message|>"))
        XCTAssertTrue(rendered.contains("<|message_model|>read<|content_invoke_tool_json|>{\"name\":\"read\",\"args\":{\"path\":\"a.txt\"}}<|end_message|>"))
        XCTAssertTrue(rendered.contains("<|message_tool|>read<|content_text|>data<|end_message|>"))
        XCTAssertTrue(rendered.hasSuffix("<|message_user|><|content_image|><|unused_200054|><|end_message|>"
            + "<|message_user|><|content_text|>Describe<|end_message|>"))
        XCTAssertFalse(rendered.contains("<|content_text|><|end_message|>"))
    }

    func testToolDeclarationUsesCompactPythonJSONWithoutHTMLEscaping() throws {
        let tools = [ToolDefinition(name: "read", description: "été <&>", parameters: [:])]
        let prompt = try render([ChatMessage(role: .user, content: "Read")], tools: tools)
        XCTAssertTrue(prompt.hasPrefix("<|message_system|>tool_declare<|content_xml|>"))
        XCTAssertTrue(prompt.contains(#""description":"été <&>""#))
        XCTAssertTrue(prompt.contains(#""name":"read","parameters":{"properties":{},"required":[],"type":"object"},"type":"function""#))
        XCTAssertFalse(prompt.contains(#"\u00e9"#))
    }

    func testToolLikeTextInReasoningIsNeverExecutedAndCallsFollowPolicy() {
        let tool = ToolDefinition(name: "read", description: "Read", parameters: ["path": .init(type: "string", description: "Path")], required: ["path"])
        let invocation = #"<|content_invoke_tool_json|>{"name":"read","args":{"path":"a.txt"}}"#
        let text = "<|message_model|><|content_thinking|>Example: " + invocation + "<|end_message|>"
            + "<|message_model|>read" + invocation + "<|end_message|>"
            + "<|message_model|>read" + invocation + "<|end_message|>"
            + "<|message_model|><|content_text|>Done.<|end_message|>"
        let parsed = Inkling2OutputParser.parse(text, tools: [tool], parallel: false)
        XCTAssertEqual(parsed.visible, "Done.")
        XCTAssertEqual(parsed.toolCalls.count, 1)
        XCTAssertTrue(parsed.reasoning?.contains(invocation) == true)
        XCTAssertTrue(Inkling2OutputParser.parse(text).toolCalls.isEmpty)
    }

    func testStopSequencesMatchBeforeVisibleWhitespaceIsTrimmed() {
        let parsed = Inkling2OutputParser.parse("<|message_model|><|content_text|>Answer END <|end_message|>")
        let stopped = TextGenerationStopSequences.trimming(parsed.visible, sequences: ["END "])
        XCTAssertEqual(stopped.text, "Answer")
        XCTAssertEqual(stopped.matchedSequence, "END ")
    }

    func testUnsupportedControlsFailBeforeCheckpointLoading() async throws {
        let generator = Inkling2Generator()
        var request = ChatRequest(messages: [ChatMessage(role: .user, content: "hello")])
        request.kvCacheMode = .affine4
        do {
            _ = try await generator.chat(request, modelPath: "/missing", progressHandler: nil)
            XCTFail("Expected refusal")
        } catch { XCTAssertEqual((error as? ChatRequestIssue)?.field, "request") }
    }
}
