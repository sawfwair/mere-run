import Jinja
import XCTest
@testable import MereRunCore

final class ChatTemplateLoopContextTests: XCTestCase {
    func testQwenToolEncodingGuidancePreservesConversationAndAppliesOnlyWithTools() throws {
        let messages = [ChatMessage(role: .system, content: "Original policy."),
                        ChatMessage(role: .user, content: "Edit the file."),
                        ChatMessage(role: .system, content: "Later policy.")]
        let adapted = Q35SystemPrompt.messagesWithToolEncoding(messages)
        XCTAssertEqual(adapted.dropLast().map(\.content), messages.map(\.content))
        XCTAssertEqual(adapted.dropLast().map(\.role), messages.map(\.role))
        XCTAssertEqual(adapted.last?.role, .system)
        XCTAssertTrue(adapted.last?.content.contains("lowercase true or false") == true)
        let withoutSystem = Q35SystemPrompt.messagesWithToolEncoding(Array(messages.dropFirst()))
        XCTAssertEqual(withoutSystem.last?.role, .system)
        XCTAssertEqual(withoutSystem.dropLast().map(\.content), messages.dropFirst().map(\.content))
        for hasTools in [false, true] {
            _ = try Q35SystemPrompt.encode(messages: Array(messages.prefix(2)), addGenerationPrompt: true, hasTools: hasTools,
                render: { prepared, _ in
                    XCTAssertFalse(prepared.contains { $0.content == Q35SystemPrompt.toolParameterEncoding })
                    return [1]
                }, encodeText: { text in
                    XCTAssertTrue(hasTools)
                    XCTAssertTrue(text.contains(Q35SystemPrompt.toolParameterEncoding))
                    return []
                })
        }
    }

    func testQwenPreservesLeadingAndMidConversationSystemTurns() throws {
        let messages = [
            ChatMessage(role: .system, content: "First policy."),
            ChatMessage(role: .system, content: "Second policy."),
            ChatMessage(role: .user, content: "Inspect."),
            ChatMessage(role: .system, content: "New instruction."),
            ChatMessage(role: .assistant, content: "Observed."),
            ChatMessage(role: .system, content: "Final instruction."),
        ]
        let tokens = try Q35SystemPrompt.encode(messages: messages, addGenerationPrompt: true,
            render: { messages, generationPrompt in
                XCTAssertFalse(messages.dropFirst().contains { $0.role == .system })
                let text = messages.map { "<|im_start|>\($0.role.rawValue)\n\($0.content)<|im_end|>\n" }.joined()
                    + (generationPrompt ? "<|im_start|>assistant\n" : "")
                return text.utf8.map(Int.init)
            }, encodeText: { $0.utf8.map(Int.init) })
        XCTAssertEqual(String(decoding: tokens.map(UInt8.init), as: UTF8.self), """
        <|im_start|>system
        First policy.
        Second policy.<|im_end|>
        <|im_start|>user
        Inspect.<|im_end|>
        <|im_start|>system
        New instruction.<|im_end|>
        <|im_start|>assistant
        Observed.<|im_end|>
        <|im_start|>system
        Final instruction.<|im_end|>
        <|im_start|>assistant

        """)
    }

    func testQwenRejectsTemplateWithUnstableSystemBoundary() {
        XCTAssertThrowsError(try Q35SystemPrompt.encode(messages: [
            ChatMessage(role: .user, content: "Inspect."),
            ChatMessage(role: .system, content: "Policy."),
        ], addGenerationPrompt: true, render: { _, generationPrompt in
            generationPrompt ? [1, 2] : [3]
        }, encodeText: { $0.utf8.map(Int.init) })) { error in
            XCTAssertEqual((error as? ChatRequestIssue)?.field, "messages")
        }
    }

    // These neighbor checks are the tool-result grouping contract used by Ornith 1.5.
    private let templateSource = #"""
    {%- for message in messages %}
        {%- if message.role == "tool" %}
            {%- if loop.previtem and loop.previtem.role != "tool" %}
                {{- '<|im_start|>user' }}
            {%- endif %}
            {{- '\n<tool_response>\n' + message.content + '\n</tool_response>' }}
            {%- if not loop.last and loop.nextitem.role != "tool" %}
                {{- '<|im_end|>\n' }}
            {%- elif loop.last %}
                {{- '<|im_end|>\n' }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
        {%- endif %}
    {%- endfor %}
    {{- '<|im_start|>assistant\n' }}
    """#

    func testSingleAndConsecutiveToolResultsRetainRoleBoundaries() throws {
        let messages = [
            ChatMessage(role: .user, content: "Inspect the site."),
            ChatMessage(role: .assistant, content: "Read two files."),
            ChatMessage(role: .tool, content: "First file."),
            ChatMessage(role: .tool, content: "Second file."),
            ChatMessage(role: .assistant, content: "Open the page."),
            ChatMessage(role: .tool, content: "Page opened."),
        ]
        let rendered = try Template(templateSource).render([
            "messages": Value(any: Q35TokenizerAndTemplate.renderMessages(messages)),
        ])
        XCTAssertEqual(rendered, """
        <|im_start|>user
        Inspect the site.<|im_end|>
        <|im_start|>assistant
        Read two files.<|im_end|>
        <|im_start|>user
        <tool_response>
        First file.
        </tool_response>
        <tool_response>
        Second file.
        </tool_response><|im_end|>
        <|im_start|>assistant
        Open the page.<|im_end|>
        <|im_start|>user
        <tool_response>
        Page opened.
        </tool_response><|im_end|>
        <|im_start|>assistant

        """)
    }
}
