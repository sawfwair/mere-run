import Jinja
import XCTest
@testable import MereRunCore

final class ChatTemplateLoopContextTests: XCTestCase {
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
