import XCTest
@testable import MereRunCore

final class Gemma4TextSFTTokenizerTests: XCTestCase {
    func testAssistantTargetTextStripsThinkingChannels() {
        XCTAssertEqual(
            Gemma4TextSFTTokenizer.assistantTargetText("Before <|channel>thought\nhidden\n<channel|> after "),
            "Before  after"
        )
    }

    func testCommonPrefixLengthStopsAtFirstDifference() {
        XCTAssertEqual(
            Gemma4TextSFTTokenizer.commonPrefixLength([1, 2, 3, 9], [1, 2, 4, 5]),
            2
        )
    }

    func testCommonPrefixLengthHandlesIdenticalPrefix() {
        XCTAssertEqual(
            Gemma4TextSFTTokenizer.commonPrefixLength([1, 2], [1, 2, 3, 4]),
            2
        )
    }

    func testOfficialTokenizerIncludesSchemaAndTypedToolTargetWhenAvailable() async throws {
        guard let path = ProcessInfo.processInfo.environment[
            "MERERUN_GEMMA4_TOKENIZER_PATH"
        ] else {
            throw XCTSkip(
                "Set MERERUN_GEMMA4_TOKENIZER_PATH to run the official Gemma 4 tool-SFT test."
            )
        }
        let tokenizer = try await Gemma4TokenizerAndTemplate.load(
            from: URL(fileURLWithPath: path),
            maxLengthOverride: 512
        )
        let tool = ToolDefinition(
            name: "create_status_summary",
            description: "Create a concise status summary.",
            parameters: [
                "project": ToolParameterProperty(
                    type: "string",
                    description: "Project identifier"
                ),
            ],
            required: ["project"]
        )
        let messages = [
            ChatMessage(role: .system, content: "Use available tools."),
            ChatMessage(role: .user, content: "Summarize the project status."),
            ChatMessage(
                role: .assistant,
                content: "",
                toolCalls: [ChatMessageToolCall(
                    id: "call-1",
                    name: tool.name,
                    arguments: ["project": .string("Example App")]
                )]
            ),
        ]
        let prefixTokens = try tokenizer.encodeForGeneration(
            messages: Array(messages.dropLast()),
            tools: [tool],
            addGenerationPrompt: true,
            includeThinking: false,
            maxLength: 512
        )
        let fullTokens = try tokenizer.encodeForGeneration(
            messages: messages,
            tools: [tool],
            addGenerationPrompt: false,
            includeThinking: false,
            maxLength: 512
        )
        let common = Gemma4TextSFTTokenizer.commonPrefixLength(prefixTokens, fullTokens)
        XCTAssertGreaterThan(
            common,
            prefixTokens.count - 8,
            "prefix=\(tokenizer.decode(tokens: Array(prefixTokens.suffix(16)))) "
                + "full=\(tokenizer.decode(tokens: Array(fullTokens[common..<min(fullTokens.count, common + 16)])))"
        )
        let tokenized = try XCTUnwrap(Gemma4TextSFTTokenizer.tokenize(
            [TextSFTExample(
                id: "official-tool-tokenizer",
                sources: ["test"],
                messages: messages,
                tools: [tool]
            )],
            tokenizerAndTemplate: tokenizer,
            maxSequenceLength: 512
        ).first)

        let prefix = tokenizer.decode(tokens: zip(tokenized.inputTokenIds, tokenized.lossMask)
            .compactMap { token, mask in mask == 0 ? token : nil })
        let target = tokenizer.decode(tokens: zip(tokenized.labelTokenIds, tokenized.lossMask)
            .compactMap { token, mask in mask == 1 ? token : nil })
        XCTAssertTrue(prefix.contains(tool.name), prefix)
        XCTAssertTrue(prefix.contains("Project identifier"), prefix)
        XCTAssertTrue(target.contains(tool.name), target)
        XCTAssertTrue(target.contains("Example App"), target)
    }
}
