import Foundation

enum FlintToolParser {
    static func parse(_ text: String, tools: [ToolDefinition], parallel: Bool) -> [ToolCall] {
        let visible = ChatReasoningMarkup.splitThinkBlocks(in: text).visibleContent
        return ToolCallPolicy.validatedCalls(LagunaToolParser.parseToolCalls(visible), tools: tools, parallelToolCalls: parallel)
    }
}
