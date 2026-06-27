@testable import MereRunApp
import XCTest

final class ConversationTranscriptTests: XCTestCase {
    func testFirstTurnRendersUserMessageVerbatim() {
        let only = StudioMessage(role: .user, content: "what is swift?")
        let rendered = ConversationTranscript.render(messages: [only])
        XCTAssertEqual(rendered.prompt, "what is swift?")
        XCTAssertEqual(rendered.droppedCount, 0)
        XCTAssertEqual(rendered.includedMessageIDs, [only.id])
    }

    func testMultiTurnRendersLabeledDialogueEndingOnLatestUser() {
        let messages = [
            StudioMessage(role: .user, content: "hi"),
            StudioMessage(role: .assistant, content: "hello"),
            StudioMessage(role: .user, content: "and now?"),
        ]
        let rendered = ConversationTranscript.render(messages: messages)
        XCTAssertEqual(rendered.prompt, "User: hi\n\nAssistant: hello\n\nUser: and now?")
        XCTAssertEqual(rendered.droppedCount, 0)
        XCTAssertEqual(rendered.includedMessageIDs.count, 3)
    }

    func testOverBudgetDropsOldestKeepsLatestAndReports() {
        let oldest = StudioMessage(role: .user, content: String(repeating: "a", count: 200))
        let middle = StudioMessage(role: .assistant, content: String(repeating: "b", count: 200))
        let latest = StudioMessage(role: .user, content: "keep me")
        let rendered = ConversationTranscript.render(
            messages: [oldest, middle, latest],
            budgetChars: 100
        )
        XCTAssertEqual(rendered.prompt, "keep me")
        XCTAssertEqual(rendered.droppedCount, 2)
        XCTAssertEqual(rendered.includedMessageIDs, [latest.id])
        XCTAssertFalse(rendered.includedMessageIDs.contains(oldest.id))
    }

    func testStripThinkTagsRemovesCompleteBlock() {
        let text = "<think>reasoning here</think>The answer is 42."
        XCTAssertEqual(ConversationTranscript.stripThinkTags(text), "The answer is 42.")
    }

    func testStripThinkTagsRemovesMultilineAndMultipleBlocks() {
        let text = "<think>line1\nline2</think>Hello<think>more</think> world"
        XCTAssertEqual(ConversationTranscript.stripThinkTags(text), "Hello world")
    }

    func testStripThinkTagsHidesTrailingUnclosedBlock() {
        let text = "Partial answer <think>still reasoning..."
        XCTAssertEqual(ConversationTranscript.stripThinkTags(text), "Partial answer")
    }

    func testStripThinkTagsLeavesPlainTextUntouched() {
        XCTAssertEqual(ConversationTranscript.stripThinkTags("just text"), "just text")
    }

    func testSystemPromptIsReservedAgainstTheBudget() {
        let messages = [
            StudioMessage(role: .user, content: String(repeating: "x", count: 60)),
            StudioMessage(role: .user, content: "latest"),
        ]
        // With a big system reserve the older message no longer fits.
        let withReserve = ConversationTranscript.render(
            messages: messages,
            systemPrompt: String(repeating: "s", count: 90),
            budgetChars: 120
        )
        XCTAssertEqual(withReserve.droppedCount, 1)
        XCTAssertEqual(withReserve.prompt, "latest")
        XCTAssertGreaterThanOrEqual(withReserve.approxChars, 90)
    }
}
