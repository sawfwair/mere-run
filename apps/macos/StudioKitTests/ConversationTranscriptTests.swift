@testable import StudioKit
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

    func testStripThinkTagsHidesTrailingUnclosedBlockWhileStreaming() {
        let text = "Partial answer <think>still reasoning..."
        XCTAssertEqual(ConversationTranscript.stripThinkTags(text, streaming: true), "Partial answer")
    }

    func testStripThinkTagsKeepsLiteralUnclosedTagAtFinalize() {
        // A completed code reply discussing the tag must not be truncated.
        let text = "Use the <think> tag to mark reasoning."
        XCTAssertEqual(ConversationTranscript.stripThinkTags(text), "Use the <think> tag to mark reasoning.")
    }

    func testStripThinkTagsRemovesLeadingOrphanClose() {
        // Some models pre-fill the opening tag and emit only the close.
        let text = "hidden reasoning here</think>The visible answer."
        XCTAssertEqual(ConversationTranscript.stripThinkTags(text), "The visible answer.")
    }

    func testRenderSkipsFailedAssistantTurns() {
        let messages = [
            StudioMessage(role: .user, content: "first"),
            StudioMessage(role: .assistant, content: "boom", failed: true),
            StudioMessage(role: .user, content: "second"),
        ]
        let rendered = ConversationTranscript.render(messages: messages)
        // The failed assistant turn is never replayed into the prompt.
        XCTAssertEqual(rendered.prompt, "User: first\n\nUser: second")
        XCTAssertFalse(rendered.prompt.contains("boom"))
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

    func testBudgetDerivesFromContextWindowLessTheReplyRoom() {
        XCTAssertEqual(
            ConversationTranscript.budgetChars(contextTokens: 32_768, maxOutputTokens: 2_048),
            (32_768 - 2_048) * ConversationTranscript.charsPerContextToken
        )
        // No context known: the fixed default, exactly as before.
        XCTAssertEqual(
            ConversationTranscript.budgetChars(contextTokens: nil, maxOutputTokens: 2_048),
            ConversationTranscript.defaultBudgetChars
        )
        XCTAssertEqual(
            ConversationTranscript.budgetChars(contextTokens: 0, maxOutputTokens: 2_048),
            ConversationTranscript.defaultBudgetChars
        )
        // A tiny context still leaves room for the latest turn.
        XCTAssertEqual(
            ConversationTranscript.budgetChars(contextTokens: 1_024, maxOutputTokens: 4_096),
            ConversationTranscript.minimumBudgetChars
        )
    }

    func testContextTokensPreferExplicitSizeThenInventoryThenNothing() {
        let inventory = [
            StudioModelInventoryRow(
                id: "text-chat-qwen3.6-4b", category: "text-chat", status: "installed", size: "2.4 GB",
                usageTerms: nil, contextWindow: 40_960
            ),
            StudioModelInventoryRow(
                id: "text-chat-unknown", category: "text-chat", status: "installed", size: "1 GB", usageTerms: nil
            ),
        ]
        XCTAssertEqual(
            ConversationTranscript.contextTokens(
                requestedContextSize: 8_192, model: "text-chat-qwen3.6-4b", inventory: inventory
            ),
            8_192
        )
        XCTAssertEqual(
            ConversationTranscript.contextTokens(requestedContextSize: 0, model: "text-chat-qwen3.6-4b", inventory: inventory),
            40_960
        )
        XCTAssertNil(
            ConversationTranscript.contextTokens(requestedContextSize: 0, model: "text-chat-unknown", inventory: inventory)
        )
        XCTAssertNil(ConversationTranscript.contextTokens(requestedContextSize: 0, model: "", inventory: inventory))
    }

    func testDecodeSpeedIsReadFromTheLatestStatsLine() {
        let lines = [
            "Loading model…",
            "time=3.10s load=0.40s prefill=0.20s decode=2.50s tokens=98 decode_tps=39.20 e2e_tps=31.61",
            "time=2.90s load=0.00s prefill=0.20s decode=2.40s tokens=99 decode_tps=41.25 e2e_tps=34.13 prefill_tps=812.00",
        ]
        XCTAssertEqual(ConversationTranscript.decodeTokensPerSecond(in: lines), 41.25)
        XCTAssertNil(ConversationTranscript.decodeTokensPerSecond(in: ["no stats here"]))
        XCTAssertNil(ConversationTranscript.decodeTokensPerSecond(in: ["decode_tps=0.00"]))
    }

    func testSplitThinkingKeepsReasoningBesideTheAnswer() {
        let reply = ConversationTranscript.splitThinking("<think>plan A\nplan B</think>The answer is 42.<think>check</think> Done.")
        XCTAssertEqual(reply.answer, "The answer is 42. Done.")
        XCTAssertEqual(reply.reasoning, "plan A\nplan B\n\ncheck")
        XCTAssertFalse(reply.isThinking)

        // A pre-filled opening tag: the text before the orphan close is the reasoning.
        let orphan = ConversationTranscript.splitThinking("hidden reasoning</think>The visible answer.")
        XCTAssertEqual(orphan.answer, "The visible answer.")
        XCTAssertEqual(orphan.reasoning, "hidden reasoning")

        let plain = ConversationTranscript.splitThinking("just text")
        XCTAssertEqual(plain.answer, "just text")
        XCTAssertNil(plain.reasoning)
    }

    func testSplitThinkingReportsAnUnclosedBlockWhileStreamingAndKeepsItAtFinalize() {
        let streaming = ConversationTranscript.splitThinking("Partial answer <think>still reasoning...", streaming: true)
        XCTAssertEqual(streaming.answer, "Partial answer")
        XCTAssertEqual(streaming.reasoning, "still reasoning...")
        XCTAssertTrue(streaming.isThinking)

        // The moment the tag opens there is no reasoning yet, but the model is thinking.
        let opened = ConversationTranscript.splitThinking("<think>", streaming: true)
        XCTAssertEqual(opened.answer, "")
        XCTAssertNil(opened.reasoning)
        XCTAssertTrue(opened.isThinking)

        // A completed reply's leftover tag is literal text, exactly as stripThinkTags treats it.
        let final = ConversationTranscript.splitThinking("Use the <think> tag to mark reasoning.")
        XCTAssertEqual(final.answer, "Use the <think> tag to mark reasoning.")
        XCTAssertNil(final.reasoning)
        XCTAssertFalse(final.isThinking)

        XCTAssertEqual(streaming.hidingReasoning, ConversationTranscript.Reply(answer: "Partial answer", reasoning: nil, isThinking: false))
    }

    func testSplitThinkingHoldsBackATagSplitAcrossChunksWhileStreaming() {
        // A chunk boundary inside a tag must not flash the fragment as literal text.
        XCTAssertEqual(
            ConversationTranscript.splitThinking("The answer <", streaming: true),
            ConversationTranscript.Reply(answer: "The answer", reasoning: nil, isThinking: false)
        )
        XCTAssertEqual(
            ConversationTranscript.splitThinking("<think>plan</thin", streaming: true),
            ConversationTranscript.Reply(answer: "", reasoning: "plan", isThinking: true)
        )
        // A pre-filled block closing mid-stream: the reasoning is done, the answer has begun.
        XCTAssertEqual(
            ConversationTranscript.splitThinking("hidden</think>The vis", streaming: true),
            ConversationTranscript.Reply(answer: "The vis", reasoning: "hidden", isThinking: false)
        )
        // Only a fragment is held back; a whole tag and ordinary angle brackets are not.
        XCTAssertEqual(ConversationTranscript.splitThinking("<think>", streaming: true).isThinking, true)
        XCTAssertEqual(ConversationTranscript.splitThinking("a < b", streaming: true).answer, "a < b")
        XCTAssertEqual(ConversationTranscript.splitThinking("The answer <", streaming: false).answer, "The answer <")
    }

    func testReasoningAndFailureDiagnosticsAreNeverReplayed() {
        let messages = [
            StudioMessage(role: .user, content: "first"),
            StudioMessage(role: .assistant, content: "The answer.", reasoning: "SECRET PLAN"),
            StudioMessage(role: .user, content: "second"),
            StudioMessage(
                role: .assistant, content: "partial", failed: true,
                failureReason: "STDERR REASON", logTail: ["error: STDERR LINE"]
            ),
            StudioMessage(role: .user, content: "third"),
        ]
        let prompt = ConversationTranscript.render(messages: messages).prompt
        XCTAssertEqual(prompt, "User: first\n\nAssistant: The answer.\n\nUser: second\n\nUser: third")
        XCTAssertFalse(prompt.contains("SECRET PLAN"))
        XCTAssertFalse(prompt.contains("STDERR"))
    }
}
