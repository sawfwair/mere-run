import Foundation
import StudioTestSupport
@testable import StudioKit
import XCTest

/// The composer's prompt history: which prompts a page recalls and in what order, how ↑ and ↓
/// step through them, where on the prompt's lines they take over from the caret, and that a
/// recall is an undo step.
@MainActor
final class StudioPromptHistoryTests: XCTestCase {
    private let start = Date(timeIntervalSince1970: 1_800_000_000)

    private func run(
        _ prompt: String, mode: StudioMode = .createImage, template: CommandTemplateID? = nil, minutes: Double
    ) -> StudioLibraryItem {
        var item = StudioLibraryItem(
            id: UUID(), mode: mode, prompt: prompt, inputURL: nil, outputURL: nil,
            createdAt: start.addingTimeInterval(minutes * 60), updatedAt: start, status: .completed, exitCode: 0,
            commandPreview: "mere.run", outputText: nil
        )
        item.templateID = template
        return item
    }

    // MARK: History

    func testNewestFirstEachPromptOnceAndBlankPromptsLeftOut() {
        let items = [
            run("a lighthouse at dusk", minutes: 1),
            run("  a ceramic mug  ", minutes: 5),
            run("", minutes: 6),
            run("a lighthouse at dusk", minutes: 7),
            run("a fox in snow", minutes: 3),
        ]
        XCTAssertEqual(
            StudioPromptHistory.prompts(for: .imageGenerate, in: items),
            ["a lighthouse at dusk", "a ceramic mug", "a fox in snow"],
            "the newest run of a repeated prompt places it; whitespace does not make a second entry"
        )
    }

    func testEachPageRecallsTheRunsItsFeedShows() {
        let items = [
            run("a mug", minutes: 1),
            run("a drone shot of a coast", mode: .video, minutes: 2),
            run("the quick brown fox", mode: .readImage, template: .textEmbed, minutes: 3),
            run("Call Ada at 555-0100", mode: .readImage, template: .textAnonymize, minutes: 4),
        ]
        XCTAssertEqual(StudioPromptHistory.prompts(for: .imageGenerate, in: items), ["a mug"])
        XCTAssertEqual(StudioPromptHistory.prompts(for: .videoGenerate, in: items), ["a drone shot of a coast"])
        XCTAssertEqual(StudioPromptHistory.prompts(for: .textEmbeddings, in: items), ["the quick brown fox"],
                       "a task-draft run belongs to the page of its command, not the mode it was filed under")
        XCTAssertEqual(StudioPromptHistory.prompts(for: .textAnonymize, in: items), ["Call Ada at 555-0100"])
        XCTAssertTrue(StudioPromptHistory.prompts(for: .musicCompose, in: items).isEmpty)
    }

    func testAThreadsUserTurnsAreItsPrompts() {
        var chat = run("", mode: .chat, minutes: 0)
        chat.messages = [
            StudioMessage(role: .user, content: "What is a roofline?", createdAt: start.addingTimeInterval(60)),
            StudioMessage(role: .assistant, content: "A bound on throughput.", createdAt: start.addingTimeInterval(61)),
            StudioMessage(role: .user, content: "Show one for M4", createdAt: start.addingTimeInterval(120)),
        ]
        var code = run("", mode: .code, minutes: 0)
        code.messages = [StudioMessage(role: .user, content: "Write a parser", createdAt: start.addingTimeInterval(90))]
        XCTAssertEqual(StudioPromptHistory.prompts(for: .chatChat, in: [chat, code]), ["Show one for M4", "What is a roofline?"])
        XCTAssertEqual(StudioPromptHistory.prompts(for: .chatCode, in: [chat, code]), ["Write a parser"])
    }

    func testMenuTitlesAreOneLineAndCut() {
        XCTAssertEqual(StudioPromptHistory.menuTitle(for: "a mug\n\non a table"), "a mug on a table")
        let long = String(repeating: "word ", count: 30)
        let title = StudioPromptHistory.menuTitle(for: long)
        XCTAssertEqual(title.count, StudioPromptHistory.menuTitleLength)
        XCTAssertTrue(title.hasSuffix("…"))
    }

    // MARK: Stepping

    func testUpStepsOlderAndDownComesBackToWhatWasTyped() {
        let history = ["newest", "middle", "oldest"]
        var recall = StudioPromptRecall()
        XCTAssertEqual(recall.older(than: "half typed", in: history), "newest")
        XCTAssertEqual(recall.older(than: "newest", in: history), "middle")
        XCTAssertEqual(recall.older(than: "middle", in: history), "oldest")
        XCTAssertNil(recall.older(than: "oldest", in: history), "nothing older: ↑ does what it would have")
        XCTAssertEqual(recall.newer(than: "oldest", in: history), "middle")
        XCTAssertEqual(recall.newer(than: "middle", in: history), "newest")
        XCTAssertEqual(recall.newer(than: "newest", in: history), "half typed", "↓ past the newest puts the typed text back")
        XCTAssertNil(recall.newer(than: "half typed", in: history), "not recalling: ↓ moves the caret")
    }

    func testTheFirstUpSkipsThePromptTheFieldAlreadyShows() {
        var recall = StudioPromptRecall()
        XCTAssertEqual(recall.older(than: "newest", in: ["newest", "older"]), "older",
                       "the prompt just run is still in the field; ↑ shows the one before it")
        var idle = StudioPromptRecall()
        XCTAssertNil(idle.newer(than: "", in: ["newest"]))
        var empty = StudioPromptRecall()
        XCTAssertNil(empty.older(than: "", in: []), "no history: ↑ moves the caret")
    }

    func testEditingARecalledPromptStartsOverFromTheNewest() {
        let history = ["newest", "middle", "oldest"]
        var recall = StudioPromptRecall()
        _ = recall.older(than: "", in: history)
        XCTAssertEqual(recall.older(than: "newest", in: history), "middle")
        XCTAssertEqual(recall.older(than: "middle, edited", in: history), "newest", "an edit is new text to come back to")
        XCTAssertEqual(recall.newer(than: "newest", in: history), "middle, edited")
    }

    // MARK: Caret

    func testUpAndDownRecallOnlyFromTheFirstAndLastLines() {
        let text = "first line\nsecond line\nthird line"
        XCTAssertTrue(StudioPromptRecall.isOnFirstLine("", caret: 0))
        XCTAssertTrue(StudioPromptRecall.isOnLastLine("", caret: 0))
        XCTAssertTrue(StudioPromptRecall.isOnFirstLine(text, caret: 5))
        XCTAssertTrue(StudioPromptRecall.isOnFirstLine(text, caret: 10), "at the end of the first line, before its break")
        XCTAssertFalse(StudioPromptRecall.isOnFirstLine(text, caret: 11), "the second line: ↑ moves the caret up")
        XCTAssertFalse(StudioPromptRecall.isOnLastLine(text, caret: 15), "the middle line: ↓ moves the caret down")
        XCTAssertTrue(StudioPromptRecall.isOnLastLine(text, caret: 24))
        XCTAssertTrue(StudioPromptRecall.isOnLastLine(text, caret: (text as NSString).length))
        XCTAssertFalse(StudioPromptRecall.isOnLastLine("line\n", caret: 2), "a trailing break opens a last, empty line")
    }

    // MARK: Undo

    func testARecallIsAnUndoStepOnBothKindsOfPage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("prompt-history-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let manager = UndoManager()
        let controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: RecordingProcessRunner(),
            resolvesCLIOnInit: false, taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json")))
        defer { controller.terminateAllProcesses() }
        controller.modelIdentities.use(nil)
        controller.taskSessions.undo.manager = manager
        let library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
        let prompt = StudioPromptTaskController(controller: controller, library: library)
        _ = prompt.activate(.createImage, preferredID: nil)
        var typed = prompt.draft
        typed.prompt = "half typed"
        prompt.draft = typed
        RunLoop.current.run(until: Date())

        prompt.recallPrompt("a ceramic mug")
        XCTAssertEqual(prompt.draft.prompt, "a ceramic mug")
        XCTAssertEqual(manager.undoActionName, StudioPageDefaults.recallUndoName)
        manager.undo()
        XCTAssertEqual(prompt.draft.prompt, "half typed")
        manager.redo()
        XCTAssertEqual(prompt.draft.prompt, "a ceramic mug")
        RunLoop.current.run(until: Date())

        let sessions = controller.taskSessions
        let before = sessions.taskDraft(for: .textEmbeddings)?.prompt
        RunLoop.current.run(until: Date())
        sessions.recallPrompt("hello from the archive", for: .textEmbeddings)
        XCTAssertEqual(sessions.taskDraft(for: .textEmbeddings)?.prompt, "hello from the archive")
        XCTAssertEqual(manager.undoActionName, StudioPageDefaults.recallUndoName)
        manager.undo()
        XCTAssertEqual(sessions.taskDraft(for: .textEmbeddings)?.prompt, before)
    }
}
