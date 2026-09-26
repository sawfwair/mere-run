import Foundation
import StudioTestSupport
@testable import StudioKit
import XCTest

/// Undo and redo through a real `UndoManager` that groups by event, as the window's does; each
/// `event` closes its group the way the end of an event would. Everything lives in a temporary
/// folder, the Trash included.
@MainActor
final class StudioUndoTests: XCTestCase {
    private var root: URL!
    private var trash: URL!
    private var manager: UndoManager!
    private var controller: MereRunController!
    private var library: StudioLibraryStore!
    private var prompt: StudioPromptTaskController!

    override func setUp() async throws {
        try await MainActor.run {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("studio-undo-\(UUID())")
            trash = root.appendingPathComponent("Trash", isDirectory: true)
            try FileManager.default.createDirectory(at: trash, withIntermediateDirectories: true)
            manager = UndoManager()
            controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: RecordingProcessRunner(),
                resolvesCLIOnInit: false, taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json")))
            controller.modelIdentities.use(nil)
            controller.taskSessions.undo.manager = manager
            library = makeLibrary()
            library.undo.manager = manager
            prompt = StudioPromptTaskController(controller: controller, library: library)
        }
    }

    override func tearDown() async throws {
        try await MainActor.run {
            controller.terminateAllProcesses()
            controller.taskSessions.flush()
            prompt = nil
            library = nil
            controller = nil
            manager = nil
            try FileManager.default.removeItem(at: root)
        }
    }

    /// A Library over the temporary folder whose Trash is `trash`: a file moves there under a
    /// unique name, as the Finder's Trash would take it.
    private func makeLibrary() -> StudioLibraryStore {
        let trash = trash!
        return StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"), trashItem: { url in
            let landed = trash.appendingPathComponent(UUID().uuidString + "-" + url.lastPathComponent)
            try FileManager.default.moveItem(at: url, to: landed)
            return landed
        })
    }

    /// One user event: everything `body` registers is one step, and an event that registers
    /// nothing leaves no step behind.
    private func event(_ body: () -> Void) {
        body()
        // The manager closes the event's group on the run loop's next pass.
        RunLoop.current.run(until: Date())
        XCTAssertEqual(manager.groupingLevel, 0)
    }

    private func run(prompt text: String, file: URL) -> StudioLibraryItem {
        var item = StudioLibraryItem(
            id: UUID(), mode: .createImage, prompt: text, inputURL: nil, outputURL: file,
            createdAt: Date(), updatedAt: Date(), status: .completed, exitCode: 0,
            commandPreview: "mere.run image generate", outputText: nil
        )
        item.artifactURLs = [file]
        return item
    }

    // MARK: - Library

    func testDeleteThenUndoRestoresTheRowAndMovesItsFilesBackOutOfTheTrash() throws {
        let file = root.appendingPathComponent("mug.png")
        try Data([1, 2, 3]).write(to: file)
        let newer = run(prompt: "newer", file: root.appendingPathComponent("absent.png"))
        let mug = run(prompt: "a ceramic mug", file: file)
        library.upsert(mug)
        library.upsert(newer)
        XCTAssertEqual(library.items.map(\.prompt), ["newer", "a ceramic mug"])

        event { XCTAssertTrue(library.delete(ids: [mug.id], trashingFiles: true).isEmpty) }
        XCTAssertEqual(manager.undoActionName, "Delete Run")
        XCTAssertEqual(library.items.map(\.prompt), ["newer"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path), "the file is in the Trash, not kept aside")
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: trash.path).count, 1)
        XCTAssertEqual(makeLibrary().items.map(\.prompt), ["newer"], "a deletion nobody undoes is already on disk")

        manager.undo()
        XCTAssertEqual(library.items.map(\.id), [newer.id, mug.id], "the row returns to its place")
        XCTAssertEqual(try Data(contentsOf: file), Data([1, 2, 3]))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: trash.path).isEmpty)
        XCTAssertEqual(makeLibrary().items.map(\.id), [newer.id, mug.id], "the restored row is saved")
        XCTAssertEqual(manager.redoActionName, "Delete Run")

        manager.redo()
        XCTAssertEqual(library.items.map(\.id), [newer.id])
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
        XCTAssertEqual(manager.undoActionName, "Delete Run")
        manager.undo()
        XCTAssertEqual(library.items.map(\.id), [newer.id, mug.id])
        XCTAssertTrue(FileManager.default.fileExists(atPath: file.path))
    }

    func testDeletingThreadsAndKeepingFilesUndoesAsOneNamedStep() {
        let first = UUID()
        let second = UUID()
        library.appendUser(conversationID: first, mode: .chat, model: nil, systemPrompt: nil, content: "Hello")
        library.appendUser(conversationID: second, mode: .chat, model: nil, systemPrompt: nil, content: "Again")

        event { library.delete(ids: [first, second], trashingFiles: false) }
        XCTAssertEqual(manager.undoActionName, "Delete Threads")
        XCTAssertTrue(library.items.isEmpty)
        manager.undo()
        XCTAssertEqual(library.items.map(\.id), [second, first])
        XCTAssertEqual(library.items.last?.messages?.first?.content, "Hello")
    }

    func testRenameAndFavoriteUndoAndRedo() {
        let item = run(prompt: "a lighthouse", file: root.appendingPathComponent("lighthouse.png"))
        library.upsert(item)

        event { library.rename(id: item.id, title: "Harbor light") }
        event { library.setFavorite(id: item.id, isFavorite: true) }
        XCTAssertEqual(manager.undoActionName, "Add to Favorites")

        manager.undo()
        XCTAssertEqual(library.items[0].isFavorite, nil)
        XCTAssertEqual(manager.undoActionName, "Rename")
        manager.undo()
        XCTAssertNil(library.items[0].customTitle)
        manager.redo()
        XCTAssertEqual(library.items[0].customTitle, "Harbor light")
        XCTAssertEqual(manager.redoActionName, "Add to Favorites")
        manager.redo()
        XCTAssertEqual(library.items[0].isFavorite, true)
    }

    // MARK: - Drafts

    func testADraftChangeUndoesToThePreviousValueAndRedoes() {
        _ = prompt.activate(.createImage, preferredID: nil)
        let before = prompt.draft
        event { prompt.draft.steps = before.steps + 3 }
        XCTAssertEqual(manager.undoActionName, "Change Steps")

        // Typing into the prompt afterwards is the text field's to undo, and survives this one.
        prompt.draft.prompt = "a quiet harbor at dawn"
        manager.undo()
        XCTAssertEqual(prompt.draft.steps, before.steps)
        XCTAssertEqual(prompt.draft.prompt, "a quiet harbor at dawn")
        XCTAssertEqual(controller.taskSessions.value(for: StudioTask.imageGenerate.rawValue + ".draft", default: StudioDraft()).steps, before.steps,
                       "the stored draft follows")

        manager.redo()
        XCTAssertEqual(prompt.draft.steps, before.steps + 3)
        XCTAssertEqual(manager.undoActionName, "Change Steps")
    }

    func testTypingIsLeftToTheTextFieldsOwnUndo() throws {
        _ = prompt.activate(.createImage, preferredID: nil)
        event { prompt.draft.prompt = "a quiet harbor" }
        event { prompt.draft.seed = "4242" }
        _ = prompt.activate(.chat, preferredID: nil)
        event { prompt.draft.secondaryText = "Answer in one line." }
        let sessions = controller.taskSessions
        var enhance = try XCTUnwrap(sessions.taskDraft(for: .audioEnhance))
        enhance.form.extraArguments = "--verbose"
        event { sessions.setTaskDraft(enhance, for: .audioEnhance) }
        XCTAssertFalse(manager.canUndo, "the prompt, the seed, the system prompt, and extra arguments are typed")
    }

    /// A text field registers its typing before its binding writes the draft, so a draft change
    /// arriving in an event that already holds another registrar's step is that field's edit.
    func testAChangeInAnEventATextFieldRegisteredInIsLeftToTheField() {
        _ = prompt.activate(.createImage, preferredID: nil)
        let textField = NSObject()
        event {
            manager.registerUndo(withTarget: textField) { _ in }
            manager.setActionName("Typing")
            prompt.draft.steps += 1
        }
        XCTAssertEqual(manager.undoActionName, "Typing")
        event { prompt.draft.steps += 1 }
        XCTAssertEqual(manager.undoActionName, "Change Steps", "the next event's change is the draft's again")
    }

    func testModelSwitchesAndAttachmentsAreNamedSteps() {
        _ = prompt.activate(.readImage, preferredID: nil)
        let before = prompt.draft
        event { prompt.draft.model = "vision-some-other-model" }
        XCTAssertEqual(manager.undoActionName, "Change Model")
        event { prompt.draft.inputPath = root.appendingPathComponent("photo.png").path }
        XCTAssertEqual(manager.undoActionName, "Add Attachment")

        manager.undo()
        XCTAssertEqual(prompt.draft.inputPath, before.inputPath)
        XCTAssertEqual(prompt.draft.model, "vision-some-other-model")
        manager.undo()
        XCTAssertEqual(prompt.draft, before)
    }

    func testASliderDragIsOneStep() {
        _ = prompt.activate(.createImage, preferredID: nil)
        let before = prompt.draft.steps
        for value in before + 1...before + 6 {
            event { prompt.draft.steps = value }
        }
        manager.undo()
        XCTAssertEqual(prompt.draft.steps, before)
        XCTAssertFalse(manager.canUndo, "the whole drag was one step")
        manager.redo()
        XCTAssertEqual(prompt.draft.steps, before + 6)
    }

    func testAResetIsOneNamedStep() {
        _ = prompt.activate(.createImage, preferredID: nil)
        let baseline = prompt.draft
        event {
            prompt.draft.steps = baseline.steps + 2
            prompt.draft.cfgScale = baseline.cfgScale + 1
        }
        event {
            controller.taskSessions.undo.naming("Reset Settings") { prompt.draft = baseline }
        }
        XCTAssertEqual(manager.undoActionName, "Reset Settings")
        manager.undo()
        XCTAssertEqual(prompt.draft.steps, baseline.steps + 2)
        XCTAssertEqual(prompt.draft.cfgScale, baseline.cfgScale + 1)
    }

    func testATaskDraftChangeUndoesThroughTheSessions() throws {
        let sessions = controller.taskSessions
        let before = try XCTUnwrap(sessions.taskDraft(for: .audioEnhance))
        var next = before
        next.model = "audio-some-other-model"
        event { sessions.setTaskDraft(next, for: .audioEnhance) }
        XCTAssertEqual(manager.undoActionName, "Change Model")
        manager.undo()
        XCTAssertEqual(sessions.taskDraft(for: .audioEnhance), before)
        manager.redo()
        XCTAssertEqual(sessions.taskDraft(for: .audioEnhance)?.model, "audio-some-other-model")
    }

    func testUseTheseSettingsUndoBringsThePreviousDraftBack() throws {
        _ = prompt.activate(.createImage, preferredID: nil)
        event { prompt.draft.steps = 7 }
        prompt.draft.prompt = "my own prompt"
        let mine = prompt.draft

        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var recorded = template.defaultDraft()
        recorded.prompt = "a green ceramic bowl"
        recorded.steps = 9
        let item = library.start(request: StudioRunRequest(mode: .createImage, templateID: .imageGenerate,
            template: template, draft: recorded), commandPreview: "fixture", source: .contract)

        event { XCTAssertTrue(prompt.useSettings(from: item)) }
        XCTAssertEqual(manager.undoActionName, "Use These Settings")
        XCTAssertEqual(prompt.draft.prompt, "a green ceramic bowl")

        manager.undo()
        XCTAssertEqual(prompt.draft, mine, "the whole previous draft, typed prompt included")
        XCTAssertEqual(controller.taskSessions.value(for: StudioTask.imageGenerate.rawValue + ".draft", default: StudioDraft()), mine)
        XCTAssertEqual(manager.undoActionName, "Change Steps", "one step, not one per value it wrote")

        manager.redo()
        XCTAssertEqual(prompt.draft.prompt, "a green ceramic bowl")
        XCTAssertEqual(prompt.draft.steps, 9)
    }

    func testUseTheseSettingsOnATaskDraftUndoes() throws {
        let sessions = controller.taskSessions
        let before = try XCTUnwrap(sessions.taskDraft(for: .audioEnhance))
        let template = try XCTUnwrap(CommandCatalog.template(id: .audioEnhance))
        var recorded = template.defaultDraft()
        recorded.inputPath = root.appendingPathComponent("voice.wav").path
        let item = library.start(request: StudioRunRequest(mode: template.libraryMode, templateID: .audioEnhance,
            template: template, draft: recorded), commandPreview: "fixture", source: .contract)

        event { XCTAssertTrue(prompt.useTaskSettings(from: item, task: .audioEnhance)) }
        XCTAssertNotEqual(sessions.taskDraft(for: .audioEnhance), before)
        XCTAssertEqual(manager.undoActionName, "Use These Settings")
        manager.undo()
        XCTAssertEqual(sessions.taskDraft(for: .audioEnhance), before)
        XCTAssertFalse(manager.canUndo)
    }

    // MARK: - Coalescing

    func testChangesToTheSameFieldCoalesceOnlyWithinTheInterval() {
        var now = Date(timeIntervalSinceReferenceDate: 0)
        let undo = StudioUndo(now: { now })
        undo.manager = manager
        var value = 0
        func change(to next: Int) {
            let previous = value
            value = next
            event { undo.register("Change Steps", coalescing: "steps") { value = previous } }
        }
        change(to: 1)
        now += 0.5
        change(to: 2)
        now += StudioUndo.coalescingInterval + 0.1
        change(to: 3)

        manager.undo()
        XCTAssertEqual(value, 2, "a pause starts a new step")
        manager.undo()
        XCTAssertEqual(value, 0, "the first two changes were one step")
        XCTAssertFalse(manager.canUndo)
    }
}
