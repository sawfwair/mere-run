import Foundation
import MereRunContract
import StudioTestSupport
@testable import StudioKit
import XCTest

/// "Save as my defaults" and "Restore app defaults": what a save keeps and leaves with the
/// draft, where fresh drafts and Reset start afterwards, that model scope still hides and
/// withholds a kept value the model does not use, and that both writes are undo steps kept in
/// the app's own task-session file.
@MainActor
final class StudioPageDefaultsTests: XCTestCase {
    private var root: URL!
    private var manager: UndoManager!
    private var controller: MereRunController!
    private var library: StudioLibraryStore!
    private var prompt: StudioPromptTaskController!

    private var sessionsURL: URL { root.appendingPathComponent("sessions.json") }
    private var sessions: StudioTaskSessions { controller.taskSessions }

    override func setUp() async throws {
        try await MainActor.run {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("page-defaults-\(UUID())")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            manager = UndoManager()
            controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: RecordingProcessRunner(),
                resolvesCLIOnInit: false, taskSessions: StudioTaskSessions(url: sessionsURL))
            controller.modelIdentities.use(nil)
            controller.taskSessions.undo.manager = manager
            library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
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

    /// One user event: what `body` registers is one undo step, as the window's manager groups it.
    private func event(_ body: () -> Void) {
        body()
        RunLoop.current.run(until: Date())
    }

    // MARK: Prompt pages

    /// An Image draft with every kind of value set: settings, and what a run is about.
    private func editedImageDraft() -> StudioDraft {
        var draft = prompt.draft
        draft.prompt = "a ceramic mug in soft morning light"
        draft.inputPath = "/tmp/mug.png"
        draft.referenceImagePaths = "/tmp/reference.png"
        draft.seed = "1234"
        draft.model = "image-krea2-raw"
        draft.width = 768
        draft.height = 512
        draft.steps = 37
        draft.secondaryText = "blurry, watermark"
        return draft
    }

    func testASaveKeepsSettingsAndLeavesWhatARunIsAboutWithTheDraft() throws {
        _ = prompt.activate(.createImage, preferredID: nil)
        event { prompt.draft = editedImageDraft() }
        XCTAssertTrue(prompt.pageDefaultsStatus(for: .createImage).canSave)
        XCTAssertFalse(prompt.pageDefaultsStatus(for: .createImage).hasSaved)

        event { prompt.savePageDefaults() }
        let saved = try XCTUnwrap(sessions.pageDefaults(for: .imageGenerate))
        XCTAssertEqual(saved.values["width"], .integer(768))
        XCTAssertEqual(saved.values["height"], .integer(512))
        XCTAssertEqual(saved.values["steps"], .integer(37))
        XCTAssertNil(saved.values["secondaryText"], "Krea 2 takes no negative prompt, so a save on it keeps none")
        for kept in ["prompt", "inputPath", "referenceImagePaths", "imageMaskPath", "seed", "model"] {
            XCTAssertNil(saved.values[kept], "\(kept) stays with the draft")
        }
        let status = prompt.pageDefaultsStatus(for: .createImage)
        XCTAssertTrue(status.hasSaved)
        XCTAssertFalse(status.canSave, "the draft is where new drafts now start")
        XCTAssertTrue(status.canRestore)
    }

    func testFreshDraftsAndResetStartFromTheSavedDefaults() {
        _ = prompt.activate(.createImage, preferredID: nil)
        let app = prompt.appFreshDraft(for: .createImage)
        event { prompt.draft = editedImageDraft() }
        event { prompt.savePageDefaults() }

        let fresh = prompt.freshDraft(for: .createImage)
        XCTAssertEqual(fresh.width, 768)
        XCTAssertEqual(fresh.steps, 37)
        XCTAssertEqual(fresh.prompt, app.prompt, "a fresh draft never inherits a prompt")
        XCTAssertEqual(fresh.inputPath, "")
        XCTAssertEqual(fresh.seed, app.seed)
        XCTAssertEqual(fresh.model, app.model)

        // The inspector's Reset reads against the fresh draft: every section back to the save.
        var draft = prompt.draft
        draft.width = 1024
        draft.height = 1024
        draft.steps = 8
        let source = controller.scopeSource
        for section in StudioInspectorSchema.sections(for: .createImage, draft: draft, source: source) {
            section.reset(&draft, to: fresh)
        }
        XCTAssertEqual(draft.width, 768)
        XCTAssertEqual(draft.height, 512)
        XCTAssertEqual(draft.steps, 37)
        XCTAssertEqual(draft.prompt, "a ceramic mug in soft morning light", "Reset leaves the prompt alone")
    }

    func testSavingAndRestoringAreUndoSteps() {
        _ = prompt.activate(.createImage, preferredID: nil)
        let app = prompt.appFreshDraft(for: .createImage)
        event { prompt.draft = editedImageDraft() }

        event { prompt.savePageDefaults() }
        XCTAssertEqual(manager.undoActionName, StudioPageDefaults.saveUndoName)
        manager.undo()
        XCTAssertNil(sessions.pageDefaults(for: .imageGenerate))
        manager.redo()
        XCTAssertNotNil(sessions.pageDefaults(for: .imageGenerate))

        event { prompt.restoreAppDefaults() }
        XCTAssertEqual(manager.undoActionName, StudioPageDefaults.restoreUndoName)
        XCTAssertNil(sessions.pageDefaults(for: .imageGenerate), "the saved defaults are forgotten")
        XCTAssertEqual(prompt.draft.width, app.width, "the settings go back to the app's")
        XCTAssertEqual(prompt.draft.steps, app.steps)
        XCTAssertEqual(prompt.draft.secondaryText, app.secondaryText, "including those the model hides")
        XCTAssertEqual(prompt.draft.prompt, "a ceramic mug in soft morning light", "what the run is about stays")
        XCTAssertEqual(prompt.draft.inputPath, "/tmp/mug.png")
        XCTAssertEqual(prompt.draft.seed, "1234")
        XCTAssertEqual(prompt.draft.model, "image-krea2-raw")
        XCTAssertFalse(prompt.pageDefaultsStatus(for: .createImage).canRestore)

        manager.undo()
        XCTAssertNotNil(sessions.pageDefaults(for: .imageGenerate), "one step brings the saved defaults back")
        XCTAssertEqual(prompt.draft.width, 768, "and the draft with them")
        XCTAssertEqual(prompt.draft.steps, 37)
        XCTAssertEqual(prompt.draft.secondaryText, "blurry, watermark")
    }

    func testDefaultsAreKeptInTheAppsOwnTaskSessionFile() throws {
        _ = prompt.activate(.createImage, preferredID: nil)
        event { prompt.draft = editedImageDraft() }
        event { prompt.savePageDefaults() }
        sessions.flush()

        let reopened = StudioTaskSessions(url: sessionsURL)
        XCTAssertEqual(reopened.pageDefaults(for: .imageGenerate), sessions.pageDefaults(for: .imageGenerate))
        XCTAssertNil(reopened.pageDefaults(for: .videoGenerate), "each page keeps its own")
    }

    // MARK: Model scope

    func testAKeptValueTheModelDoesNotUseStaysHiddenAndIsNotRun() throws {
        typealias Music = StudioScopeContracts.Music
        let source = StudioScopeContracts.source([Music.capability])
        var ace = StudioDraft.baseline(for: .music)
        ace.prompt = "Acoustic folk waltz"
        ace.model = "music-acestep"
        ace.musicQuality = "high"
        let defaults = StudioPageDefaults(capturing: ace, mode: .music, source: source)
        XCTAssertEqual(defaults.values["musicQuality"], .text("high"), "ACE-Step uses a quality preset, so the save keeps it")

        var yue = StudioDraft.baseline(for: .music)
        yue.prompt = "Acoustic folk waltz"
        yue.model = "music-yue2"
        defaults.apply(to: &yue, mode: .music)
        XCTAssertEqual(yue.musicQuality, "high", "the draft holds the kept value")
        XCTAssertFalse(StudioContractSchema.fields(for: .music, draft: yue, source: source).contains { $0.flag == "--quality" },
                       "YuE2 does not use it, so the inspector does not show it")
        let onYuE = try StudioCommandAdapter.makeRequest(mode: .music, draft: yue, source: source)
        XCTAssertNotEqual(onYuE.draft.musicQuality, "high", "and the run leaves it out")

        yue.model = "music-acestep"
        let onACE = try StudioCommandAdapter.makeRequest(mode: .music, draft: yue, source: source)
        XCTAssertEqual(onACE.draft.musicQuality, "high", "back on ACE-Step the kept value runs")

        var yueSave = StudioDraft.baseline(for: .music)
        yueSave.model = "music-yue2"
        yueSave.musicQuality = "high"
        XCTAssertNil(StudioPageDefaults(capturing: yueSave, mode: .music, source: source).values["musicQuality"],
                     "a save keeps only what the draft's model uses: what its inspector shows")
    }

    // MARK: Task-draft pages

    func testATaskPageKeepsItsSettingsPerTemplateAndLeavesInputsSeedModelAndOutput() throws {
        let universr = "audio-enhance-universr-audio"
        var draft = try XCTUnwrap(sessions.taskDraft(for: .audioEnhance))
        let app = StudioTaskDraft(templateID: draft.templateID)
        draft.setArgument(0, "/tmp/voice.wav")
        draft.model = universr
        draft.form["--ode-steps"] = .integer(20)
        draft.form["--dtype"] = .text("float16")
        draft.form["--seed"] = .integer(7)
        draft.form["--output"] = .text("/tmp/out.wav")
        event { sessions.setTaskDraft(draft, for: .audioEnhance) }
        XCTAssertTrue(sessions.pageDefaultsStatus(for: .audioEnhance, draft: draft, source: .contract).canSave)

        event { sessions.savePageDefaults(for: .audioEnhance, draft: draft, source: .contract) }
        XCTAssertEqual(manager.undoActionName, StudioPageDefaults.saveUndoName)
        let saved = try XCTUnwrap(sessions.pageDefaults(for: .audioEnhance, templateID: draft.templateID))
        XCTAssertEqual(saved.values["--ode-steps"], .integer(20))
        XCTAssertEqual(saved.values["--dtype"], .text("float16"))
        for kept in ["--seed", "--model", "--model-path", "--output", "--overlap"] {
            XCTAssertNil(saved.values[kept], "\(kept) is not kept")
        }

        // A fresh draft starts from the save, on the app's model, with no input.
        let fresh = sessions.freshTaskDraft(for: .audioEnhance, templateID: draft.templateID)
        XCTAssertEqual(fresh.form["--ode-steps"], .integer(20))
        XCTAssertEqual(fresh.form["--dtype"], .text("float16"))
        XCTAssertEqual(fresh.model, app.model)
        XCTAssertEqual(fresh.argument(0), "")
        XCTAssertEqual(fresh.form["--seed"], app.form["--seed"])
        // The app's model is AP-BWE, which has no ODE: the kept steps are hidden and not run.
        XCTAssertFalse(StudioTaskSchema.fields(for: .audioEnhance, draft: fresh, source: .contract).contains { $0.flag == "--ode-steps" })
        XCTAssertFalse(fresh.arguments(source: .contract).contains("--ode-steps"))
        var onUniverSR = fresh
        onUniverSR.model = universr
        let arguments = onUniverSR.arguments(source: .contract)
        XCTAssertEqual(arguments.firstIndex(of: "--ode-steps").map { arguments[$0 + 1] }, "20")

        // Reset reads against the fresh draft.
        var edited = draft
        edited.form["--dtype"] = .text("float32")
        for section in StudioTaskSchema.sections(for: .audioEnhance, draft: edited, source: .contract) {
            section.reset(&edited, to: fresh)
        }
        XCTAssertEqual(edited.form["--dtype"], .text("float16"))
        XCTAssertEqual(edited.argument(0), "/tmp/voice.wav")
    }

    func testRestoringATaskPagesAppDefaultsIsOneUndoStep() throws {
        var draft = try XCTUnwrap(sessions.taskDraft(for: .audioEnhance))
        let app = StudioTaskDraft(templateID: draft.templateID)
        draft.setArgument(0, "/tmp/voice.wav")
        draft.model = "audio-enhance-universr-audio"
        draft.form["--ode-steps"] = .integer(20)
        event { sessions.setTaskDraft(draft, for: .audioEnhance) }
        event { sessions.savePageDefaults(for: .audioEnhance, draft: draft, source: .contract) }

        event { sessions.restoreAppDefaults(for: .audioEnhance, source: .contract) }
        XCTAssertEqual(manager.undoActionName, StudioPageDefaults.restoreUndoName)
        let restored = try XCTUnwrap(sessions.taskDraft(for: .audioEnhance))
        XCTAssertNil(sessions.pageDefaults(for: .audioEnhance, templateID: draft.templateID))
        XCTAssertEqual(restored.form["--ode-steps"], app.form["--ode-steps"])
        XCTAssertEqual(restored.argument(0), "/tmp/voice.wav", "the input stays")
        XCTAssertEqual(restored.model, "audio-enhance-universr-audio", "the model stays")
        XCTAssertFalse(sessions.pageDefaultsStatus(for: .audioEnhance, draft: restored, source: .contract).canRestore)

        manager.undo()
        XCTAssertNotNil(sessions.pageDefaults(for: .audioEnhance, templateID: draft.templateID))
        XCTAssertEqual(sessions.taskDraft(for: .audioEnhance)?.form["--ode-steps"], .integer(20))
    }

    func testANewTaskDraftStartsFromTheSavedDefaults() throws {
        let templateID = try XCTUnwrap(StudioTask.textEmbeddings.variantTemplates.first?.id)
        let defaults = StudioPageDefaults(values: ["--dimensions": .integer(256)])
        sessions.set(Optional(defaults), for: StudioTaskSessions.pageDefaultsKey(.textEmbeddings, templateID: templateID))
        XCTAssertEqual(sessions.taskDraft(for: .textEmbeddings)?.form["--dimensions"], .integer(256),
                       "a page with nothing parked yet opens on its saved defaults")
    }
}
