import Foundation
import StudioTestSupport
@testable import StudioKit
import XCTest

/// One runner for every task: it names the destination, applies Command edits, validates, files
/// the run under its template's own Library mode, remembers it for Stop, and reports a
/// destination that had to move — for a task draft and for a legacy page's request alike.
@MainActor
final class StudioTaskRunnerTests: XCTestCase {
    private var root: URL!
    private var processRunner: RecordingProcessRunner!
    private var controller: MereRunController!
    private var library: StudioLibraryStore!
    private var runner: StudioTaskRunner!

    override func setUp() async throws {
        try await MainActor.run {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("task-runner-\(UUID())")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            StudioTestDefaults.redirectOutputs(under: root)
            processRunner = RecordingProcessRunner()
            controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: processRunner, resolvesCLIOnInit: false,
                taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json")))
            library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
            library.observe(controller: controller)
            runner = StudioTaskRunner(controller: controller, library: library)
        }
    }

    override func tearDown() async throws {
        try await MainActor.run {
            controller.terminateAllProcesses()
            StudioTestDefaults.restore()
            runner = nil
            library = nil
            controller = nil
            processRunner = nil
            try FileManager.default.removeItem(at: root)
        }
    }

    private func enhanceDraft() throws -> StudioTaskDraft {
        let input = root.appendingPathComponent("voice-memo.wav")
        try Data().write(to: input)
        var draft = StudioTaskDraft(templateID: .audioEnhance)
        draft.setArgument(0, input.path)
        return draft
    }

    func testRunningATaskDraftNamesRecordsAndRemembersTheRun() async throws {
        controller.readinessByTask[.audioEnhance] = .ready
        let draft = try enhanceDraft()

        let request = try runner.run(draft, task: .audioEnhance)

        XCTAssertEqual(request.mode, .listen, "attributed by the template, not a hand-picked mode")
        XCTAssertEqual(request.templateID, .audioEnhance)
        XCTAssertTrue(request.draft.outputPath.hasPrefix(root.appendingPathComponent("outputs/Audio").path), request.draft.outputPath)
        XCTAssertTrue(request.draft.outputPath.contains("voice-memo"), "named after the input")
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("outputs/Audio").path), "the folder is created")
        XCTAssertEqual(processRunner.starts.count, 1)
        XCTAssertTrue(processRunner.starts[0].configuration.arguments.contains(request.draft.outputPath))
        let row = try XCTUnwrap(library.items.first { $0.id == request.id })
        XCTAssertEqual(row.templateID, .audioEnhance)
        XCTAssertEqual(row.commandArguments, request.execution?.arguments)
        XCTAssertEqual(row.inputURL?.lastPathComponent, "voice-memo.wav")
        XCTAssertEqual(
            controller.taskSessions.value(for: StudioTask.audioEnhance.rawValue + ".requestID", default: Optional<UUID>.none),
            request.id
        )
        XCTAssertEqual(runner.currentJob(for: .audioEnhance)?.request.requestID, request.id)
        XCTAssertNil(controller.outputFallbackReason)

        runner.stop(task: .audioEnhance)
        XCTAssertEqual(processRunner.processes[0].terminateCallCount, 1, "Stop terminates the remembered job's process")
        processRunner.starts[0].termination(15)
        for _ in 0..<6 { await Task.yield() }
        XCTAssertNil(runner.currentJob(for: .audioEnhance), "a cancelled job is no longer the task's current one")
    }

    /// The Command view's "Will run" is the launch preview: the draft's own blank destination
    /// named the way the runner names it, so the argv it shows is the argv that runs.
    func testTheLaunchPreviewShowsTheDestinationTheRunWrites() throws {
        controller.readinessByTask[.audioEnhance] = .ready
        let draft = try enhanceDraft()
        XCTAssertEqual(draft.text("--output"), "", "the draft keeps no destination")

        let preview = StudioTaskRunner.launchPreview(draft, source: .contract)
        let previewed = preview.text("--output")
        XCTAssertTrue(previewed.hasPrefix(root.appendingPathComponent("outputs/Audio").path), previewed)
        let request = try runner.run(draft, task: .audioEnhance)
        XCTAssertEqual(request.draft.outputPath, previewed)
        XCTAssertEqual(request.execution?.arguments, preview.arguments(source: .contract))
    }

    /// Audio ▸ Separate runs Music ▸ Separate's command. Stop on either acts on that task's own
    /// run: ⌘. in one never cancels the other's, even once its own run has ended.
    func testStopActsOnlyOnTheTasksOwnRunWhenTwoTasksShareACommand() async throws {
        controller.readinessByTask[.musicSeparate] = .ready
        controller.readinessByTask[.audioSeparate] = .ready
        let input = root.appendingPathComponent("harbor-lights.wav")
        try Data().write(to: input)
        var draft = StudioTaskDraft(templateID: .musicSeparate)
        let slot = try XCTUnwrap(StudioTaskSchema.primarySlot(for: .musicSeparate))
        draft.setAttachmentText(input.path, for: slot.storage)

        let music = try runner.run(draft, task: .musicSeparate)
        XCTAssertNil(runner.currentJob(for: .audioSeparate), "Music ▸ Separate's run is not Audio ▸ Separate's to stop")
        runner.stop(task: .audioSeparate)
        XCTAssertEqual(processRunner.processes[0].terminateCallCount, 0)

        let audio = try runner.run(draft, task: .audioSeparate)
        XCTAssertEqual(runner.currentJob(for: .audioSeparate)?.request.requestID, audio.id)
        XCTAssertEqual(runner.currentJob(for: .musicSeparate)?.request.requestID, music.id)
        processRunner.starts[0].termination(0)
        for _ in 0..<6 { await Task.yield() }
        XCTAssertNil(runner.currentJob(for: .musicSeparate), "Audio ▸ Separate's run does not become Music ▸ Separate's")
    }

    func testBlockedReadinessAndAnIncompleteCommandLeaveHistoryUntouched() throws {
        controller.readinessByTask[.audioEnhance] = .missingModel("audio-enhance-ap-bwe-16kto48k")
        XCTAssertThrowsError(try runner.run(try enhanceDraft(), task: .audioEnhance)) { error in
            XCTAssertTrue(error.localizedDescription.contains("isn't on this Mac"))
        }
        controller.readinessByTask[.audioEnhance] = .ready
        // The well gate speaks first for an empty input; the contract's own validation still
        // guards the rest (`testALegacyPagesInvalidRequestIsRecordedAndFailedByAdmission`).
        XCTAssertThrowsError(try runner.run(StudioTaskDraft(templateID: .audioEnhance), task: .audioEnhance)) { error in
            XCTAssertEqual(error as? StudioValidationError, StudioValidationError(message: "Attach audio first."))
        }
        var badOverlap = try enhanceDraft()
        badOverlap.form["--overlap"] = .text("not-a-number")
        XCTAssertThrowsError(try runner.run(badOverlap, task: .audioEnhance)) { error in
            XCTAssertEqual(error as? StudioValidationError, StudioValidationError(message: "AP-BWE overlap must be a whole number."))
        }
        XCTAssertTrue(library.items.isEmpty)
        XCTAssertTrue(controller.jobs.all.isEmpty)
        XCTAssertTrue(processRunner.starts.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("outputs").path), "no folder before a run")
    }

    /// `music transcribe`'s positional is optional in the contract only because of
    /// `--list-instruments`; the task's surface requires the well filled, so an empty one is
    /// refused with the composer's own words before anything is recorded or launched.
    func testAnEmptyWellIsRefusedWhenTheTaskRequiresAnInput() throws {
        controller.readinessByTask[.musicTranscribe] = .ready
        XCTAssertThrowsError(try runner.run(StudioTaskDraft(templateID: .musicTranscribe), task: .musicTranscribe)) { error in
            XCTAssertEqual(error as? StudioValidationError, StudioValidationError(message: "Attach audio first."))
        }
        XCTAssertTrue(library.items.isEmpty)
        XCTAssertTrue(processRunner.starts.isEmpty)

        controller.readinessByTask[.soundCondition] = .ready
        let condition = try runner.run(StudioTaskDraft(templateID: .sfxConditionText), task: .soundCondition)
        XCTAssertEqual(condition.templateID, .sfxConditionText, "a task with no input slot runs on its prompt")
    }

    /// Faces ▸ Batch takes its pictures in the well or as a list file (`--input-list`); a run
    /// with only the list is not refused as an empty well, one with neither still is.
    func testAListFileFillsFacesBatchesWell() throws {
        controller.readinessByTask[.visionFaces] = .ready
        var batch = StudioTaskDraft(templateID: .visionFaceBatch)
        XCTAssertThrowsError(try runner.run(batch, task: .visionFaces)) { error in
            XCTAssertEqual((error as? StudioValidationError)?.message.hasPrefix("Attach"), true, error.localizedDescription)
        }
        let list = root.appendingPathComponent("portraits.txt")
        try Data("/tmp/a.png\n/tmp/b.png\n".utf8).write(to: list)
        batch.form["--input-list"] = .text(list.path)
        let request = try runner.run(batch, task: .visionFaces)
        XCTAssertEqual(request.templateID, .visionFaceBatch)
    }

    /// A typed input is the run's whole subject: `text anonymize` with nothing typed would launch
    /// and wait on stdin, so an empty editor is refused before anything is recorded.
    func testAnEmptyTypedInputIsRefused() throws {
        controller.readinessByTask[.textAnonymize] = .ready
        var draft = StudioTaskDraft(templateID: .textAnonymize)
        draft.prompt = ""
        XCTAssertThrowsError(try runner.run(draft, task: .textAnonymize)) { error in
            XCTAssertEqual(error as? StudioValidationError, StudioValidationError(message: "Type the text first."))
        }
        XCTAssertTrue(library.items.isEmpty)
        XCTAssertTrue(processRunner.starts.isEmpty)
    }

    func testALegacyPagesRequestRunsThroughTheSamePath() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .sfxAEEncode))
        var draft = template.defaultDraft()
        draft.inputPath = root.appendingPathComponent("hit.wav").path
        draft.outputPath = root.appendingPathComponent("outputs/Sound/hit.npy").path
        let base = StudioRunRequest(mode: .sfx, templateID: .sfxAEEncode, template: template, draft: draft)

        let request = try runner.run(request: base, task: .soundEncode)

        XCTAssertEqual(request.mode, .sfx, "a page's attribution is kept")
        XCTAssertEqual(library.items.first?.templateID, .sfxAEEncode)
        XCTAssertEqual(processRunner.starts.count, 1)
        XCTAssertEqual(runner.currentJob(for: .soundEncode)?.request.requestID, request.id)
    }

    /// A task-specific page's Run with an incomplete command (`validating: false`) records the
    /// row and lets admission fail it; the Command view's validated path throws instead.
    func testAnInvalidPageRequestIsRecordedAndFailedByAdmission() async throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .sfxAEEncode))
        var draft = template.defaultDraft()
        draft.outputPath = root.appendingPathComponent("outputs/Sound/hit.npy").path
        let base = StudioRunRequest(mode: .sfx, templateID: .sfxAEEncode, template: template, draft: draft)

        XCTAssertThrowsError(try runner.run(request: base, task: .soundEncode), "the Command view's Run validates first")
        XCTAssertTrue(library.items.isEmpty)

        let request = try runner.run(request: base, task: .soundEncode, validating: false)
        for _ in 0..<6 { await Task.yield() }

        let row = try XCTUnwrap(library.items.first { $0.id == request.id })
        XCTAssertEqual(row.status, .failed)
        XCTAssertEqual(row.outputText?.contains("required"), true, row.outputText ?? "")
        XCTAssertTrue(processRunner.starts.isEmpty, "admission failed it before a process started")
        XCTAssertEqual(controller.jobs.job(requestID: request.id)?.state.isPreflightFailure, true)
    }

    func testALaunchRefusalFailsTheLibraryRow() async throws {
        controller.readinessByTask[.audioEnhance] = .ready
        processRunner.launchError = NSError(domain: "StudioTaskRunnerTests", code: 1)
        let detachedLibrary = StudioLibraryStore(libraryURL: root.appendingPathComponent("refused-library.json"))
        let detachedRunner = StudioTaskRunner(controller: controller, library: detachedLibrary)

        let request = try detachedRunner.run(try enhanceDraft(), task: .audioEnhance)
        for _ in 0..<6 { await Task.yield() }

        let row = try XCTUnwrap(detachedLibrary.items.first { $0.id == request.id })
        XCTAssertEqual(row.status, .failed)
        XCTAssertEqual(row.exitCode, -1)
        XCTAssertEqual(row.outputText, controller.jobs.job(requestID: request.id)?.result?.outputText)
        XCTAssertTrue(processRunner.starts.isEmpty)
    }

    /// The same announcement the prompt controller makes: a destination that cannot be created
    /// moves the run to App Outputs and the reason reaches the shell's banner.
    func testAnUnwritableDestinationFallsBackAndIsAnnounced() throws {
        let blocker = root.appendingPathComponent("outputs")
        try Data("not a folder".utf8).write(to: blocker)
        controller.readinessByTask[.audioEnhance] = .ready

        let request = try runner.run(try enhanceDraft(), task: .audioEnhance)

        XCTAssertNotNil(controller.outputFallbackReason)
        XCTAssertTrue(request.draft.outputPath.hasPrefix(StudioOutputLocation.appOutputsRoot().path), request.draft.outputPath)
        XCTAssertEqual(request.execution?.arguments.contains(request.draft.outputPath), true, "the argv moved with it")
    }

    /// The live-acceptance builder and the app prepare a request identically.
    func testPrepareIsTheOneSequence() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .speechDiarize))
        var draft = template.defaultDraft()
        draft.inputPath = root.appendingPathComponent("standup.wav").path
        draft.outputPath = root.appendingPathComponent("outputs/Audio/standup.json").path
        let base = StudioRunRequest(mode: .listen, templateID: .speechDiarize, template: template, draft: draft)
        let prepared = try StudioTaskRunner.prepare(base, sessions: controller.taskSessions, source: .contract)
        XCTAssertNil(prepared.fallbackReason)
        XCTAssertEqual(prepared.request.draft, draft)
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("outputs/Audio").path))
    }
}
