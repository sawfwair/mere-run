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
    private var defaults: UserDefaults!
    private var suiteName: String!

    override func setUp() async throws {
        try await MainActor.run {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("task-runner-\(UUID())")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            suiteName = "StudioTaskRunnerTests-\(UUID().uuidString)"
            defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
            defaults.set(root.appendingPathComponent("outputs").path, forKey: StudioOutputLocation.rootDefaultsKey)
            StudioOutputLocation.defaults = defaults
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
            StudioOutputLocation.defaults = .standard
            defaults.removePersistentDomain(forName: suiteName)
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
        let prepared = try StudioTaskRunner.prepare(base, sessions: controller.taskSessions)
        XCTAssertNil(prepared.fallbackReason)
        XCTAssertEqual(prepared.request.draft, draft)
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("outputs/Audio").path))
    }
}
