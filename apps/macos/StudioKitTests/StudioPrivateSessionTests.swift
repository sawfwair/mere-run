import Foundation
import StudioTestSupport
import XCTest
@testable import StudioKit

@MainActor
final class StudioPrivateSessionTests: XCTestCase {
    private var root: URL!
    private var controller: MereRunController!
    private var library: StudioLibraryStore!
    private var runner: RecordingProcessRunner!

    override func setUp() async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("private-tests-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        StudioTestDefaults.redirectOutputs(under: root)
        runner = RecordingProcessRunner()
        controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: runner, resolvesCLIOnInit: false,
            taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json")))
        controller.cliPath = "/usr/bin/true"
        controller.modelIdentities.use(nil)
        library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
        library.observe(controller: controller)
    }

    override func tearDown() async throws {
        controller.terminateAllProcesses()
        controller = nil
        library = nil
        runner = nil
        StudioTestDefaults.restore()
        try FileManager.default.removeItem(at: root)
    }

    private func request(_ templateID: CommandTemplateID = .imageGenerate, prompt: String = "private prompt") throws -> StudioRunRequest {
        let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
        var draft = template.defaultDraft()
        draft.prompt = prompt
        draft.outputPath = root.appendingPathComponent("public/result.png").path
        return StudioRunRequest(mode: template.libraryMode, templateID: templateID, template: template, draft: draft)
    }

    private func completed(_ prompt: String) throws -> StudioLibraryItem {
        let session = try XCTUnwrap(controller.privateSession)
        let request = controller.taskSessions.privatizing(try request(prompt: prompt), source: .contract)
        try session.prepareDirectories(for: request)
        session.library.start(request: request, commandPreview: "private command", source: .contract)
        let output = URL(fileURLWithPath: request.draft.outputPath)
        try Data(prompt.utf8).write(to: output)
        session.library.complete(id: request.id, exitCode: 0, outputURL: output, outputText: nil, commandPreview: "private command")
        return try XCTUnwrap(session.library.items.first { $0.id == request.id })
    }

    func testPrivatePromptsAndDraftsNeverReachSavedFilesAndPublicStateReturns() throws {
        let publicID = UUID()
        library.appendUser(conversationID: publicID, mode: .chat, model: nil, systemPrompt: nil, content: "saved conversation")
        controller.taskSessions.set("saved draft", for: "image.draft")
        controller.taskSessions.flush()
        let history = try Data(contentsOf: library.libraryURL)
        let settings = try Data(contentsOf: root.appendingPathComponent("sessions.json"))
        try controller.setPrivateMode(true, savedLibrary: library)
        let session = try XCTUnwrap(controller.privateSession)
        XCTAssertTrue(session.library.items.isEmpty)
        controller.taskSessions.set("secret draft", for: "image.draft")
        controller.taskSessions.flush()
        session.library.appendUser(conversationID: UUID(), mode: .chat, model: "private model", systemPrompt: "secret system", content: "secret chat")
        library.start(request: try request(), commandPreview: "secret command", source: .contract)
        XCTAssertEqual(session.library.items.count, 2, "Even an older launcher holding the saved Library forwards private runs")
        XCTAssertEqual(try Data(contentsOf: library.libraryURL), history)
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("sessions.json")), settings)
        XCTAssertFalse(FileManager.default.fileExists(atPath: session.library.libraryURL.path))
        try controller.setPrivateMode(false, savedLibrary: library)
        XCTAssertEqual(controller.taskSessions.value(for: "image.draft", default: ""), "saved draft")
        XCTAssertEqual(library.items.map(\.id), [publicID])
        XCTAssertFalse(FileManager.default.fileExists(atPath: session.directory.path))
    }

    func testEveryMediaDestinationAndSidecarIsTemporaryAndInputsStayUnchanged() throws {
        try controller.setPrivateMode(true, savedLibrary: library)
        let session = try XCTUnwrap(controller.privateSession)
        for templateID in [CommandTemplateID.imageGenerate, .speechSynthesize, .sfxGenerate, .musicGenerate, .videoGenerate, .visionDepth] {
            let base = try request(templateID)
            let privateRequest = controller.taskSessions.privatizing(base, source: .contract)
            XCTAssertTrue(privateRequest.draft.outputPath.hasPrefix(session.directory.path + "/"), templateID.rawValue)
            XCTAssertEqual(controller.taskSessions.privatizing(privateRequest, source: .contract), privateRequest, "Relocation is idempotent")
        }
        let base = try request()
        let input = root.appendingPathComponent("input.png")
        try Data("user input".utf8).write(to: input)
        let execution = StudioExecution(templateID: .imageGenerate, arguments: [
            "image", "generate", "--prompt", "secret", "--output", base.draft.outputPath,
            "--input", input.path, "--run-dir", root.appendingPathComponent("public/run").path,
            "--structured-prompt-output", root.appendingPathComponent("public/prompt.txt").path
        ])
        let explicit = StudioRunRequest(id: base.id, mode: base.mode, templateID: base.templateID, template: base.template,
            draft: base.draft, execution: execution)
        let redirected = controller.taskSessions.privatizing(explicit, source: .contract)
        let form = try XCTUnwrap(redirected.execution?.form)
        for flag in ["--output", "--run-dir", "--structured-prompt-output"] {
            XCTAssertTrue(form.text(flag).hasPrefix(session.directory.path + "/"), flag)
        }
        XCTAssertEqual(form.text("--input"), input.path)
        try controller.discardPrivateWork()
        XCTAssertEqual(try String(contentsOf: input, encoding: .utf8), "user input")
    }

    func testKeepCopiesOnlyTheSelectedResultAndSurvivesCapabilityChangesAndQuit() throws {
        try controller.setPrivateMode(true, savedLibrary: library)
        let session = try XCTUnwrap(controller.privateSession)
        let first = try completed("keep this")
        let second = try completed("discard this")
        try session.keep(first)
        let kept = try XCTUnwrap(library.items.first)
        let output = try XCTUnwrap(kept.outputURL)
        XCTAssertEqual(try String(contentsOf: output, encoding: .utf8), "keep this")
        XCTAssertFalse(output.path.hasPrefix(session.directory.path + "/"))
        let keptFiles = try FileManager.default.subpathsOfDirectory(atPath: StudioOutputLocation.appOutputsRoot().path)
        XCTAssertFalse(keptFiles.contains { $0.contains(second.id.uuidString) })
        try controller.discardPrivateWork()
        XCTAssertTrue(controller.privateSession?.library.items.isEmpty == true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: session.directory.path))
        controller.terminateAllProcesses()
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.path))
        let reloaded = StudioLibraryStore(libraryURL: library.libraryURL)
        XCTAssertEqual(reloaded.items.map(\.id), [first.id])
        XCTAssertEqual(reloaded.items.first?.prompt, "keep this")
    }

    func testKeepConversationIsExplicitAndPrivateTurnsAreNotSavedBeforeKeep() throws {
        try controller.setPrivateMode(true, savedLibrary: library)
        let session = try XCTUnwrap(controller.privateSession)
        let id = UUID()
        session.library.appendUser(conversationID: id, mode: .chat, model: "model", systemPrompt: "system", content: "secret question")
        session.library.appendAssistant(conversationID: id, content: "secret reply", exitCode: 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: library.libraryURL.path))
        let item = try XCTUnwrap(session.library.items.first)
        try session.keep(item)
        try controller.setPrivateMode(false, savedLibrary: library)
        let reloaded = StudioLibraryStore(libraryURL: library.libraryURL)
        XCTAssertEqual(reloaded.items.first?.messages?.map(\.content), ["secret question", "secret reply"])
    }

    func testPermanentVoiceProfileCreationDoesNotWriteFromPrivateMode() throws {
        try controller.setPrivateMode(true, savedLibrary: library)
        XCTAssertFalse(controller.run(studio: try request(.speechProfileCreate)))
        XCTAssertTrue(runner.starts.isEmpty)
        XCTAssertTrue(controller.status.contains("Turn off private mode"))
        XCTAssertTrue(library.items.isEmpty)
    }

    func testVariationsPrepareDirectoriesInsideThePrivateWorkspace() throws {
        try controller.setPrivateMode(true, savedLibrary: library)
        let session = try XCTUnwrap(controller.privateSession)
        let base = try request(.imageTrainLoRA)
        let taskRunner = StudioTaskRunner(controller: controller, library: session.library)
        let submission = taskRunner.submitVariations([base], task: base.templateID.studioTask)
        XCTAssertTrue(submission.requests.first?.draft.outputPath.hasPrefix(session.directory.path + "/") == true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("public").path))
    }

    func testServicesStartedPrivatelyStopAtTheWorkspaceBoundary() throws {
        try controller.setPrivateMode(true, savedLibrary: library)
        let template = try XCTUnwrap(CommandCatalog.template(id: .apiServe))
        _ = controller.startService(template: template, draft: template.defaultDraft())
        XCTAssertEqual(runner.starts.count, 1)
        try controller.discardPrivateWork()
        XCTAssertEqual(runner.processes.first?.terminateCallCount, 1)
        runner.starts[0].termination(130)
    }

    func testDiscardingPrivateWorkDuringARunLeavesNothingMarkedRunning() throws {
        try controller.setPrivateMode(true, savedLibrary: library)
        let base = try request()
        XCTAssertTrue(controller.run(studio: base))
        XCTAssertTrue(controller.isRunning)
        try controller.discardPrivateWork()
        XCTAssertFalse(controller.isRunning)
        XCTAssertNil(controller.activeRunRequestID)
    }

    func testAPublicRunStaysStoppableWhenPrivateModeStarts() throws {
        let base = try request()
        XCTAssertTrue(controller.run(studio: base))
        try controller.setPrivateMode(true, savedLibrary: library)
        XCTAssertTrue(controller.isRunning)
        XCTAssertEqual(controller.activeRunRequestID, base.id)
        controller.cancel()
        XCTAssertEqual(runner.processes.first?.terminateCallCount, 1)
    }

    func testKeepCopiesEveryPathItRewrites() throws {
        try controller.setPrivateMode(true, savedLibrary: library)
        let session = try XCTUnwrap(controller.privateSession)
        var item = try completed("keep inputs")
        let recording = session.directory.appendingPathComponent("Recordings/take.png")
        try FileManager.default.createDirectory(at: recording.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("input".utf8).write(to: recording)
        item.commandArguments = (item.commandArguments ?? []) + ["--input", recording.path]
        try session.keep(item)
        let kept = try XCTUnwrap(library.items.first { $0.id == item.id })
        let arguments = try XCTUnwrap(kept.commandArguments)
        let index = try XCTUnwrap(arguments.firstIndex(of: "--input"))
        XCTAssertFalse(arguments[index + 1].hasPrefix(session.directory.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: arguments[index + 1]))
    }

    func testACommandWithoutDeclaredOutputsIsRefusedPrivately() throws {
        try controller.setPrivateMode(true, savedLibrary: library)
        let base = try request(.custom)
        XCTAssertFalse(controller.run(studio: base))
        XCTAssertTrue(runner.starts.isEmpty)
    }

    func testPreferredModelsApplyInPrivateWorkspaces() throws {
        controller.taskSessions.setPreferredModel("image-klein-nano", for: .createImage)
        try controller.setPrivateMode(true, savedLibrary: library)
        XCTAssertEqual(controller.taskSessions.preferredModel(for: .createImage), "image-klein-nano")
        try controller.discardPrivateWork()
        XCTAssertEqual(controller.taskSessions.preferredModel(for: .createImage), "image-klein-nano")
    }

    func testServiceOutputsHaveTheirDirectoryPrepared() throws {
        try controller.setPrivateMode(true, savedLibrary: library)
        let session = try XCTUnwrap(controller.privateSession)
        let base = try request()
        _ = controller.startService(template: base.template, draft: base.draft)
        let arguments = try XCTUnwrap(runner.starts.first).configuration.arguments
        let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
        XCTAssertTrue(output.hasPrefix(session.directory.path + "/"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: URL(fileURLWithPath: output).deletingLastPathComponent().path))
    }

    func testLaunchRemovesOnlyWorkspacesWhoseOwnerIsGone() throws {
        let parent = root.appendingPathComponent("temporary", isDirectory: true)
        let abandoned = parent.appendingPathComponent("mere-run-private-123-\(UUID())", isDirectory: true)
        let active = parent.appendingPathComponent("mere-run-private-456-\(UUID())", isDirectory: true)
        let unrelated = parent.appendingPathComponent("mere-run-other", isDirectory: true)
        for url in [abandoned, active, unrelated] {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }
        StudioPrivateSession.removeAbandonedWorkspaces(in: parent) { $0 == 456 }
        XCTAssertFalse(FileManager.default.fileExists(atPath: abandoned.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: active.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: unrelated.path))
    }

    func testDeletingPrivateResultsDoesNotMoveTheirFilesToSystemTrash() throws {
        try controller.setPrivateMode(true, savedLibrary: library)
        let item = try completed("temporary result")
        let session = try XCTUnwrap(controller.privateSession)
        let output = try XCTUnwrap(item.outputURL)
        _ = session.library.delete(ids: [item.id], trashingFiles: true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.path), "The private workspace still owns the file for Undo")
        try controller.discardPrivateWork()
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
    }

    func testSwitchingCapabilityCancelsWritersAndLateCompletionCannotRestorePrivateHistory() async throws {
        try controller.setPrivateMode(true, savedLibrary: library)
        let session = try XCTUnwrap(controller.privateSession)
        let base = try request()
        session.library.start(request: base, commandPreview: "private", source: .contract)
        XCTAssertTrue(controller.run(studio: base))
        let launch = try XCTUnwrap(runner.starts.first)
        let outputIndex = try XCTUnwrap(launch.configuration.arguments.firstIndex(of: "--output"))
        XCTAssertTrue(launch.configuration.arguments[outputIndex + 1].hasPrefix(session.directory.path + "/"))
        try controller.discardPrivateWork()
        XCTAssertEqual(runner.processes.first?.terminateCallCount, 1)
        launch.termination(130)
        for _ in 0..<100 where FileManager.default.fileExists(atPath: session.directory.path) {
            try await Task.sleep(for: .milliseconds(1))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: session.directory.path))
        XCTAssertTrue(controller.privateSession?.library.items.isEmpty == true)
        XCTAssertTrue(library.items.isEmpty)
        XCTAssertNil(controller.jobs.job(requestID: base.id))
        XCTAssertNil(controller.lastRunResult)
    }
}
