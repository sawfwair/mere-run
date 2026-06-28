@testable import MereRunApp
import Combine
import Foundation
import XCTest

@MainActor
final class MereRunControllerTests: XCTestCase {
    func testPackageRootSearchStopsAtFilesystemRoot() {
        let startedAt = Date()

        XCTAssertNil(
            CLIResolver.nearestPackageRoot(
                from: URL(fileURLWithPath: "/", isDirectory: true),
                fileManager: .default
            )
        )
        XCTAssertLessThan(Date().timeIntervalSince(startedAt), 0.25)
    }

    func testPackageRootSearchFindsNearestPackageManifest() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("MereRunControllerTests-\(UUID().uuidString)", isDirectory: true)
        let nested = root
            .appendingPathComponent("a", isDirectory: true)
            .appendingPathComponent("b", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try "// swift-tools-version: 6.0\n".write(
            to: root.appendingPathComponent("Package.swift", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )

        XCTAssertEqual(
            CLIResolver.nearestPackageRoot(from: nested, fileManager: .default),
            root
        )
    }

    func testReadinessCheckDoesNotStartDuplicateInFlightRequest() {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var draft = StudioDraft()
        draft.reset(for: .createImage)

        controller.checkReadiness(for: .createImage, draft: draft)
        controller.checkReadiness(for: .createImage, draft: draft)

        XCTAssertEqual(runner.starts.count, 1)
        XCTAssertEqual(
            Array(runner.starts[0].configuration.arguments.suffix(4)),
            ["model", "capabilities", "--all", "--json"]
        )
    }

    func testReadinessCheckTerminatesStaleRequestAndIgnoresItsResult() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var draft = StudioDraft()
        draft.reset(for: .createImage)

        controller.checkReadiness(for: .createImage, draft: draft)
        draft.model = "video-ltx-av"
        controller.checkReadiness(for: .createImage, draft: draft)

        XCTAssertEqual(runner.starts.count, 2)
        XCTAssertEqual(runner.processes.first?.terminateCallCount, 1)

        runner.starts[0].stdout(supportedCapabilitiesOutput(for: "image-zimage-nano", minimum: 12))
        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()
        XCTAssertEqual(controller.readinessByMode[.createImage], .checking)

        runner.starts[1].stdout(supportedCapabilitiesOutput(for: "video-ltx-av", minimum: 64))
        runner.starts[1].termination(0)
        await Task.yield()
        await Task.yield()
        XCTAssertEqual(runner.starts.count, 3)

        runner.starts[2].stdout("ID Category Status Size\nvideo-ltx-av media installed 12 GB\n")
        runner.starts[2].termination(0)
        await Task.yield()
        await Task.yield()
        XCTAssertEqual(controller.readinessByMode[.createImage], .ready)
    }

    func testReadinessBlocksUnsupportedCapabilityBeforeModelList() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var draft = StudioDraft()
        draft.reset(for: .createImage)

        controller.checkReadiness(for: .createImage, draft: draft)
        runner.starts[0].stdout(
            unsupportedCapabilitiesOutput(
                for: "image-zimage-nano",
                minimum: 12,
                detected: 8
            )
        )
        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()

        XCTAssertEqual(runner.starts.count, 1)
        XCTAssertEqual(
            controller.readinessByMode[.createImage],
            .unsupported("Requires at least 12 GB unified memory; detected 8 GB.")
        )
        XCTAssertEqual(controller.modelCapabilitiesByID["image-zimage-nano"]?.minimumUnifiedMemoryGB, 12)
    }

    func testReadinessCapturesRecommendedChatModelFromJSONCapabilities() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let chatTemplate = try XCTUnwrap(CommandCatalog.template(id: .textChat))
        controller.select(chatTemplate)
        XCTAssertEqual(controller.draft.model, StudioChatDefaults.fallbackModelID)
        var draft = StudioDraft()
        draft.reset(for: .chat)

        controller.checkReadiness(for: .chat, draft: draft)
        XCTAssertEqual(
            Array(runner.starts[0].configuration.arguments.suffix(4)),
            ["model", "capabilities", "--all", "--json"]
        )

        runner.starts[0].stdout(jsonCapabilitiesOutput(
            recommendedChatModelID: "text-agent-deepseek-v4-flash",
            supportedModelID: StudioChatDefaults.fallbackModelID
        ))
        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()

        XCTAssertEqual(controller.recommendedChatModelID, "text-agent-deepseek-v4-flash")
        XCTAssertEqual(controller.draft.model, "text-agent-deepseek-v4-flash")
        XCTAssertEqual(runner.starts.count, 2)
    }

    func testReadImageAutoDownloadActionIsReadyWithoutStartingCLI() {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var draft = StudioDraft()
        draft.reset(for: .readImage)
        draft.readImageAction = .inspect

        // Inspect auto-downloads its vision-language model via the CLI, so it is ready and
        // needs no managed-model readiness probe (no child CLI launched).
        controller.checkReadiness(for: .readImage, draft: draft)

        XCTAssertTrue(runner.starts.isEmpty)
        XCTAssertEqual(controller.readinessByMode[.readImage], .ready)
    }

    func testFailedRunResultIncludesStderrWhenStdoutIsEmpty() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        controller.select(template)
        controller.draft.extraArguments = "--version"

        controller.run()
        runner.starts[0].stderr("unsupported model\n")
        runner.starts[0].termination(64)
        await Task.yield()
        await Task.yield()

        XCTAssertEqual(controller.lastRunResult?.exitCode, 64)
        XCTAssertEqual(controller.lastRunResult?.outputText, "unsupported model")
    }

    func testValidationFailurePublishesRunResultWithoutStartingProcess() throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        controller.select(template)
        controller.draft.prompt = ""

        XCTAssertFalse(controller.run())

        XCTAssertTrue(runner.starts.isEmpty)
        XCTAssertFalse(controller.isRunning)
        XCTAssertEqual(controller.lastRunResult?.templateID, .imageGenerate)
        XCTAssertEqual(controller.lastRunResult?.exitCode, 64)
        XCTAssertEqual(controller.lastRunResult?.outputText, "Prompt is required.")
    }

    func testOutputPreparationFailurePublishesRunResultWithoutStartingProcess() throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("MereRunControllerTests-\(UUID().uuidString)", isDirectory: true)
        let fileParent = temp.appendingPathComponent("not-a-directory", isDirectory: false)
        defer { try? FileManager.default.removeItem(at: temp) }
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        XCTAssertTrue(FileManager.default.createFile(atPath: fileParent.path, contents: Data()))
        controller.select(template)
        controller.draft.prompt = "a ceramic coffee mug"
        controller.draft.outputPath = fileParent.appendingPathComponent("image.png").path

        XCTAssertFalse(controller.run())

        XCTAssertTrue(runner.starts.isEmpty)
        XCTAssertFalse(controller.isRunning)
        XCTAssertEqual(controller.lastRunResult?.templateID, .imageGenerate)
        XCTAssertEqual(controller.lastRunResult?.exitCode, -1)
        XCTAssertTrue(controller.lastRunResult?.outputText?.contains("Could not create output directory") == true)
    }

    func testModelPullUsesDownloadingStatusWhileRunning() throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .modelPull))
        controller.select(template)
        controller.draft.model = "image-zimage-nano"

        controller.run()

        XCTAssertTrue(controller.isRunning)
        XCTAssertEqual(controller.status, "Downloading model")
        XCTAssertEqual(
            runner.starts.first?.configuration.arguments,
            ["model", "pull", "image-zimage-nano"]
        )
    }

    func testStudioRunsExecuteConcurrentlyUpToCapThenQueue() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        func request(_ arg: String, mode: StudioMode) -> StudioRunRequest {
            var draft = template.defaultDraft()
            draft.extraArguments = arg
            return StudioRunRequest(mode: mode, templateID: .custom, template: template, draft: draft)
        }
        XCTAssertEqual(MereRunController.maxConcurrentRuns, 2)
        let first = request("first", mode: .chat)
        let second = request("second", mode: .code)
        let third = request("third", mode: .chat)

        // Up to maxConcurrentRuns run at once; the foreground follows the latest started.
        XCTAssertTrue(controller.run(studio: first))
        XCTAssertTrue(controller.run(studio: second))
        XCTAssertEqual(runner.starts.count, 2)
        XCTAssertEqual(controller.queuedRunCount, 0)
        XCTAssertTrue(controller.isRunning)
        XCTAssertEqual(controller.activeRunRequestID, second.id)

        // A run beyond the cap queues.
        XCTAssertTrue(controller.run(studio: third))
        XCTAssertEqual(runner.starts.count, 2)
        XCTAssertEqual(controller.queuedRunCount, 1)

        // A background run completing still publishes its result (the library keys by request id)
        // and frees a slot, so the queued run starts.
        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()

        XCTAssertEqual(controller.lastRunResult?.requestID, first.id)
        XCTAssertEqual(controller.queuedRunCount, 0)
        XCTAssertEqual(runner.starts.count, 3)
        XCTAssertEqual(runner.starts[2].configuration.arguments, ["third"])
    }

    func testQueuedConversationTurnIsTrackedInFlightAtSubmission() throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        func request(_ arg: String, _ conversationID: UUID) -> StudioRunRequest {
            var draft = template.defaultDraft()
            draft.extraArguments = arg
            return StudioRunRequest(
                mode: .chat, templateID: .custom, template: template, draft: draft,
                conversationID: conversationID
            )
        }
        let queued = UUID()
        XCTAssertTrue(controller.run(studio: request("a", UUID())))
        XCTAssertTrue(controller.run(studio: request("b", UUID())))
        XCTAssertTrue(controller.run(studio: request("c", queued)))
        // The third turn queues behind the cap but is still tracked in-flight, so a second send
        // into that thread is blocked and a pending bubble can show.
        XCTAssertEqual(controller.queuedRunCount, 1)
        XCTAssertTrue(controller.runningConversationIDs.contains(queued))
    }

    func testConcurrentCompletionsAreAllDeliveredLosslessly() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        var received: [UUID] = []
        let cancellable = controller.runCompletions.sink { result in
            if let conversationID = result.conversationID { received.append(conversationID) }
        }
        defer { cancellable.cancel() }
        func request(_ conversationID: UUID) -> StudioRunRequest {
            var draft = template.defaultDraft()
            draft.extraArguments = "x"
            return StudioRunRequest(
                mode: .chat, templateID: .custom, template: template, draft: draft,
                conversationID: conversationID
            )
        }
        let first = UUID()
        let second = UUID()
        XCTAssertTrue(controller.run(studio: request(first)))
        XCTAssertTrue(controller.run(studio: request(second)))

        // Both finish back-to-back; the lossless stream must deliver both (lastRunResult alone
        // would coalesce to the latter).
        runner.starts[0].termination(0)
        runner.starts[1].termination(0)
        for _ in 0..<6 { await Task.yield() }
        XCTAssertEqual(Set(received), [first, second])
    }

    func testFailedConversationTurnReplyIsThinkStripped() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        var draft = template.defaultDraft()
        draft.extraArguments = "x"
        let conversationID = UUID()
        let request = StudioRunRequest(
            mode: .chat, templateID: .custom, template: template, draft: draft,
            conversationID: conversationID
        )
        XCTAssertTrue(controller.run(studio: request))
        runner.starts[0].stdout("<think>oops</think>partial reply")
        await Task.yield()
        runner.starts[0].termination(1)
        await Task.yield()
        await Task.yield()
        // Even on failure the reply is think-stripped (no leak into the next turn's prompt).
        XCTAssertEqual(controller.lastRunResult?.exitCode, 1)
        XCTAssertEqual(controller.lastRunResult?.outputText, "partial reply")
    }

    func testConversationIDFlowsToResultAndTracksInFlight() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        let conversationID = UUID()
        var draft = template.defaultDraft()
        draft.extraArguments = "turn"
        let request = StudioRunRequest(
            mode: .chat, templateID: .custom, template: template, draft: draft,
            conversationID: conversationID
        )

        XCTAssertTrue(controller.run(studio: request))
        XCTAssertTrue(controller.runningConversationIDs.contains(conversationID))

        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()

        XCTAssertEqual(controller.lastRunResult?.conversationID, conversationID)
        XCTAssertFalse(controller.runningConversationIDs.contains(conversationID))
    }

    func testQueuedConversationTurnStillCarriesConversationID() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        func request(_ arg: String, _ conversationID: UUID) -> StudioRunRequest {
            var draft = template.defaultDraft()
            draft.extraArguments = arg
            return StudioRunRequest(
                mode: .chat, templateID: .custom, template: template, draft: draft,
                conversationID: conversationID
            )
        }
        let queuedConversation = UUID()
        // Fill the concurrency cap, then a third turn queues.
        XCTAssertTrue(controller.run(studio: request("a", UUID())))
        XCTAssertTrue(controller.run(studio: request("b", UUID())))
        XCTAssertTrue(controller.run(studio: request("c", queuedConversation)))
        XCTAssertEqual(controller.queuedRunCount, 1)

        // Freeing a slot dequeues the third turn — it must still carry its conversation id.
        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()
        XCTAssertEqual(runner.starts.count, 3)
        XCTAssertTrue(controller.runningConversationIDs.contains(queuedConversation))

        runner.starts[2].termination(0)
        await Task.yield()
        await Task.yield()
        XCTAssertEqual(controller.lastRunResult?.conversationID, queuedConversation)
    }

    func testConversationReplyKeepsFullOutputBeyondConsoleBuffer() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        var draft = template.defaultDraft()
        draft.extraArguments = "turn"
        let conversationID = UUID()
        let request = StudioRunRequest(
            mode: .chat, templateID: .custom, template: template, draft: draft,
            conversationID: conversationID
        )
        XCTAssertTrue(controller.run(studio: request))

        // A reply far larger than the 32 KB console buffer must survive in full for the thread.
        let head = String(repeating: "A", count: 40_000)
        let tail = "TAIL-MARKER"
        runner.starts[0].stdout(head + tail)
        await Task.yield()
        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()

        let captured = try XCTUnwrap(controller.lastRunResult?.outputText)
        XCTAssertEqual(controller.lastRunResult?.conversationID, conversationID)
        XCTAssertGreaterThan(captured.count, 39_000)
        XCTAssertTrue(captured.hasSuffix(tail))
        XCTAssertTrue(captured.hasPrefix("A"))
    }

    func testConversationReplyStripsThinkTagsLiveAndFinal() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        var draft = template.defaultDraft()
        draft.extraArguments = "turn"
        let conversationID = UUID()
        let request = StudioRunRequest(
            mode: .chat, templateID: .custom, template: template, draft: draft,
            conversationID: conversationID
        )
        XCTAssertTrue(controller.run(studio: request))

        runner.starts[0].stdout("<think>deliberating</think>Final answer.")
        await Task.yield()
        // Live, think-stripped text is published for the streaming bubble.
        XCTAssertEqual(controller.conversationLiveText[conversationID], "Final answer.")

        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()
        XCTAssertEqual(controller.lastRunResult?.outputText, "Final answer.")
        // Cleared once finalized so the bubble switches to the persisted message.
        XCTAssertNil(controller.conversationLiveText[conversationID])
    }

    func testConcurrentRunsKeepIsolatedOutputAndResults() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        func request(_ arg: String) -> StudioRunRequest {
            var draft = template.defaultDraft()
            draft.extraArguments = arg
            return StudioRunRequest(mode: .chat, templateID: .custom, template: template, draft: draft)
        }
        let background = request("background")
        let foreground = request("foreground")

        XCTAssertTrue(controller.run(studio: background))   // starts[0]
        XCTAssertTrue(controller.run(studio: foreground))   // starts[1], now the foreground
        XCTAssertEqual(runner.starts.count, 2)

        runner.starts[0].stdout("background-only-line\n")
        runner.starts[1].stdout("foreground-line\n")
        await Task.yield()

        // The console shows only the foreground run's output; the background run's is isolated.
        XCTAssertTrue(controller.logs.contains { $0.text == "foreground-line" })
        XCTAssertFalse(controller.logs.contains { $0.text == "background-only-line" })

        // The background run still publishes its own result, keyed by its request id.
        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()
        XCTAssertEqual(controller.lastRunResult?.requestID, background.id)
    }

    func testStudioRunDoesNotClobberEditingState() throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"

        // The user is editing a template/draft in the Advanced surface.
        let editing = try XCTUnwrap(CommandCatalog.template(id: .custom))
        var editingDraft = editing.defaultDraft()
        editingDraft.extraArguments = "user-is-editing-this"
        controller.selectedTemplate = editing
        controller.draft = editingDraft

        // A Studio run of a different draft starts from its own snapshot.
        var runDraft = editing.defaultDraft()
        runDraft.extraArguments = "studio-run"
        let request = StudioRunRequest(mode: .chat, templateID: .custom, template: editing, draft: runDraft)
        XCTAssertTrue(controller.run(studio: request))

        // It runs with the request's arguments...
        XCTAssertEqual(runner.starts.first?.configuration.arguments, ["studio-run"])
        // ...while the user's editing state is left untouched.
        XCTAssertEqual(controller.draft.extraArguments, "user-is-editing-this")
    }

    func testRuntimeEndpointIsOwnedNotDerivedFromDraft() {
        let controller = MereRunController(processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false)
        controller.runtimeHost = "example.local"
        controller.runtimePort = 9000
        controller.runtimeAPIKey = "secret"
        // The transient command draft must not influence the runtime endpoint.
        controller.draft.host = "10.0.0.1"
        controller.draft.port = 1234
        controller.draft.apiKey = "draft-key"

        let url = controller.runtimeURL(path: "/runtime/models/foo/load")
        XCTAssertEqual(url.absoluteString, "http://example.local:9000/runtime/models/foo/load")
        XCTAssertEqual(controller.runtimeAuthorizationHeader, "Bearer secret")
    }

    func testRuntimeEndpointFallsBackAndOmitsEmptyAuth() {
        let controller = MereRunController(processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false)
        controller.runtimeHost = "   "
        controller.runtimeAPIKey = "  "
        XCTAssertEqual(controller.runtimeURL(path: "/x").host, "127.0.0.1")
        XCTAssertNil(controller.runtimeAuthorizationHeader)
    }

    func testDetectOutputURLPrefersStdoutContractPath() {
        let probe = StubFileProbe()
        probe.existingPaths = ["/out/render.png"]
        let controller = MereRunController(
            processRunner: RecordingProcessRunner(), fileSystem: probe, resolvesCLIOnInit: false
        )
        let detected = controller.detectOutputURL(expected: nil, stdout: "loading model\n/out/render.png\n")
        XCTAssertEqual(detected?.path, "/out/render.png")
    }

    func testDetectOutputURLResolvesArrowPairRightSide() {
        let probe = StubFileProbe()
        probe.existingPaths = ["/out/page.txt"]
        let controller = MereRunController(
            processRunner: RecordingProcessRunner(), fileSystem: probe, resolvesCLIOnInit: false
        )
        // A whole "in -> out" line is never a path; only the contract parser resolves this.
        let detected = controller.detectOutputURL(expected: nil, stdout: "/in/page.png -> /out/page.txt\n")
        XCTAssertEqual(detected?.path, "/out/page.txt")
    }

    func testDetectOutputURLPrefersExpectedOutputWhenPresent() {
        let probe = StubFileProbe()
        probe.existingPaths = ["/want/out.wav", "/other/x.wav"]
        let controller = MereRunController(
            processRunner: RecordingProcessRunner(), fileSystem: probe, resolvesCLIOnInit: false
        )
        let detected = controller.detectOutputURL(
            expected: URL(fileURLWithPath: "/want/out.wav"), stdout: "/other/x.wav\n"
        )
        XCTAssertEqual(detected?.path, "/want/out.wav")
    }

    func testCLIResolverInjectionIsUsedForResolution() {
        let stubLaunch = MereRunLaunch.executable(URL(fileURLWithPath: "/stub/mere.run"))
        let controller = MereRunController(
            processRunner: RecordingProcessRunner(),
            cliResolver: { _ in stubLaunch },
            resolvesCLIOnInit: false
        )
        controller.refreshResolvedCLI()
        XCTAssertEqual(controller.resolvedCLI, "/stub/mere.run")
    }

    func testRunningStudioRunPublishesOutputBeforeProcessExits() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("MereRunControllerTests-\(UUID().uuidString)", isDirectory: true)
        let output = temp.appendingPathComponent("image.png", isDirectory: false)
        defer { try? FileManager.default.removeItem(at: temp) }
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        var draft = template.defaultDraft()
        draft.prompt = "a blue plate"
        draft.outputPath = output.path
        let request = StudioRunRequest(mode: .createImage, templateID: .imageGenerate, template: template, draft: draft)

        XCTAssertTrue(controller.run(studio: request))
        XCTAssertTrue(controller.isRunning)
        XCTAssertNil(controller.lastOutputURL)

        try Data("png".utf8).write(to: output)
        runner.starts[0].stdout("wrote \(output.path)\n")
        await Task.yield()

        XCTAssertTrue(controller.isRunning)
        XCTAssertEqual(controller.lastOutputURL?.path, output.path)
    }
}

private func supportedCapabilitiesOutput(for modelID: String, minimum: Int) -> String {
    """
    Machine
      processor: M2 Max
      unifiedMemory: 128 GB
      appleSiliconMac: true

    Model capabilities
    - \(modelID) [supported]
      title: \(modelID)
      category: test
      memory: minimum \(minimum) GB, recommended \(minimum) GB
      download: Hugging Face snapshot
    """
}

private func jsonCapabilitiesOutput(recommendedChatModelID: String, supportedModelID: String) -> String {
    """
    {
      "recommendedChatModel" : {
        "modelID" : "\(recommendedChatModelID)"
      },
      "models" : [
        {
          "download" : "hugging-face",
          "id" : "\(supportedModelID)",
          "minimumUnifiedMemoryGB" : 24,
          "reasons" : [],
          "recommendedUnifiedMemoryGB" : 32,
          "supported" : true
        }
      ]
    }
    """
}

private func unsupportedCapabilitiesOutput(for modelID: String, minimum: Int, detected: Int) -> String {
    """
    Machine
      processor: M2 Max
      unifiedMemory: \(detected) GB
      appleSiliconMac: true

    Model capabilities
    - \(modelID) [unsupported]
      title: \(modelID)
      category: test
      memory: minimum \(minimum) GB, recommended \(minimum) GB
      download: Hugging Face snapshot
      reason: Requires at least \(minimum) GB unified memory; detected \(detected) GB.
    """
}

private struct RecordingStart {
    let configuration: MereRunProcessConfiguration
    let stdout: @Sendable (String) -> Void
    let stderr: @Sendable (String) -> Void
    let termination: @Sendable (Int32) -> Void
}

private final class RecordingProcessRunner: MereRunProcessRunning {
    private(set) var starts: [RecordingStart] = []
    private(set) var processes: [RecordingProcess] = []

    func start(
        configuration: MereRunProcessConfiguration,
        stdout: @escaping @Sendable (String) -> Void,
        stderr: @escaping @Sendable (String) -> Void,
        termination: @escaping @Sendable (Int32) -> Void
    ) throws -> MereRunRunningProcess {
        let process = RecordingProcess()
        starts.append(
            RecordingStart(
                configuration: configuration,
                stdout: stdout,
                stderr: stderr,
                termination: termination
            )
        )
        processes.append(process)
        return process
    }
}

private final class RecordingProcess: MereRunRunningProcess {
    private(set) var terminateCallCount = 0

    func terminate() {
        terminateCallCount += 1
    }
}

private final class StubFileProbe: MereRunFileProbing {
    var existingPaths: Set<String> = []
    func fileExists(atPath path: String) -> Bool { existingPaths.contains(path) }
}
