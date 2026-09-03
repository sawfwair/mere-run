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

    func testReadinessProbesRunInTheProbeLaneAndReuseTheInFlightProbeAcrossAModelChange() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var draft = StudioDraft()
        draft.reset(for: .createImage)

        controller.checkReadiness(for: .createImage, draft: draft)
        draft.model = "video-ltx-av"
        controller.checkReadiness(for: .createImage, draft: draft)

        // `model capabilities --all --json` does not depend on the model, so the second request
        // rides the probe already in flight instead of killing it and launching an identical one.
        XCTAssertEqual(runner.starts.count, 1)
        XCTAssertEqual(runner.processes[0].terminateCallCount, 0)
        XCTAssertEqual(controller.jobs.running(in: .probe).count, 1)
        XCTAssertEqual(controller.jobs.running(in: .probe).first?.request.dedupeKey, StudioMode.createImage.rawValue)
        XCTAssertTrue(controller.jobs.running(in: .utility).isEmpty)
        XCTAssertTrue(controller.jobs.running(in: .inference).isEmpty)
        XCTAssertEqual(controller.readinessByMode[.createImage], .checking)

        runner.starts[0].stdout(supportedCapabilitiesOutput(for: "video-ltx-av", minimum: 64))
        runner.starts[0].termination(0)
        await settle()
        XCTAssertEqual(controller.readinessByMode[.createImage], .checking)
        XCTAssertEqual(runner.starts.count, 2)
        XCTAssertEqual(Array(runner.starts[1].configuration.arguments.suffix(2)), ["model", "list"])

        // The result is evaluated against the model current at completion: the new one is
        // installed, the original is not.
        runner.starts[1].stdout(
            "ID Category Status Size\nvideo-ltx-av media installed 12 GB\nimage-zimage-nano media missing 1 GB\n"
        )
        runner.starts[1].termination(0)
        await settle()
        XCTAssertEqual(controller.readinessByMode[.createImage], .ready)
        XCTAssertTrue(controller.jobs.running.isEmpty)
    }

    func testReadinessSettingsChangeSupersedesTheInFlightProbeAndIgnoresItsResult() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        // Settings persist through UserDefaults; leave the test domain as it was found.
        let originalModelsRoot = controller.modelsRoot
        defer { controller.modelsRoot = originalModelsRoot }
        controller.modelsRoot = ""
        var draft = StudioDraft()
        draft.reset(for: .createImage)

        controller.checkReadiness(for: .createImage, draft: draft)
        controller.modelsRoot = "/Volumes/Models"
        controller.checkReadiness(for: .createImage, draft: draft)

        XCTAssertEqual(runner.starts.count, 2)
        XCTAssertEqual(runner.processes[0].terminateCallCount, 1)
        XCTAssertFalse(runner.starts[0].configuration.arguments.contains("--models-root"))
        XCTAssertTrue(runner.starts[1].configuration.arguments.contains("--models-root"))

        // The superseded probe still reports before it dies; nothing of it is applied.
        runner.starts[0].stdout(supportedCapabilitiesOutput(for: "image-zimage-nano", minimum: 12))
        runner.starts[0].termination(0)
        await settle()
        XCTAssertEqual(controller.readinessByMode[.createImage], .checking)
        XCTAssertTrue(controller.modelCapabilitiesByID.isEmpty)
        XCTAssertEqual(runner.starts.count, 2)

        runner.starts[1].stdout(supportedCapabilitiesOutput(for: "image-zimage-nano", minimum: 12))
        runner.starts[1].termination(0)
        await settle()
        XCTAssertEqual(controller.modelCapabilitiesByID["image-zimage-nano"]?.minimumUnifiedMemoryGB, 12)
        XCTAssertEqual(runner.starts.count, 3)

        runner.starts[2].stdout("ID Category Status Size\nimage-zimage-nano media installed 1 GB\n")
        runner.starts[2].termination(0)
        await settle()
        XCTAssertEqual(controller.readinessByMode[.createImage], .ready)
    }

    func testReadinessProbeIsCancelledWhenTheModeStopsNeedingAModel() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var draft = StudioDraft()
        draft.reset(for: .readImage)
        draft.readImageAction = .ocr

        controller.checkReadiness(for: .readImage, draft: draft)
        XCTAssertEqual(runner.starts.count, 1)
        XCTAssertEqual(controller.readinessByMode[.readImage], .checking)

        draft.readImageAction = .inspect
        controller.checkReadiness(for: .readImage, draft: draft)
        XCTAssertEqual(controller.readinessByMode[.readImage], .ready)
        XCTAssertEqual(runner.processes[0].terminateCallCount, 1)

        runner.starts[0].stdout(supportedCapabilitiesOutput(for: "vision-ocr-lighton", minimum: 8))
        runner.starts[0].termination(15)
        await settle()
        XCTAssertEqual(controller.readinessByMode[.readImage], .ready)
        XCTAssertEqual(runner.starts.count, 1)
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
        await settle()

        XCTAssertEqual(runner.starts.count, 1)
        XCTAssertEqual(
            controller.readinessByMode[.createImage],
            .unsupported("Requires at least 12 GB unified memory; detected 8 GB.")
        )
        XCTAssertEqual(controller.modelCapabilitiesByID["image-zimage-nano"]?.minimumUnifiedMemoryGB, 12)
    }

    func testReadinessCapturesRecommendedModelsFromJSONCapabilities() async throws {
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
            recommendedCodeModelID: "text-code-north-mini",
            supportedModelID: StudioChatDefaults.fallbackModelID
        ))
        runner.starts[0].termination(0)
        await settle()

        XCTAssertEqual(controller.recommendedChatModelID, "text-agent-deepseek-v4-flash")
        XCTAssertEqual(controller.recommendedCodeModelID, "text-code-north-mini")
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

    func testSavingHuggingFaceTokenKeepsSecretOutOfProcessArguments() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"

        let task = Task { await controller.saveHuggingFaceToken(" hf_secret ") }
        await Task.yield()

        XCTAssertEqual(runner.starts.count, 1)
        let configuration = runner.starts[0].configuration
        XCTAssertEqual(
            Array(configuration.arguments.suffix(5)),
            ["config", "set", "hf-token", "--from-env", "MERERUN_CONFIG_VALUE"]
        )
        XCTAssertFalse(configuration.arguments.contains("hf_secret"))
        XCTAssertEqual(configuration.environment["MERERUN_CONFIG_VALUE"], "hf_secret")

        runner.starts[0].termination(0)
        let didSave = await task.value
        XCTAssertTrue(didSave)
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

    func testUtilityCommandCanBeCancelledByStableID() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let commandID = UUID()

        let task = Task {
            await controller.utilityCommandResult(
                args: ["model", "pull", "image-klein-nano"],
                commandID: commandID
            )
        }
        await Task.yield()

        XCTAssertEqual(runner.starts.count, 1)
        XCTAssertTrue(controller.cancelUtilityCommand(commandID))
        XCTAssertEqual(runner.processes.first?.terminateCallCount, 1)
        XCTAssertFalse(controller.cancelUtilityCommand(UUID()))

        runner.starts[0].termination(15)
        let result = await task.value
        XCTAssertEqual(result.exitCode, 15)
    }

    func testResidentVideoSessionKeepsInputOpenAndPublishesEachResult() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .videoSession))
        controller.select(template)
        controller.draft.prompt = "A paper kite crossing a storm front"
        controller.draft.outputPath = "/tmp/resident-kite.mp4"
        controller.draft.imagePath = "/tmp/start.png"
        controller.draft.endImagePath = "/tmp/end.png"
        controller.draft.seed = "73"

        XCTAssertTrue(controller.run())
        XCTAssertEqual(runner.starts.count, 1)
        XCTAssertTrue(runner.starts[0].configuration.keepsStandardInputOpen)

        runner.starts[0].stderr("LTX video session ready: /tmp/model\n")
        await Task.yield()
        XCTAssertEqual(controller.status, "Resident session ready")
        XCTAssertTrue(controller.canSubmitVideoSessionRequest)
        XCTAssertTrue(controller.submitVideoSessionRequest())

        let input = try XCTUnwrap(runner.processes[0].standardInputs.first)
        let payload = try XCTUnwrap(input.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: payload) as? [String: Any])
        XCTAssertEqual(json["prompt"] as? String, "A paper kite crossing a storm front")
        XCTAssertEqual(json["output"] as? String, "/tmp/resident-kite.mp4")
        XCTAssertEqual(json["image"] as? String, "/tmp/start.png")
        XCTAssertEqual(json["end_image"] as? String, "/tmp/end.png")
        XCTAssertEqual(json["seed"] as? Int, 73)

        runner.starts[0].stdout(#"{"output":"/tmp/resident-kite.mp4","status":"res"#)
        runner.starts[0].stdout("ult\"}\n")
        await Task.yield()
        await Task.yield()

        XCTAssertEqual(controller.lastOutputURL?.path, "/tmp/resident-kite.mp4")
        XCTAssertEqual(controller.status, "Generated: resident-kite.mp4")
        XCTAssertTrue(controller.canSubmitVideoSessionRequest)
    }

    func testRealtimeMusicKeepsInputOpenAndAcceptsTargetedLiveControls() throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicRealtime))
        var draft = template.defaultDraft()
        draft.prompt = "ambient glass percussion"
        draft.durationSeconds = 60
        draft.musicInteractive = true
        let request = StudioRunRequest(
            mode: .music,
            templateID: .musicRealtime,
            template: template,
            draft: draft
        )

        XCTAssertTrue(controller.run(studio: request))
        XCTAssertTrue(runner.starts[0].configuration.keepsStandardInputOpen)
        XCTAssertTrue(controller.canSteerRealtimeMusic(requestID: request.id))
        XCTAssertTrue(controller.submitRealtimeMusicCommand("temp 0.8", requestID: request.id))
        XCTAssertEqual(runner.processes[0].standardInputs, ["temp 0.8\n"])
    }

    func testServiceRunCanBeRediscoveredLoggedAndCancelledByRequestID() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .apiServe))
        var draft = template.defaultDraft()
        draft.prompt = "http://127.0.0.1:8080"
        let request = StudioRunRequest(
            mode: .chat,
            templateID: .apiServe,
            template: template,
            draft: draft
        )

        XCTAssertTrue(controller.run(studio: request))
        XCTAssertTrue(controller.isRequestRunning(request.id))
        XCTAssertEqual(controller.runningRequestID(for: .apiServe), request.id)

        runner.starts[0].stderr("Starting local API service\n")
        await Task.yield()
        XCTAssertTrue(controller.logs(for: request.id).contains {
            $0.text.contains("Starting local API service")
        })

        XCTAssertTrue(controller.cancel(requestID: request.id))
        XCTAssertEqual(runner.processes[0].terminateCallCount, 1)
        XCTAssertFalse(controller.cancel(requestID: UUID()))
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

    func testBackgroundStudioRunPublishesProgressByRequestID() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        func request(_ arg: String) -> StudioRunRequest {
            var draft = template.defaultDraft()
            draft.extraArguments = arg
            return StudioRunRequest(mode: .createImage, templateID: .custom, template: template, draft: draft)
        }
        let background = request("background")
        let foreground = request("foreground")

        XCTAssertTrue(controller.run(studio: background))
        XCTAssertTrue(controller.run(studio: foreground))
        runner.starts[0].stderr("Training (2/4) loss 0.123456\n")
        await Task.yield()

        let progress = try XCTUnwrap(controller.progressByRequestID[background.id])
        XCTAssertEqual(progress.label, "Training")
        XCTAssertEqual(progress.fractionCompleted, 0.5)
        XCTAssertTrue(progress.detail?.contains("loss 0.123456") == true)
        XCTAssertNil(controller.currentProgress)

        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()
        XCTAssertNil(controller.progressByRequestID[background.id])
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

    func testAdvancedSurfaceFollowsStudioModeChanges() {
        let controller = MereRunController(processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false)
        var imageDraft = StudioDraft()
        imageDraft.reset(for: .createImage)
        imageDraft.prompt = "an image prompt"
        controller.syncAdvanced(to: .createImage, from: imageDraft)

        XCTAssertEqual(controller.selectedTemplate.id, .imageGenerate)
        XCTAssertEqual(controller.draft.prompt, "an image prompt")

        var videoDraft = StudioDraft()
        videoDraft.reset(for: .video)
        videoDraft.prompt = "a video prompt"
        videoDraft.inputPath = "/tmp/start.png"
        videoDraft.audioPath = "/tmp/song.wav"
        videoDraft.videoOutputMode = .audioVideo
        controller.syncAdvanced(to: .video, from: videoDraft)

        XCTAssertEqual(controller.selectedTemplate.id, .videoGenerate)
        XCTAssertEqual(controller.draft.prompt, "a video prompt")
        XCTAssertEqual(controller.draft.inputPath, "/tmp/start.png")
        XCTAssertEqual(controller.draft.audioPath, "/tmp/song.wav")
        XCTAssertEqual(controller.draft.videoOutputMode, .audioVideo)
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

    func testUtilityCommandCanStreamProgressWhileStillReturningCapturedOutput() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var streamed: [String] = []

        let pending = Task {
            await controller.utilityCommandResult(
                args: ["executor", "login", "relay:fleet", "--json"],
                onOutput: { streamed.append($0) }
            )
        }
        while runner.starts.isEmpty {
            await Task.yield()
        }
        XCTAssertEqual(runner.starts.count, 1)

        runner.starts[0].stderr("Open https://relay.example/device and enter code MERE-42.\n")
        runner.starts[0].stdout(#"{"executor":"relay:fleet"}"#)
        runner.starts[0].termination(0)
        let result = await pending.value
        await Task.yield()

        XCTAssertEqual(result.exitCode, 0)
        XCTAssertTrue(result.stdout.contains("relay:fleet"))
        XCTAssertTrue(streamed.joined().contains("https://relay.example/device"))
    }

    func testUtilityCommandCanStreamStandardOutputWithoutStandardError() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var streamed: [String] = []

        let pending = Task {
            await controller.utilityCommandResult(
                args: ["speech", "listen", "--jsonl"],
                onStandardOutput: { streamed.append($0) }
            )
        }
        while runner.starts.isEmpty { await Task.yield() }

        runner.starts[0].stderr("Loading private checkpoint path.\n")
        runner.starts[0].stdout(#"{"protocol":1,"type":"ready"}"# + "\n")
        runner.starts[0].termination(0)
        _ = await pending.value
        await Task.yield()

        XCTAssertEqual(streamed, [#"{"protocol":1,"type":"ready"}"# + "\n"])
    }

    func testUtilityCommandIsAUtilityLaneJobThatNeverTouchesTheConsole() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var completions = 0
        let subscription = controller.runCompletions.sink { _ in completions += 1 }
        defer { subscription.cancel() }
        let commandID = UUID()

        let pending = Task {
            await controller.utilityCommandResult(args: ["model", "list"], commandID: commandID)
        }
        while runner.starts.isEmpty { await Task.yield() }

        let job = try XCTUnwrap(controller.jobs.job(JobID(raw: commandID)))
        XCTAssertEqual(job.lane, .utility)
        XCTAssertNil(job.request.templateID)
        XCTAssertFalse(job.detectsArtifacts)
        XCTAssertEqual(job.displayCommand, "/usr/bin/true model list")
        XCTAssertEqual(Array(runner.starts[0].configuration.arguments.suffix(2)), ["model", "list"])
        XCTAssertFalse(runner.starts[0].configuration.keepsStandardInputOpen)
        XCTAssertFalse(controller.isRunning)
        XCTAssertEqual(controller.status, "Idle")
        XCTAssertTrue(controller.logs.isEmpty)

        runner.starts[0].stdout("ID Category\nimage-zimage-nano image\n")
        runner.starts[0].stderr("warning: slow disk\n")
        runner.starts[0].termination(0)
        let result = await pending.value

        XCTAssertEqual(result.exitCode, 0)
        XCTAssertEqual(result.stdout, "ID Category\nimage-zimage-nano image\n")
        XCTAssertEqual(result.stderr, "warning: slow disk\n")
        XCTAssertEqual(result.commandPreview, "/usr/bin/true model list")
        XCTAssertEqual(job.result?.templateID, .custom)
        XCTAssertTrue(job.artifacts.isEmpty)
        XCTAssertEqual(completions, 0)
        XCTAssertNil(controller.lastRunResult)
        XCTAssertNil(controller.lastExitCode)
        XCTAssertTrue(controller.logs.isEmpty)
        XCTAssertEqual(controller.status, "Idle")
    }

    func testUtilityLaneQueuesAFifthCommandUntilASlotFrees() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"

        let pending = (0..<5).map { index in
            Task { await controller.utilityCommandResult(args: ["guide", "topic-\(index)", "--json"]) }
        }
        while runner.starts.count < 4 { await Task.yield() }
        await settle()

        XCTAssertEqual(JobLane.utility.capacity, 4)
        XCTAssertEqual(runner.starts.count, 4)
        XCTAssertEqual(controller.jobs.running(in: .utility).count, 4)
        XCTAssertEqual(controller.jobs.queued(in: .utility).count, 1)
        XCTAssertEqual(controller.queuedRunCount, 0, "utility queueing is not the run queue")

        runner.starts[0].termination(0)
        await settle()
        XCTAssertEqual(runner.starts.count, 5)
        XCTAssertTrue(controller.jobs.queued(in: .utility).isEmpty)

        for start in runner.starts.dropFirst() { start.termination(0) }
        for task in pending {
            let result = await task.value
            XCTAssertEqual(result.exitCode, 0)
        }
    }

    func testCancellingAQueuedUtilityCommandResolvesTheAwaitingCaller() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let occupants = (0..<4).map { index in
            Task { await controller.utilityCommandResult(args: ["guide", "topic-\(index)"]) }
        }
        while runner.starts.count < 4 { await Task.yield() }
        let commandID = UUID()
        let queued = Task {
            await controller.utilityCommandResult(args: ["model", "list"], commandID: commandID)
        }
        while controller.jobs.queued(in: .utility).isEmpty { await Task.yield() }

        XCTAssertTrue(controller.cancelUtilityCommand(commandID))
        let result = await queued.value
        XCTAssertEqual(result.exitCode, JobResult.cancelledBeforeStartExitCode)
        XCTAssertEqual(runner.starts.count, 4, "the cancelled command never launched")
        XCTAssertFalse(controller.cancelUtilityCommand(commandID), "a settled command cannot be cancelled twice")

        for start in runner.starts { start.termination(0) }
        for task in occupants { _ = await task.value }
    }

    func testServerStatusProbeDedupesConcurrentRefreshesAndParsesTheSnapshot() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let restoreRuntimeSettings = runtimeSettingsRestorer(for: controller)
        defer { restoreRuntimeSettings() }
        controller.runtimeHost = "127.0.0.1"
        controller.runtimePort = 8080
        controller.runtimeAPIKey = "secret-key"

        let first = Task { await controller.refreshServerStatus() }
        let second = Task { await controller.refreshServerStatus() }
        while runner.starts.isEmpty { await Task.yield() }
        await settle()

        XCTAssertEqual(runner.starts.count, 1)
        XCTAssertEqual(controller.jobs.running(in: .probe).count, 1)
        XCTAssertEqual(
            Array(runner.starts[0].configuration.arguments.suffix(8)),
            ["status", "--json", "--host", "127.0.0.1", "--port", "8080", "--api-key", "secret-key"]
        )
        XCTAssertEqual(
            controller.jobs.running(in: .probe).first?.displayCommand,
            "/usr/bin/true status --json --host 127.0.0.1 --port 8080 --api-key '••••••••'"
        )

        runner.starts[0].stdout("probing...\n")
        runner.starts[0].stdout(
            #"{"server":{"health":"ok","loadedModels":["text-agent-deepseek-v4-flash"]},"#
                + #""installedModels":[{"id":"a"},{"id":"b"}]}"# + "\n"
        )
        runner.starts[0].termination(0)
        await first.value
        await second.value

        XCTAssertEqual(
            controller.serverStatus,
            StudioServerStatus(health: "ok", loadedModels: ["text-agent-deepseek-v4-flash"], installedCount: 2)
        )
        XCTAssertTrue(controller.jobs.running.isEmpty)
    }

    func testSupersededServerStatusProbeKeepsTheLastSnapshot() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let restoreRuntimeSettings = runtimeSettingsRestorer(for: controller)
        defer { restoreRuntimeSettings() }
        controller.runtimePort = 8080

        let initial = Task { await controller.refreshServerStatus() }
        while runner.starts.isEmpty { await Task.yield() }
        runner.starts[0].stdout(#"{"server":{"health":"ok"},"installedModels":[]}"#)
        runner.starts[0].termination(0)
        await initial.value
        XCTAssertEqual(controller.serverStatus?.health, "ok")

        let stale = Task { await controller.refreshServerStatus() }
        while runner.starts.count < 2 { await Task.yield() }
        controller.runtimePort = 9090
        let fresh = Task { await controller.refreshServerStatus() }
        while runner.starts.count < 3 { await Task.yield() }
        XCTAssertEqual(runner.processes[1].terminateCallCount, 1)
        XCTAssertTrue(runner.starts[2].configuration.arguments.contains("9090"))

        runner.starts[1].termination(15)
        await stale.value
        XCTAssertEqual(controller.serverStatus?.health, "ok", "a superseded poll does not blank the pill")

        runner.starts[2].stdout(#"{"server":{"health":"unreachable"},"installedModels":[]}"#)
        runner.starts[2].termination(0)
        await fresh.value
        XCTAssertEqual(controller.serverStatus?.health, "unreachable")
    }

    func testTerminateAllProcessesStopsUtilityCommandsAndReadinessProbes() async {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var draft = StudioDraft()
        draft.reset(for: .createImage)

        let listening = Task {
            await controller.utilityCommandResult(args: ["speech", "listen", "--jsonl"])
        }
        controller.checkReadiness(for: .createImage, draft: draft)
        while runner.starts.count < 2 { await Task.yield() }

        controller.terminateAllProcesses()
        XCTAssertEqual(runner.processes.map(\.terminateCallCount), [1, 1])

        runner.starts[0].termination(15)
        runner.starts[1].termination(15)
        let result = await listening.value
        XCTAssertEqual(result.exitCode, 15)
        await settle()
        XCTAssertTrue(controller.jobs.running.isEmpty)
    }

    func testStudioRunIsObservableAsAJobWhileTheFacadeMirrorsIt() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        var draft = template.defaultDraft()
        draft.extraArguments = "observe"
        let request = StudioRunRequest(mode: .chat, templateID: .custom, template: template, draft: draft)

        XCTAssertTrue(controller.run(studio: request))
        let job = try XCTUnwrap(controller.jobs.job(requestID: request.id))
        XCTAssertEqual(job.lane, .inference)
        XCTAssertTrue(job.state.isRunning)
        // The job carries the launch snapshot the controller built from Settings.
        XCTAssertEqual(job.request.configuration.executableURL.path, "/usr/bin/true")
        XCTAssertEqual(job.request.configuration.arguments, ["observe"])
        XCTAssertTrue(job.request.configuration.environment["PATH"]?.hasPrefix("/opt/homebrew/bin:") == true)
        XCTAssertEqual(job.displayCommand, "/usr/bin/true observe")
        XCTAssertEqual(controller.logs.map(\.text), job.log.lines.map(\.text))

        runner.starts[0].stdout("hello\n")
        await Task.yield()
        XCTAssertEqual(job.liveText, "hello\n")
        XCTAssertEqual(controller.liveOutputText, job.liveText)
        XCTAssertEqual(controller.status, job.status)

        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()
        XCTAssertEqual(job.state.exitCode, 0)
        XCTAssertEqual(controller.lastExitCode, 0)
        XCTAssertFalse(controller.isRunning)
        XCTAssertEqual(controller.lastRunResult, job.result)
        // Finished jobs stay observable, so their logs are still readable by request id.
        XCTAssertTrue(controller.logs(for: request.id).contains { $0.text == "Completed with exit code 0." })
    }

    func testQueuedRunKeepsPreviousForegroundUntilItStarts() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        func request(_ arg: String) -> StudioRunRequest {
            var draft = template.defaultDraft()
            draft.extraArguments = arg
            return StudioRunRequest(mode: .createImage, templateID: .custom, template: template, draft: draft)
        }
        let first = request("first")
        let second = request("second")
        let queued = request("queued")

        XCTAssertTrue(controller.run(studio: first))
        XCTAssertTrue(controller.run(studio: second))
        XCTAssertTrue(controller.run(studio: queued))
        XCTAssertEqual(controller.activeRunRequestID, second.id)
        XCTAssertEqual(controller.queuedRunCount, 1)
        XCTAssertTrue(controller.logs.contains { $0.text == "Queued create image job." })
        XCTAssertEqual(controller.jobs.job(requestID: queued.id)?.state, .queued)
        // A queued run is not "running" for the row-level cancel, exactly as before.
        XCTAssertFalse(controller.isRequestRunning(queued.id))
        XCTAssertFalse(controller.cancel(requestID: queued.id))

        runner.starts[1].termination(0)
        await Task.yield()
        await Task.yield()
        // The dequeued run becomes the foreground the moment it starts.
        XCTAssertEqual(controller.activeRunRequestID, queued.id)
        XCTAssertTrue(controller.isRunning)
        XCTAssertEqual(controller.queuedRunCount, 0)
        XCTAssertEqual(controller.status, "Running")
    }

    func testNonInferenceJobsNeverTouchTheConsoleOrCompletionStream() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var completions = 0
        let subscription = controller.runCompletions.sink { _ in completions += 1 }
        defer { subscription.cancel() }
        let template = try XCTUnwrap(CommandCatalog.template(id: .modelList))
        let draft = template.defaultDraft()
        let request = JobRequest(
            lane: .utility,
            template: template,
            draft: draft,
            configuration: MereRunProcessConfiguration(
                executableURL: URL(fileURLWithPath: "/usr/bin/true"),
                arguments: template.arguments(from: draft),
                currentDirectoryURL: FileManager.default.temporaryDirectory,
                environment: [:],
                keepsStandardInputOpen: false
            ),
            displayCommand: "mere.run model list"
        )

        let id = controller.jobs.submit(request)
        XCTAssertTrue(controller.jobs.job(id)?.state.isRunning == true)
        XCTAssertFalse(controller.isRunning)
        XCTAssertEqual(controller.status, "Idle")
        XCTAssertTrue(controller.logs.isEmpty)

        runner.starts[0].stdout("ID Category\n")
        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()

        XCTAssertEqual(controller.jobs.job(id)?.result?.exitCode, 0)
        XCTAssertNil(controller.lastRunResult)
        XCTAssertNil(controller.lastExitCode)
        XCTAssertEqual(completions, 0)
        XCTAssertTrue(controller.logs.isEmpty)
    }

    func testAdvancedRunRefusesWhenTheInferenceLaneIsFull() throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        controller.select(template)
        controller.draft.extraArguments = "advanced"

        XCTAssertTrue(controller.run())
        XCTAssertTrue(controller.run())
        // Studio runs queue beyond the cap; the Advanced console declines instead.
        XCTAssertFalse(controller.run())
        XCTAssertEqual(runner.starts.count, 2)
        XCTAssertEqual(controller.queuedRunCount, 0)
        XCTAssertTrue(controller.jobs.queued(in: .inference).isEmpty)
    }

    func testDiagnosticsOmitConsoleTextAndCommandArguments() {
        let controller = MereRunController(processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false)
        controller.logs = [
            LogLine(stream: .stdout, text: "generated private answer"),
            LogLine(stream: .stderr, text: "Bearer secret-token")
        ]
        let item = StudioLibraryItem(
            id: UUID(),
            mode: .chat,
            prompt: "private prompt",
            inputURL: nil,
            outputURL: nil,
            createdAt: Date(),
            updatedAt: Date(),
            status: .failed,
            exitCode: 1,
            commandPreview: "mere.run text chat --prompt 'private prompt' --api-key secret-token",
            outputText: "generated private answer"
        )

        let report = controller.diagnosticsReport(libraryItems: [item])

        XCTAssertFalse(report.contains("private prompt"))
        XCTAssertFalse(report.contains("generated private answer"))
        XCTAssertFalse(report.contains("secret-token"))
        XCTAssertTrue(report.contains("Captured lines: 2"))
        XCTAssertTrue(report.contains("Console text omitted"))
    }
}

private extension MereRunControllerTests {
    /// Lets the main-actor hops between a process callback, the store's completion and an
    /// awaiting submitter's continuation run to completion.
    func settle() async {
        for _ in 0..<10 { await Task.yield() }
    }

    /// Runtime settings persist through UserDefaults; a test that changes them puts them back.
    func runtimeSettingsRestorer(for controller: MereRunController) -> @MainActor () -> Void {
        let host = controller.runtimeHost
        let port = controller.runtimePort
        let apiKey = controller.runtimeAPIKey
        return {
            controller.runtimeHost = host
            controller.runtimePort = port
            controller.runtimeAPIKey = apiKey
        }
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

private func jsonCapabilitiesOutput(
    recommendedChatModelID: String,
    recommendedCodeModelID: String = "text-code-north-mini",
    supportedModelID: String
) -> String {
    """
    {
      "recommendedChatModel" : {
        "modelID" : "\(recommendedChatModelID)"
      },
      "recommendedCodeModel" : {
        "id" : "\(recommendedCodeModelID)"
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
