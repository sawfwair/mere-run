@testable import MereRunApp
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
            Array(runner.starts[0].configuration.arguments.suffix(3)),
            ["model", "capabilities", "--all"]
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

    func testStudioRunsQueueBehindActiveProcess() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        var firstDraft = template.defaultDraft()
        firstDraft.extraArguments = "first"
        var secondDraft = template.defaultDraft()
        secondDraft.extraArguments = "second"
        let first = StudioRunRequest(mode: .chat, templateID: .custom, template: template, draft: firstDraft)
        let second = StudioRunRequest(mode: .code, templateID: .custom, template: template, draft: secondDraft)

        XCTAssertTrue(controller.run(studio: first))
        XCTAssertTrue(controller.isRunning)
        XCTAssertEqual(controller.activeRunRequestID, first.id)
        XCTAssertEqual(runner.starts.count, 1)

        XCTAssertTrue(controller.run(studio: second))
        XCTAssertEqual(controller.queuedRunCount, 1)
        XCTAssertEqual(runner.starts.count, 1)

        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()

        XCTAssertEqual(controller.lastRunResult?.requestID, first.id)
        XCTAssertEqual(controller.activeRunRequestID, second.id)
        XCTAssertEqual(controller.queuedRunCount, 0)
        XCTAssertEqual(runner.starts.count, 2)
        XCTAssertEqual(runner.starts[1].configuration.arguments, ["second"])

        runner.starts[1].termination(0)
        await Task.yield()
        await Task.yield()

        XCTAssertFalse(controller.isRunning)
        XCTAssertNil(controller.activeRunRequestID)
        XCTAssertEqual(controller.lastRunResult?.requestID, second.id)
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
