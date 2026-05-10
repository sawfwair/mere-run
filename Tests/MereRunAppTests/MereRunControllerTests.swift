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

        runner.starts[0].stdout(supportedCapabilitiesOutput(for: "image-zimage-max", minimum: 48))
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
                for: "image-zimage-max",
                minimum: 48,
                detected: 32
            )
        )
        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()

        XCTAssertEqual(runner.starts.count, 1)
        XCTAssertEqual(
            controller.readinessByMode[.createImage],
            .unsupported("Requires at least 48 GB unified memory; detected 32 GB.")
        )
        XCTAssertEqual(controller.modelCapabilitiesByID["image-zimage-max"]?.minimumUnifiedMemoryGB, 48)
    }

    func testReadinessBlocksUnavailableStudioCapabilityWithoutStartingCLI() {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var draft = StudioDraft()
        draft.reset(for: .readImage)
        let expectedMessage = "Inspect uses an automatic vision-language model download "
            + "that is not listed in the managed capability catalog yet."

        controller.checkReadiness(for: .readImage, draft: draft)

        XCTAssertTrue(runner.starts.isEmpty)
        XCTAssertEqual(
            controller.readinessByMode[.readImage],
            .unsupported(expectedMessage)
        )
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
