@testable import MereRunApp
import Foundation
import XCTest

@MainActor
final class MereRunControllerTests: XCTestCase {
    func testReadinessCheckDoesNotStartDuplicateInFlightRequest() {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        var draft = StudioDraft()
        draft.reset(for: .createImage)

        controller.checkReadiness(for: .createImage, draft: draft)
        controller.checkReadiness(for: .createImage, draft: draft)

        XCTAssertEqual(runner.starts.count, 1)
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

        runner.starts[0].stdout("ID Category Status Size\nimage-zimage-nano image installed 4 GB\n")
        runner.starts[0].termination(0)
        await Task.yield()
        await Task.yield()
        XCTAssertEqual(controller.readinessByMode[.createImage], .checking)

        runner.starts[1].stdout("ID Category Status Size\nvideo-ltx-av media installed 12 GB\n")
        runner.starts[1].termination(0)
        await Task.yield()
        await Task.yield()
        XCTAssertEqual(controller.readinessByMode[.createImage], .ready)
    }
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
