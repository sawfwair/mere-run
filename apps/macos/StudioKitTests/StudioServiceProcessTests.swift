import Foundation
import StudioTestSupport
import XCTest
@testable import StudioKit

@MainActor
final class StudioServiceProcessTests: XCTestCase {
    func testStopInvalidatesARestartWaitingForExit() async throws {
        let (controller, runner, service, draft) = try setup()
        defer { controller.terminateAllProcesses() }
        service.start(draft: draft)
        let restart = Task { await service.restart(draft: draft) }
        await settle()
        XCTAssertTrue(service.stop())
        runner.starts[0].termination(15)
        let restarted = await restart.value
        XCTAssertFalse(restarted)
        XCTAssertEqual(runner.starts.count, 1, "Stop must revoke the pending launch")
    }

    func testNewestRestartOwnsTheReplacementConfiguration() async throws {
        let (controller, runner, service, draft) = try setup()
        defer { controller.terminateAllProcesses() }
        service.start(draft: draft)
        var older = draft
        older.port = 9011
        var newer = draft
        newer.port = 9012
        let first = Task { await service.restart(draft: older) }
        await settle()
        let second = Task { await service.restart(draft: newer) }
        await settle()
        service.start(draft: draft) // Start remains a no-op while the old process is still running.
        runner.starts[0].termination(15)
        let firstRestarted = await first.value
        let secondRestarted = await second.value
        XCTAssertFalse(firstRestarted)
        XCTAssertTrue(secondRestarted)
        XCTAssertEqual(runner.starts.count, 2)
        XCTAssertTrue(runner.starts.last?.configuration.arguments.contains("9012") == true)
    }

    func testCancelledRestartDoesNotLaunchAfterExit() async throws {
        let (controller, runner, service, draft) = try setup()
        defer { controller.terminateAllProcesses() }
        service.start(draft: draft)
        let restart = Task { await service.restart(draft: draft) }
        await settle()
        restart.cancel()
        runner.starts[0].termination(15)
        let restarted = await restart.value
        XCTAssertFalse(restarted)
        XCTAssertEqual(runner.starts.count, 1)
    }

    private func setup() throws -> (MereRunController, RecordingProcessRunner, StudioServiceProcess, CommandDraft) {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: runner, resolvesCLIOnInit: false,
                                           taskSessions: StudioTaskSessions())
        controller.cliPath = "/usr/bin/true"
        let service = StudioServiceProcess(templateID: .apiServe, controller: controller)
        let draft = try XCTUnwrap(CommandCatalog.template(id: .apiServe)).defaultDraft()
        return (controller, runner, service, draft)
    }

    private func settle() async {
        for _ in 0..<10 { await Task.yield() }
    }
}
