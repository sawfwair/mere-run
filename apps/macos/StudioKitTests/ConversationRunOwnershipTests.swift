import Combine
import Foundation
import StudioTestSupport
import XCTest
@testable import StudioKit

@MainActor
final class ConversationRunOwnershipTests: XCTestCase {
    private var root: URL!
    private var runner: RecordingProcessRunner!
    private var controller: MereRunController!

    override func setUp() async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("conversation-runs-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        runner = RecordingProcessRunner()
        controller = MereRunController(
            secretStore: InMemorySecretStore(), processRunner: runner, resolvesCLIOnInit: false,
            taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json"))
        )
        controller.cliPath = "/usr/bin/true"
    }

    override func tearDown() async throws {
        controller.terminateAllProcesses()
        controller = nil
        runner = nil
        try FileManager.default.removeItem(at: root)
    }

    func testControllerRejectsASecondTurnWhileTheConversationIsRunning() throws {
        let conversation = UUID()
        XCTAssertTrue(controller.run(studio: try request(conversation)))
        XCTAssertFalse(controller.run(studio: try request(conversation)))
        XCTAssertEqual(runner.starts.count, 1)
        XCTAssertEqual(controller.runningConversationIDs, [conversation])
    }

    func testQueuedTurnAlsoOwnsItsConversation() throws {
        XCTAssertTrue(controller.run(studio: try request(UUID())))
        XCTAssertTrue(controller.run(studio: try request(UUID())))
        let conversation = UUID()
        XCTAssertTrue(controller.run(studio: try request(conversation)))
        XCTAssertFalse(controller.run(studio: try request(conversation)))
        XCTAssertEqual(controller.queuedRunCount, 1)
        XCTAssertEqual(runner.starts.count, 2)
        XCTAssertTrue(controller.runningConversationIDs.contains(conversation))
    }

    func testConversationRemainsOwnedThroughoutCompletionDelivery() async throws {
        let controller = try XCTUnwrap(self.controller)
        let conversation = UUID()
        let next = try request(conversation)
        XCTAssertTrue(controller.run(studio: try request(conversation)))
        var delivered = false
        var acceptedDuringDelivery: Bool?
        let subscription = controller.runCompletions.sink { [controller] result in
            guard result.conversationID == conversation else { return }
            delivered = true
            XCTAssertTrue(controller.runningConversationIDs.contains(conversation))
            acceptedDuringDelivery = controller.run(studio: next)
        }
        defer { subscription.cancel() }

        runner.starts[0].termination(0)
        await settle()

        XCTAssertTrue(delivered)
        XCTAssertEqual(acceptedDuringDelivery, false)
        XCTAssertFalse(controller.runningConversationIDs.contains(conversation))
        XCTAssertTrue(controller.run(studio: next), "the next turn is admitted after all reply subscribers return")
        XCTAssertEqual(runner.starts.count, 2)
    }

    func testReservationIsVisibleToSynchronousChangeObservers() throws {
        let controller = try XCTUnwrap(self.controller)
        let conversation = UUID()
        let second = try request(conversation)
        var attempted = false
        var accepted: Bool?
        let subscription = controller.objectWillChange.sink {
            guard !attempted else { return }
            attempted = true
            accepted = controller.run(studio: second)
        }
        defer { subscription.cancel() }

        XCTAssertTrue(controller.run(studio: try request(conversation)))
        XCTAssertTrue(attempted)
        XCTAssertEqual(accepted, false)
        XCTAssertEqual(runner.starts.count, 1)
    }

    func testRefusalWithoutAJobDoesNotLeaveTheConversationBusy() throws {
        let library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
        try controller.setPrivateMode(true, savedLibrary: library)
        let refused = try request(UUID())

        XCTAssertFalse(controller.run(studio: refused), "custom commands cannot declare private outputs")
        XCTAssertNil(controller.jobs.job(requestID: refused.id))
        XCTAssertTrue(controller.runningConversationIDs.isEmpty)
        XCTAssertTrue(runner.starts.isEmpty)

        try controller.setPrivateMode(false, savedLibrary: library)
        XCTAssertTrue(controller.run(studio: refused), "a rejected submission must not reserve the thread")
    }

    func testLaunchFailureReleasesTheConversationForRetry() throws {
        let conversation = UUID()
        runner.launchError = StubLaunchError()
        XCTAssertFalse(controller.run(studio: try request(conversation)))
        XCTAssertFalse(controller.runningConversationIDs.contains(conversation))
        XCTAssertTrue(controller.run(studio: try request(conversation)))
        XCTAssertEqual(runner.starts.count, 1)
    }

    func testLateOutputFromAPreviousTurnCannotReplaceTheCurrentReply() async throws {
        let conversation = UUID()
        XCTAssertTrue(controller.run(studio: try request(conversation)))
        runner.starts[0].termination(0)
        await settle()
        XCTAssertTrue(controller.run(studio: try request(conversation)))
        runner.starts[1].stdout("new reply")
        await settle()
        runner.starts[0].stdout("old reply")
        await settle()

        XCTAssertEqual(controller.conversationLiveReplies[conversation]?.answer, "new reply")
        XCTAssertTrue(controller.runningConversationIDs.contains(conversation))
    }

    private func request(_ conversationID: UUID) throws -> StudioRunRequest {
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        var draft = template.defaultDraft()
        draft.extraArguments = "turn"
        return StudioRunRequest(mode: .chat, templateID: .custom, template: template, draft: draft,
                                conversationID: conversationID)
    }

    private func settle() async {
        for _ in 0..<10 { await Task.yield() }
    }
}
