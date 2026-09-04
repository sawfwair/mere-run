@testable import StudioKit
@testable import StudioUI
import Foundation
import StudioTestSupport
import XCTest

/// The feed's cards derive from Library rows plus the jobs still alive in the store; the failure
/// card leads with one meaningful line; the running card's status comes from structured progress.
@MainActor
final class StudioFeedCardTests: XCTestCase {
    func testCardsFollowJobStateWhileAliveAndTheRowOnceGone() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let running = try makeRequest(requestID: UUID(), prompt: "running")
        let second = try makeRequest(requestID: UUID(), prompt: "also running")
        let queued = try makeRequest(requestID: UUID(), prompt: "queued")
        store.submit(running)
        store.submit(second)
        store.submit(queued)

        let now = Date()
        let items = [
            row(id: try XCTUnwrap(queued.requestID), prompt: "queued", status: .queued, createdAt: now),
            row(id: try XCTUnwrap(running.requestID), prompt: "running", status: .running, createdAt: now.addingTimeInterval(-10)),
            row(id: UUID(), prompt: "done earlier", status: .completed, createdAt: now.addingTimeInterval(-60)),
            row(id: UUID(), prompt: "failed earlier", status: .failed, createdAt: now.addingTimeInterval(-30)),
            row(id: UUID(), prompt: "other mode", status: .completed, createdAt: now, mode: .video),
        ]
        let cards = StudioFeedCardBuilder.cards(items: items, mode: .createImage, job: store.job(requestID:))

        XCTAssertEqual(cards.map(\.item.prompt), ["done earlier", "failed earlier", "running", "queued"], "oldest first, this mode only")
        XCTAssertEqual(cards.map(\.kind), [.generation, .failed, .running, .queued])
        XCTAssertNotNil(cards[2].job)
        XCTAssertNil(cards[0].job, "rows from an earlier session have no job")
        XCTAssertEqual(StudioFeedCardBuilder.queuePosition(of: cards[3], in: cards), 0)

        // The job's state wins over a stale row status while the store still has the job.
        runner.starts[0].termination(0)
        for _ in 0..<6 { await Task.yield() }
        XCTAssertEqual(StudioFeedCardBuilder.kind(for: items[1], job: store.job(requestID: running.requestID!)), .generation)
    }

    func testConversationsNeverAppearInTheFeed() {
        let thread = StudioLibraryItem(
            id: UUID(), mode: .chat, prompt: "", inputURL: nil, outputURL: nil, createdAt: Date(), updatedAt: Date(),
            status: .completed, exitCode: 0, commandPreview: "", outputText: nil,
            messages: [StudioMessage(role: .user, content: "hi", createdAt: Date())]
        )
        XCTAssertEqual(StudioFeedCardBuilder.cards(items: [thread], mode: .chat, job: { _ in nil }), [])
    }

    func testFailureSummaryPicksTheLastMeaningfulLine() {
        let text = """
        Loading model image-zimage-nano
        {"event":"progress","stage":"denoising","step":2,"total_steps":4}
        Traceback (most recent call last):
          File "x.py", line 3
        error: model image-zimage-nano is not installed. Run `mere.run model pull image-zimage-nano`.

        STDERR
        Exited with code 1.
        """
        XCTAssertEqual(
            StudioFailureSummary.summary(outputText: text, exitCode: 1),
            "Model image-zimage-nano is not installed. Run `mere.run model pull image-zimage-nano`."
        )
        XCTAssertEqual(StudioFailureSummary.summary(outputText: nil, exitCode: 15), "Cancelled.")
        XCTAssertEqual(StudioFailureSummary.summary(outputText: "", exitCode: 64), "The request was invalid.")
        XCTAssertEqual(StudioFailureSummary.summary(outputText: "   ", exitCode: 3), "The run exited with code 3.")
        XCTAssertEqual(
            StudioFailureSummary.summary(outputText: "old text", logLines: ["Generating (1/4)", "mere.run: out of memory"], exitCode: 1),
            "Out of memory",
            "log lines are newer than the captured text and win"
        )
    }

    func testRunningStatusCompactsStepProgress() {
        let progress = StudioProgressParser.parse(#"{"event":"progress","stage":"denoising","step":14,"total_steps":24}"#)
        XCTAssertEqual(StudioRunningStatus.text(progress: progress, fallback: "Running"), "Denoising 15/24")
        XCTAssertEqual(progress?.fractionCompleted ?? 0, 0.625, accuracy: 0.001)
        XCTAssertEqual(StudioRunningStatus.text(progress: nil, fallback: "Running"), "Running")
        let download = StudioRunProgress(label: "Downloading", fractionCompleted: nil, detail: "1.2 GB / 4.8 GB")
        XCTAssertEqual(StudioRunningStatus.text(progress: download, fallback: ""), "Downloading · 1.2 GB / 4.8 GB")
    }

    func testFeedChipsReadTheRunsOwnCommand() throws {
        var draft = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate)).defaultDraft()
        draft.width = 1024
        draft.height = 1024
        draft.steps = 4
        draft.seed = "8812"
        draft.model = "image-zimage-nano"
        var item = row(id: UUID(), prompt: "p", status: .completed, createdAt: Date())
        item.commandDraft = draft
        XCTAssertEqual(StudioFeedChips.chips(for: item), ["1024×1024", "4 steps", "seed 8812", "Zimage Nano"])

        draft.seed = ""
        item.commandDraft = draft
        XCTAssertEqual(StudioFeedChips.chips(for: item)[2], "seed random")
        item.commandDraft = nil
        XCTAssertEqual(StudioFeedChips.chips(for: item), [])
    }

    func testFeedTimeShowsClockTodayAndDayOtherwise() {
        let calendar = Calendar(identifier: .gregorian)
        let now = Date()
        XCTAssertFalse(StudioFeedTime.label(for: now, now: now, calendar: calendar).isEmpty)
        let earlier = calendar.date(byAdding: .day, value: -40, to: now)!
        let label = StudioFeedTime.label(for: earlier, now: now, calendar: calendar)
        XCTAssertFalse(label.contains(":"), "an older run shows its day, not a clock time: \(label)")
    }

    // MARK: Helpers

    private func row(
        id: UUID, prompt: String, status: StudioLibraryStatus, createdAt: Date, mode: StudioMode = .createImage
    ) -> StudioLibraryItem {
        StudioLibraryItem(
            id: id, mode: mode, prompt: prompt, inputURL: nil, outputURL: nil, createdAt: createdAt,
            updatedAt: createdAt, status: status, exitCode: status == .failed ? 1 : nil,
            commandPreview: "mere.run image generate", outputText: nil, templateID: .imageGenerate
        )
    }

    private func makeRequest(requestID: UUID?, prompt: String) throws -> JobRequest {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.prompt = prompt
        draft.outputPath = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).png").path
        let args = template.arguments(from: draft)
        return JobRequest(
            lane: .inference,
            template: template,
            draft: draft,
            requestID: requestID,
            configuration: MereRunProcessConfiguration(
                executableURL: URL(fileURLWithPath: "/usr/bin/true"),
                arguments: args,
                currentDirectoryURL: FileManager.default.temporaryDirectory,
                environment: [:],
                keepsStandardInputOpen: false
            ),
            displayCommand: (["mere.run"] + args).shellQuoted()
        )
    }
}
