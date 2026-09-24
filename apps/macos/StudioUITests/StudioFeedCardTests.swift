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

    /// What `text chat` wrote for a model that is not installed, and what `music train-adapter`
    /// wrote for an argument it could not read: ArgumentParser's usage trailer follows the error
    /// in both, and the trailer is never the reason.
    func testFailureSummarySkipsArgumentParsersUsageTrailer() {
        let notInstalled = """
        Queued by machine admission: text chat (4/4 permits active, 1 queued).
        Machine admission granted: text chat.
        Error: Model 'text-chat-does-not-exist-xyz' is not installed. Run 'mere.run model pull text-chat-does-not-exist-xyz' explicitly.
        Usage: mere.run [--models-root <models-root>] <subcommand>
          See 'mere.run --help' for more information.
        """
        XCTAssertEqual(
            StudioFailureSummary.summary(outputText: nil, logLines: notInstalled.components(separatedBy: .newlines), exitCode: 64),
            "Model 'text-chat-does-not-exist-xyz' is not installed. Run 'mere.run model pull text-chat-does-not-exist-xyz' explicitly."
        )
        let missingValue = """
        Error: Missing value for '--factor <factor>'
        Help:  --factor <factor>  LoKr factorization target; -1 chooses the closest balanced factors.
        Usage: mere.run music train-adapter [<options>] --dataset <dataset> --output <output>
          See 'mere.run music train-adapter --help' for more information.
        """
        XCTAssertEqual(StudioFailureSummary.summary(outputText: missingValue, exitCode: 64), "Missing value for '--factor <factor>'")
        XCTAssertEqual(StudioFailureSummary.lastMeaningfulLine(in: "USAGE: mere.run vision segment <image>\nSee 'mere.run --help'"), nil)
    }

    /// What `speech diarize` wrote for a model that is not on this Mac: the reason, then the
    /// upstream repo and every location it searched as a list. The reason leads the card; a
    /// searched path, the `Searched:` heading, and the repo line are never the summary.
    func testFailureSummaryPrefersTheErrorLineOverTheLocationsItSearched() {
        let stderr = """
        Error: Model not found: speech-diarization-nemotron3
        Upstream repo: nvidia/diar_streaming_sortformer_4spk-v2.1
        Searched:
        - /Users/example/.mere.run/models/speech-diarization-nemotron3
        - /Volumes/SALVATION/models/speech-diarization-nemotron3
        """
        XCTAssertEqual(
            StudioFailureSummary.summary(outputText: nil, logLines: stderr.components(separatedBy: .newlines), exitCode: 1),
            "Model not found: speech-diarization-nemotron3"
        )
        XCTAssertEqual(StudioFailureSummary.summary(outputText: stderr, exitCode: 1), "Model not found: speech-diarization-nemotron3")
        XCTAssertEqual(StudioFailureSummary.lastMeaningfulLine(in: stderr), "Model not found: speech-diarization-nemotron3")
        XCTAssertFalse(StudioFailureSummary.isMeaningful("- /Volumes/SALVATION/models/speech-diarization-nemotron3"))
        XCTAssertFalse(StudioFailureSummary.isMeaningful("Searched:"))
        XCTAssertFalse(StudioFailureSummary.isMeaningful("Upstream repo: nvidia/diar_streaming_sortformer_4spk-v2.1"))
        // An error line wins over prose printed after it, but only when the CLI marked one.
        XCTAssertEqual(
            StudioFailureSummary.summary(outputText: "error: no such root\nCleaning up temporary files", exitCode: 1),
            "No such root"
        )
        XCTAssertEqual(StudioFailureSummary.summary(outputText: "Loading model\nOut of memory", exitCode: 1), "Out of memory")
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
        XCTAssertEqual(StudioFeedChips.chips(for: item, titles: .none), ["1024×1024", "4 steps", "seed 8812", "Zimage Nano"])

        draft.seed = ""
        item.commandDraft = draft
        XCTAssertEqual(StudioFeedChips.chips(for: item, titles: .none)[2], "seed random")
        item.commandDraft = nil
        XCTAssertEqual(StudioFeedChips.chips(for: item, titles: .none), [])
    }

    /// A failed card offers Get the model only when that is what went wrong: the run's model
    /// is one the inventory lists as missing (not invalid, offline, or awaiting conversion,
    /// which a pull would not fix) and the failure line names it. A model removed after an
    /// unrelated failure keeps the CLI's own reason.
    func testFailureCardOffersTheModelOnlyWhenTheRunFailedForWantOfIt() throws {
        var draft = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate)).defaultDraft()
        draft.model = "image-zimage-turbo"
        var failed = row(id: UUID(), prompt: "p", status: .failed, createdAt: Date())
        failed.commandDraft = draft
        func inventory(_ status: String) -> [StudioModelInventoryRow] {
            [StudioModelInventoryRow(id: "image-zimage-turbo", category: "image", status: status, size: "—", usageTerms: nil)]
        }
        let aboutTheModel = "Image-zimage-turbo is not installed. Run `mere.run model pull image-zimage-turbo`."
        let unrelated = "Out of memory"

        XCTAssertEqual(
            StudioFailureSummary.missingModel(for: failed, in: inventory("missing"), failureLine: aboutTheModel)?.id,
            "image-zimage-turbo"
        )
        for status in ["invalid", "offline", "conversion-required", "installed"] {
            XCTAssertNil(StudioFailureSummary.missingModel(for: failed, in: inventory(status), failureLine: aboutTheModel), status)
        }
        XCTAssertNil(
            StudioFailureSummary.missingModel(for: failed, in: inventory("missing"), failureLine: unrelated),
            "a model removed after an unrelated failure keeps the CLI's reason"
        )
        XCTAssertNil(StudioFailureSummary.missingModel(for: failed, in: [], failureLine: aboutTheModel), "a model the inventory does not list")

        var cancelled = failed
        cancelled.status = .cancelled
        XCTAssertNil(StudioFailureSummary.missingModel(for: cancelled, in: inventory("missing"), failureLine: aboutTheModel))
        failed.commandDraft = nil
        XCTAssertNil(StudioFailureSummary.missingModel(for: failed, in: inventory("missing"), failureLine: aboutTheModel), "no recorded model")
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
