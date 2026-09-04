@testable import StudioKit
@testable import StudioUI
import StudioTestSupport
import Foundation
import XCTest

/// What the Activity popover reads out of a `JobStore`: which rows exist and in what order, the
/// copy on each row, the header count, and the footer's version handshake.
@MainActor
final class StudioActivityTests: XCTestCase {
    func testRowsListRunningJobsBeforeQueuedAndNeverListProbes() throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)

        let first = store.submit(try makeRequest(templateID: .imageGenerate))
        let second = store.submit(try makeRequest(templateID: .imageGenerate))
        let queued = store.submit(try makeRequest(templateID: .imageGenerate))
        let alsoQueued = store.submit(try makeRequest(templateID: .imageGenerate))
        let utility = store.submit(try makeRequest(lane: .utility, templateID: .modelList))
        _ = store.submit(try makeRequest(lane: .probe, templateID: .modelCapabilities, dedupeKey: "readiness"))

        let rows = StudioActivity.rows(in: store)

        XCTAssertEqual(rows.map(\.id), [first, second, utility, queued, alsoQueued])
        XCTAssertEqual(rows.map(\.isRunning), [true, true, true, false, false])
        XCTAssertEqual(rows.filter(\.isNextInQueue).map(\.id), [queued], "only the head of the queue is next")
        XCTAssertEqual(StudioActivity.summary(rows), "5 jobs · 2 queued")
    }

    func testRowTitlesNameTheDomainAndTheTask() throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)

        let generate = store.submit(try makeRequest(templateID: .imageGenerate))
        let transcribe = store.submit(try makeRequest(lane: .utility, templateID: .speechTranscribe))
        let pull = store.submit(try makeRequest(lane: .utility, templateID: .modelPull) { draft in
            draft.model = "vision-chat-qwen3.6-vl-4b"
        })
        let gate = store.submit(try makeRequest(lane: .utility, templateID: .qualityGate))

        XCTAssertEqual(StudioActivity.title(for: try XCTUnwrap(store.job(generate))), "Image · Generate")
        XCTAssertEqual(StudioActivity.title(for: try XCTUnwrap(store.job(transcribe))), "Audio · Transcribe")
        XCTAssertEqual(
            StudioActivity.title(for: try XCTUnwrap(store.job(pull))),
            "Models · Pull Qwen3.6-VL 4B",
            "a pull names its model the way the composer's model chip does"
        )
        // A job with no task of its own falls back to the template's title.
        XCTAssertEqual(StudioActivity.title(for: try XCTUnwrap(store.job(gate))), "Models · Quality gate")
    }

    func testQueuedRowsSayWhereTheyAreInTheQueue() throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        _ = store.submit(try makeRequest(templateID: .imageGenerate))
        _ = store.submit(try makeRequest(templateID: .imageGenerate))
        let next = store.submit(try makeRequest(templateID: .imageGenerate))
        let behind = store.submit(try makeRequest(templateID: .imageGenerate))

        XCTAssertEqual(
            StudioActivity.detail(for: try XCTUnwrap(store.job(next)), elapsed: nil, isNextInQueue: true),
            "Queued · next"
        )
        XCTAssertEqual(
            StudioActivity.detail(for: try XCTUnwrap(store.job(behind)), elapsed: nil, isNextInQueue: false),
            "Queued"
        )
    }

    func testRunningRowReportsStepProgressAndElapsedTime() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let id = store.submit(try makeRequest(templateID: .imageGenerate))

        runner.starts[0].stderr("{\"event\":\"progress\",\"stage\":\"denoising\",\"step\":14,\"total_steps\":24}\n")
        await settle()

        let job = try XCTUnwrap(store.job(id))
        XCTAssertEqual(job.progress?.fractionCompleted, 15.0 / 24.0)
        XCTAssertEqual(
            StudioActivity.detail(for: job, elapsed: 41, isNextInQueue: false),
            "Denoising 15/24 · 0:41"
        )
        // Before the first progress line there is still the job's own status.
        let fresh = try XCTUnwrap(store.job(store.submit(try makeRequest(templateID: .imageGenerate))))
        XCTAssertEqual(StudioActivity.detail(for: fresh, elapsed: 3, isNextInQueue: false), "Running · 0:03")
    }

    func testRunningPullReportsTransferredBytesAndTimeLeft() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let id = store.submit(try makeRequest(lane: .utility, templateID: .modelPull) { draft in
            draft.model = "vision-chat-qwen3.6-vl-4b"
        })

        runner.starts[0].stderr("[vision-chat-qwen3.6-vl-4b] 25%  1.2 GB / 4.8 GB  9.7 MB/s  ETA 3m 20s\n")
        await settle()

        let job = try XCTUnwrap(store.job(id))
        XCTAssertEqual(job.progress?.fractionCompleted, 0.25)
        XCTAssertEqual(
            StudioActivity.detail(for: job, elapsed: 96, isNextInQueue: false),
            "1.2 of 4.8 GB · 3 min left"
        )
    }

    func testDownloadDetailShortensTheCLIProgressLine() {
        XCTAssertEqual(
            StudioActivity.downloadDetail(progress(detail: "900 MB / 4.8 GB 9.7 MB/s ETA 45s")),
            "900 MB of 4.8 GB · 45 sec left"
        )
        XCTAssertEqual(
            StudioActivity.downloadDetail(progress(detail: "2.1 GB / 12.4 GB 4 MB/s ETA 1h 12m")),
            "2.1 of 12.4 GB · 1 hr left"
        )
        XCTAssertEqual(StudioActivity.downloadDetail(progress(detail: "1.2 GB / 4.8 GB")), "1.2 of 4.8 GB")
        XCTAssertEqual(StudioActivity.downloadDetail(progress(detail: "extracting…")), "Extracting…")
        // Nothing recognisable leaves the caller to fall back to the job's status line.
        XCTAssertNil(StudioActivity.downloadDetail(progress(detail: "resolving manifest")))
        XCTAssertNil(StudioActivity.downloadDetail(nil))
    }

    func testHeaderSummaryCountsJobsAndTheQueue() {
        XCTAssertEqual(StudioActivity.summary([]), "0 jobs")
        XCTAssertEqual(StudioActivity.summary([row(running: true)]), "1 job")
        XCTAssertEqual(StudioActivity.summary([row(running: true), row(running: true)]), "2 jobs")
        XCTAssertEqual(
            StudioActivity.summary([row(running: true), row(running: true), row(running: false)]),
            "3 jobs · 1 queued"
        )
    }

    func testCancellingARowStopsThatJobAndDropsItFromTheList() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        _ = store.submit(try makeRequest(templateID: .imageGenerate))
        let running = store.submit(try makeRequest(templateID: .imageGenerate))
        let queued = store.submit(try makeRequest(templateID: .imageGenerate))

        // The row id is the job id, so the popover's stop button cancels exactly its own job.
        let queuedRow = try XCTUnwrap(StudioActivity.rows(in: store).first { !$0.isRunning })
        XCTAssertEqual(queuedRow.id, queued)
        XCTAssertTrue(store.cancel(queuedRow.id))
        XCTAssertFalse(StudioActivity.rows(in: store).contains { $0.id == queued })

        let runningRow = try XCTUnwrap(StudioActivity.rows(in: store).first { $0.id == running })
        XCTAssertTrue(runningRow.isRunning)
        XCTAssertTrue(store.cancel(runningRow.id))
        XCTAssertEqual(runner.processes[1].terminateCallCount, 1)

        runner.starts[1].termination(15)
        await settle()
        XCTAssertFalse(StudioActivity.rows(in: store).contains { $0.id == running })
    }

    func testFooterReportsTheVersionHandshake() {
        XCTAssertEqual(
            StudioActivity.versionLine(appVersion: "0.50.0", cliVersion: "0.50.0"),
            "mere.run 0.50.0 · CLI matched"
        )
        XCTAssertEqual(
            StudioActivity.versionLine(appVersion: "0.50.0", cliVersion: "0.49.1"),
            "mere.run 0.50.0 · CLI 0.49.1"
        )
        XCTAssertEqual(
            StudioActivity.versionLine(appVersion: "0.50.0", cliVersion: nil),
            "mere.run 0.50.0 · CLI not resolved"
        )
        XCTAssertEqual(
            StudioActivity.versionLine(appVersion: "0.50.0", cliVersion: "  "),
            "mere.run 0.50.0 · CLI not resolved"
        )
    }

    // MARK: - Footer pill

    func testFooterPillCountsRunningJobs() {
        let ready = StudioMachineStatus.ready(installedModels: 92)
        XCTAssertEqual(ready.summary(runningJobs: 0), "Ready · 92 models")
        XCTAssertEqual(ready.summary(runningJobs: 2), "Ready · 92 models · 2 running")
        XCTAssertEqual(ready.summary(runningJobs: 1), "Ready · 92 models · 1 running")
        XCTAssertEqual(
            StudioMachineStatus.serving(installedModels: 3, loadedModel: "gemma4-e4b").summary(runningJobs: 1),
            "Serving · 3 models · 1 running"
        )
        XCTAssertEqual(StudioMachineStatus.unreachable.summary(runningJobs: 1), "Server unreachable · 1 running")
    }

    func testTheWorkCountIsItsOwnLineOnlyWhileJobsRun() {
        XCTAssertNil(StudioMachineStatus.workCount(0), "an idle machine keeps the one-line pill")
        XCTAssertEqual(StudioMachineStatus.workCount(1), "1 running")
        XCTAssertEqual(StudioMachineStatus.workCount(2), "2 running")
    }

    func testOnlyAnUnreachableServerColoursTheFooterPill() {
        XCTAssertEqual(StudioMachineStatus.unreachable.summaryColor, MereRunTheme.red)
        XCTAssertEqual(StudioMachineStatus.checking.summaryColor, MereRunTheme.textSecondary)
        XCTAssertEqual(StudioMachineStatus.ready(installedModels: 1).summaryColor, MereRunTheme.textSecondary)
    }

    // MARK: - Helpers

    private func row(running: Bool) -> StudioActivityRow {
        StudioActivityRow(id: JobID(), title: "Image · Generate", isRunning: running, isNextInQueue: false)
    }

    private func progress(detail: String?) -> StudioRunProgress? {
        detail.map { StudioRunProgress(label: "model", fractionCompleted: 0.25, detail: $0) }
    }

    private func makeRequest(
        lane: JobLane = .inference,
        templateID: CommandTemplateID,
        dedupeKey: String? = nil,
        configure: (inout CommandDraft) -> Void = { _ in }
    ) throws -> JobRequest {
        let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
        var draft = template.defaultDraft()
        draft.prompt = draft.prompt.isEmpty ? "a ceramic coffee mug in soft morning light" : draft.prompt
        draft.inputPath = "/tmp/input.wav"
        configure(&draft)
        let args = template.arguments(from: draft)
        return JobRequest(
            lane: lane,
            template: template,
            draft: draft,
            requestID: UUID(),
            configuration: MereRunProcessConfiguration(
                executableURL: URL(fileURLWithPath: "/usr/bin/true"),
                arguments: args,
                currentDirectoryURL: FileManager.default.temporaryDirectory,
                environment: [:],
                keepsStandardInputOpen: false
            ),
            displayCommand: (["mere.run"] + args).shellQuoted(),
            dedupeKey: dedupeKey
        )
    }

    private func settle() async {
        for _ in 0..<6 { await Task.yield() }
    }
}
