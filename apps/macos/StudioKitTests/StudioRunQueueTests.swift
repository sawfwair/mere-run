@testable import StudioKit
import Combine
import Foundation
import StudioTestSupport
import XCTest

/// The run queue's model: how it groups a `JobStore`'s work, where each job stands, how a queued
/// job is moved and whether admission then follows the new order, how Stop reaches running and
/// queued jobs, and when — and only when — it can say how long a run has left.
@MainActor
final class StudioRunQueueTests: XCTestCase {
    // MARK: Grouping and positions

    func testSectionsGroupByLaneWithRunningFirstThenQueuePositionsAndNeverProbes() throws {
        let store = JobStore(processRunner: RecordingProcessRunner())
        let first = store.submit(try makeRequest(extra: "first"))
        let second = store.submit(try makeRequest(extra: "second"))
        let next = store.submit(try makeRequest(extra: "next"))
        let behind = store.submit(try makeRequest(extra: "behind"))
        let read = store.submit(try makeRequest(lane: .utility))
        _ = store.submit(try makeRequest(lane: .probe, templateID: .modelCapabilities, dedupeKey: "readiness"))
        let server = store.submit(try makeRequest(lane: .service))

        let sections = StudioRunQueue.sections(in: store)

        XCTAssertEqual(sections.map(\.lane), [.inference, .utility, .service])
        XCTAssertEqual(sections[0].entries.map(\.id), [first, second, next, behind])
        XCTAssertEqual(sections[0].entries.map(\.status), [.running, .running, .queued(position: 0), .queued(position: 1)])
        XCTAssertEqual(sections[0].queuedCount, 2)
        XCTAssertEqual(sections[1].entries.map(\.id), [read])
        XCTAssertEqual(sections[2].entries.map(\.id), [server])
        XCTAssertEqual(sections[0].entries.map(\.canMoveUp), [false, false, false, true])
        XCTAssertEqual(sections[0].entries.map(\.canMoveDown), [false, false, true, false])
        XCTAssertEqual(store.job(next)?.queuePosition, 0)
        XCTAssertEqual(store.job(behind)?.queuePosition, 1)
        XCTAssertNil(store.job(first)?.queuePosition, "a running job has left the queue")
        XCTAssertEqual(StudioRunQueue.activeRunCount(in: store), 4, "the count is the inference lane's runs only")
    }

    func testAnEmptyStoreHasNoSections() {
        XCTAssertTrue(StudioRunQueue.sections(in: JobStore(processRunner: RecordingProcessRunner())).isEmpty)
    }

    func testARunWaitingOnMachineAdmissionReadsAsWaitingForMemoryUntilGranted() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let id = store.submit(try makeRequest(templateID: .imageGenerate))

        runner.starts[0].stderr("Queued by machine admission: image generate (2/2 permits active, 1 queued).\n")
        await settle()
        XCTAssertEqual(StudioRunQueue.sections(in: store).first?.entries.first?.status, .waitingForMemory)

        runner.starts[0].stderr("Machine admission granted: image generate.\n")
        await settle()
        XCTAssertEqual(StudioRunQueue.sections(in: store).first?.entries.first?.status, .running)
        XCTAssertEqual(store.job(id)?.isAwaitingMachineAdmission, false)
    }

    // MARK: Reordering

    func testMovingAQueuedJobChangesTheOrderAdmissionStartsThemIn() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        var reordered: [JobLane] = []
        let subscription = store.events.sink { event in
            if case .reordered(let lane) = event { reordered.append(lane) }
        }
        defer { subscription.cancel() }
        _ = store.submit(try makeRequest(extra: "a"))
        _ = store.submit(try makeRequest(extra: "b"))
        let c = store.submit(try makeRequest(extra: "c"))
        let d = store.submit(try makeRequest(extra: "d"))
        let e = store.submit(try makeRequest(extra: "e"))

        XCTAssertTrue(store.moveQueued(e, by: -2))
        XCTAssertEqual(store.queued(in: .inference).map(\.id), [e, c, d])
        XCTAssertEqual([e, c, d].map { store.job($0)?.queuePosition }, [0, 1, 2])
        XCTAssertEqual(reordered, [.inference])

        runner.starts[0].termination(0)
        await settle()
        XCTAssertEqual(runner.starts[2].configuration.arguments.last, "e", "the moved job starts first")

        runner.starts[1].termination(0)
        await settle()
        XCTAssertEqual(runner.starts[3].configuration.arguments.last, "c")
        XCTAssertEqual(store.queued(in: .inference).map(\.id), [d])
        XCTAssertEqual(store.job(d)?.queuePosition, 0)
    }

    func testOnlyAQueuedJobMovesAndNeverPastEitherEndOfItsQueue() throws {
        let store = JobStore(processRunner: RecordingProcessRunner())
        let running = store.submit(try makeRequest(extra: "a"))
        _ = store.submit(try makeRequest(extra: "b"))
        let head = store.submit(try makeRequest(extra: "c"))
        let tail = store.submit(try makeRequest(extra: "d"))

        XCTAssertFalse(store.moveQueued(running, by: 1), "a running job has no place in the queue")
        XCTAssertFalse(store.moveQueued(head, by: -1))
        XCTAssertFalse(store.moveQueued(tail, by: 1))
        XCTAssertFalse(store.moveQueued(head, by: 0))
        XCTAssertFalse(store.moveQueued(JobID(), by: 1))
        XCTAssertEqual(store.queued(in: .inference).map(\.id), [head, tail])

        XCTAssertTrue(store.moveQueued(tail, by: -1))
        XCTAssertEqual(store.queued(in: .inference).map(\.id), [tail, head])
    }

    func testARunNeverMovesAheadOfTheQueuedDownloadOfItsModel() throws {
        let store = JobStore(processRunner: RecordingProcessRunner())
        _ = store.submit(try makeRequest(extra: "a"))
        _ = store.submit(try makeRequest(extra: "b"))
        let pull = store.submit(try makeRequest(templateID: .modelPull) { $0.model = "image-zimage-nano" })
        let other = store.submit(try makeRequest(templateID: .imageGenerate) { $0.model = "image-flux2-klein" })
        let needsPull = store.submit(try makeRequest(templateID: .imageGenerate) { $0.model = "image-zimage-nano" })

        // It may pass a run of another model, but not the pull of its own, however far it jumps.
        XCTAssertFalse(store.moveQueued(needsPull, by: -2))
        XCTAssertTrue(store.moveQueued(needsPull, by: -1))
        XCTAssertFalse(store.canMoveQueued(needsPull, by: -1))
        XCTAssertFalse(store.moveQueued(needsPull, by: -1))
        XCTAssertFalse(store.moveQueued(pull, by: 1), "nor may the pull drop behind the run that needs it")
        XCTAssertEqual(store.queued(in: .inference).map(\.id), [pull, needsPull, other])

        let entry = try XCTUnwrap(StudioRunQueue.sections(in: store).first?.entries.first { $0.id == needsPull })
        XCTAssertFalse(entry.canMoveUp)
        XCTAssertTrue(entry.waitsForModelDownload)
        let otherEntry = try XCTUnwrap(StudioRunQueue.sections(in: store).first?.entries.first { $0.id == other })
        XCTAssertTrue(otherEntry.canMoveUp)
        XCTAssertFalse(otherEntry.waitsForModelDownload)
    }

    func testTheFeedNumbersQueuedCardsInTheQueuesOwnOrder() throws {
        let store = JobStore(processRunner: RecordingProcessRunner())
        _ = store.submit(try makeRequest(extra: "a"))
        _ = store.submit(try makeRequest(extra: "b"))
        let earlier = try makeRequest(extra: "c")
        let later = try makeRequest(extra: "d")
        _ = store.submit(earlier)
        let laterID = store.submit(later)
        store.moveQueued(laterID, by: -1)

        let now = Date()
        let cards = [earlier, later].enumerated().map { index, request in
            StudioFeedCard(
                item: StudioLibraryItem(
                    id: request.requestID!, mode: .createImage, prompt: "\(index)", inputURL: nil, outputURL: nil,
                    createdAt: now.addingTimeInterval(Double(index)), updatedAt: now, status: .queued, exitCode: nil,
                    commandPreview: ""
                ),
                kind: .queued,
                job: store.job(requestID: request.requestID!)
            )
        }

        XCTAssertEqual(StudioFeedCardBuilder.queuePosition(of: cards[1], in: cards), 0, "moved to the front")
        XCTAssertEqual(StudioFeedCardBuilder.queuePosition(of: cards[0], in: cards), 1)
    }

    // MARK: Cancelling

    func testStopRemovesAQueuedJobAndTerminatesARunningOne() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let running = store.submit(try makeRequest(extra: "a"))
        _ = store.submit(try makeRequest(extra: "b"))
        let queued = store.submit(try makeRequest(extra: "c"))

        StudioRunQueue.stop(try XCTUnwrap(store.job(queued)), in: store)
        XCTAssertEqual(store.job(queued)?.state.exitCode, JobResult.cancelledBeforeStartExitCode)
        XCTAssertEqual(runner.starts.count, 2, "removing a queued job launches nothing")

        StudioRunQueue.stop(try XCTUnwrap(store.job(running)), in: store)
        XCTAssertEqual(runner.processes[0].terminateCallCount, 1)
        XCTAssertEqual(runner.processes[0].interruptCallCount, 0)
    }

    func testStopInterruptsASessionFirstTheWayItsPageDoes() throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let session = store.submit(try makeRequest(templateID: .speechListen))

        StudioRunQueue.stop(try XCTUnwrap(store.job(session)), in: store)

        XCTAssertEqual(runner.processes[0].interruptCallCount, 1, "SIGINT so the CLI flushes its last output")
        XCTAssertEqual(runner.processes[0].terminateCallCount, 0, "SIGTERM only after the grace period")
    }

    func testCancelAllQueuedLeavesRunningJobsAlone() throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let first = store.submit(try makeRequest(extra: "a"))
        let second = store.submit(try makeRequest(extra: "b"))
        _ = store.submit(try makeRequest(extra: "c"))
        _ = store.submit(try makeRequest(extra: "d"))

        XCTAssertEqual(StudioRunQueue.cancelAllQueued(in: store), 2)
        XCTAssertTrue(store.queued(in: .inference).isEmpty)
        XCTAssertEqual(store.running(in: .inference).map(\.id), [first, second])
        XCTAssertEqual(runner.processes.map(\.terminateCallCount), [0, 0])
        XCTAssertEqual(StudioRunQueue.activeRunCount(in: store), 2)
    }

    func testRecentlyFinishedListsTheNewestInferenceRunsOnly() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let older = store.submit(try makeRequest(extra: "older"))
        runner.starts[0].termination(0)
        await settle()
        let newer = store.submit(try makeRequest(extra: "newer"))
        runner.starts[1].termination(1)
        await settle()
        _ = store.submit(try makeRequest(lane: .utility))
        runner.starts[2].termination(0)
        await settle()

        XCTAssertEqual(StudioRunQueue.recentlyFinished(in: store).map(\.id), [newer, older])
        XCTAssertEqual(StudioRunQueue.recentlyFinished(in: store, limit: 1).map(\.id), [newer])
    }

    func testTheCounterPublishesOnlyWhenTheCountChanges() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let counter = StudioRunQueueCounter()
        counter.attach(store)
        var published: [Int] = []
        let subscription = counter.$activeRunCount.dropFirst().sink { published.append($0) }
        defer { subscription.cancel() }

        _ = store.submit(try makeRequest(extra: "a"))
        _ = store.submit(try makeRequest(extra: "b"))
        _ = store.submit(try makeRequest(extra: "c"))
        runner.starts[0].stderr("{\"event\":\"progress\",\"stage\":\"denoising\",\"step\":1,\"total_steps\":4}\n")
        await settle()
        runner.starts[0].termination(0)
        await settle()

        XCTAssertEqual(published, [1, 2, 3, 2], "progress chatter publishes nothing")
        XCTAssertEqual(StudioRunQueue.badgeLabel(activeRuns: 2), "2")
        XCTAssertNil(StudioRunQueue.badgeLabel(activeRuns: 0))
    }

    // MARK: Time left

    func testTheCLIDownloadEstimateIsUsedAsIs() {
        let progress = StudioRunProgress(label: "model", fractionCompleted: 0.25, detail: "1.2 GB / 4.8 GB 9.7 MB/s ETA 3m 20s")
        XCTAssertEqual(
            StudioRunETA.estimate(progress: progress, stage: nil, elapsed: 30, typicalDuration: nil, now: Date()),
            StudioRunETA(remaining: 200, source: .download)
        )
        XCTAssertEqual(StudioRunETA.downloadSecondsLeft("ETA 1h 12m"), 4_320)
        XCTAssertEqual(StudioRunETA.downloadSecondsLeft("ETA 45s"), 45)
        XCTAssertNil(StudioRunETA.downloadSecondsLeft("1.2 GB / 4.8 GB"))
    }

    func testAStageEstimateComesFromThisRunsStepRateInThatStage() {
        let start = Date(timeIntervalSince1970: 1_000)
        let stage = StudioProgressStage(label: "Denoising", startedAt: start, startFraction: 0.25)
        let progress = StudioRunProgress(label: "Denoising", fractionCompleted: 0.5, detail: "Step 12 of 24")

        // A quarter of the stage took 10 s, so the remaining half takes 20 s.
        XCTAssertEqual(
            StudioRunETA.estimate(progress: progress, stage: stage, elapsed: 40, typicalDuration: nil, now: start.addingTimeInterval(10)),
            StudioRunETA(remaining: 20, source: .stage("Denoising"))
        )
    }

    func testNoStepRateMeansNoStageEstimate() {
        let start = Date(timeIntervalSince1970: 1_000)
        let now = start.addingTimeInterval(10)
        let first = StudioRunProgress(label: "Denoising", fractionCompleted: 0.25, detail: "Step 6 of 24")
        // Only one update so far: there is no rate yet.
        XCTAssertNil(StudioRunETA.estimate(
            progress: first, stage: StudioProgressStage(label: "Denoising", startedAt: start, startFraction: 0.25),
            elapsed: 10, typicalDuration: nil, now: now
        ))
        // An indeterminate stage (token streaming) has nothing to extrapolate.
        XCTAssertNil(StudioRunETA.estimate(
            progress: StudioRunProgress(label: "Generating", fractionCompleted: nil, detail: "Step 40"),
            stage: nil, elapsed: 10, typicalDuration: nil, now: now
        ))
        // A stage measured under another label says nothing about this one.
        XCTAssertNil(StudioRunETA.estimate(
            progress: StudioRunProgress(label: "Decoding", fractionCompleted: 0.5, detail: nil),
            stage: StudioProgressStage(label: "Denoising", startedAt: start, startFraction: 0.25),
            elapsed: 10, typicalDuration: nil, now: now
        ))
        XCTAssertNil(StudioRunETA.estimate(progress: nil, stage: nil, elapsed: 10, typicalDuration: nil, now: now))
    }

    func testHistoryGivesTimeLeftOnlyWhileTheRunIsInsideItsTypicalDuration() {
        XCTAssertEqual(
            StudioRunETA.estimate(progress: nil, stage: nil, elapsed: 30, typicalDuration: 90, now: Date()),
            StudioRunETA(remaining: 60, source: .history)
        )
        XCTAssertNil(
            StudioRunETA.estimate(progress: nil, stage: nil, elapsed: 120, typicalDuration: 90, now: Date()),
            "a run that outlasted its history gets no invented estimate"
        )
        XCTAssertEqual(StudioRunETA.typicalDuration(of: [80, 100, 90]), 90)
        XCTAssertEqual(StudioRunETA.typicalDuration(of: [80, 100]), 90)
        XCTAssertEqual(StudioRunETA.typicalDuration(of: [10, 20, 30, 40, 50, 1_000]), 30, "only the five newest count")
        XCTAssertNil(StudioRunETA.typicalDuration(of: []))
    }

    func testHistoryIsTheSameTemplateAndModelsSuccessfulRunsOnly() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        func run(_ template: CommandTemplateID, model: String, exit: Int32) async throws {
            _ = store.submit(try makeRequest(templateID: template) { $0.model = model })
            runner.starts.last?.termination(exit)
            await settle()
        }
        try await run(.imageGenerate, model: "image-zimage-nano", exit: 0)
        try await run(.imageGenerate, model: "image-zimage-nano", exit: 1)
        try await run(.imageGenerate, model: "image-flux2-klein", exit: 0)
        try await run(.custom, model: "image-zimage-nano", exit: 0)

        let current = try XCTUnwrap(store.job(store.submit(try makeRequest(templateID: .imageGenerate) {
            $0.model = "image-zimage-nano"
        })))
        XCTAssertEqual(StudioRunETA.recentDurations(like: current, in: store).count, 1)

        let turn = try XCTUnwrap(store.job(store.submit(try makeRequest(templateID: .imageGenerate, conversationID: UUID()) {
            $0.model = "image-zimage-nano"
        })))
        XCTAssertTrue(StudioRunETA.recentDurations(like: turn, in: store).isEmpty, "a turn's length is its reply's")
    }

    func testTheJobKeepsItsStageStartWhileTheSameStageMovesForward() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let job = try XCTUnwrap(store.job(store.submit(try makeRequest(templateID: .imageGenerate))))

        runner.starts[0].stderr("{\"event\":\"progress\",\"stage\":\"denoising\",\"step\":2,\"total_steps\":24}\n")
        await settle()
        let started = try XCTUnwrap(job.progressStage)
        XCTAssertEqual(started.label, "Denoising")
        XCTAssertEqual(started.startFraction, 3.0 / 24.0)

        runner.starts[0].stderr("{\"event\":\"progress\",\"stage\":\"denoising\",\"step\":9,\"total_steps\":24}\n")
        await settle()
        XCTAssertEqual(job.progressStage, started, "the same stage keeps its start")

        runner.starts[0].stderr("{\"event\":\"progress\",\"stage\":\"decoding\",\"step\":0,\"total_steps\":1}\n")
        await settle()
        XCTAssertEqual(job.progressStage?.label, "Decoding", "a new stage starts over")

        runner.starts[0].stderr("{\"event\":\"progress\",\"stage\":\"generating\",\"step\":3,\"total_steps\":0}\n")
        await settle()
        XCTAssertNil(job.progressStage, "an indeterminate stage has no rate")
    }
}

private extension StudioRunQueueTests {
    func makeRequest(
        lane: JobLane = .inference,
        templateID: CommandTemplateID = .custom,
        extra: String = "x",
        conversationID: UUID? = nil,
        dedupeKey: String? = nil,
        configure: (inout CommandDraft) -> Void = { _ in }
    ) throws -> JobRequest {
        let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
        var draft = template.defaultDraft()
        draft.prompt = draft.prompt.isEmpty ? "a ceramic coffee mug in soft morning light" : draft.prompt
        draft.extraArguments = extra
        configure(&draft)
        let args = template.arguments(from: draft, source: .contract)
        return JobRequest(
            lane: lane,
            template: template,
            draft: draft,
            requestID: UUID(),
            conversationID: conversationID,
            configuration: MereRunProcessConfiguration(
                executableURL: URL(fileURLWithPath: "/usr/bin/true"),
                arguments: args,
                currentDirectoryURL: FileManager.default.temporaryDirectory,
                environment: [:],
                keepsStandardInputOpen: false
            ),
            displayCommand: (["mere.run"] + args).shellQuoted(),
            dedupeKey: dedupeKey,
            scopeSource: .contract
        )
    }

    func settle() async {
        for _ in 0..<6 { await Task.yield() }
    }
}
