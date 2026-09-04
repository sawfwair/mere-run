@testable import MereRunApp
import Combine
import Foundation
import XCTest

@MainActor
final class JobStoreTests: XCTestCase {
    func testInferenceLaneRunsUpToCapacityThenQueuesInSubmissionOrder() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)

        let first = store.submit(try makeRequest(extra: "first"))
        let second = store.submit(try makeRequest(extra: "second"))
        let third = store.submit(try makeRequest(extra: "third"))
        let fourth = store.submit(try makeRequest(extra: "fourth"))

        XCTAssertEqual(JobLane.inference.capacity, 2)
        XCTAssertEqual(runner.starts.count, 2)
        XCTAssertEqual(store.running(in: .inference).map(\.id), [first, second])
        XCTAssertEqual(store.queued(in: .inference).map(\.id), [third, fourth])
        XCTAssertEqual(store.job(third)?.state, .queued)
        XCTAssertFalse(store.hasCapacity(in: .inference))

        runner.starts[0].termination(0)
        await settle()

        XCTAssertEqual(runner.starts.count, 3)
        XCTAssertEqual(runner.starts[2].configuration.arguments, ["third"])
        XCTAssertEqual(store.running(in: .inference).map(\.id), [second, third])
        XCTAssertEqual(store.queued(in: .inference).map(\.id), [fourth])
        XCTAssertTrue(store.job(first)?.state.isTerminal == true)
    }

    func testCancellingQueuedJobDeliversResultAndLeavesTheQueueOrderIntact() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        var results: [JobResult] = []
        let subscription = store.completions.sink { results.append($0) }
        defer { subscription.cancel() }

        _ = store.submit(try makeRequest(extra: "a"))
        _ = store.submit(try makeRequest(extra: "b"))
        let cancelledRequestID = UUID()
        let queued = store.submit(try makeRequest(extra: "c", requestID: cancelledRequestID))
        let survivor = store.submit(try makeRequest(extra: "d"))

        XCTAssertTrue(store.cancel(queued))
        let cancelled = try XCTUnwrap(results.first)
        XCTAssertEqual(store.job(queued)?.state, .cancelled(exit: JobResult.cancelledBeforeStartExitCode, at: cancelled.completedAt))
        XCTAssertEqual(store.job(queued)?.status, "Cancelled")
        XCTAssertEqual(results.map(\.requestID), [cancelledRequestID])
        XCTAssertEqual(cancelled.exitCode, JobResult.cancelledBeforeStartExitCode)
        XCTAssertEqual(store.queued(in: .inference).map(\.id), [survivor])
        XCTAssertFalse(store.cancel(queued), "a settled job cannot be cancelled twice")

        runner.starts[0].termination(0)
        await settle()
        XCTAssertEqual(runner.starts.count, 3)
        XCTAssertEqual(runner.starts[2].configuration.arguments, ["d"])
    }

    func testCancellingRunningJobTerminatesTheProcessAndSettlesAsCancelled() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let id = store.submit(try makeRequest())

        XCTAssertTrue(store.cancel(id))
        XCTAssertEqual(runner.processes[0].terminateCallCount, 1)
        XCTAssertTrue(store.job(id)?.log.lines.contains { $0.text == "Termination requested." } == true)
        XCTAssertTrue(store.job(id)?.state.isRunning == true, "the job stays running until the process exits")

        runner.starts[0].termination(15)
        await settle()

        let job = try XCTUnwrap(store.job(id))
        guard case .cancelled(let exit, _) = job.state else {
            return XCTFail("expected cancelled, got \(job.state)")
        }
        XCTAssertEqual(exit, 15)
        XCTAssertEqual(job.status, "Exited 15")
        XCTAssertEqual(job.result?.exitCode, 15)
    }

    func testCompletionsDeliverEveryJobInTerminationOrder() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        var completed: [UUID?] = []
        let subscription = store.completions.sink { completed.append($0.requestID) }
        defer { subscription.cancel() }

        let first = UUID()
        let second = UUID()
        _ = store.submit(try makeRequest(extra: "first", requestID: first))
        _ = store.submit(try makeRequest(extra: "second", requestID: second))

        // Back-to-back exits in reverse order must both arrive, in the order they finished.
        runner.starts[1].termination(0)
        runner.starts[0].termination(3)
        await settle()

        XCTAssertEqual(completed, [second, first])
    }

    func testUtilityLaneHasItsOwnCapacityIndependentOfInference() throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)

        _ = store.submit(try makeRequest(extra: "infer-1"))
        _ = store.submit(try makeRequest(extra: "infer-2"))
        _ = store.submit(try makeRequest(extra: "infer-3"))
        for index in 0..<5 {
            _ = store.submit(try makeRequest(lane: .utility, extra: "util-\(index)"))
        }

        XCTAssertEqual(JobLane.utility.capacity, 4)
        XCTAssertEqual(store.running(in: .inference).count, 2)
        XCTAssertEqual(store.queued(in: .inference).count, 1)
        XCTAssertEqual(store.running(in: .utility).count, 4)
        XCTAssertEqual(store.queued(in: .utility).count, 1)
        XCTAssertEqual(runner.starts.count, 6)
        XCTAssertEqual(store.running.count, 6)
    }

    func testProbeSubmissionsDedupeByKeyAndSupersedeWhenTheConfigurationChanges() throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)

        let first = store.submit(try makeRequest(lane: .probe, extra: "capabilities", dedupeKey: "createImage"))
        let duplicate = store.submit(try makeRequest(lane: .probe, extra: "capabilities", dedupeKey: "createImage"))
        XCTAssertEqual(duplicate, first)
        XCTAssertEqual(runner.starts.count, 1)

        let superseding = store.submit(try makeRequest(lane: .probe, extra: "list", dedupeKey: "createImage"))
        XCTAssertNotEqual(superseding, first)
        XCTAssertEqual(runner.processes[0].terminateCallCount, 1)
        XCTAssertEqual(runner.starts.count, 2)

        let otherKey = store.submit(try makeRequest(lane: .probe, extra: "capabilities", dedupeKey: "video"))
        XCTAssertNotEqual(otherKey, superseding)
        XCTAssertEqual(runner.starts.count, 3)
        XCTAssertEqual(store.running(in: .probe).count, 3, "probes are never queued")
    }

    func testProbeDedupeNeverReturnsASupersededProbeThatIsStillDying() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)

        let first = store.submit(try makeRequest(lane: .probe, extra: "capabilities", dedupeKey: "createImage"))
        let second = store.submit(try makeRequest(lane: .probe, extra: "list", dedupeKey: "createImage"))
        XCTAssertEqual(runner.processes[0].terminateCallCount, 1)
        XCTAssertTrue(store.job(first)?.state.isRunning == true, "the superseded probe has not exited yet")

        runner.starts[1].termination(0)
        await settle()

        // Same configuration as the dying probe: a fresh launch, not the SIGTERM'd job.
        let third = store.submit(try makeRequest(lane: .probe, extra: "capabilities", dedupeKey: "createImage"))
        XCTAssertNotEqual(third, first)
        XCTAssertNotEqual(third, second)
        XCTAssertEqual(runner.starts.count, 3)

        runner.starts[0].termination(15)
        runner.starts[2].termination(0)
        await settle()
        XCTAssertEqual(store.job(first)?.state.exitCode, 15)
        XCTAssertEqual(store.job(third)?.result?.exitCode, 0)
    }

    func testSendWritesToStandardInputOnlyWhileRunning() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let steerable = store.submit(try makeRequest(templateID: .musicRealtime, extra: "") { draft in
            draft.prompt = "ambient glass percussion"
            draft.musicInteractive = true
        })
        let plain = store.submit(try makeRequest(extra: "plain"))

        try store.send("temp 0.8\n", to: steerable)
        XCTAssertEqual(runner.processes[0].standardInputs, ["temp 0.8\n"])
        XCTAssertTrue(store.sendLiveControl("bpm 92", to: steerable))
        XCTAssertEqual(runner.processes[0].standardInputs, ["temp 0.8\n", "bpm 92\n"])
        XCTAssertTrue(store.job(steerable)?.log.lines.contains { $0.text == "Live control → bpm 92" } == true)

        XCTAssertThrowsError(try store.send("x", to: plain)) { error in
            XCTAssertTrue(error is MereRunProcessInputError)
        }
        XCTAssertFalse(store.sendLiveControl("x", to: plain))

        runner.starts[0].termination(0)
        await settle()
        XCTAssertThrowsError(try store.send("late", to: steerable)) { error in
            XCTAssertTrue(error is JobStoreError)
        }
        XCTAssertThrowsError(try store.send("nobody", to: JobID()))
    }

    func testProgressLinesBecomeStructuredProgressAndClearOnFinish() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let id = store.submit(try makeRequest())
        let job = try XCTUnwrap(store.job(id))

        runner.starts[0].stderr("Training (2/4) loss 0.123456\n")
        await settle()

        let progress = try XCTUnwrap(job.progress)
        XCTAssertEqual(progress.label, "Training")
        XCTAssertEqual(progress.fractionCompleted, 0.5)
        XCTAssertFalse(job.log.lines.contains { $0.text.contains("Training (2/4)") }, "progress lines stay out of the log")

        runner.starts[0].termination(0)
        await settle()
        XCTAssertNil(job.progress)
        XCTAssertEqual(job.status, "Completed")
    }

    func testPrimaryArtifactFollowsStdoutAndSidecarsAreCollectedOnFinish() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("JobStoreTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: temp) }
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        let loose = temp.appendingPathComponent("loose.png", isDirectory: false)
        let page = temp.appendingPathComponent("page.txt", isDirectory: false)
        try Data("png".utf8).write(to: loose)
        try Data("text".utf8).write(to: page)
        let id = store.submit(try makeRequest())
        let job = try XCTUnwrap(store.job(id))

        // A bare trailing line that resolves to a file is the last-resort probe.
        runner.starts[0].stdout("\(loose.path)\n")
        await settle()
        XCTAssertEqual(job.primaryArtifactURL?.path, loose.path)
        XCTAssertEqual(job.status, "Generated: loose.png")

        // The stdout contract's `input -> output` pair is newer, so it takes over as primary.
        runner.starts[0].stdout("/in/page.png -> \(page.path)\n")
        await settle()
        XCTAssertEqual(job.primaryArtifactURL?.path, page.path)

        runner.starts[0].termination(0)
        await settle()
        XCTAssertEqual(job.result?.outputURL?.path, page.path)
        XCTAssertEqual(job.result?.artifactURLs.map(\.path), [page.path, loose.path])
        XCTAssertEqual(job.artifacts.map(\.url.path), [page.path, loose.path])
        XCTAssertEqual(job.artifacts.map(\.role), [.primary, .sidecar])
        XCTAssertEqual(job.status, "Completed: page.txt")
    }

    func testPrimaryArtifactPrefersTheExpectedOutputOnceItExists() async throws {
        let probe = StubFileProbe()
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner, fileSystem: probe)
        let id = store.submit(try makeRequest(templateID: .imageGenerate, extra: "") { draft in
            draft.prompt = "a blue plate"
            draft.outputPath = "/tmp/JobStoreTests-expected.png"
        })
        let job = try XCTUnwrap(store.job(id))
        XCTAssertEqual(job.expectedOutput?.path, "/tmp/JobStoreTests-expected.png")

        probe.existingPaths = ["/tmp/other.png"]
        runner.starts[0].stdout("/tmp/other.png\n")
        await settle()
        XCTAssertEqual(job.primaryArtifactURL?.path, "/tmp/other.png")

        probe.existingPaths.insert("/tmp/JobStoreTests-expected.png")
        runner.starts[0].stdout("saving\n")
        await settle()
        XCTAssertEqual(job.primaryArtifactURL?.path, "/tmp/JobStoreTests-expected.png")
        XCTAssertEqual(job.status, "Generated: JobStoreTests-expected.png")
    }

    func testOutputWatchDetectsExpectedOutputWhileTheProcessIsSilent() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner, outputWatchInterval: .milliseconds(10))
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("JobStoreTests-\(UUID().uuidString)", isDirectory: true)
        let output = temp.appendingPathComponent("image.png", isDirectory: false)
        defer { try? FileManager.default.removeItem(at: temp) }
        let id = store.submit(try makeRequest(templateID: .imageGenerate, extra: "") { draft in
            draft.prompt = "a blue plate"
            draft.outputPath = output.path
        })
        let job = try XCTUnwrap(store.job(id))
        let detected = expectation(description: "the watch publishes the detected artifact")
        detected.assertForOverFulfill = true
        let subscription = store.events.sink { event in
            if case .changed(let changed) = event, changed.id == id, changed.primaryArtifactURL != nil {
                detected.fulfill()
            }
        }
        defer { subscription.cancel() }

        XCTAssertTrue(FileManager.default.fileExists(atPath: temp.path), "preflight creates the output directory")
        XCTAssertNil(job.primaryArtifactURL)

        try Data("png".utf8).write(to: output)
        await fulfillment(of: [detected], timeout: 2)

        XCTAssertEqual(job.primaryArtifactURL?.path, output.path)
        XCTAssertEqual(job.status, "Generated: image.png")
        XCTAssertTrue(job.state.isRunning)
        // A few more ticks must not republish the same artifact.
        try await Task.sleep(for: .milliseconds(50))
    }

    func testValidationFailureFailsPreflightWithoutLaunchingAndAdvancesTheQueue() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        var results: [JobResult] = []
        let subscription = store.completions.sink { results.append($0) }
        defer { subscription.cancel() }

        _ = store.submit(try makeRequest(extra: "a"))
        _ = store.submit(try makeRequest(extra: "b"))
        let invalid = store.submit(try makeRequest(templateID: .imageGenerate, extra: "") { draft in
            draft.prompt = ""
        })
        let next = store.submit(try makeRequest(extra: "d"))
        XCTAssertEqual(store.job(invalid)?.state, .queued, "validation runs when the job would start")

        runner.starts[0].termination(0)
        await settle()

        XCTAssertEqual(store.job(invalid)?.state, .preflightFailed(.invalidRequest("Prompt is required.")))
        XCTAssertEqual(results.map(\.exitCode), [0, 64])
        XCTAssertEqual(results[1].outputText, "Prompt is required.")
        XCTAssertEqual(runner.starts.count, 3, "the invalid job never launched; the next one took its slot")
        XCTAssertEqual(store.job(next)?.state.isRunning, true)
    }

    func testLaunchFailureFinishesWithExitMinusOneAndRecordsTheError() async throws {
        let runner = RecordingProcessRunner()
        runner.launchError = StubLaunchError()
        let store = JobStore(processRunner: runner)

        let id = store.submit(try makeRequest())
        let job = try XCTUnwrap(store.job(id))

        XCTAssertEqual(job.state.exitCode, -1)
        XCTAssertTrue(job.log.lines.contains { $0.stream == .stderr && $0.text == "mere.run could not be launched." })
        XCTAssertEqual(job.result?.exitCode, -1)
        XCTAssertEqual(job.result?.outputText, "mere.run could not be launched.")
        XCTAssertEqual(job.status, "Exited -1")
        XCTAssertTrue(store.running.isEmpty)
    }

    func testResultForAwaitsCompletionAndIsNilForUnknownJobs() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let requestID = UUID()
        let id = store.submit(try makeRequest(requestID: requestID))

        let pending = Task { await store.result(for: id) }
        await Task.yield()
        runner.starts[0].stdout("done\n")
        runner.starts[0].termination(0)
        let pendingValue = await pending.value
        let awaited = try XCTUnwrap(pendingValue)
        XCTAssertEqual(awaited.requestID, requestID)
        XCTAssertEqual(awaited.outputText, "done")

        let immediate = await store.result(for: id)
        XCTAssertEqual(immediate, awaited)
        let unknown = await store.result(for: JobID())
        XCTAssertNil(unknown)
    }

    func testFinishedJobsAreRetainedPerLaneOldestFirstOut() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        // A finished run the user may still be looking at.
        let run = store.submit(try makeRequest(extra: "finished-run", requestID: UUID()))
        runner.starts[0].termination(0)
        await settle()

        var ids: [JobID] = []
        for index in 0...JobStore.finishedJobRetentionLimit {
            let id = store.submit(try makeRequest(lane: .utility, extra: "\(index)"))
            ids.append(id)
            runner.starts[index + 1].termination(0)
            await settle()
        }
        let survivor = store.submit(try makeRequest(lane: .utility, extra: "running"))

        XCTAssertNil(store.job(ids[0]), "the oldest finished utility job is evicted")
        XCTAssertNotNil(store.job(ids[1]))
        XCTAssertNotNil(store.job(run), "utility churn never evicts a finished run from another lane")
        XCTAssertEqual(store.all.filter { $0.lane == .utility && $0.state.isTerminal }.count, JobStore.finishedJobRetentionLimit)
        XCTAssertNotNil(store.job(survivor))
        XCTAssertEqual(store.order.count, JobStore.finishedJobRetentionLimit + 2)
    }

    func testTerminateAllSettlesRunningJobsAsCancelled() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let id = store.submit(try makeRequest())

        store.terminateAll()
        runner.starts[0].termination(15)
        await settle()

        guard case .cancelled(let exit, _) = try XCTUnwrap(store.job(id)).state else {
            return XCTFail("expected cancelled")
        }
        XCTAssertEqual(exit, 15)
    }

    func testInterruptSendsSIGINTOnlyToRunningJobs() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let id = store.submit(try makeRequest())

        XCTAssertTrue(store.interrupt(id))
        XCTAssertEqual(runner.processes[0].interruptCallCount, 1)
        XCTAssertEqual(runner.processes[0].terminateCallCount, 0)

        runner.starts[0].termination(0)
        await settle()
        XCTAssertFalse(store.interrupt(id))
        XCTAssertFalse(store.interrupt(JobID()))
    }

    func testTerminateAllDropsQueuedJobsAndTerminatesRunningOnes() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        var results: [JobResult] = []
        let subscription = store.completions.sink { results.append($0) }
        defer { subscription.cancel() }
        _ = store.submit(try makeRequest(extra: "a"))
        _ = store.submit(try makeRequest(extra: "b"))
        let queued = store.submit(try makeRequest(extra: "c"))

        store.terminateAll()

        XCTAssertEqual(runner.processes.map(\.terminateCallCount), [1, 1])
        XCTAssertEqual(store.job(queued)?.state.exitCode, JobResult.cancelledBeforeStartExitCode)
        XCTAssertEqual(results.count, 1)

        runner.starts[0].termination(15)
        runner.starts[1].termination(15)
        await settle()
        XCTAssertEqual(runner.starts.count, 2, "nothing launches after terminateAll")
        XCTAssertTrue(store.running.isEmpty)
    }

    func testConversationTurnPublishesLiveThinkStrippedTextAndKeepsTheFullReply() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let conversationID = UUID()
        let id = store.submit(try makeRequest(conversationID: conversationID))
        let job = try XCTUnwrap(store.job(id))

        runner.starts[0].stdout("<think>deliberating</think>Final answer.")
        await settle()
        XCTAssertEqual(job.conversationLiveText, "Final answer.")
        XCTAssertEqual(job.liveText, "<think>deliberating</think>Final answer.")

        let head = String(repeating: "A", count: 40_000)
        runner.starts[0].stdout(head + "/tmp\n")
        await settle()
        runner.starts[0].termination(0)
        await settle()

        let result = try XCTUnwrap(job.result)
        XCTAssertEqual(result.conversationID, conversationID)
        XCTAssertNil(result.outputURL, "a path-like reply never becomes an artifact")
        XCTAssertTrue(job.artifacts.isEmpty)
        XCTAssertTrue(result.outputText?.hasPrefix("Final answer.AAAA") == true)
        XCTAssertGreaterThan(result.outputText?.count ?? 0, 40_000)
        XCTAssertNil(job.conversationLiveText)
    }

    func testJobLookupByRequestIDPrefersTheActiveRunOverAnEarlierOne() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        let requestID = UUID()
        let first = store.submit(try makeRequest(extra: "first", requestID: requestID))
        runner.starts[0].termination(1)
        await settle()
        XCTAssertEqual(store.job(requestID: requestID)?.id, first)

        let rerun = store.submit(try makeRequest(extra: "rerun", requestID: requestID))
        XCTAssertEqual(store.job(requestID: requestID)?.id, rerun)
        XCTAssertNil(store.job(requestID: UUID()))
        XCTAssertTrue(store.isRunning(template: .custom))
        XCTAssertFalse(store.isRunning(template: .imageGenerate))
    }

    func testStartedEventPrecedesTheLaunchedCommandLogLineAndOutputChunksPrecedeChanged() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)
        var trace: [String] = []
        let subscription = store.events.sink { event in
            switch event {
            case .started(let job): trace.append("started:\(job.log.lines.count)")
            case .output(let job, let stream, let text): trace.append("output:\(stream.label):\(text):\(job.log.lines.count)")
            case .changed(let job): trace.append("changed:\(job.log.lines.count)")
            case .finished: trace.append("finished")
            }
        }
        defer { subscription.cancel() }

        let id = store.submit(try makeRequest(extra: "trace"))

        XCTAssertEqual(trace, ["started:0", "changed:1"])
        XCTAssertEqual(store.job(id)?.log.lines.first?.text, "mere.run trace")
        XCTAssertEqual(store.job(id)?.status, "Running")

        // A chunk is delivered verbatim, after the job folded it into its log, before `.changed`.
        runner.starts[0].stdout("hello\n")
        runner.starts[0].stderr("careful\n")
        await settle()
        XCTAssertEqual(
            Array(trace.dropFirst(2)),
            ["output:out:hello\n:2", "changed:2", "output:err:careful\n:3", "changed:3"]
        )
    }

    func testRawRequestsSkipPreflightAndArtifactDetectionAndCaptureCompleteOutput() async throws {
        let runner = RecordingProcessRunner()
        let probe = StubFileProbe()
        probe.existingPaths = ["/tmp/config.json"]
        let store = JobStore(processRunner: runner, fileSystem: probe)

        let id = store.submit(
            .utility(
                arguments: ["config", "path"],
                configuration: makeConfiguration(arguments: ["config", "path"]),
                displayCommand: "mere.run config path"
            )
        )
        let job = try XCTUnwrap(store.job(id))
        XCTAssertEqual(job.lane, .utility)
        XCTAssertTrue(job.state.isRunning, "a raw command has no template to validate against")
        XCTAssertNil(job.request.templateID)
        XCTAssertNil(job.request.template)
        XCTAssertNil(job.request.draft)
        XCTAssertNil(job.expectedOutput)
        XCTAssertFalse(job.detectsArtifacts)
        XCTAssertEqual(job.status, "Running")

        let long = String(repeating: "x", count: 40 * 1024)
        runner.starts[0].stdout("/tmp/config.json\n")
        runner.starts[0].stdout(long)
        runner.starts[0].stderr("warning: stale cache\n")
        await settle()
        XCTAssertTrue(job.artifacts.isEmpty, "an existing path on stdout is data, not an artifact")
        XCTAssertEqual(job.status, "Running")

        runner.starts[0].termination(1)
        await settle()
        let result = try XCTUnwrap(job.result)
        XCTAssertEqual(result.exitCode, 1)
        XCTAssertEqual(result.templateID, .custom)
        XCTAssertNil(result.outputURL)
        XCTAssertTrue(result.artifactURLs.isEmpty)
        XCTAssertEqual(result.standardOutput, "/tmp/config.json\n" + long, "stdout is captured whole, past the console cap")
        XCTAssertEqual(result.standardError, "warning: stale cache\n")
        XCTAssertEqual(result.outputText?.hasSuffix("\n\nSTDERR\nwarning: stale cache"), true)
        XCTAssertEqual(job.status, "Exited 1")

        // Catalog commands keep reporting through `outputText` only.
        let templated = store.submit(try makeRequest(extra: "templated"))
        runner.starts[1].stdout("done\n")
        runner.starts[1].termination(0)
        await settle()
        XCTAssertNil(store.job(templated)?.result?.standardOutput)
        XCTAssertNil(store.job(templated)?.result?.standardError)
        XCTAssertEqual(store.job(templated)?.result?.outputText, "done")
    }

    func testProbeRequestsCarryTheirDedupeKeyAndSubmitHonorsAnExplicitID() throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner)

        let probe = JobRequest.probe(
            arguments: ["status", "--json"],
            configuration: makeConfiguration(arguments: ["status", "--json"]),
            displayCommand: "mere.run status --json",
            dedupeKey: "server-status"
        )
        XCTAssertEqual(probe.lane, .probe)
        XCTAssertEqual(probe.dedupeKey, "server-status")
        XCTAssertNil(probe.requestID)
        XCTAssertNil(probe.conversationID)

        let first = store.submit(probe)
        let duplicate = store.submit(probe, id: JobID())
        XCTAssertEqual(duplicate, first, "a deduplicated probe keeps the in-flight job's id")
        XCTAssertEqual(runner.starts.count, 1)

        let named = JobID()
        XCTAssertEqual(store.submit(try makeRequest(lane: .utility, extra: "named"), id: named), named)
        XCTAssertTrue(store.job(named)?.state.isRunning == true)
        XCTAssertTrue(store.cancel(named))
        XCTAssertEqual(runner.processes[1].terminateCallCount, 1)
    }
}

private extension JobStoreTests {
    /// Builds a request the way the controller does, without a controller: the template's argv,
    /// a fixed executable, and stdin open for the resident templates.
    func makeRequest(
        lane: JobLane = .inference,
        templateID: CommandTemplateID = .custom,
        extra: String = "x",
        requestID: UUID? = UUID(),
        conversationID: UUID? = nil,
        dedupeKey: String? = nil,
        configure: (inout CommandDraft) -> Void = { _ in }
    ) throws -> JobRequest {
        let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
        var draft = template.defaultDraft()
        draft.extraArguments = extra
        configure(&draft)
        let args = template.arguments(from: draft)
        return JobRequest(
            lane: lane,
            template: template,
            draft: draft,
            requestID: requestID,
            conversationID: conversationID,
            configuration: MereRunProcessConfiguration(
                executableURL: URL(fileURLWithPath: "/usr/bin/true"),
                arguments: args,
                currentDirectoryURL: FileManager.default.temporaryDirectory,
                environment: [:],
                keepsStandardInputOpen: templateID == .musicRealtime || templateID == .videoSession
            ),
            displayCommand: (["mere.run"] + args).shellQuoted(),
            dedupeKey: dedupeKey
        )
    }

    func makeConfiguration(arguments: [String]) -> MereRunProcessConfiguration {
        MereRunProcessConfiguration(
            executableURL: URL(fileURLWithPath: "/usr/bin/true"),
            arguments: arguments,
            currentDirectoryURL: FileManager.default.temporaryDirectory,
            environment: [:],
            keepsStandardInputOpen: false
        )
    }

    /// Lets the main-actor hops the process callbacks take run to completion.
    func settle() async {
        for _ in 0..<6 { await Task.yield() }
    }
}
