import Combine
import Foundation

package enum JobStoreError: LocalizedError {
    case notRunning

    package var errorDescription: String? {
        "This job is not running."
    }
}

/// Owns every job's lifecycle: lane capacities and FIFO queues, child processes behind the
/// `MereRunProcessRunning` seam, output consumption, live artifact detection, cancellation and
/// completion. Jobs stay observable after they finish (the most recent
/// `finishedJobRetentionLimit` per lane), so a view can keep showing a result without the store
/// growing without bound and without a burst of probes evicting a run the user is looking at.
///
/// Delivery order on completion: `events` receives `.finished(job, result)` first, then
/// `completions` receives the result, then any `result(for:)` awaiters resume. All three run
/// synchronously on the main actor from the same call, so a subscriber to any of them sees the
/// job's final published state.
@MainActor
package final class JobStore: ObservableObject {
    /// Synchronous lifecycle notifications, sent after the job's published state has changed.
    /// A subscriber therefore reads consistent values without an extra main-actor hop.
    package enum Event {
        /// The job left the queue and its process launch was attempted.
        case started(Job)
        /// One chunk of process output arrived, exactly as the child wrote it. Sent after the job
        /// folded it into its log and live text and before the matching `.changed`, so a
        /// submitter that needs the raw stream (a JSONL protocol, a progress feed) reads chunks
        /// rather than diffing the capped `liveText`.
        case output(Job, LogStream, String)
        /// Log, progress, status, live text or artifacts changed while running.
        case changed(Job)
        /// The job reached a terminal state; `completions` receives the same result right after.
        case finished(Job, JobResult)
    }

    package let events = PassthroughSubject<Event, Never>()
    /// Lossless completion stream: every submitted job produces exactly one result here, whether
    /// it ran, failed preflight or was cancelled while queued.
    package let completions = PassthroughSubject<JobResult, Never>()

    @Published package private(set) var jobs: [JobID: Job] = [:]
    /// Every retained job id in submission order.
    @Published package private(set) var order: [JobID] = []

    /// How many finished jobs per lane stay observable before that lane's oldest are dropped.
    /// Eviction never crosses lanes, so probe and utility churn cannot drop a finished run.
    package static let finishedJobRetentionLimit = 50
    /// How often a running job is re-probed for its `--output` file while the CLI is silent.
    /// Only commands that print no `--receipt` line are watched this way; see `start`.
    package static let defaultOutputWatchInterval: Duration = .milliseconds(350)

    private let processRunner: MereRunProcessRunning
    private let resolver: ArtifactResolver
    private let outputWatchInterval: Duration
    private var queues: [JobLane: [JobID]] = [:]
    private var runningIDs: [JobLane: [JobID]] = [:]
    private var pendingResults: [JobID: [CheckedContinuation<JobResult, Never>]] = [:]

    package init(
        processRunner: MereRunProcessRunning = FoundationMereRunProcessRunner(),
        fileSystem: MereRunFileProbing = FileManager.default,
        outputWatchInterval: Duration = JobStore.defaultOutputWatchInterval
    ) {
        self.processRunner = processRunner
        resolver = ArtifactResolver(fileSystem: fileSystem)
        self.outputWatchInterval = outputWatchInterval
    }

    // MARK: Reading

    /// Every retained job, oldest first.
    package var all: [Job] {
        order.compactMap { jobs[$0] }
    }

    package var running: [Job] {
        JobLane.allCases.flatMap { running(in: $0) }
    }

    package func running(in lane: JobLane) -> [Job] {
        (runningIDs[lane] ?? []).compactMap { jobs[$0] }
    }

    package func queued(in lane: JobLane) -> [Job] {
        (queues[lane] ?? []).compactMap { jobs[$0] }
    }

    package func hasCapacity(in lane: JobLane) -> Bool {
        (runningIDs[lane]?.count ?? 0) < lane.capacity
    }

    package func job(_ id: JobID) -> Job? {
        jobs[id]
    }

    /// The job for a durable Studio request id, preferring one still active over an older run of
    /// the same request.
    package func job(requestID: UUID) -> Job? {
        let matches = all.filter { $0.request.requestID == requestID }
        return matches.last { $0.state.isActive } ?? matches.last
    }

    package func isRunning(capability: String) -> Bool {
        running.contains { $0.request.templateID?.capabilityID == capability }
    }

    package func isRunning(template: CommandTemplateID) -> Bool {
        running.contains { $0.request.templateID == template }
    }

    // MARK: Submitting

    /// Registers the job and starts it when its lane has a free slot; otherwise it waits in FIFO
    /// order. Probe jobs with a `dedupeKey` return the matching in-flight job instead of starting
    /// a duplicate, or supersede it when the configuration differs.
    ///
    /// `id` lets a submitter name the job before submitting it (so it can hand the id to a
    /// cancel button up front); it must be unique for the session. A deduplicated probe keeps
    /// the in-flight job's id, which is what the call returns.
    @discardableResult
    package func submit(_ request: JobRequest, id: JobID = JobID()) -> JobID {
        if request.lane == .probe, let key = request.dedupeKey,
           let existing = all.last(where: {
               // A superseded probe stays `.running` until its process dies; it must not be
               // handed back to a new submitter that would then await its SIGTERM result.
               $0.state.isActive && !$0.cancelRequested && $0.lane == .probe && $0.request.dedupeKey == key
           }) {
            if existing.request.configuration == request.configuration {
                return existing.id
            }
            cancel(existing.id)
        }

        precondition(jobs[id] == nil, "Job \(id.raw) was already submitted in this session.")
        let job = Job(id: id, request: request)
        jobs[job.id] = job
        order.append(job.id)
        queues[request.lane, default: []].append(job.id)
        pump(request.lane)
        return job.id
    }

    /// Terminates a running job (SIGTERM) or removes a queued one. Returns false when the job is
    /// unknown or already finished.
    @discardableResult
    package func cancel(_ id: JobID) -> Bool {
        guard let job = jobs[id] else { return false }
        switch job.state {
        case .queued:
            queues[job.lane]?.removeAll { $0 == id }
            complete(job, with: job.cancelBeforeStart())
            return true
        case .running:
            guard let process = job.process else { return false }
            job.cancelRequested = true
            process.terminate()
            job.note("Termination requested.", stream: .system)
            events.send(.changed(job))
            return true
        case .finished, .cancelled, .preflightFailed:
            return false
        }
    }

    /// Sends SIGINT to a running job so a CLI that traps it can finish cleanly.
    @discardableResult
    package func interrupt(_ id: JobID) -> Bool {
        guard let job = jobs[id], job.state.isRunning, let process = job.process else { return false }
        process.interrupt()
        return true
    }

    /// Writes raw text to a running job's stdin. Throws when the job is not running or its
    /// process was launched without stdin.
    package func send(_ text: String, to id: JobID) throws {
        guard let job = jobs[id], job.state.isRunning, let process = job.process else {
            throw JobStoreError.notRunning
        }
        try process.sendStandardInput(text)
    }

    /// Adds a note to a job's log (the console's own commentary, not process output).
    package func annotate(_ id: JobID, _ text: String, stream: LogStream = .system) {
        guard let job = jobs[id] else { return }
        job.note(text, stream: stream)
        events.send(.changed(job))
    }

    /// The job's result, waiting for completion if needed. Nil only when the id is unknown
    /// (never submitted, or already dropped from the retained history).
    package func result(for id: JobID) async -> JobResult? {
        guard let job = jobs[id] else { return nil }
        if let result = job.result { return result }
        return await withCheckedContinuation { continuation in
            pendingResults[id, default: []].append(continuation)
        }
    }

    /// Drops every queued job and terminates every running one. Called on app termination so
    /// child CLIs are never orphaned and nothing launches during teardown.
    package func terminateAll() {
        for lane in JobLane.allCases {
            for job in queued(in: lane) {
                cancel(job.id)
            }
        }
        for job in running {
            job.cancelRequested = true
            job.process?.terminate()
        }
    }

    // MARK: Lifecycle

    private func pump(_ lane: JobLane) {
        while hasCapacity(in: lane), let next = queues[lane]?.first {
            queues[lane]?.removeFirst()
            guard let job = jobs[next] else { continue }
            start(job)
        }
    }

    private func start(_ job: Job) {
        let request = job.request
        // Preflight is a catalog concept: a raw command has nothing to validate and no output
        // location to prepare.
        if case .templated(let template, let draft) = request.command {
            if let message = template.validationMessage(for: draft) {
                complete(job, with: job.failPreflight(.invalidRequest(message)))
                return
            }
            if let message = Self.prepareOutputLocation(template: template, draft: draft) {
                complete(job, with: job.failPreflight(.outputLocationUnavailable(message)))
                return
            }
        }

        runningIDs[job.lane, default: []].append(job.id)
        job.markRunning(status: request.templateID == .modelPull ? "Downloading model" : "Running")
        events.send(.started(job))
        job.note(job.displayCommand, stream: .system)
        events.send(.changed(job))
        // A run that prints a receipt names its outputs on stdout, so `consume` settles them the
        // moment the line arrives; only the commands without one need the filesystem poll.
        if job.detectsArtifacts, !job.expectsRunReceipt {
            startOutputWatch(for: job)
        }

        do {
            job.process = try processRunner.start(
                configuration: request.configuration,
                stdout: { [weak self, weak job] text in
                    Task { @MainActor in
                        guard let self, let job else { return }
                        self.consume(text, stream: .stdout, for: job)
                    }
                },
                stderr: { [weak self, weak job] text in
                    Task { @MainActor in
                        guard let self, let job else { return }
                        self.consume(text, stream: .stderr, for: job)
                    }
                },
                termination: { [weak self, weak job] code in
                    Task { @MainActor in
                        guard let self, let job else { return }
                        self.finish(job, exitCode: code)
                    }
                }
            )
        } catch {
            // Through the stderr path, not `note`, so the message reaches the result's
            // `outputText` (and the library row) as well as the log.
            job.consume(error.localizedDescription + "\n", stream: .stderr, resolver: resolver)
            finish(job, exitCode: -1)
        }
    }

    private func consume(_ text: String, stream: LogStream, for job: Job) {
        guard job.state.isRunning else { return }
        job.consume(text, stream: stream, resolver: resolver)
        events.send(.output(job, stream, text))
        events.send(.changed(job))
    }

    private func finish(_ job: Job, exitCode: Int32) {
        guard job.state.isRunning else { return }
        job.outputWatchTask?.cancel()
        job.outputWatchTask = nil
        job.process = nil
        runningIDs[job.lane]?.removeAll { $0 == job.id }
        complete(job, with: job.finish(exitCode: exitCode, resolver: resolver))
    }

    private func complete(_ job: Job, with result: JobResult) {
        events.send(.finished(job, result))
        completions.send(result)
        pendingResults.removeValue(forKey: job.id)?.forEach { $0.resume(returning: result) }
        evictFinishedJobsIfNeeded(in: job.lane)
        pump(job.lane)
    }

    private func evictFinishedJobsIfNeeded(in lane: JobLane) {
        let finished = order.filter { id in
            guard let job = jobs[id] else { return false }
            return job.lane == lane && job.state.isTerminal
        }
        let excess = finished.count - Self.finishedJobRetentionLimit
        guard excess > 0 else { return }
        let evicted = Set(finished.prefix(excess))
        order.removeAll { evicted.contains($0) }
        for id in evicted {
            jobs[id] = nil
        }
    }

    private func startOutputWatch(for job: Job) {
        let interval = outputWatchInterval
        job.outputWatchTask = Task { [weak self, weak job] in
            while !Task.isCancelled {
                let alive = await MainActor.run { () -> Bool in
                    guard let self, let job else { return false }
                    if job.refreshPrimaryArtifact(resolver: self.resolver) {
                        self.events.send(.changed(job))
                    }
                    return true
                }
                guard alive else { return }
                try? await Task.sleep(for: interval)
            }
        }
    }

    /// Ensures the run's output directory exists. Returns `nil` on success, or a diagnostic
    /// message when the directory could not be created (so the caller can surface it per run).
    private static func prepareOutputLocation(template: CommandTemplate, draft: CommandDraft) -> String? {
        guard !draft.outputPath.isBlank else {
            return nil
        }

        let outputURL = URL(fileURLWithPath: NSString(string: draft.outputPath).expandingTildeInPath)
        let directoryURL: URL
        switch template.outputKind {
        case .file:
            directoryURL = outputURL.deletingLastPathComponent()
        case .directory:
            directoryURL = outputURL
        case .none:
            return nil
        }

        do {
            try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
            return nil
        } catch {
            return "Could not create output directory \(directoryURL.path): \(error.localizedDescription)"
        }
    }
}

// MARK: - Interactive sessions

extension JobStore {
    /// Sends one steering line to a resident process (Magenta RT live controls) and records it.
    @discardableResult
    package func sendLiveControl(_ line: String, to id: JobID) -> Bool {
        guard let job = jobs[id] else { return false }
        do {
            try send(line + "\n", to: id)
            job.note("Live control → \(line)", stream: .system)
            events.send(.changed(job))
            return true
        } catch {
            job.note(error.localizedDescription, stream: .stderr)
            events.send(.changed(job))
            return false
        }
    }

    /// Submits one render request to a resident `video session` process over its stdin protocol.
    @discardableResult
    package func sendVideoSessionRequest(_ request: StudioVideoSessionRequest, to id: JobID) -> Bool {
        guard let job = jobs[id] else { return false }
        do {
            try FileManager.default.createDirectory(
                at: URL(fileURLWithPath: request.output).deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let encoder = JSONEncoder()
            encoder.keyEncodingStrategy = .convertToSnakeCase
            encoder.outputFormatting = [.sortedKeys]
            let data = try encoder.encode(request)
            guard let line = String(data: data, encoding: .utf8) else {
                throw CocoaError(.fileWriteInapplicableStringEncoding)
            }
            try send(line + "\n", to: id)
            job.setStatus("Rendering resident request")
            job.setPrimaryArtifact(nil)
            job.note("Submitted resident render → \(request.output)", stream: .system)
            events.send(.changed(job))
            return true
        } catch {
            job.setStatus("Session submission failed")
            job.note(error.localizedDescription, stream: .stderr)
            events.send(.changed(job))
            return false
        }
    }
}
