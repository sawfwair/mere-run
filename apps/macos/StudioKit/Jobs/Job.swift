import Combine
import Foundation

/// Identity of one job for the lifetime of the app session. Distinct from the durable Studio
/// request id (`JobRequest.requestID`), which names a library row and may be re-run.
package struct JobID: Hashable, Codable, Sendable {
    package let raw: UUID

    package init(raw: UUID = UUID()) {
        self.raw = raw
    }
}

/// The concurrency lane a job executes in. Lanes are independent: a saturated inference lane
/// never delays a `model list`, and a readiness probe never takes an inference slot.
package enum JobLane: Hashable, CaseIterable, Sendable {
    /// Memory-heavy model runs: generation, chat turns, training, pulls. Small cap, FIFO queue.
    case inference
    /// Short CLI reads and writes: list, status, config, guide. Larger cap, FIFO queue.
    case utility
    /// Readiness and status probes. Never queued; deduplicated by `JobRequest.dedupeKey`.
    case probe

    /// How many jobs may execute at once in this lane; further submissions wait in FIFO order.
    package var capacity: Int {
        switch self {
        case .inference: return 2
        case .utility: return 4
        case .probe: return Int.max
        }
    }
}

/// What a job runs: a catalog command, or raw CLI arguments a Studio surface built itself.
package enum JobCommand {
    /// A catalog command. The template and draft are the immutable snapshot the lifecycle reads
    /// for validation, output detection, sidecar discovery and interactive protocols.
    case templated(CommandTemplate, CommandDraft)
    /// Raw `mere.run` arguments with no catalog template: utility reads and writes (`model list`,
    /// `config set`) and readiness/status probes. There is nothing to validate, no output file
    /// to detect and no sidecars; the complete stdout and stderr are captured for the submitter
    /// instead (`JobResult.standardOutput` / `standardError`).
    case raw(arguments: [String])

    package var template: CommandTemplate? {
        if case .templated(let template, _) = self { return template }
        return nil
    }

    package var draft: CommandDraft? {
        if case .templated(_, let draft) = self { return draft }
        return nil
    }

    package var rawArguments: [String]? {
        if case .raw(let arguments) = self { return arguments }
        return nil
    }

    package var isRaw: Bool {
        if case .raw = self { return true }
        return false
    }
}

/// Everything a job needs to run, captured at submission and never mutated afterwards. A queued
/// job executes from this snapshot, so it never observes later edits to the Advanced draft or
/// Settings.
package struct JobRequest {
    package let lane: JobLane
    package let command: JobCommand
    /// Durable Studio request id (a library row, or a per-turn id for chat). Nil for runs started
    /// from the Advanced console, which have no library row, and for raw commands.
    package let requestID: UUID?
    /// The conversation this run is a turn of, when chat/code. Conversation turns skip artifact
    /// detection and keep the full reply beyond the console buffer.
    package let conversationID: UUID?
    /// The fully resolved child-process launch: executable, argv, working directory, environment
    /// (secrets travel here, never in argv) and whether stdin stays open for steering.
    package let configuration: MereRunProcessConfiguration
    /// Shell-quoted command line for the console, library rows and results (secrets masked).
    package let displayCommand: String
    /// For `.probe` jobs: a resubmission with the same key and the same configuration returns the
    /// in-flight job instead of starting another; a different configuration supersedes it.
    package let dedupeKey: String?

    /// A catalog command.
    package init(
        lane: JobLane,
        template: CommandTemplate,
        draft: CommandDraft,
        requestID: UUID? = nil,
        conversationID: UUID? = nil,
        configuration: MereRunProcessConfiguration,
        displayCommand: String,
        dedupeKey: String? = nil
    ) {
        self.lane = lane
        command = .templated(template, draft)
        self.requestID = requestID
        self.conversationID = conversationID
        self.configuration = configuration
        self.displayCommand = displayCommand
        self.dedupeKey = dedupeKey
    }

    private init(
        lane: JobLane,
        arguments: [String],
        configuration: MereRunProcessConfiguration,
        displayCommand: String,
        dedupeKey: String?
    ) {
        self.lane = lane
        command = .raw(arguments: arguments)
        requestID = nil
        conversationID = nil
        self.configuration = configuration
        self.displayCommand = displayCommand
        self.dedupeKey = dedupeKey
    }

    /// A raw-argument command in the utility lane: a short CLI read or write whose complete
    /// output the submitter awaits. `arguments` are the `mere.run` arguments as typed (the
    /// launcher prefix, if any, lives in `configuration.arguments`).
    package static func utility(
        arguments: [String],
        configuration: MereRunProcessConfiguration,
        displayCommand: String
    ) -> JobRequest {
        JobRequest(
            lane: .utility,
            arguments: arguments,
            configuration: configuration,
            displayCommand: displayCommand,
            dedupeKey: nil
        )
    }

    /// A raw-argument probe: never queued, deduplicated by `dedupeKey` (see `JobStore.submit`).
    package static func probe(
        arguments: [String],
        configuration: MereRunProcessConfiguration,
        displayCommand: String,
        dedupeKey: String
    ) -> JobRequest {
        JobRequest(
            lane: .probe,
            arguments: arguments,
            configuration: configuration,
            displayCommand: displayCommand,
            dedupeKey: dedupeKey
        )
    }

    package var template: CommandTemplate? { command.template }
    package var draft: CommandDraft? { command.draft }
    package var rawArguments: [String]? { command.rawArguments }
    package var templateID: CommandTemplateID? { command.template?.id }
}

/// Why a job never launched its process.
package enum JobPreflightFailure: Equatable {
    /// The template rejected the draft (for example "Prompt is required."). Reported as exit 64,
    /// the conventional usage-error status.
    case invalidRequest(String)
    /// The requested output directory could not be created.
    case outputLocationUnavailable(String)

    package var message: String {
        switch self {
        case .invalidRequest(let message), .outputLocationUnavailable(let message):
            return message
        }
    }

    package var exitCode: Int32 {
        switch self {
        case .invalidRequest: return 64
        case .outputLocationUnavailable: return -1
        }
    }
}

package enum JobState: Equatable {
    case queued
    case running(since: Date)
    case finished(exit: Int32, at: Date)
    /// Terminated at the user's request, or removed from the queue before it started. `exit` is
    /// the process's termination status (15 for SIGTERM), or `JobResult.cancelledBeforeStartExitCode`
    /// for a job that never launched.
    case cancelled(exit: Int32, at: Date)
    case preflightFailed(JobPreflightFailure)

    package var isQueued: Bool {
        if case .queued = self { return true }
        return false
    }

    package var isRunning: Bool {
        if case .running = self { return true }
        return false
    }

    /// Queued or running: the job still occupies (or waits for) a lane slot.
    package var isActive: Bool {
        isQueued || isRunning
    }

    package var isTerminal: Bool {
        !isActive
    }

    package var isPreflightFailure: Bool {
        if case .preflightFailed = self { return true }
        return false
    }

    package var exitCode: Int32? {
        switch self {
        case .queued, .running:
            return nil
        case .finished(let exit, _), .cancelled(let exit, _):
            return exit
        case .preflightFailed(let failure):
            return failure.exitCode
        }
    }
}

package enum LogStream {
    case system
    case stdout
    case stderr

    package var label: String {
        switch self {
        case .system: return "mere"
        case .stdout: return "out"
        case .stderr: return "err"
        }
    }
}

package struct LogLine: Identifiable, Equatable {
    package let id = UUID()
    package let date = Date()
    package let stream: LogStream
    package let text: String
}

/// A bounded, append-only console log. Once `capacity` lines are held the oldest are dropped,
/// so a chatty multi-hour job never grows without limit.
package struct LogRing: Equatable {
    package static let defaultCapacity = 1200

    package let capacity: Int
    package private(set) var lines: [LogLine] = []

    package init(capacity: Int = LogRing.defaultCapacity) {
        self.capacity = capacity
    }

    package var isEmpty: Bool { lines.isEmpty }

    package mutating func append(_ line: LogLine) {
        lines.append(line)
        if lines.count > capacity {
            lines.removeFirst(lines.count - capacity)
        }
    }

    /// Appends every non-blank line of `text` (carriage returns count as line breaks, so
    /// progress redraws become separate lines rather than one growing line).
    package mutating func append(_ text: String, stream: LogStream) {
        for line in Self.normalizedLines(text) {
            append(LogLine(stream: stream, text: line))
        }
    }

    /// Splits a raw chunk into trimmed, non-empty lines.
    package static func normalizedLines(_ text: String) -> [String] {
        text
            .replacingOccurrences(of: "\r", with: "\n")
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}

/// One file a job produced. The primary artifact is the run's main output (the `--output` path
/// or the path the CLI printed); sidecars are the related files discovered next to it.
package struct Artifact: Hashable, Identifiable {
    package enum Role: Hashable {
        case primary
        case sidecar
    }

    package let url: URL
    package let role: Role
    /// What this sidecar is, in the CLI receipt's vocabulary (`detections`, `masks`,
    /// `daw-bundle`, …). Nil for the primary artifact, and for a sidecar found by probing whose
    /// purpose could not be inferred. See `StudioArtifactRole`.
    package let sidecarRole: String?

    package var id: URL { url }

    /// A display label for the sidecar's role ("Structured prompt", "DAW bundle"), or nil when
    /// the run reported none and the result surface should fall back to the file name.
    package var roleLabel: String? {
        StudioArtifactRole.label(for: sidecarRole)
    }

    package init(url: URL, role: Role, sidecarRole: String? = nil) {
        self.url = url
        self.role = role
        self.sidecarRole = role == .primary ? nil : sidecarRole
    }
}

/// The durable outcome of one job, delivered exactly once through `JobStore.completions`.
package struct JobResult: Identifiable, Equatable {
    /// Exit status reported for a job cancelled while still queued. It matches the SIGTERM status
    /// a running job reports when terminated, so library rows show the same failure either way.
    package static let cancelledBeforeStartExitCode: Int32 = 15

    package let id: UUID
    package let requestID: UUID?
    /// The catalog command this was a run of. Raw-argument jobs report `.custom`, the catalog's
    /// own raw-argument command.
    package let templateID: CommandTemplateID
    package let commandPreview: String
    package let exitCode: Int32
    package let outputURL: URL?
    package let artifactURLs: [URL]
    /// What each sidecar in `artifactURLs` is, keyed by its standardized path. Empty for runs
    /// that produced no sidecars and for CLIs older than `--receipt` whose sidecar roles could
    /// not be inferred.
    package let artifactRoles: [String: String]
    package let outputText: String?
    package let completedAt: Date
    /// When this run was a chat/code turn, the conversation it belongs to (so completion routes
    /// to the thread rather than the legacy single-result path).
    package let conversationID: UUID?
    /// For raw-argument jobs, the complete stdout, uncapped, so the submitter can parse structured
    /// output (`--json`, `model list`) whole. Nil for catalog commands, which report `outputText`.
    package let standardOutput: String?
    /// For raw-argument jobs, the complete stderr. Nil for catalog commands.
    package let standardError: String?

    package init(
        id: UUID = UUID(),
        requestID: UUID? = nil,
        templateID: CommandTemplateID,
        commandPreview: String,
        exitCode: Int32,
        outputURL: URL?,
        artifactURLs: [URL] = [],
        artifactRoles: [String: String] = [:],
        outputText: String?,
        completedAt: Date = Date(),
        conversationID: UUID? = nil,
        standardOutput: String? = nil,
        standardError: String? = nil
    ) {
        self.id = id
        self.requestID = requestID
        self.templateID = templateID
        self.commandPreview = commandPreview
        self.exitCode = exitCode
        self.outputURL = outputURL
        self.artifactURLs = artifactURLs
        self.artifactRoles = artifactRoles
        self.outputText = outputText
        self.completedAt = completedAt
        self.conversationID = conversationID
        self.standardOutput = standardOutput
        self.standardError = standardError
    }
}

/// The pre-`JobStore` name of `JobResult`, kept while views still spell it this way.
package typealias MereRunRunResult = JobResult

/// One render request on the stdin of a resident `video session` process.
package struct StudioVideoSessionRequest: Encodable {
    package let id: String
    package let prompt: String
    package let output: String
    package let width: Int
    package let height: Int
    package let numFrames: Int
    package let fps: Int
    package let seed: Int?
    package let image: String?
    package let imageStrength: Double?
    package let endImage: String?
    package let endImageStrength: Double?
}

package struct StudioVideoSessionResponse: Decodable {
    package let status: String
    package let output: String?
    package let error: String?
}

/// One submitted command and everything observable about it: state, status line, progress, log,
/// live stdout, artifacts and result. One `ObservableObject` per job, so a view observing a
/// background job never re-renders for the foreground one.
///
/// `JobStore` is the only writer. The mutating methods below are the store's, grouped by
/// lifecycle stage; views read the published properties.
@MainActor
package final class Job: ObservableObject, Identifiable {
    package let id: JobID
    package let request: JobRequest
    package let submittedAt: Date
    /// The explicit `--output` path the request asked for, when the template produces one.
    package let expectedOutput: URL?

    @Published package private(set) var state: JobState = .queued
    /// Human status line: "Queued", "Running", "Downloading model", "Generated: x.png",
    /// "Completed", "Exited 64", "Resident session ready".
    @Published package private(set) var status = "Queued"
    @Published package private(set) var progress: StudioRunProgress?
    @Published package private(set) var log = LogRing()
    /// Capped, NUL-stripped stdout for the console's live pane.
    @Published package private(set) var liveText = ""
    @Published package private(set) var artifacts: [Artifact] = []
    /// Think-stripped streaming reply for conversation turns; nil for other jobs and once finished.
    @Published package private(set) var conversationLiveText: String?
    @Published package private(set) var result: JobResult?
    /// When the process launched; nil for jobs that never ran (queued, cancelled early, preflight).
    package private(set) var startedAt: Date?

    // Store-owned process state. Not published: none of it is meaningful to a view.
    package var process: MereRunRunningProcess?
    package var outputWatchTask: Task<Void, Never>?
    package var cancelRequested = false

    private var stdoutBuffer = ""
    private var stderrBuffer = ""
    /// Unbounded stdout accumulator for conversation turns (so a long reply is captured in full)
    /// and raw-argument jobs (so a submitter can parse `--json` output whole) — `stdoutBuffer`
    /// is capped at 32 KB for the console.
    private var fullOutput = ""
    /// Unbounded stderr accumulator for raw-argument jobs, whose submitters read stderr as data.
    private var fullErrorOutput = ""
    /// Length of `fullOutput` at the last live think-strip; used to throttle re-stripping the
    /// whole accumulator on every chunk (it would otherwise be O(n²) over a long reply).
    private var lastLiveStripLength = 0
    private var interactiveOutputBuffer = ""

    private static let stdoutBufferByteLimit = 32 * 1024
    private static let stderrBufferByteLimit = 32 * 1024
    /// Min growth in a conversation reply before the live think-stripped text is re-published,
    /// bounding per-chunk re-strip cost. The final reply is always published in full on finish.
    private static let liveStripGranularity = 80

    package init(id: JobID = JobID(), request: JobRequest, submittedAt: Date = Date()) {
        self.id = id
        self.request = request
        self.submittedAt = submittedAt
        if case .templated(let template, let draft) = request.command {
            expectedOutput = ArtifactResolver.expectedOutput(template: template, draft: draft)
        } else {
            expectedOutput = nil
        }
    }

    package var lane: JobLane { request.lane }
    package var displayCommand: String { request.displayCommand }
    package var exitCode: Int32? { state.exitCode }
    package var primaryArtifactURL: URL? {
        artifacts.first { $0.role == .primary }?.url
    }
    /// Whether output-file detection applies: catalog commands other than conversation turns.
    /// Conversation replies are prose, and raw commands have no output contract to read.
    package var detectsArtifacts: Bool { !request.command.isRaw && !isConversationTurn }
    /// Whether this run will print the `--receipt` result line naming everything it wrote. Such
    /// a run needs no filesystem poll: the receipt arrives on stdout and settles the artifacts.
    package var expectsRunReceipt: Bool {
        detectsArtifacts
            && request.templateID?.emitsRunReceipt == true
            && request.configuration.arguments.contains(StudioMachineOutputFlags.receipt)
    }
    private var isConversationTurn: Bool { request.conversationID != nil }
    private var isRawCommand: Bool { request.command.isRaw }
    /// The template a result is attributed to; raw commands report the catalog's raw-argument one.
    private var resultTemplateID: CommandTemplateID { request.templateID ?? .custom }

    // MARK: Lifecycle (JobStore only)

    package func markRunning(status: String, at date: Date = Date()) {
        state = .running(since: date)
        startedAt = date
        self.status = status
    }

    package func setStatus(_ status: String) {
        self.status = status
    }

    /// Appends a controller-side note ("Termination requested.", the launched command) without
    /// progress parsing, so a note that happens to look like a progress line stays in the log.
    package func note(_ text: String, stream: LogStream) {
        log.append(text, stream: stream)
    }

    package func setPrimaryArtifact(_ url: URL?) {
        let sidecars = artifacts.filter { $0.role == .sidecar }
        artifacts = (url.map { [Artifact(url: $0, role: .primary)] } ?? []) + sidecars
    }

    /// Consumes one chunk of process output: buffers it, parses progress, updates the live pane
    /// and interactive-session status, and appends the remaining lines to the log.
    package func consume(_ text: String, stream: LogStream, resolver: ArtifactResolver) {
        switch stream {
        case .stdout:
            if request.templateID == .videoSession {
                consumeVideoSessionOutput(text)
            }
            stdoutBuffer = Self.trimmed(stdoutBuffer + text, toByteLimit: Self.stdoutBufferByteLimit)
            liveText = stdoutBuffer.replacingOccurrences(of: "\0", with: "")
            if isConversationTurn || isRawCommand {
                fullOutput += text
            }
            if isConversationTurn {
                // Conversation turns publish a live, think-stripped view of the full reply so a
                // streaming bubble renders even when backgrounded. The streaming variant hides an
                // in-progress (unclosed) reasoning block. Re-stripping the whole accumulator on
                // every chunk is O(n²), so throttle to ~every 80 chars of growth; finish always
                // publishes the complete, fully-stripped reply.
                // Publish immediately on the first chunk (fast first render), then throttle.
                if lastLiveStripLength == 0
                    || fullOutput.count - lastLiveStripLength >= Self.liveStripGranularity {
                    lastLiveStripLength = fullOutput.count
                    conversationLiveText = ConversationTranscript.stripThinkTags(
                        fullOutput.replacingOccurrences(of: "\0", with: ""),
                        streaming: true
                    )
                }
            } else if detectsArtifacts {
                refreshPrimaryArtifact(resolver: resolver)
            }
        case .stderr:
            stderrBuffer = Self.trimmed(stderrBuffer + text, toByteLimit: Self.stderrBufferByteLimit)
            if isRawCommand {
                fullErrorOutput += text
            }
            if request.templateID == .videoSession,
               text.localizedCaseInsensitiveContains("session ready") {
                status = "Resident session ready"
            }
        case .system:
            break
        }

        for line in LogRing.normalizedLines(text) {
            // Collapse repeated carriage-return progress updates into one structured value
            // instead of flooding the log with hundreds of lines.
            if let progress = StudioProgressParser.parse(line) {
                self.progress = progress
                continue
            }
            // The receipt is the app's transport for reading the run's outputs, not something a
            // person asked the CLI to print, so it stays out of the console.
            if StudioRunReceipt.isReceiptLine(line) { continue }
            log.append(LogLine(stream: stream, text: line))
        }
    }

    /// Re-runs output detection against what has been written so far. Returns true when a new
    /// primary artifact appeared, so the caller can publish it while the process is still alive.
    @discardableResult
    package func refreshPrimaryArtifact(resolver: ArtifactResolver) -> Bool {
        guard state.isRunning else { return false }
        guard let detected = resolver.primaryOutput(expected: expectedOutput, stdout: stdoutBuffer),
              detected != primaryArtifactURL else {
            return false
        }
        setPrimaryArtifact(detected)
        status = "Generated: \(detected.lastPathComponent)"
        return true
    }

    /// Finalizes the job after its process exited: resolves artifacts and the captured text,
    /// settles the state and status, and records the result.
    package func finish(exitCode: Int32, at date: Date = Date(), resolver: ArtifactResolver) -> JobResult {
        progress = nil

        // Conversation replies are prose, not artifacts — never run output-file detection on them
        // (a path-like substring in a reply must not become a bogus artifact or status). Raw
        // commands have no output contract to read either.
        let resolution: ArtifactResolution
        if detectsArtifacts, case .templated(let template, let draft) = request.command {
            resolution = resolver.resolve(
                template: template,
                draft: draft,
                expected: expectedOutput,
                stdout: stdoutBuffer
            )
        } else {
            resolution = .empty
        }
        let detectedOutput = resolution.primary
        // Conversation turns finalize from the unbounded, think-stripped accumulator so long
        // replies are not clipped by the 32 KB console buffer; raw commands from their complete
        // stdout/stderr for the same reason; other modes use the console buffers.
        let outputText: String?
        if isConversationTurn {
            // Strip reasoning for conversation turns on EVERY exit path (not just success) so
            // <think> blocks and STDERR never leak into the next turn's replayed prompt.
            outputText = ConversationTranscript.stripThinkTags(fullOutput.replacingOccurrences(of: "\0", with: ""))
        } else if isRawCommand {
            outputText = Self.capturedResultText(stdout: fullOutput, stderr: fullErrorOutput, exitCode: exitCode)
        } else {
            // The receipt is the app's own transport, not part of the run's output: it must not
            // land in a library row or in a transcript the CLI printed to stdout.
            outputText = Self.capturedResultText(
                stdout: StudioRunReceipt.strippingReceiptLines(from: stdoutBuffer),
                stderr: stderrBuffer,
                exitCode: exitCode
            )
        }

        // A resident session (`video session`) reports its render over its own stdin protocol,
        // so keep the live primary when nothing was resolved from the run's output.
        let primary = detectedOutput ?? primaryArtifactURL
        let sidecars = resolution.sidecars.filter { $0.url != primary }
        artifacts = (primary.map { [Artifact(url: $0, role: .primary)] } ?? []) + sidecars
        let artifactURLs = artifacts.map(\.url)
        let artifactRoles = Dictionary(
            sidecars.compactMap { artifact in
                artifact.sidecarRole.map { (artifact.url.standardizedFileURL.path, $0) }
            },
            uniquingKeysWith: { first, _ in first }
        )

        if exitCode == 0 {
            status = detectedOutput.map { "Completed: \($0.lastPathComponent)" } ?? "Completed"
            log.append(LogLine(stream: .system, text: "Completed with exit code 0."))
        } else {
            status = "Exited \(exitCode)"
            log.append(LogLine(stream: .system, text: "Exited with code \(exitCode)."))
        }
        conversationLiveText = nil
        state = cancelRequested && exitCode != 0
            ? .cancelled(exit: exitCode, at: date)
            : .finished(exit: exitCode, at: date)
        let result = JobResult(
            requestID: request.requestID,
            templateID: resultTemplateID,
            commandPreview: displayCommand,
            exitCode: exitCode,
            outputURL: detectedOutput,
            artifactURLs: artifactURLs,
            artifactRoles: artifactRoles,
            outputText: outputText,
            completedAt: date,
            conversationID: request.conversationID,
            standardOutput: isRawCommand ? fullOutput : nil,
            standardError: isRawCommand ? fullErrorOutput : nil
        )
        releaseBuffers()
        self.result = result
        return result
    }

    package func failPreflight(_ failure: JobPreflightFailure, at date: Date = Date()) -> JobResult {
        state = .preflightFailed(failure)
        status = failure.message
        log.append(failure.message, stream: .system)
        let result = JobResult(
            requestID: request.requestID,
            templateID: resultTemplateID,
            commandPreview: displayCommand,
            exitCode: failure.exitCode,
            outputURL: nil,
            outputText: failure.message,
            completedAt: date,
            conversationID: request.conversationID
        )
        self.result = result
        return result
    }

    package func cancelBeforeStart(at date: Date = Date()) -> JobResult {
        state = .cancelled(exit: JobResult.cancelledBeforeStartExitCode, at: date)
        status = "Cancelled"
        log.append(LogLine(stream: .system, text: "Cancelled before start."))
        let result = JobResult(
            requestID: request.requestID,
            templateID: resultTemplateID,
            commandPreview: displayCommand,
            exitCode: JobResult.cancelledBeforeStartExitCode,
            outputURL: nil,
            outputText: nil,
            completedAt: date,
            conversationID: request.conversationID
        )
        self.result = result
        return result
    }

    // MARK: Private

    private func consumeVideoSessionOutput(_ text: String) {
        interactiveOutputBuffer += text
        let lines = interactiveOutputBuffer.components(separatedBy: .newlines)
        interactiveOutputBuffer = lines.last ?? ""

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        for line in lines.dropLast() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty,
                  let data = trimmed.data(using: .utf8),
                  let response = try? decoder.decode(StudioVideoSessionResponse.self, from: data) else {
                continue
            }

            if response.status == "result", let output = response.output {
                let url = URL(fileURLWithPath: output)
                setPrimaryArtifact(url)
                status = "Generated: \(url.lastPathComponent)"
            } else if response.status == "error" {
                status = "Resident render failed"
                if let error = response.error, !error.isBlank {
                    log.append(LogLine(stream: .stderr, text: error))
                }
            }
        }
    }

    /// The captured stdout, with stderr appended only when the command failed.
    private static func capturedResultText(stdout: String, stderr: String, exitCode: Int32) -> String? {
        let stdout = nonEmptyTrimmed(stdout)
        guard exitCode != 0, let stderr = nonEmptyTrimmed(stderr) else {
            return stdout
        }
        if let stdout {
            return "\(stdout)\n\nSTDERR\n\(stderr)"
        }
        return stderr
    }

    /// Drops the raw accumulators once the result is settled; the log, live pane, artifacts and
    /// result keep everything a view still shows.
    private func releaseBuffers() {
        stdoutBuffer = ""
        stderrBuffer = ""
        fullOutput = ""
        fullErrorOutput = ""
        interactiveOutputBuffer = ""
    }

    private static func trimmed(_ buffer: String, toByteLimit limit: Int) -> String {
        guard buffer.utf8.count > limit else { return buffer }
        return String(decoding: buffer.utf8.suffix(limit), as: UTF8.self)
    }

    private static func nonEmptyTrimmed(_ buffer: String) -> String? {
        let trimmed = buffer
            .replacingOccurrences(of: "\0", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
