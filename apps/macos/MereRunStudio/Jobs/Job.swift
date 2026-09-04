import Combine
import Foundation

/// Identity of one job for the lifetime of the app session. Distinct from the durable Studio
/// request id (`JobRequest.requestID`), which names a library row and may be re-run.
struct JobID: Hashable, Codable, Sendable {
    let raw: UUID

    init(raw: UUID = UUID()) {
        self.raw = raw
    }
}

/// The concurrency lane a job executes in. Lanes are independent: a saturated inference lane
/// never delays a `model list`, and a readiness probe never takes an inference slot.
enum JobLane: Hashable, CaseIterable, Sendable {
    /// Memory-heavy model runs: generation, chat turns, training, pulls. Small cap, FIFO queue.
    case inference
    /// Short CLI reads and writes: list, status, config, guide. Larger cap, FIFO queue.
    case utility
    /// Readiness and status probes. Never queued; deduplicated by `JobRequest.dedupeKey`.
    case probe

    /// How many jobs may execute at once in this lane; further submissions wait in FIFO order.
    var capacity: Int {
        switch self {
        case .inference: return 2
        case .utility: return 4
        case .probe: return Int.max
        }
    }
}

/// What a job runs: a catalog command, or raw CLI arguments a Studio surface built itself.
enum JobCommand {
    /// A catalog command. The template and draft are the immutable snapshot the lifecycle reads
    /// for validation, output detection, sidecar discovery and interactive protocols.
    case templated(CommandTemplate, CommandDraft)
    /// Raw `mere.run` arguments with no catalog template: utility reads and writes (`model list`,
    /// `config set`) and readiness/status probes. There is nothing to validate, no output file
    /// to detect and no sidecars; the complete stdout and stderr are captured for the submitter
    /// instead (`JobResult.standardOutput` / `standardError`).
    case raw(arguments: [String])

    var template: CommandTemplate? {
        if case .templated(let template, _) = self { return template }
        return nil
    }

    var draft: CommandDraft? {
        if case .templated(_, let draft) = self { return draft }
        return nil
    }

    var rawArguments: [String]? {
        if case .raw(let arguments) = self { return arguments }
        return nil
    }

    var isRaw: Bool {
        if case .raw = self { return true }
        return false
    }
}

/// Everything a job needs to run, captured at submission and never mutated afterwards. A queued
/// job executes from this snapshot, so it never observes later edits to the Advanced draft or
/// Settings.
struct JobRequest {
    let lane: JobLane
    let command: JobCommand
    /// Durable Studio request id (a library row, or a per-turn id for chat). Nil for runs started
    /// from the Advanced console, which have no library row, and for raw commands.
    let requestID: UUID?
    /// The conversation this run is a turn of, when chat/code. Conversation turns skip artifact
    /// detection and keep the full reply beyond the console buffer.
    let conversationID: UUID?
    /// The fully resolved child-process launch: executable, argv, working directory, environment
    /// (secrets travel here, never in argv) and whether stdin stays open for steering.
    let configuration: MereRunProcessConfiguration
    /// Shell-quoted command line for the console, library rows and results (secrets masked).
    let displayCommand: String
    /// For `.probe` jobs: a resubmission with the same key and the same configuration returns the
    /// in-flight job instead of starting another; a different configuration supersedes it.
    let dedupeKey: String?

    /// A catalog command.
    init(
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
    static func utility(
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
    static func probe(
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

    var template: CommandTemplate? { command.template }
    var draft: CommandDraft? { command.draft }
    var rawArguments: [String]? { command.rawArguments }
    var templateID: CommandTemplateID? { command.template?.id }
}

/// Why a job never launched its process.
enum JobPreflightFailure: Equatable {
    /// The template rejected the draft (for example "Prompt is required."). Reported as exit 64,
    /// the conventional usage-error status.
    case invalidRequest(String)
    /// The requested output directory could not be created.
    case outputLocationUnavailable(String)

    var message: String {
        switch self {
        case .invalidRequest(let message), .outputLocationUnavailable(let message):
            return message
        }
    }

    var exitCode: Int32 {
        switch self {
        case .invalidRequest: return 64
        case .outputLocationUnavailable: return -1
        }
    }
}

enum JobState: Equatable {
    case queued
    case running(since: Date)
    case finished(exit: Int32, at: Date)
    /// Terminated at the user's request, or removed from the queue before it started. `exit` is
    /// the process's termination status (15 for SIGTERM), or `JobResult.cancelledBeforeStartExitCode`
    /// for a job that never launched.
    case cancelled(exit: Int32, at: Date)
    case preflightFailed(JobPreflightFailure)

    var isQueued: Bool {
        if case .queued = self { return true }
        return false
    }

    var isRunning: Bool {
        if case .running = self { return true }
        return false
    }

    /// Queued or running: the job still occupies (or waits for) a lane slot.
    var isActive: Bool {
        isQueued || isRunning
    }

    var isTerminal: Bool {
        !isActive
    }

    var isPreflightFailure: Bool {
        if case .preflightFailed = self { return true }
        return false
    }

    var exitCode: Int32? {
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

enum LogStream {
    case system
    case stdout
    case stderr

    var label: String {
        switch self {
        case .system: return "mere"
        case .stdout: return "out"
        case .stderr: return "err"
        }
    }
}

struct LogLine: Identifiable, Equatable {
    let id = UUID()
    let date = Date()
    let stream: LogStream
    let text: String
}

/// A bounded, append-only console log. Once `capacity` lines are held the oldest are dropped,
/// so a chatty multi-hour job never grows without limit.
struct LogRing: Equatable {
    static let defaultCapacity = 1200

    let capacity: Int
    private(set) var lines: [LogLine] = []

    init(capacity: Int = LogRing.defaultCapacity) {
        self.capacity = capacity
    }

    var isEmpty: Bool { lines.isEmpty }

    mutating func append(_ line: LogLine) {
        lines.append(line)
        if lines.count > capacity {
            lines.removeFirst(lines.count - capacity)
        }
    }

    /// Appends every non-blank line of `text` (carriage returns count as line breaks, so
    /// progress redraws become separate lines rather than one growing line).
    mutating func append(_ text: String, stream: LogStream) {
        for line in Self.normalizedLines(text) {
            append(LogLine(stream: stream, text: line))
        }
    }

    /// Splits a raw chunk into trimmed, non-empty lines.
    static func normalizedLines(_ text: String) -> [String] {
        text
            .replacingOccurrences(of: "\r", with: "\n")
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}

/// One file a job produced. The primary artifact is the run's main output (the `--output` path
/// or the path the CLI printed); sidecars are the related files discovered next to it.
struct Artifact: Hashable, Identifiable {
    enum Role: Hashable {
        case primary
        case sidecar
    }

    let url: URL
    let role: Role

    var id: URL { url }
}

/// The durable outcome of one job, delivered exactly once through `JobStore.completions`.
struct JobResult: Identifiable, Equatable {
    /// Exit status reported for a job cancelled while still queued. It matches the SIGTERM status
    /// a running job reports when terminated, so library rows show the same failure either way.
    static let cancelledBeforeStartExitCode: Int32 = 15

    let id: UUID
    let requestID: UUID?
    /// The catalog command this was a run of. Raw-argument jobs report `.custom`, the catalog's
    /// own raw-argument command.
    let templateID: CommandTemplateID
    let commandPreview: String
    let exitCode: Int32
    let outputURL: URL?
    let artifactURLs: [URL]
    let outputText: String?
    let completedAt: Date
    /// When this run was a chat/code turn, the conversation it belongs to (so completion routes
    /// to the thread rather than the legacy single-result path).
    let conversationID: UUID?
    /// For raw-argument jobs, the complete stdout, uncapped, so the submitter can parse structured
    /// output (`--json`, `model list`) whole. Nil for catalog commands, which report `outputText`.
    let standardOutput: String?
    /// For raw-argument jobs, the complete stderr. Nil for catalog commands.
    let standardError: String?

    init(
        id: UUID = UUID(),
        requestID: UUID? = nil,
        templateID: CommandTemplateID,
        commandPreview: String,
        exitCode: Int32,
        outputURL: URL?,
        artifactURLs: [URL] = [],
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
        self.outputText = outputText
        self.completedAt = completedAt
        self.conversationID = conversationID
        self.standardOutput = standardOutput
        self.standardError = standardError
    }
}

/// The pre-`JobStore` name of `JobResult`, kept while views still spell it this way.
typealias MereRunRunResult = JobResult

/// One render request on the stdin of a resident `video session` process.
struct StudioVideoSessionRequest: Encodable {
    let id: String
    let prompt: String
    let output: String
    let width: Int
    let height: Int
    let numFrames: Int
    let fps: Int
    let seed: Int?
    let image: String?
    let imageStrength: Double?
    let endImage: String?
    let endImageStrength: Double?
}

struct StudioVideoSessionResponse: Decodable {
    let status: String
    let output: String?
    let error: String?
}

/// One submitted command and everything observable about it: state, status line, progress, log,
/// live stdout, artifacts and result. One `ObservableObject` per job, so a view observing a
/// background job never re-renders for the foreground one.
///
/// `JobStore` is the only writer. The mutating methods below are the store's, grouped by
/// lifecycle stage; views read the published properties.
@MainActor
final class Job: ObservableObject, Identifiable {
    let id: JobID
    let request: JobRequest
    let submittedAt: Date
    /// The explicit `--output` path the request asked for, when the template produces one.
    let expectedOutput: URL?

    @Published private(set) var state: JobState = .queued
    /// Human status line: "Queued", "Running", "Downloading model", "Generated: x.png",
    /// "Completed", "Exited 64", "Resident session ready".
    @Published private(set) var status = "Queued"
    @Published private(set) var progress: StudioRunProgress?
    @Published private(set) var log = LogRing()
    /// Capped, NUL-stripped stdout for the console's live pane.
    @Published private(set) var liveText = ""
    @Published private(set) var artifacts: [Artifact] = []
    /// Think-stripped streaming reply for conversation turns; nil for other jobs and once finished.
    @Published private(set) var conversationLiveText: String?
    @Published private(set) var result: JobResult?
    /// When the process launched; nil for jobs that never ran (queued, cancelled early, preflight).
    private(set) var startedAt: Date?

    // Store-owned process state. Not published: none of it is meaningful to a view.
    var process: MereRunRunningProcess?
    var outputWatchTask: Task<Void, Never>?
    var cancelRequested = false

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

    init(id: JobID = JobID(), request: JobRequest, submittedAt: Date = Date()) {
        self.id = id
        self.request = request
        self.submittedAt = submittedAt
        if case .templated(let template, let draft) = request.command {
            expectedOutput = ArtifactResolver.expectedOutput(template: template, draft: draft)
        } else {
            expectedOutput = nil
        }
    }

    var lane: JobLane { request.lane }
    var displayCommand: String { request.displayCommand }
    var exitCode: Int32? { state.exitCode }
    var primaryArtifactURL: URL? {
        artifacts.first { $0.role == .primary }?.url
    }
    /// Whether output-file detection applies: catalog commands other than conversation turns.
    /// Conversation replies are prose, and raw commands have no output contract to read.
    var detectsArtifacts: Bool { !request.command.isRaw && !isConversationTurn }
    private var isConversationTurn: Bool { request.conversationID != nil }
    private var isRawCommand: Bool { request.command.isRaw }
    /// The template a result is attributed to; raw commands report the catalog's raw-argument one.
    private var resultTemplateID: CommandTemplateID { request.templateID ?? .custom }

    // MARK: Lifecycle (JobStore only)

    func markRunning(status: String, at date: Date = Date()) {
        state = .running(since: date)
        startedAt = date
        self.status = status
    }

    func setStatus(_ status: String) {
        self.status = status
    }

    /// Appends a controller-side note ("Termination requested.", the launched command) without
    /// progress parsing, so a note that happens to look like a progress line stays in the log.
    func note(_ text: String, stream: LogStream) {
        log.append(text, stream: stream)
    }

    func setPrimaryArtifact(_ url: URL?) {
        let sidecars = artifacts.filter { $0.role == .sidecar }
        artifacts = (url.map { [Artifact(url: $0, role: .primary)] } ?? []) + sidecars
    }

    /// Consumes one chunk of process output: buffers it, parses progress, updates the live pane
    /// and interactive-session status, and appends the remaining lines to the log.
    func consume(_ text: String, stream: LogStream, resolver: ArtifactResolver) {
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
            log.append(LogLine(stream: stream, text: line))
        }
    }

    /// Re-runs output detection against what has been written so far. Returns true when a new
    /// primary artifact appeared, so the caller can publish it while the process is still alive.
    @discardableResult
    func refreshPrimaryArtifact(resolver: ArtifactResolver) -> Bool {
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
    func finish(exitCode: Int32, at date: Date = Date(), resolver: ArtifactResolver) -> JobResult {
        progress = nil

        // Conversation replies are prose, not artifacts — never run output-file detection on them
        // (a path-like substring in a reply must not become a bogus artifact or status). Raw
        // commands have no output contract to read either.
        let detectedOutput = detectsArtifacts
            ? resolver.primaryOutput(expected: expectedOutput, stdout: stdoutBuffer)
            : nil
        let reportedOutputs = detectsArtifacts ? resolver.reportedOutputs(stdout: stdoutBuffer) : []
        let artifactURLs: [URL]
        if case .templated(let template, let draft) = request.command {
            artifactURLs = StudioArtifactDiscovery.urls(
                templateID: template.id,
                draft: draft,
                primaryOutput: detectedOutput,
                reportedOutputs: reportedOutputs
            )
        } else {
            artifactURLs = []
        }
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
            outputText = Self.capturedResultText(stdout: stdoutBuffer, stderr: stderrBuffer, exitCode: exitCode)
        }

        let primary = detectedOutput ?? primaryArtifactURL
        artifacts = (primary.map { [Artifact(url: $0, role: .primary)] } ?? [])
            + artifactURLs.filter { $0 != primary }.map { Artifact(url: $0, role: .sidecar) }

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

    func failPreflight(_ failure: JobPreflightFailure, at date: Date = Date()) -> JobResult {
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

    func cancelBeforeStart(at date: Date = Date()) -> JobResult {
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
