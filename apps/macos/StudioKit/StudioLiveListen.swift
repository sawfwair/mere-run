import Combine
import Foundation
import MereRunContract

// What Audio ▸ Live reads and launches: the microphones `speech listen --list-devices` prints,
// the two event streams the live commands write on stdout, and the switches the page turns on
// when it starts a session. All of it is plain data, so the page's parsing is tested here rather
// than through the view.

/// One input device, as `speech listen --list-devices` prints it: `* <uid>\t<name>` for the
/// system default, two spaces for the rest.
package struct StudioListenDevice: Equatable, Identifiable {
    package let uid: String
    package let name: String
    package let isDefault: Bool

    package init(uid: String, name: String, isDefault: Bool) {
        self.uid = uid
        self.name = name
        self.isDefault = isDefault
    }

    package var id: String { uid }

    package static func parseList(_ output: String) -> [StudioListenDevice] {
        output.split(separator: "\n").compactMap { rawLine in
            let line = String(rawLine)
            guard let marker = line.first else { return nil }
            let fields = line.dropFirst()
                .trimmingCharacters(in: .whitespaces)
                .split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
            guard fields.count == 2 else { return nil }
            let uid = String(fields[0]).trimmingCharacters(in: .whitespacesAndNewlines)
            let name = String(fields[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !uid.isEmpty, !name.isEmpty else { return nil }
            return StudioListenDevice(uid: uid, name: name, isDefault: marker == "*")
        }
    }
}

/// The transcript `speech listen --jsonl` streams: partials replace one another per utterance
/// until a commit closes it. Chunks arrive split anywhere, so lines are assembled first.
package struct StudioLiveTranscriptAccumulator: Equatable {
    private struct Event: Decodable {
        let protocolVersion: Int
        let type: String
        let utteranceID: String?
        let revision: Int?
        let text: String?
        let message: String?

        enum CodingKeys: String, CodingKey {
            case protocolVersion = "protocol"
            case type
            case utteranceID = "utteranceId"
            case legacyUtteranceID = "utterance_id"
            case revision
            case text
            case message
        }

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            protocolVersion = try values.decode(Int.self, forKey: .protocolVersion)
            type = try values.decode(String.self, forKey: .type)
            utteranceID = try values.decodeIfPresent(String.self, forKey: .utteranceID)
                ?? values.decodeIfPresent(String.self, forKey: .legacyUtteranceID)
            revision = try values.decodeIfPresent(Int.self, forKey: .revision)
            text = try values.decodeIfPresent(String.self, forKey: .text)
            message = try values.decodeIfPresent(String.self, forKey: .message)
        }
    }

    private var buffer = ""
    private var committedUtteranceIDs: Set<String> = []
    private var latestRevisions: [String: Int] = [:]
    private var committedSegments: [String] = []
    package private(set) var partialText = ""
    package private(set) var errorMessage: String?

    package init() {}

    package var committedText: String { committedSegments.joined(separator: "\n") }

    package var displayText: String {
        [committedText, partialText]
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    package mutating func beginSession() {
        buffer = ""
        partialText = ""
        errorMessage = nil
        latestRevisions.removeAll(keepingCapacity: true)
    }

    package mutating func clear() {
        self = StudioLiveTranscriptAccumulator()
    }

    package mutating func receive(_ chunk: String) {
        buffer += chunk
        while let newline = buffer.firstIndex(of: "\n") {
            let line = String(buffer[..<newline])
            buffer.removeSubrange(...newline)
            receiveLine(line)
        }
    }

    private mutating func receiveLine(_ line: String) {
        guard let data = line.data(using: .utf8),
              let event = try? JSONDecoder().decode(Event.self, from: data),
              event.protocolVersion == 1 else { return }

        switch event.type {
        case "partial":
            guard let id = event.utteranceID,
                  !committedUtteranceIDs.contains(id),
                  let text = event.text else { return }
            let revision = event.revision ?? 0
            guard revision >= latestRevisions[id, default: -1] else { return }
            latestRevisions[id] = revision
            partialText = text
        case "commit":
            guard let id = event.utteranceID,
                  !committedUtteranceIDs.contains(id),
                  let text = event.text?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !text.isEmpty else { return }
            committedUtteranceIDs.insert(id)
            committedSegments.append(text)
            partialText = ""
        case "error":
            errorMessage = event.message ?? "Live transcription failed."
        default:
            break
        }
    }
}

/// The speaker activity `speech diarize-live` streams: every `activity` event's segments, in
/// order, with the running speaker count and the seconds of audio heard so far.
package struct StudioLiveDiarizationAccumulator: Equatable {
    private var buffer = ""
    package private(set) var segments: [DiarizationStreamEvent.Segment] = []
    package private(set) var speakerCount = 0
    package private(set) var audioSeconds = 0.0
    package private(set) var errorMessage: String?

    package init() {}

    package var displayText: String {
        segments.map { segment in
            String(format: "%.2f–%.2f  %@", segment.startSeconds, segment.endSeconds, segment.speaker)
        }.joined(separator: "\n")
    }

    package mutating func beginSession() { self = Self() }
    package mutating func clear() { self = Self() }

    package mutating func receive(_ chunk: String) {
        buffer += chunk
        while let newline = buffer.firstIndex(of: "\n") {
            let line = String(buffer[..<newline])
            buffer.removeSubrange(...newline)
            guard let data = line.data(using: .utf8),
                  let event = try? JSONDecoder().decode(DiarizationStreamEvent.self, from: data),
                  event.schemaVersion == 1 else { continue }
            switch event.type {
            case .activity:
                segments.append(contentsOf: event.segments ?? [])
                speakerCount = event.speakerCount ?? speakerCount
                audioSeconds = event.audioSeconds ?? audioSeconds
            case .final:
                speakerCount = event.speakerCount ?? speakerCount
                audioSeconds = event.audioSeconds ?? audioSeconds
            case .error:
                errorMessage = event.message ?? "Live diarization failed."
            case .ready:
                break
            }
        }
    }
}

extension StudioTaskDraft {
    /// The draft Audio ▸ Live launches: the parked settings plus the switches a session needs on
    /// stdout — `--jsonl` for `speech listen`, whose events are what the transcript reads, and
    /// `--quiet` for both commands, so nothing but events reaches the page. Applied at Start and
    /// never parked, so the inspector does not count them as changed settings.
    package func liveListenLaunch() -> StudioTaskDraft {
        var launch = self
        launch.form["--quiet"] = .flag(true)
        if templateID == .speechListen {
            launch.form["--jsonl"] = .flag(true)
        }
        return launch
    }

    /// The flags Audio ▸ Live owns itself and keeps out of its options popover: the device chip,
    /// the two probing switches, and the two the launch sets.
    package static let liveListenOwnedFlags: Set<String> = ["--device", "--list-devices", "--stdin", "--jsonl", "--quiet"]
}

/// Audio ▸ Live's session: which job it is, every event it has streamed since Start, and the
/// microphones the CLI lists. The controller owns one, so it outlives the page — the transcript
/// keeps growing while the user is elsewhere, and coming back shows all of it. A session started
/// without the page (the Command view) is adopted when the page appears, with what the job has
/// kept of its stdout for the part streamed before then. When the session ends its text is
/// written beside the domain's other outputs and becomes the Library row's artifact, so the row
/// reads like a transcript rather than the raw event stream.
@MainActor
package final class StudioLiveListenModel: ObservableObject {
    @Published package private(set) var transcript = StudioLiveTranscriptAccumulator()
    @Published package private(set) var activity = StudioLiveDiarizationAccumulator()
    @Published package private(set) var requestID: UUID?
    @Published package private(set) var templateID: CommandTemplateID?
    @Published package private(set) var devices: [StudioListenDevice] = []
    @Published package private(set) var devicesUnavailable = false
    @Published package private(set) var now = Date()
    /// Where the last finished session's text was written, once it has been.
    @Published package private(set) var transcriptURL: URL?

    private weak var controller: MereRunController?
    private weak var library: StudioLibraryStore?
    private var subscription: AnyCancellable?
    /// How long Stop waits for the CLI to finish on SIGINT before the job is terminated.
    package var stopGrace: Duration = StudioTaskRunner.sessionStopGrace

    package init(controller: MereRunController) {
        self.controller = controller
        subscription = controller.jobs.events.sink { [weak self] event in
            self?.handle(event)
        }
    }

    package var job: Job? {
        guard let requestID else { return nil }
        return controller?.jobs.job(requestID: requestID)
    }

    package var isActive: Bool {
        job?.state.isActive ?? false
    }

    /// When the session's clock and log start counting.
    package var startedAt: Date? {
        guard let job else { return nil }
        return job.startedAt ?? job.submittedAt
    }

    /// Seconds the session has run, frozen at its end.
    package var elapsed: TimeInterval {
        guard let job, let started = job.startedAt else { return 0 }
        switch job.state {
        case .finished(_, let ended), .cancelled(_, let ended):
            return max(0, ended.timeIntervalSince(started))
        case .queued, .running, .preflightFailed:
            return max(0, now.timeIntervalSince(started))
        }
    }

    package var logLines: [LogLine] {
        job?.log.lines ?? []
    }

    package var errorMessage: String? {
        templateID == .speechDiarizeLive ? activity.errorMessage : transcript.errorMessage
    }

    /// The text a session produced so far: the committed transcript, or the speaker activity.
    package var text: String {
        templateID == .speechDiarizeLive ? activity.displayText : transcript.committedText
    }

    /// A session the page just started.
    package func begin(_ request: StudioRunRequest, library: StudioLibraryStore) {
        requestID = request.id
        templateID = request.templateID
        self.library = library
        transcriptURL = nil
        transcript.beginSession()
        activity.beginSession()
        now = Date()
    }

    /// The task's current job, when it is not already this session: one started from the
    /// Command view, or before this model existed. What the job has printed so far becomes the
    /// transcript.
    package func adoptCurrentSession(runner: StudioTaskRunner) {
        guard let job = runner.currentJob(for: .audioLive), job.request.requestID != requestID else { return }
        adopt(job, library: runner.library)
    }

    package func adopt(_ job: Job, library: StudioLibraryStore) {
        requestID = job.request.requestID
        templateID = job.request.templateID
        self.library = library
        transcriptURL = nil
        transcript = StudioLiveTranscriptAccumulator()
        activity = StudioLiveDiarizationAccumulator()
        receive(job.liveText)
        now = Date()
    }

    /// Stops the session the way Ctrl-C does — SIGINT, which `speech listen` traps to flush its
    /// last events and exit cleanly — and terminates it if it has not ended after `stopGrace`.
    package func stop() {
        guard let controller, let job, job.state.isActive else { return }
        controller.jobs.interruptThenCancel(job.id, after: stopGrace)
    }

    package func clear() {
        transcript.clear()
        activity.clear()
    }

    package func tick() {
        now = Date()
    }

    package func refreshDevices() async {
        guard let controller else { return }
        let result = await controller.utilityCommandResult(args: ["speech", "listen", "--list-devices"])
        devicesUnavailable = result.exitCode != 0
        guard result.exitCode == 0 else { return }
        devices = StudioListenDevice.parseList(result.stdout)
    }

    private func handle(_ event: JobStore.Event) {
        switch event {
        case .output(let job, let stream, let text):
            guard stream == .stdout, let requestID, job.request.requestID == requestID else { return }
            receive(text)
        case .started(let job), .changed(let job):
            if let requestID, job.request.requestID == requestID { objectWillChange.send() }
        case .finished(let job, _):
            guard let requestID, job.request.requestID == requestID else { return }
            objectWillChange.send()
            // After the Library has recorded the run (its own sink runs synchronously first).
            Task { @MainActor [weak self] in self?.recordTranscript(of: job) }
        }
    }

    private func receive(_ text: String) {
        guard !text.isEmpty else { return }
        if templateID == .speechDiarizeLive {
            activity.receive(text)
        } else {
            transcript.receive(text)
        }
    }

    /// Writes the finished session's text as `live-transcript-<stamp>.txt` (or `live-speakers-`)
    /// in the Audio folder and makes it the row's output and artifact, in place of the capped
    /// event stream the job printed. A session that heard nothing leaves the row as it is.
    private func recordTranscript(of job: Job) {
        guard let library, let requestID = job.request.requestID,
              let item = library.items.first(where: { $0.id == requestID }) else { return }
        let text = self.text
        guard !text.isEmpty else { return }
        let url = StudioOutputLocation.specialistFile(
            domain: .audio,
            name: templateID == .speechDiarizeLive ? "live-speakers" : "live-transcript",
            fileExtension: "txt"
        )
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try text.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            return
        }
        transcriptURL = url
        // A session stopped by termination keeps its cancelled status; the text is still its result.
        let status = item.status
        library.complete(
            id: requestID, exitCode: item.exitCode ?? job.exitCode ?? 0, outputURL: url, outputText: text,
            commandPreview: item.commandPreview, artifactURLs: [url]
        )
        if status == .cancelled { library.setStatus(status, id: requestID) }
    }
}
