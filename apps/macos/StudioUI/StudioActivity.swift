import StudioKit
import SwiftUI

/// One line of the Activity popover: a job that is running or waiting in one of the two work
/// lanes. Probe jobs never appear — a readiness check is not work the user started.
struct StudioActivityRow: Identifiable, Equatable {
    let id: JobID
    /// "Image · Generate", "Models · Pull image-zimage-nano": the job's domain and its task.
    let title: String
    let isRunning: Bool
    /// The first job waiting for a lane slot, which reads "Queued · next".
    let isNextInQueue: Bool
}

/// How the Activity popover reads a `JobStore`: which jobs it lists, in what order, and the copy
/// each row and the header carry. Pure functions over the store and one job, so every string the
/// popover shows is testable without a view.
enum StudioActivity {
    /// System Settings ▸ Privacy & Security ▸ Files & Folders, where removable- and
    /// network-volume access for MereRun is granted.
    static let filesAndFoldersSettingsURL = URL(
        string: "x-apple.systempreferences:com.apple.preference.security?Privacy_FilesAndFolders"
    )!

    /// The lanes whose jobs are the user's work. `.probe` is deliberately absent.
    static let lanes: [JobLane] = [.inference, .utility]

    /// Running jobs first (lane order, then start order), then the queue in FIFO order — the order
    /// the work will actually finish in.
    @MainActor
    static func rows(in store: JobStore, titles: StudioModelTitles) -> [StudioActivityRow] {
        let running = lanes.flatMap { store.running(in: $0) }
        let queued = lanes.flatMap { store.queued(in: $0) }
        return running.map {
            StudioActivityRow(id: $0.id, title: title(for: $0, titles: titles), isRunning: true, isNextInQueue: false)
        } + queued.enumerated().map { index, job in
            StudioActivityRow(id: job.id, title: title(for: job, titles: titles), isRunning: false, isNextInQueue: index == 0)
        }
    }

    /// "3 jobs · 1 queued" beside the popover's title.
    static func summary(_ rows: [StudioActivityRow]) -> String {
        let jobs = rows.count == 1 ? "1 job" : "\(rows.count) jobs"
        let queued = rows.filter { !$0.isRunning }.count
        return queued == 0 ? jobs : "\(jobs) · \(queued) queued"
    }

    /// The domain and task a job belongs to, so a row names the work rather than the command.
    /// A raw utility read or write has no template, so it names its own CLI subcommand.
    @MainActor
    static func title(for job: Job, titles: StudioModelTitles) -> String {
        guard let templateID = job.request.templateID else { return rawTitle(for: job) }
        return "\(StudioDomain(templateID: templateID).title) · \(task(for: job, titles: titles))"
    }

    /// The line under the title: step progress and elapsed time for a run, transferred bytes and
    /// time left for a pull, queue position for a job that has not started.
    @MainActor
    static func detail(for job: Job, elapsed: TimeInterval?, isNextInQueue: Bool) -> String {
        guard !job.state.isQueued else {
            return isNextInQueue ? "Queued · next" : "Queued"
        }
        if job.request.templateID == .modelPull, let download = downloadDetail(job.progress) {
            return download
        }
        let status = StudioRunningStatus.text(progress: job.progress, fallback: job.status)
        guard let elapsed else { return status }
        return "\(status) · \(StudioTimeFormat.string(elapsed))"
    }

    /// "mere.run 0.50.0 · CLI matched" in the popover's footer: the app's version and whether the
    /// CLI beside it reports the same one (a mismatch means the CLI came from PATH, not the bundle).
    static func versionLine(appVersion: String, cliVersion: String?) -> String {
        guard let cliVersion, !cliVersion.isBlank else {
            return "mere.run \(appVersion) · CLI not resolved"
        }
        return cliVersion == appVersion
            ? "mere.run \(appVersion) · CLI matched"
            : "mere.run \(appVersion) · CLI \(cliVersion)"
    }

    /// Rewrites the CLI's download progress detail ("1.2 GB / 4.8 GB 9.7 MB/s ETA 3m 20s") as the
    /// popover's shorter "1.2 of 4.8 GB · 3 min left". Returns nil when the line carries neither a
    /// byte count nor an ETA, so the caller falls back to the job's own status.
    static func downloadDetail(_ progress: StudioRunProgress?) -> String? {
        guard let detail = progress?.detail else { return nil }
        var parts: [String] = []
        if let bytes = transferredBytes(in: detail) { parts.append(bytes) }
        if let eta = timeLeft(in: detail) { parts.append(eta) }
        if parts.isEmpty, detail.localizedCaseInsensitiveContains("extracting") {
            parts.append("Extracting…")
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    // MARK: - Private

    @MainActor
    private static func task(for job: Job, titles: StudioModelTitles) -> String {
        guard let template = job.request.template else { return rawTitle(for: job) }
        if template.id == .modelPull {
            let model = (job.request.draft?.model ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            // The same friendly name the composer's model chip shows, so one model reads the same
            // way wherever it appears.
            return model.isEmpty ? "Pull model" : "Pull \(StudioModelNaming.displayName(model, titles: titles))"
        }
        // A prompt task names itself the way the task control does ("Generate", "Transcribe");
        // everything else falls back to the template's own title.
        if let task = StudioTask.allCases.first(where: { $0.mode?.defaultTemplateID == template.id }) {
            return task.title
        }
        return template.title
    }

    /// A raw-argument job (`model list`, `config set`) has no template to name it, so it is named
    /// by what Studio is doing with it — "System · Checking models" — never by the argv.
    @MainActor
    private static func rawTitle(for job: Job) -> String {
        "System · \(taskName(for: job.request.rawArguments ?? []))"
    }

    /// The plain name of the hand-built CLI read or write behind `arguments`: what the Activity
    /// row and the menu bar show while it runs.
    static func taskName(for arguments: [String]) -> String {
        let words = arguments.prefix { !$0.hasPrefix("-") }.map { $0.lowercased() }
        switch (words.first, words.dropFirst().first, words.dropFirst(2).first) {
        case ("model", "list", _), ("model", "capabilities", _): return "Checking models"
        case ("model", "info", _): return "Reading model details"
        case ("model", "storage", _): return "Measuring model storage"
        case ("model", "gc", _): return "Cleaning up model storage"
        case ("model", "runtime", "get"): return "Reading runtime settings"
        case ("model", "runtime", _): return "Saving runtime settings"
        case ("model", "remove", _): return "Removing a model"
        case ("model", "pull", _): return "Getting a model"
        case ("model", "location", _): return "Updating model locations"
        case ("model", _, _): return "Checking models"
        case ("adapter", _, _): return "Checking adapters"
        case ("config", "path", _), ("config", "list", _), ("config", "get", _): return "Reading settings"
        case ("config", _, _): return "Saving settings"
        case ("guide", _, _): return "Loading the guide"
        case ("gate", _, _): return "Checking quality gates"
        case ("executor", _, _): return "Checking executors"
        case ("run", "inspect", _): return "Inspecting a run"
        case ("run", "list", _): return "Finding runs"
        case ("run", "fetch", _): return "Fetching run outputs"
        case ("run", "cancel", _): return "Cancelling a run"
        case ("run", "retry", _): return "Retrying a run"
        case ("run", _, _): return "Checking runs"
        case ("agent", _, _): return "Checking agents"
        case ("speech", "profile", _): return "Loading voices"
        case ("speech", "listen", _) where arguments.contains("--list-devices"): return "Finding microphones"
        case ("music", "realtime", _) where arguments.contains("--list-midi-inputs"): return "Finding MIDI inputs"
        case ("plugin", _, _): return "Running a plugin"
        case (nil, _, _) where arguments.contains("--version"): return "Checking the CLI version"
        case (nil, _, _): return "Working"
        case (let first?, _, _): return "Running \(first)"
        }
    }

    /// "1.2 GB / 4.8 GB" → "1.2 of 4.8 GB" (one unit when both sides agree, both when they differ).
    private static func transferredBytes(in detail: String) -> String? {
        let pattern = #"(\d+(?:[.,]\d+)?)\s*(B|KB|MB|GB|TB)\s*/\s*(\d+(?:[.,]\d+)?)\s*(B|KB|MB|GB|TB)"#
        guard let match = detail.range(of: pattern, options: .regularExpression) else { return nil }
        let fields = detail[match].split(separator: "/").map { $0.trimmingCharacters(in: .whitespaces) }
        guard fields.count == 2 else { return nil }
        let completed = fields[0].split(separator: " ").map(String.init)
        let total = fields[1]
        guard completed.count == 2 else { return nil }
        return completed[1] == total.split(separator: " ").last.map(String.init)
            ? "\(completed[0]) of \(total)"
            : "\(fields[0]) of \(total)"
    }

    /// "ETA 3m 20s" → "3 min left"; the largest unit is enough at a glance.
    private static func timeLeft(in detail: String) -> String? {
        StudioRunETA.downloadSecondsLeft(detail).map { "\(roughDuration($0)) left" }
    }
}


// MARK: - Run queue copy

/// Every string the run queue shows, as pure functions of a job and where it stands.
extension StudioActivity {
    /// The heading over one lane's jobs.
    static func laneTitle(_ lane: JobLane) -> String {
        switch lane {
        case .inference: return "Model runs"
        case .utility: return "Background tasks"
        case .service: return "Servers"
        case .probe: return "Checks"
        }
    }

    /// "3 running · 2 queued" beside the popover's title; "Nothing running" when idle.
    static func queueSummary(_ sections: [StudioRunQueueSection]) -> String {
        let queued = sections.reduce(0) { $0 + $1.queuedCount }
        let running = sections.reduce(0) { $0 + $1.entries.count } - queued
        switch (running, queued) {
        case (0, 0): return "Nothing running"
        case (_, 0): return "\(running) running"
        case (0, _): return "\(queued) queued"
        default: return "\(running) running · \(queued) queued"
        }
    }

    /// Where the job stands: its stage while it runs, why it is waiting while it waits.
    @MainActor
    static func statusText(for job: Job, status: StudioRunQueueStatus) -> String {
        switch status {
        case .running:
            // A pull's progress line is the CLI's raw transfer readout; its time line says it
            // better, so the status stays "Downloading model".
            if job.request.templateID == .modelPull { return job.status }
            return StudioRunningStatus.text(progress: job.progress, fallback: job.status)
        case .waitingForMemory:
            return "Waiting for memory"
        case .queued(let position):
            let place = position == 0 ? "next" : "\(ordinal(position + 1)) in line"
            return job.lane == .inference ? "Waiting for a GPU slot · \(place)" : "Queued · \(place)"
        }
    }

    /// The line under a running job's status: how long it has run and, when it can be measured,
    /// how long it has left. A pull's line is the CLI's own byte count and estimate.
    @MainActor
    static func timeLine(for job: Job, elapsed: TimeInterval, eta: StudioRunETA?) -> String {
        if job.request.templateID == .modelPull, let download = downloadDetail(job.progress) {
            return download
        }
        let clock = StudioTimeFormat.string(elapsed)
        return eta.map { "\(clock) · \(etaText($0))" } ?? clock
    }

    /// "3 min left" from the CLI, "40 sec left in denoising" from this run's steps, "about 2 min
    /// left" from recent runs of the same template and model.
    static func etaText(_ eta: StudioRunETA) -> String {
        let amount = roughDuration(eta.remaining)
        switch eta.source {
        case .download: return "\(amount) left"
        case .stage(let label): return "\(amount) left in \(label.lowercased())"
        case .history: return "about \(amount) left"
        }
    }

    /// How a finished run ended: "Completed in 1:32", "Failed · exit 1", "Cancelled", or why it
    /// never started.
    @MainActor
    static func outcomeText(for job: Job) -> String {
        switch job.state {
        case .finished(0, let endedAt):
            return job.startedAt.map { "Completed in \(StudioTimeFormat.string(endedAt.timeIntervalSince($0)))" } ?? "Completed"
        case .finished(let exit, _):
            return "Failed · exit \(exit)"
        case .cancelled:
            return "Cancelled"
        case .preflightFailed(let failure):
            return "Didn't start · \(failure.message)"
        case .queued, .running:
            return job.status
        }
    }

    /// The friendly name of the model a job runs, or nil for work that names none. A pull already
    /// names its model in its title.
    @MainActor
    static func modelName(for job: Job, titles: StudioModelTitles) -> String? {
        guard job.request.templateID != .modelPull,
              let model = job.request.draft?.model.trimmingCharacters(in: .whitespacesAndNewlines),
              !model.isEmpty else { return nil }
        return StudioModelNaming.displayName(model, titles: titles)
    }

    /// The glyph of the domain a job belongs to; a raw CLI read has none, so it gets a gear.
    @MainActor
    static func systemImage(for job: Job) -> String {
        job.request.templateID.map { StudioDomain(templateID: $0).systemImage } ?? "gearshape"
    }

    /// "45 sec", "3 min", "1 hr": the largest unit, which is all a glance needs.
    static func roughDuration(_ seconds: TimeInterval) -> String {
        let total = max(Int(seconds.rounded()), 1)
        if total >= 3_600 { return "\(total / 3_600) hr" }
        if total >= 60 { return "\(total / 60) min" }
        return "\(total) sec"
    }

    /// "2nd", "3rd", "11th".
    static func ordinal(_ number: Int) -> String {
        let suffix: String
        switch (number % 10, number % 100) {
        case (_, 11...13): suffix = "th"
        case (1, _): suffix = "st"
        case (2, _): suffix = "nd"
        case (3, _): suffix = "rd"
        default: suffix = "th"
        }
        return "\(number)\(suffix)"
    }
}

/// The Activity popover, which is the run queue: everything running and waiting across every
/// page, grouped by lane, with Stop, Open and — for a waiting job — Move up and Move down on each
/// row, and the last few finished runs below with a way into the Library. With nothing in flight
/// it shows the machine's own state above those, in the same shape.
///
/// It observes the `JobStore` directly — the lane contents for which rows exist, each `Job` for its
/// own progress — rather than any mirrored copy of that state.
struct StudioActivityPopover: View {
    @ObservedObject var jobs: JobStore
    let status: StudioMachineStatus
    let appVersion: String
    let cliVersion: String?
    let modelsRoot: String
    let resolvedCLI: String
    let onOpenServer: () -> Void
    let onOpenModels: () -> Void
    /// Opens a job's page, or a finished run's row in the Library.
    var onOpen: (Job) -> Void = { _ in }
    /// Batches with files still running or waiting, each shown once above the lanes.
    var batches: [StudioBatchProgress] = []
    var onStopBatch: (UUID) -> Void = { _ in }
    @Environment(\.studioModelTitles) private var titles

    /// Bumped on every job event: lane membership and queue order are not themselves published,
    /// so the row list is re-derived from the store's own event stream.
    @State private var generation = 0

    static let width: CGFloat = 400
    static let maxListHeight: CGFloat = 520
    static let cornerRadius: CGFloat = MereRunTheme.Radius.popover

    var body: some View {
        // Reading `generation` here is what ties the row list to the store's events.
        _ = generation
        let sections = StudioRunQueue.sections(in: jobs)
        let recent = StudioRunQueue.recentlyFinished(in: jobs)
        return VStack(alignment: .leading, spacing: 0) {
            header(sections)
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    if sections.isEmpty {
                        machineDetails
                    }
                    if !batches.isEmpty {
                        eyebrow(batches.count == 1 ? "Batch" : "Batches")
                        ForEach(batches) { batch in
                            StudioBatchQueueRow(progress: batch, onStop: { onStopBatch(batch.group) })
                        }
                    }
                    ForEach(sections) { section in
                        eyebrow(StudioActivity.laneTitle(section.lane))
                        ForEach(section.entries) { entry in
                            queueRow(entry)
                        }
                    }
                    if !recent.isEmpty {
                        eyebrow("Recently finished")
                        ForEach(recent) { job in
                            StudioRecentRunRow(
                                job: job,
                                title: StudioActivity.title(for: job, titles: titles),
                                model: StudioActivity.modelName(for: job, titles: titles),
                                onOpen: { onOpen(job) }
                            )
                        }
                    }
                }
                .padding(.bottom, 4)
            }
            .scrollBounceBehavior(.basedOnSize)
            .frame(maxHeight: Self.maxListHeight)
            .fixedSize(horizontal: false, vertical: true)
            Divider()
                .overlay(MereRunTheme.border.opacity(0.4))
            footer
        }
        .padding(.vertical, 6)
        .frame(width: Self.width)
        // The shadow belongs to the panel, not to the panel's contents: `shadow` applied to a
        // stack is inherited by every leaf inside it, which would darken a halo around each row.
        .background {
            RoundedRectangle(cornerRadius: Self.cornerRadius)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: Self.cornerRadius)
                        .strokeBorder(MereRunTheme.border, lineWidth: 1)
                }
                .mereShadow(radius: 12, y: 8)
        }
        .onReceive(jobs.events) { _ in generation &+= 1 }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Activity")
    }

    private func queueRow(_ entry: StudioRunQueueEntry) -> some View {
        let job = entry.job
        return StudioQueueRow(
            job: job,
            entry: entry,
            title: StudioActivity.title(for: job, titles: titles),
            model: StudioActivity.modelName(for: job, titles: titles),
            typicalDuration: StudioRunETA.typicalDuration(of: StudioRunETA.recentDurations(like: job, in: jobs)),
            onStop: { StudioRunQueue.stop(job, in: jobs) },
            onMove: { jobs.moveQueued(job.id, by: $0) },
            onOpen: job.request.templateID == nil ? nil : { onOpen(job) }
        )
    }

    private func header(_ sections: [StudioRunQueueSection]) -> some View {
        let queued = sections.reduce(0) { $0 + $1.queuedCount }
        return HStack(spacing: 8) {
            Text("Activity")
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.textPrimary)
                .accessibilityAddTraits(.isHeader)
            Text(StudioActivity.queueSummary(sections))
                .font(.caption.weight(.medium))
                .monospacedDigit()
                .foregroundStyle(MereRunTheme.textMuted)
            Spacer(minLength: 12)
            if queued > 0 {
                Button("Cancel all queued") { StudioRunQueue.cancelAllQueued(in: jobs) }
                    .buttonStyle(.plain)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.accent)
                    .help("Take every waiting job out of the queue; running jobs keep going")
                    .accessibilityLabel(queued == 1 ? "Cancel 1 queued job" : "Cancel \(queued) queued jobs")
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 8)
        .padding(.bottom, 2)
    }

    private func eyebrow(_ title: String) -> some View {
        MereEyebrow(title)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 14)
            .padding(.top, 10)
            .padding(.bottom, 2)
    }

    /// What the popover says when no job is in flight: the local server and the models, drawn as
    /// Activity rows. The CLI's path is a diagnostic, so it is the footer's tooltip rather than a
    /// row.
    private var machineDetails: some View {
        VStack(alignment: .leading, spacing: 0) {
            detailRow(dot: status.dotColor, title: "Local server", detail: status.serverDetail)
            Button(action: onOpenModels) {
                detailRow(dot: nil, title: "Models", detail: modelsDetail)
            }
            .buttonStyle(.plain)
            .help("Open Models ▸ Installed")
            if let notice = status.locationNotice {
                Button {
                    NSWorkspace.shared.open(StudioActivity.filesAndFoldersSettingsURL)
                } label: {
                    detailRow(dot: MereRunTheme.yellow, title: notice.title, detail: notice.detail)
                }
                .buttonStyle(.plain)
                .help("Open Privacy & Security ▸ Files & Folders")
            }
        }
    }

    private var modelsDetail: String {
        let root = modelsRoot.trimmingCharacters(in: .whitespacesAndNewlines)
        let location = root.isEmpty
            ? "default location"
            : (root as NSString).abbreviatingWithTildeInPath
        return "\(status.modelsDetail) · \(location)"
    }

    private func detailRow(dot: Color?, title: String, detail: String) -> some View {
        HStack(spacing: 10) {
            Circle()
                .fill(dot ?? .clear)
                .frame(width: 8, height: 8)
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                Text(detail)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(2)
                    .truncationMode(.middle)
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
    }

    private var footer: some View {
        HStack(spacing: 8) {
            Text(StudioActivity.versionLine(appVersion: appVersion, cliVersion: cliVersion))
                .font(.caption.weight(.medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
                .help(resolvedCLI.isBlank ? "The mere.run command line was not found" : "Command line: \(resolvedCLI)")
            Spacer(minLength: 12)
            Button("Open Server", action: onOpenServer)
                .buttonStyle(.plain)
                .font(.caption.weight(.medium))
                .foregroundStyle(MereRunTheme.accent)
                .help("Open the Server page")
        }
        .padding(.horizontal, 14)
        .padding(.top, 8)
        .padding(.bottom, 4)
    }
}

/// One running or waiting job in the run queue: its task glyph, title and model, where it stands,
/// its progress, elapsed time and honest time left, and its controls. It observes its own `Job`,
/// so a chatty run redraws this row and nothing else.
struct StudioQueueRow: View {
    @Environment(\.studioReferenceDate) private var referenceDate

    @ObservedObject var job: Job
    let entry: StudioRunQueueEntry
    let title: String
    let model: String?
    /// The median duration of recent successful runs like this one, for the history estimate.
    let typicalDuration: TimeInterval?
    let onStop: () -> Void
    let onMove: (Int) -> Void
    /// Nil for work with no page of its own (a raw CLI read).
    let onOpen: (() -> Void)?

    private var isWaiting: Bool { job.state.isQueued }
    private var status: String { StudioActivity.statusText(for: job, status: entry.status) }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            StudioQueueGlyph(systemImage: StudioActivity.systemImage(for: job), tint: tint)
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let model {
                    Text(model)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                if entry.status == .running, let progress = job.progress {
                    StudioProgressBar(fraction: progress.fractionCompleted)
                        .padding(.vertical, 2)
                }
                Text(status)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(entry.status == .running ? MereRunTheme.textMuted : MereRunTheme.yellow)
                    .lineLimit(1)
                if let startedAt = job.startedAt, job.state.isRunning {
                    TimelineView(.periodic(from: startedAt, by: 1)) { context in
                        timeLine(now: referenceDate ?? context.date, startedAt: startedAt)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            controls
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 7)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(title), \(status)")
    }

    private var tint: Color {
        entry.status == .running ? MereRunTheme.accent : MereRunTheme.yellow
    }

    private func timeLine(now: Date, startedAt: Date) -> some View {
        let elapsed = now.timeIntervalSince(startedAt)
        let eta = StudioRunETA.estimate(
            progress: job.progress,
            stage: job.progressStage,
            elapsed: elapsed,
            typicalDuration: typicalDuration,
            now: now
        )
        return Text(StudioActivity.timeLine(for: job, elapsed: elapsed, eta: eta))
            .font(.caption.weight(.medium))
            .monospacedDigit()
            .foregroundStyle(MereRunTheme.textMuted)
            .lineLimit(1)
    }

    private var controls: some View {
        HStack(spacing: 0) {
            if isWaiting {
                iconButton(
                    "chevron.up",
                    help: entry.waitsForModelDownload ? "Waits for its model download." : "Move up in the queue",
                    label: "Move \(title) up",
                    enabled: entry.canMoveUp
                ) {
                    onMove(-1)
                }
                iconButton("chevron.down", help: "Move down in the queue", label: "Move \(title) down", enabled: entry.canMoveDown) {
                    onMove(1)
                }
            }
            if let onOpen {
                iconButton("arrow.up.forward.square", help: "Open this job's page", label: "Open \(title)", enabled: true, action: onOpen)
            }
            iconButton(
                isWaiting ? "xmark" : "stop",
                help: isWaiting ? "Remove this job from the queue" : "Stop this job",
                label: isWaiting ? "Remove \(title) from the queue" : "Stop \(title)",
                enabled: true,
                action: onStop
            )
        }
    }

    private func iconButton(
        _ systemImage: String,
        help: String,
        label: String,
        enabled: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 12, weight: .medium))
                .frame(width: 26, height: 26)
        }
        .buttonStyle(.mereIcon)
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.35)
        .help(help)
        .accessibilityLabel(label)
    }
}

/// A finished run under the queue: how it ended, and the way to its row in the Library.
struct StudioRecentRunRow: View {
    let job: Job
    let title: String
    let model: String?
    let onOpen: () -> Void

    private var succeeded: Bool { job.state.exitCode == 0 }
    private var outcome: String { StudioActivity.outcomeText(for: job) }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            StudioQueueGlyph(
                systemImage: StudioActivity.systemImage(for: job),
                tint: succeeded ? MereRunTheme.green : MereRunTheme.textMuted
            )
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text([model, outcome].compactMap { $0 }.joined(separator: " · "))
                    .font(.caption.weight(.medium))
                    .monospacedDigit()
                    .foregroundStyle(succeeded ? MereRunTheme.textMuted : MereRunTheme.red)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Button("Show in Library", action: onOpen)
                .buttonStyle(.plain)
                .font(.caption.weight(.medium))
                .foregroundStyle(MereRunTheme.accent)
                .padding(.top, 2)
                .help("Open this run in the Library")
                .accessibilityLabel("Show \(title) in the Library")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(title), \(outcome)")
    }
}

/// A job's domain glyph on a soft tile, tinted by where the job stands.
private struct StudioQueueGlyph: View {
    let systemImage: String
    let tint: Color

    var body: some View {
        Image(systemName: systemImage)
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(tint)
            .frame(width: 26, height: 26)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                    .fill(MereRunTheme.surfaceRaised)
            }
            .accessibilityHidden(true)
    }
}

/// One job in the Activity popover and the menu bar panel. It observes its own `Job`, so a chatty
/// run redraws this row and nothing else in the shell.
struct StudioActivityJobRow: View {
    @Environment(\.studioReferenceDate) private var referenceDate

    @ObservedObject var job: Job
    let row: StudioActivityRow
    let onCancel: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(row.isRunning ? MereRunTheme.accent : MereRunTheme.yellow)
                .frame(width: 8, height: 8)
            VStack(alignment: .leading, spacing: 3) {
                Text(row.title)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let progress = job.progress {
                    StudioProgressBar(fraction: progress.fractionCompleted)
                        .frame(height: 4)
                }
                detail
                    .font(.caption.weight(.medium))
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }
            // A queued row has no progress bar to fill the width, so the stop control still needs
            // pushing to the trailing edge where every other row's sits.
            .frame(maxWidth: .infinity, alignment: .leading)
            Button(action: onCancel) {
                Image(systemName: row.isRunning ? "stop" : "xmark")
                    .font(.callout.weight(.medium))
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.mereIcon)
            .help(row.isRunning ? "Stop this job" : "Remove this job from the queue")
            .accessibilityLabel(row.isRunning ? "Stop \(row.title)" : "Remove \(row.title) from the queue")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(row.title)
    }

    /// A running job's elapsed time ticks; a queued one has nothing to count.
    @ViewBuilder
    private var detail: some View {
        if row.isRunning, let startedAt = job.startedAt {
            TimelineView(.periodic(from: startedAt, by: 1)) { context in
                Text(StudioActivity.detail(
                    for: job,
                    elapsed: (referenceDate ?? context.date).timeIntervalSince(startedAt),
                    isNextInQueue: row.isNextInQueue
                ))
            }
        } else {
            Text(StudioActivity.detail(for: job, elapsed: nil, isNextInQueue: row.isNextInQueue))
        }
    }
}
