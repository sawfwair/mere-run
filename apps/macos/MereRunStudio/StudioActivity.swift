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
    /// The lanes whose jobs are the user's work. `.probe` is deliberately absent.
    static let lanes: [JobLane] = [.inference, .utility]

    /// Running jobs first (lane order, then start order), then the queue in FIFO order — the order
    /// the work will actually finish in.
    @MainActor
    static func rows(in store: JobStore) -> [StudioActivityRow] {
        let running = lanes.flatMap { store.running(in: $0) }
        let queued = lanes.flatMap { store.queued(in: $0) }
        return running.map {
            StudioActivityRow(id: $0.id, title: title(for: $0), isRunning: true, isNextInQueue: false)
        } + queued.enumerated().map { index, job in
            StudioActivityRow(id: job.id, title: title(for: job), isRunning: false, isNextInQueue: index == 0)
        }
    }

    /// "3 jobs · 1 queued" beside the popover's title.
    static func summary(_ rows: [StudioActivityRow]) -> String {
        let jobs = rows.count == 1 ? "1 job" : "\(rows.count) jobs"
        let queued = rows.filter { !$0.isRunning }.count
        return queued == 0 ? jobs : "\(jobs) · \(queued) queued"
    }

    /// The domain and task a job belongs to, so a row names the work rather than the command.
    @MainActor
    static func title(for job: Job) -> String {
        "\(StudioDomain(templateID: job.request.template.id).title) · \(task(for: job))"
    }

    /// The line under the title: step progress and elapsed time for a run, transferred bytes and
    /// time left for a pull, queue position for a job that has not started.
    @MainActor
    static func detail(for job: Job, elapsed: TimeInterval?, isNextInQueue: Bool) -> String {
        guard !job.state.isQueued else {
            return isNextInQueue ? "Queued · next" : "Queued"
        }
        if job.request.template.id == .modelPull, let download = downloadDetail(job.progress) {
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
    private static func task(for job: Job) -> String {
        let template = job.request.template
        if template.id == .modelPull {
            let model = job.request.draft.model.trimmingCharacters(in: .whitespacesAndNewlines)
            // The same friendly name the composer's model chip shows, so one model reads the same
            // way wherever it appears.
            return model.isEmpty ? "Pull model" : "Pull \(StudioModelNaming.displayName(model))"
        }
        // A prompt task names itself the way the task control does ("Generate", "Transcribe");
        // everything else falls back to the template's own title.
        if let task = StudioTask.allCases.first(where: { $0.mode?.defaultTemplateID == template.id }) {
            return task.title
        }
        return template.title
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
        guard let eta = detail.range(of: #"ETA\s+\d+[hms]"#, options: .regularExpression) else { return nil }
        let value = detail[eta].dropFirst(3).trimmingCharacters(in: .whitespaces)
        let amount = value.prefix { $0.isNumber }
        guard !amount.isEmpty, let unit = value.dropFirst(amount.count).first else { return nil }
        switch unit {
        case "h": return "\(amount) hr left"
        case "m": return "\(amount) min left"
        default: return "\(amount) sec left"
        }
    }
}

/// The Activity popover: what this Mac is working on right now, one row per running or queued job
/// with the control to stop it, over the app↔CLI version handshake and a way into the Server page.
/// With nothing running it shows the machine's own state instead, in the same shape.
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

    /// Bumped whenever a job starts or finishes: lane membership is not itself published, so the
    /// row list is re-derived from the store's own event stream.
    @State private var generation = 0

    static let width: CGFloat = 340
    static let cornerRadius: CGFloat = MereRunTheme.Radius.popover

    var body: some View {
        // Reading `generation` here is what ties the row list to the store's start/finish events.
        _ = generation
        let rows = StudioActivity.rows(in: jobs)
        return VStack(alignment: .leading, spacing: 0) {
            header(rows)
            if rows.isEmpty {
                machineDetails
            } else {
                ForEach(rows) { row in
                    if let job = jobs.job(row.id) {
                        StudioActivityJobRow(job: job, row: row) { jobs.cancel(row.id) }
                    }
                }
            }
            Divider()
                .overlay(MereRunTheme.border.opacity(0.4))
                .padding(.top, 4)
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

    private func header(_ rows: [StudioActivityRow]) -> some View {
        HStack(spacing: 8) {
            Text("Activity")
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.textPrimary)
            Spacer(minLength: 12)
            Text(rows.isEmpty ? "Nothing running" : StudioActivity.summary(rows))
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
        }
        .padding(.horizontal, 14)
        .padding(.top, 8)
        .padding(.bottom, 6)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }

    /// What the popover says when no job is in flight: the same three facts the machine-status
    /// details carried, drawn as Activity rows.
    private var machineDetails: some View {
        VStack(alignment: .leading, spacing: 0) {
            detailRow(dot: status.dotColor, title: "Local server", detail: status.serverDetail)
            Button(action: onOpenModels) {
                detailRow(dot: nil, title: "Models", detail: modelsDetail)
            }
            .buttonStyle(.plain)
            .help("Open Models ▸ Installed")
            detailRow(dot: nil, title: "CLI", detail: resolvedCLI.isBlank ? "Not resolved" : resolvedCLI)
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
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                Text(detail)
                    .font(.system(size: 11, weight: .medium))
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
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
            Spacer(minLength: 12)
            Button("Open Server", action: onOpenServer)
                .buttonStyle(.plain)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(MereRunTheme.accent)
                .help("Open the Server page")
        }
        .padding(.horizontal, 14)
        .padding(.top, 8)
        .padding(.bottom, 4)
    }
}

/// One job in the Activity popover. It observes its own `Job`, so a chatty run redraws this row
/// and nothing else in the shell.
private struct StudioActivityJobRow: View {
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
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let progress = job.progress {
                    StudioProgressBar(fraction: progress.fractionCompleted)
                        .frame(height: 4)
                }
                detail
                    .font(.system(size: 11, weight: .medium))
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }
            // A queued row has no progress bar to fill the width, so the stop control still needs
            // pushing to the trailing edge where every other row's sits.
            .frame(maxWidth: .infinity, alignment: .leading)
            Button(action: onCancel) {
                Image(systemName: row.isRunning ? "stop" : "xmark")
                    .font(.system(size: 12, weight: .medium))
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
                    elapsed: context.date.timeIntervalSince(startedAt),
                    isNextInQueue: row.isNextInQueue
                ))
            }
        } else {
            Text(StudioActivity.detail(for: job, elapsed: nil, isNextInQueue: row.isNextInQueue))
        }
    }
}
