import AppKit
import Quartz
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

package enum StudioSpecialistFiles {
    @MainActor
    static func chooseFile(
        title: String,
        allowedContentTypes: [UTType] = [],
        allowsMultipleSelection: Bool = false
    ) -> [URL] {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = allowsMultipleSelection
        if !allowedContentTypes.isEmpty {
            panel.allowedContentTypes = allowedContentTypes
        }
        return panel.runModal() == .OK ? panel.urls : []
    }

    @MainActor
    static func chooseDirectory(title: String) -> URL? {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        return panel.runModal() == .OK ? panel.url : nil
    }

    @MainActor
    package static func saveFile(
        title: String,
        suggestedName: String,
        allowedContentTypes: [UTType] = []
    ) -> URL? {
        let panel = NSSavePanel()
        panel.title = title
        panel.nameFieldStringValue = suggestedName
        if !allowedContentTypes.isEmpty {
            panel.allowedContentTypes = allowedContentTypes
        }
        return panel.runModal() == .OK ? panel.url : nil
    }

    /// A fresh directory for a specialist run, in `domain`'s folder wherever Settings ▸ General
    /// says generations go, stamped with the display clock so the snapshot boards render a
    /// stable path.
    @MainActor
    static func outputDirectory(domain: StudioDomain, name: String) -> URL {
        StudioOutputLocation.specialistDirectory(domain: domain, name: name, now: StudioDisplayClock.now)
    }

    /// One output file for a specialist run, filed the same way as `outputDirectory`.
    @MainActor
    static func outputFile(domain: StudioDomain, name: String, fileExtension: String) -> URL {
        StudioOutputLocation.specialistFile(
            domain: domain,
            name: name,
            fileExtension: fileExtension,
            now: StudioDisplayClock.now
        )
    }
}

struct StudioPathField: View {
    let label: String
    let placeholder: String
    @Binding var path: String
    var picksDirectory = false
    var allowsMultipleSelection = false
    var allowedContentTypes: [UTType] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            HStack(spacing: 8) {
                TextField(placeholder, text: $path)
                    .mereField()
                Button("Choose…") {
                    if picksDirectory {
                        if let url = StudioSpecialistFiles.chooseDirectory(title: label) {
                            path = url.path
                        }
                    } else {
                        let urls = StudioSpecialistFiles.chooseFile(
                            title: label,
                            allowedContentTypes: allowedContentTypes,
                            allowsMultipleSelection: allowsMultipleSelection
                        )
                        if !urls.isEmpty {
                            path = urls.map(\.path).joined(separator: "\n")
                        }
                    }
                }
                .buttonStyle(.mereSecondary)
            }
        }
    }
}

struct StudioEmbeddedQuickLookPreview: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> QLPreviewView {
        let view = QLPreviewView(frame: .zero, style: .normal)!
        view.autostarts = true
        view.previewItem = url as NSURL
        return view
    }

    func updateNSView(_ view: QLPreviewView, context: Context) {
        view.previewItem = url as NSURL
    }
}

struct StudioSpecialistResultView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @Environment(\.studioReferenceDate) private var referenceDate

    let requestID: UUID?
    var preferredKinds: [StudioOutputFileKind] = [.video, .image, .model3D, .audio, .text]
    @State private var selection: URL?

    private var item: StudioLibraryItem? {
        guard let requestID else { return nil }
        return library.items.first { $0.id == requestID }
    }

    private var artifacts: [URL] {
        item?.allArtifactURLs.filter {
            (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        } ?? []
    }

    private var activeURL: URL? {
        if let selection, artifacts.contains(selection) { return selection }
        for kind in preferredKinds {
            if let match = artifacts.first(where: { StudioOutputFileKind.classify($0) == kind }) {
                return match
            }
        }
        return artifacts.first
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let item {
                HStack {
                    Text(item.status.rawValue.capitalized)
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(statusColor(item.status))
                    // A finished run says when it ran, so a result kept from an earlier session
                    // (a failure days old, most tellingly) is never mistaken for a fresh one.
                    if item.status != .running, item.status != .queued {
                        Text("· \(StudioFeedTime.label(for: item.updatedAt, now: referenceDate ?? Date()))")
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .accessibilityLabel("Ran \(StudioFeedTime.label(for: item.updatedAt, now: referenceDate ?? Date()))")
                    }
                    if let progress = requestID.flatMap({ controller.progressByRequestID[$0] }) {
                        Text(progress.label)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textSecondary)
                        if let fraction = progress.fractionCompleted {
                            ProgressView(value: fraction)
                                .frame(maxWidth: 180)
                        } else {
                            ProgressView()
                                .controlSize(.small)
                        }
                        if let detail = progress.detail {
                            Text(detail)
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                                .lineLimit(1)
                        }
                    }
                    Spacer()
                    if let activeURL {
                        Button {
                            QuickLookCoordinator.shared.preview(activeURL)
                        } label: {
                            Label("Quick Look", systemImage: "eye")
                        }
                        .buttonStyle(.mereSecondary)
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([activeURL])
                        } label: {
                            Label("Reveal", systemImage: "folder")
                        }
                        .buttonStyle(.mereSecondary)
                    }
                }

                preview
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(MereRunTheme.surface.opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg))

                if artifacts.count > 1 {
                    // Which file the preview shows: a segmented row, the way every other
                    // either-or choice in Studio is drawn.
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 2) {
                            ForEach(artifacts, id: \.self) { url in
                                MereSegment(title: url.lastPathComponent, isSelected: activeURL == url) {
                                    selection = url
                                }
                                .help(url.path)
                            }
                        }
                        .padding(2)
                        .background {
                            RoundedRectangle(cornerRadius: 7)
                                .fill(MereRunTheme.surfaceRaised)
                        }
                    }
                    .accessibilityElement(children: .contain)
                    .accessibilityLabel("Result files")
                }
            } else {
                ContentUnavailableView(
                    "No specialist run yet",
                    systemImage: "sparkles.rectangle.stack",
                    description: Text("Configure the workflow and start a run. Progress and artifacts stay in the Library.")
                )
            }
        }
    }

    @ViewBuilder
    private var preview: some View {
        if let item, item.status == .failed || item.status == .interrupted {
            StudioSpecialistFailureView(item: item, models: controller.modelStore)
        } else if let url = activeURL {
            switch StudioOutputFileKind.classify(url) {
            case .image:
                StudioAsyncImagePreview(
                    url: url,
                    maxPixelSize: 2_000,
                    contentMode: .fit,
                    fallbackSystemImage: "photo"
                )
                .padding(10)
            case .audio:
                StudioAudioPlayerView(url: url)
            case .video:
                StudioVideoPlayerView(url: url)
            case .text:
                StudioTextFilePreview(url: url)
            case .model3D:
                StudioEmbeddedQuickLookPreview(url: url)
            case .other:
                filePlaceholder(url)
            }
        } else if let text = item?.outputText, !text.isBlank {
            ScrollView {
                Text(text)
                    .font(MereRunTheme.monoFont)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
        } else {
            ContentUnavailableView(
                "Working",
                systemImage: "hourglass",
                description: Text("The first artifact will appear here.")
            )
        }
    }

    private func filePlaceholder(_ url: URL) -> some View {
        VStack(spacing: 10) {
            Image(systemName: "doc")
                .font(.system(size: 42, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
            Text(url.lastPathComponent)
                .font(MereRunTheme.sectionFont)
            Text(url.path)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func statusColor(_ status: StudioLibraryStatus) -> Color {
        switch status {
        case .queued: MereRunTheme.textMuted
        case .running: MereRunTheme.yellow
        case .completed: MereRunTheme.green
        case .failed: MereRunTheme.red
        case .cancelled, .interrupted: MereRunTheme.textSecondary
        }
    }
}

enum StudioSpecialistRunner {
    @MainActor
    static func submit(
        templateID: CommandTemplateID,
        mode: StudioMode,
        draft: CommandDraft,
        controller: MereRunController,
        library: StudioLibraryStore
    ) -> UUID? {
        guard let template = CommandCatalog.template(id: templateID) else { return nil }
        let base = StudioRunRequest(
            mode: mode,
            templateID: templateID,
            template: template,
            draft: draft
        )
        // The same destination preparation a prompt task gets: the folder is created, or the run
        // moves to App Outputs and the shell says why.
        let prepared = StudioOutputLocation.preparing(controller.taskSessions.resolving(base))
        if let reason = prepared.fallbackReason { controller.noteOutputFallback(reason) }
        let request = prepared.request
        let preview = controller.commandPreview(arguments: request.execution?.arguments ?? template.arguments(from: request.draft), masksSecrets: true)
        let status: StudioLibraryStatus = controller.isRunning || controller.queuedRunCount > 0
            ? .queued
            : .running
        library.start(request: request, commandPreview: preview, status: status)
        _ = controller.run(studio: request)
        return request.id
    }
}

/// What a specialist page shows when its run failed: why, in one line; the log behind it; and,
/// when the run's model is not on this Mac, the way to get it. Specialist pages have no Library
/// column, so this is where the diagnosis has to be.
struct StudioSpecialistFailureView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var navigation: NavigationModel
    @ObservedObject private var models: StudioModelStore
    let item: StudioLibraryItem
    @State private var showLog = false
    @State private var pullProblem: String?

    init(item: StudioLibraryItem, models: StudioModelStore) {
        self.item = item
        _models = ObservedObject(wrappedValue: models)
    }

    private var job: Job? { controller.jobs.job(requestID: item.id) }

    private var logLines: [String] {
        if let job, !job.log.isEmpty { return job.log.lines.map(\.text) }
        return (item.outputText ?? "").components(separatedBy: .newlines).filter { !$0.isBlank }
    }

    private var summary: String {
        if item.status == .interrupted { return "Interrupted when Studio closed. Run it again to start over." }
        return StudioFailureSummary.summary(
            outputText: item.outputText,
            logLines: job?.log.lines.map(\.text) ?? [],
            exitCode: job?.exitCode ?? item.exitCode
        )
    }

    /// The model the run used: the one it was given, or its command's default.
    private var modelID: String? {
        let chosen = item.commandDraft?.model.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !chosen.isEmpty { return chosen }
        return item.templateID.flatMap { CommandCatalog.template(id: $0)?.defaultModel }
    }

    /// A managed model id rather than a path to a converted model.
    private var usesManagedModel: Bool {
        guard let modelID else { return false }
        return !modelID.contains("/") && !modelID.hasPrefix("~") && !modelID.hasPrefix(".")
    }

    /// The inventory row of a managed model that is not installed — the likeliest reason for the
    /// failure, and one Studio can fix. Local paths and installed models return nil.
    private var missingModel: StudioModelInventoryRow? {
        guard let modelID else { return nil }
        return models.rows.first { $0.id == modelID && !$0.isInstalled }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(MereRunTheme.red)
                    .padding(.top, 1)
                Text(summary)
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            if let missing = missingModel {
                missingModelRow(missing)
            }
            if let pullProblem {
                Text(pullProblem)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !logLines.isEmpty {
                Button {
                    withAnimation(MereRunTheme.Motion.quick) { showLog.toggle() }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: showLog ? "chevron.down" : "chevron.right")
                            .font(.system(size: 9, weight: .semibold))
                        Text(showLog ? "Hide log" : "Show log")
                            .font(.caption.weight(.medium))
                    }
                    .foregroundStyle(MereRunTheme.textMuted)
                }
                .buttonStyle(.plain)
                if showLog {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 3) {
                            ForEach(Array(logLines.suffix(120).enumerated()), id: \.offset) { _, line in
                                Text(line)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(MereRunTheme.textMuted)
                                    .textSelection(.enabled)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 260)
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        // The inventory says whether a managed model is missing; read it once, not per failure.
        .task(id: item.id) {
            guard usesManagedModel, !models.hasInventory, !models.isRefreshing, models.error == nil else { return }
            await models.refresh()
        }
    }

    @ViewBuilder
    private func missingModelRow(_ row: StudioModelInventoryRow) -> some View {
        let name = StudioModelNaming.displayName(row)
        HStack(spacing: 10) {
            if let pull = models.download(modelID: row.id) {
                ProgressView()
                    .controlSize(.small)
                Text(pull.progress?.label ?? "Getting \(name)…")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
            } else {
                Text("\(name) isn't on this Mac yet.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                Spacer(minLength: 8)
                if row.usageTerms == nil {
                    Button {
                        getModel(row.id)
                    } label: {
                        Label("Get the model", systemImage: "arrow.down.circle.fill")
                    }
                    .buttonStyle(.merePrimary)
                } else {
                    // Models with usage terms are accepted on the Models page, where the terms are.
                    Button("Open in Models") { navigation.open(task: .modelsInstalled) }
                        .buttonStyle(.mereSecondary)
                }
            }
        }
        .padding(10)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                .fill(MereRunTheme.surfaceRaised.opacity(0.6))
        }
    }

    private func getModel(_ modelID: String) {
        guard let template = CommandCatalog.template(id: .modelPull) else { return }
        var draft = template.defaultDraft()
        draft.model = modelID
        let started = models.startPull(StudioRunRequest(mode: item.mode, templateID: .modelPull, template: template, draft: draft))
        // A refused submission explains itself in the controller's status.
        let reason = controller.status
        pullProblem = started ? nil : (reason.isBlank || reason == "Idle" ? "Studio could not start the download." : reason)
    }
}
