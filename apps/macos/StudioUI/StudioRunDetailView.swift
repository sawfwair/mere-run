import AppKit
import StudioKit
import SwiftUI

/// One recorded run inside a Project or Manage page. Its status, artifacts, and failure
/// diagnosis come from the Library row that the shared runner wrote.
struct StudioRunDetailView: View {
    @EnvironmentObject private var controller: MereRunController
    @Environment(\.studioReferenceDate) private var referenceDate

    let item: StudioLibraryItem
    var preferredKinds: [StudioOutputFileKind] = [.video, .image, .model3D, .audio, .text]
    @State private var selection: URL?

    private var artifacts: [URL] {
        item.allArtifactURLs.filter {
            (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }
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
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Text(item.status.rawValue.capitalized)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(statusColor)
                if item.status != .running, item.status != .queued {
                    Text(StudioFeedTime.label(for: item.updatedAt, now: referenceDate ?? Date()))
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                if let progress = controller.progressByRequestID[item.id] {
                    Text(progress.label)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                }
                Spacer()
                if let activeURL {
                    Button("Quick Look") { QuickLookCoordinator.shared.preview(activeURL) }
                        .buttonStyle(.mereSecondary)
                    Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([activeURL]) }
                        .buttonStyle(.mereSecondary)
                }
            }

            content
                .frame(maxWidth: .infinity, minHeight: 180, alignment: .topLeading)

            StudioLineageLinks(item: item)

            if artifacts.count > 1 {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(artifacts, id: \.self) { url in
                            MereSegment(title: url.lastPathComponent, isSelected: activeURL == url) {
                                selection = url
                            }
                                .help(url.path)
                        }
                    }
                }
                .accessibilityLabel("Result files")
            }
        }
        .padding(MereRunTheme.Spacing.md)
        .merePanel()
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(item.status.rawValue.capitalized) run: \(item.displayTitle)")
    }

    @ViewBuilder
    private var content: some View {
        if item.status == .failed || item.status == .interrupted {
            StudioRunFailureDetail(models: controller.modelStore, item: item)
        } else if let activeURL {
            switch StudioOutputFileKind.classify(activeURL) {
            case .image:
                StudioAsyncImagePreview(
                    url: activeURL, maxPixelSize: 2_000, contentMode: .fit, fallbackSystemImage: "photo"
                )
            case .audio:
                StudioAudioPlayerView(url: activeURL)
            case .video:
                StudioVideoPlayerView(url: activeURL)
            case .text:
                StudioTextFilePreview(url: activeURL)
            case .model3D:
                StudioEmbeddedQuickLookPreview(url: activeURL)
            case .other:
                StudioResultFileRow(url: activeURL)
            }
        } else if let output = item.outputText, !output.isBlank {
            // What the CLI printed — a benchmark's table, a quality gate's report — shown as
            // printed: Markdown would read its `*`, `_`, and `|` as formatting.
            ScrollView {
                Text(output)
                    .font(MereRunTheme.monoFont)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            ContentUnavailableView(
                "Working", systemImage: "hourglass",
                description: Text("The run's output appears here.")
            )
        }
    }

    private var statusColor: Color {
        switch item.status {
        case .queued: MereRunTheme.textMuted
        case .running: MereRunTheme.yellow
        case .completed: MereRunTheme.green
        case .failed: MereRunTheme.red
        case .cancelled, .interrupted: MereRunTheme.textSecondary
        }
    }
}

/// A failed run's CLI reason, recent log, and the missing-model action when the inventory
/// identifies one. The same run remains recorded in the Library.
struct StudioRunFailureDetail: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var navigation: NavigationModel
    @ObservedObject var models: StudioModelStore
    let item: StudioLibraryItem
    @State private var showsLog = false
    @State private var pullProblem: String?

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
    private var modelID: String? {
        let chosen = item.commandDraft?.model.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !chosen.isEmpty { return chosen }
        return item.templateID.flatMap { CommandCatalog.template(id: $0)?.defaultModel }
    }
    private var usesManagedModel: Bool {
        guard let modelID else { return false }
        return !modelID.contains("/") && !modelID.hasPrefix("~") && !modelID.hasPrefix(".")
    }
    private var missingModel: StudioModelInventoryRow? {
        guard let modelID, usesManagedModel else { return nil }
        return models.rows.first { $0.id == modelID && !$0.isInstalled }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(summary, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(MereRunTheme.red)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            if let missingModel {
                HStack {
                    if let pull = models.download(modelID: missingModel.id) {
                        ProgressView().controlSize(.small)
                        Text(pull.progress?.label ?? "Getting \(StudioModelNaming.displayName(missingModel))…")
                    } else {
                        Text("\(StudioModelNaming.displayName(missingModel)) isn't on this Mac yet.")
                        Spacer()
                        if missingModel.usageTerms == nil {
                            Button("Get the model") { getModel(missingModel.id) }
                                .buttonStyle(.merePrimary)
                        } else {
                            Button("Open in Models") { navigation.open(task: .modelsInstalled) }
                                .buttonStyle(.mereSecondary)
                        }
                    }
                }
                .font(MereRunTheme.captionFont)
            }
            if let pullProblem {
                Text(pullProblem).font(MereRunTheme.captionFont).foregroundStyle(MereRunTheme.red)
            }
            if !logLines.isEmpty {
                DisclosureGroup("Log", isExpanded: $showsLog) {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 3) {
                            ForEach(Array(logLines.suffix(120).enumerated()), id: \.offset) { _, line in
                                Text(line)
                                    .font(.system(size: 11, design: .monospaced))
                                    .textSelection(.enabled)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 240)
                }
            }
        }
        .task(id: item.id) {
            guard usesManagedModel, !models.hasInventory, !models.isRefreshing, models.error == nil else { return }
            await models.refresh()
        }
    }

    private func getModel(_ modelID: String) {
        guard let template = CommandCatalog.template(id: .modelPull) else { return }
        var draft = template.defaultDraft()
        draft.model = modelID
        let request = StudioRunRequest(mode: item.mode, templateID: .modelPull, template: template, draft: draft)
        let started = models.startPull(request)
        let reason = controller.status
        pullProblem = started ? nil : (reason.isBlank || reason == "Idle" ? "Studio could not start the download." : reason)
    }
}
