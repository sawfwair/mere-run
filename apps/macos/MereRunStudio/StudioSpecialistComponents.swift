import AppKit
import Quartz
import SwiftUI
import UniformTypeIdentifiers

enum StudioSpecialistFiles {
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
    static func saveFile(
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

    static func timestampedDirectory(component: String) -> URL {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Movies/MereRun", isDirectory: true)
            .appendingPathComponent(component, isDirectory: true)
            .appendingPathComponent(formatter.string(from: Date()), isDirectory: true)
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
                .buttonStyle(.bordered)
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
                        .buttonStyle(.bordered)
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([activeURL])
                        } label: {
                            Label("Reveal", systemImage: "folder")
                        }
                        .buttonStyle(.bordered)
                    }
                }

                preview
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(MereRunTheme.surface.opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg))

                if artifacts.count > 1 {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 7) {
                            ForEach(artifacts, id: \.self) { url in
                                Button(url.lastPathComponent) { selection = url }
                                    .buttonStyle(.bordered)
                                    .tint(activeURL == url ? MereRunTheme.accent : nil)
                                    .help(url.path)
                            }
                        }
                    }
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
        if let url = activeURL {
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
                item?.status == .failed ? "Run failed" : "Working",
                systemImage: item?.status == .failed ? "exclamationmark.triangle" : "hourglass",
                description: Text(item?.status == .failed ? "Open the Library row for diagnostics." : "The first artifact will appear here.")
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
        let request = StudioRunRequest(
            mode: mode,
            templateID: templateID,
            template: template,
            draft: draft
        )
        let preview = controller.commandPreview(template: template, draft: draft, masksSecrets: true)
        let status: StudioLibraryStatus = controller.isRunning || controller.queuedRunCount > 0
            ? .queued
            : .running
        library.start(request: request, commandPreview: preview, status: status)
        _ = controller.run(studio: request)
        return request.id
    }
}
