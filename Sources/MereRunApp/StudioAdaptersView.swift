import AppKit
import SwiftUI

struct StudioAdapterCatalogPayload: Decodable, Equatable {
    let schemaVersion: Int
    let adapterStore: String
    let adapters: [StudioAdapterRow]
}

struct StudioAdapterRow: Decodable, Equatable, Identifiable {
    let id: String
    let title: String
    let version: String
    let summary: String
    let baseModelID: String
    let format: String
    let license: String
    let byteCount: Int64
    let installed: Bool
    let path: String?
}

/// First-class adapter hub shared by Image, Chat/Code, Music, and specialist video workflows.
/// Catalog mutation still runs through the CLI contract, while every pull/training launch is a
/// durable Library job rather than an invisible utility process.
struct StudioAdaptersSheet: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @Environment(\.dismiss) private var dismiss

    let activeModelID: String
    let onUse: (StudioAdapterRow) -> Void
    let onUseLocal: (String) -> Void
    let onTrain: (CommandTemplateID) -> Void

    @State private var payload: StudioAdapterCatalogPayload?
    @State private var selectedID: String?
    @State private var searchText = ""
    @State private var statusMessage = "Loading adapters"
    @State private var isRefreshing = false
    @State private var pullingID: String?
    @State private var selectedTrainingID: UUID?

    private var rows: [StudioAdapterRow] {
        let all = payload?.adapters ?? []
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return all }
        return all.filter {
            $0.id.lowercased().contains(query)
                || $0.title.lowercased().contains(query)
                || $0.baseModelID.lowercased().contains(query)
                || $0.summary.lowercased().contains(query)
        }
    }

    private var selectedRow: StudioAdapterRow? {
        payload?.adapters.first { $0.id == selectedID }
    }

    private var trainingRuns: [StudioLibraryItem] {
        let ids: Set<CommandTemplateID> = [.imageTrainLoRA, .textTrainLoRA, .musicTrainAdapter]
        return library.items.filter { item in
            guard let templateID = item.templateID else { return false }
            return ids.contains(templateID)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.6))

            HStack(spacing: 0) {
                catalogList
                    .frame(width: 350)
                Divider().overlay(MereRunTheme.border.opacity(0.6))
                detailPane
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(width: 980, height: 650)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task { await refresh() }
        .onReceive(controller.runCompletions) { result in
            guard result.templateID == .adapterPull else { return }
            pullingID = nil
            Task { await refresh() }
        }
    }

    private var header: some View {
        HStack(spacing: MereRunTheme.Spacing.md) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Adapters")
                    .font(MereRunTheme.titleFont)
                Text(statusMessage)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            Spacer()
            Button {
                revealAdapterStore()
            } label: {
                Label("Store", systemImage: "folder")
            }
            .buttonStyle(.bordered)
            .disabled(payload == nil)

            Button {
                chooseLocalAdapter()
            } label: {
                Label("Use local…", systemImage: "doc.badge.plus")
            }
            .buttonStyle(.bordered)

            Button {
                Task { await refresh() }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.bordered)
            .disabled(isRefreshing)

            Button("Done") { dismiss() }
                .buttonStyle(.borderedProminent)
                .tint(MereRunTheme.accent)
        }
        .padding(18)
    }

    private var catalogList: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 7) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                TextField("Search adapters", text: $searchText)
                    .textFieldStyle(.plain)
                if !searchText.isEmpty {
                    Button {
                        searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
                }
            }
            .padding(.horizontal, MereRunTheme.Spacing.sm)
            .frame(height: 32)
            .background {
                Capsule()
                    .fill(MereRunTheme.surface)
                    .overlay {
                        Capsule().strokeBorder(MereRunTheme.border.opacity(0.7), lineWidth: 1)
                    }
            }
            .padding(12)

            HStack {
                Text("Verified catalog")
                    .font(MereRunTheme.sectionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                Spacer()
                Text("\(rows.count)")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 7)

            if rows.isEmpty {
                ContentUnavailableView(
                    "No adapters",
                    systemImage: "square.stack.3d.up",
                    description: Text(searchText.isEmpty ? "The CLI returned an empty catalog." : "No catalog entries match.")
                )
            } else {
                ScrollView {
                    LazyVStack(spacing: 5) {
                        ForEach(rows) { row in
                            Button {
                                selectedID = row.id
                            } label: {
                                HStack(spacing: 10) {
                                    Image(systemName: row.installed ? "checkmark.seal.fill" : "arrow.down.circle")
                                        .foregroundStyle(row.installed ? MereRunTheme.green : MereRunTheme.accent)
                                        .frame(width: 22)
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(row.title)
                                            .font(.system(size: 12.5, weight: .semibold))
                                            .foregroundStyle(MereRunTheme.textPrimary)
                                            .lineLimit(1)
                                        Text(row.baseModelID)
                                            .font(MereRunTheme.captionFont)
                                            .foregroundStyle(MereRunTheme.textMuted)
                                            .lineLimit(1)
                                    }
                                    Spacer()
                                }
                                .padding(8)
                                .background {
                                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                                        .fill(selectedID == row.id ? MereRunTheme.accentSoft : Color.clear)
                                }
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 9)
                    .padding(.bottom, 12)
                }
            }
        }
    }

    private var detailPane: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                if let selectedRow {
                    adapterDetail(selectedRow)
                } else {
                    ContentUnavailableView(
                        "Choose an adapter",
                        systemImage: "slider.horizontal.3",
                        description: Text("Inspect compatibility, download it, and apply it to the active workspace.")
                    )
                    .frame(maxWidth: .infinity, minHeight: 230)
                }

                Divider().overlay(MereRunTheme.border.opacity(0.5))
                trainingSection
            }
            .padding(20)
        }
    }

    private func adapterDetail(_ row: StudioAdapterRow) -> some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(row.title)
                        .font(.system(size: 20, weight: .semibold))
                    Text("\(row.id) · v\(row.version)")
                        .font(MereRunTheme.monoFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
                Text(row.installed ? "Installed" : "Available")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(row.installed ? MereRunTheme.green : MereRunTheme.accent)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background {
                        Capsule().fill((row.installed ? MereRunTheme.green : MereRunTheme.accent).opacity(0.13))
                    }
            }

            Text(row.summary)
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textSecondary)

            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 7) {
                detailRow("Base model", row.baseModelID)
                detailRow("Format", row.format)
                detailRow("License", row.license)
                detailRow("Size", ByteCountFormatter.string(fromByteCount: row.byteCount, countStyle: .file))
            }

            if !activeModelID.isBlank, activeModelID != row.baseModelID {
                Label(
                    "The active model is \(activeModelID). This adapter targets \(row.baseModelID).",
                    systemImage: "exclamationmark.triangle.fill"
                )
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.yellow)
            }

            HStack(spacing: 10) {
                if row.installed {
                    Button {
                        onUse(row)
                        dismiss()
                    } label: {
                        Label("Use in Studio", systemImage: "checkmark.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(MereRunTheme.accent)

                    Button {
                        reveal(row)
                    } label: {
                        Label("Reveal", systemImage: "magnifyingglass")
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button {
                        pull(row)
                    } label: {
                        Label(pullingID == row.id ? "Queued" : "Download", systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(MereRunTheme.accent)
                    .disabled(pullingID != nil)
                }
            }
        }
        .padding(16)
        .merePanel()
    }

    @ViewBuilder
    private func detailRow(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Text(value)
                .font(MereRunTheme.monoFont)
                .textSelection(.enabled)
        }
    }

    private var trainingSection: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Train your own")
                        .font(.system(size: 16, weight: .semibold))
                    Text("Launch a typed training run; progress and outputs stay in the Library.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
            }

            HStack(spacing: 10) {
                trainingButton("Image LoRA", symbol: "photo.badge.plus", templateID: .imageTrainLoRA)
                trainingButton("Text LoRA", symbol: "text.badge.plus", templateID: .textTrainLoRA)
                trainingButton("Music Adapter", symbol: "music.note.list", templateID: .musicTrainAdapter)
            }

            if !trainingRuns.isEmpty {
                VStack(alignment: .leading, spacing: 7) {
                    Text("Recent training")
                        .font(MereRunTheme.sectionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    ForEach(trainingRuns.prefix(4)) { item in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Image(systemName: item.displaySystemImage)
                                    .foregroundStyle(MereRunTheme.accent)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(item.displayTitle)
                                        .font(.system(size: 12.5, weight: .medium))
                                        .lineLimit(1)
                                    Text("\(item.displayKindTitle) · \(item.status.rawValue)")
                                        .font(MereRunTheme.captionFont)
                                        .foregroundStyle(MereRunTheme.textMuted)
                                }
                                Spacer()
                                if item.status == .running,
                                   controller.activeRunRequestID == item.id,
                                   let progress = controller.currentProgress {
                                    Text("\(Int(((progress.fractionCompleted ?? 0) * 100).rounded()))%")
                                        .font(MereRunTheme.captionFont)
                                        .monospacedDigit()
                                        .foregroundStyle(MereRunTheme.accent)
                                }
                                Button(selectedTrainingID == item.id ? "Hide" : "Inspect") {
                                    selectedTrainingID = selectedTrainingID == item.id ? nil : item.id
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                                if let output = adapterOutput(for: item) {
                                    Button("Use") {
                                        onUseLocal(output.path)
                                        dismiss()
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .tint(MereRunTheme.accent)
                                    .controlSize(.small)
                                }
                                if let output = item.outputURL {
                                    Button("Reveal") {
                                        NSWorkspace.shared.activateFileViewerSelecting([output])
                                    }
                                    .buttonStyle(.bordered)
                                    .controlSize(.small)
                                }
                            }
                            if selectedTrainingID == item.id {
                                StudioTrainingRunDetail(item: item)
                            }
                        }
                        .padding(9)
                        .merePanel()
                    }
                }
            }
        }
    }

    private func trainingButton(
        _ title: String,
        symbol: String,
        templateID: CommandTemplateID
    ) -> some View {
        Button {
            dismiss()
            onTrain(templateID)
        } label: {
            Label(title, systemImage: symbol)
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
    }

    @MainActor
    private func refresh() async {
        isRefreshing = true
        statusMessage = "Refreshing adapter catalog"
        let result = await controller.utilityCommandResult(args: ["adapter", "list", "--json"])
        isRefreshing = false

        guard result.exitCode == 0,
              let data = result.stdout.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(StudioAdapterCatalogPayload.self, from: data) else {
            statusMessage = "Could not load adapters"
            return
        }
        payload = decoded
        if selectedID == nil || !decoded.adapters.contains(where: { $0.id == selectedID }) {
            selectedID = decoded.adapters.first?.id
        }
        let installed = decoded.adapters.filter(\.installed).count
        statusMessage = "\(installed) installed · \(decoded.adapters.count) verified"
    }

    private func pull(_ row: StudioAdapterRow) {
        guard let template = CommandCatalog.template(id: .adapterPull) else { return }
        var draft = template.defaultDraft()
        draft.prompt = row.id
        let request = StudioRunRequest(
            mode: .chat,
            templateID: template.id,
            template: template,
            draft: draft
        )
        let preview = controller.commandPreview(template: template, draft: draft, masksSecrets: true)
        let status: StudioLibraryStatus = controller.isRunning || controller.queuedRunCount > 0
            ? .queued
            : .running
        pullingID = row.id
        library.start(request: request, commandPreview: preview, status: status)
        _ = controller.run(studio: request)
    }

    private func reveal(_ row: StudioAdapterRow) {
        guard let path = row.path else { return }
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }

    private func revealAdapterStore() {
        guard let store = payload?.adapterStore else { return }
        NSWorkspace.shared.open(URL(fileURLWithPath: store, isDirectory: true))
    }

    private func chooseLocalAdapter() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.data]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url {
            onUseLocal(url.path)
            dismiss()
        }
    }

    private func adapterOutput(for item: StudioLibraryItem) -> URL? {
        item.allArtifactURLs.first {
            ["safetensors", "ckpt", "pt"].contains($0.pathExtension.lowercased())
        }
    }
}

private struct StudioTrainingLossPoint: Identifiable {
    let step: Double
    let loss: Double
    var id: Double { step }
}

private struct StudioTrainingRunDetail: View {
    let item: StudioLibraryItem
    @State private var discoveredArtifacts: [URL] = []

    private var artifactURLs: [URL] {
        var seen = Set<String>()
        return (item.allArtifactURLs + discoveredArtifacts).filter {
            seen.insert($0.standardizedFileURL.path).inserted
        }
    }

    private var lossPoints: [StudioTrainingLossPoint] {
        guard let url = artifactURLs.first(where: {
            $0.lastPathComponent.contains(".loss.") && $0.pathExtension.lowercased() == "csv"
        }),
        let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return text.components(separatedBy: .newlines).compactMap { line in
            let values = line.split(separator: ",")
            guard values.count >= 2,
                  let step = Double(values[0].trimmingCharacters(in: .whitespaces)),
                  let loss = Double(values[1].trimmingCharacters(in: .whitespaces)) else { return nil }
            return StudioTrainingLossPoint(step: step, loss: loss)
        }
    }

    private var samples: [URL] {
        artifactURLs.filter {
            ["png", "jpg", "jpeg", "webp"].contains($0.pathExtension.lowercased())
                && $0.lastPathComponent.lowercased().contains("sample")
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if lossPoints.count >= 2 {
                HStack {
                    Text("Loss")
                        .font(MereRunTheme.sectionFont)
                    Spacer()
                    if let latest = lossPoints.last {
                        Text(String(format: "%.5f", latest.loss))
                            .font(MereRunTheme.monoFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                }
                lossChart
                    .frame(height: 92)
            }

            if !samples.isEmpty {
                Text("Preview samples")
                    .font(MereRunTheme.sectionFont)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(samples, id: \.self) { url in
                            Button {
                                QuickLookCoordinator.shared.preview(url)
                            } label: {
                                StudioAsyncImagePreview(
                                    url: url,
                                    maxPixelSize: 360,
                                    contentMode: .fill,
                                    fallbackSystemImage: "photo"
                                )
                                .frame(width: 92, height: 76)
                                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }

            if lossPoints.isEmpty && samples.isEmpty {
                Text("Metrics and sample images will appear here as the trainer writes them.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .padding(.top, 4)
        .task(id: item.status) {
            while !Task.isCancelled {
                refreshArtifacts()
                guard item.status == .running || item.status == .queued else { return }
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    private var lossChart: some View {
        Canvas { context, size in
            guard let minLoss = lossPoints.map(\.loss).min(),
                  let maxLoss = lossPoints.map(\.loss).max(),
                  let minStep = lossPoints.map(\.step).min(),
                  let maxStep = lossPoints.map(\.step).max() else { return }
            let lossRange = max(0.000_001, maxLoss - minLoss)
            let stepRange = max(1, maxStep - minStep)
            var path = Path()
            for (index, point) in lossPoints.enumerated() {
                let x = ((point.step - minStep) / stepRange) * size.width
                let y = size.height - (((point.loss - minLoss) / lossRange) * (size.height - 8)) - 4
                if index == 0 {
                    path.move(to: CGPoint(x: x, y: y))
                } else {
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }
            context.stroke(path, with: .color(MereRunTheme.accent), lineWidth: 2)
        }
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                        .strokeBorder(MereRunTheme.border.opacity(0.55), lineWidth: 1)
                }
        }
    }

    @MainActor
    private func refreshArtifacts() {
        guard let templateID = item.templateID, let draft = item.commandDraft else { return }
        let primary = item.outputURL ?? (
            draft.outputPath.isBlank
                ? nil
                : URL(fileURLWithPath: NSString(string: draft.outputPath).expandingTildeInPath)
        )
        discoveredArtifacts = StudioArtifactDiscovery.urls(
            templateID: templateID,
            draft: draft,
            primaryOutput: primary
        )
    }
}
