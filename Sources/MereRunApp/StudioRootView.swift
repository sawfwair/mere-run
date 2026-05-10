import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct StudioRootView: View {
    @EnvironmentObject private var controller: MereRunController
    @StateObject private var library = StudioLibraryStore()
    @State private var mode: StudioMode = .createImage
    @State private var draft = StudioDraft()
    @State private var showLibrary = true
    @State private var showAdvanced = false
    @State private var showOptions = false
    @State private var activeLibraryID: UUID?
    @State private var selectedLibraryID: UUID?
    @State private var studioError: String?

    private var selectedItem: StudioLibraryItem? {
        guard let selectedLibraryID else { return library.items.first }
        return library.items.first { $0.id == selectedLibraryID }
    }

    private var readiness: ModelReadinessState {
        controller.readinessByMode[mode] ?? .unknown("Readiness has not been checked yet.")
    }

    var body: some View {
        HStack(spacing: 0) {
            if showLibrary {
                StudioLibraryPanel(
                    items: library.items,
                    selectedID: $selectedLibraryID,
                    isVisible: $showLibrary
                )
                .frame(width: 292)

                Divider()
                    .overlay(MereRunTheme.border.opacity(0.55))
            }

            VStack(spacing: 0) {
                StudioTopBar(
                    mode: $mode,
                    showLibrary: $showLibrary,
                    showAdvanced: $showAdvanced,
                    readiness: readiness,
                    resolvedCLI: controller.resolvedCLI
                )

                Divider()
                    .overlay(MereRunTheme.border.opacity(0.45))

                StudioCanvas(
                    mode: mode,
                    item: selectedItem,
                    isRunning: controller.isRunning,
                    status: controller.status,
                    readiness: readiness,
                    error: studioError,
                    logs: controller.logs,
                    liveOutputText: controller.liveOutputText,
                    onOpen: openSelectedOutput,
                    onReveal: revealSelectedOutput,
                    onPullModel: pullModel,
                    onShowDetails: { showAdvanced = true }
                )

                StudioPromptBar(
                    mode: mode,
                    draft: $draft,
                    showOptions: $showOptions,
                    isRunning: controller.isRunning,
                    readiness: readiness,
                    onRun: runStudioCommand,
                    onStop: controller.cancel,
                    onAttach: chooseAttachment
                )
                .padding(.horizontal, 24)
                .padding(.bottom, 22)
            }
            .frame(minWidth: 680)

            if showAdvanced {
                Divider()
                    .overlay(MereRunTheme.border.opacity(0.55))
                AdvancedControlSurface()
                    .frame(width: 560)
            }
        }
        .background {
            ZStack {
                MereRunTheme.background
                LinearGradient(
                    colors: [
                        MereRunTheme.surfaceRaised.opacity(0.28),
                        MereRunTheme.background,
                        MereRunTheme.surface.opacity(0.2)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            }
            .ignoresSafeArea()
        }
        .sheet(isPresented: $showOptions) {
            StudioOptionsSheet(mode: mode, draft: $draft)
                .frame(width: 500)
        }
        .onAppear {
            draft.reset(for: mode)
            controller.checkReadiness(for: mode, draft: draft)
        }
        .onChange(of: mode) { _, newMode in
            var nextDraft = StudioDraft()
            nextDraft.reset(for: newMode)
            draft = nextDraft
            studioError = nil
            controller.checkReadiness(for: newMode, draft: nextDraft)
        }
        .onChange(of: draft.model) { _, _ in
            controller.checkReadiness(for: mode, draft: draft)
        }
        .onChange(of: controller.lastRunResult) { _, result in
            guard let result, let activeLibraryID else { return }
            library.complete(
                id: activeLibraryID,
                exitCode: result.exitCode,
                outputURL: result.outputURL,
                outputText: result.outputText,
                commandPreview: result.commandPreview.maskingAPIKeyValue()
            )
            selectedLibraryID = activeLibraryID
            self.activeLibraryID = nil
            controller.checkReadiness(for: mode, draft: draft)
        }
    }

    private func runStudioCommand() {
        studioError = nil

        if readiness.blocksRun {
            studioError = readiness.message
            return
        }

        do {
            let request = try StudioCommandAdapter.makeRequest(mode: mode, draft: draft)
            let preview = controller
                .commandPreview(template: request.template, draft: request.draft, masksSecrets: true)
            library.start(request: request, commandPreview: preview)
            activeLibraryID = request.id
            selectedLibraryID = request.id
            controller.run(studio: request)
        } catch {
            studioError = error.localizedDescription
        }
    }

    private func pullModel() {
        do {
            guard let request = try StudioCommandAdapter.pullRequest(for: mode, draft: draft) else {
                studioError = "This mode does not need a managed model."
                return
            }
            showAdvanced = true
            controller.run(studio: request)
        } catch {
            studioError = error.localizedDescription
        }
    }

    private func chooseAttachment() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = mode.acceptedTypes.isEmpty ? [.item] : mode.acceptedTypes
        if panel.runModal() == .OK, let url = panel.url {
            draft.inputPath = url.path
            studioError = nil
        }
    }

    private func openSelectedOutput() {
        guard let url = selectedItem?.outputURL else { return }
        NSWorkspace.shared.open(url)
    }

    private func revealSelectedOutput() {
        guard let url = selectedItem?.outputURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
}

private struct StudioTopBar: View {
    @Binding var mode: StudioMode
    @Binding var showLibrary: Bool
    @Binding var showAdvanced: Bool
    let readiness: ModelReadinessState
    let resolvedCLI: String

    var body: some View {
        VStack(spacing: 14) {
            HStack(spacing: 12) {
                Button {
                    withAnimation(.easeOut(duration: 0.18)) {
                        showLibrary.toggle()
                    }
                } label: {
                    Image(systemName: "sidebar.left")
                        .frame(width: 30, height: 30)
                }
                .buttonStyle(.plain)
                .help("Show library")

                VStack(alignment: .leading, spacing: 2) {
                    Text("mere.run")
                        .font(.system(size: 20, weight: .semibold))
                    Text("Create anything. Locally.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }

                Spacer()

                StudioStatusPill(
                    title: readiness.title,
                    detail: readiness.message,
                    systemImage: readiness.blocksRun ? "arrow.down.circle" : "checkmark.circle",
                    color: readiness.blocksRun ? MereRunTheme.yellow : MereRunTheme.green
                )

                StudioStatusPill(
                    title: "CLI",
                    detail: resolvedCLI,
                    systemImage: "terminal",
                    color: MereRunTheme.accent
                )

                Button {
                    withAnimation(.easeOut(duration: 0.18)) {
                        showAdvanced.toggle()
                    }
                } label: {
                    Label("Advanced", systemImage: "slider.horizontal.3")
                }
                .buttonStyle(.bordered)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(StudioMode.allCases) { candidate in
                        StudioModeChip(
                            mode: candidate,
                            isSelected: candidate == mode
                        ) {
                            withAnimation(.easeOut(duration: 0.18)) {
                                mode = candidate
                            }
                        }
                    }
                }
                .padding(.horizontal, 2)
            }
        }
        .padding(.horizontal, 22)
        .padding(.top, 16)
        .padding(.bottom, 14)
    }
}

private struct StudioModeChip: View {
    let mode: StudioMode
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 7) {
                Image(systemName: mode.systemImage)
                    .font(.system(size: 12, weight: .semibold))
                Text(mode.title)
                    .font(.system(size: 12, weight: .semibold))
            }
            .foregroundStyle(isSelected ? MereRunTheme.background : MereRunTheme.textSecondary)
            .padding(.horizontal, 12)
            .frame(height: 32)
            .background {
                Capsule()
                    .fill(isSelected ? MereRunTheme.accent : MereRunTheme.surface)
                    .overlay {
                        Capsule()
                            .strokeBorder(MereRunTheme.border.opacity(isSelected ? 0 : 0.8), lineWidth: 1)
                    }
            }
        }
        .buttonStyle(.plain)
    }
}

private struct StudioStatusPill: View {
    let title: String
    let detail: String
    let systemImage: String
    let color: Color

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: systemImage)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(color)
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.system(size: 11, weight: .semibold))
                Text(detail)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 10)
        .frame(height: 38)
        .frame(maxWidth: 210)
        .merePanel(cornerRadius: 18)
    }
}

private struct StudioCanvas: View {
    let mode: StudioMode
    let item: StudioLibraryItem?
    let isRunning: Bool
    let status: String
    let readiness: ModelReadinessState
    let error: String?
    let logs: [LogLine]
    let liveOutputText: String
    let onOpen: () -> Void
    let onReveal: () -> Void
    let onPullModel: () -> Void
    let onShowDetails: () -> Void

    private var visibleLiveOutputText: String? {
        guard isRunning, mode == .chat || mode == .code else { return nil }
        let text = liveOutputText
            .replacingOccurrences(of: "\0", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }

    var body: some View {
        ZStack {
            if let item {
                StudioOutputView(item: item, liveOutputText: visibleLiveOutputText, onOpen: onOpen, onReveal: onReveal)
                    .padding(32)
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
            } else {
                StudioEmptyState(mode: mode)
                    .padding(32)
            }

            if isRunning && visibleLiveOutputText == nil {
                StudioRunningOverlay(status: status, latestLog: logs.last?.text)
                    .transition(.opacity)
            }

            if readiness.blocksRun || error != nil {
                StudioReadinessOverlay(
                    title: error == nil ? readiness.title : "Needs attention",
                    message: error ?? readiness.message,
                    canPull: readiness.blocksRun,
                    onPullModel: onPullModel,
                    onShowDetails: onShowDetails
                )
                .padding(32)
                .transition(.opacity.combined(with: .scale(scale: 0.98)))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .animation(.easeOut(duration: 0.18), value: isRunning)
        .animation(.easeOut(duration: 0.18), value: readiness.blocksRun)
    }
}

private struct StudioEmptyState: View {
    let mode: StudioMode

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: mode.systemImage)
                .font(.system(size: 48, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
                .frame(width: 92, height: 92)
                .background {
                    Circle()
                        .fill(MereRunTheme.surfaceRaised)
                }

            VStack(spacing: 8) {
                Text(mode.emptyTitle)
                    .font(.system(size: 30, weight: .semibold))
                Text(mode.emptyMessage)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 460)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct StudioRunningOverlay: View {
    let status: String
    let latestLog: String?

    var body: some View {
        VStack(spacing: 14) {
            ProgressView()
                .controlSize(.large)
            Text(status)
                .font(.system(size: 18, weight: .semibold))
            if let latestLog {
                Text(latestLog)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(3)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 420)
            }
        }
        .padding(28)
        .merePanel(cornerRadius: 18)
        .shadow(color: .black.opacity(0.28), radius: 28, y: 12)
    }
}

private struct StudioReadinessOverlay: View {
    let title: String
    let message: String
    let canPull: Bool
    let onPullModel: () -> Void
    let onShowDetails: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: canPull ? "arrow.down.circle" : "exclamationmark.triangle")
                .font(.system(size: 32, weight: .semibold))
                .foregroundStyle(canPull ? MereRunTheme.yellow : MereRunTheme.red)
            Text(title)
                .font(.system(size: 22, weight: .semibold))
            Text(message)
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 280)
            HStack(spacing: 10) {
                if canPull {
                    Button {
                        onPullModel()
                    } label: {
                        Label("Pull Model", systemImage: "arrow.down.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(MereRunTheme.accent)
                }
                Button {
                    onShowDetails()
                } label: {
                    Label("Details", systemImage: "terminal")
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(28)
        .merePanel(cornerRadius: 18)
        .shadow(color: .black.opacity(0.3), radius: 30, y: 12)
    }
}

private struct StudioOutputView: View {
    let item: StudioLibraryItem
    let liveOutputText: String?
    let onOpen: () -> Void
    let onReveal: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            outputPreview
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(MereRunTheme.surface.opacity(0.45))
                .clipShape(RoundedRectangle(cornerRadius: 18))
                .overlay {
                    RoundedRectangle(cornerRadius: 18)
                        .strokeBorder(MereRunTheme.border.opacity(0.65), lineWidth: 1)
                }

            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.mode.title)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    Text(item.displayTitle)
                        .font(.system(size: 16, weight: .semibold))
                        .lineLimit(1)
                }
                Spacer()
                statusBadge
                if item.outputURL != nil {
                    Button("Open", action: onOpen)
                        .buttonStyle(.bordered)
                    Button {
                        onReveal()
                    } label: {
                        Image(systemName: "finder")
                    }
                    .buttonStyle(.bordered)
                    .help("Reveal in Finder")
                }
            }
        }
    }

    @ViewBuilder
    private var outputPreview: some View {
        if let url = item.outputURL {
            switch StudioOutputFileKind.classify(url) {
            case .image:
                StudioAsyncImagePreview(
                    url: url,
                    maxPixelSize: 2_200,
                    contentMode: .fit,
                    fallbackSystemImage: iconName(for: url)
                )
                .padding(22)
            case .text:
                StudioTextFilePreview(url: url)
            case .other:
                filePlaceholder(for: url)
            }
        } else if let text = liveOutputText ?? item.outputText, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            ScrollView {
                Text(text)
                    .font(item.mode == .chat ? MereRunTheme.bodyFont : MereRunTheme.monoFont)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(22)
            }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "doc")
                    .font(.system(size: 46, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
                Text(item.status == .failed ? "Run did not produce a file." : "Output will appear here.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
    }

    private func filePlaceholder(for url: URL) -> some View {
        VStack(spacing: 12) {
            Image(systemName: iconName(for: url))
                .font(.system(size: 54, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
            Text(url.lastPathComponent)
                .font(.system(size: 16, weight: .semibold))
                .lineLimit(1)
            Text(url.deletingLastPathComponent().path)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
        }
        .padding(22)
    }

    private var statusBadge: some View {
        Text(item.status.rawValue.capitalized)
            .font(MereRunTheme.captionFont)
            .foregroundStyle(statusColor)
            .padding(.horizontal, 10)
            .frame(height: 28)
            .background {
                Capsule()
                    .fill(statusColor.opacity(0.14))
            }
    }

    private var statusColor: Color {
        switch item.status {
        case .running: return MereRunTheme.yellow
        case .completed: return MereRunTheme.green
        case .failed: return MereRunTheme.red
        }
    }

    private func iconName(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "wav", "mp3", "m4a": return "waveform"
        case "mp4", "mov": return "film"
        case "json": return "curlybraces"
        case "safetensors": return "shippingbox"
        default: return "doc"
        }
    }
}

private enum StudioImageContentMode: Equatable {
    case fit
    case fill
}

private enum StudioImageLoadState {
    case loading
    case loaded(NSImage)
    case unavailable
}

private struct StudioAsyncImagePreview: View {
    let url: URL
    let maxPixelSize: CGFloat
    let contentMode: StudioImageContentMode
    let fallbackSystemImage: String

    @State private var loadState = StudioImageLoadState.loading

    var body: some View {
        Group {
            switch loadState {
            case .loading:
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .loaded(let image):
                if contentMode == .fill {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFit()
                }
            case .unavailable:
                Image(systemName: fallbackSystemImage)
                    .font(.system(size: 32, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task(id: url) {
            loadState = .loading
            let loaded = await Task.detached(priority: .userInitiated) {
                StudioImagePreviewLoader.downsampledImage(from: url, maxPixelSize: maxPixelSize)
            }.value
            guard !Task.isCancelled else { return }
            if let image = loaded?.image {
                loadState = .loaded(image)
            } else {
                loadState = .unavailable
            }
        }
    }
}

private struct StudioTextFilePreview: View {
    let url: URL

    @State private var text: String?
    @State private var didLoad = false

    var body: some View {
        ScrollView {
            if let text {
                Text(text)
                    .font(MereRunTheme.monoFont)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(22)
            } else if didLoad {
                Text("Text preview unavailable.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(22)
            } else {
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, minHeight: 180)
                    .padding(22)
            }
        }
        .task(id: url) {
            text = nil
            didLoad = false
            let preview = await Task.detached(priority: .userInitiated) {
                StudioTextPreviewReader.previewText(from: url)
            }.value
            guard !Task.isCancelled else { return }
            text = preview
            didLoad = true
        }
    }
}

private struct StudioPromptBar: View {
    let mode: StudioMode
    @Binding var draft: StudioDraft
    @Binding var showOptions: Bool
    let isRunning: Bool
    let readiness: ModelReadinessState
    let onRun: () -> Void
    let onStop: () -> Void
    let onAttach: () -> Void

    var body: some View {
        VStack(spacing: 10) {
            if !draft.inputPath.isBlank {
                HStack {
                    Label(URL(fileURLWithPath: draft.inputPath).lastPathComponent, systemImage: "paperclip")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                    Spacer()
                    Button {
                        draft.inputPath = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 14)
            }

            HStack(spacing: 12) {
                Button(action: onAttach) {
                    Image(systemName: mode.requiresAttachment ? "paperclip.circle.fill" : "paperclip")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 34, height: 34)
                }
                .buttonStyle(.plain)
                .disabled(mode.acceptedTypes.isEmpty)
                .help(mode.requiresAttachment ? "Attach required input" : "Attach reference")

                if mode == .listen {
                    Text(mode.promptPlaceholder)
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    TextField(mode.promptPlaceholder, text: $draft.prompt, axis: .vertical)
                        .textFieldStyle(.plain)
                        .font(.system(size: 15, weight: .medium))
                        .lineLimit(1...4)
                        .onSubmit(onRun)
                }

                Button {
                    showOptions = true
                } label: {
                    Image(systemName: "slider.horizontal.3")
                        .frame(width: 34, height: 34)
                }
                .buttonStyle(.plain)
                .help("Options")

                Button {
                    isRunning ? onStop() : onRun()
                } label: {
                    Image(systemName: isRunning ? "stop.fill" : "arrow.up")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundStyle(MereRunTheme.background)
                        .frame(width: 38, height: 38)
                        .background {
                            Circle()
                                .fill(isRunning ? MereRunTheme.red : (readiness.blocksRun ? MereRunTheme.yellow : MereRunTheme.accent))
                        }
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.return, modifiers: .command)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background {
                RoundedRectangle(cornerRadius: 20)
                    .fill(MereRunTheme.surfaceRaised)
                    .overlay {
                        RoundedRectangle(cornerRadius: 20)
                            .strokeBorder(MereRunTheme.border.opacity(0.75), lineWidth: 1)
                    }
                    .shadow(color: .black.opacity(0.22), radius: 20, y: 10)
            }
        }
    }
}

private struct StudioOptionsSheet: View {
    let mode: StudioMode
    @Binding var draft: StudioDraft
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text("\(mode.title) Options")
                    .font(MereRunTheme.titleFont)
                Spacer()
                Button("Done") {
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }

            if mode == .readImage {
                Picker("Task", selection: $draft.readImageAction) {
                    ForEach(StudioReadImageAction.allCases) { action in
                        Text(action.title).tag(action)
                    }
                }
                .pickerStyle(.segmented)
            }

            if secondaryLabel != nil {
                TextField(secondaryLabel!, text: $draft.secondaryText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
            }

            TextField("Model", text: $draft.model)
                .textFieldStyle(.plain)
                .padding(10)
                .merePanel()

            if [.createImage, .video].contains(mode) {
                HStack(spacing: 10) {
                    Stepper("Width \(draft.width)", value: $draft.width, in: 64...4096, step: 64)
                    Stepper("Height \(draft.height)", value: $draft.height, in: 64...4096, step: 64)
                }
            }

            if [.createImage, .music].contains(mode) {
                Stepper("Steps \(draft.steps)", value: $draft.steps, in: 1...80, step: 1)
            }

            if mode == .music {
                TextField("Duration seconds", value: $draft.durationSeconds, format: .number)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
            }

            if [.createImage, .music, .video].contains(mode) {
                TextField("Seed", text: $draft.seed)
                    .textFieldStyle(.plain)
                    .padding(10)
                    .merePanel()
            }

            Spacer()
        }
        .padding(22)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
    }

    private var secondaryLabel: String? {
        switch mode {
        case .createImage: return "Negative prompt"
        case .chat, .code: return "System"
        case .speak: return "Voice"
        case .music: return "Lyrics"
        default: return nil
        }
    }
}

private struct StudioLibraryPanel: View {
    let items: [StudioLibraryItem]
    @Binding var selectedID: UUID?
    @Binding var isVisible: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Library")
                        .font(.system(size: 20, weight: .semibold))
                    Text("\(items.count) local runs")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
                Button {
                    withAnimation(.easeOut(duration: 0.18)) {
                        isVisible = false
                    }
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.plain)
            }
            .padding(18)

            Divider()
                .overlay(MereRunTheme.border.opacity(0.55))

            if items.isEmpty {
                VStack(spacing: 10) {
                    Image(systemName: "rectangle.stack")
                        .font(.system(size: 30, weight: .semibold))
                        .foregroundStyle(MereRunTheme.textMuted)
                    Text("Runs you create will land here.")
                        .font(MereRunTheme.bodyFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(22)
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(items) { item in
                            StudioLibraryRow(
                                item: item,
                                isSelected: selectedID == item.id
                            ) {
                                selectedID = item.id
                            }
                        }
                    }
                    .padding(12)
                }
            }
        }
        .background(MereRunTheme.background)
    }
}

private struct StudioLibraryRow: View {
    let item: StudioLibraryItem
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                thumbnail
                    .frame(width: 46, height: 38)
                    .background(MereRunTheme.surface)
                    .clipShape(RoundedRectangle(cornerRadius: 7))

                VStack(alignment: .leading, spacing: 4) {
                    Text(item.displayTitle)
                        .font(.system(size: 12, weight: .semibold))
                        .lineLimit(1)
                    Text("\(item.mode.title) · \(item.status.rawValue)")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Spacer()
            }
            .padding(9)
            .background {
                RoundedRectangle(cornerRadius: 9)
                    .fill(isSelected ? MereRunTheme.surfaceRaised : Color.clear)
            }
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var thumbnail: some View {
        if let url = item.outputURL, StudioOutputFileKind.classify(url) == .image {
            StudioAsyncImagePreview(
                url: url,
                maxPixelSize: 160,
                contentMode: .fill,
                fallbackSystemImage: item.mode.systemImage
            )
        } else {
            Image(systemName: item.mode.systemImage)
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
        }
    }
}

private extension String {
    func maskingAPIKeyValue() -> String {
        var words = ShellWords.split(self)
        words = words.maskingSecrets()
        return words.shellQuoted()
    }
}
