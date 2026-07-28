import AppKit
import SwiftUI

/// The center stage: conversation thread for chat/code, otherwise the latest output with
/// empty / running / readiness states layered as needed.
struct StudioCanvas: View {
    let mode: StudioMode
    var isCompact = false
    let item: StudioLibraryItem?
    let conversationItem: StudioLibraryItem?
    let conversationLiveText: String?
    let isConversationRunning: Bool
    let isRunning: Bool
    let status: String
    let readiness: ModelReadinessState
    let error: String?
    let logs: [LogLine]
    let liveOutputText: String
    let progress: StudioRunProgress?
    let onOpen: () -> Void
    let onReveal: () -> Void
    let onQuickLook: () -> Void
    let onPullModel: () -> Void
    let onShowDetails: () -> Void
    let onNewChat: () -> Void
    let onCopy: (String) -> Void
    let onRetry: () -> Void
    let onRerunItem: (StudioLibraryItem) -> Void
    let onEditRun: (StudioLibraryItem) -> Void
    let onEdit: (UUID) -> Void
    let onStop: () -> Void
    let onUseExample: (String) -> Void
    let onAttach: () -> Void

    private var visibleLiveOutputText: String? {
        guard isRunning else { return nil }
        let text = liveOutputText
            .replacingOccurrences(of: "\0", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }

    private var recentLogLines: [String] {
        logs.suffix(4).map(\.text)
    }

    private var shouldShowReadinessOverlay: Bool {
        !isRunning && (readiness.blocksRun || error != nil)
    }

    private var showsConversation: Bool {
        mode.isConversational && (item == nil || item?.isConversation == true)
    }

    var body: some View {
        ZStack {
            if showsConversation {
                StudioConversationView(
                    item: conversationItem,
                    liveText: conversationLiveText,
                    isRunning: isConversationRunning,
                    mode: mode,
                    onNewChat: onNewChat,
                    onCopy: onCopy,
                    onRetry: onRetry,
                    onEdit: onEdit,
                    onUseExample: onUseExample
                )
                .transition(.opacity)
            } else if let item {
                StudioOutputView(
                    item: item,
                    liveOutputText: visibleLiveOutputText,
                    onOpen: onOpen,
                    onReveal: onReveal,
                    onQuickLook: onQuickLook,
                    onCopy: onCopy,
                    onRerun: { onRerunItem(item) },
                    onEdit: { onEditRun(item) }
                )
                .padding(isCompact ? MereRunTheme.Spacing.md : MereRunTheme.Spacing.xxxl)
                .transition(.opacity.combined(with: .scale(scale: 0.985)))
            } else {
                StudioEmptyState(
                    mode: mode,
                    isCompact: isCompact,
                    onUseExample: onUseExample,
                    onAttach: onAttach
                )
                .padding(isCompact ? MereRunTheme.Spacing.md : MereRunTheme.Spacing.xxxl)
            }

            if isRunning && visibleLiveOutputText == nil && !showsConversation {
                StudioRunningOverlay(
                    mode: mode,
                    status: status,
                    progress: progress,
                    recentLogs: recentLogLines,
                    onStop: onStop
                )
                .transition(.opacity.combined(with: .scale(scale: 0.98)))
            }

            if shouldShowReadinessOverlay {
                StudioReadinessOverlay(
                    title: error == nil ? readiness.title : "Needs attention",
                    message: error ?? readiness.message,
                    canPull: error == nil && readiness.canPull,
                    isChecking: error == nil && readiness.isChecking,
                    onPullModel: onPullModel,
                    onShowDetails: onShowDetails
                )
                .padding(isCompact ? MereRunTheme.Spacing.md : MereRunTheme.Spacing.xxxl)
                .transition(.opacity.combined(with: .scale(scale: 0.98)))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .animation(MereRunTheme.Motion.standard, value: isRunning)
        .animation(MereRunTheme.Motion.standard, value: shouldShowReadinessOverlay)
    }
}

/// A blank canvas that teaches: serif headline, one line of guidance, and prompts you can
/// click straight into the composer.
struct StudioEmptyState: View {
    let mode: StudioMode
    var isCompact = false
    var onUseExample: ((String) -> Void)?
    var onAttach: (() -> Void)?

    var body: some View {
        VStack(spacing: isCompact ? MereRunTheme.Spacing.md : MereRunTheme.Spacing.xl) {
            Image(systemName: mode.systemImage)
                .font(.system(size: isCompact ? 28 : 42, weight: .medium))
                .foregroundStyle(MereRunTheme.accent)
                .frame(width: isCompact ? 64 : 96, height: isCompact ? 64 : 96)
                .background {
                    Circle().fill(MereRunTheme.accentSoft.opacity(0.75))
                }
                .overlay {
                    Circle().strokeBorder(MereRunTheme.accent.opacity(0.18), lineWidth: 1)
                }

            VStack(spacing: MereRunTheme.Spacing.xs) {
                Text(mode.emptyTitle)
                    .font(isCompact ? MereRunTheme.displaySmallFont : MereRunTheme.displayFont)
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .multilineTextAlignment(.center)
                Text(mode.emptyMessage)
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 460)
            }

            if mode.requiresAttachment, let onAttach {
                Button(action: onAttach) {
                    Label(attachLabel, systemImage: "paperclip")
                }
                .buttonStyle(.merePrimary)
            }

            if !mode.examplePrompts.isEmpty, let onUseExample {
                FlowLayout(spacing: MereRunTheme.Spacing.xs, lineSpacing: 8) {
                    examplePrompts(onUseExample: onUseExample)
                }
                .frame(maxWidth: isCompact ? 320 : 620)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func examplePrompts(onUseExample: @escaping (String) -> Void) -> some View {
        ForEach(mode.examplePrompts, id: \.self) { example in
            StudioExampleChip(text: example) { onUseExample(example) }
        }
    }

    private var attachLabel: String {
        switch mode {
        case .listen: return "Choose audio…"
        case .track: return "Choose video…"
        default: return "Choose image…"
        }
    }
}

/// One clickable example prompt. Filling, never auto-running, keeps the user in charge.
private struct StudioExampleChip: View {
    let text: String
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Text("“\(text)”")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(hovering ? MereRunTheme.textPrimary : MereRunTheme.textSecondary)
                .lineLimit(2)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 300, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .fill(hovering ? MereRunTheme.surfaceRaised : MereRunTheme.surface.opacity(0.7))
                        .overlay {
                            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                                .strokeBorder(MereRunTheme.border.opacity(0.65), lineWidth: 1)
                        }
                }
                .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .help("Use this prompt")
        .accessibilityLabel("Use example prompt: \(text)")
    }
}

/// Places example chips left-to-right and wraps to a new centered row when the next
/// chip would overflow the available width — so prompts size to their content and
/// never truncate mid-word regardless of window width.
struct FlowLayout: Layout {
    var spacing: CGFloat = 8
    var lineSpacing: CGFloat = 8

    private struct Row {
        var indices: [Int] = []
        var width: CGFloat = 0
        var height: CGFloat = 0
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        let rows = arrange(subviews: subviews, maxWidth: maxWidth)
        let height = rows.map(\.height).reduce(0, +)
            + CGFloat(max(0, rows.count - 1)) * lineSpacing
        let width = rows.map(\.width).max() ?? 0
        return CGSize(width: min(width, maxWidth), height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) {
        let rows = arrange(subviews: subviews, maxWidth: bounds.width)
        var y = bounds.minY
        for row in rows {
            var x = bounds.minX + (bounds.width - row.width) / 2
            for index in row.indices {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(
                    at: CGPoint(x: x, y: y),
                    anchor: .topLeading,
                    proposal: ProposedViewSize(size)
                )
                x += size.width + spacing
            }
            y += row.height + lineSpacing
        }
    }

    private func arrange(subviews: Subviews, maxWidth: CGFloat) -> [Row] {
        var rows: [Row] = []
        var current = Row()
        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            let projected = current.indices.isEmpty ? size.width : current.width + spacing + size.width
            if !current.indices.isEmpty, projected > maxWidth {
                rows.append(current)
                current = Row()
            }
            current.width = current.indices.isEmpty ? size.width : current.width + spacing + size.width
            current.height = max(current.height, size.height)
            current.indices.append(index)
        }
        if !current.indices.isEmpty { rows.append(current) }
        return rows
    }
}

/// The in-flight card: a breathing mode glyph, plain-words status, real progress when the CLI
/// reports it, elapsed time, cancel right here, and the raw activity tucked behind a disclosure.
struct StudioRunningOverlay: View {
    let mode: StudioMode
    let status: String
    let progress: StudioRunProgress?
    let recentLogs: [String]
    let onStop: () -> Void

    @State private var startedAt: Date?
    @State private var showActivity = false

    var body: some View {
        VStack(spacing: MereRunTheme.Spacing.md) {
            Image(systemName: mode.systemImage)
                .font(.system(size: 30, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
                .symbolEffect(.pulse, options: .repeating)

            Text(statusHeadline)
                .font(.system(size: 17, weight: .semibold))
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)

            if let progress, let fraction = progress.fractionCompleted {
                VStack(spacing: 6) {
                    ProgressView(value: fraction)
                        .progressViewStyle(.linear)
                        .tint(MereRunTheme.accent)
                        .frame(width: 320)
                    HStack {
                        Text(progress.label)
                            .lineLimit(1)
                        Spacer()
                        Text("\(Int((fraction * 100).rounded()))%")
                            .monospacedDigit()
                    }
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(width: 320)
                    if let detail = progress.detail {
                        Text(detail)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .lineLimit(1)
                    }
                }
            } else {
                ProgressView()
                    .controlSize(.small)
                if let detail = progress?.detail {
                    Text(detail)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                }
            }

            HStack(spacing: MereRunTheme.Spacing.sm) {
                elapsedLabel
                Button("Cancel", action: onStop)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Stop this run (⌘.)")
            }

            if !recentLogs.isEmpty {
                DisclosureGroup(isExpanded: $showActivity) {
                    VStack(alignment: .leading, spacing: 5) {
                        ForEach(Array(recentLogs.enumerated()), id: \.offset) { _, line in
                            Text(line)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(MereRunTheme.textMuted)
                                .lineLimit(2)
                        }
                    }
                    .frame(width: 420, alignment: .leading)
                    .padding(.top, 6)
                } label: {
                    Text("Activity")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                .disclosureGroupStyle(.automatic)
                .frame(width: 440)
            }
        }
        .padding(MereRunTheme.Spacing.xxl)
        .merePanel(cornerRadius: MereRunTheme.Radius.xl)
        .mereShadow(radius: 28, y: 12)
        .onAppear { if startedAt == nil { startedAt = Date() } }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Running: \(statusHeadline)")
    }

    private var statusHeadline: String {
        if progress?.fractionCompleted == nil, let detail = progress?.detail, !detail.isEmpty {
            return status
        }
        return status
    }

    @ViewBuilder
    private var elapsedLabel: some View {
        if let startedAt {
            TimelineView(.periodic(from: startedAt, by: 1)) { context in
                Text(StudioTimeFormat.string(context.date.timeIntervalSince(startedAt)))
                    .font(MereRunTheme.captionFont)
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
    }
}

/// Blocking states (no model, missing CLI, run errors) presented as a calm card with the
/// one action that fixes it front and center.
struct StudioReadinessOverlay: View {
    let title: String
    let message: String
    let canPull: Bool
    let isChecking: Bool
    let onPullModel: () -> Void
    let onShowDetails: () -> Void

    var body: some View {
        VStack(spacing: MereRunTheme.Spacing.md) {
            Image(systemName: statusImage)
                .font(.system(size: 26, weight: .semibold))
                .foregroundStyle(statusColor)
                .frame(width: 64, height: 64)
                .background {
                    Circle().fill(statusColor.opacity(0.12))
                }

            Text(title)
                .font(.system(size: 20, weight: .semibold))
            Text(message)
                .font(MereRunTheme.bodyFont)
                .foregroundStyle(MereRunTheme.textSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 320)

            HStack(spacing: MereRunTheme.Spacing.sm) {
                if canPull {
                    Button {
                        onPullModel()
                    } label: {
                        Label("Get the model", systemImage: "arrow.down.circle.fill")
                    }
                    .buttonStyle(.merePrimary)
                }
                Button {
                    onShowDetails()
                } label: {
                    Label("Details", systemImage: "terminal")
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(MereRunTheme.Spacing.xxl)
        .merePanel(cornerRadius: MereRunTheme.Radius.xl)
        .mereShadow(radius: 30, y: 12)
    }

    private var statusImage: String {
        if isChecking { return "hourglass" }
        return canPull ? "arrow.down.circle" : "exclamationmark.triangle"
    }

    private var statusColor: Color {
        if isChecking { return MereRunTheme.yellow }
        return canPull ? MereRunTheme.yellow : MereRunTheme.red
    }
}

/// The finished-work stage: a big preview you can Quick Look, drag straight to Finder, copy,
/// or open — with the run's identity in one quiet meta row.
struct StudioOutputView: View {
    let item: StudioLibraryItem
    let liveOutputText: String?
    let onOpen: () -> Void
    let onReveal: () -> Void
    let onQuickLook: () -> Void
    var onCopy: ((String) -> Void)?
    var onRerun: (() -> Void)?
    var onEdit: (() -> Void)?

    @State private var copied = false
    @State private var selectedArtifactURL: URL?

    private var activeArtifactURL: URL? {
        if let selectedArtifactURL, item.allArtifactURLs.contains(selectedArtifactURL) {
            return selectedArtifactURL
        }
        let files = item.allArtifactURLs.filter {
            (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }
        if let outputURL = item.outputURL, files.contains(outputURL) {
            return outputURL
        }
        let preferredKinds: [StudioOutputFileKind] = [.video, .image, .model3D, .audio, .text]
        for kind in preferredKinds {
            if let artifact = files.first(where: { StudioOutputFileKind.classify($0) == kind }) {
                return artifact
            }
        }
        if let outputURL = item.outputURL,
           (try? outputURL.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
            return outputURL
        }
        return files.first
    }

    var body: some View {
        VStack(spacing: MereRunTheme.Spacing.md) {
            outputPreview
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(MereRunTheme.surface.opacity(0.45))
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.xl))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.xl)
                        .strokeBorder(MereRunTheme.border.opacity(0.65), lineWidth: 1)
                }
                .contextMenu { previewContextMenu }

            if item.allArtifactURLs.count > 1 {
                artifactStrip
            }

            HStack(spacing: MereRunTheme.Spacing.sm) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(item.displayTitle)
                        .font(.system(size: 15, weight: .semibold))
                        .lineLimit(1)
                    HStack(spacing: 5) {
                        Text(item.displayKindTitle)
                        Text("·")
                        Text(item.updatedAt, format: .relative(presentation: .named))
                    }
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                }

                Spacer()

                if item.status != .completed {
                    statusBadge
                }

                if canCopy {
                    Button {
                        copyOutput()
                    } label: {
                        Image(systemName: copied ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 13, weight: .semibold))
                            .frame(width: 30, height: 30)
                    }
                    .buttonStyle(.mereIcon)
                    .help(copied ? "Copied" : "Copy output")
                    .accessibilityLabel("Copy output")
                }

                if activeArtifactURL != nil {
                    Button {
                        if let activeArtifactURL {
                            QuickLookCoordinator.shared.preview(activeArtifactURL)
                        } else {
                            onQuickLook()
                        }
                    } label: {
                        Image(systemName: "eye")
                            .font(.system(size: 13, weight: .semibold))
                            .frame(width: 30, height: 30)
                    }
                    .buttonStyle(.mereIcon)
                    .help("Quick Look (Space)")
                    .accessibilityLabel("Quick Look")

                    Button {
                        if let activeArtifactURL {
                            NSWorkspace.shared.activateFileViewerSelecting([activeArtifactURL])
                        } else {
                            onReveal()
                        }
                    } label: {
                        Image(systemName: "folder")
                            .font(.system(size: 13, weight: .semibold))
                            .frame(width: 30, height: 30)
                    }
                    .buttonStyle(.mereIcon)
                    .help("Reveal in Finder")
                    .accessibilityLabel("Reveal in Finder")

                    Button("Open") {
                        if let activeArtifactURL {
                            NSWorkspace.shared.open(activeArtifactURL)
                        } else {
                            onOpen()
                        }
                    }
                        .buttonStyle(.mereSecondary)
                }

                if let onRerun {
                    Button {
                        onRerun()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 13, weight: .semibold))
                            .frame(width: 30, height: 30)
                    }
                    .buttonStyle(.mereIcon)
                    .help("Run again")
                }

                if let onEdit {
                    Button("Edit", action: onEdit)
                        .buttonStyle(.mereSecondary)
                }
            }
        }
        // Cap to a comfortable reading measure and center, so a single result
        // doesn't stretch into a lonely island on very wide windows.
        .frame(maxWidth: 900)
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    private var previewContextMenu: some View {
        if let activeArtifactURL {
            Button("Open") { NSWorkspace.shared.open(activeArtifactURL) }
            Button("Quick Look") { QuickLookCoordinator.shared.preview(activeArtifactURL) }
            Button("Reveal in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([activeArtifactURL])
            }
            if canCopy {
                Button("Copy") { copyOutput() }
            }
        } else if canCopy {
            Button("Copy") { copyOutput() }
        }
    }

    @ViewBuilder
    private var outputPreview: some View {
        if let url = activeArtifactURL {
            filePreview(url)
                .onDrag { NSItemProvider(contentsOf: url) ?? NSItemProvider() }
        } else if let text = liveOutputText ?? item.outputText,
                  !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            ScrollView {
                Text(text)
                    .font(item.mode == .chat ? MereRunTheme.bodyFont : MereRunTheme.monoFont)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(MereRunTheme.Spacing.xl)
            }
        } else {
            VStack(spacing: MereRunTheme.Spacing.sm) {
                Image(systemName: "doc")
                    .font(.system(size: 42, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
                Text(item.status == .failed ? "Run did not produce a file." : "Output will appear here.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
    }

    private var artifactStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(item.allArtifactURLs, id: \.self) { url in
                    Button {
                        selectedArtifactURL = url
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: iconName(for: url))
                                .font(.system(size: 11, weight: .semibold))
                            Text(url.lastPathComponent)
                                .font(MereRunTheme.captionFont)
                                .lineLimit(1)
                        }
                        .foregroundStyle(activeArtifactURL == url ? MereRunTheme.accent : MereRunTheme.textSecondary)
                        .padding(.horizontal, 10)
                        .frame(height: 30)
                        .background {
                            Capsule()
                                .fill(activeArtifactURL == url ? MereRunTheme.accentSoft : MereRunTheme.surface)
                                .overlay {
                                    Capsule().strokeBorder(MereRunTheme.border.opacity(0.6), lineWidth: 1)
                                }
                        }
                    }
                    .buttonStyle(.plain)
                    .help(url.path)
                }
            }
        }
        .accessibilityLabel("Run artifacts")
    }

    @ViewBuilder
    private func filePreview(_ url: URL) -> some View {
        switch StudioOutputFileKind.classify(url) {
        case .image:
            StudioAsyncImagePreview(
                url: url,
                maxPixelSize: 2_200,
                contentMode: .fit,
                fallbackSystemImage: iconName(for: url)
            )
            .padding(MereRunTheme.Spacing.xl)
        case .audio:
            StudioAudioPlayerView(url: url)
        case .video:
            StudioVideoPlayerView(url: url)
        case .text:
            StudioTextFilePreview(url: url)
        case .model3D:
            StudioEmbeddedQuickLookPreview(url: url)
        case .other:
            filePlaceholder(for: url)
        }
    }

    private func filePlaceholder(for url: URL) -> some View {
        VStack(spacing: MereRunTheme.Spacing.sm) {
            Image(systemName: iconName(for: url))
                .font(.system(size: 50, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
            Text(url.lastPathComponent)
                .font(.system(size: 15, weight: .semibold))
                .lineLimit(1)
            Text(url.deletingLastPathComponent().path)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .padding(MereRunTheme.Spacing.xl)
    }

    private var statusBadge: some View {
        Text(item.status.rawValue.capitalized)
            .font(MereRunTheme.captionFont)
            .foregroundStyle(statusColor)
            .padding(.horizontal, 10)
            .frame(height: 26)
            .background {
                Capsule().fill(statusColor.opacity(0.14))
            }
    }

    private var statusColor: Color {
        switch item.status {
        case .queued: return MereRunTheme.textMuted
        case .running: return MereRunTheme.yellow
        case .completed: return MereRunTheme.green
        case .failed: return MereRunTheme.red
        }
    }

    private var canCopy: Bool {
        if let url = activeArtifactURL {
            let kind = StudioOutputFileKind.classify(url)
            return kind == .image || kind == .text
        }
        let text = (liveOutputText ?? item.outputText)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return !text.isEmpty
    }

    private func copyOutput() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        if let url = activeArtifactURL {
            pasteboard.writeObjects([url as NSURL])
        } else if let text = liveOutputText ?? item.outputText {
            pasteboard.setString(text, forType: .string)
        }
        copied = true
        Task {
            try? await Task.sleep(nanoseconds: 1_400_000_000)
            copied = false
        }
    }

    private func iconName(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "wav", "mp3", "m4a": return "waveform"
        case "mp4", "mov": return "film"
        case "json": return "curlybraces"
        case "glb", "gltf", "obj", "ply", "stl", "usdz": return "cube.transparent"
        case "safetensors": return "shippingbox"
        default: return "doc"
        }
    }
}

enum StudioImageContentMode: Equatable {
    case fit
    case fill
}

enum StudioImageLoadState {
    case loading
    case loaded(NSImage)
    case unavailable
}

/// Downsampled, off-main image preview with a gentle entrance once decoded.
struct StudioAsyncImagePreview: View {
    let url: URL
    let maxPixelSize: CGFloat
    let contentMode: StudioImageContentMode
    let fallbackSystemImage: String

    @State private var loadState = StudioImageLoadState.loading

    private var stateKey: Int {
        switch loadState {
        case .loading: return 0
        case .loaded: return 1
        case .unavailable: return 2
        }
    }

    var body: some View {
        Group {
            switch loadState {
            case .loading:
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .loaded(let image):
                Group {
                    if contentMode == .fill {
                        Image(nsImage: image)
                            .resizable()
                            .scaledToFill()
                    } else {
                        Image(nsImage: image)
                            .resizable()
                            .scaledToFit()
                    }
                }
                .transition(.opacity.combined(with: .scale(scale: 0.985)))
            case .unavailable:
                Image(systemName: fallbackSystemImage)
                    .font(.system(size: 32, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .animation(MereRunTheme.Motion.gentleSpring, value: stateKey)
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

struct StudioTextFilePreview: View {
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
                    .padding(MereRunTheme.Spacing.xl)
            } else if didLoad {
                Text("Text preview unavailable.")
                    .font(MereRunTheme.bodyFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(MereRunTheme.Spacing.xl)
            } else {
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, minHeight: 180)
                    .padding(MereRunTheme.Spacing.xl)
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
