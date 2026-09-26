import AppKit
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

// Batch inputs on screen: the well's stack for a slot holding several files, its list, the Run
// button that counts the runs, the confirmation that names the files that can't run, and the bar
// that follows a batch in flight on its page and in the run queue. What batches, and how a batch
// runs, is StudioKit's (`StudioInputBatch.swift`, `StudioTaskRunner+Batch.swift`).

// MARK: - Well

/// A batching slot holding several files: the first file on a small stack of cards with the
/// count on it. A click lists the files; a drop or paste adds to them.
struct StudioBatchWellTile<Draft: StudioAttachmentDraft>: View {
    let slot: StudioAttachmentSlot
    @Binding var draft: Draft
    /// The disk choice for Add files…: the caller's open panel into the draft.
    let onAddFromDisk: () -> Void

    @State private var showsList = false
    @State private var isDropTargeted = false

    static var side: CGFloat { 48 }
    private static var card: CGFloat { 42 }
    private static var cornerRadius: CGFloat { MereRunTheme.Radius.base }

    private var paths: [String] { slot.runPaths(in: draft) }
    private var countLabel: String { "\(paths.count) files" }

    var body: some View {
        Button { showsList = true } label: { stack }
            .buttonStyle(StudioPressDimButtonStyle())
            .popover(isPresented: $showsList, arrowEdge: .top) {
                StudioBatchFileList(slot: slot, draft: $draft, onAddFromDisk: onAddFromDisk)
            }
            .dropDestination(for: URL.self) { urls, _ in
                let accepted = urls.filter(slot.accepts)
                guard !accepted.isEmpty else { return false }
                slot.attach(accepted, to: &draft)
                return true
            } isTargeted: { targeted in
                withAnimation(MereRunTheme.Motion.quick) { isDropTargeted = targeted }
            }
            .onPasteCommand(of: [.fileURL, .image, .audio]) { _ in
                StudioAttachmentPaste.paste(into: &draft, slots: [slot], allowsText: false)
            }
            .contextMenu {
                Button("Show files…") { showsList = true }
                Button("Add files from disk…", action: onAddFromDisk)
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting(paths.map { URL(fileURLWithPath: $0) })
                }
                Divider()
                Button("Clear") { slot.clear(in: &draft) }
            }
            .help("\(countLabel) · each runs on its own. Click to see or change them.")
            .accessibilityLabel(slot.label)
            .accessibilityValue(countLabel)
            .accessibilityHint("Shows the files. Each runs on its own when you run the task.")
    }

    /// Two cards peeking out behind the first file, and the count on its corner.
    private var stack: some View {
        ZStack(alignment: .bottomLeading) {
            backCard.offset(x: 6, y: -6)
            backCard.offset(x: 3, y: -3)
            StudioBatchThumbnail(path: paths.first ?? "", isFolder: false)
                .frame(width: Self.card, height: Self.card)
                .clipShape(RoundedRectangle(cornerRadius: Self.cornerRadius))
                .overlay {
                    RoundedRectangle(cornerRadius: Self.cornerRadius)
                        .strokeBorder(isDropTargeted ? MereRunTheme.accent : MereRunTheme.border,
                                      style: isDropTargeted ? StrokeStyle(lineWidth: 2, dash: [5, 4]) : StrokeStyle(lineWidth: 1))
                }
        }
        .frame(width: Self.side, height: Self.side, alignment: .bottomLeading)
        .overlay(alignment: .topTrailing) {
            Text("\(paths.count)")
                .font(.system(size: 10.5, weight: .bold))
                .monospacedDigit()
                .foregroundStyle(MereRunTheme.onAccent)
                .padding(.horizontal, 5)
                .frame(minWidth: 18, minHeight: 18)
                .background(Capsule().fill(MereRunTheme.accent))
                .overlay(Capsule().strokeBorder(MereRunTheme.surface, lineWidth: 1.5))
                .offset(x: 4, y: -4)
                .accessibilityHidden(true)
        }
        .contentShape(Rectangle())
    }

    private var backCard: some View {
        RoundedRectangle(cornerRadius: Self.cornerRadius)
            .fill(MereRunTheme.surfaceRaised)
            .overlay {
                RoundedRectangle(cornerRadius: Self.cornerRadius)
                    .strokeBorder(MereRunTheme.border, lineWidth: 1)
            }
            .frame(width: Self.card, height: Self.card)
    }
}

/// A file's picture, or its kind's glyph for sound, video, and anything else.
struct StudioBatchThumbnail: View {
    let path: String
    let isFolder: Bool

    var body: some View {
        let url = URL(fileURLWithPath: path)
        if isFolder {
            glyph("folder")
        } else {
            switch StudioOutputFileKind.classify(url) {
            case .image:
                StudioAsyncImagePreview(url: url, maxPixelSize: 160, contentMode: .fill, fallbackSystemImage: "photo")
            case .audio:
                glyph("waveform")
            case .video:
                glyph("film")
            default:
                glyph("doc")
            }
        }
    }

    private func glyph(_ systemImage: String) -> some View {
        ZStack {
            MereRunTheme.surfaceRaised
            Image(systemName: systemImage)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(MereRunTheme.accent)
        }
    }
}

/// The files of a batch, in the order they run: each with its folder and a remove button, then
/// Add files… and Clear.
struct StudioBatchFileList<Draft: StudioAttachmentDraft>: View {
    let slot: StudioAttachmentSlot
    @Binding var draft: Draft
    let onAddFromDisk: () -> Void

    private enum Metrics {
        static var width: CGFloat { 340 }
        static var maxListHeight: CGFloat { 300 }
        static var rowHeight: CGFloat { 38 }
        static var thumbnail: CGFloat { 26 }
    }

    private var paths: [String] { slot.runPaths(in: draft) }

    var body: some View {
        let paths = paths
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 2) {
                Text("\(paths.count) files")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textPrimary)
                Text("Each runs on its own, in this order.")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 14)
            .padding(.top, 12)
            .padding(.bottom, 8)
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isHeader)
            Divider().overlay(MereRunTheme.border.opacity(0.5))
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(paths.enumerated()), id: \.element) { index, path in
                        row(path, number: index + 1)
                    }
                }
                .padding(.vertical, 4)
            }
            .frame(height: min(Metrics.maxListHeight, CGFloat(paths.count) * Metrics.rowHeight + 8))
            Divider().overlay(MereRunTheme.border.opacity(0.5))
            HStack(spacing: 8) {
                StudioAttachMenu(target: StudioAttachTarget(slot: slot, draft: $draft), chooseFromDisk: onAddFromDisk) { opensMenu in
                    MereSecondaryMenuLabel("Add files…", systemImage: "plus", showsChevron: opensMenu)
                }
                .accessibilityLabel("Add files")
                Spacer(minLength: 8)
                Button("Clear") { slot.clear(in: &draft) }
                    .buttonStyle(.mereSecondary)
                    .help("Remove every file from the batch")
                    .accessibilityLabel("Clear all \(paths.count) files")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .frame(width: Metrics.width)
        .background(MereRunTheme.background)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Batch of \(paths.count) files")
    }

    private func row(_ path: String, number: Int) -> some View {
        let url = URL(fileURLWithPath: path)
        let folder = (url.deletingLastPathComponent().path as NSString).abbreviatingWithTildeInPath
        return HStack(spacing: 10) {
            Text("\(number)")
                .font(.caption.weight(.medium))
                .monospacedDigit()
                .foregroundStyle(MereRunTheme.textMuted)
                .frame(width: 18, alignment: .trailing)
            StudioBatchThumbnail(path: path, isFolder: false)
                .frame(width: Metrics.thumbnail, height: Metrics.thumbnail)
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))
            VStack(alignment: .leading, spacing: 1) {
                Text(url.lastPathComponent)
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(folder)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                    .truncationMode(.head)
            }
            Spacer(minLength: 4)
            Button {
                slot.removeFromBatch(path, in: &draft)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .semibold))
                    .frame(width: 22, height: 22)
            }
            .buttonStyle(.mereIcon)
            .help("Remove from the batch")
            .accessibilityLabel("Remove \(url.lastPathComponent)")
        }
        .padding(.horizontal, 12)
        .frame(height: Metrics.rowHeight)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(number). \(url.lastPathComponent)")
    }
}

// MARK: - Run

/// Run for a batch: the arrow and how many runs it makes, "Run 12".
struct StudioBatchRunButton: View {
    let count: Int
    let isEnabled: Bool
    /// Why Run is blocked, or nil.
    let blockedReason: String?
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: "arrow.up")
                    .font(.system(size: 12, weight: .bold))
                Text("Run \(count)")
                    .font(.system(size: 13, weight: .semibold))
                    .monospacedDigit()
            }
            .foregroundStyle(isEnabled ? MereRunTheme.onAccent : MereRunTheme.textMuted)
            .padding(.horizontal, 13)
            .frame(height: 32)
            .background(Capsule().fill(isEnabled ? MereRunTheme.accent : MereRunTheme.surfaceRaised))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .help(blockedReason ?? "Run once for each of the \(count) files (⌘↩)")
        .accessibilityLabel("Run \(count) files")
        .accessibilityHint(blockedReason ?? "Runs the task once for each file")
        .keyboardShortcut(.return, modifiers: .command)
    }
}

extension View {
    /// Asks before running a batch some of whose files can't run: which ones and why, then Skip
    /// and run the rest, or Cancel.
    func studioBatchConfirmation(_ review: Binding<StudioBatchReview?>, onRun: @escaping ([String]) -> Void) -> some View {
        alert(
            review.wrappedValue?.confirmationTitle ?? "",
            isPresented: Binding(get: { review.wrappedValue != nil }, set: { if !$0 { review.wrappedValue = nil } }),
            presenting: review.wrappedValue
        ) { pending in
            Button("Cancel", role: .cancel) {}
            Button(pending.skipTitle) { onRun(pending.runnable) }
                .keyboardShortcut(.defaultAction)
        } message: { pending in
            Text(pending.confirmationMessage())
        }
    }
}

/// What Run does with a checked batch: run it, ask about the files that can't run (by setting
/// `pending`, which `studioBatchConfirmation` shows), or answer why none can.
@MainActor
enum StudioBatchLaunch {
    static func start(_ review: StudioBatchReview, pending: inout StudioBatchReview?, run: ([String]) -> Void) -> String? {
        switch review.decision {
        case .runAll:
            run(review.runnable)
            return nil
        case .confirmSkipping:
            pending = review
            return nil
        case .refuse(let message):
            return message
        }
    }

    /// The banner for files that passed the checks but could not be prepared when their turn came.
    static func failureMessage(_ submission: StudioBatchSubmission) -> String? {
        guard let first = submission.failures.first else { return nil }
        let others = submission.failures.count - 1
        return "\(first.fileName) couldn't start: \(first.problem ?? "")" + (others > 0 ? " (and \(others) more)" : "")
    }
}

// MARK: - Following a batch

/// The bar over a page's composer while one of its batches is in flight: what it is, how far it
/// has come, and Stop batch.
struct StudioBatchStatusBar: View {
    let progress: StudioBatchProgress
    let onStop: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "square.stack.3d.down.right")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(MereRunTheme.accent)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 8) {
                    Text(progress.title)
                        .font(.callout.weight(.medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                    Text(progress.summary)
                        .font(.caption.weight(.medium))
                        .monospacedDigit()
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                }
                StudioProgressBar(fraction: progress.fractionFinished)
            }
            Button("Stop batch", action: onStop)
                .buttonStyle(.mereSecondary)
                .help("Take the waiting files out of the queue and stop the ones running")
                .accessibilityLabel("Stop batch, \(progress.total - progress.finished) files left")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .strokeBorder(MereRunTheme.border, lineWidth: 1)
                }
        }
        .padding(.horizontal, 24)
        .padding(.top, 8)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(progress.title), \(progress.summary)")
    }
}

/// A batch in the run queue: its task and size, how far it has come, and Stop batch.
struct StudioBatchQueueRow: View {
    let progress: StudioBatchProgress
    let onStop: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            ZStack {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                    .fill(MereRunTheme.accentSoft)
                Image(systemName: "square.stack.3d.down.right")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.accent)
            }
            .frame(width: 26, height: 26)
            .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 4) {
                Text(progress.title)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                StudioProgressBar(fraction: progress.fractionFinished)
                Text(progress.summary)
                    .font(.caption.weight(.medium))
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Button("Stop batch", action: onStop)
                .buttonStyle(.mereSecondary)
                .help("Take the batch's waiting files out of the queue and stop the ones running")
                .accessibilityLabel("Stop \(progress.title)")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 7)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(progress.title), \(progress.summary)")
    }
}
