import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// What a feed card can ask the workspace to do.
struct StudioFeedActions {
    /// Re-run with a fresh seed.
    let vary: (StudioLibraryItem) -> Void
    /// Re-run with the same command.
    let rerun: (StudioLibraryItem) -> Void
    /// Load an output into the composer's well.
    let useAsInput: (URL) -> Void
    /// Copy an output to a place the user picks.
    let saveTo: (URL) -> Void
    let cancel: (Job) -> Void
    /// Take a queued run out of the queue (or drop a stale queued row).
    let remove: (StudioFeedCard) -> Void
    let retry: (StudioLibraryItem) -> Void
    let delete: (StudioLibraryItem) -> Void
    let pullModel: () -> Void
    let showDetails: () -> Void
    let useExample: (String) -> Void
    let attach: () -> Void
}

/// The feed of generations for a prompt mode: oldest at the top, the newest just above the
/// composer, each run its own card with its own progress, Cancel, or Remove. The readiness
/// state is a card at the bottom rather than an overlay, so it never hides earlier work.
struct StudioFeedCanvas: View {
    let mode: StudioMode
    let cards: [StudioFeedCard]
    let readiness: ModelReadinessState
    /// The `model pull` the readiness card reports, while one runs for this mode's model.
    let pullJob: Job?
    /// A card to scroll to and briefly outline: the Library row the user just picked.
    let highlightedID: UUID?
    /// A run that finished while its card was off-screen; cleared once the card is seen.
    @Binding var newResultID: UUID?
    let actions: StudioFeedActions

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var visibleCardIDs: Set<UUID> = []
    @State private var scrollTarget: UUID?

    private enum Metrics {
        static let cardSpacing: CGFloat = 14
        static let insets = EdgeInsets(top: 22, leading: 24, bottom: 6, trailing: 24)
    }

    private var showsReadinessCard: Bool {
        readiness.blocksRun || pullJob != nil
    }

    private var pendingNewResultIsOffscreen: Bool {
        guard let newResultID else { return false }
        return !visibleCardIDs.contains(newResultID)
    }

    var body: some View {
        Group {
            if cards.isEmpty && !showsReadinessCard {
                StudioEmptyState(mode: mode, onUseExample: actions.useExample, onAttach: actions.attach)
                    .padding(MereRunTheme.Spacing.xxxl)
            } else {
                feed
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onChange(of: highlightedID) { _, id in
            guard let id else { return }
            scrollTarget = id
        }
        .onChange(of: visibleCardIDs) { _, visible in
            if let newResultID, visible.contains(newResultID) {
                self.newResultID = nil
            }
        }
        .onChange(of: newResultID) { _, id in
            // A result that finished in view needs no pill.
            if let id, visibleCardIDs.contains(id) {
                newResultID = nil
            }
        }
    }

    private var feed: some View {
        ScrollViewReader { proxy in
            GeometryReader { container in
                ScrollView {
                    VStack(spacing: Metrics.cardSpacing) {
                        if cards.isEmpty {
                            StudioEmptyState(mode: mode, onUseExample: actions.useExample, onAttach: actions.attach)
                                .padding(.vertical, MereRunTheme.Spacing.xl)
                        }
                        ForEach(cards) { card in
                            cardView(card)
                                .id(card.id)
                                .background {
                                    GeometryReader { geometry in
                                        Color.clear.preference(
                                            key: StudioFeedCardFramesKey.self,
                                            value: [card.id: geometry.frame(in: .named("feed"))]
                                        )
                                    }
                                }
                        }
                        if showsReadinessCard {
                            StudioReadinessCard(
                                readiness: readiness,
                                pullJob: pullJob,
                                onPullModel: actions.pullModel,
                                onShowDetails: actions.showDetails,
                                onCancelPull: { job in actions.cancel(job) }
                            )
                            .id(StudioReadinessCard.feedID)
                        }
                    }
                    .padding(Metrics.insets)
                    .frame(maxWidth: StudioLayoutPolicy.feedMaxWidth)
                    .frame(maxWidth: .infinity)
                    // The composer sits just below; the newest card should end up beside it.
                    .frame(minHeight: container.size.height, alignment: .bottom)
                }
                .coordinateSpace(name: "feed")
                .defaultScrollAnchor(.bottom)
                .onPreferenceChange(StudioFeedCardFramesKey.self) { frames in
                    let bounds = CGRect(origin: .zero, size: container.size)
                    let visible = Set(frames.filter { $0.value.intersects(bounds) }.map(\.key))
                    if visible != visibleCardIDs { visibleCardIDs = visible }
                }
            }
            .overlay(alignment: .bottom) {
                if pendingNewResultIsOffscreen, let newResultID {
                    StudioNewResultPill {
                        scrollTarget = newResultID
                    }
                    .padding(.bottom, 10)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .onChange(of: scrollTarget) { _, target in
                guard let target else { return }
                if reduceMotion {
                    proxy.scrollTo(target, anchor: .center)
                } else {
                    withAnimation(MereRunTheme.Motion.standard) { proxy.scrollTo(target, anchor: .center) }
                }
                scrollTarget = nil
            }
            .animation(reduceMotion ? nil : MereRunTheme.Motion.standard, value: pendingNewResultIsOffscreen)
        }
    }

    @ViewBuilder
    private func cardView(_ card: StudioFeedCard) -> some View {
        let highlighted = highlightedID == card.id
        switch card.kind {
        case .generation:
            StudioGenerationCard(
                mode: mode,
                item: card.item,
                isHighlighted: highlighted,
                actions: actions
            )
        case .running:
            if let job = card.job {
                StudioRunningCard(item: card.item, job: job, isHighlighted: highlighted) {
                    actions.cancel(job)
                }
            } else {
                // The store no longer has the job (a row from an earlier launch): it cannot be
                // running any more, so offer the same recovery as a failure.
                StudioFailureCard(item: card.item, job: nil, isHighlighted: highlighted, actions: actions)
            }
        case .queued:
            StudioQueuedRow(
                item: card.item,
                position: StudioFeedCardBuilder.queuePosition(of: card, in: cards) ?? 0,
                isHighlighted: highlighted
            ) {
                actions.remove(card)
            }
        case .failed:
            StudioFailureCard(item: card.item, job: card.job, isHighlighted: highlighted, actions: actions)
        }
    }
}

private struct StudioFeedCardFramesKey: PreferenceKey {
    static let defaultValue: [UUID: CGRect] = [:]

    static func reduce(value: inout [UUID: CGRect], nextValue: () -> [UUID: CGRect]) {
        value.merge(nextValue(), uniquingKeysWith: { _, new in new })
    }
}

// MARK: - Card chrome

/// The board's `panel`: `surface` on a 1pt `border` at 80%, radius 10. Running cards tint the
/// border accent; a just-picked Library row outlines its card for a moment.
private struct StudioFeedPanel: ViewModifier {
    var borderColor: Color = MereRunTheme.border.opacity(0.8)
    var isHighlighted = false

    func body(content: Content) -> some View {
        content
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .fill(MereRunTheme.surface)
                    .overlay {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                            .strokeBorder(
                                isHighlighted ? MereRunTheme.accent.opacity(0.7) : borderColor,
                                lineWidth: isHighlighted ? 1.5 : 1
                            )
                    }
            }
            .animation(MereRunTheme.Motion.standard, value: isHighlighted)
    }
}

private extension View {
    func feedPanel(borderColor: Color = MereRunTheme.border.opacity(0.8), isHighlighted: Bool = false) -> some View {
        modifier(StudioFeedPanel(borderColor: borderColor, isHighlighted: isHighlighted))
    }
}

/// Prompt (13.5 medium) with the time on the right (11.5 muted), then the run's parameter chips.
private struct StudioCardHeader: View {
    let item: StudioLibraryItem
    let when: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                Text(item.displayTitle)
                    .font(.system(size: 13.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                Text(when)
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                    .fixedSize()
            }
            let chips = StudioFeedChips.chips(for: item)
            if !chips.isEmpty {
                HStack(spacing: 6) {
                    ForEach(chips, id: \.self) { chip in
                        Text(chip)
                            .font(.system(size: 10.5, weight: .medium))
                            .foregroundStyle(MereRunTheme.textPrimary)
                            .lineLimit(1)
                            .padding(.horizontal, 7)
                            .frame(height: 20)
                            .background {
                                RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                                    .fill(MereRunTheme.surfaceRaised)
                            }
                    }
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel("Settings: \(chips.joined(separator: ", "))")
            }
        }
    }
}

/// "12:43 PM" today, "Jul 11" on another day, "now" for a run in flight.
enum StudioFeedTime {
    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        return formatter
    }()

    private static let dayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.setLocalizedDateFormatFromTemplate("MMM d")
        return formatter
    }()

    static func label(for date: Date, now: Date = Date(), calendar: Calendar = .current) -> String {
        calendar.isDate(date, inSameDayAs: now) ? timeFormatter.string(from: date) : dayFormatter.string(from: date)
    }
}

/// The parameter chips a card shows under its prompt, read from the run's own command draft so
/// a card always says what actually ran.
enum StudioFeedChips {
    static func chips(for item: StudioLibraryItem) -> [String] {
        guard let draft = item.commandDraft else { return [] }
        var chips: [String] = []
        switch item.mode {
        case .createImage, .video:
            chips.append("\(draft.width)×\(draft.height)")
            if item.mode == .video {
                chips.append(draft.useDuration
                    ? "\(StudioComposerPresets.secondsText(draft.durationSeconds)) s"
                    : "\(draft.numFrames) frames")
            }
            chips.append(stepsChip(draft.steps))
            chips.append(seedChip(draft.seed))
        case .music:
            if draft.useDuration { chips.append("\(StudioComposerPresets.secondsText(draft.durationSeconds)) s") }
            if draft.musicOverrideSteps { chips.append(stepsChip(draft.steps)) }
            chips.append(seedChip(draft.seed))
        case .sfx:
            chips.append("\(StudioComposerPresets.secondsText(draft.durationSeconds)) s")
            chips.append(stepsChip(draft.steps))
            chips.append(seedChip(draft.seed))
        case .findObjects, .segment, .track:
            chips.append("threshold \(StudioComposerPresets.decimalText(draft.visionThreshold))")
        case .speak:
            chips.append(draft.voiceMode == "clone" ? "cloned voice" : "preset voice")
        case .listen:
            if !draft.language.isBlank, draft.language != "auto" { chips.append(draft.language) }
        case .chat, .code, .readImage:
            break
        }
        if !draft.model.isBlank {
            chips.append(StudioComposer.displayModelName(draft.model))
        }
        return chips
    }

    private static func stepsChip(_ steps: Int) -> String {
        steps == 1 ? "1 step" : "\(steps) steps"
    }

    private static func seedChip(_ seed: String) -> String {
        let trimmed = seed.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "seed random" : "seed \(trimmed)"
    }
}

// MARK: - Generation card

/// A finished run: prompt, chips, every output in a grid, and the actions that continue from it.
struct StudioGenerationCard: View {
    let mode: StudioMode
    let item: StudioLibraryItem
    let isHighlighted: Bool
    let actions: StudioFeedActions

    @State private var copied = false

    static let tileSide: CGFloat = 236

    private var files: [URL] {
        item.allArtifactURLs.filter {
            (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }
    }

    private var mediaFiles: [URL] {
        files.filter { [.image, .video, .audio, .model3D].contains(StudioOutputFileKind.classify($0)) }
    }

    private var textFiles: [URL] {
        files.filter { StudioOutputFileKind.classify($0) == .text }
    }

    private var sidecars: [URL] {
        files.filter { StudioOutputFileKind.classify($0) == .other }
    }

    private var primaryURL: URL? {
        if let outputURL = item.outputURL, files.contains(outputURL) { return outputURL }
        return mediaFiles.first ?? textFiles.first ?? files.first
    }

    private var outputText: String? {
        guard let text = item.outputText?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
            return nil
        }
        return text
    }

    private var canUseAsInput: Bool {
        guard let primaryURL else { return false }
        return mode.attachmentSlots.contains { $0.accepts(primaryURL) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            StudioCardHeader(item: item, when: StudioFeedTime.label(for: item.createdAt))
            outputs
            actionRow
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 16)
        .feedPanel(isHighlighted: isHighlighted)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Generation: \(item.displayTitle)")
    }

    @ViewBuilder
    private var outputs: some View {
        if !mediaFiles.isEmpty {
            StudioOutputGrid(urls: mediaFiles, tileSide: Self.tileSide)
        }
        ForEach(textFiles, id: \.self) { url in
            StudioTextFilePreview(url: url)
                .frame(maxHeight: 240)
                .background(MereRunTheme.surfaceRaised.opacity(0.6))
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
        }
        if mediaFiles.isEmpty, textFiles.isEmpty, let outputText {
            StudioMarkdownText(content: outputText, bodyFont: .system(size: 13))
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(MereRunTheme.surfaceRaised.opacity(0.6))
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
        }
        if !sidecars.isEmpty {
            HStack(spacing: 6) {
                ForEach(sidecars, id: \.self) { url in
                    Button {
                        QuickLookCoordinator.shared.preview(url)
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "doc")
                                .font(.system(size: 9, weight: .semibold))
                            Text(url.lastPathComponent)
                                .font(.system(size: 10.5, weight: .medium))
                                .lineLimit(1)
                        }
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .padding(.horizontal, 7)
                        .frame(height: 20)
                        .background {
                            RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                                .fill(MereRunTheme.surfaceRaised)
                        }
                    }
                    .buttonStyle(.plain)
                    .help(url.path)
                    .onDrag { NSItemProvider(contentsOf: url) ?? NSItemProvider() }
                }
            }
        }
    }

    private var actionRow: some View {
        HStack(spacing: 2) {
            if item.commandDraft != nil, item.templateID != nil {
                cardIcon("shuffle", help: "Vary — run again with a new seed") { actions.vary(item) }
                cardIcon("arrow.clockwise", help: "Rerun") { actions.rerun(item) }
            }
            if let primaryURL {
                cardIcon("arrow.right.to.line", help: "Use as input") { actions.useAsInput(primaryURL) }
                    .disabled(!canUseAsInput)
                    .opacity(canUseAsInput ? 1 : 0.4)
                cardIcon("eye", help: "Quick Look (Space)") { QuickLookCoordinator.shared.preview(primaryURL) }
                cardIcon("folder", help: "Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([primaryURL])
                }
            }
            if primaryURL != nil || outputText != nil {
                cardIcon(copied ? "checkmark" : "doc.on.doc", help: copied ? "Copied" : "Copy") { copyOutput() }
            }
            Spacer(minLength: 8)
            if let primaryURL {
                Button("Save to…") { actions.saveTo(primaryURL) }
                    .buttonStyle(.mereSecondary)
                    .help("Save a copy of the output somewhere else")
            }
        }
        .contextMenu {
            if item.commandDraft != nil, item.templateID != nil {
                Button("Vary") { actions.vary(item) }
                Button("Rerun") { actions.rerun(item) }
            }
            if let primaryURL {
                Button("Open") { NSWorkspace.shared.open(primaryURL) }
                Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([primaryURL]) }
            }
            Divider()
            Button("Delete", role: .destructive) { actions.delete(item) }
        }
    }

    private func cardIcon(_ systemImage: String, help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 13, weight: .medium))
                .frame(width: 28, height: 28)
        }
        .buttonStyle(.mereIcon)
        .help(help)
        .accessibilityLabel(help)
    }

    private func copyOutput() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        if let primaryURL {
            pasteboard.writeObjects([primaryURL as NSURL])
        } else if let outputText {
            pasteboard.setString(outputText, forType: .string)
        }
        copied = true
        Task {
            try? await Task.sleep(nanoseconds: 1_400_000_000)
            copied = false
        }
    }
}

/// 236pt tiles (radius 8) for images, video, and 3D; audio takes a full-width waveform player.
private struct StudioOutputGrid: View {
    let urls: [URL]
    let tileSide: CGFloat

    private var tiled: [URL] {
        urls.filter { StudioOutputFileKind.classify($0) != .audio }
    }

    private var audio: [URL] {
        urls.filter { StudioOutputFileKind.classify($0) == .audio }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !tiled.isEmpty {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: tileSide, maximum: tileSide), spacing: 10, alignment: .leading)],
                    alignment: .leading,
                    spacing: 10
                ) {
                    ForEach(tiled, id: \.self) { url in
                        tile(url)
                    }
                }
            }
            ForEach(audio, id: \.self) { url in
                StudioAudioPlayerView(url: url)
                    .padding(.vertical, 12)
                    .frame(maxWidth: .infinity)
                    .background(MereRunTheme.surfaceRaised.opacity(0.6))
                    .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
                    .onDrag { NSItemProvider(contentsOf: url) ?? NSItemProvider() }
            }
        }
    }

    @ViewBuilder
    private func tile(_ url: URL) -> some View {
        Group {
            switch StudioOutputFileKind.classify(url) {
            case .image:
                StudioAsyncImagePreview(url: url, maxPixelSize: 720, contentMode: .fill, fallbackSystemImage: "photo")
            case .video:
                StudioVideoPlayerView(url: url)
            case .model3D:
                StudioEmbeddedQuickLookPreview(url: url)
            default:
                Image(systemName: "doc")
                    .font(.system(size: 28, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .frame(width: tileSide, height: tileSide)
        .background(MereRunTheme.surfaceRaised)
        .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
        .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
        .onTapGesture { QuickLookCoordinator.shared.preview(url) }
        .onDrag { NSItemProvider(contentsOf: url) ?? NSItemProvider() }
        .contextMenu {
            Button("Open") { NSWorkspace.shared.open(url) }
            Button("Quick Look") { QuickLookCoordinator.shared.preview(url) }
            Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
        }
        .help(url.lastPathComponent)
        .accessibilityLabel("Output \(url.lastPathComponent)")
        .accessibilityHint("Click for Quick Look; drag to copy the file out")
    }
}

// MARK: - Running card

/// A run in flight. Observes its `Job` directly: progress, status, and the log tail update here
/// without the workspace re-rendering.
struct StudioRunningCard: View {
    let item: StudioLibraryItem
    @ObservedObject var job: Job
    let isHighlighted: Bool
    let onCancel: () -> Void

    @State private var showActivity = false

    private var fraction: Double? { job.progress?.fractionCompleted }

    private var statusText: String {
        StudioRunningStatus.text(progress: job.progress, fallback: job.status)
    }

    private var logTail: [LogLine] {
        Array(job.log.lines.suffix(6))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            StudioCardHeader(item: item, when: "now")

            HStack(spacing: 12) {
                StudioProgressBar(fraction: fraction)
                    .frame(maxWidth: .infinity)
                HStack(spacing: 0) {
                    Text(statusText)
                    elapsed
                }
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
                .fixedSize()
                Button("Cancel", action: onCancel)
                    .buttonStyle(.mereSecondary)
                    .help("Stop this run (⌘.)")
            }

            VStack(alignment: .leading, spacing: 6) {
                Button {
                    withAnimation(MereRunTheme.Motion.quick) { showActivity.toggle() }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: showActivity ? "chevron.down" : "chevron.right")
                            .font(.system(size: 9, weight: .semibold))
                        Text("Activity")
                            .font(.system(size: 11.5, weight: .medium))
                    }
                    .foregroundStyle(MereRunTheme.textMuted)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(showActivity ? "Hide activity" : "Show activity")
                if showActivity {
                    VStack(alignment: .leading, spacing: 3) {
                        if logTail.isEmpty {
                            Text("Waiting for the first line of output…")
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(MereRunTheme.textMuted)
                        }
                        ForEach(logTail) { line in
                            Text(line.text)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(MereRunTheme.textMuted)
                                .lineLimit(2)
                                .textSelection(.enabled)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 16)
        .feedPanel(borderColor: MereRunTheme.accent.opacity(0.4), isHighlighted: isHighlighted)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Running: \(item.displayTitle), \(statusText)")
    }

    @ViewBuilder
    private var elapsed: some View {
        if let startedAt = job.startedAt {
            TimelineView(.periodic(from: startedAt, by: 1)) { context in
                Text(" · \(StudioTimeFormat.string(context.date.timeIntervalSince(startedAt)))")
                    .monospacedDigit()
            }
        }
    }
}

/// A 4pt track on `surfaceRaised` with an accent fill; indeterminate progress shows a soft
/// pulse instead of a false percentage.
struct StudioProgressBar: View {
    let fraction: Double?
    var tint: Color = MereRunTheme.accent

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(MereRunTheme.surfaceRaised)
                if let fraction {
                    Capsule()
                        .fill(tint)
                        .frame(width: max(4, geometry.size.width * min(max(fraction, 0), 1)))
                        .animation(reduceMotion ? nil : MereRunTheme.Motion.standard, value: fraction)
                } else {
                    Capsule()
                        .fill(tint.opacity(0.55))
                        .frame(width: max(4, geometry.size.width * 0.3))
                }
            }
        }
        .frame(height: 4)
        .accessibilityElement()
        .accessibilityLabel("Progress")
        .accessibilityValue(fraction.map { "\(Int(($0 * 100).rounded())) percent" } ?? "In progress")
    }
}

// MARK: - Queued row

struct StudioQueuedRow: View {
    let item: StudioLibraryItem
    /// 0 is next in line.
    let position: Int
    let isHighlighted: Bool
    let onRemove: () -> Void

    private var positionText: String {
        position == 0 ? "Queued · next" : "Queued · \(position + 1) ahead"
    }

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(MereRunTheme.yellow)
                .frame(width: 8, height: 8)
            Text(positionText)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
                .fixedSize()
            Text(item.displayTitle)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(MereRunTheme.textPrimary)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button("Remove", action: onRemove)
                .buttonStyle(.mereSecondary)
                .help("Take this run out of the queue")
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 16)
        .feedPanel(isHighlighted: isHighlighted)
        .opacity(0.85)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(positionText): \(item.displayTitle)")
    }
}

// MARK: - Failure card

/// A run that did not finish: one line that says why, the log behind a disclosure, and Retry.
struct StudioFailureCard: View {
    let item: StudioLibraryItem
    let job: Job?
    let isHighlighted: Bool
    let actions: StudioFeedActions

    @State private var showLog = false

    private var logLines: [String] {
        if let job, !job.log.isEmpty { return job.log.lines.map(\.text) }
        return (item.outputText ?? "").components(separatedBy: .newlines).filter { !$0.isBlank }
    }

    private var summary: String {
        StudioFailureSummary.summary(
            outputText: item.outputText,
            logLines: job?.log.lines.map(\.text) ?? [],
            exitCode: job?.exitCode ?? item.exitCode
        )
    }

    private var wasCancelled: Bool {
        if let job, case .cancelled = job.state { return true }
        return item.exitCode == JobResult.cancelledBeforeStartExitCode
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            StudioCardHeader(item: item, when: StudioFeedTime.label(for: item.createdAt))

            HStack(alignment: .top, spacing: 8) {
                Image(systemName: wasCancelled ? "stop.circle" : "exclamationmark.triangle.fill")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(wasCancelled ? MereRunTheme.textMuted : MereRunTheme.red)
                    .padding(.top, 1)
                Text(summary)
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack(alignment: .top, spacing: 8) {
                VStack(alignment: .leading, spacing: 6) {
                    if !logLines.isEmpty {
                        Button {
                            withAnimation(MereRunTheme.Motion.quick) { showLog.toggle() }
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: showLog ? "chevron.down" : "chevron.right")
                                    .font(.system(size: 9, weight: .semibold))
                                Text(showLog ? "Hide log" : "Show log")
                                    .font(.system(size: 11.5, weight: .medium))
                            }
                            .foregroundStyle(MereRunTheme.textMuted)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(showLog ? "Hide log" : "Show log")
                    }
                    if showLog {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 3) {
                                ForEach(Array(logLines.suffix(80).enumerated()), id: \.offset) { _, line in
                                    Text(line)
                                        .font(.system(size: 11, design: .monospaced))
                                        .foregroundStyle(MereRunTheme.textMuted)
                                        .textSelection(.enabled)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxHeight: 200)
                    }
                }
                Spacer(minLength: 8)
                if item.commandDraft != nil, item.templateID != nil {
                    Button("Retry") { actions.retry(item) }
                        .buttonStyle(.mereSecondary)
                        .help("Run the same command again")
                }
            }
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 16)
        .feedPanel(
            borderColor: wasCancelled ? MereRunTheme.border.opacity(0.8) : MereRunTheme.red.opacity(0.4),
            isHighlighted: isHighlighted
        )
        .contextMenu {
            Button("Delete", role: .destructive) { actions.delete(item) }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(wasCancelled ? "Cancelled" : "Failed"): \(item.displayTitle). \(summary)")
    }
}

// MARK: - Readiness card

/// The bottom card while the mode cannot run: the model to get (with the pull's own progress
/// once it is downloading), or what went wrong checking. Never covers earlier cards.
struct StudioReadinessCard: View {
    let readiness: ModelReadinessState
    let pullJob: Job?
    let onPullModel: () -> Void
    let onShowDetails: () -> Void
    let onCancelPull: (Job) -> Void

    static let feedID = UUID()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: statusImage)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(statusColor)
                    .frame(width: 32, height: 32)
                    .background { Circle().fill(statusColor.opacity(0.12)) }
                VStack(alignment: .leading, spacing: 3) {
                    Text(pullJob == nil ? readiness.title : "Getting the model")
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(MereRunTheme.textPrimary)
                    Text(readiness.message)
                        .font(.system(size: 12))
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 8)
                if pullJob == nil {
                    if readiness.canPull {
                        Button {
                            onPullModel()
                        } label: {
                            Label("Get the model", systemImage: "arrow.down.circle.fill")
                        }
                        .buttonStyle(.merePrimary)
                    }
                    Button("Details", action: onShowDetails)
                        .buttonStyle(.mereSecondary)
                        .help("Show the command this task would run")
                }
            }
            if let pullJob {
                StudioPullProgressRow(job: pullJob) { onCancelPull(pullJob) }
            }
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 16)
        .feedPanel(borderColor: statusColor.opacity(0.4))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(readiness.title): \(readiness.message)")
    }

    private var statusImage: String {
        if pullJob != nil { return "arrow.down.circle" }
        if readiness.isChecking { return "hourglass" }
        return readiness.canPull ? "arrow.down.circle" : "exclamationmark.triangle"
    }

    private var statusColor: Color {
        if pullJob != nil || readiness.isChecking { return MereRunTheme.accent }
        return readiness.canPull ? MereRunTheme.yellow : MereRunTheme.red
    }
}

/// The pull's bar, bytes, and Cancel, observing the `model pull` job itself.
private struct StudioPullProgressRow: View {
    @ObservedObject var job: Job
    let onCancel: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            StudioProgressBar(fraction: job.progress?.fractionCompleted)
                .frame(maxWidth: .infinity)
            Text(job.progress?.detail ?? job.status)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
                .fixedSize()
            Button("Cancel", action: onCancel)
                .buttonStyle(.mereSecondary)
        }
    }
}

// MARK: - New result pill

private struct StudioNewResultPill: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text("New result ↓")
                .font(.system(size: 11.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.onAccent)
                .padding(.horizontal, 12)
                .frame(height: 26)
                .background { Capsule().fill(MereRunTheme.accent) }
                .mereShadow(radius: 8, y: 3)
        }
        .buttonStyle(.plain)
        .help("Scroll to the result that just finished")
        .accessibilityLabel("New result. Scroll to it")
    }
}
