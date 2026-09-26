import AVFoundation
import AVKit
import StudioKit
import SwiftUI

/// How Compare lays out pictures: every pane side by side, or two over each other with a
/// draggable divider.
enum StudioCompareImageLayout: String, CaseIterable {
    case grid, slider

    var title: String {
        switch self {
        case .grid: return "Side by side"
        case .slider: return "A/B slider"
        }
    }
}

/// Two to eight finished runs of one kind, compared in place of the canvas: pictures with one
/// zoom and pan (or an A/B slider), sounds as stacked waveforms on one playhead with A/B
/// listening, videos side by side on one transport. Every pane names what it ran with that the
/// others did not, and can be kept, reused, or sent on.
struct StudioCompareView: View {
    let items: [StudioLibraryItem]
    let onClose: () -> Void
    /// Keep: star the row, or unstar it.
    let onKeep: (StudioLibraryItem) -> Void
    /// Use these settings: the row's recorded command back in its task's composer.
    let onUseSettings: (StudioLibraryItem) -> Void
    var initialImageLayout: StudioCompareImageLayout = .grid

    @Environment(\.studioScopeSource) private var scopeSource
    @State private var imageLayout: StudioCompareImageLayout?
    @State private var zoom: CGFloat = 1
    @State private var pan: CGSize = .zero
    @State private var split: CGFloat = 0.5
    @State private var sliderPanes = (left: 0, right: 1)
    @StateObject private var audio = StudioCompareAudioDeck()
    @StateObject private var video = StudioCompareVideoDeck()
    @FocusState private var keyboardFocused: Bool
    private let ticker = Timer.publish(every: 0.05, on: .main, in: .common).autoconnect()

    @State private var paneMemo = StudioComparePaneMemo()
    /// Read once per change to the rows: the playhead republishes twenty times a second while a
    /// side plays, and reading every pane's argv again each time would be wasted work.
    private var panes: [StudioComparePane] { paneMemo.panes(for: items, source: scopeSource) }
    private var media: StudioCompareMedia? { items.first.flatMap(StudioCompare.media(of:)) }
    private var layout: StudioCompareImageLayout { imageLayout ?? initialImageLayout }
    /// Panes name their run only when the runs are named differently.
    private var showsTitles: Bool { Set(items.map(\.displayTitle)).count > 1 }

    var body: some View {
        let panes = panes
        VStack(spacing: 0) {
            toolbar(panes)
            Divider()
            switch media {
            case .image: imageContent(panes)
            case .audio: audioContent(panes)
            case .video: videoContent(panes)
            case nil: EmptyView()
            }
        }
        .background(MereRunTheme.background)
        .focusable()
        .focusEffectDisabled()
        .focused($keyboardFocused)
        .onKeyPress(action: handleKey)
        .onAppear { keyboardFocused = true }
        .task(id: panes.map(\.url)) {
            switch media {
            case .audio: await audio.load(panes.map(\.url))
            case .video: await video.load(panes.map(\.url))
            case .image, nil: break
            }
        }
        .onReceive(ticker) { _ in
            audio.refresh()
            video.refresh()
        }
        .onDisappear {
            audio.stop()
            video.stop()
        }
        .onExitCommand(perform: onClose)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Compare \(media?.count(panes.count) ?? "results")")
    }

    // MARK: Toolbar

    private func toolbar(_ panes: [StudioComparePane]) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) { back; title(panes); Spacer(); tools(panes) }
            VStack(alignment: .leading, spacing: 10) {
                HStack { back; title(panes); Spacer() }
                HStack { tools(panes); Spacer() }
            }
        }
        .padding(14)
    }

    private var back: some View {
        Button(action: onClose) { Label("Results", systemImage: "chevron.left") }
            .buttonStyle(.mereSecondary)
            .keyboardShortcut(.cancelAction)
            .help("Return to your results and draft (Escape)")
    }

    private func title(_ panes: [StudioComparePane]) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Compare").font(.headline)
            Text(subtitle(panes))
                .font(.caption)
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }

    /// "4 images · Variations of harbor at dusk", or the count alone for a hand-picked set.
    private func subtitle(_ panes: [StudioComparePane]) -> String {
        let count = media?.count(panes.count) ?? ""
        let groups = Set(panes.map(\.item.variationGroup))
        if groups.count == 1, groups.first != nil, let first = panes.first {
            return "\(count) · Variations of \(first.item.displayTitle)"
        }
        return count
    }

    @ViewBuilder
    private func tools(_ panes: [StudioComparePane]) -> some View {
        switch media {
        case .image:
            HStack(spacing: 12) {
                MereSegmentedControl(
                    StudioCompareImageLayout.allCases,
                    selection: Binding(get: { layout }, set: { imageLayout = $0 }),
                    accessibilityLabel: "Layout"
                ) { $0.title }
                .fixedSize()
                zoomControls
            }
        case .audio:
            StudioCompareTransportBar(
                transport: audio.transport, letters: panes.map(\.letter), showsScrubber: false,
                onToggle: audio.togglePlay, onSelect: audio.select, onSeek: audio.seek(to:)
            )
        case .video:
            StudioCompareTransportBar(
                transport: video.transport, letters: panes.map(\.letter), showsScrubber: true,
                onToggle: video.togglePlay, onSelect: video.select, onSeek: video.seek(to:)
            )
        case nil:
            EmptyView()
        }
    }

    private var zoomControls: some View {
        HStack(spacing: 10) {
            Button { zoom = max(1, zoom / 1.5); if zoom == 1 { pan = .zero } } label: {
                Image(systemName: "minus.magnifyingglass")
            }
            .keyboardShortcut("-", modifiers: .command)
            .help("Zoom out")
            .accessibilityLabel("Zoom out")
            .disabled(zoom <= 1)
            Button { zoom = 1; pan = .zero } label: { Text(zoom == 1 ? "Fit" : "\(Int(zoom * 100))%").monospacedDigit().fixedSize() }
                .keyboardShortcut("0", modifiers: .command)
                .help("Fit every image")
                .accessibilityLabel("Fit images")
                .accessibilityValue("\(Int(zoom * 100)) percent")
            Button { zoom = min(8, zoom * 1.5) } label: { Image(systemName: "plus.magnifyingglass") }
                .keyboardShortcut("+", modifiers: .command)
                .help("Zoom in")
                .accessibilityLabel("Zoom in")
                .disabled(zoom >= 8)
        }
        .buttonStyle(.borderless)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Linked zoom")
    }

    /// 1…8 hand playback to that pane (sounds and videos) at the same moment; Space plays or
    /// pauses (`StudioShortcutContext.compare` in the shortcut table). The comparison takes
    /// keyboard focus when it opens, so the keys never reach a text field or the menu bar.
    private func handleKey(_ press: KeyPress) -> KeyPress.Result {
        guard media == .audio || media == .video else { return .ignored }
        if press.key == .space {
            media == .audio ? audio.togglePlay() : video.togglePlay()
            return .handled
        }
        guard let digit = Int(press.characters), digit >= 1, digit <= items.count else { return .ignored }
        media == .audio ? audio.select(digit - 1) : video.select(digit - 1)
        return .handled
    }

    // MARK: Images

    @ViewBuilder
    private func imageContent(_ panes: [StudioComparePane]) -> some View {
        switch layout {
        case .grid:
            StudioCompareGrid(count: panes.count) { index in
                let pane = panes[index]
                VStack(spacing: 0) {
                    StudioResultImagePane(url: pane.url, label: pane.letter, zoom: $zoom, pan: $pan)
                    footer(pane)
                }
            }
        case .slider:
            sliderContent(panes)
        }
    }

    private func sliderContent(_ panes: [StudioComparePane]) -> some View {
        let left = panes[min(sliderPanes.left, panes.count - 1)]
        let right = panes[min(sliderPanes.right, panes.count - 1)]
        return VStack(spacing: 0) {
            if panes.count > 2 {
                HStack(spacing: 10) {
                    sliderPicker("Left", panes: panes, selection: Binding(get: { sliderPanes.left }, set: { sliderPanes.left = $0 }))
                    sliderPicker("Right", panes: panes, selection: Binding(get: { sliderPanes.right }, set: { sliderPanes.right = $0 }))
                    Spacer()
                }
                .padding(.horizontal, 14)
                .padding(.bottom, 10)
            }
            StudioCompareSlider(left: left, right: right, split: $split, zoom: $zoom, pan: $pan)
            HStack(spacing: 1) {
                footer(left)
                footer(right)
            }
            .background(MereRunTheme.border)
        }
    }

    private func sliderPicker(_ side: String, panes: [StudioComparePane], selection: Binding<Int>) -> some View {
        Picker(side, selection: selection) {
            ForEach(panes.indices, id: \.self) { index in
                Text("\(panes[index].letter) · \(showsTitles ? panes[index].item.displayTitle : seedTitle(panes[index]))").tag(index)
            }
        }
        .fixedSize()
    }

    /// "Seed 4517", what tells variations apart in the slider's pickers.
    private func seedTitle(_ pane: StudioComparePane) -> String {
        pane.settings.first { $0.kind == .seed }.map { "Seed \($0.value)" } ?? pane.item.displayTitle
    }

    // MARK: Audio

    private func audioContent(_ panes: [StudioComparePane]) -> some View {
        ScrollView {
            VStack(spacing: 10) {
                ForEach(panes.indices, id: \.self) { index in
                    audioRow(panes[index], index: index)
                }
            }
            .padding(14)
        }
    }

    private func audioRow(_ pane: StudioComparePane, index: Int) -> some View {
        let heard = audio.transport.active == index
        return VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                StudioCompareLetter(letter: pane.letter, isActive: heard) { audio.select(index) }
                Group {
                    if let peaks = audio.peaks(of: index) {
                        StudioWaveformView(peaks: peaks, progress: audio.transport.progress(of: index)) { fraction in
                            audio.seek(fraction: fraction, in: index)
                        }
                    } else {
                        StudioWaveformView(peaks: StudioWaveformView.placeholderPeaks, progress: 0).opacity(0.35)
                    }
                }
                .frame(height: 64)
                .opacity(heard ? 1 : 0.6)
                Text(StudioTimeFormat.string(audio.transport.durations.indices.contains(index) ? audio.transport.durations[index] : 0))
                    .font(MereRunTheme.captionFont)
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(minWidth: 40, alignment: .trailing)
            }
            StudioComparePaneFooter(pane: pane, showsLetter: false, showsTitle: showsTitles,
                                    onKeep: onKeep, onUseSettings: onUseSettings)
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .strokeBorder(heard ? MereRunTheme.accent.opacity(0.7) : MereRunTheme.border.opacity(0.8), lineWidth: heard ? 1.5 : 1)
                }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(pane.letter): \(pane.item.displayTitle)\(heard ? ", playing side" : "")")
    }

    // MARK: Video

    private func videoContent(_ panes: [StudioComparePane]) -> some View {
        StudioCompareGrid(count: panes.count) { index in
            let pane = panes[index]
            let heard = video.transport.active == index
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    if let player = video.player(for: index) {
                        StudioComparePlayerSurface(player: player)
                    } else {
                        MereRunTheme.surfaceRaised
                    }
                    StudioCompareLetter(letter: pane.letter, isActive: heard) { video.select(index) }
                        .padding(10)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black)
                .overlay {
                    if heard { Rectangle().strokeBorder(MereRunTheme.accent.opacity(0.7), lineWidth: 2) }
                }
                footer(pane)
            }
        }
    }

    private func footer(_ pane: StudioComparePane) -> some View {
        StudioComparePaneFooter(pane: pane, showsTitle: showsTitles, onKeep: onKeep, onUseSettings: onUseSettings)
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(MereRunTheme.surface)
    }
}

/// The feed's floating bar while cards are picked for Compare: how many, Compare once two to
/// four of one kind are picked, and Clear.
struct StudioCompareSelectionBar: View {
    let items: [StudioLibraryItem]
    let onCompare: ([StudioLibraryItem]) -> Void
    let onClear: () -> Void

    var body: some View {
        let reason = StudioCompare.unavailableReason(items)
        HStack(spacing: 10) {
            Text("\(items.count) selected")
                .font(.callout.weight(.medium))
                .foregroundStyle(MereRunTheme.textSecondary)
                .monospacedDigit()
            Button {
                onCompare(items)
                onClear()
            } label: {
                Label("Compare", systemImage: StudioVariationSymbols.compare)
            }
            .buttonStyle(.merePrimary)
            .disabled(reason != nil)
            .help(reason ?? "Compare the selected results side by side")
            Button("Clear", action: onClear)
                .buttonStyle(.mereSecondary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background {
            Capsule()
                .fill(MereRunTheme.surface)
                .overlay { Capsule().strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1) }
                .mereShadow(radius: 10, y: 4)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(items.count) results selected to compare")
    }
}

/// The panes for the rows last asked about, kept until the rows change (a Keep, a rename).
@MainActor
private final class StudioComparePaneMemo {
    private var items: [StudioLibraryItem]?
    private var panes: [StudioComparePane] = []

    func panes(for items: [StudioLibraryItem], source: StudioScopeSource) -> [StudioComparePane] {
        if items != self.items {
            self.items = items
            panes = StudioCompare.panes(for: items, source: source)
        }
        return panes
    }
}

// MARK: - Layout

/// Panes in rows of `StudioCompare.columns`, filling the space below the toolbar.
private struct StudioCompareGrid<Pane: View>: View {
    let count: Int
    @ViewBuilder let pane: (Int) -> Pane

    var body: some View {
        GeometryReader { geometry in
            let columns = StudioCompare.columns(count: count, width: geometry.size.width)
            let rows = stride(from: 0, to: count, by: columns).map { Array($0..<min($0 + columns, count)) }
            VStack(spacing: 1) {
                ForEach(rows, id: \.self) { row in
                    HStack(spacing: 1) {
                        ForEach(row, id: \.self) { index in pane(index) }
                        // A short last row keeps its panes the width of the others.
                        ForEach(row.count..<columns, id: \.self) { _ in MereRunTheme.background }
                    }
                }
            }
        }
        .background(MereRunTheme.border)
    }
}

/// A pane's letter, which is also how its side is picked for listening.
private struct StudioCompareLetter: View {
    let letter: String
    let isActive: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(letter)
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .foregroundStyle(isActive ? MereRunTheme.onAccent : MereRunTheme.textPrimary)
                .frame(width: 26, height: 26)
                .background(Circle().fill(isActive ? MereRunTheme.accent : MereRunTheme.surfaceRaised))
        }
        .buttonStyle(.plain)
        .help("Listen to \(letter)")
        .accessibilityLabel("Play \(letter)")
        .accessibilityAddTraits(isActive ? [.isButton, .isSelected] : .isButton)
    }
}

/// Under every pane: its letter, its title when the panes' titles differ (variations share
/// theirs, which the toolbar already says), what it ran with that the others did not, and Keep,
/// Use these settings, and Send to.
private struct StudioComparePaneFooter: View {
    let pane: StudioComparePane
    /// False where the pane shows its letter already (a sound's row).
    var showsLetter = true
    let showsTitle: Bool
    let onKeep: (StudioLibraryItem) -> Void
    let onUseSettings: (StudioLibraryItem) -> Void
    @Environment(\.studioModelTitles) private var titles

    var body: some View {
        if showsTitle {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    letter
                    Text(pane.item.displayTitle)
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Spacer(minLength: 6)
                    actions
                }
                settings
            }
        } else {
            HStack(spacing: 8) {
                letter
                // What the run ran with wins the width; the actions drop their words first.
                settings.layoutPriority(1)
                Spacer(minLength: 6)
                actions
            }
        }
    }

    @ViewBuilder private var letter: some View {
        if showsLetter {
            Text(pane.letter)
                .font(.caption.weight(.bold))
                .foregroundStyle(MereRunTheme.accent)
        }
    }

    private var settings: some View {
        settingsText
            .font(.caption)
            .lineLimit(2)
            .textSelection(.enabled)
            .accessibilityLabel("Settings: " + pane.settings.map { "\($0.title) \(value($0))" }.joined(separator: ", "))
    }

    private var actions: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 6) { keep(labelled: true); useSettings(labelled: true); StudioSendToButton(url: pane.url) }
            HStack(spacing: 2) { keep(labelled: false); useSettings(labelled: false); StudioSendToButton(url: pane.url) }
        }
    }

    private func keep(labelled: Bool) -> some View {
        let kept = pane.item.isStarred
        return Button { onKeep(pane.item) } label: {
            if labelled {
                Label(kept ? "Kept" : "Keep", systemImage: kept ? "star.fill" : "star")
            } else {
                Image(systemName: kept ? "star.fill" : "star").frame(width: 24, height: 24)
            }
        }
        .buttonStyle(.mereIcon(tint: kept ? MereRunTheme.yellow : MereRunTheme.textSecondary))
        .help(kept ? "Remove from Favorites" : "Keep this one in Favorites")
        .accessibilityLabel(kept ? "Remove \(pane.letter) from Favorites" : "Keep \(pane.letter)")
    }

    private func useSettings(labelled: Bool) -> some View {
        Button { onUseSettings(pane.item) } label: {
            if labelled {
                Label("Use these settings", systemImage: "slider.horizontal.3")
            } else {
                Image(systemName: "slider.horizontal.3").frame(width: 24, height: 24)
            }
        }
        .buttonStyle(.mereIcon(tint: MereRunTheme.textSecondary))
        .disabled(!StudioLibraryDraftRestoration.canRestore(pane.item))
        .help("Put \(pane.letter)'s prompt, model, and options in the composer")
        .accessibilityLabel("Use \(pane.letter)'s settings")
    }

    /// "Seed 1234 · Z-Image Turbo · Steps 28": seed and model always, then what differs.
    private var settingsText: Text {
        pane.settings.enumerated().reduce(Text("")) { text, entry in
            let (offset, setting) = entry
            let separator = offset == 0 ? Text("") : Text("  ·  ").foregroundStyle(MereRunTheme.textMuted)
            let title = setting.kind == .model ? Text("") : Text(setting.title + " ").foregroundStyle(MereRunTheme.textMuted)
            return text + separator + title + Text(value(setting)).foregroundStyle(MereRunTheme.textPrimary)
        }
    }

    private func value(_ setting: StudioCompareSetting) -> String {
        setting.kind == .model && setting.value != "Default"
            ? StudioModelNaming.displayName(setting.value, titles: titles) : setting.value
    }
}

/// Play/pause, the shared time, which pane is heard, and — for video — one scrubber for all.
private struct StudioCompareTransportBar: View {
    let transport: StudioCompareTransport
    let letters: [String]
    let showsScrubber: Bool
    let onToggle: () -> Void
    let onSelect: (Int) -> Void
    let onSeek: (Double) -> Void

    var body: some View {
        HStack(spacing: 12) {
            Button(action: onToggle) {
                Image(systemName: transport.isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(MereRunTheme.onAccent)
                    .frame(width: 30, height: 30)
                    .background(Circle().fill(MereRunTheme.accent))
                    .contentTransition(.symbolEffect(.replace))
            }
            .buttonStyle(.plain)
            .help(transport.isPlaying ? "Pause (Space)" : "Play (Space)")
            .accessibilityLabel(transport.isPlaying ? "Pause" : "Play")
            if showsScrubber {
                Slider(value: Binding(get: { transport.position }, set: { onSeek($0) }), in: 0...max(transport.span, 0.1))
                    .frame(minWidth: 160, maxWidth: 280)
                    .tint(MereRunTheme.accent)
                    .accessibilityLabel("Position")
            }
            Text("\(StudioTimeFormat.string(transport.position)) / \(StudioTimeFormat.string(transport.span))")
                .font(MereRunTheme.captionFont)
                .monospacedDigit()
                .foregroundStyle(MereRunTheme.textSecondary)
            MereSegmentedControl(
                Array(letters.indices),
                selection: Binding(get: { transport.active }, set: { onSelect($0) }),
                accessibilityLabel: "Playing side"
            ) { letters[$0] }
            .fixedSize()
            .help("Switch sides at the same moment (1–\(letters.count))")
        }
    }
}

// MARK: - A/B slider

/// Two pictures over each other, the left one shown up to a draggable divider, both at the one
/// zoom and pan.
private struct StudioCompareSlider: View {
    let left: StudioComparePane
    let right: StudioComparePane
    @Binding var split: CGFloat
    @Binding var zoom: CGFloat
    @Binding var pan: CGSize
    @State private var dragOrigin: CGSize?

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            ZStack(alignment: .topLeading) {
                layer(right, size: geometry.size)
                layer(left, size: geometry.size)
                    .mask(alignment: .leading) { Rectangle().frame(width: width * split) }
                label(left.letter).padding(10)
                label(right.letter)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                handle(height: geometry.size.height)
                    .position(x: width * split, y: geometry.size.height / 2)
                    .gesture(DragGesture(coordinateSpace: .named("compare-slider")).onChanged { value in
                        split = min(1, max(0, value.location.x / max(width, 1)))
                    })
            }
            .frame(width: width, height: geometry.size.height)
            .clipped()
            .contentShape(Rectangle())
            .coordinateSpace(name: "compare-slider")
            .gesture(DragGesture().onChanged { value in
                guard zoom > 1 else { return }
                let origin = dragOrigin ?? pan
                dragOrigin = origin
                pan = StudioResultViewport.clampedPan(
                    CGSize(width: origin.width + value.translation.width, height: origin.height + value.translation.height),
                    zoom: zoom, size: geometry.size)
            }.onEnded { _ in dragOrigin = nil })
        }
        .background(MereRunTheme.surfaceRaised)
    }

    private func layer(_ pane: StudioComparePane, size: CGSize) -> some View {
        StudioAsyncImagePreview(url: pane.url, maxPixelSize: 4096, contentMode: .fit, fallbackSystemImage: "photo")
            .frame(width: size.width, height: size.height)
            .scaleEffect(zoom)
            .offset(pan)
            .frame(width: size.width, height: size.height)
            .accessibilityHidden(true)
    }

    private func label(_ letter: String) -> some View {
        Text(letter)
            .font(.caption.weight(.semibold))
            .padding(8)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
    }

    private func handle(height: CGFloat) -> some View {
        ZStack {
            Rectangle().fill(Color.white).frame(width: 2, height: height).shadow(radius: 2)
            Image(systemName: "arrow.left.and.right")
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(Color.black)
                .frame(width: 30, height: 30)
                .background(Circle().fill(Color.white).shadow(radius: 3))
        }
        .frame(width: 30, height: height)
        .contentShape(Rectangle())
        .accessibilityElement()
        .accessibilityLabel("Divider between \(left.letter) and \(right.letter)")
        .accessibilityValue("\(Int((split * 100).rounded())) percent \(left.letter)")
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: split = min(1, split + 0.05)
            case .decrement: split = max(0, split - 0.05)
            @unknown default: break
            }
        }
    }
}

// MARK: - Playback

/// One `AVAudioPlayer` per pane, each prepared up front so switching sides starts at once. Only
/// the heard pane plays; the others wait at the shared position.
@MainActor
final class StudioCompareAudioDeck: ObservableObject {
    @Published private(set) var transport = StudioCompareTransport(durations: [])
    @Published private var peaks: [[Float]?] = []
    private var players: [AVAudioPlayer?] = []
    private var urls: [URL] = []

    func peaks(of pane: Int) -> [Float]? {
        peaks.indices.contains(pane) ? peaks[pane] : nil
    }

    func load(_ urls: [URL]) async {
        guard urls != self.urls else { return }
        stop()
        self.urls = urls
        // A file AVFoundation cannot open is a pane with no sound, as the feed's player shows it.
        players = urls.map { try? AVAudioPlayer(contentsOf: $0) }
        players.forEach { $0?.prepareToPlay() }
        transport = StudioCompareTransport(durations: players.map { $0?.duration ?? 0 })
        peaks = Array(repeating: nil, count: urls.count)
        for (index, url) in urls.enumerated() {
            let loaded = await Task.detached(priority: .userInitiated) { StudioWaveformLoader.peaks(url: url) }.value
            guard self.urls == urls else { return }
            peaks[index] = loaded
        }
    }

    func togglePlay() {
        transport.isPlaying ? pause() : play()
    }

    func play() {
        guard let player = player(transport.active) else { return }
        if transport.position >= transport.durations[transport.active] { transport.seek(to: 0) }
        player.currentTime = transport.position
        player.play()
        transport.isPlaying = true
    }

    func pause() {
        refresh()
        player(transport.active)?.pause()
        transport.isPlaying = false
    }

    /// Hands playback to `pane` at the moment the heard one had reached.
    func select(_ pane: Int) {
        refresh()
        let previous = transport.active
        transport.select(pane)
        guard transport.isPlaying, previous != pane else { return }
        player(previous)?.pause()
        if let next = player(pane) {
            next.currentTime = transport.position
            next.play()
        }
    }

    func seek(to seconds: Double) {
        transport.seek(to: seconds)
        player(transport.active)?.currentTime = transport.position
    }

    func seek(fraction: Double, in pane: Int) {
        let previous = transport.active
        transport.seek(fraction: fraction, in: pane)
        if transport.isPlaying, previous != pane {
            player(previous)?.pause()
            player(pane)?.play()
        }
        player(pane)?.currentTime = transport.position
    }

    func refresh() {
        guard transport.isPlaying, let player = player(transport.active) else { return }
        transport.advance(to: player.currentTime, stillPlaying: player.isPlaying)
    }

    func stop() {
        players.forEach { $0?.stop() }
        transport.isPlaying = false
    }

    private func player(_ pane: Int) -> AVAudioPlayer? {
        players.indices.contains(pane) ? players[pane] : nil
    }
}

/// One `AVPlayer` per pane on one transport: play, pause, and seek reach every pane, and only
/// the heard pane is unmuted.
@MainActor
final class StudioCompareVideoDeck: ObservableObject {
    @Published private(set) var transport = StudioCompareTransport(durations: [])
    private var players: [AVPlayer] = []
    private var urls: [URL] = []

    func player(for pane: Int) -> AVPlayer? {
        players.indices.contains(pane) ? players[pane] : nil
    }

    func load(_ urls: [URL]) async {
        guard urls != self.urls else { return }
        stop()
        self.urls = urls
        players = urls.map { AVPlayer(url: $0) }
        transport = StudioCompareTransport(durations: urls.map { _ in 0 })
        applyMute()
        var durations: [Double] = []
        for url in urls {
            // A clip whose length cannot be read scrubs as zero seconds long.
            let duration = try? await AVURLAsset(url: url).load(.duration)
            durations.append(duration.map(CMTimeGetSeconds) ?? 0)
        }
        guard self.urls == urls else { return }
        transport.durations = durations
    }

    func togglePlay() {
        transport.isPlaying ? pause() : play()
    }

    func play() {
        // Every pane starts from the shared position, which also rewinds a clip that ended.
        seek(to: transport.position >= transport.span ? 0 : transport.position)
        players.forEach { $0.play() }
        transport.isPlaying = true
    }

    func pause() {
        refresh()
        players.forEach { $0.pause() }
        transport.isPlaying = false
    }

    func select(_ pane: Int) {
        transport.select(pane)
        applyMute()
    }

    func seek(to seconds: Double) {
        transport.seek(to: seconds)
        for (index, player) in players.enumerated() {
            let limit = transport.durations.indices.contains(index) ? transport.durations[index] : 0
            let time = CMTime(seconds: min(transport.position, limit), preferredTimescale: 600)
            player.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero)
        }
    }

    func refresh() {
        guard transport.isPlaying, let player = player(for: transport.active) else { return }
        let seconds = CMTimeGetSeconds(player.currentTime())
        let ended = player.rate == 0 && seconds >= transport.durations[transport.active] - 0.05
        transport.advance(to: seconds, stillPlaying: !ended)
        if ended {
            players.forEach { $0.pause() }
            seek(to: 0)
        }
    }

    func stop() {
        players.forEach { $0.pause() }
        transport.isPlaying = false
    }

    private func applyMute() {
        for (index, player) in players.enumerated() { player.isMuted = index != transport.active }
    }
}

/// A player's picture with no controls of its own: Compare's transport drives every pane.
private struct StudioComparePlayerSurface: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .none
        view.videoGravity = .resizeAspect
        view.player = player
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if view.player !== player { view.player = player }
    }
}
