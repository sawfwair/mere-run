import AppKit
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

// Library runs as inputs. Every attachment entry point — a well slot, an empty state's primary
// button, the Analyze canvas's Replace, a contract path row — offers "From Disk…" and
// "From Library…" through `StudioAttachMenu`; the Library choice opens `StudioLibraryInputPicker`,
// and a pick fills the slot exactly as a disk pick would. Library rows and output tiles drag as
// file URLs (`studioFileDrag`), so the same wells take them by drop.

// MARK: - Environment

private struct StudioLibraryItemsKey: EnvironmentKey {
    static let defaultValue: [StudioLibraryItem] = []
}

extension EnvironmentValues {
    /// The Library's rows, for the attachment entry points' "From Library…". The window sets it
    /// from its `StudioLibraryStore`; a view hosted without it offers the disk alone.
    package var studioLibraryItems: [StudioLibraryItem] {
        get { self[StudioLibraryItemsKey.self] }
        set { self[StudioLibraryItemsKey.self] = newValue }
    }
}

// MARK: - Target

/// Where an attachment entry point sends what the user picks: what it takes, and the write
/// that stores the files (a slot's `attach`, a path row's binding).
struct StudioAttachTarget {
    let requirement: StudioAttachmentRequirement
    let attach: ([URL]) -> Void

    init(requirement: StudioAttachmentRequirement, attach: @escaping ([URL]) -> Void) {
        self.requirement = requirement
        self.attach = attach
    }

    /// A well slot's target: picks land in `slot` on `draft`, through the slot's own `attach`.
    init<Draft: StudioAttachmentDraft>(slot: StudioAttachmentSlot, draft: Binding<Draft>) {
        self.init(requirement: StudioAttachmentRequirement(slot: slot)) { urls in
            slot.attach(urls, to: &draft.wrappedValue)
        }
    }
}

/// The open panel every attachment entry point shares: files of the slot's types, or a folder
/// for a directory slot, several at once for a list slot.
@MainActor
enum StudioAttachmentPicker {
    static func chooseFromDisk(for requirement: StudioAttachmentRequirement) -> [URL] {
        let panel = NSOpenPanel()
        panel.message = "Choose \(requirement.label.lowercased())"
        panel.canChooseFiles = !requirement.picksDirectory
        panel.canChooseDirectories = requirement.picksDirectory
        panel.canCreateDirectories = requirement.picksDirectory
        panel.allowsMultipleSelection = requirement.allowsMultiple
        if !requirement.picksDirectory, !requirement.acceptedTypes.isEmpty {
            panel.allowedContentTypes = requirement.acceptedTypes
        }
        return panel.runModal() == .OK ? panel.urls : []
    }

    static func pick<Draft: StudioAttachmentDraft>(for slot: StudioAttachmentSlot, into draft: inout Draft) {
        let urls = chooseFromDisk(for: StudioAttachmentRequirement(slot: slot))
        guard !urls.isEmpty else { return }
        slot.attach(urls, to: &draft)
    }
}

// MARK: - Menu

/// An attachment entry point: a menu of "From Disk…" and "From Library…" (plus `extraItems`, such
/// as Record…) when the Library holds a finished run with a file the target takes, and a plain
/// button straight to the open panel when it holds none, so an empty Library never costs a
/// click. `face` draws the control and is told whether it opens a menu.
struct StudioAttachMenu<Face: View, ExtraItems: View>: View {
    let target: StudioAttachTarget
    /// Runs the disk choice; defaults to the shared open panel into `target`.
    var chooseFromDisk: (() -> Void)?
    /// Lets a context menu open the same picker; nil keeps the state here.
    var isPickingFromLibrary: Binding<Bool>?
    var arrowEdge: Edge = .bottom
    @ViewBuilder let face: (_ opensMenu: Bool) -> Face
    @ViewBuilder var extraItems: () -> ExtraItems

    @Environment(\.studioLibraryItems) private var libraryItems
    @State private var localPicking = false

    private var pickingBinding: Binding<Bool> { isPickingFromLibrary ?? $localPicking }

    private var hasLibraryChoices: Bool {
        StudioLibraryInputs.hasCandidates(in: libraryItems, for: target.requirement)
    }

    var body: some View {
        Group {
            if hasLibraryChoices {
                Menu {
                    Button(action: diskChoice) {
                        Label("From Disk…", systemImage: "folder")
                    }
                    Button {
                        pickingBinding.wrappedValue = true
                    } label: {
                        Label("From Library…", systemImage: "rectangle.stack")
                    }
                    extraItems()
                } label: {
                    face(true)
                }
                .menuStyle(.button)
                .buttonStyle(.plain)
                .menuIndicator(.hidden)
                .fixedSize()
                .accessibilityHint("Choose from disk or from the Library")
            } else {
                Button(action: diskChoice) { face(false) }
                    .buttonStyle(StudioPressDimButtonStyle())
            }
        }
        .popover(isPresented: pickingBinding, arrowEdge: arrowEdge) {
            StudioLibraryInputPicker(
                requirement: target.requirement,
                items: libraryItems,
                onPick: { url in
                    pickingBinding.wrappedValue = false
                    target.attach([url])
                },
                onChooseFromDisk: {
                    pickingBinding.wrappedValue = false
                    diskChoice()
                }
            )
        }
    }

    private func diskChoice() {
        if let chooseFromDisk {
            chooseFromDisk()
            return
        }
        let urls = StudioAttachmentPicker.chooseFromDisk(for: target.requirement)
        if !urls.isEmpty { target.attach(urls) }
    }
}

extension StudioAttachMenu where ExtraItems == EmptyView {
    init(
        target: StudioAttachTarget,
        chooseFromDisk: (() -> Void)? = nil,
        isPickingFromLibrary: Binding<Bool>? = nil,
        arrowEdge: Edge = .bottom,
        @ViewBuilder face: @escaping (_ opensMenu: Bool) -> Face
    ) {
        self.init(
            target: target, chooseFromDisk: chooseFromDisk, isPickingFromLibrary: isPickingFromLibrary,
            arrowEdge: arrowEdge, face: face, extraItems: { EmptyView() }
        )
    }
}

/// A titled attachment control in the Studio's primary or secondary chrome: the empty state's
/// "Choose audio…", the Analyze canvas's Replace, a path row's Choose…. The pull-down chevron
/// appears only when the click opens the menu.
struct StudioAttachButton: View {
    enum Prominence { case primary, secondary }

    let target: StudioAttachTarget
    let title: String
    var systemImage: String?
    var prominence: Prominence = .secondary
    var chooseFromDisk: (() -> Void)?

    var body: some View {
        StudioAttachMenu(target: target, chooseFromDisk: chooseFromDisk) { opensMenu in
            face(opensMenu: opensMenu)
        }
        .accessibilityLabel(title)
    }

    @ViewBuilder
    private func face(opensMenu: Bool) -> some View {
        switch prominence {
        case .primary:
            MerePrimaryMenuLabel(title, systemImage: systemImage, showsChevron: opensMenu)
        case .secondary:
            MereSecondaryMenuLabel(title, systemImage: systemImage, showsChevron: opensMenu)
        }
    }
}

/// A face that draws its own chrome dims while pressed, so the plain-button fallback of an
/// attach control acknowledges the click the way the Studio's button styles do.
struct StudioPressDimButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .opacity(configuration.isPressed ? 0.8 : 1)
            .animation(MereRunTheme.Motion.quick, value: configuration.isPressed)
    }
}

// MARK: - Picker

/// "From Library…": the finished runs whose files the slot takes, newest first, searchable. A run
/// with one such file is one row; a run with several (stems, a batch of pictures) lists each file
/// under it. Type to search, ↑/↓ to move, Return to choose, Escape to close.
struct StudioLibraryInputPicker: View {
    let requirement: StudioAttachmentRequirement
    let items: [StudioLibraryItem]
    let onPick: (URL) -> Void
    let onChooseFromDisk: () -> Void

    @State private var query = ""
    @State private var highlighted: StudioLibraryInputChoice?
    @FocusState private var searchFocused: Bool
    @Environment(\.studioModelTitles) private var titles
    @Environment(\.studioReferenceDate) private var referenceDate

    private enum Metrics {
        static let width: CGFloat = 360
        static let listHeight: CGFloat = 320
        static let thumbnail: CGFloat = 36
        static let fileThumbnail: CGFloat = 24
    }

    private var groups: [StudioLibraryInputGroup] {
        StudioLibraryInputs.groups(in: items, for: requirement, query: query, titles: titles)
    }

    var body: some View {
        let groups = groups
        let choices = StudioLibraryInputs.choices(in: groups)
        VStack(alignment: .leading, spacing: 0) {
            header
            searchField(choices: choices)
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            Divider().overlay(MereRunTheme.border.opacity(0.5))
            if groups.isEmpty {
                emptyState
            } else {
                list(groups: groups)
            }
            Divider().overlay(MereRunTheme.border.opacity(0.5))
            footer(runCount: groups.count)
        }
        .frame(width: Metrics.width)
        .background(MereRunTheme.background)
        .onAppear {
            highlighted = choices.first
            searchFocused = true
        }
        .onChange(of: query) { _, _ in
            let refreshed = StudioLibraryInputs.choices(in: self.groups)
            if let highlighted, refreshed.contains(highlighted) { return }
            highlighted = refreshed.first
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Choose \(requirement.label.lowercased()) from the Library")
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Choose from Library")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(MereRunTheme.textPrimary)
            Text(subtitle)
                .font(.caption.weight(.medium))
                .foregroundStyle(MereRunTheme.textMuted)
        }
        .padding(.horizontal, 14)
        .padding(.top, 12)
        .padding(.bottom, 8)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }

    /// "Audio from finished runs" when the slot is named for its media; "Reference · Images from
    /// finished runs" when it is named for its role.
    private var subtitle: String {
        let media = "\(requirement.mediaNoun.capitalizedFirst) from finished runs"
        let label = requirement.label.lowercased()
        let singular = requirement.mediaNoun.hasSuffix("s") ? String(requirement.mediaNoun.dropLast()) : requirement.mediaNoun
        return label == requirement.mediaNoun || label == singular ? media : "\(requirement.label) · \(media)"
    }

    private func searchField(choices: [StudioLibraryInputChoice]) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.callout.weight(.medium))
                .foregroundStyle(MereRunTheme.textMuted)
            TextField("Search Library", text: $query)
                .textFieldStyle(.plain)
                .font(.callout)
                .foregroundStyle(MereRunTheme.textPrimary)
                .focused($searchFocused)
                .focusEffectDisabled()
                .onSubmit {
                    if let choice = highlighted ?? choices.first { onPick(choice.url) }
                }
                .onKeyPress(.downArrow) { move(by: 1, in: choices) }
                .onKeyPress(.upArrow) { move(by: -1, in: choices) }
                .accessibilityLabel("Search Library")
                .accessibilityHint("Up and down arrows move through the results; Return chooses")
            if !query.isEmpty {
                Button {
                    query = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.caption)
                }
                .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, 10)
        .frame(height: 28)
        .background {
            Capsule()
                .fill(MereRunTheme.surface)
                .overlay {
                    Capsule().strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                }
        }
    }

    private func list(groups: [StudioLibraryInputGroup]) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(groups) { group in
                        if group.files.count == 1, let url = group.files.first {
                            runRow(group: group, url: url)
                        } else {
                            multiFileRun(group: group)
                        }
                    }
                }
                .padding(6)
            }
            .frame(height: Metrics.listHeight)
            .onChange(of: highlighted) { _, choice in
                guard let choice else { return }
                proxy.scrollTo(choice.id)
            }
        }
    }

    /// A run with one file the slot takes: the whole row chooses it.
    private func runRow(group: StudioLibraryInputGroup, url: URL) -> some View {
        let choice = StudioLibraryInputChoice(itemID: group.id, url: url)
        return StudioLibraryInputRow(
            isHighlighted: highlighted == choice,
            action: { onPick(url) },
            onHover: { if $0 { highlighted = choice } }
        ) {
            HStack(spacing: 10) {
                thumbnail(url: url, item: group.item, side: Metrics.thumbnail)
                VStack(alignment: .leading, spacing: 1) {
                    Text(group.item.displayTitle)
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Text(meta(for: group.item))
                        .font(.caption.weight(.medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
            }
        }
        .id(choice.id)
        .accessibilityLabel("\(group.item.displayTitle), \(url.lastPathComponent)")
        .accessibilityValue(meta(for: group.item))
        .accessibilityHint("Uses this file as the \(requirement.label.lowercased())")
    }

    /// A run with several files the slot takes: its heading, then one row per file.
    private func multiFileRun(group: StudioLibraryInputGroup) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 10) {
                thumbnail(url: group.files[0], item: group.item, side: Metrics.thumbnail)
                VStack(alignment: .leading, spacing: 1) {
                    Text(group.item.displayTitle)
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Text("\(meta(for: group.item)) · \(group.files.count) files")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .accessibilityElement(children: .combine)
            .accessibilityLabel("\(group.item.displayTitle), \(group.files.count) files, \(meta(for: group.item))")

            ForEach(group.files, id: \.self) { url in
                fileRow(group: group, url: url)
            }
        }
        .accessibilityElement(children: .contain)
    }

    private func fileRow(group: StudioLibraryInputGroup, url: URL) -> some View {
        let choice = StudioLibraryInputChoice(itemID: group.id, url: url)
        let role = group.item.artifactRoleLabel(for: url)
        return StudioLibraryInputRow(
            isHighlighted: highlighted == choice,
            action: { onPick(url) },
            onHover: { if $0 { highlighted = choice } }
        ) {
            HStack(spacing: 8) {
                thumbnail(url: url, item: group.item, side: Metrics.fileThumbnail)
                Text(url.lastPathComponent)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 4)
                if let role {
                    Text(role)
                        .font(.caption2.weight(.medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                }
            }
            .padding(.leading, Metrics.thumbnail - Metrics.fileThumbnail + 10)
        }
        .id(choice.id)
        .accessibilityLabel(url.lastPathComponent)
        .accessibilityValue([role, group.item.displayTitle].compactMap { $0 }.joined(separator: ", "))
        .accessibilityHint("Uses this file as the \(requirement.label.lowercased())")
    }

    private func thumbnail(url: URL, item: StudioLibraryItem, side: CGFloat) -> some View {
        StudioFileThumbnail(url: url, side: side, fallbackSystemImage: item.displaySystemImage)
            .frame(width: side, height: side)
            .background(MereRunTheme.surfaceRaised)
            .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))
    }

    private var emptyState: some View {
        VStack(spacing: 6) {
            Image(systemName: query.isBlank ? "rectangle.stack" : "magnifyingglass")
                .font(.system(size: 20, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            Text(query.isBlank
                 ? "No \(requirement.mediaNoun) in the Library yet"
                 : "No \(requirement.mediaNoun) match “\(query.trimmingCharacters(in: .whitespaces))”")
                .font(.system(size: 12.5, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
                .multilineTextAlignment(.center)
            Text("Finished runs whose files this slot takes appear here.")
                .font(.caption)
                .foregroundStyle(MereRunTheme.textMuted)
                .multilineTextAlignment(.center)
        }
        .padding(.horizontal, 24)
        .frame(maxWidth: .infinity)
        .frame(height: Metrics.listHeight)
        .accessibilityElement(children: .combine)
    }

    private func footer(runCount: Int) -> some View {
        HStack(spacing: 8) {
            if runCount > 0 {
                Text(runCount == 1 ? "1 run" : "\(runCount) runs")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            Spacer(minLength: 8)
            Button(action: onChooseFromDisk) {
                Label("From Disk…", systemImage: "folder")
            }
            .buttonStyle(.mereSecondary)
            .help("Choose a file in Finder instead")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private func move(by offset: Int, in choices: [StudioLibraryInputChoice]) -> KeyPress.Result {
        guard !choices.isEmpty else { return .ignored }
        let current = highlighted.flatMap { choices.firstIndex(of: $0) } ?? (offset > 0 ? -1 : choices.count)
        highlighted = choices[min(max(current + offset, 0), choices.count - 1)]
        return .handled
    }

    /// "Compose · 12:43 PM" for today's runs, "Compose · Sep 3, 12:43 PM" for earlier ones.
    private func meta(for item: StudioLibraryItem) -> String {
        let now = referenceDate ?? Date()
        let formatter = Calendar.current.isDate(item.createdAt, inSameDayAs: now) ? Self.timeFormatter : Self.dayFormatter
        return StudioLibraryRowMeta.text(
            item: item,
            kindTitle: StudioLibraryRowMeta.kindTitle(for: item),
            progress: nil,
            formatter: formatter
        )
    }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        return formatter
    }()

    private static let dayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.setLocalizedDateFormatFromTemplate("MMMd jmm")
        return formatter
    }()
}

/// One choosable row of the picker: the Library row's hover and selection fills, a button for
/// VoiceOver and the keyboard, and hover that moves the keyboard highlight so the two agree.
private struct StudioLibraryInputRow<Content: View>: View {
    let isHighlighted: Bool
    let action: () -> Void
    let onHover: (Bool) -> Void
    @ViewBuilder let content: () -> Content

    var body: some View {
        Button(action: action) {
            content()
                .padding(.vertical, 5)
                .padding(.horizontal, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                        .fill(isHighlighted ? MereRunTheme.accentSoft : .clear)
                }
                .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
        }
        .buttonStyle(.plain)
        .onHover(perform: onHover)
        .accessibilityElement(children: .ignore)
        .accessibilityAddTraits(isHighlighted ? [.isButton, .isSelected] : .isButton)
    }
}

private extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}

// MARK: - Drag source

extension View {
    /// Drags `url` as a file: onto a Studio well, composer, or canvas (which read the file URL
    /// and route it to the slot that takes it), or out to Finder and other apps. Inert when there
    /// is no file, rather than dragging an empty promise.
    func studioFileDrag(_ url: URL?) -> some View {
        modifier(StudioFileDragSource(url: url))
    }
}

struct StudioFileDragSource: ViewModifier {
    let url: URL?

    func body(content: Content) -> some View {
        if let url {
            content
                .onDrag { Self.provider(for: url) }
                .accessibilityDragPoint(
                    .center,
                    description: Text("Drag \(url.lastPathComponent) onto an input or into Finder")
                )
        } else {
            content
        }
    }

    /// The file's own type (so Finder and other apps copy the file) plus its file URL (so a
    /// Studio drop target reads the original path, not a copy).
    static func provider(for url: URL) -> NSItemProvider {
        let provider = NSItemProvider(contentsOf: url) ?? NSItemProvider(object: url as NSURL)
        provider.suggestedName = url.lastPathComponent
        return provider
    }
}
