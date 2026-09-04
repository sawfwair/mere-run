import AppKit
import SwiftUI

/// Run history as a column beside the current task: filtered to the current domain (or All) and
/// to a kind or favorites, searchable, shown as rows or as a grid of thumbnails, grouped by day,
/// keyboard-navigable (arrows move, Space previews), with Quick Look surfacing on hover. Picking a
/// row of another domain switches the destination; ⌘ and ⇧ build a batch to reveal, save, or
/// delete in one go, and any row can be dragged out to Finder or another app.
struct StudioLibraryPanel: View {
    let items: [StudioLibraryItem]
    let domain: StudioDomain
    @Binding var scope: StudioLibraryScope
    @Binding var viewMode: StudioLibraryViewMode
    @Binding var kind: StudioLibraryKind
    @Binding var favoritesOnly: Bool
    let progressByID: [UUID: StudioRunProgress]
    @Binding var selectedID: UUID?
    /// A row the user picked (click or arrow keys), as opposed to a programmatic selection.
    let onSelect: (StudioLibraryItem) -> Void
    /// Rows to forget, and whether their files go to the Trash with them.
    let onDelete: (Set<UUID>, Bool) -> Void
    let onRename: (UUID, String) -> Void
    let onToggleFavorite: (UUID) -> Void
    let onQuickLook: (URL) -> Void
    let onReveal: ([URL]) -> Void
    /// "Save to…": copy these rows' artifacts somewhere the user chooses.
    let onExport: ([StudioLibraryItem]) -> Void
    let onRetry: (StudioLibraryItem) -> Void
    let onEdit: (StudioLibraryItem) -> Void
    /// Extra leading space for the header while the window's traffic lights sit over it.
    var leadingInset: CGFloat = 0

    @State private var searchText = ""
    @State private var renamingID: UUID?
    @State private var renameText = ""
    /// The batch ⌘/⇧ click builds. Always contains the open row when there is one.
    @State private var batch: Set<UUID> = []
    @State private var anchorID: UUID?
    @State private var pendingDelete: StudioLibraryDeleteRequest?
    @FocusState private var renameFocused: Bool
    @Environment(\.studioLibrarySeed) private var seed

    /// The layout the column draws in — the user's, unless a render is staging the other one.
    private var effectiveViewMode: StudioLibraryViewMode {
        seed?.viewMode ?? viewMode
    }

    private var effectiveScope: StudioLibraryScope {
        seed?.scope ?? scope
    }

    private var currentFilter: StudioLibraryFilter {
        StudioLibraryFilter(
            scope: effectiveScope,
            domain: domain,
            kind: kind,
            favoritesOnly: favoritesOnly,
            query: searchText
        )
    }

    private var scopedItems: [StudioLibraryItem] {
        StudioLibraryPresenter.scoped(items, scope: effectiveScope, domain: domain)
    }

    private var filteredItems: [StudioLibraryItem] {
        StudioLibraryPresenter.filter(items, with: currentFilter)
    }

    private var daySections: [(day: Date, title: String, items: [StudioLibraryItem])] {
        let calendar = Calendar.current
        var order: [Date] = []
        var grouped: [Date: [StudioLibraryItem]] = [:]
        for item in filteredItems {
            let day = calendar.startOfDay(for: item.createdAt)
            if grouped[day] == nil { order.append(day) }
            grouped[day, default: []].append(item)
        }
        return order.map { day in
            (day: day, title: Self.sectionFormatter.string(from: day), items: grouped[day] ?? [])
        }
    }

    private static let sectionFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .none
        formatter.doesRelativeDateFormatting = true
        return formatter
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            searchRow

            if filteredItems.isEmpty {
                emptyState
            } else {
                content
            }

            if batch.count > 1 {
                batchBar
            }
        }
        .background(MereRunTheme.background)
        .confirmationDialog(
            pendingDelete.map(\.title) ?? "",
            isPresented: deleteBinding,
            titleVisibility: .visible
        ) {
            Button("Delete and Move Files to Trash", role: .destructive) {
                if let pendingDelete { onDelete(pendingDelete.ids, true) }
                clearBatch()
            }
            Button("Delete, Keep Files", role: .destructive) {
                if let pendingDelete { onDelete(pendingDelete.ids, false) }
                clearBatch()
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        } message: {
            Text("Deleting removes the run from the Library. Its files stay on disk unless you move them to the Trash.")
        }
        .onAppear(perform: applyBatchSeed)
        .onChange(of: selectedID) { _, newValue in
            // A programmatic selection (a run finishing, a deep link) replaces the batch, so the
            // batch bar never claims rows the user can no longer see selected. A seeded batch is
            // the harness staging a render, and holds.
            guard seed?.batchCount == nil, let newValue, !batch.contains(newValue) else { return }
            batch = [newValue]
            anchorID = newValue
        }
    }

    private var deleteBinding: Binding<Bool> {
        Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } })
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 8) {
            Text("Library")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(MereRunTheme.textPrimary)
            Text("\(scopedItems.count)")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .accessibilityLabel("\(scopedItems.count) runs")
            Spacer(minLength: 0)
            MereSegmentedControl(
                StudioLibraryScope.allCases,
                selection: Binding(get: { effectiveScope }, set: { scope = $0 }),
                accessibilityLabel: "Library scope"
            ) { scope in
                scope == .domain ? domain.title : "All"
            }
        }
        .padding(.top, 14)
        .padding(.leading, 14 + leadingInset)
        .padding(.trailing, 14)
        .padding(.bottom, 8)
    }

    /// The search field with the kind/favorites filter and the list-or-grid toggle beside it, so
    /// the column keeps the header row the design calls for.
    private var searchRow: some View {
        HStack(spacing: 4) {
            searchField
            filterMenu
            viewModeButton
        }
        .frame(height: 28)
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
    }

    private var searchField: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            TextField("Search", text: $searchText)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .foregroundStyle(MereRunTheme.textPrimary)
            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 11))
                }
                .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, 10)
        .frame(maxWidth: .infinity)
        .frame(height: 28)
        .background {
            Capsule()
                .fill(MereRunTheme.surface)
                .overlay {
                    Capsule().strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                }
        }
    }

    private var isFiltering: Bool { kind != .all || favoritesOnly }

    private var filterMenu: some View {
        Menu {
            Picker("Kind", selection: $kind) {
                ForEach(StudioLibraryKind.allCases) { option in
                    Label(option.title, systemImage: option.systemImage).tag(option)
                }
            }
            .pickerStyle(.inline)
            Divider()
            Toggle("Favorites only", isOn: $favoritesOnly)
        } label: {
            Image(systemName: isFiltering ? "line.3.horizontal.decrease.circle.fill" : "line.3.horizontal.decrease.circle")
                .font(.system(size: 12, weight: .medium))
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .frame(width: 22, height: 24)
        .foregroundStyle(isFiltering ? MereRunTheme.accent : MereRunTheme.textMuted)
        .help("Filter the Library by kind or favorites")
        .accessibilityLabel("Filter")
        .accessibilityValue(favoritesOnly ? "\(kind.title), favorites only" : kind.title)
    }

    private var viewModeButton: some View {
        Button {
            viewMode = viewMode == .list ? .grid : .list
        } label: {
            Image(systemName: effectiveViewMode == .list ? StudioLibraryViewMode.grid.systemImage : StudioLibraryViewMode.list.systemImage)
                .font(.system(size: 12, weight: .medium))
                .frame(width: 22, height: 24)
        }
        .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
        .help(effectiveViewMode == .list ? "Show as a grid" : "Show as a list")
        .accessibilityLabel(effectiveViewMode == .list ? "Show as a grid" : "Show as a list")
    }

    private var emptyState: some View {
        VStack(spacing: MereRunTheme.Spacing.sm) {
            Image(systemName: "rectangle.stack")
                .font(.system(size: 28, weight: .semibold))
                .foregroundStyle(MereRunTheme.textMuted)
            Text(emptyMessage)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(MereRunTheme.Spacing.xl)
    }

    private var emptyMessage: String {
        if items.isEmpty { return "Runs you create will land here." }
        if scopedItems.isEmpty { return "No \(domain.title) runs yet. Choose All to see every run." }
        if favoritesOnly { return "No favorites here yet. Star a run to keep it close." }
        if kind != .all { return "No \(kind.title.lowercased()) here. Choose All kinds to see every run." }
        return "No matching runs."
    }

    // MARK: - Rows and tiles

    private var content: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 2, pinnedViews: []) {
                ForEach(daySections, id: \.day) { section in
                    MereEyebrow(section.title)
                        .padding(.horizontal, 8)
                        .padding(.top, 6)
                        .padding(.bottom, 2)

                    if effectiveViewMode == .list {
                        ForEach(section.items) { item in
                            row(for: item)
                        }
                    } else {
                        grid(for: section.items)
                    }
                }
            }
            .padding(.horizontal, 6)
            .padding(.bottom, MereRunTheme.Spacing.sm)
        }
        .focusable()
        // Finder-style keys: arrows move the selection, Space previews the selected output
        // (ignored when it has none, so it never swallows the key destructively).
        .onKeyPress(.space) {
            guard let id = selectedID,
                  let url = items.first(where: { $0.id == id })?.outputURL else { return .ignored }
            onQuickLook(url)
            return .handled
        }
        .onKeyPress(.upArrow) { moveSelection(by: -1) }
        .onKeyPress(.downArrow) { moveSelection(by: 1) }
    }

    private func row(for item: StudioLibraryItem) -> some View {
        StudioLibraryRow(
            item: item,
            progress: progressByID[item.id],
            isSelected: isSelected(item.id),
            isRenaming: renamingID == item.id,
            renameText: $renameText,
            renameFocused: $renameFocused,
            onQuickLook: item.outputURL.map { url in { onQuickLook(url) } },
            onToggleFavorite: { onToggleFavorite(item.id) },
            onCommitRename: { commitRename(for: item) },
            onCancelRename: { renamingID = nil },
            action: { click(item) }
        )
        .modifier(StudioLibraryDragOut(url: item.outputURL))
        .contextMenu { menu(for: item) }
    }

    private func grid(for sectionItems: [StudioLibraryItem]) -> some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 3), spacing: 6) {
            ForEach(sectionItems) { item in
                StudioLibraryTile(
                    item: item,
                    progress: progressByID[item.id],
                    isSelected: isSelected(item.id),
                    onToggleFavorite: { onToggleFavorite(item.id) },
                    action: { click(item) }
                )
                .modifier(StudioLibraryDragOut(url: item.outputURL))
                .contextMenu { menu(for: item) }
            }
        }
        .padding(.horizontal, 2)
        .padding(.bottom, 4)
    }

    @ViewBuilder
    private func menu(for item: StudioLibraryItem) -> some View {
        if batch.count > 1, batch.contains(item.id) {
            Button("Reveal \(batch.count) in Finder") { onReveal(urls(in: batch)) }
            Button("Save \(batch.count) to…") { onExport(items(in: batch)) }
            Divider()
            Button("Delete \(batch.count)…", role: .destructive) {
                pendingDelete = StudioLibraryDeleteRequest(ids: batch, count: batch.count)
            }
        } else {
            if let url = item.outputURL {
                Button("Quick Look") { onQuickLook(url) }
                Button("Reveal in Finder") { onReveal([url]) }
                Button("Save to…") { onExport([item]) }
            }
            Button(item.isStarred ? "Remove from Favorites" : "Add to Favorites") {
                onToggleFavorite(item.id)
            }
            if item.commandDraft != nil, item.templateID != nil {
                Divider()
                Button("Run again") { onRetry(item) }
                Button("Edit command…") { onEdit(item) }
            }
            Divider()
            Button("Rename") { beginRename(item) }
            Button("Delete…", role: .destructive) {
                pendingDelete = StudioLibraryDeleteRequest(ids: [item.id], count: 1)
            }
        }
    }

    private var batchBar: some View {
        HStack(spacing: 6) {
            Text("\(batch.count) selected")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MereRunTheme.textSecondary)
            Spacer(minLength: 0)
            // Icons, not labels: three words do not fit a 248pt column beside the count.
            batchAction(systemImage: "folder", label: "Reveal in Finder", tint: MereRunTheme.textSecondary) {
                onReveal(urls(in: batch))
            }
            .disabled(urls(in: batch).isEmpty)
            batchAction(systemImage: "square.and.arrow.down", label: "Save to…", tint: MereRunTheme.textSecondary) {
                onExport(items(in: batch))
            }
            .disabled(urls(in: batch).isEmpty)
            batchAction(systemImage: "trash", label: "Delete \(batch.count) runs", tint: MereRunTheme.red) {
                pendingDelete = StudioLibraryDeleteRequest(ids: batch, count: batch.count)
            }
        }
        .padding(.horizontal, 12)
        .frame(height: 36)
        .background(MereRunTheme.surface)
        .overlay(alignment: .top) {
            Rectangle().fill(MereRunTheme.border.opacity(0.55)).frame(height: 1)
        }
    }

    private func batchAction(
        systemImage: String,
        label: String,
        tint: Color,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 12, weight: .medium))
                .frame(width: 26, height: 24)
        }
        .buttonStyle(.mereIcon(tint: tint))
        .help(label)
        .accessibilityLabel(label)
    }

    // MARK: - Selection

    private func applyBatchSeed() {
        guard let batchCount = seed?.batchCount, batchCount > 1 else { return }
        let visible = filteredItems.prefix(batchCount).map(\.id)
        batch = Set(visible)
        anchorID = visible.first
    }

    private func isSelected(_ id: UUID) -> Bool {
        selectedID == id || batch.contains(id)
    }

    private func click(_ item: StudioLibraryItem) {
        let result = StudioLibrarySelection.click(
            on: item.id,
            visible: filteredItems.map(\.id),
            selection: batch,
            anchor: anchorID,
            modifiers: StudioLibrarySelection.Modifiers(event: NSEvent.modifierFlags)
        )
        batch = result.selection
        anchorID = result.anchor
        if result.opened != nil { onSelect(item) }
    }

    private func clearBatch() {
        pendingDelete = nil
        batch = selectedID.map { [$0] } ?? []
    }

    private func moveSelection(by offset: Int) -> KeyPress.Result {
        let visible = filteredItems
        guard !visible.isEmpty else { return .ignored }
        guard let selectedID, let index = visible.firstIndex(where: { $0.id == selectedID }) else {
            if let edge = offset > 0 ? visible.first : visible.last { select(edge) }
            return .handled
        }
        let next = min(max(index + offset, 0), visible.count - 1)
        select(visible[next])
        return .handled
    }

    private func select(_ item: StudioLibraryItem) {
        batch = [item.id]
        anchorID = item.id
        onSelect(item)
    }

    private func items(in ids: Set<UUID>) -> [StudioLibraryItem] {
        filteredItems.filter { ids.contains($0.id) }
    }

    private func urls(in ids: Set<UUID>) -> [URL] {
        items(in: ids).compactMap(\.outputURL)
    }

    // MARK: - Rename

    private func beginRename(_ item: StudioLibraryItem) {
        renameText = item.displayTitle
        renamingID = item.id
        renameFocused = true
    }

    private func commitRename(for item: StudioLibraryItem) {
        guard renamingID == item.id else { return }
        onRename(item.id, renameText)
        renamingID = nil
    }
}

/// The state a render wants the column in. Only the snapshot harness sets this — the column's own
/// view mode lives in `@SceneStorage`, which no scene backs offscreen, and a batch is built by
/// ⌘-clicking, which a render cannot do. The app leaves it nil.
struct StudioLibrarySeed: Equatable {
    var viewMode: StudioLibraryViewMode?
    var scope: StudioLibraryScope?
    /// How many of the visible rows start out selected together.
    var batchCount: Int?
}

private struct StudioLibrarySeedKey: EnvironmentKey {
    static let defaultValue: StudioLibrarySeed? = nil
}

extension EnvironmentValues {
    var studioLibrarySeed: StudioLibrarySeed? {
        get { self[StudioLibrarySeedKey.self] }
        set { self[StudioLibrarySeedKey.self] = newValue }
    }
}

/// One pending Delete, so the dialog knows how many rows it is about.
private struct StudioLibraryDeleteRequest: Identifiable {
    let ids: Set<UUID>
    let count: Int

    var id: Int { ids.hashValue }

    var title: String {
        count == 1 ? "Delete this run?" : "Delete \(count) runs?"
    }
}

/// Drag a row or tile straight into Finder, Mail, or another app. Rows with no file are inert
/// rather than dragging an empty promise.
private struct StudioLibraryDragOut: ViewModifier {
    let url: URL?

    func body(content: Content) -> some View {
        if let url {
            content.onDrag { NSItemProvider(contentsOf: url) ?? NSItemProvider() }
        } else {
            content
        }
    }
}

/// One Library row: a 40pt thumbnail (or a glyph tile), the title on one line, and a meta line
/// that carries a status dot while the run is queued, running, or failed. Hovering reveals the
/// favorite star and Quick Look; renaming happens in place, never in a dialog.
private struct StudioLibraryRow: View {
    let item: StudioLibraryItem
    let progress: StudioRunProgress?
    let isSelected: Bool
    let isRenaming: Bool
    @Binding var renameText: String
    var renameFocused: FocusState<Bool>.Binding
    let onQuickLook: (() -> Void)?
    let onToggleFavorite: () -> Void
    let onCommitRename: () -> Void
    let onCancelRename: () -> Void
    let action: () -> Void

    @State private var hovering = false

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        return formatter
    }()

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                StudioLibraryThumbnail(item: item, side: 40)
                    .frame(width: 40, height: 40)
                    .background(MereRunTheme.surfaceRaised)
                    .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))

                VStack(alignment: .leading, spacing: 1) {
                    if isRenaming {
                        TextField("Name", text: $renameText)
                            .textFieldStyle(.plain)
                            .font(.system(size: 12.5, weight: .medium))
                            .foregroundStyle(MereRunTheme.textPrimary)
                            .focused(renameFocused)
                            .onSubmit(onCommitRename)
                            .onExitCommand(perform: onCancelRename)
                            .onChange(of: renameFocused.wrappedValue) { _, focused in
                                if !focused { onCommitRename() }
                            }
                    } else {
                        Text(item.displayTitle)
                            .font(.system(size: 12.5, weight: .medium))
                            .foregroundStyle(MereRunTheme.textPrimary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                    HStack(spacing: 4) {
                        if let statusColor {
                            Circle()
                                .fill(statusColor)
                                .frame(width: 8, height: 8)
                        }
                        Text(meta)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(MereRunTheme.textMuted)
                            .lineLimit(1)
                    }
                }

                Spacer(minLength: 0)

                if hovering || item.isStarred {
                    Button(action: onToggleFavorite) {
                        Image(systemName: item.isStarred ? "star.fill" : "star")
                            .font(.system(size: 11, weight: .semibold))
                            .frame(width: 20, height: 24)
                    }
                    .buttonStyle(.mereIcon(tint: item.isStarred ? MereRunTheme.yellow : MereRunTheme.textMuted))
                    .help(item.isStarred ? "Remove from Favorites" : "Add to Favorites")
                    .accessibilityLabel(item.isStarred ? "Remove from Favorites" : "Add to Favorites")
                    .transition(.opacity)
                }

                if hovering, let onQuickLook {
                    Button {
                        onQuickLook()
                    } label: {
                        Image(systemName: "eye")
                            .font(.system(size: 11, weight: .semibold))
                            .frame(width: 24, height: 24)
                    }
                    .buttonStyle(.mereIcon)
                    .help("Quick Look (Space)")
                    .accessibilityLabel("Quick Look")
                    .transition(.opacity)
                }
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 8)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(rowFill)
            }
            .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(item.displayKindTitle), \(item.status.rawValue), \(item.displayTitle)")
        .accessibilityValue(item.isStarred ? "\(meta), favorite" : meta)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private var rowFill: Color {
        if isSelected { return MereRunTheme.accentSoft }
        if hovering { return MereRunTheme.hoverFill }
        return .clear
    }

    /// The task that made the row ("Generate", "Find"), or the command's own title when a
    /// specialist command other than the mode's prompt template produced it.
    private var kindTitle: String {
        if let templateID = item.templateID, templateID != item.mode.defaultTemplateID {
            return item.displayKindTitle
        }
        return item.mode.destination.task.title
    }

    /// "Generate · 12:43 PM" once done; "Running · 62%" / "Queued" / "Failed · 12:43 PM" otherwise.
    private var meta: String {
        StudioLibraryRowMeta.text(
            item: item,
            kindTitle: kindTitle,
            progress: progress,
            formatter: Self.timeFormatter
        )
    }

    private var statusColor: Color? {
        StudioLibraryRowMeta.statusColor(for: item.status)
    }
}

/// One grid tile: the thumbnail alone until the pointer arrives, then the title over it.
private struct StudioLibraryTile: View {
    let item: StudioLibraryItem
    let progress: StudioRunProgress?
    let isSelected: Bool
    let onToggleFavorite: () -> Void
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            // The square comes from an empty spacer, not from the thumbnail: a `scaledToFill`
            // picture has no size of its own to fit, and would otherwise stretch the tile.
            Color.clear
                .frame(maxWidth: .infinity)
                .aspectRatio(1, contentMode: .fit)
                .background(MereRunTheme.surfaceRaised)
                .overlay { StudioLibraryThumbnail(item: item, side: 72) }
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))
                .overlay(alignment: .bottom) {
                    if hovering {
                        Text(item.displayTitle)
                            .font(.system(size: 9.5, weight: .medium))
                            .foregroundStyle(.white)
                            .lineLimit(2)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 3)
                            .background(Color.black.opacity(0.55))
                            .transition(.opacity)
                    }
                }
                .overlay(alignment: .topLeading) {
                    if let statusColor = StudioLibraryRowMeta.statusColor(for: item.status) {
                        Circle()
                            .fill(statusColor)
                            .frame(width: 7, height: 7)
                            .padding(4)
                    }
                }
                .overlay(alignment: .topTrailing) {
                    if hovering || item.isStarred {
                        Button(action: onToggleFavorite) {
                            Image(systemName: item.isStarred ? "star.fill" : "star")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundStyle(item.isStarred ? MereRunTheme.yellow : .white)
                                .padding(3)
                                .background(Circle().fill(Color.black.opacity(0.4)))
                        }
                        .buttonStyle(.plain)
                        .padding(3)
                        .help(item.isStarred ? "Remove from Favorites" : "Add to Favorites")
                        .accessibilityLabel(item.isStarred ? "Remove from Favorites" : "Add to Favorites")
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                        .strokeBorder(
                            isSelected ? MereRunTheme.accent : MereRunTheme.border.opacity(hovering ? 0.9 : 0.5),
                            lineWidth: isSelected ? 2 : 1
                        )
                }
                .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(item.displayKindTitle), \(item.status.rawValue), \(item.displayTitle)")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }
}

/// The row meta line and status dot, shared by the list rows and the grid tiles.
enum StudioLibraryRowMeta {
    static func text(
        item: StudioLibraryItem,
        kindTitle: String,
        progress: StudioRunProgress?,
        formatter: DateFormatter
    ) -> String {
        switch item.status {
        case .completed:
            let origin = item.source.map { "\($0.title) · " } ?? ""
            return "\(origin)\(kindTitle) · \(formatter.string(from: item.createdAt))"
        case .running:
            if let fraction = progress?.fractionCompleted {
                return "Running · \(Int((fraction * 100).rounded()))%"
            }
            if let detail = progress?.detail { return "Running · \(detail)" }
            return "Running"
        case .queued:
            return "Queued"
        case .failed:
            return "Failed · \(formatter.string(from: item.createdAt))"
        }
    }

    static func statusColor(for status: StudioLibraryStatus) -> Color? {
        switch status {
        case .queued: return MereRunTheme.yellow
        case .running: return MereRunTheme.accent
        case .completed: return nil
        case .failed: return MereRunTheme.red
        }
    }
}
