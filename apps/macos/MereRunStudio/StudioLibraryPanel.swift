import AppKit
import SwiftUI

/// Run history as a column beside the current task: filtered to the current domain (or All),
/// searchable, grouped by day, keyboard-navigable (arrows move, Space previews), with Quick Look
/// surfacing on hover. Picking a row of another domain switches the destination.
struct StudioLibraryPanel: View {
    let items: [StudioLibraryItem]
    let domain: StudioDomain
    @Binding var scope: StudioLibraryScope
    let progressByID: [UUID: StudioRunProgress]
    @Binding var selectedID: UUID?
    /// A row the user picked (click or arrow keys), as opposed to a programmatic selection.
    let onSelect: (StudioLibraryItem) -> Void
    let onDelete: (UUID) -> Void
    let onRename: (UUID, String) -> Void
    let onQuickLook: (URL) -> Void
    let onRetry: (StudioLibraryItem) -> Void
    let onEdit: (StudioLibraryItem) -> Void

    @State private var searchText = ""
    @State private var renamingID: UUID?
    @State private var renameText = ""

    private var scopedItems: [StudioLibraryItem] {
        switch scope {
        case .all: return items
        case .domain: return items.filter { $0.mode.destination.domain == domain }
        }
    }

    private var filteredItems: [StudioLibraryItem] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return scopedItems }
        return scopedItems.filter { item in
            item.displayTitle.lowercased().contains(query)
                || item.displayKindTitle.lowercased().contains(query)
                || item.prompt.lowercased().contains(query)
        }
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

            Divider()
                .overlay(MereRunTheme.border.opacity(0.5))

            searchField

            if filteredItems.isEmpty {
                emptyState
            } else {
                list
            }
        }
        .background(MereRunTheme.background)
        .alert("Rename run", isPresented: renameBinding) {
            TextField("Name", text: $renameText)
            Button("Save") {
                if let renamingID { onRename(renamingID, renameText) }
                renamingID = nil
            }
            Button("Cancel", role: .cancel) { renamingID = nil }
        }
    }

    private var renameBinding: Binding<Bool> {
        Binding(get: { renamingID != nil }, set: { if !$0 { renamingID = nil } })
    }

    private var header: some View {
        HStack(spacing: MereRunTheme.Spacing.xs) {
            Text("Library")
                .font(.system(size: 15, weight: .semibold))
            Text("\(scopedItems.count)")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            Spacer()
            Picker("Scope", selection: $scope) {
                Text(domain.title).tag(StudioLibraryScope.domain)
                Text("All").tag(StudioLibraryScope.all)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.small)
            .fixedSize()
            .accessibilityLabel("Library scope")
        }
        .padding(.horizontal, MereRunTheme.Spacing.md)
        .frame(height: 52)
    }

    private var searchField: some View {
        HStack(spacing: 7) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            TextField("Search runs", text: $searchText)
                .textFieldStyle(.plain)
                .font(.system(size: 12.5))
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
        .padding(.horizontal, MereRunTheme.Spacing.sm)
        .frame(height: 30)
        .background {
            Capsule()
                .fill(MereRunTheme.surface)
                .overlay {
                    Capsule().strokeBorder(MereRunTheme.border.opacity(0.7), lineWidth: 1)
                }
        }
        .padding(.horizontal, MereRunTheme.Spacing.sm)
        .padding(.vertical, MereRunTheme.Spacing.sm)
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
        return "No matching runs."
    }

    private var list: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 4, pinnedViews: []) {
                ForEach(daySections, id: \.day) { section in
                    Text(section.title)
                        .font(.system(size: 10.5, weight: .semibold))
                        .kerning(0.5)
                        .foregroundStyle(MereRunTheme.textMuted)
                        .textCase(.uppercase)
                        .padding(.horizontal, 10)
                        .padding(.top, MereRunTheme.Spacing.sm)
                        .padding(.bottom, 2)
                        .accessibilityAddTraits(.isHeader)

                    ForEach(section.items) { item in
                        StudioLibraryRow(
                            item: item,
                            progress: progressByID[item.id],
                            isSelected: selectedID == item.id,
                            onQuickLook: item.outputURL.map { url in { onQuickLook(url) } }
                        ) {
                            onSelect(item)
                        }
                        .contextMenu {
                            if let url = item.outputURL {
                                Button("Quick Look") { onQuickLook(url) }
                            }
                            if item.commandDraft != nil, item.templateID != nil {
                                Button("Run again") { onRetry(item) }
                                Button("Edit command…") { onEdit(item) }
                            }
                            Button("Rename") {
                                renameText = item.displayTitle
                                renamingID = item.id
                            }
                            Button("Delete", role: .destructive) {
                                onDelete(item.id)
                            }
                        }
                    }
                }
            }
            .padding(.horizontal, MereRunTheme.Spacing.xs)
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

    private func moveSelection(by offset: Int) -> KeyPress.Result {
        let visible = filteredItems
        guard !visible.isEmpty else { return .ignored }
        guard let selectedID, let index = visible.firstIndex(where: { $0.id == selectedID }) else {
            if let edge = offset > 0 ? visible.first : visible.last { onSelect(edge) }
            return .handled
        }
        let next = min(max(index + offset, 0), visible.count - 1)
        onSelect(visible[next])
        return .handled
    }
}

private struct StudioLibraryRow: View {
    let item: StudioLibraryItem
    let progress: StudioRunProgress?
    let isSelected: Bool
    let onQuickLook: (() -> Void)?
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
            HStack(spacing: MereRunTheme.Spacing.sm) {
                thumbnail
                    .frame(width: 52, height: 42)
                    .background(isSelected ? MereRunTheme.surface : MereRunTheme.accentSoft.opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))

                VStack(alignment: .leading, spacing: 3) {
                    Text(item.displayTitle)
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                    HStack(spacing: 4) {
                        if item.status != .completed {
                            Circle()
                                .fill(statusColor)
                                .frame(width: 5, height: 5)
                        }
                        Text(subtitle)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .lineLimit(1)
                    }
                    if item.status == .running, let progress {
                        ProgressView(value: progress.fractionCompleted)
                            .progressViewStyle(.linear)
                            .tint(MereRunTheme.accent)
                            .accessibilityLabel(progress.label)
                            .accessibilityValue(progressAccessibilityValue(progress))
                    }
                }

                Spacer(minLength: 0)

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
            .padding(6)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .fill(rowFill)
                    .overlay {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                            .strokeBorder(
                                isSelected ? MereRunTheme.accent.opacity(0.4) : Color.clear,
                                lineWidth: 1
                            )
                    }
            }
            .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(item.displayKindTitle), \(item.status.rawValue), \(item.displayTitle)")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private var rowFill: Color {
        if isSelected { return MereRunTheme.accentSoft }
        if hovering { return MereRunTheme.hoverFill }
        return .clear
    }

    private var subtitle: String {
        if item.status == .completed {
            let origin = item.source.map { "\($0.title) · " } ?? ""
            return "\(origin)\(item.displayKindTitle) · \(Self.timeFormatter.string(from: item.createdAt))"
        }
        return "\(item.displayKindTitle) · \(item.status.rawValue)"
    }

    private var statusColor: Color {
        switch item.status {
        case .queued: return MereRunTheme.textMuted
        case .running: return MereRunTheme.yellow
        case .completed: return MereRunTheme.green
        case .failed: return MereRunTheme.red
        }
    }

    private func progressAccessibilityValue(_ progress: StudioRunProgress) -> String {
        if let detail = progress.detail { return detail }
        guard let fraction = progress.fractionCompleted else { return "In progress" }
        return "\(Int((fraction * 100).rounded())) percent"
    }

    @ViewBuilder
    private var thumbnail: some View {
        if let url = item.outputURL, StudioOutputFileKind.classify(url) == .image {
            StudioAsyncImagePreview(
                url: url,
                maxPixelSize: 180,
                contentMode: .fill,
                fallbackSystemImage: item.displaySystemImage
            )
        } else {
            Image(systemName: item.displaySystemImage)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(MereRunTheme.accent)
        }
    }
}
