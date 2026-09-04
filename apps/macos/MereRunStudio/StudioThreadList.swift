import SwiftUI

/// Pure rules for the Converse thread list: which Library rows are threads, how they group,
/// and what their meta line says. Everything here is unit-testable without a view.
enum StudioThreadListPresenter {
    struct Section: Equatable {
        let title: String
        let threads: [StudioLibraryItem]
    }

    /// The conversation rows, most recently active first. Chat and Code threads share one list:
    /// Code is a preset inside Converse, not a second thread pool.
    static func threads(in items: [StudioLibraryItem]) -> [StudioLibraryItem] {
        items.filter(\.isConversation).sorted { $0.updatedAt > $1.updatedAt }
    }

    /// The rows the media Library shows: everything that is not a thread.
    static func mediaItems(in items: [StudioLibraryItem]) -> [StudioLibraryItem] {
        items.filter { !$0.isConversation }
    }

    /// Threads whose title or any turn contains `query` (case-insensitive); all of them for an
    /// empty query.
    static func filter(_ threads: [StudioLibraryItem], query: String) -> [StudioLibraryItem] {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !needle.isEmpty else { return threads }
        return threads.filter { thread in
            thread.displayTitle.lowercased().contains(needle)
                || (thread.messages ?? []).contains { $0.content.lowercased().contains(needle) }
        }
    }

    /// "Today" for threads active today, "Earlier" for the rest, in that order; a group is
    /// omitted when empty.
    static func sections(
        _ threads: [StudioLibraryItem],
        now: Date = Date(),
        calendar: Calendar = .current
    ) -> [Section] {
        let today = threads.filter { calendar.isDate($0.updatedAt, inSameDayAs: now) }
        let earlier = threads.filter { !calendar.isDate($0.updatedAt, inSameDayAs: now) }
        var sections: [Section] = []
        if !today.isEmpty { sections.append(Section(title: "Today", threads: today)) }
        if !earlier.isEmpty { sections.append(Section(title: "Earlier", threads: earlier)) }
        return sections
    }

    /// "Qwen3.6 4B · 1:20 PM" for a chat thread active today; "Code · Gemma 4 · Yesterday" for a
    /// Code thread from yesterday; older threads show their day ("Aug 30").
    static func meta(
        for thread: StudioLibraryItem,
        now: Date = Date(),
        calendar: Calendar = .current
    ) -> String {
        var parts: [String] = []
        if thread.mode == .code { parts.append("Code") }
        parts.append(modelLabel(for: thread))
        parts.append(activityLabel(for: thread.updatedAt, now: now, calendar: calendar))
        return parts.joined(separator: " · ")
    }

    /// The friendly name of the model the thread last ran with, or the preset's default.
    static func modelLabel(for thread: StudioLibraryItem) -> String {
        let identity = StudioModelNaming.resolvedModelID(for: thread.mode, model: thread.model ?? "")
        return identity.isEmpty ? "Auto" : StudioModelNaming.displayName(identity)
    }

    static func activityLabel(for date: Date, now: Date, calendar: Calendar) -> String {
        if calendar.isDate(date, inSameDayAs: now) {
            return timeFormatter.string(from: date)
        }
        if let yesterday = calendar.date(byAdding: .day, value: -1, to: now),
           calendar.isDate(date, inSameDayAs: yesterday) {
            return "Yesterday"
        }
        return dayFormatter.string(from: date)
    }

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
}

/// The Converse column beside the transcript: every thread, searchable, grouped Today / Earlier,
/// with a compose button for a new thread (also ⌘N). Replaces the Library column in the Chat
/// domain; threads never appear in the media Library.
struct StudioThreadList: View {
    /// Every Library row; the list keeps the threads.
    let items: [StudioLibraryItem]
    let selectedID: UUID?
    /// A row the user picked (click or arrow keys), as opposed to a programmatic selection.
    let onSelect: (StudioLibraryItem) -> Void
    let onNewThread: () -> Void
    let onDelete: (UUID) -> Void
    let onRename: (UUID, String) -> Void
    /// Extra leading space for the header while the window's traffic lights sit over it.
    var leadingInset: CGFloat = 0

    @State private var searchText = ""
    @State private var renamingID: UUID?
    @State private var renameText = ""

    private var threads: [StudioLibraryItem] {
        StudioThreadListPresenter.threads(in: items)
    }

    private var visibleThreads: [StudioLibraryItem] {
        StudioThreadListPresenter.filter(threads, query: searchText)
    }

    private var sections: [StudioThreadListPresenter.Section] {
        StudioThreadListPresenter.sections(visibleThreads)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            searchField
            if visibleThreads.isEmpty {
                emptyState
            } else {
                list
            }
        }
        .background(MereRunTheme.background)
        .alert("Rename thread", isPresented: renameBinding) {
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
        HStack(spacing: 8) {
            Text("Threads")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(MereRunTheme.textPrimary)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 0)
            Button(action: onNewThread) {
                Image(systemName: "square.and.pencil")
                    .font(.system(size: 14, weight: .medium))
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.mereIcon)
            .help("New thread (⌘N)")
            .accessibilityLabel("New thread")
        }
        .padding(.top, 14)
        .padding(.leading, 14 + leadingInset)
        .padding(.trailing, 14)
        .padding(.bottom, 8)
    }

    private var searchField: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            TextField("Search threads", text: $searchText)
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
        .frame(height: 28)
        .background {
            Capsule()
                .fill(MereRunTheme.surface)
                .overlay {
                    Capsule().strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                }
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
    }

    private var emptyState: some View {
        VStack(spacing: MereRunTheme.Spacing.sm) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 28, weight: .semibold))
                .foregroundStyle(MereRunTheme.textMuted)
            Text(threads.isEmpty ? "Threads you start will land here." : "No matching threads.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(MereRunTheme.Spacing.xl)
    }

    private var list: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 2) {
                ForEach(Array(sections.enumerated()), id: \.element.title) { index, section in
                    MereEyebrow(section.title)
                        .padding(.horizontal, 8)
                        .padding(.top, index == 0 ? 6 : 10)
                        .padding(.bottom, 2)

                    ForEach(section.threads) { thread in
                        StudioThreadRow(thread: thread, isSelected: selectedID == thread.id) {
                            onSelect(thread)
                        }
                        .contextMenu {
                            Button("Rename") {
                                renameText = thread.displayTitle
                                renamingID = thread.id
                            }
                            Button("Delete", role: .destructive) {
                                onDelete(thread.id)
                            }
                        }
                    }
                }
            }
            .padding(.horizontal, 6)
            .padding(.bottom, MereRunTheme.Spacing.sm)
        }
        .focusable()
        .onKeyPress(.upArrow) { moveSelection(by: -1) }
        .onKeyPress(.downArrow) { moveSelection(by: 1) }
    }

    private func moveSelection(by offset: Int) -> KeyPress.Result {
        let visible = visibleThreads
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

/// One thread: the title on one line and a meta line naming the preset, model, and last activity.
private struct StudioThreadRow: View {
    let thread: StudioLibraryItem
    let isSelected: Bool
    let action: () -> Void

    @State private var hovering = false

    private var meta: String { StudioThreadListPresenter.meta(for: thread) }

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 2) {
                Text(thread.displayTitle)
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Text(meta)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 8)
            .padding(.horizontal, 10)
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
        .accessibilityLabel("Thread, \(thread.displayTitle)")
        .accessibilityValue(meta)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private var rowFill: Color {
        if isSelected { return MereRunTheme.accentSoft }
        if hovering { return MereRunTheme.hoverFill }
        return .clear
    }
}
