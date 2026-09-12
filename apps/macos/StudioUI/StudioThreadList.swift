import StudioKit
import SwiftUI

/// The Converse column beside the transcript: every thread, searchable, grouped Today / Earlier,
/// with a compose button for a new thread (also ⌘N). Replaces the Library column in the Chat
/// domain; threads never appear in the media Library.
struct StudioThreadList: View {
    @Environment(\.studioReferenceDate) private var referenceDate

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
        StudioThreadListPresenter.sections(visibleThreads, now: referenceDate ?? Date())
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
                .font(.callout.weight(.semibold))
                .foregroundStyle(MereRunTheme.textPrimary)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 0)
            Button(action: onNewThread) {
                Image(systemName: "square.and.pencil")
                    .font(.body.weight(.medium))
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
                .font(.callout.weight(.medium))
                .foregroundStyle(MereRunTheme.textMuted)
            TextField("Search threads", text: $searchText)
                .textFieldStyle(.plain)
                .font(.callout)
                .foregroundStyle(MereRunTheme.textPrimary)
            if !searchText.isEmpty {
                Button {
                    searchText = ""
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
    @Environment(\.studioReferenceDate) private var referenceDate

    let thread: StudioLibraryItem
    let isSelected: Bool
    let action: () -> Void

    @State private var hovering = false

    private var meta: String { StudioThreadListPresenter.meta(for: thread, now: referenceDate ?? Date()) }

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 2) {
                Text(thread.displayTitle)
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Text(meta)
                    .font(.caption.weight(.medium))
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
