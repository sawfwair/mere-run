import SwiftUI

/// The 52pt header row at the top of the content column, beside the Library: the domain glyph,
/// title, and one-line subtitle leading; the task control centered; the Library, Inspector, and
/// Command toggles trailing; a hairline below. It lives in the content column, not the window
/// toolbar, so the Library column can run to the top of the window.
struct StudioContentHeader: View {
    let domain: StudioDomain
    /// The one-line subtitle under the title: the domain's tagline, or a live fact such as the
    /// Models inventory.
    let subtitle: String
    @Binding var task: StudioTask
    let showsLibraryToggle: Bool
    let isLibraryShown: Bool
    let isConsoleOpen: Bool
    /// Extra leading space when the traffic lights and sidebar toggle sit over this header.
    var leadingInset: CGFloat = 0
    let onToggleLibrary: () -> Void
    let onOpenConsole: () -> Void

    static let height: CGFloat = 52
    /// Clears the traffic lights and the sidebar toggle while the sidebar is collapsed.
    static let collapsedSidebarInset: CGFloat = 112

    var body: some View {
        // Balanced like the design (220pt leading and trailing blocks, so the pill sits at the
        // column's center) when the column is wide enough; otherwise the blocks shrink to their
        // content so the pill never truncates at the default window width.
        ViewThatFits(in: .horizontal) {
            row(sideBlockWidth: 220)
            row(sideBlockWidth: nil)
        }
        .padding(.leading, 18 + leadingInset)
        .padding(.trailing, 16)
        .frame(height: Self.height)
        .frame(maxWidth: .infinity)
        .background(MereRunTheme.background)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(MereRunTheme.border.opacity(0.53))
                .frame(height: 1)
        }
    }

    private func row(sideBlockWidth: CGFloat?) -> some View {
        HStack(spacing: 12) {
            title
                .frame(minWidth: sideBlockWidth, alignment: .leading)
            Spacer(minLength: 0)
            taskControl
                .fixedSize()
            Spacer(minLength: 0)
            toggles
                .frame(minWidth: sideBlockWidth, alignment: .trailing)
        }
    }

    private var title: some View {
        HStack(spacing: MereRunTheme.Spacing.sm) {
            Image(systemName: domain.systemImage)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(MereRunTheme.accent)
                .frame(width: 16, height: 16)

            VStack(alignment: .leading, spacing: 2) {
                Text(domain.title)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textPrimary)
                Text(subtitle)
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }

    @ViewBuilder
    private var taskControl: some View {
        switch StudioTaskControlStyle.style(for: domain) {
        case .none:
            EmptyView()
        case .segmented:
            StudioTaskControl(domain: domain, selection: $task, visibleLimit: nil)
        case .segmentedWithOverflow(let visible):
            StudioTaskControl(domain: domain, selection: $task, visibleLimit: visible)
        }
    }

    private var toggles: some View {
        HStack(spacing: 4) {
            if showsLibraryToggle {
                MereToolbarIconButton(
                    systemImage: "sidebar.left",
                    isActive: isLibraryShown,
                    action: onToggleLibrary
                )
                .help(isLibraryShown ? "Hide Library (⌥⌘L)" : "Show Library (⌥⌘L)")
                .accessibilityLabel(isLibraryShown ? "Hide Library" : "Show Library")
            }

            // The inspector column arrives with the feed canvas; the toggle holds its place.
            MereToolbarIconButton(systemImage: "sidebar.right", isActive: false, action: {})
                .disabled(true)
                .opacity(0.45)
                .help("Inspector — coming soon")
                .accessibilityLabel("Inspector")
                .accessibilityHint("Not available yet")

            MereToolbarIconButton(
                systemImage: "terminal",
                isActive: isConsoleOpen,
                action: onOpenConsole
            )
            .help("Command Console (⌥⌘C)")
            .accessibilityLabel("Command Console")
        }
    }
}

/// The header's task control: one `MereSegmentedControl`-styled pill listing the domain's tasks.
/// Domains with more tasks than fit show the first `visibleLimit` as segments and the rest behind
/// a "More" segment that opens a menu; when the current task is one of those, the segment shows
/// its title and draws selected, so the header always names where you are.
struct StudioTaskControl: View {
    let domain: StudioDomain
    @Binding var selection: StudioTask
    /// nil shows every task as a segment.
    let visibleLimit: Int?

    private var visibleTasks: [StudioTask] {
        guard let visibleLimit else { return domain.tasks }
        return Array(domain.tasks.prefix(visibleLimit))
    }

    private var overflowTasks: [StudioTask] {
        guard let visibleLimit else { return [] }
        return Array(domain.tasks.dropFirst(visibleLimit))
    }

    private var overflowSelection: StudioTask? {
        overflowTasks.first { $0 == selection }
    }

    var body: some View {
        HStack(spacing: 2) {
            ForEach(visibleTasks) { task in
                MereSegment(title: task.title, isSelected: task == selection) {
                    selection = task
                }
                .accessibilityLabel(task.title)
            }
            if !overflowTasks.isEmpty {
                overflowMenu
            }
        }
        .padding(2)
        .background {
            RoundedRectangle(cornerRadius: 7)
                .fill(MereRunTheme.surfaceRaised)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(domain.title) task")
    }

    private var overflowMenu: some View {
        Menu {
            ForEach(overflowTasks) { task in
                Button {
                    selection = task
                } label: {
                    if task == selection {
                        Label(task.title, systemImage: "checkmark")
                    } else {
                        Text(task.title)
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Text(overflowSelection?.title ?? "More")
                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .font(.system(size: 12, weight: overflowSelection == nil ? .medium : .semibold))
            .foregroundStyle(overflowSelection == nil ? MereRunTheme.textSecondary : MereRunTheme.textPrimary)
            .lineLimit(1)
            .padding(.horizontal, 12)
            .frame(height: 24)
            .background {
                RoundedRectangle(cornerRadius: 5.5)
                    .fill(overflowSelection == nil ? Color.clear : MereRunTheme.segmentedSelection)
                    .shadow(
                        color: overflowSelection == nil ? .clear : MereRunTheme.shadowColor,
                        radius: 1,
                        y: 1
                    )
            }
            .contentShape(RoundedRectangle(cornerRadius: 5.5))
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .accessibilityLabel(overflowSelection.map { "More tasks, \($0.title) selected" } ?? "More tasks")
        .accessibilityAddTraits(overflowSelection == nil ? [] : .isSelected)
    }
}
