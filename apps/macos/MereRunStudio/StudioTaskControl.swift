import SwiftUI

/// The toolbar's task control: one `MereSegmentedControl`-styled pill listing the domain's tasks.
/// Domains with more tasks than fit show the first `visibleLimit` as segments and the rest behind
/// a "More" segment that opens a menu; when the current task is one of those, the segment shows
/// its title and draws selected, so the toolbar always names where you are.
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

/// A toolbar item that draws its own chrome: on macOS 26 the shared glass platter is hidden so the
/// Studio's flat title, pill, and 28pt icon tiles sit directly on the toolbar.
struct StudioToolbarItem<Content: View>: ToolbarContent {
    let placement: ToolbarItemPlacement
    @ViewBuilder let content: () -> Content

    init(placement: ToolbarItemPlacement, @ViewBuilder content: @escaping () -> Content) {
        self.placement = placement
        self.content = content
    }

    var body: some ToolbarContent {
        if #available(macOS 26.0, *) {
            ToolbarItem(placement: placement, content: content)
                .sharedBackgroundVisibility(.hidden)
        } else {
            ToolbarItem(placement: placement, content: content)
        }
    }
}
