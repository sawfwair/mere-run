import StudioKit
import SwiftUI

/// The Library's lineage and a way to go to a linked run, as the window root hands them to every
/// result surface. A view hosted without it shows no links. The lineage is read from the store
/// only when a view asks, so a window showing no links never builds it.
struct StudioLibraryLinks {
    let library: StudioLibraryStore
    /// Shows a run the way picking its Library row would.
    let open: (StudioLibraryItem) -> Void
    /// Takes one "Made from" link off a run: `(source, run)`.
    let removeLink: (UUID, UUID) -> Void

    @MainActor var lineage: StudioLibraryLineage { library.lineage }
}

private struct StudioLibraryLinksKey: EnvironmentKey {
    static var defaultValue: StudioLibraryLinks? { nil }
}

extension EnvironmentValues {
    var studioLibraryLinks: StudioLibraryLinks? {
        get { self[StudioLibraryLinksKey.self] }
        set { self[StudioLibraryLinksKey.self] = newValue }
    }
}

/// "Made from" and "Used in" for one run, under a run's detail and a focused result: each linked
/// run is a link that opens it. A Made from link's context menu removes it.
struct StudioLineageLinks: View {
    let item: StudioLibraryItem
    /// How many links a row shows before the rest move into a "more" menu.
    var visibleLimit = 3

    @Environment(\.studioLibraryLinks) private var links

    var body: some View {
        if let links, links.lineage.hasLinks(item.id) {
            VStack(alignment: .leading, spacing: 6) {
                let sources = links.lineage.madeFrom(item.id)
                if !sources.isEmpty {
                    row("Made from", sources, links: links, removable: true)
                }
                let uses = links.lineage.usedIn(item.id)
                if !uses.isEmpty {
                    row("Used in", uses, links: links, removable: false)
                }
            }
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Lineage")
        }
    }

    private func row(_ title: String, _ linked: [StudioLibraryItem], links: StudioLibraryLinks, removable: Bool) -> some View {
        HStack(alignment: .center, spacing: 6) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(MereRunTheme.textMuted)
                .frame(width: 66, alignment: .leading)
            ForEach(linked.prefix(visibleLimit)) { other in
                StudioLineageLink(item: other, relation: title) { links.open(other) }
                    .contextMenu {
                        if removable {
                            Button("Remove link") { links.removeLink(other.id, item.id) }
                        }
                    }
            }
            if linked.count > visibleLimit {
                Menu("+\(linked.count - visibleLimit) more") {
                    ForEach(linked.dropFirst(visibleLimit)) { other in
                        Button(other.displayTitle) { links.open(other) }
                    }
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .font(.caption.weight(.medium))
                .accessibilityLabel("\(linked.count - visibleLimit) more runs \(title.lowercased())")
            }
            Spacer(minLength: 0)
        }
    }
}

/// One linked run: its kind's symbol and its title, as a link.
private struct StudioLineageLink: View {
    let item: StudioLibraryItem
    let relation: String
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: item.displaySystemImage)
                    .font(.system(size: 10, weight: .semibold))
                Text(item.displayTitle)
                    .font(.caption.weight(.medium))
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            .foregroundStyle(MereRunTheme.accent)
            .padding(.horizontal, 8)
            .frame(height: 22)
            .background {
                Capsule().fill(hovering ? MereRunTheme.accentSoft : MereRunTheme.surfaceRaised)
            }
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .help("Open \(item.displayTitle)")
        .accessibilityLabel("\(relation) \(item.displayTitle)")
        .accessibilityHint("Opens that run")
    }
}

/// A result card's one-line lineage under its settings: the run it was made from, and how many
/// runs have used it since. Each part opens the run it names.
struct StudioLineageBreadcrumb: View {
    let item: StudioLibraryItem

    @Environment(\.studioLibraryLinks) private var links

    var body: some View {
        if let links, links.lineage.hasLinks(item.id) {
            let sources = links.lineage.madeFrom(item.id)
            let uses = links.lineage.usedIn(item.id)
            HStack(spacing: 10) {
                if let source = sources.first {
                    Button { links.open(source) } label: {
                        Label {
                            Text(sources.count > 1 ? "From \(source.displayTitle) +\(sources.count - 1)" : "From \(source.displayTitle)")
                                .lineLimit(1)
                                .truncationMode(.tail)
                        } icon: {
                            Image(systemName: "arrow.turn.down.right")
                        }
                    }
                    .buttonStyle(.plain)
                    .help("Open \(source.displayTitle)")
                    .accessibilityLabel("Made from \(source.displayTitle)")
                }
                if let only = uses.first, uses.count == 1 {
                    Button { links.open(only) } label: {
                        Label("Used in \(only.displayTitle)", systemImage: "arrow.branch").lineLimit(1)
                    }
                    .buttonStyle(.plain)
                    .help("Open \(only.displayTitle)")
                } else if !uses.isEmpty {
                    Menu {
                        ForEach(uses) { use in
                            Button(use.displayTitle) { links.open(use) }
                        }
                    } label: {
                        Label("Used in \(uses.count) runs", systemImage: "arrow.branch")
                    }
                    .menuStyle(.borderlessButton)
                    .menuIndicator(.hidden)
                    .fixedSize()
                }
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(MereRunTheme.textSecondary)
            .labelStyle(StudioLineageLabelStyle())
        }
    }
}

/// A tight icon-and-title label for the breadcrumb, closer than the default spacing.
private struct StudioLineageLabelStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 4) {
            configuration.icon.font(.system(size: 10, weight: .semibold))
            configuration.title
        }
    }
}
