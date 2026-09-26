import StudioKit
import SwiftUI

/// The Library column's collections, as a row of chips under the search field: a click shows
/// only that collection's runs (a second click shows everything again), a run dragged onto a chip
/// joins it, and each chip's context menu renames or deletes it. The row is only drawn once the
/// user has a collection; the first one starts from a run's "Add to collection" menu or the plus.
struct StudioLibraryCollectionChips: View {
    let collections: [StudioLibraryCollection]
    let memberCount: (StudioLibraryCollection) -> Int
    @Binding var selectedID: UUID?
    let onNew: () -> Void
    let onRename: (StudioLibraryCollection) -> Void
    let onDelete: (StudioLibraryCollection) -> Void
    /// Adds the runs that made the dropped files; false when none of them came from the Library.
    let onDrop: (StudioLibraryCollection, [URL]) -> Bool

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(collections) { collection in
                    StudioCollectionChip(
                        collection: collection,
                        count: memberCount(collection),
                        isSelected: selectedID == collection.id,
                        action: { selectedID = selectedID == collection.id ? nil : collection.id },
                        onDrop: { onDrop(collection, $0) }
                    )
                    .contextMenu {
                        Button("Rename…") { onRename(collection) }
                        Button("Delete collection") {
                            if selectedID == collection.id { selectedID = nil }
                            onDelete(collection)
                        }
                    }
                }
                Button(action: onNew) {
                    Image(systemName: "plus")
                        .font(.caption.weight(.semibold))
                        .frame(width: 24, height: 24)
                }
                .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
                .help("New collection")
                .accessibilityLabel("New collection")
            }
            .padding(.horizontal, 12)
        }
        .frame(height: 26)
        .padding(.bottom, 8)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Collections")
    }
}

/// One collection chip: its name and how many runs it holds. Selected, it is the column's filter;
/// under a drag it lights up to say a drop will add the run.
private struct StudioCollectionChip: View {
    let collection: StudioLibraryCollection
    let count: Int
    let isSelected: Bool
    let action: () -> Void
    let onDrop: ([URL]) -> Bool

    @State private var hovering = false
    @State private var isTargeted = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: isSelected ? "rectangle.stack.fill" : "rectangle.stack")
                    .font(.system(size: 10, weight: .semibold))
                Text(collection.name)
                    .font(.caption.weight(isSelected ? .semibold : .medium))
                    .lineLimit(1)
                Text("\(count)")
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(isSelected ? MereRunTheme.accent : MereRunTheme.textMuted)
            }
            .foregroundStyle(isSelected ? MereRunTheme.accent : MereRunTheme.textSecondary)
            .padding(.horizontal, 9)
            .frame(height: 24)
            .background {
                Capsule().fill(fill)
                    .overlay {
                        Capsule().strokeBorder(
                            isSelected || isTargeted ? MereRunTheme.accent : MereRunTheme.border.opacity(0.8),
                            lineWidth: isTargeted ? 1.5 : 1
                        )
                    }
            }
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .dropDestination(for: URL.self) { urls, _ in
            onDrop(urls)
        } isTargeted: { isTargeted = $0 }
        .animation(MereRunTheme.Motion.quick, value: isTargeted)
        .help(isSelected ? "Show every run" : "Show only \(collection.name). Drop a run here to add it.")
        .accessibilityLabel("\(collection.name) collection")
        .accessibilityValue(count == 1 ? "1 run" : "\(count) runs")
        .accessibilityHint("Shows only the runs in this collection")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private var fill: Color {
        if isSelected || isTargeted { return MereRunTheme.accentSoft }
        if hovering { return MereRunTheme.hoverFill }
        return MereRunTheme.surface
    }
}

/// "Add to collection": every collection, checked when the runs are all in it (choosing a checked
/// one takes them out), then New collection….
struct StudioAddToCollectionMenu: View {
    let itemIDs: [UUID]
    let collections: [StudioLibraryCollection]
    let onToggle: (StudioLibraryCollection) -> Void
    let onNew: () -> Void

    var body: some View {
        Menu {
            StudioCollectionMenuItems(itemIDs: itemIDs, collections: collections, onToggle: onToggle, onNew: onNew)
        } label: {
            Label(itemIDs.count > 1 ? "Add \(itemIDs.count) to collection" : "Add to collection",
                  systemImage: "rectangle.stack.badge.plus")
        }
    }
}

/// The items of the Add to collection menu, shared by the context menu and the batch bar.
struct StudioCollectionMenuItems: View {
    let itemIDs: [UUID]
    let collections: [StudioLibraryCollection]
    let onToggle: (StudioLibraryCollection) -> Void
    let onNew: () -> Void

    var body: some View {
        ForEach(collections) { collection in
            Toggle(collection.name, isOn: Binding(
                get: { !itemIDs.isEmpty && itemIDs.allSatisfy(collection.contains) },
                set: { _ in onToggle(collection) }
            ))
        }
        if !collections.isEmpty { Divider() }
        Button("New collection…", action: onNew)
    }
}

/// What the collection name sheet is asking for: a new collection holding some runs, or a new
/// name for an existing one.
enum StudioCollectionNamePrompt: Identifiable, Equatable {
    case create(itemIDs: [UUID])
    case rename(StudioLibraryCollection)

    var id: String {
        switch self {
        case .create(let ids): return "create-" + ids.map(\.uuidString).joined(separator: ",")
        case .rename(let collection): return "rename-" + collection.id.uuidString
        }
    }
}

extension View {
    /// Asks for a collection's name and writes it through the store, so the new collection or
    /// the rename is one undo step.
    func studioCollectionNamePrompt(_ prompt: Binding<StudioCollectionNamePrompt?>, library: StudioLibraryStore) -> some View {
        modifier(StudioCollectionNameAlert(prompt: prompt, library: library))
    }
}

private struct StudioCollectionNameAlert: ViewModifier {
    @Binding var prompt: StudioCollectionNamePrompt?
    let library: StudioLibraryStore
    @State private var name = ""

    func body(content: Content) -> some View {
        content
            .alert(title, isPresented: Binding(get: { prompt != nil }, set: { if !$0 { prompt = nil } })) {
                TextField("Name", text: $name)
                Button(confirmTitle, action: commit)
                Button("Cancel", role: .cancel) { prompt = nil }
            } message: {
                Text(message)
            }
            .onChange(of: prompt) { _, next in
                switch next {
                case .create: name = library.suggestedCollectionName
                case .rename(let collection): name = collection.name
                case nil: break
                }
            }
    }

    private var title: String {
        if case .rename = prompt { return "Rename collection" }
        return "New collection"
    }

    private var confirmTitle: String {
        if case .rename = prompt { return "Rename" }
        return "Create"
    }

    private var message: String {
        switch prompt {
        case .create(let ids) where ids.count == 1: return "The run is added to it."
        case .create(let ids) where ids.count > 1: return "The \(ids.count) runs are added to it."
        case .rename: return "Its runs stay as they are."
        default: return "Drag runs onto it to add them."
        }
    }

    private func commit() {
        switch prompt {
        case .create(let ids): library.createCollection(named: name, adding: ids)
        case .rename(let collection): library.renameCollection(id: collection.id, to: name)
        case nil: break
        }
        prompt = nil
    }
}
