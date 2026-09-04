import AppKit
import Foundation

/// How the Library column lays its rows out.
package enum StudioLibraryViewMode: String, CaseIterable, Identifiable, Codable {
    case list
    case grid

    package var id: String { rawValue }

    package var systemImage: String {
        switch self {
        case .list: return "list.bullet"
        case .grid: return "square.grid.2x2"
        }
    }

    package var title: String {
        switch self {
        case .list: return "List"
        case .grid: return "Grid"
        }
    }
}

/// The media a Library row holds, as the column's filter names it. Rows are classified by their
/// primary output file, so a Find run that wrote an annotated PNG counts as an image and its JSON
/// sidecar does not split it out.
package enum StudioLibraryKind: String, CaseIterable, Identifiable, Codable {
    case all
    case images
    case video
    case audio
    case text

    package var id: String { rawValue }

    package var title: String {
        switch self {
        case .all: return "All kinds"
        case .images: return "Images"
        case .video: return "Video"
        case .audio: return "Audio"
        case .text: return "Text"
        }
    }

    package var systemImage: String {
        switch self {
        case .all: return "square.stack.3d.up"
        case .images: return "photo"
        case .video: return "film"
        case .audio: return "waveform"
        case .text: return "text.alignleft"
        }
    }

    /// Whether a row of `fileKind` (nil when the run wrote no file) belongs to this filter.
    package func matches(fileKind: StudioOutputFileKind?, hasText: Bool) -> Bool {
        switch self {
        case .all: return true
        case .images: return fileKind == .image
        case .video: return fileKind == .video
        case .audio: return fileKind == .audio
        case .text: return fileKind == .text || (fileKind == nil && hasText)
        }
    }
}

/// Everything the column filters by at once, so the presenter can be exercised without a view.
package struct StudioLibraryFilter: Equatable {
    package var scope: StudioLibraryScope = .domain
    package var domain: StudioDomain = .image
    package var kind: StudioLibraryKind = .all
    package var favoritesOnly = false
    package var query = ""

    package init(
        scope: StudioLibraryScope = .domain,
        domain: StudioDomain = .image,
        kind: StudioLibraryKind = .all,
        favoritesOnly: Bool = false,
        query: String = ""
    ) {
        self.scope = scope
        self.domain = domain
        self.kind = kind
        self.favoritesOnly = favoritesOnly
        self.query = query
    }
}

/// Pure filtering and day-grouping for the Library column.
package enum StudioLibraryPresenter {
    package static func fileKind(of item: StudioLibraryItem) -> StudioOutputFileKind? {
        guard let url = item.outputURL else { return nil }
        return StudioOutputFileKind.classify(url)
    }

    package static func filter(_ items: [StudioLibraryItem], with filter: StudioLibraryFilter) -> [StudioLibraryItem] {
        let query = filter.query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return items.filter { item in
            if filter.scope == .domain, item.domain != filter.domain { return false }
            if filter.favoritesOnly, !item.isStarred { return false }
            let hasText = item.outputText?.isBlank == false
            if !filter.kind.matches(fileKind: fileKind(of: item), hasText: hasText) { return false }
            guard !query.isEmpty else { return true }
            return item.displayTitle.lowercased().contains(query)
                || item.displayKindTitle.lowercased().contains(query)
                || item.prompt.lowercased().contains(query)
        }
    }

    /// The rows the column shows before the kind, favorites, and search filters narrow them — the
    /// number the header counts.
    package static func scoped(_ items: [StudioLibraryItem], scope: StudioLibraryScope, domain: StudioDomain) -> [StudioLibraryItem] {
        scope == .all ? items : items.filter { $0.domain == domain }
    }
}

/// Finder-style multi-selection: plain click replaces, ⌘ toggles, ⇧ extends from the anchor.
/// Kept out of the view so the rules are testable without synthesizing mouse events.
package enum StudioLibrarySelection {
    package struct Modifiers: Equatable {
        package var command = false
        package var shift = false

        package static let none = Modifiers()
        package static let command = Modifiers(command: true, shift: false)
        package static let shift = Modifiers(command: false, shift: true)

        package init(command: Bool = false, shift: Bool = false) {
            self.command = command
            self.shift = shift
        }

        package init(event flags: NSEvent.ModifierFlags) {
            command = flags.contains(.command)
            shift = flags.contains(.shift)
        }
    }

    package struct Result: Equatable {
        /// The whole batch after the click.
        package var selection: Set<UUID>
        /// The row the anchor moves to (nil leaves it where it was).
        package var anchor: UUID?
        /// The row the canvas should open, or nil when the click only changed the batch.
        package var opened: UUID?
    }

    /// `visible` is the column's current order, so ⇧ selects the run of rows the user sees.
    package static func click(
        on id: UUID,
        visible: [UUID],
        selection: Set<UUID>,
        anchor: UUID?,
        modifiers: Modifiers
    ) -> Result {
        if modifiers.shift, let anchor, anchor != id,
           let start = visible.firstIndex(of: anchor), let end = visible.firstIndex(of: id) {
            let range = start <= end ? start...end : end...start
            return Result(selection: Set(visible[range]), anchor: anchor, opened: id)
        }
        if modifiers.command {
            var next = selection
            if next.contains(id) {
                next.remove(id)
                // Deselecting the open row leaves the canvas where it is; the batch shrinks.
                return Result(selection: next, anchor: next.isEmpty ? nil : anchor, opened: nil)
            }
            next.insert(id)
            return Result(selection: next, anchor: id, opened: id)
        }
        return Result(selection: [id], anchor: id, opened: id)
    }
}

// MARK: - Scope
/// Which Library rows the column shows: the current domain's, or everything.
package enum StudioLibraryScope: String, CaseIterable, Identifiable {
    case domain
    case all

    package var id: String { rawValue }
}
