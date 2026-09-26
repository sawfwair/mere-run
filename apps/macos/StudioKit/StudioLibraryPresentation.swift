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
    /// Only the rows in this collection.
    package var collection: StudioLibraryCollection?
    /// Only the rows that ran this model (`StudioLibraryItem.recordedModelID`).
    package var modelID: String?
    /// Only the rows this task made (`StudioLibraryItem.task`).
    package var task: StudioTask?

    package init(
        scope: StudioLibraryScope = .domain,
        domain: StudioDomain = .image,
        kind: StudioLibraryKind = .all,
        favoritesOnly: Bool = false,
        query: String = "",
        collection: StudioLibraryCollection? = nil,
        modelID: String? = nil,
        task: StudioTask? = nil
    ) {
        self.scope = scope
        self.domain = domain
        self.kind = kind
        self.favoritesOnly = favoritesOnly
        self.query = query
        self.collection = collection
        self.modelID = modelID
        self.task = task
    }
}

/// Pure filtering and day-grouping for the Library column.
package enum StudioLibraryPresenter {
    package static func fileKind(of item: StudioLibraryItem) -> StudioOutputFileKind? {
        guard let url = item.outputURL else { return nil }
        return StudioOutputFileKind.classify(url)
    }

    /// `titles` names models the way the app shows them, so "qwen" finds a Qwen run whether the
    /// user remembers the title or the id. A whole status word in the query ("failed", "running")
    /// narrows to rows in that status, and the rest of the query must still match: "failed
    /// harbor" is the failed harbor runs, not every failed run plus every harbor run.
    package static func filter(
        _ items: [StudioLibraryItem],
        with filter: StudioLibraryFilter,
        titles: StudioModelTitles
    ) -> [StudioLibraryItem] {
        let query = filter.query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let words = query.split(whereSeparator: \.isWhitespace).map(String.init)
        let statusWords = words.filter(statusRawValues.contains)
        let text = words.filter { !statusRawValues.contains($0) }.joined(separator: " ")
        return items.filter { item in
            if filter.scope == .domain, item.domain != filter.domain { return false }
            if filter.favoritesOnly, !item.isStarred { return false }
            if let collection = filter.collection, !collection.contains(item.id) { return false }
            if let modelID = filter.modelID, item.recordedModelID != modelID { return false }
            if let task = filter.task, item.task != task { return false }
            let hasText = item.outputText?.isBlank == false
            if !filter.kind.matches(fileKind: fileKind(of: item), hasText: hasText) { return false }
            guard !query.isEmpty else { return true }
            if !statusWords.isEmpty, !statusWords.contains(item.status.rawValue) { return false }
            return text.isEmpty || matchesText(item, query: text, titles: titles)
        }
    }

    private static let statusRawValues = Set(StudioLibraryStatus.allCases.map(\.rawValue))

    private static func matchesText(_ item: StudioLibraryItem, query: String, titles: StudioModelTitles) -> Bool {
        item.displayTitle.lowercased().contains(query)
            || item.displayKindTitle.lowercased().contains(query)
            || item.prompt.lowercased().contains(query)
            || matchesModel(item, query: query, titles: titles)
    }

    private static func matchesModel(_ item: StudioLibraryItem, query: String, titles: StudioModelTitles) -> Bool {
        guard let modelID = item.recordedModelID else { return false }
        return modelID.lowercased().contains(query)
            || StudioModelNaming.displayName(modelID, titles: titles).lowercased().contains(query)
    }

    /// The models the filter menu offers: every model a row in `items` ran, named the way the
    /// app shows it, in alphabetical order of that name.
    package static func modelOptions(in items: [StudioLibraryItem], titles: StudioModelTitles) -> [StudioLibraryFilterOption<String>] {
        let ids = Set(items.compactMap(\.recordedModelID))
        return ids.map { StudioLibraryFilterOption(value: $0, title: StudioModelNaming.displayName($0, titles: titles)) }
            .sorted { ($0.title.localizedLowercase, $0.value) < ($1.title.localizedLowercase, $1.value) }
    }

    /// The tasks the filter menu offers: every task a row in `items` came from, in sidebar order.
    /// Under All tasks each is named with its domain ("Video · Generate"), since several domains
    /// have a Generate.
    package static func taskOptions(in items: [StudioLibraryItem], scope: StudioLibraryScope) -> [StudioLibraryFilterOption<StudioTask>] {
        let tasks = Set(items.map(\.task))
        return StudioTask.allCases.filter(tasks.contains).map { task in
            StudioLibraryFilterOption(value: task, title: scope == .all ? "\(task.domain.title) · \(task.title)" : task.title)
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

/// One choice in the Library's task or model filter: the value it filters by and its label.
package struct StudioLibraryFilterOption<Value: Hashable>: Hashable, Identifiable {
    package let value: Value
    package let title: String

    package init(value: Value, title: String) {
        self.value = value
        self.title = title
    }

    package var id: Value { value }
}

extension StudioLibraryItem {
    /// The task a row is filed under: its command's task when it records one, otherwise the
    /// task of the mode that created it.
    package var task: StudioTask {
        templateID?.studioTask ?? mode.task
    }
}
